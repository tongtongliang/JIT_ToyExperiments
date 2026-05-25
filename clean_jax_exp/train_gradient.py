from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
from functools import partial
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data import get_or_create_dataset, project_to_2d
from .metrics import (
    normalized_noise_variance,
    sign_invariant_angle_deg,
    spectral_metrics,
    top_right_singular_vector,
    tree_get,
)
from .models import ModelConfig, forward, init_params, loss_fn, pred_to_velocity, tracked_layers


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    gradient_analysis_steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    metric_every: int = 20
    loss_every: int = 10
    print_every: int = 50
    fast_chunk_size: int = 1000
    t_min: float = 1e-3
    t_sampling: str = "sigmoid_normal"


@dataclass(frozen=True)
class AnalysisConfig:
    repr_n_samples: int = 2048
    repr_batch_size: int = 512
    repr_t_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    stability_n_clean: int = 256
    stability_n_noise: int = 8
    sample_n: int = 2048
    sample_steps: int = 100
    rank95_threshold: float = 0.95
    rank90_threshold: float = 0.90


MODES = ("x", "v", "eps")
HOOKS = ("norm", "fanin")
MATRIX_KINDS = ("gradient", "momentum", "update", "activation", "residual")


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_global_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))


def clip_grads(grads, max_norm: float):
    norm = tree_global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm, scale


def adamw_init(params):
    return {"t": jnp.array(0, dtype=jnp.int32), "m": tree_zeros_like(params), "v": tree_zeros_like(params)}


def adamw_update(params, grads, state, cfg: TrainConfig):
    clipped, grad_norm, clip_scale = clip_grads(grads, cfg.grad_clip_norm)
    t = state["t"] + jnp.array(1, dtype=jnp.int32)
    m = jax.tree_util.tree_map(lambda m, g: cfg.beta1 * m + (1.0 - cfg.beta1) * g, state["m"], clipped)
    v = jax.tree_util.tree_map(lambda v, g: cfg.beta2 * v + (1.0 - cfg.beta2) * (g * g), state["v"], clipped)
    beta1_correction = 1.0 - cfg.beta1 ** t.astype(jnp.float32)
    beta2_correction = 1.0 - cfg.beta2 ** t.astype(jnp.float32)

    def one_update(p, mi, vi):
        mhat = mi / beta1_correction
        vhat = vi / beta2_correction
        return -cfg.lr * (mhat / (jnp.sqrt(vhat) + cfg.eps) + cfg.weight_decay * p)

    updates = jax.tree_util.tree_map(one_update, params, m, v)
    new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    return new_params, {"t": t, "m": m, "v": v}, updates, grad_norm, clip_scale


def make_batch(key, x0_all, cfg: TrainConfig):
    key_idx, key_t, key_eps, key_next = jax.random.split(key, 4)
    n_total = x0_all.shape[0]
    idx = jax.random.randint(key_idx, (cfg.batch_size,), 0, n_total)
    x0 = x0_all[idx]
    if cfg.t_sampling == "sigmoid_normal":
        t = jax.nn.sigmoid(jax.random.normal(key_t, (cfg.batch_size,), dtype=jnp.float32))
    elif cfg.t_sampling == "uniform":
        t = jax.random.uniform(key_t, (cfg.batch_size,), minval=cfg.t_min, maxval=1.0 - cfg.t_min, dtype=jnp.float32)
    else:
        raise ValueError(f"unknown t_sampling: {cfg.t_sampling}")
    t = jnp.clip(t, cfg.t_min, 1.0 - cfg.t_min)
    eps = jax.random.normal(key_eps, x0.shape, dtype=jnp.float32)
    z_t = (1.0 - t[:, None]) * x0 + t[:, None] * eps
    return (x0, eps, t, z_t), key_next


def make_train_steps(model_cfg: ModelConfig, train_cfg: TrainConfig, mode: str):
    def compute_loss_and_grad(params, batch):
        return jax.value_and_grad(lambda p: loss_fn(p, batch, model_cfg, mode, train_cfg.t_min))(params)

    def fast_body(carry, _):
        params, opt_state, key, x0_all = carry
        batch, key_next = make_batch(key, x0_all, train_cfg)
        loss, grads = compute_loss_and_grad(params, batch)
        new_params, new_state, _, grad_norm, clip_scale = adamw_update(params, grads, opt_state, train_cfg)
        return (new_params, new_state, key_next, x0_all), (loss, grad_norm, clip_scale)

    @jax.jit
    def fast_step(params, opt_state, key, x0_all):
        batch, key_next = make_batch(key, x0_all, train_cfg)
        loss, grads = compute_loss_and_grad(params, batch)
        new_params, new_state, _, grad_norm, clip_scale = adamw_update(params, grads, opt_state, train_cfg)
        return new_params, new_state, key_next, loss, grad_norm, clip_scale

    @jax.jit
    def metric_step(params, opt_state, key, x0_all):
        batch, key_next = make_batch(key, x0_all, train_cfg)
        loss, grads = compute_loss_and_grad(params, batch)
        new_params, new_state, updates, grad_norm, clip_scale = adamw_update(params, grads, opt_state, train_cfg)
        return new_params, new_state, key_next, loss, grad_norm, clip_scale, grads, updates, batch

    @partial(jax.jit, static_argnames=("n_steps",))
    def fast_many_steps(params, opt_state, key, x0_all, n_steps: int):
        (new_params, new_state, key_next, _), logs = jax.lax.scan(
            fast_body,
            (params, opt_state, key, x0_all),
            None,
            length=n_steps,
        )
        return new_params, new_state, key_next, logs

    return fast_step, metric_step, fast_many_steps


def as_numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def layernorm_backward_np(dy: np.ndarray, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    norm = (x - mean) * inv_std
    return inv_std * (dy - dy.mean(axis=-1, keepdims=True) - norm * (dy * norm).mean(axis=-1, keepdims=True))


def manual_activation_residuals(params_np: dict[str, Any], batch_np: tuple[np.ndarray, ...], model_cfg: ModelConfig, mode: str, t_min: float):
    x0, eps, t, z_t = batch_np
    raw, cache = forward(params_np, jnp.asarray(z_t), jnp.asarray(t), model_cfg, return_cache=True)
    raw = np.asarray(raw)
    cache = as_numpy_tree(cache)
    bsz, data_dim = raw.shape
    t_col = np.clip(t[:, None], t_min, 1.0 - t_min)
    v_target = eps - x0
    if mode == "v":
        v_pred = raw
        d_raw = 2.0 * (v_pred - v_target) / float(bsz * data_dim)
    elif mode == "x":
        v_pred = (z_t - raw) / t_col
        d_raw = -2.0 * (v_pred - v_target) / float(bsz * data_dim) / t_col
    elif mode == "eps":
        v_pred = (raw - z_t) / (1.0 - t_col)
        d_raw = 2.0 * (v_pred - v_target) / float(bsz * data_dim) / (1.0 - t_col)
    else:
        raise ValueError(mode)

    activations: dict[str, np.ndarray] = {"output_proj": cache["final_h"]}
    residuals: dict[str, np.ndarray] = {"output_proj": d_raw.astype(np.float64)}
    d_h = d_raw @ params_np["output_proj"]["w"]

    for i in reversed(range(model_cfg.depth)):
        block = params_np["blocks"][i]
        bc = cache["blocks"][i]
        name = f"block{i + 1}_mlp0"
        d_a2 = d_h * bc["alpha"]
        d_relu = d_a2 @ block["mlp1"]["w"]
        d_a1 = d_relu * (bc["a1"] > 0.0)
        activations[name] = bc["fanin"]
        residuals[name] = d_a1.astype(np.float64)
        d_fanin = d_a1 @ block["mlp0"]["w"]
        d_norm = d_fanin * (1.0 + bc["gamma"])
        d_h = d_h + layernorm_backward_np(d_norm, bc["h_in"])

    activations["input_proj"] = z_t
    residuals["input_proj"] = d_h.astype(np.float64)
    return activations, residuals


def grad_reconstruction_error(grad_w: np.ndarray, activation: np.ndarray, residual: np.ndarray):
    estimate = residual.T @ activation
    num = np.linalg.norm(grad_w - estimate)
    den = np.linalg.norm(grad_w) + 1e-20
    return float(num / den), float(num)


def checkpoint_dump(path: Path, payload: dict[str, Any]):
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def checkpoint_load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training_for_mode(
    mode: str,
    init_params_tree,
    x0_all: jnp.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_dir: Path,
    save_checkpoints: bool,
    seed: int,
):
    params = init_params_tree
    opt_state = adamw_init(params)
    fast_step, metric_step, fast_many_steps = make_train_steps(model_cfg, train_cfg, mode)
    key = jax.random.PRNGKey(seed)

    tracked = tracked_layers(model_cfg.depth)
    previous_vectors: dict[tuple[str, str], np.ndarray | None] = {}
    loss_rows = []
    metric_rows = []
    angle_rows = []
    sanity_rows = []

    analysis_steps = min(train_cfg.steps, train_cfg.gradient_analysis_steps)
    current_step = 0

    # Early phase: record gradients, momentum, updates, activations, residuals, and angles.
    for step in range(1, analysis_steps + 1):
        current_step = step
        if step == 1 or step % train_cfg.metric_every == 0:
            old_params = params
            params, opt_state, key, loss, grad_norm, clip_scale, grads, updates, batch = metric_step(params, opt_state, key, x0_all)
            old_params_np = as_numpy_tree(old_params)
            grads_np = as_numpy_tree(grads)
            updates_np = as_numpy_tree(updates)
            m_np = as_numpy_tree(opt_state["m"])
            batch_np = tuple(np.asarray(x) for x in batch)
            activations, residuals = manual_activation_residuals(old_params_np, batch_np, model_cfg, mode, train_cfg.t_min)

            for layer_name, path, layer_label in tracked:
                grad_w = np.asarray(tree_get(grads_np, path))
                momentum_w = np.asarray(tree_get(m_np, path))
                update_w = np.asarray(tree_get(updates_np, path))
                matrices = {
                    "gradient": (grad_w, False),
                    "momentum": (momentum_w, False),
                    "update": (update_w, False),
                    "activation": (activations[layer_name], True),
                    "residual": (residuals[layer_name], True),
                }
                rel, abs_err = grad_reconstruction_error(grad_w, activations[layer_name], residuals[layer_name])
                sanity_rows.append({
                    "mode": mode,
                    "step": step,
                    "layer": layer_name,
                    "relative_error": rel,
                    "absolute_error": abs_err,
                })

                current_vectors: dict[str, np.ndarray | None] = {}
                for kind, (mat, center) in matrices.items():
                    sm = spectral_metrics(mat, center=center, rank_threshold=0.90)
                    metric_rows.append({
                        "mode": mode,
                        "step": step,
                        "layer": layer_name,
                        "layer_label": layer_label,
                        "matrix_kind": kind,
                        "stable_rank": sm["stable_rank"],
                        "rank90": sm["rank_k"],
                        "op_norm": sm["op_norm"],
                        "fro_norm": sm["fro_norm"],
                        "top1_energy": sm["top1_energy"],
                    })
                    vec = top_right_singular_vector(mat, center=center)
                    current_vectors[kind] = vec
                    prev_key = (layer_name, kind)
                    if prev_key in previous_vectors:
                        angle_rows.append({
                            "mode": mode,
                            "step": step,
                            "layer": layer_name,
                            "angle_kind": f"adjacent_{kind}",
                            "angle_deg": sign_invariant_angle_deg(previous_vectors[prev_key], vec),
                        })
                angle_rows.append({
                    "mode": mode,
                    "step": step,
                    "layer": layer_name,
                    "angle_kind": "gradient_vs_previous_momentum",
                    "angle_deg": sign_invariant_angle_deg(previous_vectors.get((layer_name, "momentum")), current_vectors["gradient"]),
                })
                for kind, vec in current_vectors.items():
                    previous_vectors[(layer_name, kind)] = vec
        else:
            params, opt_state, key, loss, grad_norm, clip_scale = fast_step(params, opt_state, key, x0_all)

        if step == 1 or step % train_cfg.loss_every == 0 or step == train_cfg.steps:
            loss_rows.append({
                "mode": mode,
                "step": step,
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "clip_scale": float(clip_scale),
            })
        if step == 1 or step % train_cfg.print_every == 0:
            print(f"[{mode}] step {step:6d}/{train_cfg.steps} loss={float(loss):.6f} grad_norm={float(grad_norm):.4f} clip={float(clip_scale):.3f}", flush=True)

    # Late phase: continue optimizing with JAX scan chunks, but only keep lightweight loss logs.
    while current_step < train_cfg.steps:
        chunk = min(train_cfg.fast_chunk_size, train_cfg.steps - current_step)
        params, opt_state, key, logs = fast_many_steps(params, opt_state, key, x0_all, n_steps=chunk)
        losses, grad_norms, clip_scales = [np.asarray(x) for x in logs]
        for i in range(chunk):
            step = current_step + i + 1
            if step % train_cfg.loss_every == 0 or step == train_cfg.steps:
                loss_rows.append({
                    "mode": mode,
                    "step": step,
                    "loss": float(losses[i]),
                    "grad_norm": float(grad_norms[i]),
                    "clip_scale": float(clip_scales[i]),
                })
            if step % train_cfg.print_every == 0 or step == train_cfg.steps:
                print(f"[{mode}] step {step:6d}/{train_cfg.steps} loss={float(losses[i]):.6f} grad_norm={float(grad_norms[i]):.4f} clip={float(clip_scales[i]):.3f}", flush=True)
        current_step += chunk

    ckpt_path = None
    if save_checkpoints:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{mode}_final.pkl"
        checkpoint_dump(ckpt_path, {"mode": mode, "params": as_numpy_tree(params), "opt_state": as_numpy_tree(opt_state), "model_config": asdict(model_cfg), "train_config": asdict(train_cfg)})

    return {
        "params": as_numpy_tree(params),
        "loss_rows": loss_rows,
        "metric_rows": metric_rows,
        "angle_rows": angle_rows,
        "sanity_rows": sanity_rows,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_repr_for_batch(params, z_t_np: np.ndarray, t_np: np.ndarray, model_cfg: ModelConfig, batch_size: int):
    collected = {hook: [[] for _ in range(model_cfg.depth)] for hook in HOOKS}
    for start in range(0, z_t_np.shape[0], batch_size):
        end = min(start + batch_size, z_t_np.shape[0])
        _, cache = forward(params, jnp.asarray(z_t_np[start:end]), jnp.asarray(t_np[start:end]), model_cfg, return_cache=True)
        cache_np = as_numpy_tree(cache)
        for i, bc in enumerate(cache_np["blocks"]):
            collected["norm"][i].append(bc["norm"])
            collected["fanin"][i].append(bc["fanin"])
    return {hook: [np.concatenate(parts, axis=0) for parts in layers] for hook, layers in collected.items()}


def make_eval_batch(x0_np: np.ndarray, n_samples: int, seed: int, t_value: float | None):
    rng = np.random.default_rng(seed)
    replace = n_samples > x0_np.shape[0]
    idx = rng.choice(x0_np.shape[0], size=n_samples, replace=replace)
    x0 = x0_np[idx]
    eps = rng.normal(size=x0.shape).astype(np.float32)
    if t_value is None:
        t = rng.normal(size=(n_samples,)).astype(np.float32)
        t = 1.0 / (1.0 + np.exp(-t))
        label = "mixed"
        t_out = np.nan
    else:
        t = np.full((n_samples,), float(t_value), dtype=np.float32)
        label = f"t={t_value:.1f}"
        t_out = float(t_value)
    z_t = (1.0 - t[:, None]) * x0 + t[:, None] * eps
    return label, t_out, z_t.astype(np.float32), t.astype(np.float32)


def run_representation_analysis(run_dir: Path, params_by_mode: dict[str, Any], x0_np: np.ndarray, model_cfg: ModelConfig, cfg: AnalysisConfig):
    rows = []
    batches = [(None, *make_eval_batch(x0_np, cfg.repr_n_samples, 10_000, None))]
    for tv in cfg.repr_t_values:
        batches.append((tv, *make_eval_batch(x0_np, cfg.repr_n_samples, 10_000 + int(tv * 1000), tv)))

    for mode, params in params_by_mode.items():
        print(f"representation analysis: mode={mode}")
        for _, label, t_out, z_t, t in batches:
            reps = collect_repr_for_batch(params, z_t, t, model_cfg, cfg.repr_batch_size)
            for hook in HOOKS:
                for layer_idx, h in enumerate(reps[hook], start=1):
                    m95 = spectral_metrics(h, center=True, rank_threshold=cfg.rank95_threshold)
                    rows.append({
                        "mode": mode,
                        "sampling": label,
                        "t": t_out,
                        "hook": hook,
                        "layer": layer_idx,
                        "stable_rank": m95["stable_rank"],
                        "rank95": m95["rank_k"],
                        "op_norm": m95["op_norm"],
                        "fro_norm": m95["fro_norm"],
                        "top1_energy": m95["top1_energy"],
                    })
            del reps
            gc.collect()
    write_csv(run_dir / "analysis" / "representation_metrics.csv", rows)
    return rows


def run_stability_analysis(run_dir: Path, params_by_mode: dict[str, Any], x0_np: np.ndarray, model_cfg: ModelConfig, cfg: AnalysisConfig):
    rows = []
    rng = np.random.default_rng(20_000)
    replace = cfg.stability_n_clean > x0_np.shape[0]
    idx = rng.choice(x0_np.shape[0], size=cfg.stability_n_clean, replace=replace)
    base_x = x0_np[idx]
    x_rep = np.repeat(base_x, cfg.stability_n_noise, axis=0)
    for mode, params in params_by_mode.items():
        print(f"stability analysis: mode={mode}")
        for tv in cfg.repr_t_values:
            eps = rng.normal(size=x_rep.shape).astype(np.float32)
            t = np.full((x_rep.shape[0],), float(tv), dtype=np.float32)
            z_t = (1.0 - t[:, None]) * x_rep + t[:, None] * eps
            reps = collect_repr_for_batch(params, z_t.astype(np.float32), t, model_cfg, cfg.repr_batch_size)
            for hook in HOOKS:
                for layer_idx, h in enumerate(reps[hook], start=1):
                    rows.append({
                        "mode": mode,
                        "t": float(tv),
                        "hook": hook,
                        "layer": layer_idx,
                        "nsv": normalized_noise_variance(h, cfg.stability_n_clean, cfg.stability_n_noise),
                    })
    write_csv(run_dir / "analysis" / "representation_stability.csv", rows)
    return rows


def make_sampler(model_cfg: ModelConfig, mode: str, t_min: float, n_steps: int):
    @partial(jax.jit, static_argnames=("n_samples",))
    def sample(params, key, n_samples: int):
        z = jax.random.normal(key, (n_samples, model_cfg.ambient_dim), dtype=jnp.float32)
        times = jnp.linspace(1.0 - t_min, t_min, n_steps, dtype=jnp.float32)

        def body(z_cur, i):
            t_now = times[i]
            t_next = times[i + 1]
            dt = t_now - t_next
            t_vec = jnp.full((n_samples,), t_now, dtype=jnp.float32)
            raw = forward(params, z_cur, t_vec, model_cfg, return_cache=False)
            v = pred_to_velocity(raw, z_cur, t_vec, mode, t_min)
            return z_cur - dt * v, None

        z, _ = jax.lax.scan(body, z, jnp.arange(n_steps - 1))
        return z

    return sample


def run_sampling_analysis(run_dir: Path, params_by_mode: dict[str, Any], data: dict[str, np.ndarray], model_cfg: ModelConfig, train_cfg: TrainConfig, cfg: AnalysisConfig):
    rows = []
    sample_arrays = {}
    train_metrics = spectral_metrics(data["x0"], center=True, rank_threshold=cfg.rank95_threshold)
    rows.append({"mode": "training_data", "stable_rank": train_metrics["stable_rank"], "rank95": train_metrics["rank_k"], "top1_energy": train_metrics["top1_energy"]})
    for mode, params in params_by_mode.items():
        print(f"sampling: mode={mode}")
        sampler = make_sampler(model_cfg, mode, train_cfg.t_min, cfg.sample_steps)
        samples = np.asarray(sampler(params, jax.random.PRNGKey(30_000 + len(mode)), cfg.sample_n))
        samples_2d = project_to_2d(samples, data["P"], data["mean"], data["std"])
        sample_arrays[f"{mode}_highd"] = samples.astype(np.float32)
        sample_arrays[f"{mode}_2d"] = samples_2d.astype(np.float32)
        sm = spectral_metrics(samples, center=True, rank_threshold=cfg.rank95_threshold)
        rows.append({"mode": mode, "stable_rank": sm["stable_rank"], "rank95": sm["rank_k"], "top1_energy": sm["top1_energy"]})
    sample_arrays["training_2d"] = data["data_2d"].astype(np.float32)
    np.savez_compressed(run_dir / "analysis" / "samples.npz", **sample_arrays)
    write_csv(run_dir / "analysis" / "sample_metrics.csv", rows)
    return rows


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)

    model_cfg = ModelConfig(ambient_dim=args.ambient_dim, width=args.width, depth=args.depth, time_embed_dim=args.time_embed_dim, zero_init_output=True)
    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, metric_every=args.metric_every, print_every=args.print_every, grad_clip_norm=args.grad_clip_norm)
    analysis_cfg = AnalysisConfig()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"clean_jax_D{args.ambient_dim}_adamw_w{args.width}_d{args.depth}_s{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "optimizer": "adamw_manual",
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
        "experiment_type": "gradient_analysis",
        "notes": "Early-training gradient-rank and principal-angle analysis only. No representation/sampling posthoc analysis is run here.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)

    x0_all = jnp.asarray(data["x0"], dtype=jnp.float32)
    init = init_params(jax.random.PRNGKey(args.seed + 123), model_cfg)
    params_by_mode = {}
    all_loss_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_angle_rows: list[dict[str, Any]] = []
    all_sanity_rows: list[dict[str, Any]] = []

    for mode in MODES:
        print(f"\n{'=' * 90}\nTraining mode={mode}\n{'=' * 90}")
        result = run_training_for_mode(
            mode=mode,
            init_params_tree=init,
            x0_all=x0_all,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            run_dir=run_dir,
            save_checkpoints=args.save_checkpoints,
            seed=args.seed + 1000,
        )
        params_by_mode[mode] = result["params"]
        all_loss_rows.extend(result["loss_rows"])
        all_metric_rows.extend(result["metric_rows"])
        all_angle_rows.extend(result["angle_rows"])
        all_sanity_rows.extend(result["sanity_rows"])
        gc.collect()

    write_csv(run_dir / "logs" / "loss.csv", all_loss_rows)
    write_csv(run_dir / "logs" / "matrix_metrics.csv", all_metric_rows)
    write_csv(run_dir / "logs" / "angle_metrics.csv", all_angle_rows)
    write_csv(run_dir / "logs" / "sanity_metrics.csv", all_sanity_rows)

    print("\nGradient-analysis run complete. Representation/stability/sampling are intentionally handled by train_representation.py.")
    print(f"Run complete: {run_dir}")
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Clean JAX toy diffusion inductive-bias experiment")
    p.add_argument("--output-root", default="results/clean_jax")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--metric-every", type=int, default=20)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--save-checkpoints", action="store_true")
    return p


def main(argv: list[str] | None = None):
    args = build_argparser().parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
