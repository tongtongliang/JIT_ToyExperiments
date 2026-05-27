from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data import get_or_create_dataset
from .metrics import sign_invariant_angle_deg, spectral_metrics, top_right_singular_vector, tree_get
from .models_longskip import LongSkipModelConfig, count_params, forward, init_params, loss_fn
from .train_gradient import (
    AnalysisConfig,
    MATRIX_KINDS,
    MODES,
    TrainConfig,
    adamw_init,
    adamw_update,
    as_numpy_tree,
    checkpoint_dump,
    grad_reconstruction_error,
    layernorm_backward_np,
    make_batch,
    write_csv,
)


def tracked_layers(depth: int):
    layers = [("input_proj", ("input_proj", "w"), "input_proj")]
    for i in range(depth):
        layers.append((f"block{i + 1}_mlp0", ("blocks", i, "mlp0", "w"), f"block {i + 1} mlp fan-in"))
    layers.append(("output_proj", ("output_proj", "w"), "output_proj"))
    layers.append(("skip_head", ("skip_head", "w"), "long-skip scalar head"))
    return layers


def make_train_steps(model_cfg: LongSkipModelConfig, train_cfg: TrainConfig, mode: str):
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


def manual_activation_residuals_longskip(params_np: dict[str, Any], batch_np: tuple[np.ndarray, ...], model_cfg: LongSkipModelConfig, mode: str, t_min: float):
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

    c_out = cache["c_out"]
    net_out = cache["net_out"]
    d_net_out = d_raw * c_out
    d_skip_raw = np.stack(
        [np.sum(d_raw * z_t, axis=1), np.sum(d_raw * net_out, axis=1)],
        axis=1,
    ).astype(np.float64)

    activations: dict[str, np.ndarray] = {
        "output_proj": cache["final_h"],
        "skip_head": cache["t_cond"],
    }
    residuals: dict[str, np.ndarray] = {
        "output_proj": d_net_out.astype(np.float64),
        "skip_head": d_skip_raw,
    }
    d_h = d_net_out @ params_np["output_proj"]["w"]

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


def run_training_for_mode(
    mode: str,
    init_params_tree,
    x0_all: jnp.ndarray,
    model_cfg: LongSkipModelConfig,
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
            activations, residuals = manual_activation_residuals_longskip(old_params_np, batch_np, model_cfg, mode, train_cfg.t_min)

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


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)

    model_cfg = LongSkipModelConfig(ambient_dim=args.ambient_dim, width=args.width, depth=args.depth, time_embed_dim=args.time_embed_dim, zero_init_output=True)
    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, metric_every=args.metric_every, print_every=args.print_every, grad_clip_norm=args.grad_clip_norm)
    analysis_cfg = AnalysisConfig()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"longskip_gradient_D{args.ambient_dim}_adamw_w{args.width}_d{args.depth}_s{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    init = init_params(jax.random.PRNGKey(args.seed + 123), model_cfg)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "optimizer": "adamw_manual",
        "model_family": "fcn_adaln_longskip",
        "model_config": asdict(model_cfg),
        "parameter_count": count_params(init),
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
        "experiment_type": "gradient_analysis_longskip",
        "tracked_layers": [name for name, _, _ in tracked_layers(model_cfg.depth)],
        "notes": "Early-training gradient-rank and principal-angle analysis for learned long-skip FCN. Includes skip_head and long-skip-specific gradient factorization sanity checks.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)

    x0_all = jnp.asarray(data["x0"], dtype=jnp.float32)
    all_loss_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_angle_rows: list[dict[str, Any]] = []
    all_sanity_rows: list[dict[str, Any]] = []

    for mode in MODES:
        print(f"\n{'=' * 90}\nLong-skip gradient analysis mode={mode}\n{'=' * 90}", flush=True)
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
        all_loss_rows.extend(result["loss_rows"])
        all_metric_rows.extend(result["metric_rows"])
        all_angle_rows.extend(result["angle_rows"])
        all_sanity_rows.extend(result["sanity_rows"])
        gc.collect()

    write_csv(run_dir / "logs" / "loss.csv", all_loss_rows)
    write_csv(run_dir / "logs" / "matrix_metrics.csv", all_metric_rows)
    write_csv(run_dir / "logs" / "angle_metrics.csv", all_angle_rows)
    write_csv(run_dir / "logs" / "sanity_metrics.csv", all_sanity_rows)

    print(f"\nLong-skip gradient-analysis run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Learned long-skip FCN early gradient-rank and principal-angle experiment")
    p.add_argument("--output-root", default="results/clean_jax_gradient_longskip")
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
