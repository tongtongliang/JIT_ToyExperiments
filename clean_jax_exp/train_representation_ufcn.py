from __future__ import annotations

import argparse
import csv
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

from .data import get_or_create_dataset, project_to_2d
from .metrics import normalized_noise_variance, spectral_metrics
from .models import pred_to_velocity
from .models_ufcn import UFCNModelConfig, count_params, forward, init_params, loss_fn, uvit_skip_pairs
from .posthoc_analysis import ensure_sample_quality_metrics, ensure_sample_subspace_metrics
from .train_gradient import AnalysisConfig, HOOKS, as_numpy_tree, checkpoint_dump
from .train_representation import (
    MODES,
    RepresentationTrainConfig,
    adamw_init,
    adamw_update,
    make_batch,
    write_csv,
)


def make_training_functions(model_cfg: UFCNModelConfig, train_cfg: RepresentationTrainConfig, mode: str):
    def body(carry, _):
        params, opt_state, key, x0_all = carry
        batch, key_next = make_batch(key, x0_all, train_cfg)
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, model_cfg, mode, train_cfg.t_min))(params)
        params, opt_state, grad_norm, clip_scale = adamw_update(params, grads, opt_state, train_cfg)
        return (params, opt_state, key_next, x0_all), (loss, grad_norm, clip_scale)

    @jax.jit
    def one_step(params, opt_state, key, x0_all):
        (params, opt_state, key, _), logs = body((params, opt_state, key, x0_all), None)
        loss, grad_norm, clip_scale = logs
        return params, opt_state, key, loss, grad_norm, clip_scale

    @partial(jax.jit, static_argnames=("n_steps",))
    def many_steps(params, opt_state, key, x0_all, n_steps: int):
        (params, opt_state, key, _), logs = jax.lax.scan(
            body,
            (params, opt_state, key, x0_all),
            None,
            length=n_steps,
        )
        return params, opt_state, key, logs

    return one_step, many_steps


def train_mode(
    mode: str,
    init_params_tree,
    x0_all,
    model_cfg: UFCNModelConfig,
    train_cfg: RepresentationTrainConfig,
    seed: int,
):
    params = init_params_tree
    opt_state = adamw_init(params)
    key = jax.random.PRNGKey(seed)
    one_step, many_steps = make_training_functions(model_cfg, train_cfg, mode)
    rows: list[dict[str, Any]] = []
    step = 0

    params, opt_state, key, loss, grad_norm, clip_scale = one_step(params, opt_state, key, x0_all)
    step = 1
    rows.append({"mode": mode, "step": step, "loss": float(loss), "grad_norm": float(grad_norm), "clip_scale": float(clip_scale)})
    print(f"[{mode}] step {step:6d}/{train_cfg.steps} loss={float(loss):.6f} grad_norm={float(grad_norm):.4f} clip={float(clip_scale):.3f}", flush=True)

    while step < train_cfg.steps:
        chunk = min(train_cfg.fast_chunk_size, train_cfg.steps - step)
        params, opt_state, key, logs = many_steps(params, opt_state, key, x0_all, n_steps=chunk)
        losses, grad_norms, clip_scales = [np.asarray(x) for x in logs]
        for i in range(chunk):
            s = step + i + 1
            if s % train_cfg.loss_every == 0 or s == train_cfg.steps:
                rows.append({"mode": mode, "step": s, "loss": float(losses[i]), "grad_norm": float(grad_norms[i]), "clip_scale": float(clip_scales[i])})
            if s % train_cfg.print_every == 0 or s == train_cfg.steps:
                print(f"[{mode}] step {s:6d}/{train_cfg.steps} loss={float(losses[i]):.6f} grad_norm={float(grad_norms[i]):.4f} clip={float(clip_scales[i]):.3f}", flush=True)
        step += chunk

    return as_numpy_tree(params), as_numpy_tree(opt_state), rows


def collect_repr_for_batch(params, z_t_np: np.ndarray, t_np: np.ndarray, model_cfg: UFCNModelConfig, batch_size: int):
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


def run_representation_analysis(run_dir: Path, params_by_mode: dict[str, Any], x0_np: np.ndarray, model_cfg: UFCNModelConfig, cfg: AnalysisConfig):
    rows = []
    batches = [(None, *make_eval_batch(x0_np, cfg.repr_n_samples, 10_000, None))]
    for tv in cfg.repr_t_values:
        batches.append((tv, *make_eval_batch(x0_np, cfg.repr_n_samples, 10_000 + int(tv * 1000), tv)))

    for mode, params in params_by_mode.items():
        print(f"representation analysis: mode={mode}", flush=True)
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


def run_stability_analysis(run_dir: Path, params_by_mode: dict[str, Any], x0_np: np.ndarray, model_cfg: UFCNModelConfig, cfg: AnalysisConfig):
    rows = []
    rng = np.random.default_rng(20_000)
    replace = cfg.stability_n_clean > x0_np.shape[0]
    idx = rng.choice(x0_np.shape[0], size=cfg.stability_n_clean, replace=replace)
    base_x = x0_np[idx]
    x_rep = np.repeat(base_x, cfg.stability_n_noise, axis=0)
    for mode, params in params_by_mode.items():
        print(f"stability analysis: mode={mode}", flush=True)
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
            del reps
            gc.collect()
    write_csv(run_dir / "analysis" / "representation_stability.csv", rows)
    return rows


def make_sampler(model_cfg: UFCNModelConfig, mode: str, t_min: float, n_steps: int):
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


def run_sampling_analysis(
    run_dir: Path,
    params_by_mode: dict[str, Any],
    data: dict[str, np.ndarray],
    model_cfg: UFCNModelConfig,
    train_cfg: RepresentationTrainConfig,
    cfg: AnalysisConfig,
):
    rows = []
    sample_arrays = {}
    train_metrics = spectral_metrics(data["x0"], center=True, rank_threshold=cfg.rank95_threshold)
    rows.append({"mode": "training_data", "stable_rank": train_metrics["stable_rank"], "rank95": train_metrics["rank_k"], "top1_energy": train_metrics["top1_energy"]})
    for mode, params in params_by_mode.items():
        print(f"sampling: mode={mode}", flush=True)
        sampler = make_sampler(model_cfg, mode, train_cfg.t_min, cfg.sample_steps)
        samples = np.asarray(sampler(params, jax.random.PRNGKey(30_000 + len(mode)), cfg.sample_n))
        samples_2d = project_to_2d(samples, data["P"], data["mean"], data["std"])
        sample_arrays[f"{mode}_highd"] = samples.astype(np.float32)
        sample_arrays[f"{mode}_2d"] = samples_2d.astype(np.float32)
        sm = spectral_metrics(samples, center=True, rank_threshold=cfg.rank95_threshold)
        rows.append({"mode": mode, "stable_rank": sm["stable_rank"], "rank95": sm["rank_k"], "top1_energy": sm["top1_energy"]})
        del samples
        gc.collect()
    sample_arrays["training_2d"] = data["data_2d"].astype(np.float32)
    np.savez_compressed(run_dir / "analysis" / "samples.npz", **sample_arrays)
    write_csv(run_dir / "analysis" / "sample_metrics.csv", rows)
    ensure_sample_quality_metrics(run_dir, force=True)
    ensure_sample_subspace_metrics(run_dir, force=True)
    return rows


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)

    model_cfg = UFCNModelConfig(
        ambient_dim=args.ambient_dim,
        width=args.width,
        depth=args.depth,
        time_embed_dim=args.time_embed_dim,
        zero_init_output=True,
        skip_init=args.skip_init,
    )
    if model_cfg.depth != 5:
        print(f"Warning: requested depth={model_cfg.depth}; symmetric U skips will be {uvit_skip_pairs(model_cfg.depth)}", flush=True)
    train_cfg = RepresentationTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_every=args.loss_every,
        print_every=args.print_every,
        grad_clip_norm=args.grad_clip_norm,
        fast_chunk_size=args.fast_chunk_size,
    )
    analysis_cfg = AnalysisConfig()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"repr_ufcn_D{args.ambient_dim}_adamw_w{args.width}_d{args.depth}_s{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    init = init_params(jax.random.PRNGKey(args.seed + 123), model_cfg)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "experiment_type": "representation_training_ufcn",
        "optimizer": "adamw_manual",
        "model_family": "fcn_adaln_uvit_hidden_skips",
        "model_config": asdict(model_cfg),
        "parameter_count": count_params(init),
        "uvit_hidden_skips": {
            "description": "U-ViT-style hidden long skips: concat encoder block output with decoder block input, then project back to width.",
            "skip_pairs_1_based": [[source + 1, target + 1] for source, target in uvit_skip_pairs(model_cfg.depth)],
            "concat_order": "[current_decoder_stream, encoder_skip_stream]",
            "projection_init": args.skip_init,
        },
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
        "notes": "U-FCN representation/stability/sampling run. For depth=5, block1->block5 and block2->block4 long skips are used.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)

    print(json.dumps({"run_dir": str(run_dir), "parameter_count": metadata["parameter_count"], "skip_pairs_1_based": metadata["uvit_hidden_skips"]["skip_pairs_1_based"]}, indent=2), flush=True)
    x0_all = jnp.asarray(data["x0"], dtype=jnp.float32)
    params_by_mode: dict[str, Any] = {}
    all_loss_rows: list[dict[str, Any]] = []

    for mode in MODES:
        print(f"\n{'=' * 90}\nU-FCN representation training mode={mode}\n{'=' * 90}", flush=True)
        params, opt_state, rows = train_mode(mode, init, x0_all, model_cfg, train_cfg, seed=args.seed + 1000)
        params_by_mode[mode] = params
        all_loss_rows.extend(rows)
        if args.save_checkpoints:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dump(
                ckpt_dir / f"{mode}_final.pkl",
                {"mode": mode, "params": params, "opt_state": opt_state, "model_config": asdict(model_cfg), "train_config": asdict(train_cfg)},
            )
        gc.collect()

    write_csv(run_dir / "logs" / "loss.csv", all_loss_rows)

    print("\nPost-training U-FCN representation analysis", flush=True)
    run_representation_analysis(run_dir, params_by_mode, data["x0"], model_cfg, analysis_cfg)
    print("\nPost-training U-FCN representation stability analysis", flush=True)
    run_stability_analysis(run_dir, params_by_mode, data["x0"], model_cfg, analysis_cfg)
    print("\nPost-training U-FCN sampling analysis", flush=True)
    run_sampling_analysis(run_dir, params_by_mode, data, model_cfg, train_cfg, analysis_cfg)

    print(f"\nU-FCN representation run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Long-training FCN representation/stability/sampling experiment with U-ViT-style hidden long skips")
    p.add_argument("--output-root", default="results/clean_jax_representation_ufcn")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--skip-init", choices=("identity_current", "zero"), default="identity_current")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--fast-chunk-size", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--save-checkpoints", action="store_true")
    return p


def main(argv: list[str] | None = None):
    args = build_argparser().parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
