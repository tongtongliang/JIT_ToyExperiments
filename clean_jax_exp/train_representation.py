from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data import get_or_create_dataset
from .models import ModelConfig, init_params, loss_fn
from .train_gradient import (
    AnalysisConfig,
    as_numpy_tree,
    checkpoint_dump,
    run_representation_analysis,
    run_sampling_analysis,
    run_stability_analysis,
)

MODES = ("x", "v", "eps")


@dataclass(frozen=True)
class RepresentationTrainConfig:
    steps: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    loss_every: int = 100
    print_every: int = 1000
    fast_chunk_size: int = 1000
    t_min: float = 1e-3
    t_sampling: str = "sigmoid_normal"


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def adamw_update(params, grads, state, cfg: RepresentationTrainConfig):
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
    return new_params, {"t": t, "m": m, "v": v}, grad_norm, clip_scale


def make_batch(key, x0_all, cfg: RepresentationTrainConfig):
    key_idx, key_t, key_eps, key_next = jax.random.split(key, 4)
    idx = jax.random.randint(key_idx, (cfg.batch_size,), 0, x0_all.shape[0])
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


def make_training_functions(model_cfg: ModelConfig, train_cfg: RepresentationTrainConfig, mode: str):
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


def train_mode(mode: str, init_params_tree, x0_all, model_cfg: ModelConfig, train_cfg: RepresentationTrainConfig, seed: int):
    params = init_params_tree
    opt_state = adamw_init(params)
    key = jax.random.PRNGKey(seed)
    one_step, many_steps = make_training_functions(model_cfg, train_cfg, mode)
    rows: list[dict[str, Any]] = []
    step = 0

    # Run step 1 separately so compile/runtime issues surface immediately and we have an initial log.
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


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)

    model_cfg = ModelConfig(
        ambient_dim=args.ambient_dim,
        width=args.width,
        depth=args.depth,
        time_embed_dim=args.time_embed_dim,
        zero_init_output=True,
    )
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
    run_name = args.run_name or f"repr_D{args.ambient_dim}_adamw_w{args.width}_d{args.depth}_s{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "experiment_type": "representation_training",
        "optimizer": "adamw_manual",
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
        "notes": "Long training run for representation/stability/sampling only. No gradient-rank logs are collected here.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)

    x0_all = jnp.asarray(data["x0"], dtype=jnp.float32)
    init = init_params(jax.random.PRNGKey(args.seed + 123), model_cfg)
    params_by_mode: dict[str, Any] = {}
    all_loss_rows: list[dict[str, Any]] = []

    for mode in MODES:
        print(f"\n{'=' * 90}\nRepresentation training mode={mode}\n{'=' * 90}", flush=True)
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

    write_csv(run_dir / "logs" / "loss.csv", all_loss_rows)

    print("\nPost-training representation analysis", flush=True)
    run_representation_analysis(run_dir, params_by_mode, data["x0"], model_cfg, analysis_cfg)
    print("\nPost-training representation stability analysis", flush=True)
    run_stability_analysis(run_dir, params_by_mode, data["x0"], model_cfg, analysis_cfg)
    print("\nPost-training sampling analysis", flush=True)
    run_sampling_analysis(run_dir, params_by_mode, data, model_cfg, train_cfg, analysis_cfg)

    print(f"\nRepresentation run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Long-training representation/stability/sampling experiment")
    p.add_argument("--output-root", default="results/clean_jax_representation")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-embed-dim", type=int, default=256)
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
