from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from clean_jax_exp.models_longskip import LongSkipModelConfig, forward
from clean_jax_exp.train_gradient import checkpoint_load
from clean_jax_exp.visualize import MODE_COLORS, MODE_LABELS, MODE_MARKERS, setup_style

MODES = ("x", "v", "eps")


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_config_from_meta(run_dir: Path) -> LongSkipModelConfig:
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    raw = dict(meta["model_config"])
    names = {f.name for f in fields(LongSkipModelConfig)}
    return LongSkipModelConfig(**{k: v for k, v in raw.items() if k in names})


def load_params(run_dir: Path) -> dict[str, Any]:
    return {mode: checkpoint_load(run_dir / "checkpoints" / f"{mode}_final.pkl")["params"] for mode in MODES}


def evaluate_coefficients(run_dir: Path, cfg: LongSkipModelConfig, params_by_mode: dict[str, Any], n_grid: int):
    t_grid = np.linspace(0.001, 0.999, n_grid, dtype=np.float32)
    z = np.zeros((n_grid, cfg.ambient_dim), dtype=np.float32)
    rows = []
    for mode, params in params_by_mode.items():
        _, cache = forward(params, jnp.asarray(z), jnp.asarray(t_grid), cfg, return_cache=True)
        c_skip = np.asarray(cache["c_skip"]).reshape(-1)
        c_out = np.asarray(cache["c_out"]).reshape(-1)
        for t, cs, co in zip(t_grid, c_skip, c_out):
            rows.append({"mode": mode, "t": float(t), "c_skip": float(cs), "c_out": float(co)})
    write_csv(run_dir / "analysis" / "longskip_coefficients.csv", rows)
    return rows


def evaluate_branch_metrics(
    run_dir: Path,
    cfg: LongSkipModelConfig,
    params_by_mode: dict[str, Any],
    n_eval: int,
    n_grid: int,
    seed: int,
):
    data = np.load(run_dir / "training_data_snapshot.npz")
    x0_all = np.asarray(data["x0"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(x0_all.shape[0], size=min(n_eval, x0_all.shape[0]), replace=False)
    x0 = x0_all[idx]
    eps_base = rng.normal(size=x0.shape).astype(np.float32)
    t_grid = np.linspace(0.02, 0.98, n_grid, dtype=np.float32)
    rows = []
    for mode, params in params_by_mode.items():
        for t in t_grid:
            t_vec = np.full((x0.shape[0],), float(t), dtype=np.float32)
            z_t = ((1.0 - t) * x0 + t * eps_base).astype(np.float32)
            out, cache = forward(params, jnp.asarray(z_t), jnp.asarray(t_vec), cfg, return_cache=True)
            out = np.asarray(out)
            c_skip = np.asarray(cache["c_skip"])
            c_out = np.asarray(cache["c_out"])
            net_out = np.asarray(cache["net_out"])
            skip_branch = c_skip * z_t
            net_branch = c_out * net_out
            skip_norm = np.linalg.norm(skip_branch, axis=1)
            net_norm = np.linalg.norm(net_branch, axis=1)
            out_norm = np.linalg.norm(out, axis=1)
            denom = np.maximum(skip_norm * net_norm, 1e-12)
            cos = np.sum(skip_branch * net_branch, axis=1) / denom
            rows.append({
                "mode": mode,
                "t": float(t),
                "mean_c_skip": float(np.mean(c_skip)),
                "mean_c_out": float(np.mean(c_out)),
                "mean_abs_c_skip": float(np.mean(np.abs(c_skip))),
                "mean_abs_c_out": float(np.mean(np.abs(c_out))),
                "mean_skip_norm": float(np.mean(skip_norm)),
                "mean_net_norm": float(np.mean(net_norm)),
                "mean_output_norm": float(np.mean(out_norm)),
                "skip_fraction": float(np.mean(skip_norm / (skip_norm + net_norm + 1e-12))),
                "branch_cosine": float(np.mean(cos)),
            })
    write_csv(run_dir / "analysis" / "longskip_branch_metrics.csv", rows)
    return rows


def _legend_inside(ax, loc="best"):
    ax.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="0.85", loc=loc)


def save(fig, run_dir: Path, name: str, save_pdf: bool):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coefficients(run_dir: Path, coeff_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], save_pdf: bool):
    setup_style()
    import pandas as pd

    coeff = pd.DataFrame(coeff_rows)
    branch = pd.DataFrame(branch_rows)
    paths = []

    for name, ylabel in [("c_skip", "Learned skip coefficient"), ("c_out", "Learned network coefficient")]:
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
        for mode in MODES:
            sub = coeff[coeff["mode"] == mode]
            ax.plot(sub["t"], sub[name], color=MODE_COLORS[mode], linewidth=2.2, label=MODE_LABELS[mode])
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
        if name == "c_out":
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.45, label="init c_out=1")
        ax.set_xlabel("Diffusion time t")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Long-Skip {name}(t)", fontweight="bold")
        ax.grid(alpha=0.25)
        _legend_inside(ax, loc="best")
        paths.append(save(fig, run_dir, f"longskip_{name}", save_pdf))

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for mode in MODES:
        sub = branch[branch["mode"] == mode]
        ax.plot(sub["t"], sub["skip_fraction"], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(sub) // 8), linewidth=2.0, label=MODE_LABELS[mode])
    ax.set_xlabel("Diffusion time t")
    ax.set_ylabel(r"Mean $||c_{skip}z_t||/(||c_{skip}z_t||+||c_{out}nnet||)$")
    ax.set_title("Actual Skip-Branch Fraction", fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    paths.append(save(fig, run_dir, "longskip_skip_fraction", save_pdf))

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for mode in MODES:
        sub = branch[branch["mode"] == mode]
        ax.plot(sub["t"], sub["branch_cosine"], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(sub) // 8), linewidth=2.0, label=MODE_LABELS[mode])
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("Diffusion time t")
    ax.set_ylabel("Mean cosine between skip and network branches")
    ax.set_title("Skip/Network Branch Alignment", fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    paths.append(save(fig, run_dir, "longskip_branch_cosine", save_pdf))

    for mode in MODES:
        sub = branch[branch["mode"] == mode]
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
        ax.plot(sub["t"], sub["mean_skip_norm"], color="#555555", linewidth=2.0, label=r"$||c_{skip}z_t||$")
        ax.plot(sub["t"], sub["mean_net_norm"], color=MODE_COLORS[mode], linewidth=2.2, label=r"$||c_{out}nnet||$")
        ax.plot(sub["t"], sub["mean_output_norm"], color="black", linewidth=1.6, linestyle="--", label=r"$||raw||$")
        ax.set_xlabel("Diffusion time t")
        ax.set_ylabel("Mean branch norm")
        ax.set_title(f"Long-Skip Branch Norms ({MODE_LABELS[mode]})", fontweight="bold")
        ax.grid(alpha=0.25)
        _legend_inside(ax, loc="best")
        paths.append(save(fig, run_dir, f"longskip_branch_norms_{mode}", save_pdf))

    return paths


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Analyze learned c_skip(t), c_out(t), and branch usage for a long-skip FCN run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-grid", type=int, default=101)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--seed", type=int, default=91_000)
    p.add_argument("--save-pdf", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = model_config_from_meta(run_dir)
    params_by_mode = load_params(run_dir)
    coeff_rows = evaluate_coefficients(run_dir, cfg, params_by_mode, args.n_grid)
    branch_rows = evaluate_branch_metrics(run_dir, cfg, params_by_mode, args.n_eval, args.n_grid, args.seed)
    paths = plot_coefficients(run_dir, coeff_rows, branch_rows, args.save_pdf)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
