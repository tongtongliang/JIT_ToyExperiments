from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from clean_jax_exp.metrics import spectral_metrics
from clean_jax_exp.visualize import MODE_COLORS, MODE_LABELS, MODE_MARKERS, MODES, setup_style
from run_transformer1d_torch_experiment import TorchTransformerConfig, TinyAdaLNTransformer1D, modulate


STREAM_LABELS = {
    "patch_embed": "Patch Embed",
    "attn_pre_norm": "Attention Pre-Norm Stream",
    "attn_norm": "Attention Norm Output",
    "attn_fanin": "Attention AdaLN Fan-In",
    "mlp_pre_norm": "MLP Pre-Norm Stream",
    "mlp_norm": "MLP Norm Output",
    "mlp_fanin": "MLP AdaLN Fan-In",
    "final_pre_norm": "Final Pre-Norm Stream",
    "final_norm": "Final Norm Output",
    "final_fanin": "Final AdaLN Fan-In",
}


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def config_from_checkpoint(checkpoint: dict[str, Any]) -> TorchTransformerConfig:
    cfg = dict(checkpoint["model_config"])
    allowed = TorchTransformerConfig.__dataclass_fields__.keys()
    return TorchTransformerConfig(**{k: cfg[k] for k in allowed if k in cfg})


def make_eval_batch(x0_all: np.ndarray, *, n_eval: int, t_value: float | None, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    idx = rng.choice(x0_all.shape[0], size=n_eval, replace=False)
    x0_np = x0_all[idx].astype(np.float32)
    x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    eps = torch.randn(x0.shape, generator=gen, device=device)
    if t_value is None:
        t = torch.randn((n_eval,), generator=gen, device=device).sigmoid()
        t_label = "mixed"
    else:
        t = torch.full((n_eval,), float(t_value), dtype=torch.float32, device=device)
        t_label = f"t={t_value:.1f}"
    z_t = (1.0 - t[:, None]) * x0 + t[:, None] * eps
    return z_t, t, t_label


@torch.no_grad()
def collect_patch_streams_manual(model: TinyAdaLNTransformer1D, z_t: torch.Tensor, t: torch.Tensor, streams: set[str]):
    # This mirrors TinyAdaLNTransformer1D.forward while exposing token clouds at
    # the normalization and AdaLN fan-in boundaries. Each recorded tensor has
    # shape [batch, patches, width] and is flattened later to patch point clouds.
    from run_transformer1d_torch_experiment import sinusoidal_embedding

    records: dict[tuple[int, str], np.ndarray] = {}
    t_emb = sinusoidal_embedding(t, model.cfg.time_embed_dim)
    t_cond = model.time_mlp1(F.silu(model.time_mlp0(t_emb)))
    h = model.patch_embed(model.patchify(z_t)) + model.pos_embed
    if "patch_embed" in streams:
        records[(-1, "patch_embed")] = h.detach().cpu().numpy()

    for layer_idx, block in enumerate(model.blocks):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.ada(t_cond).chunk(6, dim=-1)

        if "attn_pre_norm" in streams:
            records[(layer_idx, "attn_pre_norm")] = h.detach().cpu().numpy()
        attn_norm = block.norm_attn(h)
        if "attn_norm" in streams:
            records[(layer_idx, "attn_norm")] = attn_norm.detach().cpu().numpy()
        attn_fanin = modulate(attn_norm, shift_a, scale_a)
        if "attn_fanin" in streams:
            records[(layer_idx, "attn_fanin")] = attn_fanin.detach().cpu().numpy()
        h = h + gate_a[:, None, :] * block.attention(attn_fanin)

        if "mlp_pre_norm" in streams:
            records[(layer_idx, "mlp_pre_norm")] = h.detach().cpu().numpy()
        mlp_norm = block.norm_mlp(h)
        if "mlp_norm" in streams:
            records[(layer_idx, "mlp_norm")] = mlp_norm.detach().cpu().numpy()
        mlp_fanin = modulate(mlp_norm, shift_m, scale_m)
        if "mlp_fanin" in streams:
            records[(layer_idx, "mlp_fanin")] = mlp_fanin.detach().cpu().numpy()
        h = h + gate_m[:, None, :] * block.mlp1(F.gelu(block.mlp0(mlp_fanin)))

    if "final_pre_norm" in streams:
        records[(model.cfg.depth, "final_pre_norm")] = h.detach().cpu().numpy()
    final_norm = model.final_norm(h)
    if "final_norm" in streams:
        records[(model.cfg.depth, "final_norm")] = final_norm.detach().cpu().numpy()
    shift, scale = model.final_ada(t_cond).chunk(2, dim=-1)
    final_fanin = modulate(final_norm, shift, scale)
    if "final_fanin" in streams:
        records[(model.cfg.depth, "final_fanin")] = final_fanin.detach().cpu().numpy()

    return records


def summarize_patch_cloud(arr: np.ndarray, rank_threshold: float):
    matrix = arr.reshape(-1, arr.shape[-1])
    metrics = spectral_metrics(matrix, center=True, rank_threshold=rank_threshold)
    return {
        "stable_rank": metrics["stable_rank"],
        "rank95": metrics["rank_k"],
        "top1_energy": metrics["top1_energy"],
        "n_points": int(matrix.shape[0]),
        "width": int(matrix.shape[1]),
    }


def load_model(run_dir: Path, mode: str, device: torch.device):
    ckpt = torch.load(run_dir / "checkpoints" / f"{mode}_final.pt", map_location=device)
    cfg = config_from_checkpoint(ckpt)
    model = TinyAdaLNTransformer1D(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def analyze_run(args: argparse.Namespace):
    run_dir = Path(args.run_dir)
    device = pick_device(args.device)
    data = np.load(run_dir / "training_data_snapshot.npz")
    x0_all = data["x0"]
    streams = set(args.streams.split(","))
    rows: list[dict[str, Any]] = []

    t_values: list[float | None] = [None] + [float(x) for x in args.t_values.split(",") if x]
    for mode in MODES:
        print(f"loading mode={mode} on {device}", flush=True)
        model = load_model(run_dir, mode, device)
        for t_idx, t_value in enumerate(t_values):
            z_t, t, t_label = make_eval_batch(x0_all, n_eval=args.n_eval, t_value=t_value, seed=args.seed + 1000 * t_idx, device=device)
            print(f"collecting mode={mode} condition={t_label}", flush=True)
            records = collect_patch_streams_manual(model, z_t, t, streams)
            for (layer, stream), arr in records.items():
                summary = summarize_patch_cloud(arr, args.rank_threshold)
                rows.append(
                    {
                        "mode": mode,
                        "sampling": t_label,
                        "layer": int(layer),
                        "stream": stream,
                        **summary,
                    }
                )
            del z_t, t, records
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    out_path = run_dir / "analysis" / "transformer_patch_representation_metrics.csv"
    write_csv(out_path, rows)
    print(f"wrote {out_path}", flush=True)
    return out_path


def _save(fig, run_dir: Path, name: str, *, save_pdf: bool):
    fig_dir = run_dir / "figures" / "transformer_hidden"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_lines(run_dir: Path, *, stream: str, sampling: str, metric: str, save_pdf: bool):
    setup_style()
    df = pd.read_csv(run_dir / "analysis" / "transformer_patch_representation_metrics.csv")
    sub = df[(df["stream"] == stream) & (df["sampling"] == sampling)].copy()
    if sub.empty:
        raise ValueError(f"No rows for stream={stream}, sampling={sampling}")
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("layer")
        ax.plot(
            m["layer"],
            m[metric],
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            linewidth=2.0,
            markersize=5.0,
            label=MODE_LABELS[mode],
        )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Stable rank" if metric == "stable_rank" else "95% PCA rank")
    ax.set_title(f"{STREAM_LABELS.get(stream, stream)} · {sampling}", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="0.85", loc="best")
    safe = sampling.replace("=", "").replace(".", "p")
    return _save(fig, run_dir, f"patch_{stream}_{safe}_{metric}", save_pdf=save_pdf)


def plot_fixed_t_heatmap(run_dir: Path, *, stream: str, mode: str, metric: str, save_pdf: bool):
    setup_style()
    df = pd.read_csv(run_dir / "analysis" / "transformer_patch_representation_metrics.csv")
    sub = df[(df["stream"] == stream) & (df["mode"] == mode) & (df["sampling"] != "mixed")].copy()
    if sub.empty:
        raise ValueError(f"No fixed-t rows for stream={stream}, mode={mode}")
    sub["t"] = sub["sampling"].str.replace("t=", "", regex=False).astype(float)
    table = sub.pivot(index="t", columns="layer", values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([str(int(x)) for x in table.columns])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in table.index])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Fixed diffusion time t")
    ax.set_title(f"{MODE_LABELS[mode]} · {STREAM_LABELS.get(stream, stream)}", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Stable rank" if metric == "stable_rank" else "95% PCA rank")
    return _save(fig, run_dir, f"patch_{mode}_{stream}_fixed_t_{metric}_heatmap", save_pdf=save_pdf)


def make_plots(args: argparse.Namespace):
    run_dir = Path(args.run_dir)
    paths = []
    key_streams = [s for s in ("attn_pre_norm", "attn_fanin", "mlp_pre_norm", "mlp_fanin", "final_fanin") if s in set(args.streams.split(","))]
    for stream in key_streams:
        for metric in ("stable_rank", "rank95"):
            paths.append(plot_metric_lines(run_dir, stream=stream, sampling="mixed", metric=metric, save_pdf=args.save_pdf))
    for stream in ("attn_fanin", "mlp_fanin", "final_fanin"):
        if stream in set(args.streams.split(",")):
            for mode in MODES:
                paths.append(plot_fixed_t_heatmap(run_dir, stream=stream, mode=mode, metric="stable_rank", save_pdf=args.save_pdf))
                paths.append(plot_fixed_t_heatmap(run_dir, stream=stream, mode=mode, metric="rank95", save_pdf=args.save_pdf))
    print("\n".join(str(p) for p in paths), flush=True)


def build_argparser():
    p = argparse.ArgumentParser(description="Posthoc patch hidden-representation analysis for TinyAdaLNTransformer1D checkpoints.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-eval", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--t-values", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--rank-threshold", type=float, default=0.95)
    p.add_argument(
        "--streams",
        default="patch_embed,attn_pre_norm,attn_norm,attn_fanin,mlp_pre_norm,mlp_norm,mlp_fanin,final_pre_norm,final_norm,final_fanin",
    )
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--save-pdf", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.skip_analysis:
        analyze_run(args)
    if not args.skip_plots:
        make_plots(args)
