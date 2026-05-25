from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .posthoc_analysis import ensure_representation_spectrum, ensure_sample_quality_metrics

MODE_LABELS = {"x": "x-pred", "v": "v-pred", "eps": "eps-pred"}
MODE_COLORS = {"x": "#0072B2", "v": "#009E73", "eps": "#D55E00"}
MODE_HATCHES = {"x": "", "v": "//", "eps": "xx"}
MODE_MARKERS = {"x": "o", "v": "s", "eps": "^"}
MODES = ["x", "v", "eps"]

MATRIX_LABELS = {
    "gradient": "Gradient",
    "momentum": "AdamW first moment",
    "update": "Actual update",
    "activation": "Activation",
    "residual": "Residual",
}

ANGLE_LABELS = {
    "adjacent_gradient": "Adjacent gradient angle (deg)",
    "adjacent_momentum": "Adjacent momentum angle (deg)",
    "adjacent_update": "Adjacent update angle (deg)",
    "adjacent_activation": "Adjacent activation angle (deg)",
    "adjacent_residual": "Adjacent residual angle (deg)",
    "gradient_vs_previous_momentum": "Gradient vs previous momentum angle (deg)",
}


def latest_run(output_root: str | Path = "results/clean_jax/runs") -> Path:
    root = Path(output_root)
    runs = [p for p in root.glob("*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found in {root}")
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.handlelength": 1.4,
    })


def _fig_dir(run_dir: Path) -> Path:
    p = run_dir / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, run_dir: Path, name: str, *, save_pdf: bool = False):
    out = _fig_dir(run_dir) / f"{name}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    return out


def _legend_above(ax, *, ncol: int = 3):
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        borderaxespad=0.0,
    )


def _legend_inside(ax, *, loc: str = "best", ncol: int = 1):
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        edgecolor="0.85",
        loc=loc,
        ncol=ncol,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.45,
    )


def _legend_top_row_inside(ax, *, ncol: int = 3):
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.72,
        edgecolor="0.85",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=ncol,
        borderpad=0.25,
        labelspacing=0.25,
        handletextpad=0.45,
        columnspacing=1.0,
    )


def _sample_metric_box(ax, quality):
    if quality is None:
        return
    text = (
        f"Chamfer: {_format_quality(float(quality['chamfer']))}\n"
        f"Precision: {float(quality['precision']):.2f}\n"
        f"Recall: {float(quality['recall']):.2f}"
    )
    text_fn = ax.text2D if getattr(ax, "name", "") == "3d" and hasattr(ax, "text2D") else ax.text
    text_fn(
        0.035,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.82",
            "alpha": 0.80,
        },
    )


def _annotate_bars(ax, bars, *, fmt="{:.0f}", y_offset_frac: float = 0.025):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset_frac * span,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
        )


def _hook_label(hook: str) -> str:
    return {"norm": "Norm Stream", "fanin": "AdaLN Fan-In"}.get(hook, hook)


def _sampling_label(sampling: str) -> str:
    if sampling == "mixed":
        return "Mixed t"
    if sampling.startswith("t="):
        return f"Fixed {sampling}"
    return sampling


def _metric_label(metric: str) -> str:
    return {"stable_rank": "Stable Rank", "rank95": "95% PCA Rank", "rank90": "90% PCA Rank"}.get(metric, metric)


def _sample_limits(samples) -> tuple[tuple[float, float], tuple[float, float]]:
    # Match the original notebook: sample quality panels use only the ground
    # truth window plus a 10% margin. Generated outliers are intentionally
    # clipped by the view instead of shrinking the clean manifold.
    xy = samples["training_2d"]
    gt_min = xy.min(axis=0)
    gt_max = xy.max(axis=0)
    margin = (gt_max - gt_min) * 0.10
    return (
        (float(gt_min[0] - margin[0]), float(gt_max[0] + margin[0])),
        (float(gt_min[1] - margin[1]), float(gt_max[1] + margin[1])),
    )


def _rolling_mean(values, window: int):
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return arr
    k = min(window, arr.size)
    if k % 2 == 0:
        k -= 1
    k = max(k, 3)
    return pd.Series(arr).rolling(window=k, center=True, min_periods=1).mean().to_numpy()


def plot_train_loss(run_dir: str | Path, *, save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "logs" / "loss.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in MODES:
        sub = df[df["mode"] == mode]
        ax.plot(sub["step"], sub["loss"], color=MODE_COLORS[mode], linewidth=0.9, alpha=0.22)
        smooth = _rolling_mean(sub["loss"], window=max(5, len(sub) // 35))
        ax.plot(sub["step"], smooth, color=MODE_COLORS[mode], label=MODE_LABELS[mode], linewidth=2.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Velocity loss")
    ax.set_title("Velocity Loss During Training", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.28, which="both")
    _legend_inside(ax, loc="best")
    path = save_figure(fig, run_dir, "train_loss", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_representation_bar(run_dir: str | Path, *, hook: str, sampling: str, metric: str, save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "analysis" / "representation_metrics.csv")
    sub = df[(df["hook"] == hook) & (df["sampling"] == sampling)]
    if sub.empty:
        raise ValueError(f"No representation rows for hook={hook}, sampling={sampling}")
    layers = sorted(sub["layer"].unique())
    x = np.arange(len(layers), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    all_bars = []
    all_vals = []
    for i, mode in enumerate(MODES):
        vals = [float(sub[(sub["mode"] == mode) & (sub["layer"] == layer)][metric].iloc[0]) for layer in layers]
        all_vals.extend(vals)
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=MODE_COLORS[mode],
            edgecolor="black",
            linewidth=0.6,
            hatch=MODE_HATCHES[mode],
            label=MODE_LABELS[mode],
            alpha=0.88,
        )
        all_bars.extend(bars)
    ymax = max(all_vals) if all_vals else 1.0
    if metric == "rank95":
        y_top = min(300.0, max(5.0, float(np.ceil(ymax * 1.26 + 8.0))))
        if y_top <= ymax:
            y_top = ymax + 8.0
    else:
        y_top = max(ymax * 1.32, ymax + 0.12)
    ax.set_ylim(0, y_top)
    _annotate_bars(ax, all_bars, fmt="{:.0f}" if metric == "rank95" else "{:.2f}")
    ax.set_xlabel("Residual block")
    ylabel = "Stable rank" if metric == "stable_rank" else "95% PCA rank"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Hidden {_metric_label(metric)}\n{_hook_label(hook)}, {_sampling_label(sampling)}", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    if metric == "rank95":
        ax.axhline(y=2, color="black", linestyle="--", alpha=0.55, linewidth=1.2, label="true dim")
    ax.grid(alpha=0.25, axis="y")
    _legend_top_row_inside(ax, ncol=4 if metric == "rank95" else 3)
    path = save_figure(fig, run_dir, f"repr_{hook}_{sampling.replace('=', '').replace('.', 'p')}_{metric}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_stability_bar(run_dir: str | Path, *, hook: str, t_value: float, save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "analysis" / "representation_stability.csv")
    sub = df[(df["hook"] == hook) & np.isclose(df["t"], t_value)]
    layers = sorted(sub["layer"].unique())
    x = np.arange(len(layers), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    all_bars = []
    all_vals = []
    for i, mode in enumerate(MODES):
        vals = [float(sub[(sub["mode"] == mode) & (sub["layer"] == layer)]["nsv"].iloc[0]) for layer in layers]
        all_vals.extend(vals)
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=MODE_COLORS[mode],
            edgecolor="black",
            linewidth=0.6,
            hatch=MODE_HATCHES[mode],
            label=MODE_LABELS[mode],
            alpha=0.88,
        )
        all_bars.extend(bars)
    ymax = max(all_vals) if all_vals else 1.0
    ax.set_ylim(0, ymax * 1.34 + 1e-9)
    _annotate_bars(ax, all_bars, fmt="{:.2f}")
    ax.set_xlabel("Residual block")
    ax.set_ylabel("Normalized noise variance")
    ax.set_title(f"Noise-Resampling Stability\n{_hook_label(hook)}, t={t_value:.1f}", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.grid(alpha=0.25, axis="y")
    _legend_top_row_inside(ax, ncol=3)
    path = save_figure(fig, run_dir, f"stability_{hook}_t{str(t_value).replace('.', 'p')}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _format_quality(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.01):
        return f"{value:.1e}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _sample_quality_for_legend(run_dir: Path, mode: str, *, space: str):
    ensure_sample_quality_metrics(run_dir)
    df = pd.read_csv(run_dir / "analysis" / "sample_quality_metrics.csv")
    row = df[(df["mode"] == mode) & (df["space"] == space)]
    if row.empty:
        return None
    return row.iloc[0]


def plot_samples(
    run_dir: str | Path,
    *,
    mode: str,
    quality_space: str = "ambient_highd",
    save_pdf: bool = False,
    show: bool = True,
):
    setup_style()
    run_dir = Path(run_dir)
    samples = np.load(run_dir / "analysis" / "samples.npz")
    train = samples["training_2d"]
    gen = samples[f"{mode}_2d"]
    quality = _sample_quality_for_legend(run_dir, mode, space=quality_space)
    xlim, ylim = _sample_limits(samples)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(train[:, 0], train[:, 1], s=1, c="blue", alpha=0.30, label="Ground truth", rasterized=True)
    ax.scatter(gen[:, 0], gen[:, 1], s=1, c="red", alpha=0.30, label="Generated", rasterized=True)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Generated Samples ({MODE_LABELS[mode]})", fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    _sample_metric_box(ax, quality)
    _legend_inside(ax, loc="upper right")
    path = save_figure(fig, run_dir, f"samples_{mode}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _pca_scores(x: np.ndarray, n_components: int = 3):
    arr = np.asarray(x, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    centered = arr - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T


def _set_3d_equalish(ax, points: np.ndarray):
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    lo = np.quantile(finite, 0.01, axis=0)
    hi = np.quantile(finite, 0.99, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_samples_pca3d(
    run_dir: str | Path,
    *,
    mode: str,
    max_points: int = 1200,
    save_pdf: bool = False,
    show: bool = True,
):
    setup_style()
    run_dir = Path(run_dir)
    samples = np.load(run_dir / "analysis" / "samples.npz")
    data = np.load(run_dir / "training_data_snapshot.npz")
    train = data["x0"].astype(np.float32)
    gen = samples[f"{mode}_highd"].astype(np.float32)
    n_train = min(max_points, train.shape[0])
    n_gen = min(max_points, gen.shape[0])
    train = train[:n_train]
    gen = gen[:n_gen]
    scores = _pca_scores(np.concatenate([train, gen], axis=0), n_components=3)
    train_scores = scores[:n_train]
    gen_scores = scores[n_train:]
    quality = _sample_quality_for_legend(run_dir, mode, space="ambient_highd")
    fig = plt.figure(figsize=(5.6, 5.1))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_scores[:, 0], train_scores[:, 1], train_scores[:, 2], s=2, c="blue", alpha=0.24, label="Ground truth", rasterized=True)
    ax.scatter(gen_scores[:, 0], gen_scores[:, 1], gen_scores[:, 2], s=2, c="red", alpha=0.28, label="Generated", rasterized=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA Samples ({MODE_LABELS[mode]})", fontweight="bold")
    _set_3d_equalish(ax, scores)
    ax.view_init(elev=24, azim=-55)
    _sample_metric_box(ax, quality)
    _legend_inside(ax, loc="upper right")
    path = save_figure(fig, run_dir, f"samples_{mode}_pca3d", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_representation_spectrum(
    run_dir: str | Path,
    *,
    hook: str,
    sampling: str,
    layer: int,
    max_components: int = 80,
    save_pdf: bool = False,
    show: bool = True,
):
    setup_style()
    run_dir = Path(run_dir)
    ensure_representation_spectrum(run_dir)
    df = pd.read_csv(run_dir / "analysis" / "representation_spectrum.csv")
    sub = df[(df["hook"] == hook) & (df["sampling"] == sampling) & (df["layer"] == layer)]
    if sub.empty:
        raise ValueError(f"No spectrum rows for hook={hook}, sampling={sampling}, layer={layer}")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("component")
        m = m[m["component"] <= max_components]
        ax.plot(
            m["component"],
            m["relative_singular_value"],
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            markevery=max(1, len(m) // 8),
            linewidth=1.8,
            label=MODE_LABELS[mode],
        )
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title(f"Hidden Spectrum (Block {layer})\n{_hook_label(hook)}, {_sampling_label(sampling)}", fontweight="bold", pad=8)
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="upper right")
    safe_sampling = sampling.replace("=", "").replace(".", "p")
    path = save_figure(fig, run_dir, f"spectrum_{hook}_{safe_sampling}_block{layer}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_sample_metric(run_dir: str | Path, *, metric: str = "rank95", save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "analysis" / "sample_metrics.csv")
    labels = ["training_data", *MODES]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    colors = ["#777777", *[MODE_COLORS[m] for m in MODES]]
    hatches = ["", *[MODE_HATCHES[m] for m in MODES]]
    vals = [float(df[df["mode"] == label][metric].iloc[0]) for label in labels]
    bars = ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.6, alpha=0.88)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ymax = max(vals) if vals else 1.0
    ax.set_ylim(0, ymax * 1.2 + 1e-9)
    _annotate_bars(ax, bars, fmt="{:.0f}" if metric == "rank95" else "{:.2f}")
    ax.set_xlabel("Point cloud")
    ax.set_ylabel("95% PCA rank" if metric == "rank95" else "Stable rank")
    ax.set_title(f"Generated Sample {_metric_label(metric)}", fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["data", "x-pred", "v-pred", "eps-pred"], rotation=20, ha="right")
    ax.grid(alpha=0.25, axis="y")
    path = save_figure(fig, run_dir, f"sample_metric_{metric}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_matrix_metric(run_dir: str | Path, *, matrix_kind: str, layer: str, metric: str, save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "logs" / "matrix_metrics.csv")
    sub = df[(df["matrix_kind"] == matrix_kind) & (df["layer"] == layer)]
    if sub.empty:
        raise ValueError(f"No matrix rows for matrix_kind={matrix_kind}, layer={layer}")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("step")
        ax.plot(m["step"], m[metric], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(m)//8), linewidth=1.6, label=MODE_LABELS[mode])
    ax.set_xlabel("Training step")
    ylabel = "Stable rank" if metric == "stable_rank" else "90% PCA rank"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{MATRIX_LABELS.get(matrix_kind, matrix_kind)}: {layer}", fontweight="bold")
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    path = save_figure(fig, run_dir, f"matrix_{matrix_kind}_{layer}_{metric}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_angle_metric(run_dir: str | Path, *, angle_kind: str, layer: str, save_pdf: bool = False, show: bool = True):
    setup_style()
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "logs" / "angle_metrics.csv")
    sub = df[(df["angle_kind"] == angle_kind) & (df["layer"] == layer)]
    if sub.empty:
        raise ValueError(f"No angle rows for angle_kind={angle_kind}, layer={layer}")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("step")
        ax.plot(m["step"], m["angle_deg"], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(m)//8), linewidth=1.6, label=MODE_LABELS[mode])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ANGLE_LABELS.get(angle_kind, "Principal angle (deg)"))
    ax.set_title(f"{ANGLE_LABELS.get(angle_kind, angle_kind)}: {layer}", fontweight="bold")
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    path = save_figure(fig, run_dir, f"angle_{angle_kind}_{layer}", save_pdf=save_pdf)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def generate_core_figures(run_dir: str | Path, *, save_pdf: bool = False, show: bool = False):
    run_dir = Path(run_dir)
    ensure_sample_quality_metrics(run_dir)
    ensure_representation_spectrum(run_dir)
    paths = [plot_train_loss(run_dir, save_pdf=save_pdf, show=show)]
    for hook in ["norm", "fanin"]:
        for metric in ["stable_rank", "rank95"]:
            paths.append(plot_representation_bar(run_dir, hook=hook, sampling="mixed", metric=metric, save_pdf=save_pdf, show=show))
        for t in [0.1, 0.5, 0.9]:
            paths.append(plot_representation_bar(run_dir, hook=hook, sampling=f"t={t:.1f}", metric="rank95", save_pdf=save_pdf, show=show))
            paths.append(plot_stability_bar(run_dir, hook=hook, t_value=t, save_pdf=save_pdf, show=show))
    for mode in MODES:
        paths.append(plot_samples(run_dir, mode=mode, save_pdf=save_pdf, show=show))
        paths.append(plot_samples_pca3d(run_dir, mode=mode, save_pdf=save_pdf, show=show))
    for metric in ["rank95", "stable_rank"]:
        paths.append(plot_sample_metric(run_dir, metric=metric, save_pdf=save_pdf, show=show))
    for hook in ["norm", "fanin"]:
        paths.append(plot_representation_spectrum(run_dir, hook=hook, sampling="mixed", layer=5, save_pdf=save_pdf, show=show))
    return paths


def generate_sampling_figures(run_dir: str | Path, *, save_pdf: bool = False, show: bool = False):
    """Generate only loss/sample figures for sampling-only runs such as UNet1D."""
    run_dir = Path(run_dir)
    ensure_sample_quality_metrics(run_dir)
    paths = [plot_train_loss(run_dir, save_pdf=save_pdf, show=show)]
    for mode in MODES:
        paths.append(plot_samples(run_dir, mode=mode, save_pdf=save_pdf, show=show))
        paths.append(plot_samples_pca3d(run_dir, mode=mode, save_pdf=save_pdf, show=show))
    for metric in ["rank95", "stable_rank"]:
        paths.append(plot_sample_metric(run_dir, metric=metric, save_pdf=save_pdf, show=show))
    return paths


def generate_gradient_figures(run_dir: str | Path, *, layers: Iterable[str], save_pdf: bool = False, show: bool = False):
    paths = []
    for layer in layers:
        for kind in ["gradient", "momentum", "update", "activation", "residual"]:
            for metric in ["stable_rank", "rank90"]:
                paths.append(plot_matrix_metric(run_dir, matrix_kind=kind, layer=layer, metric=metric, save_pdf=save_pdf, show=show))
        for angle_kind in ["adjacent_gradient", "gradient_vs_previous_momentum", "adjacent_activation", "adjacent_residual"]:
            paths.append(plot_angle_metric(run_dir, angle_kind=angle_kind, layer=layer, save_pdf=save_pdf, show=show))
    return paths
