from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .models import ModelConfig
from .train_gradient import (
    AnalysisConfig,
    checkpoint_load,
    collect_repr_for_batch,
    make_eval_batch,
)

MODES = ("x", "v", "eps")
HOOKS = ("norm", "fanin")


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _analysis_config(meta: dict[str, Any]) -> AnalysisConfig:
    raw = dict(meta.get("analysis_config", {}))
    names = {f.name for f in fields(AnalysisConfig)}
    raw = {k: v for k, v in raw.items() if k in names}
    if "repr_t_values" in raw:
        raw["repr_t_values"] = tuple(raw["repr_t_values"])
    return AnalysisConfig(**raw)


def _model_config(meta: dict[str, Any]) -> ModelConfig:
    raw = dict(meta.get("model_config", {}))
    names = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in names})


def _subsample_rows(x: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_points, replace=False)
    return arr[idx]


def _nearest_distances(a: np.ndarray, b: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    b_norm = np.sum(b * b, axis=1, dtype=np.float32)
    out = np.empty((a.shape[0],), dtype=np.float32)
    for start in range(0, a.shape[0], chunk_size):
        end = min(start + chunk_size, a.shape[0])
        block = a[start:end]
        dist2 = (
            np.sum(block * block, axis=1, keepdims=True, dtype=np.float32)
            + b_norm[None, :]
            - 2.0 * (block @ b.T)
        )
        out[start:end] = np.sqrt(np.maximum(np.min(dist2, axis=1), 0.0))
    return out


def _self_nn_radius(x: np.ndarray, *, quantile: float, chunk_size: int = 256) -> float:
    x = np.asarray(x, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, dtype=np.float32)
    nn = np.empty((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        block = x[start:end]
        dist2 = (
            np.sum(block * block, axis=1, keepdims=True, dtype=np.float32)
            + x_norm[None, :]
            - 2.0 * (block @ x.T)
        )
        row = np.arange(end - start)
        dist2[row, np.arange(start, end)] = np.inf
        nn[start:end] = np.sqrt(np.maximum(np.min(dist2, axis=1), 0.0))
    return float(np.quantile(nn, quantile))


def _sample_quality_one(train_eval: np.ndarray, gen: np.ndarray, *, radius: float) -> dict[str, float]:
    gen_eval = np.asarray(gen, dtype=np.float32)
    gen_to_train = _nearest_distances(gen_eval, train_eval)
    train_to_gen = _nearest_distances(train_eval, gen_eval)
    return {
        "radius": radius,
        "chamfer": float(np.mean(gen_to_train) + np.mean(train_to_gen)),
        "precision": float(np.mean(gen_to_train <= radius)),
        "recall": float(np.mean(train_to_gen <= radius)),
        "mean_gen_to_data": float(np.mean(gen_to_train)),
        "mean_data_to_gen": float(np.mean(train_to_gen)),
    }


def ensure_sample_quality_metrics(
    run_dir: str | Path,
    *,
    force: bool = False,
    highd_train_max: int = 2048,
    projected_train_max: int = 4096,
    radius_quantile: float = 0.95,
) -> Path:
    """Save simple point-cloud quality metrics used by the sample legends.

    Precision is the fraction of generated samples inside the data manifold
    radius. Recall is the fraction of data samples covered by generated samples.
    The radius is the 95th percentile of data-to-data nearest-neighbor distance.
    Chamfer is the symmetric mean nearest-neighbor distance.
    """
    run_dir = Path(run_dir)
    out = run_dir / "analysis" / "sample_quality_metrics.csv"
    if out.exists() and not force:
        return out

    sample_npz = np.load(run_dir / "analysis" / "samples.npz")
    data_npz = np.load(run_dir / "training_data_snapshot.npz")
    rows: list[dict[str, Any]] = []
    spaces = [
        ("projected_2d", sample_npz["training_2d"], projected_train_max),
        ("ambient_highd", data_npz["x0"], highd_train_max),
    ]
    for space, train, train_max in spaces:
        # Use one data subset and one data-scale radius per space, shared by all
        # prediction modes. Otherwise precision/recall are not directly
        # comparable across x/v/eps.
        train_eval = _subsample_rows(train, train_max, seed=71_000)
        radius = _self_nn_radius(train_eval, quantile=radius_quantile)
        for mode_idx, mode in enumerate(MODES):
            key = f"{mode}_2d" if space == "projected_2d" else f"{mode}_highd"
            if key not in sample_npz:
                continue
            metrics = _sample_quality_one(
                train_eval,
                sample_npz[key],
                radius=radius,
            )
            rows.append({
                "mode": mode,
                "space": space,
                "radius_quantile": radius_quantile,
                "train_points": int(train_eval.shape[0]),
                **metrics,
            })
    _write_csv(out, rows)
    return out


def _top_pca_basis(x: np.ndarray, k: int = 2) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(arr, full_matrices=False)
    return vh[:k].T


def _orthonormalize(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(x, dtype=np.float64))
    return q


def _principal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = _orthonormalize(a)
    qb = _orthonormalize(b)
    s = np.linalg.svd(qa.T @ qb, full_matrices=False, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def ensure_sample_subspace_metrics(run_dir: str | Path, *, force: bool = False) -> Path:
    """Compare sample PCA subspaces against the true normalized 2D data plane.

    The dataset is embedded as ``data_2d @ P.T`` and then coordinate-wise
    normalized by ``std``. Therefore the true two-dimensional linear subspace in
    training/sample coordinates is span(diag(1 / std) @ P), not span(P).
    """
    run_dir = Path(run_dir)
    out = run_dir / "analysis" / "sample_subspace_metrics.csv"
    if out.exists() and not force:
        return out

    sample_npz = np.load(run_dir / "analysis" / "samples.npz")
    data_npz = np.load(run_dir / "training_data_snapshot.npz")
    p = np.asarray(data_npz["P"], dtype=np.float64)
    std = np.asarray(data_npz["std"], dtype=np.float64)
    true_basis = _orthonormalize(p / std[:, None])

    rows: list[dict[str, Any]] = []
    clouds = {"training_data": np.asarray(data_npz["x0"], dtype=np.float32)}
    for mode in MODES:
        key = f"{mode}_highd"
        if key in sample_npz:
            clouds[mode] = np.asarray(sample_npz[key], dtype=np.float32)

    for label, cloud in clouds.items():
        sample_basis = _top_pca_basis(cloud, k=2)
        angles = _principal_angles_deg(sample_basis, true_basis)
        overlap = float(np.sum(np.cos(np.radians(angles)) ** 2) / 2.0)
        rows.append({
            "mode": label,
            "angle1_deg": float(angles[0]),
            "angle2_deg": float(angles[1]),
            "mean_angle_deg": float(np.mean(angles)),
            "max_angle_deg": float(np.max(angles)),
            "subspace_overlap": overlap,
        })
    _write_csv(out, rows)
    return out


def ensure_representation_spectrum(run_dir: str | Path, *, force: bool = False) -> Path:
    """Save singular-value spectra for every hidden representation matrix."""
    run_dir = Path(run_dir)
    out = run_dir / "analysis" / "representation_spectrum.csv"
    if out.exists() and not force:
        return out

    meta = _metadata(run_dir)
    model_cfg = _model_config(meta)
    analysis_cfg = _analysis_config(meta)
    x0 = np.load(run_dir / "training_data_snapshot.npz")["x0"]
    batches = [(None, *make_eval_batch(x0, analysis_cfg.repr_n_samples, 10_000, None))]
    for tv in analysis_cfg.repr_t_values:
        batches.append((float(tv), *make_eval_batch(x0, analysis_cfg.repr_n_samples, 10_000 + int(float(tv) * 1000), float(tv))))

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        ckpt = checkpoint_load(run_dir / "checkpoints" / f"{mode}_final.pkl")
        params = ckpt["params"]
        for _, label, t_out, z_t, t in batches:
            reps = collect_repr_for_batch(params, z_t, t, model_cfg, analysis_cfg.repr_batch_size)
            for hook in HOOKS:
                for layer_idx, h in enumerate(reps[hook], start=1):
                    centered = np.asarray(h, dtype=np.float64)
                    centered = centered - centered.mean(axis=0, keepdims=True)
                    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
                    energy = s * s
                    total = float(energy.sum())
                    cumulative = np.cumsum(energy / total) if total > 0 else np.zeros_like(energy)
                    s0 = float(s[0]) if s.size and s[0] > 0 else 1.0
                    for component, sv in enumerate(s, start=1):
                        efrac = float(energy[component - 1] / total) if total > 0 else 0.0
                        rows.append({
                            "mode": mode,
                            "sampling": label,
                            "t": t_out,
                            "hook": hook,
                            "layer": layer_idx,
                            "component": component,
                            "singular_value": float(sv),
                            "relative_singular_value": float(sv / s0),
                            "energy_fraction": efrac,
                            "cumulative_energy": float(cumulative[component - 1]) if cumulative.size else 0.0,
                        })
    _write_csv(out, rows)
    return out
