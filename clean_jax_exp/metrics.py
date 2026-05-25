from __future__ import annotations

import math
from typing import Any

import numpy as np


def spectral_metrics(matrix: np.ndarray, *, center: bool, rank_threshold: float) -> dict[str, float]:
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    if center:
        a = a - a.mean(axis=0, keepdims=True)
    if min(a.shape) == 0:
        return {"stable_rank": 0.0, "rank_k": 0.0, "op_norm": 0.0, "fro_norm": 0.0, "top1_energy": 0.0}
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    energy = s * s
    total = float(energy.sum())
    op2 = float(energy[0]) if energy.size else 0.0
    if total <= 1e-20 or op2 <= 1e-20:
        return {"stable_rank": 0.0, "rank_k": 0.0, "op_norm": 0.0, "fro_norm": 0.0, "top1_energy": 0.0}
    p = energy / total
    rank_k = int(np.searchsorted(np.cumsum(p), rank_threshold) + 1)
    return {
        "stable_rank": float(total / op2),
        "rank_k": float(rank_k),
        "op_norm": float(math.sqrt(op2)),
        "fro_norm": float(math.sqrt(total)),
        "top1_energy": float(p[0]),
    }


def top_right_singular_vector(matrix: np.ndarray, *, center: bool) -> np.ndarray | None:
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    if center:
        a = a - a.mean(axis=0, keepdims=True)
    if min(a.shape) == 0 or np.linalg.norm(a) <= 1e-20:
        return None
    _, _, vh = np.linalg.svd(a, full_matrices=False)
    return vh[0].copy()


def sign_invariant_angle_deg(u: np.ndarray | None, v: np.ndarray | None) -> float:
    if u is None or v is None:
        return float("nan")
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-20:
        return float("nan")
    cos = abs(float(np.dot(u, v) / denom))
    cos = min(1.0, max(0.0, cos))
    return float(np.degrees(np.arccos(cos)))


def normalized_noise_variance(h: np.ndarray, n_clean: int, n_noise: int) -> float:
    arr = np.asarray(h, dtype=np.float64).reshape(n_clean, n_noise, -1)
    per_clean_var = arr.var(axis=1).sum(axis=1)
    denom = np.square(arr).sum(axis=2).mean()
    return float(per_clean_var.mean() / (denom + 1e-20))


def tree_get(tree: dict[str, Any], path: tuple[Any, ...]):
    cur: Any = tree
    for key in path:
        cur = cur[key]
    return cur
