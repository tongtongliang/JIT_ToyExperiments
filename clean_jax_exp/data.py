from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_swiss_roll


def generate_swiss_roll_2d(n_samples: int, noise: float, seed: int) -> np.ndarray:
    data_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return data_3d[:, [0, 2]].astype(np.float32)


def make_projection(ambient_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(ambient_dim, 2)).astype(np.float64)
    q, _ = np.linalg.qr(raw)
    return q[:, :2].astype(np.float32)


def embed_and_normalize(data_2d: np.ndarray, ambient_dim: int, seed: int):
    p = make_projection(ambient_dim, seed)
    x = data_2d @ p.T
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    x_norm = ((x - mean) / std).astype(np.float32)
    return x_norm, p, mean.squeeze(0), std.squeeze(0)


def project_to_2d(x_norm: np.ndarray, p: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = x_norm * std[None, :] + mean[None, :]
    return (x @ p).astype(np.float32)


def get_or_create_dataset(
    output_root: Path,
    ambient_dim: int,
    n_samples: int,
    noise: float,
    seed: int,
) -> tuple[Path, dict[str, np.ndarray]]:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"swissroll_D{ambient_dim}_n{n_samples}_seed{seed}.npz"
    if path.exists():
        loaded = dict(np.load(path))
        return path, loaded

    data_2d = generate_swiss_roll_2d(n_samples=n_samples, noise=noise, seed=seed)
    x0, p, mean, std = embed_and_normalize(data_2d, ambient_dim=ambient_dim, seed=seed)
    np.savez_compressed(
        path,
        x0=x0,
        data_2d=data_2d,
        P=p,
        mean=mean,
        std=std,
        ambient_dim=np.array(ambient_dim, dtype=np.int32),
        n_samples=np.array(n_samples, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
    )
    return path, dict(np.load(path))
