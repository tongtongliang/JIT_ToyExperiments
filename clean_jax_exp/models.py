from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    ambient_dim: int = 512
    width: int = 256
    depth: int = 5
    time_embed_dim: int = 256
    zero_init_output: bool = True


def silu(x):
    return x * jax.nn.sigmoid(x)


def xavier(key, out_dim: int, in_dim: int):
    scale = np.sqrt(2.0 / float(in_dim + out_dim))
    return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) * scale


def init_linear(key, out_dim: int, in_dim: int, *, zero: bool = False):
    if zero:
        w = jnp.zeros((out_dim, in_dim), dtype=jnp.float32)
    else:
        w = xavier(key, out_dim, in_dim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_params(key, cfg: ModelConfig):
    keys = list(jax.random.split(key, 4 + cfg.depth * 3))
    i = 0
    params: dict[str, Any] = {}
    params["time_mlp0"] = init_linear(keys[i], cfg.width, cfg.time_embed_dim); i += 1
    params["time_mlp1"] = init_linear(keys[i], cfg.width, cfg.width); i += 1
    params["input_proj"] = init_linear(keys[i], cfg.width, cfg.ambient_dim); i += 1
    blocks = []
    for _ in range(cfg.depth):
        block = {
            "mlp0": init_linear(keys[i], cfg.width, cfg.width),
            "mlp1": init_linear(keys[i + 1], cfg.width, cfg.width),
            "ada": init_linear(keys[i + 2], 3 * cfg.width, cfg.width, zero=True),
        }
        i += 3
        blocks.append(block)
    params["blocks"] = tuple(blocks)
    params["output_proj"] = init_linear(keys[i], cfg.ambient_dim, cfg.width, zero=cfg.zero_init_output)
    return params


def linear(x, layer):
    return x @ layer["w"].T + layer["b"]


def sinusoidal_embedding(t, embed_dim: int):
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * (-np.log(10000.0) / float(half_dim - 1)))
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def layer_norm(x, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def forward(params, z_t, t, cfg: ModelConfig, *, return_cache: bool = False):
    t_emb = sinusoidal_embedding(t, cfg.time_embed_dim)
    t_cond = linear(silu(linear(t_emb, params["time_mlp0"])), params["time_mlp1"])
    h = linear(z_t, params["input_proj"])
    caches = []
    for block in params["blocks"]:
        norm_h = layer_norm(h)
        gamma, beta, alpha = jnp.split(linear(t_cond, block["ada"]), 3, axis=-1)
        fanin = norm_h * (1.0 + gamma) + beta
        a1 = linear(fanin, block["mlp0"])
        relu = jax.nn.relu(a1)
        a2 = linear(relu, block["mlp1"])
        if return_cache:
            caches.append({"h_in": h, "norm": norm_h, "fanin": fanin, "a1": a1, "relu": relu, "a2": a2, "gamma": gamma, "beta": beta, "alpha": alpha})
        h = h + alpha * a2
    out = linear(h, params["output_proj"])
    if return_cache:
        return out, {"t_cond": t_cond, "input_activation": z_t, "final_h": h, "blocks": tuple(caches)}
    return out


def pred_to_velocity(raw, z_t, t, mode: str, t_min: float):
    t_col = jnp.clip(t[:, None], t_min, 1.0 - t_min)
    if mode == "v":
        return raw
    if mode == "x":
        return (z_t - raw) / t_col
    if mode == "eps":
        return (raw - z_t) / (1.0 - t_col)
    raise ValueError(f"unknown mode: {mode}")


def loss_fn(params, batch, cfg: ModelConfig, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = forward(params, z_t, t, cfg, return_cache=False)
    v_target = eps - x0
    v_pred = pred_to_velocity(raw, z_t, t, mode, t_min)
    return jnp.mean((v_pred - v_target) ** 2)


def tracked_layers(depth: int):
    layers = [("input_proj", ("input_proj", "w"), "input_proj")]
    for i in range(depth):
        layers.append((f"block{i + 1}_mlp0", ("blocks", i, "mlp0", "w"), f"block {i + 1} mlp fan-in"))
    layers.append(("output_proj", ("output_proj", "w"), "output_proj"))
    return layers


def numpy_forward_cache(params, z_t, t, cfg: ModelConfig):
    out, cache = forward(params, jnp.asarray(z_t), jnp.asarray(t), cfg, return_cache=True)
    return np.asarray(out), jax.tree_util.tree_map(lambda x: np.asarray(x), cache)
