from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .models import layer_norm, linear, pred_to_velocity, silu, sinusoidal_embedding, xavier


@dataclass(frozen=True)
class SingleFCNModelConfig:
    ambient_dim: int = 512
    width: int = 256
    depth: int = 10
    time_embed_dim: int = 256
    zero_init_output: bool = True


def init_linear(key, out_dim: int, in_dim: int, *, zero: bool = False):
    if zero:
        w = jnp.zeros((out_dim, in_dim), dtype=jnp.float32)
    else:
        w = xavier(key, out_dim, in_dim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_params(key, cfg: SingleFCNModelConfig):
    keys = list(jax.random.split(key, 4 + cfg.depth * 2))
    i = 0
    params: dict[str, Any] = {}
    params["time_mlp0"] = init_linear(keys[i], cfg.width, cfg.time_embed_dim); i += 1
    params["time_mlp1"] = init_linear(keys[i], cfg.width, cfg.width); i += 1
    params["input_proj"] = init_linear(keys[i], cfg.width, cfg.ambient_dim); i += 1
    blocks = []
    for _ in range(cfg.depth):
        block = {
            "linear": init_linear(keys[i], cfg.width, cfg.width),
            "ada": init_linear(keys[i + 1], 3 * cfg.width, cfg.width, zero=True),
        }
        i += 2
        blocks.append(block)
    params["blocks"] = tuple(blocks)
    params["output_proj"] = init_linear(keys[i], cfg.ambient_dim, cfg.width, zero=cfg.zero_init_output)
    return params


def forward(params, z_t, t, cfg: SingleFCNModelConfig, *, return_cache: bool = False):
    t_emb = sinusoidal_embedding(t, cfg.time_embed_dim)
    t_cond = linear(silu(linear(t_emb, params["time_mlp0"])), params["time_mlp1"])
    h = linear(z_t, params["input_proj"])
    caches = []
    for block in params["blocks"]:
        norm_h = layer_norm(h)
        gamma, beta, alpha = jnp.split(linear(t_cond, block["ada"]), 3, axis=-1)
        fanin = norm_h * (1.0 + gamma) + beta
        preact = linear(fanin, block["linear"])
        activation = jax.nn.relu(preact)
        if return_cache:
            caches.append({
                "h_in": h,
                "norm": norm_h,
                "fanin": fanin,
                "preact": preact,
                "activation": activation,
                "gamma": gamma,
                "beta": beta,
                "alpha": alpha,
            })
        h = h + alpha * activation
    out = linear(h, params["output_proj"])
    if return_cache:
        return out, {"t_cond": t_cond, "input_activation": z_t, "final_h": h, "blocks": tuple(caches)}
    return out


def loss_fn(params, batch, cfg: SingleFCNModelConfig, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = forward(params, z_t, t, cfg, return_cache=False)
    v_target = eps - x0
    v_pred = pred_to_velocity(raw, z_t, t, mode, t_min)
    return jnp.mean((v_pred - v_target) ** 2)


def numpy_forward_cache(params, z_t, t, cfg: SingleFCNModelConfig):
    out, cache = forward(params, jnp.asarray(z_t), jnp.asarray(t), cfg, return_cache=True)
    return np.asarray(out), jax.tree_util.tree_map(lambda x: np.asarray(x), cache)


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))
