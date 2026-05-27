from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .models import layer_norm, linear, pred_to_velocity, silu, sinusoidal_embedding, xavier


@dataclass(frozen=True)
class LongSkipModelConfig:
    ambient_dim: int = 512
    width: int = 256
    depth: int = 5
    time_embed_dim: int = 256
    zero_init_output: bool = True


def init_linear(key, out_dim: int, in_dim: int, *, zero: bool = False):
    if zero:
        w = jnp.zeros((out_dim, in_dim), dtype=jnp.float32)
    else:
        w = xavier(key, out_dim, in_dim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_params(key, cfg: LongSkipModelConfig):
    keys = list(jax.random.split(key, 5 + cfg.depth * 3))
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
    params["output_proj"] = init_linear(keys[i], cfg.ambient_dim, cfg.width, zero=cfg.zero_init_output); i += 1
    # The long-skip controller predicts two scalars per sample from t_cond.
    # It is zero-initialized and interpreted as c_skip = raw_skip,
    # c_out = 1 + raw_out, so initialization exactly matches the baseline
    # zero-output FCN while still allowing gradients into output_proj.
    params["skip_head"] = init_linear(keys[i], 2, cfg.width, zero=True)
    return params


def forward(params, z_t, t, cfg: LongSkipModelConfig, *, return_cache: bool = False):
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
    net_out = linear(h, params["output_proj"])
    skip_raw = linear(t_cond, params["skip_head"])
    c_skip = skip_raw[:, 0:1]
    c_out = 1.0 + skip_raw[:, 1:2]
    out = c_skip * z_t + c_out * net_out
    if return_cache:
        return out, {"t_cond": t_cond, "input_activation": z_t, "final_h": h, "net_out": net_out, "c_skip": c_skip, "c_out": c_out, "blocks": tuple(caches)}
    return out


def loss_fn(params, batch, cfg: LongSkipModelConfig, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = forward(params, z_t, t, cfg, return_cache=False)
    v_target = eps - x0
    v_pred = pred_to_velocity(raw, z_t, t, mode, t_min)
    return jnp.mean((v_pred - v_target) ** 2)


def numpy_forward_cache(params, z_t, t, cfg: LongSkipModelConfig):
    out, cache = forward(params, jnp.asarray(z_t), jnp.asarray(t), cfg, return_cache=True)
    return np.asarray(out), jax.tree_util.tree_map(lambda x: np.asarray(x), cache)


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))
