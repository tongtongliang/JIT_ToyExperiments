from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .models import layer_norm, linear, pred_to_velocity, silu, sinusoidal_embedding, xavier


@dataclass(frozen=True)
class UFCNModelConfig:
    ambient_dim: int = 512
    width: int = 256
    depth: int = 5
    time_embed_dim: int = 256
    zero_init_output: bool = True
    skip_init: str = "identity_current"


def uvit_skip_pairs(depth: int) -> tuple[tuple[int, int], ...]:
    """Return U-ViT-style symmetric hidden skip pairs as 0-based block ids."""
    if depth < 3:
        return tuple()
    n_pairs = depth // 2
    return tuple((i, depth - 1 - i) for i in range(n_pairs))


def init_linear(key, out_dim: int, in_dim: int, *, zero: bool = False):
    if zero:
        w = jnp.zeros((out_dim, in_dim), dtype=jnp.float32)
    else:
        w = xavier(key, out_dim, in_dim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_skip_combine(width: int, *, init: str):
    """Linear projection after concat([current_h, encoder_skip]).

    U-ViT concatenates shallow encoder features into decoder blocks and then
    projects back to the model width. The identity-current initialization keeps
    the initial hidden stream identical to the no-skip FCN while allowing the
    skip half to learn during training.
    """
    if init == "identity_current":
        w = jnp.concatenate(
            [jnp.eye(width, dtype=jnp.float32), jnp.zeros((width, width), dtype=jnp.float32)],
            axis=1,
        )
    elif init == "zero":
        w = jnp.zeros((width, 2 * width), dtype=jnp.float32)
    else:
        raise ValueError(f"unknown skip_init: {init}")
    b = jnp.zeros((width,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_params(key, cfg: UFCNModelConfig):
    pairs = uvit_skip_pairs(cfg.depth)
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
    params["skip_projs"] = tuple(init_skip_combine(cfg.width, init=cfg.skip_init) for _ in pairs)
    params["output_proj"] = init_linear(keys[i], cfg.ambient_dim, cfg.width, zero=cfg.zero_init_output)
    return params


def forward(params, z_t, t, cfg: UFCNModelConfig, *, return_cache: bool = False):
    pairs = uvit_skip_pairs(cfg.depth)
    target_to_source = {target: source for source, target in pairs}
    target_to_proj = {target: proj_idx for proj_idx, (_, target) in enumerate(pairs)}

    t_emb = sinusoidal_embedding(t, cfg.time_embed_dim)
    t_cond = linear(silu(linear(t_emb, params["time_mlp0"])), params["time_mlp1"])
    h = linear(z_t, params["input_proj"])
    skip_states: dict[int, Any] = {}
    caches = []

    for i, block in enumerate(params["blocks"]):
        skip_source = target_to_source.get(i)
        skip_used = skip_source is not None
        if skip_used:
            skip_h = skip_states[skip_source]
            h_before_skip = h
            h = linear(jnp.concatenate([h, skip_h], axis=-1), params["skip_projs"][target_to_proj[i]])
        else:
            skip_h = None
            h_before_skip = h

        norm_h = layer_norm(h)
        gamma, beta, alpha = jnp.split(linear(t_cond, block["ada"]), 3, axis=-1)
        fanin = norm_h * (1.0 + gamma) + beta
        a1 = linear(fanin, block["mlp0"])
        relu = jax.nn.relu(a1)
        a2 = linear(relu, block["mlp1"])
        if return_cache:
            caches.append({
                "h_in": h,
                "h_before_skip": h_before_skip,
                "skip_h": skip_h if skip_h is not None else jnp.zeros_like(h),
                "skip_used": jnp.asarray(skip_used),
                "norm": norm_h,
                "fanin": fanin,
                "a1": a1,
                "relu": relu,
                "a2": a2,
                "gamma": gamma,
                "beta": beta,
                "alpha": alpha,
            })
        h = h + alpha * a2
        for source, _target in pairs:
            if i == source:
                skip_states[source] = h

    out = linear(h, params["output_proj"])
    if return_cache:
        return out, {"t_cond": t_cond, "input_activation": z_t, "final_h": h, "blocks": tuple(caches)}
    return out


def loss_fn(params, batch, cfg: UFCNModelConfig, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = forward(params, z_t, t, cfg, return_cache=False)
    v_target = eps - x0
    v_pred = pred_to_velocity(raw, z_t, t, mode, t_min)
    return jnp.mean((v_pred - v_target) ** 2)


def numpy_forward_cache(params, z_t, t, cfg: UFCNModelConfig):
    out, cache = forward(params, jnp.asarray(z_t), jnp.asarray(t), cfg, return_cache=True)
    return np.asarray(out), jax.tree_util.tree_map(lambda x: np.asarray(x), cache)


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))
