from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from clean_jax_exp.data import get_or_create_dataset
from scripts.architectures.run_mixer1d_torch_experiment import (
    TinyAdaLNMixer1D,
    TorchMixerConfig,
)
from scripts.architectures.run_transformer1d_torch_experiment import (
    MODES,
    TinyAdaLNTransformer1D,
    TorchTrainConfig,
    TorchTransformerConfig,
    count_params,
    make_batch,
    pick_device,
    pred_to_velocity_torch,
    sinusoidal_embedding,
    write_csv,
)
from scripts.architectures.run_unet1d_torch_experiment import (
    TinyUNet1D,
    TorchUNetConfig,
)


MODE_COLORS = {"x": "#1f77b4", "v": "#009E73", "eps": "#D55E00"}
MODE_LABELS = {"x": "x-pred", "v": "v-pred", "eps": "eps-pred"}


@dataclass(frozen=True)
class TorchFCNConfig:
    ambient_dim: int = 512
    width: int = 256
    depth: int = 5
    time_embed_dim: int = 256
    zero_init_output: bool = True


class AdaLNFCNBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ada = nn.Linear(width, 3 * width)
        self.mlp0 = nn.Linear(width, width)
        self.mlp1 = nn.Linear(width, width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h: torch.Tensor, t_cond: torch.Tensor):
        gamma, beta, alpha = self.ada(t_cond).chunk(3, dim=-1)
        fanin = self.norm(h) * (1.0 + gamma) + beta
        return h + alpha * self.mlp1(F.relu(self.mlp0(fanin)))


class TinyAdaLNFCN(nn.Module):
    def __init__(self, cfg: TorchFCNConfig):
        super().__init__()
        self.cfg = cfg
        self.time_mlp0 = nn.Linear(cfg.time_embed_dim, cfg.width)
        self.time_mlp1 = nn.Linear(cfg.width, cfg.width)
        self.input_proj = nn.Linear(cfg.ambient_dim, cfg.width)
        self.blocks = nn.ModuleList([AdaLNFCNBlock(cfg.width) for _ in range(cfg.depth)])
        self.output_proj = nn.Linear(cfg.width, cfg.ambient_dim)
        if cfg.zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        t_emb = sinusoidal_embedding(t, self.cfg.time_embed_dim)
        t_cond = self.time_mlp1(F.silu(self.time_mlp0(t_emb)))
        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, t_cond)
        return self.output_proj(h)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def loss_one_sample(
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    model: nn.Module,
    x0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    z_t: torch.Tensor,
    mode: str,
    t_min: float,
) -> torch.Tensor:
    raw = functional_call(model, (params, buffers), (z_t.unsqueeze(0), t.unsqueeze(0))).squeeze(0)
    v_target = eps - x0
    v_pred = pred_to_velocity_torch(raw.unsqueeze(0), z_t.unsqueeze(0), t.unsqueeze(0), mode, t_min).squeeze(0)
    return torch.mean((v_pred - v_target) ** 2)


def batch_loss(model: nn.Module, batch, mode: str, t_min: float) -> torch.Tensor:
    x0, eps, t, z_t = batch
    raw = model(z_t, t)
    v_target = eps - x0
    v_pred = pred_to_velocity_torch(raw, z_t, t, mode, t_min)
    return F.mse_loss(v_pred, v_target)


def zeros_like_named_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p) for name, p in params.items()}


def tree_add_(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor], scale: float = 1.0):
    for name in dst:
        dst[name].add_(src[name], alpha=scale)


def tree_norm_sq(tree: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for value in tree.values():
        term = torch.sum(value * value)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("empty parameter tree")
    return total


def per_sample_grad_stats(
    model: nn.Module,
    batch,
    mode: str,
    t_min: float,
    chunk_size: int,
    eps_denom: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    x0, eps, t, z_t = batch
    batch_size = x0.shape[0]
    sum_grad = zeros_like_named_params(params)
    sum_grad_norm_sq = torch.zeros((), device=x0.device, dtype=x0.dtype)

    grad_fn = grad(loss_one_sample)
    vmap_grad_fn = vmap(grad_fn, in_dims=(None, None, None, 0, 0, 0, 0, None, None))

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        grads = vmap_grad_fn(
            params,
            buffers,
            model,
            x0[start:end],
            eps[start:end],
            t[start:end],
            z_t[start:end],
            mode,
            t_min,
        )
        for name, g in grads.items():
            sum_grad[name].add_(g.sum(dim=0))
            sum_grad_norm_sq = sum_grad_norm_sq + torch.sum(g.reshape(g.shape[0], -1) ** 2)
        del grads

    mean_grad = {name: value / float(batch_size) for name, value in sum_grad.items()}
    mean_grad_norm_sq = tree_norm_sq(mean_grad)
    mean_sample_grad_norm_sq = sum_grad_norm_sq / float(batch_size)
    if batch_size > 1:
        cov_trace = (float(batch_size) / float(batch_size - 1)) * (mean_sample_grad_norm_sq - mean_grad_norm_sq)
    else:
        cov_trace = torch.zeros_like(mean_grad_norm_sq)
    cov_trace = torch.clamp(cov_trace, min=0.0)
    gns = cov_trace / (mean_grad_norm_sq + eps_denom)
    stats = {
        "mean_grad_norm_sq": float(mean_grad_norm_sq.detach().cpu()),
        "mean_grad_norm": float(torch.sqrt(mean_grad_norm_sq).detach().cpu()),
        "mean_sample_grad_norm_sq": float(mean_sample_grad_norm_sq.detach().cpu()),
        "cov_trace": float(cov_trace.detach().cpu()),
        "gradient_noise_scale": float(gns.detach().cpu()),
    }
    return mean_grad, stats


def set_model_grads_from_mean(model: nn.Module, mean_grad: dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        param.grad = mean_grad[name].detach().clone()


def microbatch_grad_stats(
    model: nn.Module,
    batch,
    mode: str,
    t_min: float,
    microbatch_size: int,
    eps_denom: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Estimate sample-level GNS from gradients of equal-size microbatches.

    If a microbatch gradient is the average of m independent sample gradients,
    then Cov(g_microbatch) = Cov(g_sample) / m. We estimate the covariance
    trace across microbatch gradients and multiply by m to recover the
    sample-level covariance trace. This is much cheaper than exact per-sample
    gradients for compute-heavy models.
    """
    x0, eps, t, z_t = batch
    batch_size = x0.shape[0]
    if batch_size % microbatch_size != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by microbatch_size={microbatch_size}")
    params_tuple = tuple(model.parameters())
    param_names = [name for name, _ in model.named_parameters()]
    sum_grad = {name: torch.zeros_like(param) for name, param in zip(param_names, params_tuple)}
    sum_micro_grad_norm_sq = torch.zeros((), device=x0.device, dtype=x0.dtype)
    n_micro = batch_size // microbatch_size

    for start in range(0, batch_size, microbatch_size):
        end = start + microbatch_size
        micro_batch = (x0[start:end], eps[start:end], t[start:end], z_t[start:end])
        loss = batch_loss(model, micro_batch, mode, t_min)
        grads = torch.autograd.grad(loss, params_tuple, retain_graph=False, create_graph=False)
        micro_norm_sq = torch.zeros((), device=x0.device, dtype=x0.dtype)
        for name, g in zip(param_names, grads):
            sum_grad[name].add_(g)
            micro_norm_sq = micro_norm_sq + torch.sum(g * g)
        sum_micro_grad_norm_sq = sum_micro_grad_norm_sq + micro_norm_sq
        del grads

    mean_grad = {name: value / float(n_micro) for name, value in sum_grad.items()}
    mean_grad_norm_sq = tree_norm_sq(mean_grad)
    mean_micro_grad_norm_sq = sum_micro_grad_norm_sq / float(n_micro)
    if n_micro > 1:
        micro_cov_trace = (float(n_micro) / float(n_micro - 1)) * (mean_micro_grad_norm_sq - mean_grad_norm_sq)
    else:
        micro_cov_trace = torch.zeros_like(mean_grad_norm_sq)
    micro_cov_trace = torch.clamp(micro_cov_trace, min=0.0)
    cov_trace = float(microbatch_size) * micro_cov_trace
    gns = cov_trace / (mean_grad_norm_sq + eps_denom)
    stats = {
        "mean_grad_norm_sq": float(mean_grad_norm_sq.detach().cpu()),
        "mean_grad_norm": float(torch.sqrt(mean_grad_norm_sq).detach().cpu()),
        "mean_sample_grad_norm_sq": float("nan"),
        "cov_trace": float(cov_trace.detach().cpu()),
        "gradient_noise_scale": float(gns.detach().cpu()),
        "microbatch_cov_trace": float(micro_cov_trace.detach().cpu()),
        "microbatch_size": float(microbatch_size),
        "num_microbatches": float(n_micro),
    }
    return mean_grad, stats


def build_model(model_name: str, args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    if model_name == "fcn":
        cfg = TorchFCNConfig(
            ambient_dim=args.ambient_dim,
            width=args.fcn_width,
            depth=args.fcn_depth,
            time_embed_dim=args.time_embed_dim,
            zero_init_output=True,
        )
        return TinyAdaLNFCN(cfg), asdict(cfg)
    if model_name == "mixer":
        cfg = TorchMixerConfig(
            ambient_dim=args.ambient_dim,
            patch_size=args.mixer_patch_size,
            dim=args.mixer_dim,
            depth=args.mixer_depth,
            token_mlp_width=args.mixer_token_mlp_width,
            channel_mlp_width=args.mixer_channel_mlp_width,
            time_embed_dim=args.time_embed_dim,
            time_width=args.mixer_time_width,
            zero_init_output=True,
        )
        return TinyAdaLNMixer1D(cfg), asdict(cfg)
    if model_name == "transformer":
        cfg = TorchTransformerConfig(
            ambient_dim=args.ambient_dim,
            patch_size=args.transformer_patch_size,
            dim=args.transformer_dim,
            depth=args.transformer_depth,
            heads=args.transformer_heads,
            mlp_width=args.transformer_mlp_width,
            time_embed_dim=args.time_embed_dim,
            time_width=args.transformer_time_width,
            zero_init_output=True,
            attention_impl=args.transformer_attention_impl,
        )
        return TinyAdaLNTransformer1D(cfg), asdict(cfg)
    if model_name == "unet":
        cfg = TorchUNetConfig(
            ambient_dim=args.ambient_dim,
            patch_size=args.unet_patch_size,
            stride=args.unet_stride,
            base_channels=args.unet_base_channels,
            channel_mults=tuple(args.unet_channel_mults),
            blocks_per_level=args.unet_blocks_per_level,
            kernel_size=args.unet_kernel_size,
            time_embed_dim=args.time_embed_dim,
            time_width=args.unet_time_width,
            groups=args.unet_groups,
            zero_init_output=True,
        )
        return TinyUNet1D(cfg), asdict(cfg)
    raise ValueError(f"unknown model: {model_name}")


def plot_gns(run_dir: Path, save_pdf: bool = False):
    import pandas as pd

    df = pd.read_csv(run_dir / "logs" / "gradient_noise_scale.csv")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_name in sorted(df["model"].unique()):
        sub_model = df[df["model"] == model_name]
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for mode in MODES:
            sub = sub_model[sub_model["mode"] == mode].sort_values("step")
            if sub.empty:
                continue
            ax.plot(
                sub["step"],
                sub["gradient_noise_scale"],
                color=MODE_COLORS[mode],
                linewidth=2.0,
                label=MODE_LABELS[mode],
            )
        ax.set_title(f"Gradient Noise Scale ({model_name})", fontsize=15, weight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient noise scale")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
        fig.tight_layout()
        path = fig_dir / f"gradient_noise_scale_{model_name}.png"
        fig.savefig(path, dpi=180)
        if save_pdf:
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)

    for mode in MODES:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for model_name in sorted(df["model"].unique()):
            sub = df[(df["model"] == model_name) & (df["mode"] == mode)].sort_values("step")
            if sub.empty:
                continue
            ax.plot(sub["step"], sub["gradient_noise_scale"], linewidth=2.0, label=model_name)
        ax.set_title(f"Gradient Noise Scale ({MODE_LABELS[mode]})", fontsize=15, weight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient noise scale")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
        fig.tight_layout()
        path = fig_dir / f"gradient_noise_scale_compare_{mode}.png"
        fig.savefig(path, dpi=180)
        if save_pdf:
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    models = parse_csv(args.models)
    modes = parse_csv(args.modes)
    invalid_models = sorted(set(models) - {"fcn", "mixer", "transformer", "unet"})
    invalid_modes = sorted(set(modes) - set(MODES))
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}")
    if invalid_modes:
        raise ValueError(f"Unknown modes: {invalid_modes}")

    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)
    x0_all = torch.as_tensor(data["x0"], dtype=torch.float32, device=device)
    train_cfg = TorchTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_every=1,
        print_every=args.print_every,
        grad_clip_norm=args.grad_clip_norm,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"gns_D{args.ambient_dim}_B{args.batch_size}_s{args.steps}_"
        f"models{'-'.join(models)}_modes{'-'.join(modes)}_seed{args.seed}_{ts}"
    )
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_type": "gradient_noise_scale",
        "data_path": str(data_path),
        "models": list(models),
        "modes": list(modes),
        "device": str(device),
        "metric": "tr(per-sample gradient covariance) / ||mean per-sample gradient||^2",
        "estimator": args.estimator,
        "batch_size": args.batch_size,
        "per_sample_chunk_size": args.per_sample_chunk_size,
        "microbatch_size": args.microbatch_size,
        "train_config": asdict(train_cfg),
        "fcn_config": {
            "width": args.fcn_width,
            "depth": args.fcn_depth,
            "time_embed_dim": args.time_embed_dim,
        },
        "mixer_config": {
            "patch_size": args.mixer_patch_size,
            "dim": args.mixer_dim,
            "depth": args.mixer_depth,
            "token_mlp_width": args.mixer_token_mlp_width,
            "channel_mlp_width": args.mixer_channel_mlp_width,
            "time_width": args.mixer_time_width,
        },
        "transformer_config": {
            "patch_size": args.transformer_patch_size,
            "dim": args.transformer_dim,
            "depth": args.transformer_depth,
            "heads": args.transformer_heads,
            "mlp_width": args.transformer_mlp_width,
            "time_width": args.transformer_time_width,
            "attention_impl": args.transformer_attention_impl,
        },
        "unet_config": {
            "patch_size": args.unet_patch_size,
            "stride": args.unet_stride,
            "base_channels": args.unet_base_channels,
            "channel_mults": list(args.unet_channel_mults),
            "blocks_per_level": args.unet_blocks_per_level,
            "kernel_size": args.unet_kernel_size,
            "time_width": args.unet_time_width,
            "groups": args.unet_groups,
        },
        "notes": "Per-sample gradients are reduced immediately into sum gradients and sum squared norms; no per-sample gradient matrix is saved.",
    }

    rows: list[dict[str, Any]] = []
    for model_name in models:
        torch.manual_seed(args.seed + 123)
        init_model, model_cfg = build_model(model_name, args)
        init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
        parameter_count = count_params(init_model)
        metadata[f"{model_name}_parameter_count"] = parameter_count
        metadata[f"{model_name}_model_config"] = model_cfg
        for mode in modes:
            print(f"\n{'=' * 90}\nGNS model={model_name} mode={mode}\n{'=' * 90}", flush=True)
            model, _ = build_model(model_name, args)
            model.load_state_dict(init_state)
            model.to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.lr,
                betas=(train_cfg.beta1, train_cfg.beta2),
                eps=train_cfg.eps,
                weight_decay=train_cfg.weight_decay,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 1000)
            for step in range(1, args.steps + 1):
                batch = make_batch(x0_all, train_cfg, generator, device)
                with torch.no_grad():
                    loss_value = float(batch_loss(model, batch, mode, train_cfg.t_min).detach().cpu())
                opt.zero_grad(set_to_none=True)
                if args.estimator == "exact":
                    mean_grad, stats = per_sample_grad_stats(
                        model,
                        batch,
                        mode,
                        train_cfg.t_min,
                        args.per_sample_chunk_size,
                        args.gns_eps,
                    )
                elif args.estimator == "microbatch":
                    mean_grad, stats = microbatch_grad_stats(
                        model,
                        batch,
                        mode,
                        train_cfg.t_min,
                        args.microbatch_size,
                        args.gns_eps,
                    )
                else:
                    raise ValueError(f"unknown estimator: {args.estimator}")
                set_model_grads_from_mean(model, mean_grad)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                clip_scale = min(1.0, train_cfg.grad_clip_norm / (float(grad_norm) + 1e-12))
                opt.step()
                row = {
                    "model": model_name,
                    "mode": mode,
                    "step": step,
                    "loss": loss_value,
                    "grad_norm": float(grad_norm),
                    "clip_scale": clip_scale,
                    **stats,
                }
                rows.append(row)
                if step == 1 or step % args.print_every == 0 or step == args.steps:
                    print(
                        f"[{model_name}/{mode}] step {step:4d}/{args.steps} "
                        f"loss={loss_value:.6f} gns={stats['gradient_noise_scale']:.3e} "
                        f"cov={stats['cov_trace']:.3e} mean_norm={stats['mean_grad_norm']:.3e} "
                        f"clip={clip_scale:.3f}",
                        flush=True,
                    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_csv(run_dir / "logs" / "gradient_noise_scale.csv", rows)
    paths = plot_gns(run_dir, save_pdf=args.save_pdf)
    print(json.dumps({"run_dir": str(run_dir), "figures": [str(p) for p in paths]}, indent=2), flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Measure per-sample gradient noise scale for toy diffusion prediction modes.")
    p.add_argument("--output-root", default="results/gradient_noise_scale")
    p.add_argument("--run-name", default=None)
    p.add_argument("--models", default="fcn,mixer", help="Comma-separated subset/order of: fcn,mixer,transformer,unet")
    p.add_argument("--modes", default="x,v,eps", help="Comma-separated subset/order of: x,v,eps")
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--per-sample-chunk-size", type=int, default=4)
    p.add_argument("--estimator", choices=["exact", "microbatch"], default="exact")
    p.add_argument("--microbatch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--gns-eps", type=float, default=1e-12)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--fcn-width", type=int, default=256)
    p.add_argument("--fcn-depth", type=int, default=5)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--mixer-patch-size", type=int, default=8)
    p.add_argument("--mixer-dim", type=int, default=128)
    p.add_argument("--mixer-depth", type=int, default=5)
    p.add_argument("--mixer-token-mlp-width", type=int, default=128)
    p.add_argument("--mixer-channel-mlp-width", type=int, default=512)
    p.add_argument("--mixer-time-width", type=int, default=256)
    p.add_argument("--transformer-patch-size", type=int, default=8)
    p.add_argument("--transformer-dim", type=int, default=128)
    p.add_argument("--transformer-depth", type=int, default=5)
    p.add_argument("--transformer-heads", type=int, default=1)
    p.add_argument("--transformer-mlp-width", type=int, default=512)
    p.add_argument("--transformer-time-width", type=int, default=256)
    p.add_argument("--transformer-attention-impl", choices=("torch", "manual"), default="torch")
    p.add_argument("--unet-patch-size", type=int, default=4)
    p.add_argument("--unet-stride", type=int, default=2)
    p.add_argument("--unet-base-channels", type=int, default=56)
    p.add_argument("--unet-channel-mults", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--unet-blocks-per-level", type=int, default=2)
    p.add_argument("--unet-kernel-size", type=int, default=3)
    p.add_argument("--unet-time-width", type=int, default=256)
    p.add_argument("--unet-groups", type=int, default=8)
    p.add_argument("--save-pdf", action="store_true")
    return p


if __name__ == "__main__":
    run_experiment(build_argparser().parse_args())
