from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clean_jax_exp.data import get_or_create_dataset, project_to_2d
from clean_jax_exp.metrics import spectral_metrics
from clean_jax_exp.posthoc_analysis import ensure_sample_quality_metrics

MODES = ("x", "v", "eps")


@dataclass(frozen=True)
class TorchUNetConfig:
    ambient_dim: int = 512
    patch_size: int = 4
    stride: int = 2
    base_channels: int = 56
    channel_mults: tuple[int, int, int] = (1, 2, 3)
    blocks_per_level: int = 2
    kernel_size: int = 3
    time_embed_dim: int = 256
    time_width: int = 256
    groups: int = 8
    zero_init_output: bool = True


@dataclass(frozen=True)
class TorchTrainConfig:
    steps: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    loss_every: int = 100
    print_every: int = 1000
    t_min: float = 1e-3
    t_sampling: str = "sigmoid_normal"


@dataclass(frozen=True)
class TorchAnalysisConfig:
    sample_n: int = 2048
    sample_steps: int = 100
    rank95_threshold: float = 0.95


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sinusoidal_embedding(t: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / float(half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResAdaGNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cfg: TorchUNetConfig):
        super().__init__()
        pad = cfg.kernel_size // 2
        self.conv0 = nn.Conv1d(in_ch, out_ch, cfg.kernel_size, padding=pad)
        self.norm = nn.GroupNorm(num_groups=min(cfg.groups, out_ch), num_channels=out_ch)
        self.ada = nn.Linear(cfg.time_width, 2 * out_ch)
        self.conv1 = nn.Conv1d(out_ch, out_ch, cfg.kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(cfg.groups, out_ch), num_channels=out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor):
        skip = self.skip(x)
        h = self.conv0(x)
        h = self.norm(h)
        gamma, beta = self.ada(t_cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        h = F.silu(h)
        h = self.conv1(h)
        h = F.silu(self.norm1(h))
        return (skip + h) / math.sqrt(2.0)


def block_sequence(in_ch: int, out_ch: int, n_blocks: int, cfg: TorchUNetConfig):
    layers = []
    cur = in_ch
    for _ in range(n_blocks):
        layers.append(ResAdaGNBlock(cur, out_ch, cfg))
        cur = out_ch
    return nn.ModuleList(layers)


class TinyUNet1D(nn.Module):
    def __init__(self, cfg: TorchUNetConfig):
        super().__init__()
        self.cfg = cfg
        c0 = cfg.base_channels * cfg.channel_mults[0]
        c1 = cfg.base_channels * cfg.channel_mults[1]
        c2 = cfg.base_channels * cfg.channel_mults[2]
        self.time_mlp0 = nn.Linear(cfg.time_embed_dim, cfg.time_width)
        self.time_mlp1 = nn.Linear(cfg.time_width, cfg.time_width)
        self.input_proj = nn.Linear(cfg.patch_size, c0)
        self.enc0 = block_sequence(c0, c0, cfg.blocks_per_level, cfg)
        self.enc1 = block_sequence(c0, c1, cfg.blocks_per_level, cfg)
        self.enc2 = block_sequence(c1, c2, cfg.blocks_per_level, cfg)
        self.mid = block_sequence(c2, c2, cfg.blocks_per_level, cfg)
        self.dec1 = block_sequence(c2 + c1, c1, cfg.blocks_per_level, cfg)
        self.dec0 = block_sequence(c1 + c0, c0, cfg.blocks_per_level, cfg)
        self.output_proj = nn.Linear(c0, cfg.patch_size)
        if cfg.zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def patchify(self, x: torch.Tensor):
        cfg = self.cfg
        if cfg.patch_size == 4 and cfg.stride == 2:
            return torch.stack(
                [
                    x[:, 0 : cfg.ambient_dim - 2 : 2],
                    x[:, 1 : cfg.ambient_dim - 1 : 2],
                    x[:, 2 : cfg.ambient_dim : 2],
                    x[:, 3 : cfg.ambient_dim + 1 : 2],
                ],
                dim=-1,
            )
        return x.unfold(dimension=1, size=cfg.patch_size, step=cfg.stride)

    def unpatchify(self, patches: torch.Tensor):
        cfg = self.cfg
        if cfg.patch_size == 4 and cfg.stride == 2:
            even = torch.cat(
                [
                    patches[:, 0:1, 0],
                    0.5 * (patches[:, 1:, 0] + patches[:, :-1, 2]),
                    patches[:, -1:, 2],
                ],
                dim=1,
            )
            odd = torch.cat(
                [
                    patches[:, 0:1, 1],
                    0.5 * (patches[:, 1:, 1] + patches[:, :-1, 3]),
                    patches[:, -1:, 3],
                ],
                dim=1,
            )
            return torch.stack([even, odd], dim=-1).reshape(patches.shape[0], cfg.ambient_dim)
        raise NotImplementedError("Generic unpatchify is intentionally not used for this experiment.")

    @staticmethod
    def downsample(x: torch.Tensor):
        if x.shape[-1] % 2 == 1:
            x = torch.cat([x, x[:, :, -1:]], dim=-1)
        b, c, l = x.shape
        return x.reshape(b, c, l // 2, 2).mean(dim=-1)

    @staticmethod
    def upsample_to(x: torch.Tensor, target_len: int):
        y = x.repeat_interleave(2, dim=-1)
        if y.shape[-1] >= target_len:
            return y[:, :, :target_len]
        pad = y[:, :, -1:].repeat(1, 1, target_len - y.shape[-1])
        return torch.cat([y, pad], dim=-1)

    @staticmethod
    def apply_blocks(blocks: nn.ModuleList, x: torch.Tensor, t_cond: torch.Tensor):
        h = x
        for block in blocks:
            h = block(h, t_cond)
        return h

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        t_emb = sinusoidal_embedding(t, self.cfg.time_embed_dim)
        t_cond = self.time_mlp1(F.silu(self.time_mlp0(t_emb)))
        h = self.input_proj(self.patchify(z_t)).transpose(1, 2)
        h0 = self.apply_blocks(self.enc0, h, t_cond)
        h1 = self.apply_blocks(self.enc1, self.downsample(h0), t_cond)
        h2 = self.apply_blocks(self.enc2, self.downsample(h1), t_cond)
        hm = self.apply_blocks(self.mid, h2, t_cond)
        u1 = torch.cat([self.upsample_to(hm, h1.shape[-1]), h1], dim=1)
        u1 = self.apply_blocks(self.dec1, u1, t_cond)
        u0 = torch.cat([self.upsample_to(u1, h0.shape[-1]), h0], dim=1)
        u0 = self.apply_blocks(self.dec0, u0, t_cond)
        patches = self.output_proj(u0.transpose(1, 2))
        return self.unpatchify(patches)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pred_to_velocity_torch(raw: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, mode: str, t_min: float):
    t_col = t[:, None].clamp_min(t_min)
    if mode == "v":
        return raw
    if mode == "x":
        return (z_t - raw) / t_col
    if mode == "eps":
        return (raw - z_t) / (1.0 - t_col)
    raise ValueError(mode)


def make_batch(x0_all: torch.Tensor, cfg: TorchTrainConfig, generator: torch.Generator, device: torch.device):
    idx = torch.randint(0, x0_all.shape[0], (cfg.batch_size,), generator=generator, device=device)
    x0 = x0_all[idx]
    if cfg.t_sampling == "sigmoid_normal":
        t = torch.randn((cfg.batch_size,), generator=generator, device=device).sigmoid()
    else:
        t = torch.rand((cfg.batch_size,), generator=generator, device=device) * (1.0 - 2.0 * cfg.t_min) + cfg.t_min
    t = t.clamp(cfg.t_min, 1.0 - cfg.t_min)
    eps = torch.randn(x0.shape, generator=generator, device=device)
    z_t = (1.0 - t[:, None]) * x0 + t[:, None] * eps
    return x0, eps, t, z_t


def loss_fn(model: TinyUNet1D, batch, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = model(z_t, t)
    v_target = eps - x0
    v_pred = pred_to_velocity_torch(raw, z_t, t, mode, t_min)
    return F.mse_loss(v_pred, v_target)


def train_mode(mode: str, init_state: dict[str, torch.Tensor], x0_all: torch.Tensor, model_cfg: TorchUNetConfig, train_cfg: TorchTrainConfig, device: torch.device, seed: int):
    model = TinyUNet1D(model_cfg).to(device)
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps, weight_decay=train_cfg.weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows = []
    for step in range(1, train_cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = make_batch(x0_all, train_cfg, generator, device)
        loss = loss_fn(model, batch, mode, train_cfg.t_min)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        clip_scale = min(1.0, train_cfg.grad_clip_norm / (float(grad_norm) + 1e-12))
        opt.step()
        if step == 1 or step % train_cfg.loss_every == 0 or step == train_cfg.steps:
            rows.append({"mode": mode, "step": step, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm), "clip_scale": clip_scale})
        if step == 1 or step % train_cfg.print_every == 0 or step == train_cfg.steps:
            print(f"[{mode}] step {step:6d}/{train_cfg.steps} loss={float(loss.detach().cpu()):.6f} grad_norm={float(grad_norm):.4f} clip={clip_scale:.3f}", flush=True)
    return model, opt, rows


@torch.no_grad()
def sample_model(model: TinyUNet1D, mode: str, n_samples: int, sample_steps: int, t_min: float, device: torch.device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn((n_samples, model.cfg.ambient_dim), generator=generator, device=device)
    times = torch.linspace(1.0 - t_min, t_min, sample_steps, device=device)
    for i in range(sample_steps - 1):
        t_now = times[i]
        t_next = times[i + 1]
        dt = t_now - t_next
        t_vec = torch.full((n_samples,), float(t_now), device=device)
        raw = model(z, t_vec)
        v = pred_to_velocity_torch(raw, z, t_vec, mode, t_min)
        z = z - dt * v
    return z.detach().cpu().numpy()


def run_sampling_analysis(run_dir: Path, models: dict[str, TinyUNet1D], data: dict[str, np.ndarray], train_cfg: TorchTrainConfig, analysis_cfg: TorchAnalysisConfig, device: torch.device):
    rows = []
    sample_arrays = {}
    train_metrics = spectral_metrics(data["x0"], center=True, rank_threshold=analysis_cfg.rank95_threshold)
    rows.append({"mode": "training_data", "stable_rank": train_metrics["stable_rank"], "rank95": train_metrics["rank_k"], "top1_energy": train_metrics["top1_energy"]})
    for mode, model in models.items():
        print(f"sampling: mode={mode}", flush=True)
        samples = sample_model(model, mode, analysis_cfg.sample_n, analysis_cfg.sample_steps, train_cfg.t_min, device, seed=30_000 + len(mode))
        sample_arrays[f"{mode}_highd"] = samples.astype(np.float32)
        sample_arrays[f"{mode}_2d"] = project_to_2d(samples, data["P"], data["mean"], data["std"]).astype(np.float32)
        sm = spectral_metrics(samples, center=True, rank_threshold=analysis_cfg.rank95_threshold)
        rows.append({"mode": mode, "stable_rank": sm["stable_rank"], "rank95": sm["rank_k"], "top1_energy": sm["top1_energy"]})
    sample_arrays["training_2d"] = data["data_2d"].astype(np.float32)
    np.savez_compressed(run_dir / "analysis" / "samples.npz", **sample_arrays)
    write_csv(run_dir / "analysis" / "sample_metrics.csv", rows)
    ensure_sample_quality_metrics(run_dir, force=True)
    return rows


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    model_cfg = TorchUNetConfig(
        ambient_dim=args.ambient_dim,
        patch_size=args.patch_size,
        stride=args.stride,
        base_channels=args.base_channels,
        blocks_per_level=args.blocks_per_level,
        kernel_size=args.kernel_size,
        time_embed_dim=args.time_embed_dim,
        time_width=args.time_width,
        groups=args.groups,
        zero_init_output=True,
    )
    train_cfg = TorchTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_every=args.loss_every,
        print_every=args.print_every,
        grad_clip_norm=args.grad_clip_norm,
    )
    analysis_cfg = TorchAnalysisConfig(sample_n=args.sample_n, sample_steps=args.sample_steps)

    torch.manual_seed(args.seed + 123)
    init_model = TinyUNet1D(model_cfg)
    init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    n_params = count_params(init_model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"torch_unet1d_D{args.ambient_dim}_adamw_b{args.base_channels}_k{args.kernel_size}_p{args.patch_size}_s{args.stride}_steps{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    if args.save_checkpoints:
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "experiment_type": "torch_unet1d_sampling_training",
        "optimizer": "AdamW",
        "device": str(device),
        "model_family": "TinyUNet1D_AdaGN",
        "model_config": asdict(model_cfg),
        "parameter_count": n_params,
        "patch_count": (args.ambient_dim - args.patch_size) // args.stride + 1,
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), "parameter_count": n_params, "patch_count": metadata["patch_count"]}, indent=2), flush=True)

    x0_all = torch.as_tensor(data["x0"], dtype=torch.float32, device=device)
    models: dict[str, TinyUNet1D] = {}
    all_rows: list[dict[str, Any]] = []
    for mode in MODES:
        print(f"\n{'=' * 90}\nTorch UNet1D training mode={mode}\n{'=' * 90}", flush=True)
        model, opt, rows = train_mode(mode, init_state, x0_all, model_cfg, train_cfg, device, seed=args.seed + 1000)
        models[mode] = model
        all_rows.extend(rows)
        if args.save_checkpoints:
            torch.save(
                {"mode": mode, "model": model.state_dict(), "optimizer": opt.state_dict(), "model_config": asdict(model_cfg), "train_config": asdict(train_cfg)},
                run_dir / "checkpoints" / f"{mode}_final.pt",
            )
    write_csv(run_dir / "logs" / "loss.csv", all_rows)
    print("\nPost-training sampling analysis", flush=True)
    run_sampling_analysis(run_dir, models, data, train_cfg, analysis_cfg, device)
    print(f"\nTorch UNet1D run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="PyTorch Tiny UNet1D + AdaGN toy diffusion sampling experiment")
    p.add_argument("--output-root", default="results/torch_unet1d_sampling")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=56)
    p.add_argument("--blocks-per-level", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--time-width", type=int, default=256)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--sample-n", type=int, default=2048)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--save-checkpoints", action="store_true")
    return p


if __name__ == "__main__":
    run_experiment(build_argparser().parse_args())
