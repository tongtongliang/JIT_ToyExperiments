from __future__ import annotations

import argparse
import json
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
from run_transformer1d_torch_experiment import (
    MODES,
    TorchAnalysisConfig,
    TorchTrainConfig,
    count_params,
    make_batch,
    modulate,
    pick_device,
    pred_to_velocity_torch,
    sinusoidal_embedding,
    write_csv,
)


@dataclass(frozen=True)
class TorchMixerConfig:
    ambient_dim: int = 512
    patch_size: int = 8
    dim: int = 128
    depth: int = 5
    token_mlp_width: int = 128
    channel_mlp_width: int = 512
    time_embed_dim: int = 256
    time_width: int = 256
    zero_init_output: bool = True


class AdaLNMixerBlock(nn.Module):
    def __init__(self, cfg: TorchMixerConfig, n_patches: int):
        super().__init__()
        self.cfg = cfg
        self.n_patches = n_patches
        self.norm_token = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.norm_channel = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.ada = nn.Linear(cfg.time_width, 6 * cfg.dim)
        self.token0 = nn.Linear(n_patches, cfg.token_mlp_width)
        self.token1 = nn.Linear(cfg.token_mlp_width, n_patches)
        self.channel0 = nn.Linear(cfg.dim, cfg.channel_mlp_width)
        self.channel1 = nn.Linear(cfg.channel_mlp_width, cfg.dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def token_mlp(self, x: torch.Tensor):
        # MLP-Mixer token mixing: same token-mixing MLP is applied to every
        # channel after transposing [batch, tokens, channels] -> [batch, channels, tokens].
        y = x.transpose(1, 2)
        y = self.token1(F.gelu(self.token0(y)))
        return y.transpose(1, 2)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor):
        shift_t, scale_t, gate_t, shift_c, scale_c, gate_c = self.ada(t_cond).chunk(6, dim=-1)
        h = modulate(self.norm_token(x), shift_t, scale_t)
        x = x + gate_t[:, None, :] * self.token_mlp(h)
        h = modulate(self.norm_channel(x), shift_c, scale_c)
        x = x + gate_c[:, None, :] * self.channel1(F.gelu(self.channel0(h)))
        return x


class TinyAdaLNMixer1D(nn.Module):
    def __init__(self, cfg: TorchMixerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.ambient_dim % cfg.patch_size != 0:
            raise ValueError("ambient_dim must be divisible by patch_size")
        self.n_patches = cfg.ambient_dim // cfg.patch_size
        self.time_mlp0 = nn.Linear(cfg.time_embed_dim, cfg.time_width)
        self.time_mlp1 = nn.Linear(cfg.time_width, cfg.time_width)
        self.patch_embed = nn.Linear(cfg.patch_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, cfg.dim))
        self.blocks = nn.ModuleList([AdaLNMixerBlock(cfg, self.n_patches) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.final_ada = nn.Linear(cfg.time_width, 2 * cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, cfg.patch_size)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        if cfg.zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def patchify(self, x: torch.Tensor):
        return x.reshape(x.shape[0], self.n_patches, self.cfg.patch_size)

    def unpatchify(self, patches: torch.Tensor):
        return patches.reshape(patches.shape[0], self.cfg.ambient_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        t_emb = sinusoidal_embedding(t, self.cfg.time_embed_dim)
        t_cond = self.time_mlp1(F.silu(self.time_mlp0(t_emb)))
        h = self.patch_embed(self.patchify(z_t)) + self.pos_embed
        for block in self.blocks:
            h = block(h, t_cond)
        shift, scale = self.final_ada(t_cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        patches = self.output_proj(h)
        return self.unpatchify(patches)


def param_breakdown(model: nn.Module) -> dict[str, int]:
    items = list(model.named_parameters())
    groups = {
        "total": sum(p.numel() for _, p in items),
        "time": sum(p.numel() for name, p in items if name.startswith("time_mlp")),
        "block_ada": sum(p.numel() for name, p in items if ".ada." in name),
        "final_ada": sum(p.numel() for name, p in items if name.startswith("final_ada")),
        "patch_io_pos": sum(p.numel() for name, p in items if name.startswith(("patch_embed", "output_proj", "pos_embed"))),
        "token_mlp": sum(p.numel() for name, p in items if ".token" in name),
        "channel_mlp": sum(p.numel() for name, p in items if ".channel" in name),
    }
    groups["without_adaln_time"] = groups["total"] - groups["time"] - groups["block_ada"] - groups["final_ada"]
    return {k: int(v) for k, v in groups.items()}


def loss_fn(model: TinyAdaLNMixer1D, batch, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = model(z_t, t)
    v_target = eps - x0
    v_pred = pred_to_velocity_torch(raw, z_t, t, mode, t_min)
    return F.mse_loss(v_pred, v_target)


def train_mode(mode: str, init_state: dict[str, torch.Tensor], x0_all: torch.Tensor, model_cfg: TorchMixerConfig, train_cfg: TorchTrainConfig, device: torch.device, seed: int):
    model = TinyAdaLNMixer1D(model_cfg).to(device)
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
def sample_model(model: TinyAdaLNMixer1D, mode: str, n_samples: int, sample_steps: int, t_min: float, device: torch.device, seed: int):
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


def run_sampling_analysis(run_dir: Path, models: dict[str, TinyAdaLNMixer1D], data: dict[str, np.ndarray], train_cfg: TorchTrainConfig, analysis_cfg: TorchAnalysisConfig, device: torch.device):
    rows = []
    sample_arrays = {}
    train_metrics = spectral_metrics(data["x0"], center=True, rank_threshold=analysis_cfg.rank95_threshold)
    rows.append({"mode": "training_data", "stable_rank": train_metrics["stable_rank"], "rank95": train_metrics["rank_k"], "top1_energy": train_metrics["top1_energy"]})
    for mode, model in models.items():
        print(f"sampling: mode={mode}", flush=True)
        samples = sample_model(model, mode, analysis_cfg.sample_n, analysis_cfg.sample_steps, train_cfg.t_min, device, seed=40_000 + len(mode))
        sample_arrays[f"{mode}_highd"] = samples.astype(np.float32)
        sample_arrays[f"{mode}_2d"] = project_to_2d(samples, data["P"], data["mean"], data["std"]).astype(np.float32)
        sm = spectral_metrics(samples, center=True, rank_threshold=analysis_cfg.rank95_threshold)
        rows.append({"mode": mode, "stable_rank": sm["stable_rank"], "rank95": sm["rank_k"], "top1_energy": sm["top1_energy"]})
    sample_arrays["training_2d"] = data["data_2d"].astype(np.float32)
    np.savez_compressed(run_dir / "analysis" / "samples.npz", **sample_arrays)
    write_csv(run_dir / "analysis" / "sample_metrics.csv", rows)
    ensure_sample_quality_metrics(run_dir, force=True)


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    model_cfg = TorchMixerConfig(
        ambient_dim=args.ambient_dim,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        token_mlp_width=args.token_mlp_width,
        channel_mlp_width=args.channel_mlp_width,
        time_embed_dim=args.time_embed_dim,
        time_width=args.time_width,
        zero_init_output=True,
    )
    train_cfg = TorchTrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, loss_every=args.loss_every, print_every=args.print_every, grad_clip_norm=args.grad_clip_norm)
    analysis_cfg = TorchAnalysisConfig(sample_n=args.sample_n, sample_steps=args.sample_steps)

    torch.manual_seed(args.seed + 123)
    init_model = TinyAdaLNMixer1D(model_cfg)
    init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    breakdown = param_breakdown(init_model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"torch_mixer1d_D{args.ambient_dim}_adamw_p{args.patch_size}_d{args.dim}_L{args.depth}"
        f"_tm{args.token_mlp_width}_cm{args.channel_mlp_width}_steps{args.steps}_seed{args.seed}_{ts}"
    )
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
        "experiment_type": "torch_mixer1d_sampling_training",
        "optimizer": "AdamW",
        "device": str(device),
        "model_family": "TinyAdaLNMixer1D",
        "model_config": asdict(model_cfg),
        "parameter_count": breakdown["total"],
        "parameter_breakdown": breakdown,
        "patch_count": args.ambient_dim // args.patch_size,
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), **breakdown, "patch_count": metadata["patch_count"]}, indent=2), flush=True)

    x0_all = torch.as_tensor(data["x0"], dtype=torch.float32, device=device)
    models: dict[str, TinyAdaLNMixer1D] = {}
    all_rows: list[dict[str, Any]] = []
    for mode in MODES:
        print(f"\n{'=' * 90}\nTorch Mixer1D training mode={mode}\n{'=' * 90}", flush=True)
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
    print(f"\nTorch Mixer1D run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="PyTorch Tiny MLP-Mixer1D + AdaLN-zero toy diffusion sampling experiment")
    p.add_argument("--output-root", default="results/torch_mixer1d_sampling")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--token-mlp-width", type=int, default=128)
    p.add_argument("--channel-mlp-width", type=int, default=512)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--time-width", type=int, default=256)
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
