from __future__ import annotations

import argparse
import csv
import json
import math
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
class TorchTransformerConfig:
    ambient_dim: int = 512
    patch_size: int = 8
    dim: int = 128
    depth: int = 5
    heads: int = 1
    mlp_width: int = 512
    time_embed_dim: int = 256
    time_width: int = 256
    zero_init_output: bool = True
    attention_impl: str = "torch"


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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, cfg: TorchTransformerConfig):
        super().__init__()
        if cfg.dim % cfg.heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.cfg = cfg
        self.norm_attn = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.ada = nn.Linear(cfg.time_width, 6 * cfg.dim)
        if cfg.attention_impl == "torch":
            self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, dropout=0.0, batch_first=True)
        elif cfg.attention_impl == "manual":
            self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
            self.attn_out = nn.Linear(cfg.dim, cfg.dim)
        else:
            raise ValueError("attention_impl must be 'torch' or 'manual'")
        self.mlp0 = nn.Linear(cfg.dim, cfg.mlp_width)
        self.mlp1 = nn.Linear(cfg.mlp_width, cfg.dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def attention(self, x: torch.Tensor):
        if self.cfg.attention_impl == "torch":
            out, _ = self.attn(x, x, x, need_weights=False)
            return out
        b, n, d = x.shape
        h = self.cfg.heads
        head_dim = d // h
        qkv = self.qkv(x).reshape(b, n, 3, h, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = logits.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.attn_out(out)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(t_cond).chunk(6, dim=-1)
        h = modulate(self.norm_attn(x), shift_a, scale_a)
        x = x + gate_a[:, None, :] * self.attention(h)
        h = modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m[:, None, :] * self.mlp1(F.gelu(self.mlp0(h)))
        return x


class TinyAdaLNTransformer1D(nn.Module):
    def __init__(self, cfg: TorchTransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.ambient_dim % cfg.patch_size != 0:
            raise ValueError("ambient_dim must be divisible by patch_size")
        self.n_patches = cfg.ambient_dim // cfg.patch_size
        self.time_mlp0 = nn.Linear(cfg.time_embed_dim, cfg.time_width)
        self.time_mlp1 = nn.Linear(cfg.time_width, cfg.time_width)
        self.patch_embed = nn.Linear(cfg.patch_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, cfg.dim))
        self.blocks = nn.ModuleList([AdaLNTransformerBlock(cfg) for _ in range(cfg.depth)])
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


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def config_from_checkpoint(checkpoint: dict[str, Any]) -> TorchTransformerConfig:
    cfg = dict(checkpoint["model_config"])
    allowed = TorchTransformerConfig.__dataclass_fields__.keys()
    return TorchTransformerConfig(**{k: cfg[k] for k in allowed if k in cfg})


def infer_step_offset(run_dir: Path) -> int:
    loss_path = run_dir / "logs" / "loss.csv"
    if not loss_path.exists():
        return 0
    with open(loss_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    return int(max(float(row["step"]) for row in rows if row.get("step")))


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


def loss_fn(model: TinyAdaLNTransformer1D, batch, mode: str, t_min: float):
    x0, eps, t, z_t = batch
    raw = model(z_t, t)
    v_target = eps - x0
    v_pred = pred_to_velocity_torch(raw, z_t, t, mode, t_min)
    return F.mse_loss(v_pred, v_target)


def train_mode(
    mode: str,
    init_state: dict[str, torch.Tensor] | None,
    x0_all: torch.Tensor,
    model_cfg: TorchTransformerConfig,
    train_cfg: TorchTrainConfig,
    device: torch.device,
    seed: int,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
    step_offset: int = 0,
):
    model = TinyAdaLNTransformer1D(model_cfg).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
    elif init_state is not None:
        model.load_state_dict(init_state)
    else:
        raise ValueError("Either init_state or resume_checkpoint must be provided.")
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps, weight_decay=train_cfg.weight_decay)
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        opt.load_state_dict(resume_checkpoint["optimizer"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows = []
    for step in range(1, train_cfg.steps + 1):
        global_step = step_offset + step
        opt.zero_grad(set_to_none=True)
        batch = make_batch(x0_all, train_cfg, generator, device)
        loss = loss_fn(model, batch, mode, train_cfg.t_min)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        clip_scale = min(1.0, train_cfg.grad_clip_norm / (float(grad_norm) + 1e-12))
        opt.step()
        if step == 1 or step % train_cfg.loss_every == 0 or step == train_cfg.steps:
            rows.append({"mode": mode, "step": global_step, "local_step": step, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm), "clip_scale": clip_scale})
        if step == 1 or step % train_cfg.print_every == 0 or step == train_cfg.steps:
            print(
                f"[{mode}] step {global_step:6d} (+{step:5d}/{train_cfg.steps}) "
                f"loss={float(loss.detach().cpu()):.6f} grad_norm={float(grad_norm):.4f} clip={clip_scale:.3f}",
                flush=True,
            )
    return model, opt, rows


@torch.no_grad()
def sample_model(model: TinyAdaLNTransformer1D, mode: str, n_samples: int, sample_steps: int, t_min: float, device: torch.device, seed: int):
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


def run_sampling_analysis(run_dir: Path, models: dict[str, TinyAdaLNTransformer1D], data: dict[str, np.ndarray], train_cfg: TorchTrainConfig, analysis_cfg: TorchAnalysisConfig, device: torch.device):
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
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    resume_run_dir = Path(args.resume_run_dir) if args.resume_run_dir else None
    resume_checkpoints: dict[str, dict[str, Any]] = {}
    if resume_run_dir is not None:
        data_path = resume_run_dir / "training_data_snapshot.npz"
        data = dict(np.load(data_path))
        resume_step_offset = args.resume_step_offset if args.resume_step_offset is not None else infer_step_offset(resume_run_dir)
        for mode in MODES:
            resume_checkpoints[mode] = torch.load(resume_run_dir / "checkpoints" / f"{mode}_final.pt", map_location=device)
        model_cfg = config_from_checkpoint(resume_checkpoints["x"])
    else:
        data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)
        resume_step_offset = 0
        model_cfg = TorchTransformerConfig(
            ambient_dim=args.ambient_dim,
            patch_size=args.patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_width=args.mlp_width,
            time_embed_dim=args.time_embed_dim,
            time_width=args.time_width,
            zero_init_output=True,
            attention_impl=args.attention_impl,
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
    init_model = TinyAdaLNTransformer1D(model_cfg)
    init_state = None if resume_run_dir is not None else {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    n_params = count_params(init_model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    elif resume_run_dir is not None:
        run_name = (
            f"torch_transformer1d_resume_D{model_cfg.ambient_dim}_adamw_p{model_cfg.patch_size}_d{model_cfg.dim}"
            f"_h{model_cfg.heads}_L{model_cfg.depth}_m{model_cfg.mlp_width}_{model_cfg.attention_impl}"
            f"_from{resume_step_offset}_to{resume_step_offset + args.steps}_seed{args.seed}_{ts}"
        )
    else:
        run_name = f"torch_transformer1d_D{args.ambient_dim}_adamw_p{args.patch_size}_d{args.dim}_h{args.heads}_L{args.depth}_m{args.mlp_width}_{args.attention_impl}_steps{args.steps}_seed{args.seed}_{ts}"
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
        "experiment_type": "torch_transformer1d_sampling_training",
        "optimizer": "AdamW",
        "device": str(device),
        "model_family": "TinyAdaLNTransformer1D",
        "model_config": asdict(model_cfg),
        "parameter_count": n_params,
        "patch_count": args.ambient_dim // args.patch_size,
        "train_config": asdict(train_cfg),
        "analysis_config": asdict(analysis_cfg),
        "resume_run_dir": str(resume_run_dir) if resume_run_dir is not None else None,
        "resume_step_offset": resume_step_offset,
        "resume_target_step": resume_step_offset + args.steps,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), "parameter_count": n_params, "patch_count": metadata["patch_count"]}, indent=2), flush=True)

    x0_all = torch.as_tensor(data["x0"], dtype=torch.float32, device=device)
    models: dict[str, TinyAdaLNTransformer1D] = {}
    all_rows: list[dict[str, Any]] = []
    for mode in MODES:
        print(f"\n{'=' * 90}\nTorch Transformer1D training mode={mode}\n{'=' * 90}", flush=True)
        model, opt, rows = train_mode(
            mode,
            init_state,
            x0_all,
            model_cfg,
            train_cfg,
            device,
            seed=args.seed + 1000 + resume_step_offset,
            resume_checkpoint=resume_checkpoints.get(mode),
            step_offset=resume_step_offset,
        )
        models[mode] = model
        all_rows.extend(rows)
        if args.save_checkpoints:
            torch.save(
                {
                    "mode": mode,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(train_cfg),
                    "step": resume_step_offset + train_cfg.steps,
                    "resume_run_dir": str(resume_run_dir) if resume_run_dir is not None else None,
                },
                run_dir / "checkpoints" / f"{mode}_final.pt",
            )
    write_csv(run_dir / "logs" / "loss.csv", all_rows)
    print("\nPost-training sampling analysis", flush=True)
    run_sampling_analysis(run_dir, models, data, train_cfg, analysis_cfg, device)
    print(f"\nTorch Transformer1D run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="PyTorch Tiny Transformer1D + AdaLN-zero toy diffusion sampling experiment")
    p.add_argument("--output-root", default="results/torch_transformer1d_sampling")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ambient-dim", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--mlp-width", type=int, default=512)
    p.add_argument("--attention-impl", choices=("torch", "manual"), default="torch")
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
    p.add_argument("--resume-run-dir", default=None, help="Continue from a previous run directory containing checkpoints and training_data_snapshot.npz.")
    p.add_argument("--resume-step-offset", type=int, default=None, help="Global step offset for resumed logs. Defaults to max step in the previous loss.csv.")
    return p


if __name__ == "__main__":
    run_experiment(build_argparser().parse_args())
