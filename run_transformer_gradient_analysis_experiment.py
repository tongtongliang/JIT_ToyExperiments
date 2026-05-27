from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from clean_jax_exp.data import get_or_create_dataset
from clean_jax_exp.metrics import sign_invariant_angle_deg, spectral_metrics, top_right_singular_vector
from clean_jax_exp.visualize import ANGLE_LABELS, MATRIX_LABELS, MODE_COLORS, MODE_LABELS, MODE_MARKERS, setup_style
from run_transformer1d_torch_experiment import (
    MODES,
    TinyAdaLNTransformer1D,
    TorchTrainConfig,
    TorchTransformerConfig,
    count_params,
    loss_fn,
    make_batch,
    pick_device,
)

MATRIX_KINDS = ("gradient", "momentum", "update", "activation", "residual")
ANGLE_KINDS = (
    "adjacent_gradient",
    "adjacent_momentum",
    "adjacent_update",
    "adjacent_activation",
    "adjacent_residual",
    "gradient_vs_previous_momentum",
)


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_last_dim(x: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(-1, arr.shape[-1])


def grad_reconstruction_error(grad_w: np.ndarray, activation: np.ndarray, residual: np.ndarray):
    estimate = residual.T @ activation
    num = np.linalg.norm(grad_w - estimate)
    den = np.linalg.norm(grad_w) + 1e-20
    return float(num / den), float(num)


class LinearIORecorder:
    def __init__(self, modules: dict[str, nn.Linear]):
        self.modules = modules
        self.enabled = False
        self.activations: dict[str, torch.Tensor] = {}
        self.residuals: dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._forward_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._backward_hook(name)))

    def _forward_hook(self, name: str):
        def hook(module, inputs, output):
            if self.enabled and inputs and isinstance(inputs[0], torch.Tensor):
                self.activations[name] = inputs[0].detach()
        return hook

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if self.enabled and grad_output and isinstance(grad_output[0], torch.Tensor):
                self.residuals[name] = grad_output[0].detach()
        return hook

    def clear(self):
        self.activations.clear()
        self.residuals.clear()

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def tracked_linear_modules(model: TinyAdaLNTransformer1D, *, include_time: bool) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not include_time and (name.startswith("time_mlp") or name.endswith(".ada") or name == "final_ada"):
            continue
        modules[name] = module
    return modules


def friendly_layer_label(name: str) -> str:
    if name == "patch_embed":
        return "patch embed"
    if name == "output_proj":
        return "output proj"
    if name == "final_ada":
        return "final AdaLN"
    if name.startswith("time_mlp"):
        return name.replace("_", " ")
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blocks":
        idx = int(parts[1]) + 1
        leaf = parts[2]
        if leaf == "ada":
            return f"block {idx} AdaLN"
        if leaf == "qkv":
            return f"block {idx} QKV"
        if leaf == "attn_out":
            return f"block {idx} attention out"
        if leaf == "mlp0":
            return f"block {idx} MLP up"
        if leaf == "mlp1":
            return f"block {idx} MLP down"
    return name


def run_training_for_mode(
    mode: str,
    init_state: dict[str, torch.Tensor],
    x0_all: torch.Tensor,
    model_cfg: TorchTransformerConfig,
    train_cfg: TorchTrainConfig,
    device: torch.device,
    run_dir: Path,
    save_checkpoints: bool,
    seed: int,
    include_time_matrices: bool,
):
    model = TinyAdaLNTransformer1D(model_cfg).to(device)
    model.load_state_dict(init_state)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    modules = tracked_linear_modules(model, include_time=include_time_matrices)
    recorder = LinearIORecorder(modules)
    previous_vectors: dict[tuple[str, str], np.ndarray | None] = {}
    loss_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []

    name_to_param = dict(model.named_parameters())
    tracked_weight_names = {name: f"{name}.weight" for name in modules.keys()}

    for step in range(1, train_cfg.steps + 1):
        record = step == 1 or step % train_cfg.metric_every == 0
        opt.zero_grad(set_to_none=True)
        batch = make_batch(x0_all, train_cfg, generator, device)
        recorder.clear()
        recorder.enabled = record
        if record:
            before = {pname: name_to_param[pname].detach().cpu().clone().numpy() for pname in tracked_weight_names.values()}
        loss = loss_fn(model, batch, mode, train_cfg.t_min)
        loss.backward()
        recorder.enabled = False

        if record:
            raw_grads = {}
            for layer, pname in tracked_weight_names.items():
                grad = name_to_param[pname].grad
                raw_grads[layer] = None if grad is None else grad.detach().cpu().clone().numpy()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        clip_scale = min(1.0, train_cfg.grad_clip_norm / (float(grad_norm) + 1e-12))
        opt.step()

        if record:
            for layer, pname in tracked_weight_names.items():
                param = name_to_param[pname]
                grad_w = raw_grads[layer]
                if grad_w is None:
                    continue
                update_w = param.detach().cpu().numpy() - before[pname]
                state = opt.state.get(param, {})
                exp_avg = state.get("exp_avg")
                momentum_w = np.zeros_like(grad_w) if exp_avg is None else exp_avg.detach().cpu().numpy()
                if layer not in recorder.activations or layer not in recorder.residuals:
                    continue
                activation = flatten_last_dim(recorder.activations[layer])
                residual = flatten_last_dim(recorder.residuals[layer])
                matrices = {
                    "gradient": (grad_w, False),
                    "momentum": (momentum_w, False),
                    "update": (update_w, False),
                    "activation": (activation, True),
                    "residual": (residual, True),
                }
                rel, abs_err = grad_reconstruction_error(grad_w, activation, residual)
                sanity_rows.append({
                    "mode": mode,
                    "step": step,
                    "layer": layer,
                    "relative_error": rel,
                    "absolute_error": abs_err,
                })

                current_vectors: dict[str, np.ndarray | None] = {}
                for kind, (mat, center) in matrices.items():
                    sm = spectral_metrics(mat, center=center, rank_threshold=0.90)
                    metric_rows.append({
                        "mode": mode,
                        "step": step,
                        "layer": layer,
                        "layer_label": friendly_layer_label(layer),
                        "matrix_kind": kind,
                        "stable_rank": sm["stable_rank"],
                        "rank90": sm["rank_k"],
                        "op_norm": sm["op_norm"],
                        "fro_norm": sm["fro_norm"],
                        "top1_energy": sm["top1_energy"],
                    })
                    vec = top_right_singular_vector(mat, center=center)
                    current_vectors[kind] = vec
                    prev_key = (layer, kind)
                    if prev_key in previous_vectors:
                        angle_rows.append({
                            "mode": mode,
                            "step": step,
                            "layer": layer,
                            "angle_kind": f"adjacent_{kind}",
                            "angle_deg": sign_invariant_angle_deg(previous_vectors[prev_key], vec),
                        })
                angle_rows.append({
                    "mode": mode,
                    "step": step,
                    "layer": layer,
                    "angle_kind": "gradient_vs_previous_momentum",
                    "angle_deg": sign_invariant_angle_deg(previous_vectors.get((layer, "momentum")), current_vectors["gradient"]),
                })
                for kind, vec in current_vectors.items():
                    previous_vectors[(layer, kind)] = vec

        if step == 1 or step % train_cfg.loss_every == 0 or step == train_cfg.steps:
            loss_rows.append({
                "mode": mode,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm),
                "clip_scale": clip_scale,
            })
        if step == 1 or step % train_cfg.print_every == 0 or step == train_cfg.steps:
            print(f"[{mode}] step {step:6d}/{train_cfg.steps} loss={float(loss.detach().cpu()):.6f} grad_norm={float(grad_norm):.4f} clip={clip_scale:.3f}", flush=True)

    recorder.remove()
    ckpt_path = None
    if save_checkpoints:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{mode}_final.pt"
        torch.save({
            "mode": mode,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "step": train_cfg.steps,
        }, ckpt_path)

    return {
        "loss_rows": loss_rows,
        "metric_rows": metric_rows,
        "angle_rows": angle_rows,
        "sanity_rows": sanity_rows,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "tracked_layers": list(modules.keys()),
    }


def _legend_inside(ax, *, loc: str = "best"):
    ax.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="0.85", loc=loc)


def save_figure(fig, run_dir: Path, name: str, *, save_pdf: bool = False):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def plot_matrix_metric(run_dir: Path, df, *, matrix_kind: str, layer: str, metric: str, save_pdf: bool):
    setup_style()
    sub = df[(df["matrix_kind"] == matrix_kind) & (df["layer"] == layer)]
    if sub.empty:
        return None
    label = str(sub["layer_label"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("step")
        ax.plot(m["step"], m[metric], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(m) // 8), linewidth=1.6, label=MODE_LABELS[mode])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Stable rank" if metric == "stable_rank" else "90% PCA rank")
    ax.set_title(f"{MATRIX_LABELS.get(matrix_kind, matrix_kind)}: {label}", fontweight="bold")
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    return save_figure(fig, run_dir, f"matrix_{matrix_kind}_{safe_name(layer)}_{metric}", save_pdf=save_pdf)


def plot_angle_metric(run_dir: Path, df, *, angle_kind: str, layer: str, save_pdf: bool):
    setup_style()
    sub = df[(df["angle_kind"] == angle_kind) & (df["layer"] == layer)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in MODES:
        m = sub[sub["mode"] == mode].sort_values("step")
        ax.plot(m["step"], m["angle_deg"], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], markevery=max(1, len(m) // 8), linewidth=1.6, label=MODE_LABELS[mode])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ANGLE_LABELS.get(angle_kind, "Principal angle (deg)"))
    ax.set_title(f"{ANGLE_LABELS.get(angle_kind, angle_kind)}: {friendly_layer_label(layer)}", fontweight="bold")
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.25)
    _legend_inside(ax, loc="best")
    return save_figure(fig, run_dir, f"angle_{angle_kind}_{safe_name(layer)}", save_pdf=save_pdf)


def generate_figures(run_dir: Path, *, selected_layers: list[str] | None = None, all_layers: bool = False, save_pdf: bool = False):
    import pandas as pd

    metric_df = pd.read_csv(run_dir / "logs" / "matrix_metrics.csv")
    angle_df = pd.read_csv(run_dir / "logs" / "angle_metrics.csv")
    layers = list(dict.fromkeys(metric_df["layer"].tolist()))
    if selected_layers is None:
        selected_layers = [
            "patch_embed",
            "blocks.4.qkv",
            "blocks.4.attn_out",
            "blocks.4.mlp0",
            "blocks.4.mlp1",
            "output_proj",
        ]
    layers_to_plot = layers if all_layers else [layer for layer in selected_layers if layer in layers]
    paths = []
    for layer in layers_to_plot:
        for kind in MATRIX_KINDS:
            for metric in ("stable_rank", "rank90"):
                p = plot_matrix_metric(run_dir, metric_df, matrix_kind=kind, layer=layer, metric=metric, save_pdf=save_pdf)
                if p is not None:
                    paths.append(p)
        for angle_kind in ("adjacent_gradient", "gradient_vs_previous_momentum", "adjacent_activation", "adjacent_residual"):
            p = plot_angle_metric(run_dir, angle_df, angle_kind=angle_kind, layer=layer, save_pdf=save_pdf)
            if p is not None:
                paths.append(p)
    return paths


def run_experiment(args: argparse.Namespace):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    data_path, data = get_or_create_dataset(output_root, args.ambient_dim, args.n_samples, args.data_noise, args.seed)
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
    # Attach metric frequency without changing the shared dataclass used by the sampling runner.
    object.__setattr__(train_cfg, "metric_every", args.metric_every)

    torch.manual_seed(args.seed + 123)
    init_model = TinyAdaLNTransformer1D(model_cfg)
    init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    n_params = count_params(init_model)
    tracked = list(tracked_linear_modules(init_model, include_time=args.include_time_matrices).keys())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"torch_transformer_gradient_D{args.ambient_dim}_adamw_p{args.patch_size}_d{args.dim}_h{args.heads}_L{args.depth}_m{args.mlp_width}_{args.attention_impl}_s{args.steps}_seed{args.seed}_{ts}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "modes": list(MODES),
        "experiment_type": "torch_transformer1d_gradient_analysis",
        "optimizer": "AdamW",
        "device": str(device),
        "model_family": "TinyAdaLNTransformer1D",
        "model_config": asdict(model_cfg),
        "parameter_count": n_params,
        "patch_count": args.ambient_dim // args.patch_size,
        "train_config": {**asdict(train_cfg), "metric_every": args.metric_every},
        "tracked_layers": tracked,
        "matrix_kinds": list(MATRIX_KINDS),
        "notes": "Early gradient-rank/principal-angle analysis for the patch Transformer. Manual attention exposes qkv and attention-output matrices as ordinary Linear modules for activation/residual sanity checks.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "training_data_snapshot.npz", **data)
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), "parameter_count": n_params, "tracked_matrices": len(tracked)}, indent=2), flush=True)

    x0_all = torch.as_tensor(data["x0"], dtype=torch.float32, device=device)
    all_loss_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_angle_rows: list[dict[str, Any]] = []
    all_sanity_rows: list[dict[str, Any]] = []

    for mode in MODES:
        print(f"\n{'=' * 90}\nTorch Transformer gradient analysis mode={mode}\n{'=' * 90}", flush=True)
        result = run_training_for_mode(
            mode=mode,
            init_state=init_state,
            x0_all=x0_all,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=device,
            run_dir=run_dir,
            save_checkpoints=args.save_checkpoints,
            seed=args.seed + 1000,
            include_time_matrices=args.include_time_matrices,
        )
        all_loss_rows.extend(result["loss_rows"])
        all_metric_rows.extend(result["metric_rows"])
        all_angle_rows.extend(result["angle_rows"])
        all_sanity_rows.extend(result["sanity_rows"])
        if device.type == "mps":
            torch.mps.empty_cache()

    write_csv(run_dir / "logs" / "loss.csv", all_loss_rows)
    write_csv(run_dir / "logs" / "matrix_metrics.csv", all_metric_rows)
    write_csv(run_dir / "logs" / "angle_metrics.csv", all_angle_rows)
    write_csv(run_dir / "logs" / "sanity_metrics.csv", all_sanity_rows)

    if args.make_figures:
        paths = generate_figures(run_dir, all_layers=args.plot_all_layers, save_pdf=args.save_pdf)
        print("\n".join(str(p) for p in paths), flush=True)

    print(f"\nTorch Transformer gradient-analysis run complete: {run_dir}", flush=True)
    return run_dir


def build_argparser():
    p = argparse.ArgumentParser(description="PyTorch Tiny Transformer1D early gradient-rank and principal-angle experiment")
    p.add_argument("--output-root", default="results/torch_transformer1d_gradient")
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
    p.add_argument("--attention-impl", choices=("manual",), default="manual")
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument("--time-width", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-every", type=int, default=10)
    p.add_argument("--metric-every", type=int, default=20)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--include-time-matrices", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--make-figures", action="store_true")
    p.add_argument("--plot-all-layers", action="store_true")
    p.add_argument("--save-pdf", action="store_true")
    return p


if __name__ == "__main__":
    run_experiment(build_argparser().parse_args())
