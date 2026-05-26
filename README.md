# Diffusion Model Inductive Bias Toy Experiments

This project studies why denoising models succeed or fail under different output parameterizations. The clean workflow is organized around three deliberately separate experiments:

1. Hidden representation dimensionality, stability, and sampling after long training.
2. Early gradient rank and principal-angle dynamics.
3. Sampling-only architecture comparison with a Tiny 1D U-Net + AdaGN.
4. Sampling-only architecture comparison with a small AdaLN-zero Transformer over 1D patches.

The separation is intentional. The representation experiment needs sufficiently converged FCN models. The gradient experiment needs dense early-training diagnostics and should not be conflated with final sample quality. The U-Net and Transformer experiments are sampling-only for now, because hidden-representation and gradient diagnostics for non-FCN architectures require a separate hook design.

The main hypothesis is that successful denoising corresponds to learning corruption-stable, data-dependent low-dimensional structure, while failing parameterizations write or chase high-dimensional noise directions.

The local gradient mechanism is

```text
grad_W L = residual.T @ activation
```

The gradient-analysis script checks this factorization numerically when it records activation and residual matrices.

## Current Project Layout

```text
.
├── clean_jax_exp/
│   ├── data.py                         # Fixed Swiss-roll data generation and projection
│   ├── models.py                       # 5-block residual FCN with pre-norm AdaLN-zero
│   ├── metrics.py                      # Stable rank, PCA ranks, principal angles, NSV
│   ├── train_representation.py         # Long training + representation/stability/sampling analysis
│   ├── train_gradient.py               # Early gradient-rank and principal-angle analysis
│   ├── posthoc_analysis.py             # Sample quality and hidden-spectrum posthoc analysis
│   └── visualize.py                    # Paper-friendly independent plot functions
├── run_clean_jax_experiment.py          # Representation experiment CLI
├── run_gradient_analysis_experiment.py  # Gradient-analysis experiment CLI
├── run_unet1d_torch_experiment.py       # Tiny 1D U-Net + AdaGN sampling-only CLI
├── run_transformer1d_torch_experiment.py # AdaLN-zero Transformer sampling-only CLI
├── 01_representation_dimension_and_stability.ipynb
├── 02_gradient_rank_and_angle.ipynb
├── results/clean_jax_representation/
├── results/clean_jax_gradient/
├── results/torch_unet1d_sampling/
├── results/torch_transformer1d_sampling/
└── old_notebooks_and_data/
```

## Main Notebooks

`01_representation_dimension_and_stability.ipynb`

Origin experiment. It trains or loads the long-training representation run, then visualizes hidden representation dimension, representation stability, sampling quality, and sampled point-cloud dimension. It measures both the LayerNorm representation (`norm`) and the post-AdaLN MLP fan-in (`fanin`). Visualizations are independent single plots with local legends, no embedded titles, log-scale loss curves, aligned sample coordinates, and numeric labels on bar charts.

`02_gradient_rank_and_angle.ipynb`

Mechanistic follow-up. It reads the early-gradient run and visualizes stable rank, 90% PCA rank, and principal-angle drift for gradients, AdamW first moment, actual updates, activations, and residuals. It also displays the numerical sanity check for `grad_W = residual.T @ activation`.

## Representation Runner

Long-training run for Notebook 1:

```bash
/Users/tongtongliang/miniforge3/bin/python3.12 run_clean_jax_experiment.py \
  --output-root results/clean_jax_representation \
  --ambient-dim 512 \
  --n-samples 8192 \
  --width 256 \
  --depth 5 \
  --time-embed-dim 256 \
  --steps 100000 \
  --batch-size 256 \
  --loss-every 100 \
  --print-every 1000 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --save-checkpoints
```

This script does not record gradient-rank matrices. It saves loss, final checkpoints, representation metrics, stability metrics, generated samples, sample point-cloud metrics, and figures.

## Gradient Runner

Early-gradient run for Notebook 2:

```bash
/Users/tongtongliang/miniforge3/bin/python3.12 run_gradient_analysis_experiment.py \
  --output-root results/clean_jax_gradient \
  --ambient-dim 512 \
  --n-samples 8192 \
  --width 256 \
  --depth 5 \
  --time-embed-dim 256 \
  --steps 2000 \
  --batch-size 256 \
  --metric-every 20 \
  --print-every 50 \
  --lr 1e-4 \
  --grad-clip-norm 1.0
```

This script does not run representation/stability/sampling posthoc analysis. It saves loss, matrix-rank logs, angle logs, and gradient-factorization sanity checks.

## Tiny U-Net Runner

Sampling-only architecture comparison:

```bash
python run_unet1d_torch_experiment.py \
  --output-root results/torch_unet1d_sampling \
  --ambient-dim 512 \
  --n-samples 8192 \
  --patch-size 4 \
  --stride 2 \
  --base-channels 56 \
  --kernel-size 3 \
  --blocks-per-level 2 \
  --time-embed-dim 256 \
  --time-width 256 \
  --groups 8 \
  --steps 100000 \
  --batch-size 256 \
  --loss-every 100 \
  --print-every 1000 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --sample-n 2048 \
  --sample-steps 100 \
  --device auto \
  --save-checkpoints
```

This runner uses PyTorch and automatically selects `cuda`, then `mps`, then `cpu`. On the local Mac, MPS only works outside Codex's filesystem sandbox; the sandbox can make PyTorch mis-detect the macOS version. On a GPU server, `--device auto` should choose CUDA.

The default U-Net is a Tiny 1D U-Net over overlapping local windows:

```text
ambient dimension = 512
local window = 4
stride = 2
patch count = 255
base channels = 56
kernel size = 3
parameter count ~= 1.98M
```

This is intentionally close to the FCN parameter count of about `2.04M`. The U-Net runner saves loss curves, checkpoints, generated samples, sample point-cloud dimensions, and sample-quality metrics. It does not collect hidden-representation or gradient-rank diagnostics.

To generate figures after a U-Net run:

```python
from pathlib import Path
from clean_jax_exp.visualize import latest_run, generate_sampling_figures

run_dir = latest_run(Path("results/torch_unet1d_sampling/runs"))
generate_sampling_figures(run_dir, save_pdf=False, show=False)
```

## Implementation Details

- Training uses JAX/JIT on CPU from the `base` environment.
- The `ml` environment currently does not have JAX installed.
- The model is a 5-block residual FCN with pre-norm AdaLN-zero.
- The final output projection is zero-initialized.
- All three modes start from the same initialization.
- All three modes use the same dynamic batch, `t`, and `epsilon` random sequence within each experiment.
- Training uses dynamic diffusion-style sampling: every step resamples data indices, `t`, and Gaussian noise.
- AdamW uses global gradient clipping before the first-moment update.
- PDF export is off by default. Set `SAVE_PDF=True` in notebooks only after selecting final plot styling.
- The sampling-only U-Net runner is PyTorch-based and should be run on CUDA or MPS for full 100k-step experiments.

## Metrics

`stable_rank`

```text
||A||_F^2 / ||A||_op^2
```

Measures whether matrix energy is concentrated in a few singular directions.

`rank95`

The number of PCA components needed to explain 95% of centered hidden-representation or sample point-cloud energy.

`rank90`

The number of PCA components needed to explain 90% of matrix energy in gradient-dynamics logs.

`principal_angle`

Sign-invariant angle between dominant right singular vectors of adjacent matrices. Large values mean the leading direction is rotating over training.

`NSV`

Normalized noise variance of hidden representations under resampled corruption noise for the same clean sample. Lower values mean more corruption-stable representations.

## Archive

Older notebooks, executed notebooks, historical PyTorch scripts, previous figures, and checkpoint files are preserved in `old_notebooks_and_data/`. They are not part of the clean main workflow, but remain available for comparison.

The abandoned JAX U-Net prototype is archived under `old_notebooks_and_data/scripts/abandoned_jax_unet1d_20260525/`. It is kept for provenance, but the active U-Net path is `run_unet1d_torch_experiment.py`.

## Tiny Transformer Runner

Sampling-only architecture comparison with a simpler computation graph than the U-Net:

```bash
python run_transformer1d_torch_experiment.py \
  --output-root results/torch_transformer1d_sampling \
  --ambient-dim 512 \
  --n-samples 8192 \
  --patch-size 8 \
  --dim 128 \
  --depth 5 \
  --heads 1 \
  --mlp-width 512 \
  --attention-impl torch \
  --time-embed-dim 256 \
  --time-width 256 \
  --steps 100000 \
  --batch-size 256 \
  --loss-every 100 \
  --print-every 1000 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --sample-n 2048 \
  --sample-steps 100 \
  --device auto \
  --save-checkpoints
```

The default Transformer uses non-overlapping 1D patches:

```text
ambient dimension = 512
patch size = 8
patch count = 64
Transformer width = 128
attention heads = 1
depth = 5
MLP width = 512
attention implementation = torch.nn.MultiheadAttention
parameter count ~= 2.18M
```

Each block uses affine-free LayerNorm, AdaLN-zero shift/scale/gates from the time embedding, self-attention, and a token MLP. The default attention path uses `torch.nn.MultiheadAttention(batch_first=True)`, with `--attention-impl manual` available as a fallback. The final patch decoder is zero-initialized. Like the U-Net runner, this script saves loss curves, checkpoints, generated samples, sample point-cloud dimensions, and sample-quality metrics. It does not collect hidden-representation or gradient-rank diagnostics.

Local MPS benchmark on the MacBook:

```text
100 training steps per prediction mode, batch size 256, D=512:
wall time ~= 58 seconds including a small sampling pass
projected 100k-step run over x/v/eps ~= 16-17 hours
```

Posthoc patch hidden-representation analysis for a completed Transformer run:

```bash
python analyze_transformer_hidden_representations.py \
  --run-dir results/torch_transformer1d_sampling/runs/<run-name> \
  --device cpu \
  --n-eval 256 \
  --t-values 0.1,0.3,0.5,0.7,0.9
```

This reads `x/v/eps` checkpoints, reuses the saved training data snapshot, and records patch point-cloud metrics for:

```text
patch_embed
attention pre-norm stream
attention norm output
attention AdaLN fan-in
MLP pre-norm stream
MLP norm output
MLP AdaLN fan-in
final pre-norm stream
final norm output
final AdaLN fan-in
```

Rows are flattened over `(sample, patch)` and measured as width-dimensional point clouds. The output is saved to `analysis/transformer_patch_representation_metrics.csv`, with grouped bar-chart figures under `figures/transformer_hidden_bars/`. The default plots combine attention and MLP into 10 Transformer sublayers (`B0 Attn`, `B0 MLP`, ..., `B4 MLP`) so the layout matches the FCN representation notebook.

## Tiny MLP-Mixer Runner

Sampling-only architecture comparison that keeps the Transformer patchification, embedding width, AdaLN-zero conditioning, and final patch decoder, but replaces each attention block with an MLP-Mixer block:

```bash
python run_mixer1d_torch_experiment.py \
  --output-root results/torch_mixer1d_sampling \
  --ambient-dim 512 \
  --n-samples 8192 \
  --patch-size 8 \
  --dim 128 \
  --depth 5 \
  --token-mlp-width 128 \
  --channel-mlp-width 512 \
  --time-embed-dim 256 \
  --time-width 256 \
  --steps 100000 \
  --batch-size 256 \
  --loss-every 100 \
  --print-every 1000 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --sample-n 2048 \
  --sample-steps 100 \
  --device auto \
  --save-checkpoints
```

The default Mixer uses:

```text
ambient dimension = 512
patch size = 8
patch count = 64
Mixer width = 128
depth = 5
token-mixing MLP hidden width = 128
channel-mixing MLP hidden width = 512
parameter count ~= 1.94M
parameter count excluding AdaLN/time ~= 0.75M
```

Parameter count comparison for D=512:

| model | total params | excluding AdaLN/time |
|---|---:|---:|
| FCN width=256 depth=5 | 2.04M | 0.92M |
| Transformer d=128 depth=5 | 2.18M | 1.00M |
| MLP-Mixer d=128 depth=5 | 1.94M | 0.75M |

Local MPS benchmark on the MacBook:

```text
100 training steps per prediction mode, batch size 256, D=512:
wall time ~= 44 seconds including a small sampling pass
projected 10k-step run over x/v/eps ~= 1.2-1.5 hours
```
