# Script Entry Points

The repository keeps notebooks for interactive inspection, but experiments should be launched through scripts so that logs, metadata, checkpoints, and figures are reproducible.

## `scripts/fcn/`

FCN/JAX experiments. These are the primary controlled toy experiments.

| script | purpose |
| --- | --- |
| `run_clean_jax_experiment.py` | 100k-step representation/stability/sampling experiment for the baseline FCN |
| `run_clean_jax_longskip_experiment.py` | same long experiment with learned scalar long skip |
| `run_gradient_analysis_experiment.py` | 2000-step early gradient dynamics for the baseline FCN |
| `run_gradient_longskip_experiment.py` | 2000-step early gradient dynamics for the long-skip FCN |

## `scripts/architectures/`

PyTorch architecture comparisons and non-FCN diagnostics.

| script | purpose |
| --- | --- |
| `run_transformer1d_torch_experiment.py` | patch Transformer sampling experiment |
| `run_transformer_gradient_analysis_experiment.py` | patch Transformer early gradient dynamics with explicit QKV matrices |
| `run_mixer1d_torch_experiment.py` | patch MLP-Mixer sampling experiment |
| `run_unet1d_torch_experiment.py` | Tiny 1D U-Net sampling experiment |

## `scripts/analysis/`

Posthoc analysis over completed runs.

| script | purpose |
| --- | --- |
| `analyze_longskip_coefficients.py` | visualize learned `c_skip(t)` and `c_out(t)` plus branch usage |
| `analyze_transformer_hidden_representations.py` | patch-level hidden representation dimensionality for Transformer checkpoints |
| `analyze_gradient_effective_rank90.py` | compare FCN, long-skip FCN, and Transformer gradient `rank90` logs |
