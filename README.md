# JIT Toy Experiments: Diffusion-Model Inductive Bias

This repository contains toy diffusion/flow-matching experiments for studying why different prediction parameterizations behave differently in high-dimensional ambient space. The central empirical question is:

```text
Why does predicting clean data often recover the low-dimensional data manifold,
while predicting velocity or noise can fail or learn very different dynamics?
```

The working hypothesis is not simply that noise is harder to express. The more precise hypothesis is that the training dynamics have an implicit bias: stable low-dimensional signal directions are accumulated by the optimizer, while unstable or high-dimensional residual directions are either written into many parameter directions or compressed by architectural structure.

## Current Layout

```text
.
├── README.md
├── requirements.txt
├── 01_representation_dimension_and_stability.ipynb
├── 02_gradient_rank_and_angle.ipynb
├── clean_jax_exp/
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── models_longskip.py
│   ├── models_ufcn.py
│   ├── train_representation.py
│   ├── train_representation_longskip.py
│   ├── train_representation_ufcn.py
│   ├── train_gradient.py
│   ├── train_gradient_longskip.py
│   ├── posthoc_analysis.py
│   └── visualize.py
├── scripts/
│   ├── fcn/
│   │   ├── run_clean_jax_experiment.py
│   │   ├── run_clean_jax_longskip_experiment.py
│   │   ├── run_clean_jax_ufcn_experiment.py
│   │   ├── run_gradient_analysis_experiment.py
│   │   └── run_gradient_longskip_experiment.py
│   ├── architectures/
│   │   ├── run_transformer1d_torch_experiment.py
│   │   ├── run_transformer_gradient_analysis_experiment.py
│   │   ├── run_mixer1d_torch_experiment.py
│   │   └── run_unet1d_torch_experiment.py
│   └── analysis/
│       ├── analyze_longskip_coefficients.py
│       ├── analyze_transformer_hidden_representations.py
│       └── analyze_gradient_effective_rank90.py
├── docs/
│   └── PROJECT_SUMMARY.md
├── results/                  # local, git-ignored experiment outputs
└── old_notebooks_and_data/    # local, git-ignored historical archive
```

## Main Notebooks

`01_representation_dimension_and_stability.ipynb`

Origin experiment. It focuses on long-training FCN models and analyzes internal representation dimensionality, representation stability under resampled noise, sampling quality, sample point-cloud dimension, and learned point-cloud subspace alignment. It measures both the residual stream and the AdaLN fan-in representation.

`02_gradient_rank_and_angle.ipynb`

Mechanistic follow-up. It focuses on early training and visualizes stable rank, rank90 effective rank, principal-angle drift, and the matrix factorization sanity check for gradients. It reads saved logs rather than recomputing figures from notebooks whenever possible.

## Core Package

`clean_jax_exp/` contains the reusable FCN/JAX code:

| file | role |
| --- | --- |
| `data.py` | Swiss-roll data generation, high-dimensional projection, fixed data snapshots |
| `models.py` | 5-block residual FCN with pre-norm AdaLN-zero and zero-initialized output |
| `models_longskip.py` | FCN plus learned EDM-like scalar long skip |
| `models_ufcn.py` | FCN plus U-ViT-style hidden long skips, block1->block5 and block2->block4 at depth 5 |
| `metrics.py` | stable rank, rank90/rank95, singular vectors, principal angles |
| `train_representation.py` | long FCN training plus representation/sampling analysis |
| `train_representation_longskip.py` | long-skip FCN long training and posthoc analysis |
| `train_representation_ufcn.py` | U-FCN long training and posthoc analysis |
| `train_gradient.py` | early FCN gradient dynamics logging |
| `train_gradient_longskip.py` | early long-skip FCN gradient dynamics logging |
| `posthoc_analysis.py` | sample quality, ambient Chamfer, sample subspace analysis, spectrum analysis |
| `visualize.py` | paper-oriented plotting utilities |

## Script Groups

### FCN runners

These are the most important baseline scripts.

```bash
python scripts/fcn/run_clean_jax_experiment.py --help
python scripts/fcn/run_clean_jax_longskip_experiment.py --help
python scripts/fcn/run_clean_jax_ufcn_experiment.py --help
python scripts/fcn/run_gradient_analysis_experiment.py --help
python scripts/fcn/run_gradient_longskip_experiment.py --help
```

Typical long representation run:

```bash
/Users/tongtongliang/miniforge3/bin/python3.12 scripts/fcn/run_clean_jax_experiment.py \
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

Typical early-gradient run:

```bash
/Users/tongtongliang/miniforge3/bin/python3.12 scripts/fcn/run_gradient_analysis_experiment.py \
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

Typical U-FCN hidden-skip representation run:

```bash
/Users/tongtongliang/miniforge3/bin/python3.12 scripts/fcn/run_clean_jax_ufcn_experiment.py \
  --output-root results/clean_jax_representation_ufcn \
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

### Architecture comparison runners

These are sampling or architecture-specific diagnostics.

```bash
python scripts/architectures/run_transformer1d_torch_experiment.py --help
python scripts/architectures/run_transformer_gradient_analysis_experiment.py --help
python scripts/architectures/run_mixer1d_torch_experiment.py --help
python scripts/architectures/run_unet1d_torch_experiment.py --help
```

Transformer sampling run:

```bash
python scripts/architectures/run_transformer1d_torch_experiment.py \
  --output-root results/torch_transformer1d_sampling \
  --ambient-dim 512 \
  --n-samples 8192 \
  --patch-size 8 \
  --dim 128 \
  --depth 5 \
  --heads 1 \
  --mlp-width 512 \
  --attention-impl torch \
  --steps 100000 \
  --batch-size 256 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --device auto \
  --save-checkpoints
```

Transformer gradient run:

```bash
python scripts/architectures/run_transformer_gradient_analysis_experiment.py \
  --output-root results/torch_transformer1d_gradient \
  --ambient-dim 512 \
  --n-samples 8192 \
  --patch-size 8 \
  --dim 128 \
  --depth 5 \
  --heads 1 \
  --mlp-width 512 \
  --attention-impl manual \
  --steps 2000 \
  --batch-size 256 \
  --metric-every 20 \
  --print-every 50 \
  --lr 1e-4 \
  --grad-clip-norm 1.0 \
  --device mps \
  --make-figures
```

The gradient runner intentionally uses `attention_impl=manual`, because it exposes QKV and attention-output projections as ordinary matrices. It records gradient, AdamW first moment, actual update, activation, residual, rank metrics, angle metrics, and the numerical check for

```text
grad_W = residual.T @ activation
```

### Posthoc analysis scripts

```bash
python scripts/analysis/analyze_longskip_coefficients.py --help
python scripts/analysis/analyze_transformer_hidden_representations.py --help
python scripts/analysis/analyze_gradient_effective_rank90.py
```

These scripts read completed runs and write additional CSVs/figures under each run's `analysis/` and `figures/` folders.

## Output Contract

Every non-smoke run should write this structure:

```text
results/<experiment_family>/runs/<run_name>/
├── metadata.json
├── training_data_snapshot.npz
├── logs/
│   ├── loss.csv
│   ├── matrix_metrics.csv          # gradient runs only
│   ├── angle_metrics.csv           # gradient runs only
│   └── sanity_metrics.csv          # gradient runs only
├── checkpoints/                    # only when requested
├── analysis/
│   ├── sample_metrics.csv
│   ├── sample_quality_metrics.csv
│   ├── sample_subspace_metrics.csv
│   └── representation*.csv         # representation runs only
└── figures/
```

`results/` is intentionally git-ignored. The code and documentation are versioned; large logs, figures, and checkpoints stay local.

## Metrics

| metric | meaning |
| --- | --- |
| `stable_rank` | `||A||_F^2 / ||A||_op^2`; continuous measure of energy concentration |
| `rank90` | number of singular directions explaining 90% of matrix energy; used for gradient effective rank |
| `rank95` | number of PCA directions explaining 95% of centered point-cloud energy; used for hidden/sample point clouds |
| `principal_angle` | sign-invariant angle between dominant singular vectors across adjacent steps |
| `NSV` | normalized representation variance under resampled corruption noise |
| `ambient_chamfer` | Chamfer distance in the original D-dimensional ambient space |
| `subspace_angle` | angle between sample PCA subspace and the true 2D data plane |

## Current Empirical Picture

The short version is:

1. FCN clean prediction is strongly favored by early gradient dynamics.
2. FCN velocity/noise prediction can write high-dimensional directions into the last layer and optimizer momentum.
3. A learned long skip dramatically helps velocity prediction and partially helps noise prediction by letting the model represent large linear-in-`z_t` terms directly.
4. U-ViT-style hidden long skips inside the FCN do not, by themselves, fix the velocity/noise failure; they largely reproduce the plain FCN sample geometry.
5. Patch architectures change the story: Transformer gradient diagnostics do not reproduce the FCN final-layer rank explosion, while Mixer 20k shows that token/channel mixing can route `x`, `v`, and `eps` into near-correct low-dimensional sample geometry.
6. Stable rank alone is not enough; rank90, angles, activation/residual factorization, sample Chamfer, and sample subspace alignment each reveal different failure modes.

For the detailed experiment log and interpretation, see [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md).

For collaborator-facing oral explanations and figure/video caption language, see
[docs/EXPERIMENT_TALK_TRACK_AND_CAPTIONS.md](docs/EXPERIMENT_TALK_TRACK_AND_CAPTIONS.md).
