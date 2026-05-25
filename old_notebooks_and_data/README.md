# Archived Notebooks and Data

This directory stores files that are useful for provenance but no longer belong in the clean main workflow.

## Contents

`notebooks/`

Older exploratory notebooks, including the original flow-matching notebooks and the earlier unified notebook before the Python runner was split out.

`executed_notebooks/`

Executed notebook snapshots. These are useful as frozen records but are intentionally kept out of the main project root.

`checkpoints/`

Trained checkpoint files used by archived experiments and by the current representation-stability notebook. The active notebook points here through `checkpoint_dir='old_notebooks_and_data/checkpoints'`.

`results/`

Historical experiment logs, plots, CSVs, and intermediate outputs from earlier runs.

`scripts/`

Older training worker scripts used to generate archived checkpoints, plus abandoned prototypes kept for provenance. Current experiments should prefer the root CLIs and `clean_jax_exp/`.

`reports/`, `figures/`, `paper_panels_pdf/`, `outputs_patch_learning/`, `outputs_patch_generation/`

Historical analysis artifacts and generated figures.

Nothing here was deleted; it was moved to reduce top-level clutter.

## 2026-05-25 clean JAX rewrite archive note

The previous root notebooks and PyTorch gradient package were moved into:

- `old_notebooks_and_data/notebooks/superseded_clean_rewrite_20260525/`
- `old_notebooks_and_data/scripts/superseded_clean_rewrite_20260525/`

The active root workflow is now:

- `01_representation_dimension_and_stability.ipynb`
- `02_gradient_rank_and_angle.ipynb`
- `clean_jax_exp/`
- `run_clean_jax_experiment.py`
- `run_gradient_analysis_experiment.py`
- `run_unet1d_torch_experiment.py`

The pre-split hidden-representation result directory was also archived under:

- `old_notebooks_and_data/results/superseded_clean_rewrite_20260525/hidden_representation_dimension/`

## 2026-05-25 U-Net note

The JAX U-Net prototype was moved into:

- `old_notebooks_and_data/scripts/abandoned_jax_unet1d_20260525/`

It was correct enough for smoke tests but impractical on local CPU/JAX for full training. The active U-Net experiment is the PyTorch runner at the project root.
