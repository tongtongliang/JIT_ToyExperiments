# Project Summary: Diffusion Toy Inductive Bias Experiments

Last updated: 2026-05-27.

This document summarizes the current clean version of the project after the repository was reorganized into package code, script entry points, notebooks, and local result folders.

## 1. Scientific Motivation

The project studies a toy version of the prediction-parameterization question in diffusion and flow-matching models. The data live on a low-dimensional manifold embedded in a high-dimensional ambient space. The model observes corrupted samples and is trained with one of three prediction heads:

| mode | network output is interpreted as | velocity loss target |
| --- | --- | --- |
| `x` | clean data prediction | convert predicted `x` into velocity using `(z_t - x_pred) / t` |
| `v` | direct velocity prediction | compare directly to `eps - x` |
| `eps` | noise prediction | convert predicted noise into velocity |

The loss is kept velocity-style so that the objective is aligned across modes. The output parameterization is intentionally decoupled from the loss objective.

The original coarse explanation was: clean data lie on a low-dimensional manifold, while noise is high-dimensional. The refined hypothesis is more mechanistic:

```text
Early gradient dynamics and optimizer state selectively accumulate stable signal directions.
Clean prediction provides temporally stable low-dimensional residual structure.
Velocity/noise prediction can expose high-dimensional or unstable residual directions.
Architecture can either amplify, absorb, or compress those directions.
```

This is why the project now has three related but distinct experiment families:

| family | purpose |
| --- | --- |
| representation/stability | after long training, does the model learn low-dimensional internal and sampled structure? |
| gradient dynamics | during early training, what rank/angle structure is written into gradients, momentum, updates, activations, and residuals? |
| architecture comparisons | do local/patch architectures alter the failure mode seen in FCNs? |

## 2. Data and Training Protocol

The main D=512 experiments use a fixed low-dimensional data distribution projected into ambient space:

```text
clean sample: x in R^D, D=512
noise sample: eps ~ N(0, I)
time: t ~ sigmoid(N(0, 1)) unless fixed for analysis
corrupted input: z_t = (1 - t) x + t eps
velocity target: v = eps - x
```

Important implementation decisions:

| decision | reason |
| --- | --- |
| dynamic `eps`, `t`, and data indices every step | mimic real diffusion training and avoid fitting a fixed noise table |
| same random seed and initialization across x/v/eps within a run | isolate target parameterization |
| zero-initialized final output projection | make the initial prediction comparable and controlled |
| AdamW with gradient clipping | avoid early explosion and make optimizer-state analysis meaningful |
| train data snapshot saved per run | visualization and posthoc analysis can be reproduced |
| figures generated from saved logs | notebook visualization can change without rerunning training |

## 3. Repository Organization

The main code is divided as follows:

| path | role |
| --- | --- |
| `clean_jax_exp/` | reusable JAX FCN code, metrics, posthoc analysis, and plotting |
| `scripts/fcn/` | baseline FCN and long-skip FCN experiment runners |
| `scripts/architectures/` | PyTorch Transformer, Mixer, and U-Net architecture runners |
| `scripts/analysis/` | posthoc analysis helpers for completed runs |
| `01_representation_dimension_and_stability.ipynb` | main representation/stability visualization notebook |
| `02_gradient_rank_and_angle.ipynb` | main gradient dynamics visualization notebook |
| `results/` | local git-ignored experiment outputs |
| `old_notebooks_and_data/` | archived historical notebooks/data/scripts |

## 4. Main Metrics

| metric | used for | interpretation |
| --- | --- | --- |
| stable rank | matrices and point clouds | soft rank: `||A||_F^2 / ||A||_op^2` |
| rank90 | gradient matrices | number of singular directions explaining 90% energy |
| rank95 | representation and sample point clouds | number of PCA directions explaining 95% centered energy |
| principal angle | adjacent gradients/momenta/activations/residuals | whether dominant directions rotate across time |
| NSV | representation stability | corruption-noise variance normalized by representation norm |
| ambient Chamfer | sample quality | geometric sample/data distance in D-dimensional space |
| sample subspace angle | sample geometry | alignment between generated sample PCA plane and true data plane |

## 5. FCN Baseline: Representation Experiment

Primary run:

```text
results/clean_jax_representation/runs/repr_D512_adamw_w256_d5_s100000_seed42_20260525_103044
```

Architecture:

```text
5-block residual FCN
pre-norm AdaLN-zero
width = 256
D = 512
final output projection zero-initialized
optimizer = AdamW, lr = 1e-4, gradient clip = 1.0
steps = 100000
```

Sample results:

| mode | final loss | ambient Chamfer | sample stable rank | sample rank95 | max subspace angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | 0.0251 | 0.2399 | 1.85 | 2 | 0.0117 deg |
| v | 0.5305 | 30.5953 | 2.79 | 193 | 1.4691 deg |
| eps | 4.6310 | 29333.1133 | 140.37 | 232 | 87.6185 deg |

Interpretation:

| observation | reading |
| --- | --- |
| `x` learns a nearly perfect 2D manifold | clean prediction aligns with stable low-dimensional structure |
| `v` has acceptable 2D projected shape but huge ambient Chamfer/rank95 | it roughly finds the manifold plane but spreads mass in many ambient directions |
| `eps` fails catastrophically | noise prediction does not recover the correct 2D data plane in the FCN baseline |

This experiment is the origin of the project. The later gradient and stability experiments should be read as mechanistic probes explaining this phenomenon.

## 6. FCN Baseline: Gradient Dynamics

Primary run:

```text
results/clean_jax_gradient/runs/clean_jax_D512_adamw_w256_d5_s2000_seed42_20260525_114029
```

This run records, for each tracked matrix and every mode:

```text
gradient matrix
AdamW first moment
actual update matrix
activation matrix
residual matrix
principal-angle drift
sanity check for grad_W = residual.T @ activation
```

Final output-layer rank90 at step 2000:

| matrix kind | x | v | eps |
| --- | ---: | ---: | ---: |
| gradient | 2 | 68 | 18 |
| momentum | 1 | 108 | 63 |
| update | 1 | 112 | 63 |
| activation | 2 | 92 | 114 |
| residual | 2 | 143 | 19 |

Interpretation:

| observation | reading |
| --- | --- |
| `x` output momentum/update are essentially rank-1 | optimizer accumulates a very concentrated low-dimensional signal |
| `v` output momentum/update are extremely high-rank | velocity prediction writes many directions into the final layer |
| `eps` also has high output momentum/update rank, though less than `v` in rank90 | noise prediction still exposes a high-dimensional update burden |
| activation and residual rank are not identical to gradient rank | gradient rank is a product-structure phenomenon, not just target rank |

This supports the gradient implicit-bias story for the baseline FCN.

## 7. Learned Long-Skip FCN

Primary run:

```text
results/clean_jax_representation_longskip/runs/repr_longskip_D512_adamw_w256_d5_s100000_seed42_20260526_190226
```

Architecture change:

```text
raw(z_t, t) = c_skip(t) * z_t + c_out(t) * nnet(z_t, t)
```

The scalar controller is initialized so that:

```text
c_skip(t) = 0
c_out(t) = 1
```

Therefore the long-skip model starts like the baseline FCN, then learns whether to use a direct linear path.

Sample results:

| mode | ambient Chamfer | max subspace angle |
| --- | ---: | ---: |
| x | 0.3184 | 0.0144 deg |
| v | 0.7800 | 0.2298 deg |
| eps | 10.5241 | 0.5613 deg |

Gradient output-layer rank90 at step 2000:

| matrix kind | x | v | eps |
| --- | ---: | ---: | ---: |
| gradient | 1 | 3 | 2 |
| momentum | 2 | 3 | 6 |
| update | 2 | 3 | 8 |
| activation | 3 | 3 | 12 |
| residual | 2 | 37 | 80 |

Interpretation:

| observation | reading |
| --- | --- |
| velocity sampling improves dramatically compared with FCN | a learned linear skip can absorb the large linear-in-input component of velocity |
| eps also improves geometrically but remains less good than x/v | noise prediction still has a harder residual structure after the skip |
| output momentum/update effective rank collapses from tens/hundreds to single digits | the long skip removes the last-layer high-rank writing burden |
| learned `c_skip(t)` for v/eps behaves roughly like a time-dependent inverse-noise scaling | the model discovers an EDM-like role without hand-designed coefficients |

This is one of the strongest pieces of evidence that architectural parameterization can change the gradient-rank burden without changing the data distribution.

## 8. U-FCN Hidden Long-Skip Experiment

Primary run:

```text
results/clean_jax_representation_ufcn/runs/repr_ufcn_D512_adamw_w256_d5_s100000_seed42_20260530_145856
```

Architecture change:

```text
5-block residual FCN with pre-norm AdaLN-zero
U-ViT-style hidden long skips
block1 output -> block5 input
block2 output -> block4 input
concat([decoder stream, encoder skip]) -> linear projection back to width
```

The skip projections are initialized as `[I, 0]`, so the initial hidden stream matches the no-skip FCN while still allowing the skip half to learn. This keeps the comparison close to the baseline FCN initialization.

Default D=512 run:

```text
width = 256
depth = 5
steps = 100000
optimizer = AdamW, lr = 1e-4, gradient clip = 1.0
parameter count = 2,301,952
```

Sample results:

| mode | final loss | ambient Chamfer | sample stable rank | sample rank95 | max subspace angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | 0.0252 | 0.4371 | 1.89 | 2 | 0.0062 deg |
| v | 0.5298 | 30.5568 | 2.81 | 191 | 1.4838 deg |
| eps | 4.6227 | 29371.3848 | 140.30 | 231 | 86.9758 deg |

Comparison against the two closest FCN baselines:

| model | mode | final loss | ambient Chamfer | sample rank95 | max subspace angle |
| --- | --- | ---: | ---: | ---: | ---: |
| FCN | x | 0.0251 | 0.2399 | 2 | 0.0117 deg |
| FCN | v | 0.5305 | 30.5953 | 193 | 1.4691 deg |
| FCN | eps | 4.6310 | 29333.1133 | 232 | 87.6185 deg |
| U-FCN | x | 0.0252 | 0.4371 | 2 | 0.0062 deg |
| U-FCN | v | 0.5298 | 30.5568 | 191 | 1.4838 deg |
| U-FCN | eps | 4.6227 | 29371.3848 | 231 | 86.9758 deg |
| FCN + learned output long skip | x | 0.0248 | 0.3184 | 2 | 0.0144 deg |
| FCN + learned output long skip | v | 0.0249 | 0.7800 | 2 | 0.2298 deg |
| FCN + learned output long skip | eps | 0.0256 | 10.5241 | 2 | 0.5613 deg |

Interpretation:

| observation | reading |
| --- | --- |
| U-FCN almost exactly reproduces the plain FCN behavior for `v` and `eps` | hidden long skips alone do not remove the output-parameterization bottleneck |
| `x` still succeeds | the hidden skips do not damage clean prediction |
| `v` still has high ambient Chamfer and high sample rank95 | U-style hidden feature reuse does not absorb the large linear-in-input velocity component |
| `eps` still collapses to the wrong high-dimensional/noisy geometry | noise prediction remains a hard FCN parameterization problem |
| learned output long skip remains dramatically better for `v/eps` | the successful intervention is specifically an output-level, time-dependent linear path, not merely a U-shaped hidden architecture |

This is a useful negative result. It suggests that the FCN failure is not fixed by giving the hidden backbone more feature reuse. The important missing ingredient for velocity/noise prediction appears to be a direct route for the simple linear component of the target, as in the learned scalar output skip.

## 9. Patch Transformer Sampling

Primary completed 20k run:

```text
results/torch_transformer1d_sampling_mps_20k/runs/torch_transformer1d_resume_D512_adamw_p8_d128_h1_L5_m512_torch_from10000_to20000_seed42_20260526_082235
```

Architecture:

```text
D = 512
patch size = 8
patch count = 64
embedding dim = 128
heads = 1
depth = 5
MLP width = 512
AdaLN-zero blocks
zero-initialized patch decoder
```

Sample results:

| mode | final loss | ambient Chamfer | sample rank95 | max subspace angle |
| --- | ---: | ---: | ---: | ---: |
| x | 0.0311 | 0.8334 | 2 | 0.5475 deg |
| v | 0.0460 | 2.4751 | 2 | 0.7332 deg |
| eps | 0.0582 | 5.7374 | 1 | 88.1925 deg |

Interpretation:

| observation | reading |
| --- | --- |
| x and v are much closer in loss than in FCN | patch Transformer strongly changes the optimization landscape |
| v sample geometry is much better than FCN-v in ambient Chamfer | patch/token structure helps velocity prediction |
| eps still has a severe subspace failure | low sample rank does not imply correct data-plane alignment |
| eps rank95 can be low but wrong | the model may collapse to a low-dimensional but incorrect direction |

## 10. Transformer Gradient Dynamics

Primary run:

```text
results/torch_transformer1d_gradient/runs/torch_transformer_gradient_D512_adamw_p8_d128_h1_L5_m512_manual_s2000_seed42_20260527_094812
```

Tracked matrices:

```text
patch_embed
each block's qkv, attn_out, mlp0, mlp1
output_proj
```

Output projection rank90 at step 2000:

| matrix kind | x | v | eps |
| --- | ---: | ---: | ---: |
| gradient | 5 | 3 | 1 |
| momentum | 6 | 4 | 2 |
| update | 7 | 5 | 5 |
| activation | 37 | 34 | 40 |
| residual | 7 | 7 | 7 |

Category mean momentum rank90 at step 2000:

| category | x | v | eps |
| --- | ---: | ---: | ---: |
| patch_embed | 6.0 | 6.0 | 6.0 |
| qkv | 7.6 | 4.8 | 2.6 |
| attn_out | 4.4 | 3.0 | 1.8 |
| mlp0 | 13.4 | 10.2 | 7.2 |
| mlp1 | 8.8 | 6.8 | 5.0 |
| output_proj | 6.0 | 4.0 | 2.0 |

Interpretation:

| observation | reading |
| --- | --- |
| Transformer output rank90 reverses the FCN signature: `x > v > eps` | eps does not fail because of final-layer effective-rank explosion |
| MLP-up matrices carry the largest effective rank | internal channel mixing is the main high-rank route |
| attention matrices stay lower-rank | attention is not the dominant high-rank writer in this toy setup |
| eps remains hard despite low output rank | low rank can mean under-routing or wrong compression, not success |

This is an important correction to the original FCN story. The FCN mechanism is real, but it is architecture-dependent.

## 11. MLP-Mixer Sampling

Primary 10k run:

```text
results/torch_mixer1d_sampling_mps_10k/runs/torch_mixer1d_D512_adamw_p8_d128_L5_tm128_cm512_steps10000_seed42_20260526_104530
```

20k resumed run:

```text
results/torch_mixer1d_sampling_mps_20k/runs/torch_mixer1d_resume_D512_adamw_p8_d128_L5_tm128_cm512_from10000_to20000_seed42_20260530_210051
```

Architecture:

```text
patch size = 8
patch count = 64
width = 128
depth = 5
token MLP width = 128
channel MLP width = 512
AdaLN-zero conditioning
```

Sample results:

| run | mode | final loss | ambient Chamfer | sample rank95 | max subspace angle |
| --- | --- | ---: | ---: | ---: | ---: |
| Mixer 10k | x | 0.0284 | 0.6657 | 2 | 0.5290 deg |
| Mixer 10k | v | 0.0438 | 3.2419 | 2 | 0.5806 deg |
| Mixer 10k | eps | 0.0482 | 10.8002 | 2 | 2.3339 deg |
| Mixer 20k | x | 0.0279 | 0.5544 | 2 | 0.2951 deg |
| Mixer 20k | v | 0.0324 | 1.0531 | 2 | 0.3566 deg |
| Mixer 20k | eps | 0.0328 | 1.4664 | 2 | 0.5744 deg |

Interpretation:

| observation | reading |
| --- | --- |
| continuing Mixer from 10k to 20k substantially improves all three modes | this architecture was not saturated at 10k |
| eps ambient Chamfer drops from 10.80 to 1.47 | Mixer eventually learns a much better noise-prediction geometry than the 10k snapshot suggested |
| eps subspace angle drops from 2.33 deg to 0.57 deg | the generated eps-prediction samples are now close to the true 2D data plane |
| v ambient Chamfer drops from 3.24 to 1.05 | token/channel mixing continues improving velocity prediction with more steps |
| all modes remain rank95=2 at 20k | Mixer learns low-dimensional sample clouds for x, v, and eps |

## 12. U-Net Status

A Tiny 1D U-Net + AdaGN runner exists under:

```text
scripts/architectures/run_unet1d_torch_experiment.py
```

The intended design was:

```text
D = 512
local window = 4
stride = 2
base channels around 56
parameter count around 2M
```

This path is ready for GPU/server runs, but local Mac/MPS experiments were not pursued as heavily because the Transformer/Mixer path was simpler and faster to inspect.

## 13. Current Main Pattern

The experiments now suggest a sharper taxonomy:

| model family | x prediction | v prediction | eps prediction | likely mechanism |
| --- | --- | --- | --- | --- |
| FCN | succeeds | ambient high-rank failure | severe failure | final-layer/optimizer high-rank burden |
| FCN + learned long skip | succeeds | mostly fixed | much improved but imperfect | output skip absorbs large linear term |
| U-FCN hidden long skip | succeeds | same as FCN | same as FCN | hidden feature reuse does not absorb output-level linear burden |
| Transformer | succeeds/moderate | much improved | low-rank wrong subspace collapse | patch/token compression and routing |
| Mixer 20k | succeeds/moderate | strong improvement | strong improvement, near-correct subspace | token/channel MLP eventually routes v/eps into the right low-dimensional geometry |

The main conceptual shift is:

```text
Failure is not always high rank.
Sometimes failure is high-rank writing burden.
Sometimes failure is low-rank but wrong compression.
```

This is why future analysis should always pair rank metrics with geometry metrics: Chamfer, subspace angle, sample rank, and representation stability.

## 14. Practical Workflow Going Forward

Use this discipline for new experiments:

1. Add reusable model/training logic to package code or a script under `scripts/`.
2. Save `metadata.json`, `training_data_snapshot.npz`, losses, logs, and optional checkpoints.
3. Generate visualizations from saved CSV/NPZ files only.
4. Keep notebooks as lightweight viewers, not as the source of training logic.
5. Use `results/<family>/runs/<metadata-summary>_<timestamp>/` for every run.
6. Keep smoke/benchmark/full runs in separate output roots.
7. Do not commit `results/` or checkpoints.

## 15. Best Next Experiments

The current most valuable next steps are:

| direction | why it matters |
| --- | --- |
| run Transformer/Mixer longer on GPU | local 10k/20k may not represent converged behavior |
| add gradient dynamics for Mixer | compare token mixing vs attention in rank90 and angle drift |
| run U-Net on GPU | test whether locality plus multiscale mixing fixes v/eps more naturally |
| run gradient dynamics for U-FCN | verify whether hidden skips leave the same final-layer momentum/update ranks as the plain FCN |
| extend output long-skip design to Transformer/Mixer | test whether learned skip and patch architecture are complementary |
| add rank90/rank95 side-by-side in representation notebooks | prevent stable-rank-only interpretations |
| analyze target decomposition algebraically for eps/v | connect residual geometry with observed optimizer ranks |

## 16. Caveats

| caveat | consequence |
| --- | --- |
| local MPS sometimes needs unsandboxed execution | device availability can be misdetected inside Codex sandbox |
| some architecture runs are only 10k/20k | sampling conclusions are preliminary unless trained longer |
| `rank90` and stable rank can disagree | use both; they answer different questions |
| low rank is not automatically good | low-rank wrong-subspace collapse is a real failure mode |
| precision/recall were removed from plots | current implementation was not reliable enough for paper-level claims |

## 17. One-Sentence Summary

Clean prediction succeeds in the FCN because its early optimizer state sees stable low-dimensional directions; velocity/noise prediction can fail by forcing high-dimensional updates, but architecture and learned skip paths can transform that failure into a different regime where the model writes fewer directions yet may still compress the signal into the wrong geometry.
