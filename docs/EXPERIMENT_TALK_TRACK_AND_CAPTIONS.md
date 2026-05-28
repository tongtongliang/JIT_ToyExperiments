# Talk Track and Figure Captions for the Toy Diffusion Inductive-Bias Experiments

This note is written as collaborator-facing language. It is meant to help explain the experiment verbally while showing figures, videos, or notebook outputs. The goal is not to be maximally formal, but to make the motivation, design, and interpretation easy to communicate.

## 1. One-Minute Overview

We are studying a toy version of a common diffusion-model question: why does predicting clean data sometimes work much better than predicting velocity or noise, even when the network and the training data are the same?

The toy data live on a low-dimensional manifold embedded in a high-dimensional ambient space. At each training step, we sample a clean point `x`, fresh Gaussian noise `eps`, and a time `t`, then form a corrupted input

```text
z_t = (1 - t) x + t eps.
```

We train three otherwise identical models. The only difference is how the network output is interpreted: clean prediction, velocity prediction, or noise prediction. The loss is kept in a common velocity-objective form, so the comparison is about parameterization and gradient dynamics rather than changing the training target arbitrarily.

The main hypothesis is that the advantage of clean prediction is not just an expressivity story. It is an optimization-dynamics story. Early in training, clean prediction exposes stable low-dimensional signal directions, while velocity or noise prediction can expose higher-dimensional or less stable residual directions. AdamW momentum then accumulates these directions differently. Architecture can also change this effect: a fully connected network, a learned long-skip network, and a patch Transformer do not write the same information into their parameters.

## 2. Short Talk Track

A compact way to introduce the project is:

> The experiment asks whether the success of clean prediction comes from the geometry of the target alone, or from the way the target shapes early gradient dynamics. We take the same high-dimensional corrupted inputs and train matched models under three output parameterizations: predict clean data `x`, predict velocity `v`, or predict noise `eps`. Because the clean data lie on a low-dimensional manifold, clean prediction tends to produce stable low-rank residual structure. We then test whether this stability shows up in the gradient, AdamW momentum, and actual parameter updates. The key object is not only the final sample quality, but what directions the optimizer writes into the model during early training.

A slightly more informal version:

> I want to separate two explanations that are easy to conflate. One explanation says clean prediction works because the clean target is low-dimensional and noise is high-dimensional. That is true but incomplete. The more mechanistic explanation is that clean prediction gives the optimizer a stable low-dimensional signal early in training, while noise or velocity prediction can make the optimizer chase directions that are high-dimensional, unstable, or poorly routed by the architecture. The diagnostics here are designed to see that mechanism directly.

## 3. Experimental Design

### Data

The clean data are sampled from a low-dimensional distribution and embedded into a high-dimensional ambient space.

```text
x in R^D, usually D = 512
intrinsic dimension approximately 2
eps ~ N(0, I_D)
t ~ sigmoid(N(0, 1))
z_t = (1 - t) x + t eps
true velocity v* = eps - x
```

Fresh noise and fresh time values are sampled every training step. This is important: the model is not fitting a fixed table of noise vectors. The setup is meant to mimic the stochasticity of real diffusion training.

### Prediction Modes

We compare three output parameterizations:

| mode | network output means | velocity loss is computed by |
| --- | --- | --- |
| `x` prediction | model predicts clean data | convert predicted `x` to velocity through the interpolation identity |
| `v` prediction | model predicts velocity directly | compare directly to `eps - x` |
| `eps` prediction | model predicts noise | convert predicted `eps` to velocity through the interpolation identity |

The key design point is that the backbone architecture and data distribution are held fixed. Only the meaning of the network output changes.

### Models

The main controlled baseline is a 5-block residual fully connected network with pre-norm AdaLN-zero conditioning and a zero-initialized output layer. This is intentionally simple: it makes the last-layer gradient easy to reason about.

We then compare against architectural variants:

| model | purpose |
| --- | --- |
| FCN | clean baseline where the gradient mechanism is easiest to inspect |
| FCN + learned long skip | tests whether a learned linear-in-input path can absorb difficult velocity/noise structure |
| Patch Transformer | tests whether patch/token structure changes the rank and routing story |
| MLP-Mixer | tests whether token mixing without attention changes the geometry |
| Tiny 1D U-Net | planned/server-oriented architecture comparison for locality and multiscale mixing |

### Optimizer and Control Variables

Across matched runs, we keep the random seed, initialization, data generation, and optimizer settings as aligned as possible. The final output layer is zero-initialized. AdamW uses gradient clipping. This makes early training more interpretable because the initial prediction is controlled and the optimizer state is not dominated by numerical explosion.

## 4. What We Measure

### Sampling Metrics

These tell us whether the learned generative process recovers the target geometry.

| metric | interpretation |
| --- | --- |
| Ambient Chamfer distance | sample quality in the original high-dimensional space |
| Projected visualization | qualitative view in the true 2D data plane |
| Sample stable rank | whether generated samples concentrate in a few directions |
| Sample rank95 | number of PCA directions needed to explain 95% of sample variance |
| Sample subspace angle | whether the generated low-dimensional plane matches the true data plane |

Important caveat:

> Low rank is not automatically good. A model can collapse to a low-dimensional structure that is the wrong subspace. This is why sample rank must be read together with Chamfer distance and subspace angle.

### Representation Metrics

These tell us what the internal hidden states are doing.

| metric | interpretation |
| --- | --- |
| Hidden stable rank | whether layer representations concentrate in a few singular directions |
| Hidden rank95 | PCA dimension of hidden point clouds |
| Representation spectrum | how quickly hidden singular values decay |
| Noise stability / NSV | how much the hidden representation varies when the same clean sample is corrupted with fresh noise |
| AdaLN fan-in representation | representation after time modulation, right before the main nonlinear layer |

A useful oral explanation:

> The representation plots ask whether the model internally recovers a low-dimensional, corruption-stable coordinate system. If clean prediction succeeds because it locks onto the data manifold, we expect its hidden states to become low-dimensional and stable under resampled corruption noise.

### Gradient-Dynamics Metrics

These are the most mechanistic diagnostics.

For a linear layer, the gradient has a factorized form, up to transpose conventions:

```text
grad_W is built from activation and residual
```

In the scripts, we explicitly record:

| matrix | what it means |
| --- | --- |
| gradient | current step's raw gradient matrix |
| AdamW first moment | optimizer momentum, i.e. temporally accumulated gradient information |
| update | actual parameter update after optimizer preprocessing |
| activation | layer input/features used to form the gradient |
| residual | backpropagated output-side error used to form the gradient |

We then compute rank and angle diagnostics for these matrices.

| metric | interpretation |
| --- | --- |
| stable rank | soft measure of energy concentration |
| rank90 | number of singular directions explaining 90% matrix energy |
| principal angle | whether the dominant direction rotates over training |

A simple way to say this:

> Stable rank tells us whether the matrix energy is concentrated. Rank90 tells us how many directions actually carry most of the energy. Principal angle tells us whether the leading direction is stable from step to step. Together, these show whether the optimizer is accumulating a coherent signal or chasing changing directions.

## 5. How to Read the Main Figure Types

### Training-Loss Curves

Suggested caption:

> Training loss for the three output parameterizations under the same model and optimizer. The curves should not be read as the entire story: two models can have similar velocity loss but very different sample geometry. We use the loss curves mainly to check optimization progress, while geometry and rank diagnostics explain what kind of solution is being learned.

How to interpret:

| pattern | reading |
| --- | --- |
| `x` loss drops and samples look correct | clean prediction is both optimizing and recovering the manifold |
| `v` loss drops but ambient Chamfer remains high | the model may learn the projected geometry while spreading mass in ambient directions |
| `eps` loss drops but subspace angle is large | the model may collapse into a low-dimensional but wrong subspace |

### 2D Sample Plots

Suggested caption:

> Generated samples projected onto the true data plane. The black or reference points show the ground-truth data distribution, while colored points show samples produced by each trained model. This visualization is useful for qualitative geometry, but it can hide high-dimensional errors, so we pair it with ambient Chamfer distance.

How to interpret:

| visual pattern | reading |
| --- | --- |
| samples overlap ground truth in 2D and have low ambient Chamfer | strong recovery |
| samples look reasonable in 2D but have high ambient Chamfer | projection is misleading; there is off-manifold ambient error |
| samples form a thin line or wrong curve | low-dimensional collapse, possibly into the wrong subspace |
| samples are diffuse everywhere | failure to recover the data manifold |

### 3D PCA Sample Plots

Suggested caption:

> Samples are projected onto their leading principal components to reveal whether the generated point cloud has extra ambient spread or collapses into a wrong low-dimensional structure. This view complements the true 2D projection: the 2D projection tells us alignment with the data plane, while PCA tells us the geometry of the generated cloud itself.

How to interpret:

| pattern | reading |
| --- | --- |
| data and samples occupy similar 2D sheet | good geometric recovery |
| samples have a third-direction plume | ambient leakage or off-manifold noise |
| samples are rank-1 or line-like | collapse, even if some 2D projection looks plausible |

### Sample Rank Bar Charts

Suggested caption:

> Rank metrics of the generated sample point cloud. Stable rank measures soft energy concentration, while rank95 counts the number of PCA directions needed to explain 95% of variance. These metrics tell us whether the learned distribution is geometrically low-dimensional, but not whether it is the correct low-dimensional geometry.

Key warning:

> A low sample rank can be either good manifold recovery or bad collapse. We should read it together with Chamfer distance and subspace angle.

### Sample Subspace Angle Plots

Suggested caption:

> Angle between the generated sample PCA subspace and the true data plane. This is a direct test of whether a low-dimensional sample cloud is aligned with the actual clean-data manifold. A small angle means the model found the correct plane; a large angle means it may be low-dimensional but geometrically wrong.

Interpretation:

| angle | reading |
| --- | --- |
| near 0 degrees | generated samples lie in the correct data plane |
| moderate angle | rough alignment but imperfect geometry |
| near 90 degrees | generated samples are essentially in the wrong subspace |

### Hidden Representation Bar Charts

Suggested caption:

> Layerwise dimensionality of hidden representations. Each bar summarizes the rank or stable rank of the hidden point cloud at a given layer. We compare clean, velocity, and noise prediction to see whether successful training corresponds to internal low-dimensional structure.

How to explain:

> If the model learns the data manifold, hidden representations should often become more concentrated and more stable across corruption noise. If a prediction mode fails, we often see either high-dimensional hidden states or low-dimensional states that do not align with the right sample geometry.

### Representation Spectrum Plots

Suggested caption:

> Singular-value spectrum of hidden representations. A rapidly decaying spectrum means most representation energy lies in a few directions; a flatter spectrum means the representation uses many directions. This plot is a more detailed version of the rank bar chart.

How to interpret:

| spectrum shape | reading |
| --- | --- |
| sharp decay | low-dimensional representation |
| slow decay | high-dimensional representation |
| very sharp decay with bad samples | possible collapse rather than successful manifold learning |

### Representation Stability Plots

Suggested caption:

> Stability of hidden representations under repeated noise corruptions of the same clean sample. For each clean point, we resample noise several times and measure how much the hidden representation varies. Lower variation means the representation is more tied to the clean data and less sensitive to the particular noise draw.

Oral interpretation:

> This is a direct test of whether the model internally factors out corruption noise. If clean prediction works by learning stable data coordinates, we expect its hidden states to be more stable under resampled noise.

### Gradient Stable-Rank Curves

Suggested caption:

> Stable rank of layerwise gradient matrices during early training. Lower stable rank means the gradient energy is concentrated in fewer directions. We use this to test whether clean prediction gives the optimizer a lower-dimensional signal than velocity or noise prediction.

How to interpret:

| pattern | reading |
| --- | --- |
| clean gradient rank lower than v/eps | clean prediction exposes a more concentrated training signal |
| raw gradient noisy but momentum smooths | AdamW is filtering temporally unstable fluctuations |
| v/eps rank increases over time | optimizer is accumulating directions beyond the initial low-rank signal |

### AdamW Momentum Rank Curves

Suggested caption:

> Stable rank or rank90 of AdamW's first-moment buffer. This is one of the key diagnostics because momentum accumulates directions that are stable over time. If clean prediction has a stable low-dimensional residual signal, its momentum matrix should remain lower-rank than the corresponding velocity or noise momentum.

Oral emphasis:

> The momentum plot is more important than the raw gradient plot. The raw gradient is a single noisy snapshot. The first moment tells us what signal the optimizer is actually carrying forward across steps.

### Actual Update Rank Curves

Suggested caption:

> Rank of the actual parameter update after optimizer preprocessing. This tells us what is really being written into the network, not just what the raw gradient looked like. For AdamW and other preconditioned optimizers, the update can differ substantially from the raw gradient.

How to interpret:

| pattern | reading |
| --- | --- |
| update rank follows momentum rank | optimizer writes accumulated stable directions into parameters |
| update rank exceeds gradient rank | preconditioning or accumulation spreads energy into more directions |
| update rank stays low but samples fail | low-rank writing may be under-expressive or geometrically misaligned |

### Activation and Residual Rank Curves

Suggested caption:

> The gradient matrix is built from activations and residuals. By measuring both factors separately, we can tell whether high-rank gradients come from high-dimensional features, high-dimensional errors, or their interaction.

Useful explanation:

> This prevents us from saying simply that the target is high-rank. The gradient rank depends on both sides of the factorization: what features the layer provides and what residual signal is backpropagated through it.

### Principal-Angle Curves

Suggested caption:

> Principal-angle drift between adjacent training steps. A small angle means the dominant singular direction is stable over time; a large angle means the leading direction is rotating. This is useful for distinguishing coherent low-rank learning from low-rank but unstable stochastic directions.

Interpretation:

| pattern | reading |
| --- | --- |
| low rank and small angle | stable low-dimensional signal |
| low rank and large angle | low-dimensional but changing direction; momentum may accumulate many directions over time |
| high rank and small angle | broad but stable update structure |
| high rank and large angle | noisy or poorly aligned dynamics |

### Long-Skip Coefficient Plots

Suggested caption:

> Learned scalar skip coefficients in the long-skip FCN. The model output is decomposed as a direct input-dependent path plus a neural residual path. These plots show whether the model learns to route part of the prediction through a simple linear skip instead of forcing the MLP to represent everything.

Oral interpretation:

> The long skip is a way to test whether the FCN failure comes from making the network learn a large, simple, time-dependent linear component through ordinary hidden layers. If the skip absorbs that component, velocity prediction becomes much easier and the last-layer gradient rank collapses.

### Transformer Gradient-Rank Plots

Suggested caption:

> Gradient-rank diagnostics for the patch Transformer. Unlike the FCN, the Transformer distributes computation across patch embedding, QKV projections, attention output, MLP up/down projections, and the final patch decoder. We therefore track each matrix separately.

Key interpretation:

> The Transformer does not simply reproduce the FCN story. In the FCN, velocity and noise produce a high-rank burden at the final layer. In the Transformer, the final output rank can be small, and the effective rank is distributed across internal matrices, especially the MLP-up projections. This suggests that architecture changes the mechanism from high-rank writing at the output layer to routing and compression across patch/token features.

## 6. Main Empirical Messages

### Message 1: Clean prediction has a favorable gradient signature in FCNs.

Suggested wording:

> In the FCN baseline, clean prediction produces very low-rank output-layer momentum and updates. This means AdamW is accumulating a compact, stable signal. In contrast, velocity and noise prediction write many more directions into the output layer. This supports the idea that clean prediction has an optimization advantage, not only a representational advantage.

### Message 2: Velocity and noise failures are not identical.

Suggested wording:

> Velocity prediction can sometimes find the correct projected manifold while still having large ambient error. Noise prediction can collapse to a low-dimensional but wrong subspace. These are different failure modes, so we should not summarize everything as simply high-dimensional noise being hard.

### Message 3: The learned long skip changes the mechanism.

Suggested wording:

> The learned long skip strongly improves velocity prediction and also improves noise prediction. Mechanistically, it reduces the high-rank burden on the FCN's output layer. This suggests that part of the difficulty was forcing a plain MLP to learn a simple time-dependent linear component through ordinary nonlinear layers.

### Message 4: Patch architectures change the rank story.

Suggested wording:

> The patch Transformer and Mixer do not show the same last-layer rank explosion as the FCN. This means the FCN gradient story is real but architecture-dependent. Patch/token structure can compress or reroute the difficult directions. However, low-rank compression is not automatically good: for example, noise prediction can become low-rank but aligned with the wrong subspace.

### Message 5: Rank must be paired with geometry.

Suggested wording:

> A central lesson is that rank metrics are diagnostic, not sufficient. Low rank can mean successful manifold learning, but it can also mean collapse. High rank can mean ambient leakage, but it can also reflect a richer distributed representation. We need to read rank together with sample geometry, Chamfer distance, subspace angle, and representation stability.

## 7. Suggested Captions for a Figure Panel

### Panel A: Data and Corruption Process

> Clean samples lie on a low-dimensional manifold embedded in high-dimensional ambient space. Training inputs are generated by interpolating between clean data and fresh Gaussian noise, `z_t = (1 - t)x + t eps`. The same corrupted-input distribution is used for all prediction parameterizations.

### Panel B: Prediction Parameterizations

> We compare three models with identical architecture and initialization. The network output is interpreted as clean data `x`, velocity `v`, or noise `eps`; all modes are evaluated through a common velocity-style loss. This isolates the effect of output parameterization on training dynamics.

### Panel C: Sample Geometry

> Generated samples reveal whether each parameterization recovers the data manifold. Clean prediction recovers the correct low-dimensional geometry in the FCN baseline. Velocity may recover the projected plane but leak into ambient directions, while noise prediction can collapse into an incorrect subspace.

### Panel D: Hidden Representation Dimension

> Internal representations are measured as point clouds across samples. Successful models tend to develop lower-dimensional and more corruption-stable representations, while failing modes either remain high-dimensional or collapse into geometrically incorrect structures.

### Panel E: Gradient/Momentum Rank

> Early-training gradient diagnostics show what directions the optimizer writes into the network. In the FCN baseline, clean prediction has low-rank AdamW momentum at the final layer, while velocity and noise prediction accumulate many more effective directions.

### Panel F: Long-Skip Intervention

> Adding a learned time-dependent skip path greatly reduces the high-rank output-layer burden for velocity and noise prediction. This suggests that architecture can absorb simple linear components that are otherwise hard for the plain FCN to learn through its hidden layers.

### Panel G: Transformer Comparison

> Patch Transformer diagnostics show a different mechanism. The final projection no longer carries the same high-rank burden; instead, rank is distributed across internal patch and channel-mixing matrices. This indicates that the implicit bias depends strongly on architecture.

## 8. Common Questions and Suggested Answers

### Why not just say noise is high-dimensional and clean data are low-dimensional?

> That statement is directionally true but too coarse. The experiments show that the optimizer sees the target through activations, residuals, momentum, and architecture. In some models, noise prediction fails through high-rank parameter updates; in others, it becomes low-rank but collapses to the wrong subspace. So the mechanism is not target dimension alone.

### Why do we care about AdamW momentum instead of only gradients?

> The raw gradient is one noisy step. Momentum is the optimizer's memory of directions that persist over time. If clean prediction provides stable low-dimensional signal, this should be especially visible in the first-moment buffer.

### Why measure activation and residual separately?

> Because the weight gradient is formed from both. A high-rank gradient can come from high-dimensional features, high-dimensional residuals, or the interaction between them. Separating activation and residual lets us localize the source of the rank.

### Why can low rank still be bad?

> Low rank only says the sample or matrix energy is concentrated. It does not say the concentrated directions are correct. A model can collapse into a one- or two-dimensional structure that is not aligned with the true data manifold.

### What does the long skip teach us?

> It shows that the FCN's failure is not fixed by capacity alone. A small architectural reparameterization that gives the model a direct time-dependent linear path can dramatically change both sampling quality and gradient rank. That points to an implicit-bias and parameterization effect.

### What does the Transformer teach us?

> It shows that the FCN mechanism is architecture-dependent. Patch/token models can avoid the FCN's final-layer high-rank burden, but they may introduce a different failure mode: routing or compressing the signal into the wrong geometric subspace.

## 9. Final Takeaway

A concise final takeaway for collaborators:

> These toy experiments suggest that prediction parameterization changes the early optimization geometry. Clean prediction gives the optimizer stable low-dimensional directions in a plain FCN. Velocity and noise prediction can force the optimizer to write many directions, unless the architecture provides a better route such as a learned long skip or patch/token structure. But architecture can also over-compress the signal, so low rank by itself is not success. The right interpretation requires connecting gradient rank, optimizer momentum, hidden representation stability, and final sample geometry.
