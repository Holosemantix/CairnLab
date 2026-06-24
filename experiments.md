# SWM (Spherical World Model) Experiment Log

## Document Map

This log is organized by experiment phase:

1. **Current Working Summary**: high-level state of the SWM direction.
2. **Metric Scale References**: reference values for uniformity / spread losses.
3. **Collapse-Avoidance Experiments**: early TwoRoom collapse and anti-collapse experiments.
4. **Collapse Mechanism**: consolidated interpretation of why some losses fail.
5. **PushT Rollout-Space Consistency**: raw-vs-normalized rollout analysis.
6. **PushT Uniformity Ablation**: pair sampling, temporal masking, dimension, and temperature.
7. **Four-Task Temporal Hinge Comparison**: latest `epoch=10`, `num_eval=500` comparison batch.
8. **Appendix A**: raw PushT ablation records retained for traceability.
9. **Next Directions**: recommended follow-up experiments.
10. **Paper2 GLC adequacy baseline**: Reacher generic latent consistency
    implementation and negative gate result.
11. **Paper2 SNAP-ACPC PR-1A negative baseline**: Reacher one-step predictive
    consistency result and route decision.
12. **Paper2 paired no-aux equivalence control**: Reacher paired-view
    no-auxiliary diagnostic and next code-adaptation gate.

## 1. Current Working Summary

Goal: spherical representations (S^{d-1}) + non-SIGReg anti-collapse for world models (plan.md / plan_v2.md V0).

Base config: SphericalJEPA, ViT-Tiny, embed_dim=64, batch_size=128, T=4, Two-Room, lr=5e-5, spread weight=0.1.

Current best SWM baseline:

- `swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260415`
- core recipe: MLP projector + BatchNorm, normalized spherical prediction, uniformity regularizer with `weight=0.2`, `t=2`, `mode=temporal_masked`, `temporal_exclusion=2`, `embed_dim=64`
- benchmark at `epoch=10`, `num_eval=500`: TwoRoom `90.8`, Cube `74.0`, PushT `89.8`, Reacher `66.0`

Current working conclusion:

- temporal masking is a useful soft structural bias because it stops uniformity from pushing near-neighbor time steps apart
- fixed temporal hinge is much riskier because it actively pulls every adjacent transition together, regardless of action magnitude, contact, or task phase
- the next promising direction is action / transition-aware continuity rather than stronger global continuity

## 2. Metric Scale References

### Uniformity_loss scale reference (t=2, N=512)

| State | uniformity_loss value | Meaning |
|-------|----------------------|---------|
| Full collapse (all z_i identical) | **6.24** = log(511) | All sq_dist=0, exp(0)=1 |
| Random 64D unit vectors | **≈2.2** = log(511·e⁻⁴) | avg sq_dist≈2 |
| Well-spread | **< 2.2** | avg sq_dist > 2 |

### Mean cosine spread_loss scale reference

| State | spread_loss value |
|-------|------------------|
| Full collapse | **1.0** (all cosine sim = 1) |
| Random 64D unit vectors | **≈0.0** |

### Sliced spread_loss scale reference (`D=64`)

| State | sliced_spread_loss value | Meaning |
|-------|--------------------------|---------|
| Full collapse | **≈0.03121** (`≈ 2 / D`) | Sorted projections all equal to one constant |
| Random 64D unit vectors | **≈8e-5** | Projection order statistics match target quantiles |
| Well-spread | **→ 0** | 1D marginals close to uniform-on-sphere target |

### Results overview

| # | Approach | Loss type | Init spread | E0 fit spread | E0 val spread | Collapse broken? | Failure mode |
|---|----------|-----------|-------------|--------------|---------------|-----------------|--------------|
| 1 | Linear + mean cosine | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 2 | + detach target | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 3 | InfoNCE (τ=0.1) | InfoNCE | 16.24 | 16.24→7.55(E8) | 7.57 | E1 violent break | Destroys temporal structure |
| 4 | MLP+BN projector | cosine | 1.0 | **0.06** | **1.0** | Train only | Batch masking |
| 5 | + LayerNorm pre-L2 | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 6 | + noise (σ=1e-2) | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 7 | variance_loss on emb_raw | variance | 0.996 | 1.0 | 1.0 | No | Gradient dead zone (0/0) |
| 8 | DINO centering + uniformity | uniformity | 6.24 | **3.94** | **6.24** | Train only | Batch masking |
| 9 | uniformity_loss only (no centering) | uniformity | 6.24 | 6.24 | 6.23 | No | Gradient dead zone |
| 10 | sliced Wasserstein on sphere | sliced | 0.03296 | — | 0.03126 (E99) | No | Weak signal after L2 normalisation |
| 11 | MLP+BN + uniformity | uniformity | 4.28 | 2.83 | 4.28 | Yes | Slow BN / running-stats alignment |

### Three failure modes

| Mode | Experiments | Mechanism |
|------|------------|-----------|
| **Gradient dead zone** | 1, 2, 5, 6, 7, 9 | At collapse all z_i identical → (z_i−z_j)=0, std=0/0 → zero gradient |
| **Batch-dependent masking** | 4, 8 | BN / centering creates diversity during training only → loss gets no corrective signal → eval reveals true collapse |
| **Temporal destruction** | 3 | InfoNCE pushes ALL pairs apart (including adjacent frames) → conflicts with pred_loss |
| **Weak post-norm signal** | 10 | Sorting gives gradient, but L2 normalisation already erased most cross-sample variation |
| **Slow but real escape** | 11 | BN creates useful perturbation; eval lags early, then running stats catch up and collapse is broken |

### Root cause

Random ViT CLS tokens = shared component (v, magnitude ~10) + input-dependent variation (ε_i, magnitude ~0.01). L2 normalisation maps v+ε_i to v/||v|| ≈ same unit vector for all i, killing the ε_i signal. This is unique to spherical architectures — LeWM in R^d preserves ε_i and SIGReg's sorting mechanism detects it.

### What works (SIGReg) and why

SIGReg succeeds because of **sorting + quantile matching** inside Epps-Pulley: project embeddings to random 1D direction → sort → compare sorted values against Gaussian characteristic function. Sorting assigns different ranks to identical values, providing non-zero gradient at exact collapse. No batch-dependence, no pairwise differences.

---

## 3. Collapse-Avoidance Experiment Details

### Exp 1: V0 Baseline (Linear + mean cosine spread_loss)

**Config**: `nn.Linear(hidden_dim, embed_dim)` projector, spread_loss = mean pairwise cosine, weight=0.1.

| Stage | pred_loss | spread_loss | loss |
|-------|-----------|-------------|------|
| Init | 1.061 | 1.0 | 1.162 |
| E3 fit | 2.1e-5 | 1.0 | 0.100 |
| E3 val | 2.6e-5 | 1.0 | 0.100 |

Complete collapse. Predictor trivially outputs constant.

---

### Exp 2: + detach() on target

**Change**: `tgt_emb = emb[:, n_preds:].detach()`.

No improvement. Collapse is in the encoder, not gradient flow through target.

---

### Exp 3: InfoNCE spread_loss (τ=0.1)

**Change**: `logsumexp(sim/τ)`. InfoNCE scale: collapse=16.24, random=6.24.

| Stage | pred_loss | spread_loss |
|-------|-----------|-------------|
| Init | 1.080 | 16.236 (collapse) |
| E0 fit | 0.01 | 16.236 (still collapsed) |
| E1 fit | 0.966 | 9.286 (broke out) |
| E8 fit | 0.875 | 7.551 (near random) |
| E8 val | 0.879 | 7.573 |

Broke collapse at E1 via float noise accumulation, but pred_loss jumped to untrained level (0.97) and barely recovered. InfoNCE pushes temporal neighbours apart, directly opposing pred_loss.

---

### Exp 4: MLP projector with BatchNorm

**Change**: `MLP(hidden_dim, 2048, embed_dim, norm_fn=BatchNorm1d)`.

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 0.935 | 1.0 |
| E0 fit | 0.049 | **0.06** |
| E0 val | 0.017 | **1.0** |

Train/val split: BN decorrelates during training (batch stats) → spread sees no collapse → no corrective gradient. Eval with running stats → collapse exposed.

---

### Exp 5: LayerNorm before L2 normalize

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 1.104 | 1.0 |
| E0 fit | 4.5e-4 | 1.0 |

LayerNorm is per-sample. Doesn't fix cross-sample similarity.

---

### Exp 6: Noise injection (σ=1e-2)

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 0.880 | 1.0 |
| E0 fit | 6.0e-4 | 1.0 |

σ=1e-2 negligible vs embedding magnitude (~10). Angular perturbation ≈ 0.01 rad.

---

### Exp 7: variance_loss on pre-L2-norm embeddings

**Change**: `clamp(1 - std_per_dim, min=0).mean()` on emb_raw.

| Stage | pred_loss | spread_loss (variance) |
|-------|-----------|----------------------|
| Init | 1.138 | 0.996 |
| E0 fit | 7.9e-4 | 1.0 |

variance gradient = (x_i − mean)/(N·std) = 0/0 at collapse. PyTorch resolves to 0.

---

### Exp 8: DINO centering + uniformity_loss

**Change**: Subtract batch mean (train) / EMA center (eval) before L2 norm. uniformity_loss (t=2) on normalised embeddings.

| Stage | pred_loss | spread_loss (uniformity) |
|-------|-----------|--------------------------|
| Init | 1.053 | 6.236 (= collapse) |
| E0 fit | 0.986 | **3.935** (partially spread) |
| E0 val | 0.350 | **6.235** (= collapse) |

Same pattern as Exp 4: centering creates diversity during training (fit spread improved), but eval with EMA center still collapsed. Centering ≈ BN = batch-dependent operation that masks collapse from the loss.

---

### Exp 9: uniformity_loss only (no centering, Linear projector)

Isolate uniformity_loss without any architectural changes to confirm baseline behaviour.

| Stage | pred_loss | spread_loss (uniformity) |
|-------|-----------|--------------------------|
| Init | 0.932 | 6.236 (= collapse) |
| E0 fit | 6.2e-4 | 6.236 (= collapse) |
| E0 val | 8.3e-5 | 6.235 (= collapse) |

Complete collapse. uniformity_loss has zero gradient at collapse (same dead zone as other pairwise losses).

---

### Exp 10: sliced_spread_loss (sorting + quantile matching on sphere)

**Change**: replace pairwise/uniformity spread loss with `sliced_spread_loss()`:
- project all `B*T` unit vectors onto random directions
- sort projections independently per direction
- match sorted values against `N(0, 1/D)` quantiles

Training command:

```bash
python train_swm.py --config-name=swm.yaml \
    data=tworoom \
    subdir=ckpt/swm_v0_20260414_exp10_sliced_spread_loss \
    wandb.enabled=False
```

Config: `embed_dim=64`, `spread.weight=0.1`, `n_projections=256`.

| Stage | pred_loss | spread_loss (sliced) | loss |
|-------|-----------|----------------------|------|
| Init val sanity check | 0.9465 | 0.03296 | 0.94979 |
| E99 fit | 5.31e-6 | 0.03351 | 0.03356 |
| E99 val | 4.98e-6 | 0.03126 | 0.03131 |

Interpretation:
- `pred_loss` goes essentially to zero, so the predictor learns the trivial constant-latent solution perfectly.
- `spread_loss` stays almost exactly at the collapse baseline from start to finish.
- Unlike Exp 4 / 8, fit and val agree: this is **not** batch masking.
- Therefore sorting provides a non-zero gradient in principle, but in the current spherical pipeline that signal is too weak to move the model away from the collapsed basin.

Likely mechanism:
- The loss is applied **after** L2 normalisation on `S^{d-1}`.
- Earlier experiments already suggested raw ViT CLS features look like `v + ε`, where `||v|| >> ||ε||`.
- L2 normalisation maps `v + ε` close to `v / ||v||`, suppressing the useful variation before the spread loss sees it.
- `sliced_spread_loss` avoids the exact zero-gradient dead zone, but it still cannot overcome the trivial predictor solution once the representation has already been angularly flattened.

---

### Exp 11: MLP+BatchNorm projector + uniformity_loss

**Change**:
- use the original LeWM-style `MLP(..., BatchNorm1d)` projector and predictor projector
- keep spherical encoder / predictor outputs (`L2` normalised)
- use `uniformity_loss(t=2)` as spread loss

Training run:
- SwanLab run: `swm_v0_bn_uniform_lambda_0p1_t_2_20260415`
- URL: `https://swanlab.cn/@qunteam/worldmodels/runs/x3poay2amzei0vi6f1rlt/chart`

Config: `embed_dim=64`, `spread.weight=0.1`, `spread.type=uniformity`, `t=2.0`.

| Stage | pred_loss | spread_loss (uniformity) | loss |
|-------|-----------|--------------------------|------|
| Early fit (step 49) | 0.9712 | 2.8337 | 1.2545 |
| Early val (epoch 1) | 0.1814 | 4.2845 | 0.6098 |
| Mid fit peak spread (step 299) | 0.0247 | 6.0299 | 0.6277 |
| Final fit (step 128399) | 0.060 | 2.4655 | 0.2526 |
| Final val (epoch 99) | 0.0220 | 2.4712 | 0.2493 |

Interpretation:
- This is **not** the Exp 4 / Exp 8 failure mode.
- Validation spread does **not** stay near the collapse baseline (`6.24`); it falls to `2.47`, close to the random-spread reference (`≈2.2`).
- Validation pred loss also goes very low (`0.181 -> 0.022`), so the model is not merely decorrelated in training mode.
- Therefore BN + uniformity appears to be the first spherical variant that **really escapes collapse on validation**, not just on train.

Training dynamics:
- Early on, train and validation both fluctuate strongly.
- Fit spread initially rises toward the collapse scale (`≈6`), while validation spread is still high and noisy.
- After several epochs, validation rapidly improves and then stabilises around `2.47`.
- This suggests BN is doing more than masking: it injects enough cross-sample variation to break symmetry, but eval-mode running statistics need time to align before that benefit appears on validation.

Updated judgement on BatchNorm:
- Exp 4 was too pessimistic as a universal conclusion.
- BN still carries a train/eval mismatch risk.
- But at least in this long-run BN+uniformity setting, the mismatch is **transient**, not permanent masking.
- The most accurate description is: **slow-starting but genuinely non-collapsed**.

Remaining caveat:
- These metrics show that the representation escapes collapse.
- They do **not yet** prove better downstream planning / control performance.
- Exp 11 should therefore be treated as a promising positive result pending evaluation.

---

## 4. Collapse Mechanism: The Gradient Dead Zone

At exact collapse (all z_i identical), gradient is zero for ALL tested losses:

| Loss type | Why zero at collapse |
|-----------|---------------------|
| Pairwise (cosine, uniformity, InfoNCE) | (z_i − z_j) = 0 |
| Statistical (variance, covariance) | (x_i − mean) = 0, std = 0 → 0/0 |
| Gram matrix on post-norm | Gradient ∝ (G−I)z₀, killed by L2 Jacobian (I−z₀z₀ᵀ)z₀ = 0 |

**Mechanisms now known to work**:
- sorting + quantile matching (used inside SIGReg's Epps-Pulley)
- BatchNorm-assisted uniformity, when trained long enough for eval running statistics to align

Sorting assigns different ranks to identical values → different target quantiles → non-zero gradient. Does not involve (z_i − z_j) or batch statistics.

### Updated interpretation after Exp 10

Exp 10 refines the conclusion:
- Sorting is likely **necessary** to escape exact collapse.
- But sorting **on post-L2 spherical embeddings is not sufficient**.
- The remaining bottleneck is probably the representation pipeline itself: useful variation is destroyed by early normalisation, so the spread objective receives too little signal too late.

This points to the next design direction:
- keep spherical prediction / planning if desired
- but apply anti-collapse regularisation on **pre-normalised** projector outputs, where the small non-collapsed variation still exists
- or otherwise preserve / amplify that variation before projecting to the sphere

### Updated again after Exp 11

Exp 11 further refines the picture:
- BatchNorm can be more than batch masking.
- When combined with uniformity and trained long enough, it can provide a real route out of collapse.
- The apparent contradiction with Exp 4 is explained by time scale: early validation can look worse before BN running statistics catch up.

So the current picture is:
- post-L2 linear projector + pairwise/statistical spread losses: fail
- post-L2 sliced spread alone: fail
- BN + uniformity: **promising success**
- SIGReg in Euclidean space: still the cleanest known robust baseline

---

## 5. PushT Rollout-Space Consistency

After the early collapse-focused Two-Room experiments, we ran a stricter PushT
comparison to answer a different question:

> when prediction is trained in one latent space, does planning also need to
> roll out in that same space?

The short answer from the current runs is **yes**.

### PushT comparison snapshot

| Model | Prediction training space | Rollout state space | Final cost space | PushT eval | Main representation takeaway |
|---|---|---|---|---|---|
| LeWM / SIGReg | raw MSE in Euclidean space | raw | raw MSE | **94** | Best global geometry and strongest action-driven latent motion |
| SWM Exp A: BN + norm cosine + norm uniformity | normalized cosine | normalized | normalized cosine | **62** | Self-consistent rollout, but geometry is weaker than LeWM |
| SWM Exp B2: BN + raw MSE + norm uniformity | raw MSE | normalized | raw MSE | **0** | Train / rollout mismatch; changing only final cost does not rescue planning |
| SWM Exp C2: raw-consistent + norm uniformity | raw MSE | raw | raw MSE | poor / unstable | Space-consistent Euclidean control still does not reproduce the spherical branch |

### Representation evidence from PushT

The representation analysis makes the failure mode clearer:

- Exp A (`BN + norm_cosine + norm_uniformity`) still has imperfect geometry, but
  it retains usable local dynamics:
  - `distance_rank_corr_cross_seq = 0.046`
  - `knn_overlap_cross_seq = 0.225`
  - `latent_state_step_corr = 0.365`
  - `pred_target_cosine_mean = 0.994`
- Exp B2 (`BN + raw_mse + norm_uniformity`, but `rollout_state_space=normalized`)
  breaks much more severely:
  - `distance_rank_corr_cross_seq = 0.045`
  - `knn_overlap_cross_seq = 0.037`
  - `latent_state_step_corr = -0.133`
  - `pred_target_cosine_mean = 0.239`
  - `mean_pred_shift_norm = 0.347`

Interpretation:

- Exp A is bad mainly because its geometry is weaker than LeWM, not because the
  planner uses the wrong space.
- Exp B2 is worse because the predictor is trained with `pred.space=raw`, but
  planning still closes the autoregressive loop in `rollout_state_space=normalized`.
- This shows that matching the **prediction-loss space** to the **rollout space**
  matters more than matching only the final planning cost.
- Exp C2 matters for a different reason: even after restoring raw-space
  consistency (`pred.space=raw`, `rollout_state_space=raw`, `cost_space=raw`),
  the Euclidean control branch still fails to match the spherical branch.

### Additional raw-vs-normalized check

We then extended `repr_analysis` to report both normalized and raw embedding /
topology metrics for hybrid models.

For the current BN-based raw-MSE run, raw and normalized geometry turned out to
be almost identical. That means:

- the failure is **not** mainly caused by the final `L2 normalize()`
- `emb_raw` is already close to a fixed-norm thin shell, so raw space does not
  preserve much extra amplitude information
- with this architecture, switching only `cost_space` cannot rescue planning

This points away from "cost mismatch only" and toward two separate conclusions:

> the prediction training space and the rollout state space should be the same
> if we want stable multi-step planning.

and

> within the current `uniformity`-style recipe family, restoring Euclidean
> space consistency is not sufficient to recover the best spherical result.

### Recommended consistent experiment lines

The key comparison is therefore not the hybrid B2 setup, but two fully
self-consistent branches:

### Exp C1: spherical-consistent

Use this to measure the best version of the spherical branch without changing
its core geometry.

```yaml
encoder:
  projection_head:
    type: mlp
    norm_fn: batchnorm1d

loss:
  pred:
    type: cosine
    space: normalized
  regularizer:
    type: uniformity
    space: normalized
    weight: 0.1

wm:
  inference:
    rollout_state_space: normalized
    cost_space: normalized
    cost_type: cosine
```

Why this branch matters:

- it keeps training, rollout, and planning in the same spherical space
- it is the clean follow-up to Exp A
- any remaining gap to LeWM can then be attributed to geometry quality, not
  train / rollout inconsistency

### Exp C2: raw-consistent

Use this to test whether raw-MSE dynamics actually become useful once the
predictor is trained and rolled out in the same raw space.

```yaml
encoder:
  projection_head:
    type: mlp
    norm_fn: none

loss:
  pred:
    type: mse
    space: raw
  regularizer:
    type: uniformity
    space: normalized
    weight: 0.1

wm:
  inference:
    rollout_state_space: raw
    cost_space: raw
    cost_type: mse
```

Why this branch matters:

- it aligns prediction loss, rollout, and final planning cost in raw space
- it directly tests the "raw dynamics" hypothesis instead of only changing the
  terminal cost
- removing `BatchNorm1d` from the projector is important here, because the
  current BN-based raw run suggests `emb_raw` is already being compressed into a
  near-shell geometry before planning ever sees it
- in practice, this branch still fails to produce a competitive model under the
  current `uniformity`-style recipe, so it serves as a negative Euclidean
  control rather than a promising alternative baseline

### Updated interpretation after PushT

Current evidence supports the following rule of thumb:

- `pred.space` should match `rollout_state_space`
- `cost_space` should usually match them too
- `regularizer.space` may differ if needed

In other words, the important consistency is along the **dynamics path**:

> prediction target space = autoregressive rollout space = planning space

The current hybrid `raw pred + normalized rollout + raw cost` setup does not
meet that requirement and should not be treated as the main raw-MSE baseline.

What the current evidence **does** support:

- for this family of `uniformity`-regularized objectives, a spherical-consistent
  branch is substantially more effective than the tested Euclidean alternatives
- rollout-space consistency is necessary
- Euclidean consistency alone is not sufficient

What the current evidence **does not yet** support:

- a universal claim that spherical latent spaces are always better than
  Euclidean latent spaces
- a universal claim that any auxiliary loss or planner will work better just
  because the latent space is spherical

So the strongest defensible claim at this stage is narrower:

> under the current `uniformity + temporal masking (+ rollout / inverse-dynamics
> extensions)` recipe family, the spherical-consistent branch is the one that
> trains and evaluates well, while the tested Euclidean controls do not.

---

## 6. PushT Uniformity Ablation: Weight, Pair Sampling, and Temporal Exclusion

To understand which part of the SWM regularizer stack actually matters on
PushT, we ran a focused ablation with fixed evaluation settings:

- training checkpoint: `epoch=10`
- evaluation budget: `num_eval=500`
- dataset / task: `PushT`

Important notation for this section:

- `t` = uniformity temperature
- `temporal_masked_k` = `loss.uniformity.mode=temporal_masked` with
  `temporal_exclusion=k`

### PushT ablation snapshot

| Model | PushT eval |
|---|---:|
| `lewm` | **89.4** |
| `swm_mlp_bn_uniform_w_0p1_t_2_dim_192` | 65.6 |
| `swm_mlp_bn_uniform_w_0p2_t_2_dim_192` | 74.4 |
| `swm_mlp_bn_uniform_w_0p2_t_2_dim_64` | 69.8 |
| `swm_mlp_bn_uniform_w_0p1_t_2_cross_window_dim_192` | 74.4 |
| `swm_mlp_bn_uniform_w_0p2_t_2_cross_window_dim_192` | 80.2 |
| `swm_mlp_bn_uniform_w_0p2_t_2_cross_window_dim_64` | 82.2 |
| `swm_mlp_bn_uniform_w_0p1_t_2_temporal_masked_1_dim_192` | 71.4 |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_1_dim_192` | 80.0 |
| `swm_mlp_bn_uniform_w_0p2_t_1_temporal_masked_1_dim_64` | 64.6 |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_1_dim_64` | 81.2 |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64` | **89.8** |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_192` | 80.6 |
| `swm_mlp_bn_uniform_w_0p3_t_2_temporal_masked_2_dim_64` | 85.2 |
| `swm_mlp_bn_uniform_w_0p3_t_2_temporal_masked_2_dim_192` | 82.0 |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_3_dim_64` | 67.0 |

### Main takeaways

1. Increasing `uniform_w` from `0.1` to `0.2` helps consistently.
2. Changing pair sampling from `all_pairs` to `cross_window` or
   `temporal_masked` matters more than changing the base MLP+BN backbone alone.
3. `dim=64` is **not** universally better. It hurts the plain `all_pairs`
   baseline, but helps strongly once temporal structure is imposed.
4. `temporal_exclusion=2` is the best setting at `dim=64`, but that gain does
   not transfer cleanly to `dim=192`.
5. `uniform_w=0.2` is better calibrated than `0.3` around the best branch.
6. `temporal_exclusion=3` is too aggressive on PushT and collapses performance.
7. The current best SWM run (`89.8`) is effectively on par with `lewm` (`89.4`)
   on PushT, but the broader 4-task comparison now suggests SWM is slightly
   stronger overall.

### Controlled comparisons

#### Effect of uniformity weight

At fixed `t=2`:

| Setting | `uniform_w=0.1` | `uniform_w=0.2` | Gain |
|---|---:|---:|---:|
| `all_pairs`, `dim=192` | 65.6 | 74.4 | +8.8 |
| `cross_window`, `dim=192` | 74.4 | 80.2 | +5.8 |
| `temporal_masked_1`, `dim=192` | 71.4 | 80.0 | +8.6 |

Interpretation: `uniform_w=0.1` is too weak in this PushT setting; `0.2`
consistently improves downstream control.

#### Effect of pair sampling / structural bias

At fixed `uniform_w=0.2`, `t=2`, `dim=192`:

| Setting | PushT eval |
|---|---:|
| `all_pairs` | 74.4 |
| `cross_window` | 80.2 |
| `temporal_masked_1` | 80.0 |

Interpretation: the main benefit is not just stronger uniformity, but applying
it with a more appropriate pair-selection structure. Both `cross_window` and
`temporal_masked` avoid over-penalizing temporally related samples compared with
plain `all_pairs`, and `temporal_masked` retains headroom for larger gains once
`temporal_exclusion` is tuned.

#### Effect of latent dimension

At fixed `uniform_w=0.2`, `t=2`:

| Setting | `dim=192` | `dim=64` | Gain |
|---|---:|---:|---:|
| `all_pairs` | 74.4 | 69.8 | -4.6 |
| `cross_window` | 80.2 | 82.2 | +2.0 |
| `temporal_masked_1` | 80.0 | 81.2 | +1.2 |
| `temporal_masked_2` | 80.6 | 89.8 | +9.2 |

Interpretation: reducing latent dimension is not a standalone win. It hurts the
plain `all_pairs` branch, mildly helps `cross_window` / `temporal_masked_1`,
and helps **a lot** only for `temporal_masked_2`. This points to a strong
interaction: `dim=64` becomes useful when the temporal masking policy is
already well aligned with the task.

#### Effect of temperature `t`

For `temporal_masked_1`, `uniform_w=0.2`, `dim=64`:

| `t` | PushT eval |
|---|---:|
| 1 | 64.6 |
| 2 | 81.2 |

Interpretation: `t=2` is much better than `t=1` in the currently tested
configuration. This is a strong signal that the uniformity temperature matters,
but it is still based on one architecture slice rather than a full sweep.

#### Effect of temporal exclusion

At fixed `uniform_w=0.2`, `t=2`:

| Setting | `dim=192` | `dim=64` |
|---|---:|
| `temporal_masked_1` | 80.0 | 81.2 |
| `temporal_masked_2` | 80.6 | **89.8** |

Interpretation: increasing `temporal_exclusion` from `1` to `2` is the largest
single improvement in this ablation, but only at `dim=64`. At `dim=192`, the
gain is marginal (`80.0 -> 80.6`). So `temporal_exclusion=2` is not a universal
rule; it is a strong choice specifically in the smaller latent setting.

#### Effect of stronger uniformity weight near the best branch

At fixed `t=2`, `temporal_exclusion=2`:

| Setting | `uniform_w=0.2` | `uniform_w=0.3` | Gain |
|---|---:|---:|---:|
| `dim=64` | 89.8 | 85.2 | -4.6 |
| `dim=192` | 80.6 | 82.0 | +1.4 |

Interpretation: `uniform_w=0.3` is not a clean improvement and clearly hurts
the best `dim=64` branch. The current evidence supports `uniform_w=0.2` as the
best-calibrated value near the top-performing regime.

#### Effect of larger temporal exclusion

At fixed `uniform_w=0.2`, `t=2`, `dim=64`:

| Setting | PushT eval |
|---|---:|
| `temporal_masked_1` | 81.2 |
| `temporal_masked_2` | **89.8** |
| `temporal_masked_3` | 67.0 |

Interpretation: the exclusion range has a clear optimum. `1` is too small, `3`
is too large, and `2` is the sweet spot. Excluding too many nearby temporal
pairs appears to remove too much useful contrastive pressure and weakens the
uniformity objective substantially.

### Updated working interpretation for PushT

The strongest SWM result in this batch does not come from larger capacity. It
comes from a better match between the uniformity objective and PushT's temporal
structure:

- strong enough regularization (`uniform_w=0.2`)
- pair selection that respects temporal locality
- a moderate uniformity temperature (`t=2`)
- a **moderate** temporal exclusion (`temporal_exclusion=2`, not `1` or `3`)
- compact latent size (`dim=64`)

This suggests that for PushT, regularizer geometry and temporal sampling policy
matter more than simply scaling embedding dimension. It also suggests the
best-performing region is a fairly narrow interaction regime rather than a
single dominant scalar hyperparameter.

### Updated benchmark view across four tasks

We also evaluated the current best SWM setting on the four main downstream
tasks.

| Model | TwoRoom | Cube | PushT | Reacher |
|---|---:|---:|---:|---:|
| `lewm` | 93.0 | 69.2 | 89.4 | 62.2 |
| `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64` | 90.8 | 74.0 | 89.8 | 66.0 |

Task-wise comparison:
- `TwoRoom`: SWM is slightly worse (`90.8` vs `93.0`)
- `Cube`: SWM is clearly better (`74.0` vs `69.2`)
- `PushT`: effectively tied, with a slight SWM edge (`89.8` vs `89.4`)
- `Reacher`: SWM is clearly better (`66.0` vs `62.2`)

Interpretation:
- SWM is no longer just a PushT-specific improvement.
- The current best configuration appears to trade a small amount of TwoRoom
  performance for stronger 3D/control-task generalization.
- On a simple unweighted 4-task average, SWM is now slightly ahead of LeWM.

This makes the current SWM branch worth treating as a genuine alternative
baseline rather than a task-specific ablation curiosity.

### Remaining gaps

- The best SWM setting now looks promising across four tasks, but it still
  needs multi-seed confirmation before claiming a robust overall win.
- TwoRoom remains the one task where the current SWM variant underperforms
  LeWM, so the next dynamics-oriented improvements should be judged partly by
  whether they recover that gap without giving back gains on Cube / Reacher.
- The current sweep still leaves open whether improvements now come mainly from
  better representation geometry, better action-conditioned dynamics, or both.

---

## 7. Four-Task Temporal Hinge Comparison

This section records the latest comparison batch provided in CSV form.

Evaluation protocol:

- checkpoint epoch: `10`
- eval budget: `num_eval=500`
- tasks: TwoRoom, Cube, PushT, Reacher

Run names are preserved as recorded, including typos such as `tworroom` and
`reache`, so the table can be traced back to the original experiment names.

Naming convention:

- `w01` = `loss.temporal_hinge.weight=0.1`
- `w001` = `loss.temporal_hinge.weight=0.01`
- `m01` = `loss.temporal_hinge.margin=0.1`
- `m05` = `loss.temporal_hinge.margin=0.5`
- `m1` = `loss.temporal_hinge.margin=1.0`
- `uniform_w02_t2_temporal_masked_2` = `loss.regularizer.weight=0.2`, `loss.uniformity.t=2`, `loss.uniformity.mode=temporal_masked`, `loss.uniformity.temporal_exclusion=2`

Important implementation detail:

```text
temporal_hinge_loss = max(0, distance(z_t, z_{t+1}) - margin)^2
```

Therefore smaller margins are stronger continuity constraints. `m01` forces
adjacent embeddings to be very close, while `m05` / `m1` are weaker constraints.

### Baseline four-task comparison

| Model | TwoRoom | Cube | PushT | Reacher | Average |
|---|---:|---:|---:|---:|---:|
| `lewm` | 93.0 | 69.2 | 89.4 | 62.2 | 78.45 |
| `swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260415` | 90.8 | 74.0 | 89.8 | 66.0 | 80.15 |

Interpretation:

- SWM is slightly worse on TwoRoom (`90.8` vs `93.0`).
- SWM is better on Cube (`74.0` vs `69.2`) and Reacher (`66.0` vs `62.2`).
- PushT is effectively tied, with a small SWM edge (`89.8` vs `89.4`).
- On this single-seed four-task average, SWM is ahead by `+1.70`.

### Latest no-hinge reruns

| Model | TwoRoom | Cube | PushT | Reacher |
|---|---:|---:|---:|---:|
| `tworoom_lewm` | 93.0 |  |  |  |
| `cube_lewm` |  | 70.0 |  |  |
| `pusht_lewm` |  |  | 83.6 |  |
| `reacher_lewm` |  |  |  | 58.8 |
| `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425` | 91.0 |  |  |  |
| `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425` |  |  | 82.4 |  |
| `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425` |  |  |  | 58.4 |

Notes:

- No `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425` row was provided in this batch.
- The 2026-04-25 no-hinge reruns are lower than the 2026-04-15 SWM baseline on PushT (`89.8 -> 82.4`) and Reacher (`66.0 -> 58.4`), while TwoRoom is similar (`90.8 -> 91.0`).
- This reinforces the need for multi-seed / reproducibility checks before treating a single run as a stable task-level number.

### LeWM temporal hinge runs

| Model | weight | margin | TwoRoom | Cube | PushT | Reacher |
|---|---:|---:|---:|---:|---:|---:|
| `tworroom_lewm_temporal_hinge_w01_m01` | 0.1 | 0.1 | 32.0 |  |  |  |
| `tworoom_lewm_temporal_hinge_w001_m01` | 0.01 | 0.1 | 87.2 |  |  |  |
| `tworoom_lewm_temporal_hinge_w001_m05` | 0.01 | 0.5 | 100.0 |  |  |  |
| `cube_lewm_temporal_hinge_w001_m01` | 0.01 | 0.1 |  | 69.2 |  |  |
| `cube_lewm_temporal_hinge_w001_m05` | 0.01 | 0.5 |  | 70.2 |  |  |
| `pusht_lewm_temporal_hinge_w01_m01` | 0.1 | 0.1 |  |  | 6.2 |  |
| `pusht_lewm_temporal_hinge_w001_m01` | 0.01 | 0.1 |  |  | 13.4 |  |
| `pusht_lewm_temporal_hinge_w001_m05` | 0.01 | 0.5 |  |  | 20.0 |  |
| `reacher_lewm_temporal_hinge_w01_m01` | 0.1 | 0.1 |  |  |  | 43.6 |
| `reacher_lewm_temporal_hinge_w001_m05` | 0.01 | 0.5 |  |  |  | 54.2 |

Interpretation:

- LeWM + fixed temporal hinge is highly unstable across tasks.
- TwoRoom can benefit from weak hinge with a loose margin (`w=0.01,m=0.5` gives `100.0`), but strong hinge collapses performance (`w=0.1,m=0.1` gives `32.0`).
- Cube is almost insensitive in the tested range (`69.2` / `70.2`).
- PushT is severely damaged by all tested LeWM hinge settings (`6.2` to `20.0`).
- Reacher also degrades relative to both `lewm` (`62.2`) and the latest `reacher_lewm` rerun (`58.8`).

### SWM temporal hinge runs

All SWM rows below use the same baseline family:

```text
swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64
```

with fixed temporal hinge added.

| Model | weight | margin | TwoRoom | Cube | PushT | Reacher |
|---|---:|---:|---:|---:|---:|---:|
| `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m01_dim64` | 0.01 | 0.1 | 86.8 |  |  |  |
| `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m05_dim64` | 0.01 | 0.5 | 88.6 |  |  |  |
| `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m1_dim64` | 0.01 | 1.0 | 75.6 |  |  |  |
| `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w01_m01_dim64` | 0.1 | 0.1 |  | 73.8 |  |  |
| `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m01_dim64` | 0.01 | 0.1 |  | 73.4 |  |  |
| `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m05_dim64` | 0.01 | 0.5 |  | 72.0 |  |  |
| `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m1_dim64` | 0.01 | 1.0 |  | 71.4 |  |  |
| `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w01_m01_dim64` | 0.1 | 0.1 |  |  | 72.0 |  |
| `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m01_dim64` | 0.01 | 0.1 |  |  | 76.6 |  |
| `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m05_dim64` | 0.01 | 0.5 |  |  | 76.0 |  |
| `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m1_dim64` | 0.01 | 1.0 |  |  | 81.0 |  |
| `reache_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w01_m01_dim64` | 0.1 | 0.1 |  |  |  | 58.8 |
| `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m01_dim64` | 0.01 | 0.1 |  |  |  | 58.0 |
| `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m05_dim64` | 0.01 | 0.5 |  |  |  | 57.6 |
| `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_temporal_hinge_w001_m1_dim64` | 0.01 | 1.0 |  |  |  | 60.0 |

Interpretation:

- Relative to the best 2026-04-15 no-hinge SWM baseline, fixed temporal hinge does not improve any task in this batch.
- Relative to the 2026-04-25 no-hinge SWM reruns, fixed hinge is still worse on TwoRoom (`91.0` to at best `88.6`) and PushT (`82.4` to at best `81.0`).
- Reacher is the only latest-rerun comparison where a hinge row is slightly higher than the 2026-04-25 no-hinge rerun (`60.0` vs `58.4`), but it remains far below the 2026-04-15 SWM baseline (`66.0`).
- Cube hinge rows stay close to the 2026-04-15 SWM baseline (`74.0`) but do not beat it.
- The larger PushT / Reacher margins recover some performance, which supports the idea that stricter fixed continuity is the damaging component.

### Consolidated interpretation

The comparison separates three related but different mechanisms:

| Mechanism | What it does | Current evidence |
|---|---|---|
| SIGReg in LeWM | globally regularizes the embedding distribution in Euclidean space | strong baseline, but does not explicitly respect temporal adjacency |
| SWM uniformity + temporal mask | pushes representations apart while excluding nearby same-trajectory pairs from that repulsion | stable and currently best 4-task average in the 2026-04-15 baseline |
| fixed temporal hinge | actively pulls every adjacent transition within a fixed margin | helps only in narrow cases and usually hurts, especially PushT / Reacher |

Working hypothesis:

- `temporal_masked` is a low-risk soft temporal prior because it says "do not
  force nearby time steps apart."
- fixed temporal hinge is a stronger prior because it says "force all adjacent
  time steps to be close."
- that stronger prior conflicts with tasks where adjacent time steps can have
  different semantic or control significance, especially contact / manipulation
  tasks such as PushT.
- the next useful continuity prior should be action / transition-aware instead
  of globally applied.

Practical conclusion:

- keep `swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64` as the main SWM
  baseline family, but verify reproducibility because the 2026-04-25 no-hinge
  reruns are lower on PushT / Reacher
- do not continue broad fixed `weight,margin` sweeps for temporal hinge as the
  main path
- if continuity is revisited, make it gated or dynamic:

```text
gate_t * max(0, d(z_t, z_{t+1}) - margin_t)^2
```

where `gate_t` or `margin_t` depends on action, predicted transition scale, or
a task-normalized transition proxy.

## Appendix A. PushT Ablation Experiment Records

### Exp 12: PushT baseline sweep with MLP+BN + uniformity (`all_pairs`)

Setup:
- checkpoint epoch: `10`
- eval budget: `500`
- projector: `MLP + BatchNorm1d`
- pair mode: `all_pairs`

Results:

| Config | PushT eval |
|---|---:|
| `uniform_w=0.1, t=2, dim=192` | 65.6 |
| `uniform_w=0.2, t=2, dim=192` | 74.4 |
| `uniform_w=0.2, t=2, dim=64` | 69.8 |

Interpretation:
- The plain `all_pairs` variant is clearly weaker than the best structured
  variants.
- Even here, `uniform_w=0.2` provides a substantial gain over `0.1`.
- Lowering `dim` to `64` **without** improving the pair-selection policy hurts,
  so smaller latent size alone is not the source of the best result.

### Exp 13: PushT `cross_window` ablation

Setup:
- checkpoint epoch: `10`
- eval budget: `500`
- pair mode: `cross_window`

Results:

| Config | PushT eval |
|---|---:|
| `uniform_w=0.1, t=2, dim=192` | 74.4 |
| `uniform_w=0.2, t=2, dim=192` | 80.2 |
| `uniform_w=0.2, t=2, dim=64` | 82.2 |

Interpretation:
- Restricting uniformity pairs to cross-window pairs improves over the
  `all_pairs` baseline.
- The gain from `uniform_w=0.2` remains.
- `dim=64` is slightly stronger than `dim=192` in this structured setting.

### Exp 14: PushT `temporal_masked` ablation

Setup:
- checkpoint epoch: `10`
- eval budget: `500`
- pair mode: `temporal_masked`

Results:

| Config | PushT eval |
|---|---:|
| `uniform_w=0.1, t=2, temporal_exclusion=1, dim=192` | 71.4 |
| `uniform_w=0.2, t=2, temporal_exclusion=1, dim=192` | 80.0 |
| `uniform_w=0.2, t=1, temporal_exclusion=1, dim=64` | 64.6 |
| `uniform_w=0.2, t=2, temporal_exclusion=1, dim=64` | 81.2 |
| `uniform_w=0.2, t=2, temporal_exclusion=2, dim=64` | **89.8** |
| `uniform_w=0.2, t=2, temporal_exclusion=2, dim=192` | 80.6 |
| `uniform_w=0.3, t=2, temporal_exclusion=2, dim=64` | 85.2 |
| `uniform_w=0.3, t=2, temporal_exclusion=2, dim=192` | 82.0 |
| `uniform_w=0.2, t=2, temporal_exclusion=3, dim=64` | 67.0 |

Interpretation:
- `temporal_masked` is competitive with `cross_window` at
  `temporal_exclusion=1`.
- `t=1` is too weak or poorly calibrated for this branch.
- Increasing `temporal_exclusion` to `2` produces the best result in the whole
  SWM PushT sweep, but specifically at `dim=64`.
- Raising `uniform_w` to `0.3` does not improve the best branch.
- Pushing `temporal_exclusion` to `3` over-shoots and degrades sharply.

## 9. Next Directions

The main single-task ablation ambiguities are now mostly resolved. The next
phase should shift from "which regularizer hyperparameter wins on PushT?" to
"how do we improve action / transition-aware dynamics without losing the
current representation gains?"

### Priority 1: multi-seed confirmation of the current best setting

Recommended protocol:
- rerun `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64` with multiple
  seeds on at least PushT and Cube
- report mean and standard deviation instead of only the best single result

Why this matters:
- The current 4-task picture is promising enough to justify a stronger claim,
  but that claim should be based on stability, not a single lucky run.

### Priority 2: test transition-aware continuity instead of fixed temporal hinge

Recommended direction:
- keep the current SWM baseline geometry
- replace fixed temporal hinge with a gated or dynamic version
- avoid using a single global margin for all adjacent transitions

Candidate objective:

```text
gate_t * max(0, d(z_t, z_{t+1}) - margin_t)^2
```

Possible first versions:
- `gate_t` from task-normalized transition magnitude as a diagnostic-only
  experiment
- `margin_t = base + softplus(h(stopgrad(z_t), a_t))` as a more general
  action-conditioned version
- add `beta * margin_t` if using learned margins, so the model cannot escape by
  making every margin large

Why this matters:
- the latest four-task comparison shows fixed temporal hinge hurts PushT and
  usually does not improve SWM over the no-hinge baseline
- action magnitude and contact can make adjacent time steps represent very
  different semantic transitions
- the useful prior is not "all adjacent states should be close"; it is "small
  transitions should stay close, large transitions may move farther"

Main caveat:
- low-dimensional state / proprio deltas are useful for diagnosing the
  hypothesis, but they are not a universal final signal
- the deployable version should rely on action / latent-conditioned margins or
  another signal available in the visual world-model setting

### Priority 3: improve rollout training directly with multi-step prediction

Recommended direction:
- keep the current one-step latent prediction loss
- add a low-weight multi-step rollout consistency loss over 2-4 future steps
- compute it autoregressively in the same rollout space used at inference

Why this matters:
- Planning quality depends on repeated rollout, not only one-step accuracy.
- The current SWM gains already suggest the representation is good enough to
  make dynamics the next likely bottleneck.
- A small-horizon multi-step loss is the most direct way to align training with
  MPC usage.

Main caveat:
- start with a small weight and short horizon; large-horizon rollout losses can
  destabilize optimization and hurt the strong one-step predictor.

### Priority 4: add a lightweight inverse-dynamics auxiliary loss

Recommended direction:
- predict action from `(z_t, z_{t+1})` or from `(context_t, z_{t+1})`
- use it as an auxiliary loss with a modest weight, not as a primary objective

Why this matters:
- It can encourage the latent space to retain action-relevant information,
  which is especially useful for PushT, Cube, and Reacher.
- It is more likely to help in continuous-control settings where the same visual
  state can support multiple future outcomes depending on action.

Main caveat:
- inverse dynamics is ill-posed when multiple actions induce near-identical
  transitions, so it should remain an auxiliary bias rather than the main
  training signal.

### Priority 5: combine additions only after isolated tests

Recommended sequence:
1. baseline best SWM config with more seeds
2. best SWM + transition-aware continuity
3. best SWM + short-horizon multi-step loss
4. best SWM + inverse-dynamics auxiliary
5. only then try combined versions

Why this matters:
- these additions all target action-conditioned future identifiability from
  different angles, so testing them separately first makes it much easier to
  tell which one actually drives gains or regressions.

### Current recommendation

If compute budget is limited, the cleanest next sequence is:

1. rerun `swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64` with more
   seeds on all four tasks if possible, or at least PushT and Cube
2. test transition-aware continuity against the no-hinge SWM baseline
3. add a small-weight 2-4 step latent rollout loss in the current rollout space
4. if rollout metrics improve without harming TwoRoom, add a low-weight
   inverse-dynamics auxiliary on top

At this point, the regularizer ablation has done its job. The next gains are
more likely to come from action / transition-aware dynamics than from continuing
to search the same uniformity or fixed temporal-hinge hyperparameter grids.

## 10. Paper2 GLC Adequacy Baseline

This section records the first Paper2 adequacy baseline after the Paper1 ACPC
diagnostics. The question was intentionally narrow: before implementing a
SNAP-ACPC or APDC-style objective, test whether a minimal related-work baseline
based on encoder-level clean/noisy latent consistency is already sufficient.

### Implementation

Implemented branch:

- `loss.generic_latent_consistency.enabled`
- paired-view training path for LeWM
- `run_trainer.sh` and `run_trainer_batch.sh` CLI/env passthrough
- self-bounded auxiliary loss, so GLC has no extra tuned loss weight
- clean-anchor BatchNorm freeze fix for the detached clean branch

Training semantics:

- paired-view mode requires `loss.pred.target_view=perturbed`
- normal LeWM prediction loss and SIGReg use the noisy branch
- the clean branch is encoded under `no_grad` and used only as a detached anchor
- BN running stats are frozen only for the clean-anchor encode

Relevant commits:

- `1dc5f09 add generic latent consistency baseline`
- `f68f006 freeze glc anchor batchnorm stats`

### Reacher 0.08 evaluation

Main BN-fix run:

```text
/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-reacher/ckpt/reacher_reacher_lewm_glc_bnfix_noise_0to008_p1
```

Run notes:

- `loss.generic_latent_consistency.enabled=true`
- `image_noise.std_max=0.08`
- `loss.pred.target_view=perturbed`
- `post_train_eval_mode=full`
- `num_eval=150` in `summary.txt`
- eval rows exist for `pixels` and `pixels+goal`; no clean/origin row was
  produced for this run

Closed-loop success comparison:

| Model | Origin | Clean | `pixels_std0.08` | `pixels_goal_std0.08` |
|---|---:|---:|---:|---:|
| `reacher_lewm_20260430` |  | 59.42 | 17.17 | 14.92 |
| `reacher_lewm_noise_0to008_p1` | 81.33 |  | 83.67 | 81.00 |
| `reacher_lewm_glc_noise_0to008_p1` | 58.67 |  | 19.67 | 18.33 |
| `reacher_reacher_lewm_glc_bnfix_noise_0to008_p1` |  |  | 24.00 | 12.00 |
| `reacher_lewm_baseline_unperturbed_target_noise_0to008_p1` | 60.33 |  | 24.33 |  |

BN-fix GLC corruption sweep:

| Condition | Success |
|---|---:|
| `pixels_std0.03` | 37.33 |
| `pixels_std0.05` | 37.33 |
| `pixels_std0.08` | 24.00 |
| `pixels_goal_std0.03` | 48.00 |
| `pixels_goal_std0.05` | 26.67 |
| `pixels_goal_std0.08` | 12.00 |

### Diagnostics

Noise sensitivity shows the failure is not explained by the clean-anchor BN
side effect.

| Model | Std | Angle median | CKA | Noise-to-NN ratio | Risk |
|---|---:|---:|---:|---:|---|
| normal noise 0.08 | 0.08 | 2.51 | 0.998 | 0.014 | low |
| old GLC 0.08 | 0.08 | 80.33 | 0.447 | 12.30 | high |
| BN-fix GLC 0.08 | 0.08 | 80.13 | 0.412 | 12.58 | high |
| target-origin 0.08 | 0.08 | 79.94 | 0.407 | 13.81 | high |

Predictor sensitivity is also much worse than the normal noise-trained branch:

| Model | Std | Target L2 | Rollout T1 L2 | Rollout T8 L2 | T8 angle |
|---|---:|---:|---:|---:|---:|
| normal noise 0.08 | 0.08 | 0.0029 | 0.303 | 0.252 | 1.02 |
| BN-fix GLC 0.08 | 0.08 | 0.0078 | 14.164 | 16.685 | 81.86 |
| target-origin 0.08 | 0.08 |  |  | 12.995 | 68.85 |

### Gate decision

GLC failed the adequacy gate on Reacher.

Interpretation:

- The BN-fix corrected a real implementation side-effect, but it did not change
  the conclusion.
- Generic encoder-level latent consistency behaves like the failed
  target-origin branch, not like ordinary noise training.
- Encoder-level clean/noisy closeness is therefore too weak and too
  mis-targeted for the Paper1 ACPC failure mode.

Decision:

- stop broad GLC sweeps unless a clean/origin eval row is needed for a table
- do not promote GLC as a method contribution
- SNAP-ACPC PR-1A was the next minimal train-side check; its result is recorded
  in the following section

## 11. Paper2 SNAP-ACPC PR-1A Negative Baseline

This section records the first one-step action-conditioned predictive
consistency check after GLC failed. The question was whether matching clean and
noisy **predictions** under the same action context is already enough to close
the gap exposed by Paper1, without returning to the more complex AAAC route.

### Implementation

Implemented branch:

- `loss.snap_acpc.enabled`
- paired clean/noisy LeWM forward shared with the GLC infrastructure
- normal prediction loss and SIGReg remain on the noisy branch
- clean branch is encoded and predicted under `no_grad`
- BatchNorm running stats are frozen for the detached clean-anchor path
- self-bounded auxiliary loss, so SNAP-ACPC has no extra tuned loss weight
- `run_trainer.sh` and `run_trainer_batch.sh` passthrough via
  `loss_snap_acpc_enabled=true`

Relevant commit:

- `cae8bd2 add snap acpc preparation path`

### Reacher 0.08 evaluation

Main run:

```text
/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-reacher/ckpt/reacher_lewm_snap_acpc_noise_0to008_p1
```

Run notes:

- `loss.snap_acpc.enabled=true`
- `loss.generic_latent_consistency.enabled=false`
- `image_noise.std_max=0.08`
- `loss.pred.target_view=perturbed`
- eval rows exist for `pixels` and `pixels+goal`; no clean/origin row was
  produced for this run

Closed-loop success comparison:

| Model | Origin | Clean | `pixels_std0.08` | `pixels_goal_std0.08` |
|---|---:|---:|---:|---:|
| `reacher_lewm_20260430` |  | 59.42 | 17.17 | 14.92 |
| `reacher_lewm_noise_0to008_p1` | 81.33 |  | 83.67 | 81.00 |
| `reacher_lewm_glc_noise_0to008_p1` | 58.67 |  | 19.67 | 18.33 |
| `reacher_reacher_lewm_glc_bnfix_noise_0to008_p1` |  |  | 24.00 | 12.00 |
| `reacher_lewm_baseline_unperturbed_target_noise_0to008_p1` | 60.33 |  | 24.33 |  |
| `reacher_lewm_snap_acpc_noise_0to008_p1` |  |  | 24.67 | 19.67 |

SNAP-ACPC corruption sweep:

| Condition | Success |
|---|---:|
| `pixels_std0.03` | 40.67 |
| `pixels_std0.05` | 30.33 |
| `pixels_std0.08` | 24.67 |
| `pixels_goal_std0.03` | 37.00 |
| `pixels_goal_std0.05` | 29.67 |
| `pixels_goal_std0.08` | 19.67 |

### Diagnostics

SNAP-ACPC improves neither the behavior gate nor the main ACPC diagnostic
relative to ordinary noise training.

Noise sensitivity at std 0.08, all frames:

| Model | Angle median | CKA | Noise-to-NN cosine ratio | Clean rank | Risk |
|---|---:|---:|---:|---:|---|
| normal noise 0.08 | 2.55 | 0.998 | 0.155 | 70.31 | low |
| BN-fix GLC 0.08 | 80.23 | 0.386 | 127.96 | 64.65 | high |
| target-origin 0.08 | 79.70 | 0.383 | 123.93 | 64.48 | high |
| SNAP-ACPC 0.08 | 80.81 | 0.495 | 129.04 | 63.35 | high |

Predictor sensitivity at std 0.08:

| Model | Target L2 | Target-to-NN cosine ratio | Rollout T1 L2 | Rollout T8 L2 | T8 angle |
|---|---:|---:|---:|---:|---:|
| normal noise 0.08 | 0.0029 | 0.000009 | 0.303 | 0.252 | 1.02 |
| BN-fix GLC 0.08 | 0.0078 | 0.000028 | 14.164 | 16.685 | 81.86 |
| target-origin 0.08 | 0.0076 | 0.000027 | 12.996 | 12.995 | 68.85 |
| SNAP-ACPC 0.08 | 0.0062 | 0.000018 | 13.914 | 16.422 | 78.44 |

Task-resolution diagnostics do not show a clean collapse rescue story:

| Model | Transition ratio cosine | Transition ratio L2 | ID probe R2 | Lidar rank |
|---|---:|---:|---:|---:|
| normal noise 0.08 | 0.144 | 0.383 | 0.177 | 49.55 |
| BN-fix GLC 0.08 | 0.136 | 0.373 | 0.159 | 45.94 |
| target-origin 0.08 | 0.136 | 0.370 | 0.160 | 45.84 |
| SNAP-ACPC 0.08 | 0.139 | 0.373 | 0.167 | 45.36 |

### Gate decision

SNAP-ACPC PR-1A failed the Paper2 gate on Reacher.

Interpretation:

- It is only marginally above GLC and target-origin behavior, and far below
  ordinary noise training.
- The encoder clean/noisy geometry remains high-risk: roughly `80.8` degrees at
  std 0.08 all frames, despite CKA being slightly higher than GLC.
- Predictor rollout drift remains in the same failure regime as GLC: T8 L2 is
  `16.42`, while ordinary noise training is `0.252`.
- The failure is not a simple discriminability collapse: transition and
  inverse-dynamics probes are not dramatically worse than GLC, but the visual
  perturbation still transduces into a large predictive rollout shift.

Decision:

- close one-step self-bounded SNAP-ACPC as a negative baseline
- do not broaden this exact PR-1A path into larger sweeps by default
- do not route back to AAAC/APDC as the next Paper2 mainline
- require the next method hypothesis to explain, simplify, or beat ordinary
  noise training under matched noise settings
- before designing another loss, run `loss.paired_view_control.enabled=true` to
  test whether the paired clean/noisy in-forward path is equivalent to ordinary
  `TransformDataset` noise training when no auxiliary loss is added

## 12. Paper2 Paired No-Aux Equivalence Control

This section records the paired-view infrastructure check requested after
GLC and SNAP-ACPC both failed. The purpose was to separate the auxiliary loss
from the training path itself:

- ordinary noise training applies configured image noise through
  `TransformDataset`
- GLC / SNAP-ACPC / paired no-aux bypass `TransformDataset` and apply image
  noise inside `lejepa_forward`
- paired no-aux still encodes a clean detached branch, but adds no auxiliary
  loss

Main run:

```text
/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-reacher/ckpt/reacher_lewm_paired_noaux_noise_0to008_p1
```

### Config verification

The rerun config is the intended one:

```yaml
output_model_name: reacher_lewm_paired_noaux_noise_0to008_p1
image_noise:
  type: gaussian_noise
  std_min: 0.0
  std_max: 0.08
  noise_prob: 1.0
  apply_to_val: false
loss:
  pred:
    space: raw
    target_view: perturbed
  generic_latent_consistency:
    enabled: false
  snap_acpc:
    enabled: false
  paired_view_control:
    enabled: true
```

Other relevant controls are off (`hetero`, `action_gate`,
`adaptive_consistency`), and the run uses the standard Reacher LeWM settings
(`trainer.max_epochs=10`, `loader.batch_size=128`, `seed=3072`,
`wm.history_size=3`, `wm.num_preds=1`). The config therefore fully activates
the intended no-aux paired-view path; no rerun is needed for config reasons.

### Reacher 0.08 evaluation

Closed-loop success comparison:

| Model | `pixels_std0.08` | `pixels_goal_std0.08` | Read |
|---|---:|---:|---|
| normal noise training | 83.67 | 81.00 | strong |
| BN-fix GLC 0.08 | 24.00 | 12.00 | failed |
| SNAP-ACPC 0.08 | 24.67 | 19.67 | failed |
| paired no-aux 0.08 | 24.67 | 14.67 | failed |

Paired no-aux corruption sweep:

| Condition | Success |
|---|---:|
| `pixels_std0.03` | 44.67 |
| `pixels_std0.05` | 37.33 |
| `pixels_std0.08` | 24.67 |
| `pixels_goal_std0.03` | 49.33 |
| `pixels_goal_std0.05` | 23.33 |
| `pixels_goal_std0.08` | 14.67 |

### Diagnostics

Paired no-aux does not reproduce ordinary noise training. At std 0.08 the
clean/noisy geometry remains in the same high-risk regime as GLC/SNAP:

| Model | Rollout T8 L2 | CKA at max std | Geometry read |
|---|---:|---:|---|
| normal noise training | 0.357 | 0.997 | stable |
| BN-fix GLC 0.08 | 17.779 | 0.361 | failed |
| SNAP-ACPC 0.08 | 18.187 | 0.477 | failed |
| paired no-aux 0.08 | 14.875 | 0.433 | failed |

Additional paired no-aux summary:

| Metric | Value |
|---|---:|
| `noise_robust_radius_std` | 0.01594 |
| `noise_angle_slope_deg_per_std` | 753.23 |
| `clean_effective_rank` | 60.56 |
| `transition_resolution_ratio_cos` | 0.1355 |
| `transition_resolution_ratio_l2` | 0.3694 |
| `id_probe_r2` | 0.1650 |
| `predictor_rollout_T8_l2` | 14.875 |
| `latent_robust_radius_z` | 0.04164 |
| `latent_noise_geometry` | ambient |

At std 0.08, all-frame noise sensitivity remains high-risk:

| View | Angle median | CKA | Noise-to-NN cosine ratio | Risk |
|---|---:|---:|---:|---|
| goal | 81.22 | 0.453 | 13.25 | high |
| history | 80.94 | 0.437 | 117.19 | high |
| all | 80.98 | 0.436 | 130.79 | high |

Predictor sensitivity at std 0.08 also remains far from ordinary noise
training: `target_l2=0.006247`, `rollout_T1_l2=13.964`,
`rollout_T8_l2=13.601`, and `T8 angle=67.923`.

### Gate decision

Paired no-aux fails the equivalence gate.

Interpretation:

- GLC/SNAP failure cannot be attributed primarily to their auxiliary losses.
- The failure appears already when paired-view infrastructure is enabled with
  no auxiliary objective.
- The next debug target is therefore the training data path, not another
  consistency loss.

Decision:

- close paired no-aux as an equivalence-control failure
- do not return to AAAC/APDC as the next mainline
- add a narrower `in_forward_noise_control` path that applies the same
  configured noise inside `lejepa_forward` but does **not** encode a clean
  branch and does **not** add any auxiliary loss
- use that noisy-only control to decide whether the issue is
  `TransformDataset` versus in-forward perturbation semantics, or the extra
  clean-anchor paired forward itself
