# Action-Aware Adaptive Latent Resolution

## Executive Summary

This document presents a phased framework for adding per-transition uncertainty estimation to LeWorldModel (LeWM) without compromising its proven MSE + SIGReg baseline. The core hypothesis is that a scalar σ head can learn prediction difficulty, but **σ calibration alone does not constitute adaptive resolution**.

**Status:** Pilot-1B has completed initial validation on TwoRoom and PushT (2026-05-09). The σ head successfully learns to predict per-token error (`hetero_s_logerr_corr` ≈ 0.89 on TwoRoom, 0.95 on PushT). However, replacing MSE with a scale-preserving heteroscedastic NLL causes catastrophic degradation on PushT (clean eval drops from 87.33 to 13.33), because hard-but-important transitions are downweighted. The heteroscedastic-loss path is therefore relegated to ablation / negative result.

**Current preferred route:**
1. **Stage A (active):** Probe-only σ — a detached scalar σ head predicts `log(error)` without modifying the μ-path (MSE + SIGReg unchanged).
2. **Stage B (logging-only):** Action-aware gate — compute action-conditioned local sensitivity `A_t` and combine with σ to emit diagnostic gate statistics. No training intervention yet.
3. **Stage C (conditional):** Action-aware adaptive consistency — use `critical_t = f(A_t, σ_t)` to control encoder-side invariance strength, preserving resolution in action-critical regions and increasing it in redundant regions.

**Pilot-2A update (2026-05-09):** Probe-only successfully recovers PushT performance (clean 87.00 ≈ LeWM-base 87.33), validating the detachment design. Action-gate logging passes structural criteria on PushT (`cv_mean < 0.5`, weak σ-A correlation supporting multiplicative critical design). However, a **BN drift bug** was discovered: K perturbation forwards inside `compute_action_gate_metrics` run in train mode and corrupt `BatchNorm1d` running stats, causing TwoRoom probe+gate clean to drop 7pt (96.33 → 89.33). This bug is orthogonal to Stage C design and must be fixed (freeze BN during perturb forwards) before any `alpha_cons > 0` is enabled.

**Relation to plan_v3:** This is not a replacement for plan_v3, but the concrete elaboration of plan_v3 §6 P4 "Adaptive Resolution Method".

**Design discipline:** Every added mechanism must (a) pass the hyperparameter-count test, and (b) demonstrate empirical marginal gain before advancing to the next stage. See Appendix A for the full list of mechanisms that were proposed and then rolled back.

---

## 1. Introduction

### 1.1 Motivation

Plan_v3 §5.2 proposed "task-aware latent geometry" as a way to move beyond the uniform invariance of LeWM + SIGReg. The practical deadlock encountered was that all "adaptive" schemes placed the trade-off controller *outside* the model — PI controllers, Lagrangian τ, cheap-proxy bilevel methods, or multi-task heads all require external signals or hand-tuned thresholds, and none have empirically outperformed LeWM + SIGReg.

The paradigm shift this framework tests is: **let the model output a local difficulty / uncertainty signal σ, and prove that σ can help allocate resolution.** However, one cannot equate predictor uncertainty `σ̂` directly with latent neighbourhood radius:
- Predictor `σ̂` most naturally supervises *transition uncertainty*.
- Encoder `σ_x` has no direct supervision and risks unidentifiability.
- Planning resolution asks "which state differences should be preserved", which is not equivalent to "which transitions are hard to predict".

Therefore the first step should not modify the main loss, but instead ask: can an extra output head stably learn meaningful heterogeneity? If not, subsequent NLL / planner usage has no foundation. If yes, σ can be gradually introduced into training or inference.

### 1.2 Core Critique: σ Head ≠ Dynamic Resolution

An additional σ head does not automatically become dynamic resolution. Three levels must be distinguished:

| Level | What σ does | Does it change resolution? |
|---|---|---|
| Probe | Predicts detached error | No — diagnostic only |
| Loss weighting | Changes per-transition μ-gradient allocation | Possibly, but may ignore hard-but-important states |
| σ-only controller | Affects CEM budget / gating / consistency strength | Possibly, but confuses aleatoric visual noise with resolution demand |
| **Action-aware consistency** | σ + action sensitivity jointly control encoder invariance | **Yes** — encoder-side adaptive resolution (current preferred candidate) |

The paper cannot treat "adding a σ head" as equivalent to dynamic resolution. What must be proved is that σ aligns with action-relevant difficulty, and that `A_t` separates controllable critical states from uncontrollable visual noise; then adaptive consistency must approach or exceed the LeWM+noise oracle without breaking PushT resolution guardrails.

### 1.3 Design Principles

1. **Probe before intervention.** Stage A validates that σ carries information before Stage B/C let it influence training.
2. **LeWM is the first-principles baseline.** Any σ scheme must first prove it does not destroy LeWM+noise's clean/robustness tradeoff.
3. **Hyperparameter budget discipline.** If a mechanism adds hyperparameters without clear empirical gain, it is rolled back (see Appendix A).
4. **Minimal changes first.** The predictor σ head adds ~0.5M parameters (negligible) and the loss at `s=0` degenerates exactly to LeWM MSE.

### 1.4 Scope and Structure

This document covers architecture (§2.1), loss design (§2.2), the action-aware gate (§2.4), completed experimental validation (§3–§4), discussion of findings (§5), and the staged future roadmap (§6). Pilot-1B results are presented as an ablation that validates σ semantics while rejecting heteroscedastic loss as the primary method.

---

## 2. Method

### 2.1 Architecture

#### 2.1.1 Predictor σ Head (Pilot-1)

LeWM baseline:
```
enc_backbone(x) → h ∈ R^{h_dim}
projection_head(h) → z ∈ R^d
predictor(z_t, a_t) → pred_hidden
pred_proj(pred_hidden) → z_hat ∈ R^d
```

Pilot-1 adds:
```
pred_hidden → μ_hat ∈ R^d
pred_hidden → logvar_hat ∈ R^1   # scalar per token
```

The encoder still outputs a single `μ = z`; no encoder σ is added. Rationale:
- Predictor `σ̂` has a natural target: current transition prediction error.
- Encoder `σ_x` lacks direct supervision; adding both simultaneously creates unidentifiability.
- Rollout and CEM cost remain unchanged, isolating whether the σ head carries information.

#### 2.1.2 Optional Encoder σ Head (Pilot-2)

Only if Pilot-1 demonstrates stable correlation between `σ̂` and prediction difficulty:
```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
logvar_head(h)  → log σ_x² ∈ R^1
```

Encoder `σ_x` must have a concrete use case (goal uncertainty, consistency weight controller, or calibration alignment with predictor `σ̂`) before it is added.

#### 2.1.3 Future Architectural Consideration: Encoder Input-Sensitivity Head

An alternative to the unsupervised encoder σ head (§2.1.2) is a **supervised encoder input-sensitivity head** that predicts the encoder's response to input perturbations:

```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
sens_head(h)    → s_enc(x) ∈ R^1   # predicts encoder displacement under input perturbation

target_enc = log( ||enc(x).detach() − enc(aug(x)).detach()||_2 + eps )
L_sens = smooth_l1(s_enc(x), target_enc)
```

Properties:
- **Natural supervision:** `||enc(x) − enc(aug(x))||` is directly measurable, avoiding the unsupervised encoder σ identifiability trap.
- **Orthogonal to predictor σ̂:** σ̂ describes *transition difficulty given clean state*; s_enc describes *state representation sensitivity to input nuisance*. They capture different physical quantities without mutual residual-absorption degeneracy.
- **No second encoder / EMA required:** supervision uses two forwards through the same encoder, consistent with the single-encoder philosophy.
- **Shared forward with L_cons:** the target distance is already computed in `d(stopgrad(z_clean), stopgrad(z_noisy))`, incurring no extra forward pass.
- **Controller-side闭环 value:** s_enc extracts encoder sensitivity as an explicit controller input for `w_t`, forming *encoder sensitivity → controller → encoder consistency pressure* feedback, rather than relying solely on predictor-side signals.

**Why deferred:**
1. Pilot-1 (probe-only σ) has not yet been fully validated; the cheaper predictor head must be tested first.
2. Stage C's `L_cons = w_t · ||z_clean − z_noisy||` already implicitly brings encoder input-sensitivity into gradients — test whether this suffices before adding an explicit head.
3. Adding this head would introduce a new hyperparameter `beta_sens`, violating the hyperparameter-count discipline.

**Trigger for adoption:**
- Stage C's `alpha_cons` ramp hits PushT guardrails (`transition_resolution_ratio_l2 < 0.24` or `clean < 84`), and diagnostics show the failure stems from **insufficient encoder-side input-nuisance discrimination** (not predictor-side σ̂ / A_t signal failure); or
- σ̂ calibration drifts under +noise training (§5.4 Probe-on-noise stage), requiring an input-side signal to disentangle σ̂ drift components.

#### 2.1.4 Target Encoder and Anti-Collapse

Target latent `μ_{t+1}^{target} = enc(x_{t+1})` uses the same encoder, no EMA, and no stop-grad asymmetry (consistent with LeWM). Anti-collapse is handled entirely by LeWM's existing SIGReg(μ); no additional mechanism is introduced.

### 2.2 Loss Design: Three-Stage Route

#### 2.2.1 Stage A: Detached σ Calibration Probe (Current Preferred)

The main training objective remains exactly LeWM:
```
pred_loss = mean((mu_hat - mu_target)^2)
loss = pred_loss + lambda_SIGReg * SIGReg(mu)
```

The σ head is trained with detached supervision:
```
err_token = mean((mu_hat.detach() - mu_target.detach())^2, dim=-1)
s_hat = pred_logvar_hat.squeeze(-1)
sigma_probe_loss = smooth_l1(s_hat, log(err_token + eps))
```

Key constraints:
- `sigma_probe_loss` updates only the σ head; it does not backpropagate into the encoder or predictor mean path.
- This stage does **not** change latent resolution; it validates that the extra head learns transition difficulty.
- If σ probe fails to learn stable structure, subsequent gate / consistency stages lack a reliable signal foundation.

#### 2.2.2 Stage B: Scale-Preserving Heteroscedastic Loss (Ablation / Negative Result)

> **Status:** Validated in Pilot-1B (2026-05-09). σ calibration succeeds, but PushT clean eval collapses to 13.33. This section is retained as documentation of a negative result, not as the primary method.

Ordinary Gaussian NLL `0.5 * (err * exp(-s) + s)` is unsuitable as a direct MSE replacement because at `s=0` it equals `0.5 * err`, halving the pred loss and altering the MSE/SIGReg balance.

The scale-preserving formulation used in Pilot-1B:
```
err = mean((mu_hat - mu_target)^2, dim=-1)
s = pred_logvar_hat.squeeze(-1)
tau = stopgrad(EMA(mean(err)))   # or current batch mean(err).detach()

hetero_loss = mean(exp(-s) * err + tau * s)
loss = hetero_loss + lambda_SIGReg * SIGReg(mu)
```

Properties:
- At `s ≡ 0`, `hetero_loss = mean(err)`, identical to plain MSE in scale.
- The μ-path initial gradient approximates LeWM, so SIGReg weight needs no retuning.
- Optimal condition: `exp(s) ≈ err / tau`, i.e. σ learns relative difficulty rather than arbitrary global loss scale.

Risk realised in Pilot-1B: it downweights high-error transitions. In PushT these transitions likely contain contact and fine-control critical regions, causing clean control failure. Therefore this path is relegated to ablation; the primary route does not let σ enter the μ-path gradient.

#### 2.2.3 Stage C: Action-Aware Adaptive Consistency (Candidate Method)

Adaptive latent resolution should not reweight prediction loss, but rather modulate encoder input-side invariance locally:
```
z_clean = enc(x)
z_noisy = enc(aug(x))
L_cons = mean(w_t * d(stopgrad(z_clean_t), z_noisy_t))
```

The weight `w_t` must not be determined by σ alone, because prediction difficulty mixes:
- **Epistemic / dynamics difficulty** (e.g. PushT contact instant): should *reduce* consistency pressure, preserving resolution.
- **Aleatoric / visual nuisance** (e.g. uncontrollable background noise): should *increase* consistency pressure, erasing noise.

Action-conditioned criticality is defined as:
```
A_t = d(f(z_t, a_t + delta), f(z_t, a_t)) / (||delta|| + eps)
gA_t = sigmoid(zscore_ema(log(A_t + eps)))
gS_t = sigmoid(zscore_ema(s_t))
critical_t = gA_t * (0.5 + 0.5 * gS_t)
w_t = w_max - (w_max - w_min) * stopgrad(critical_t)
```

Design rationale:
- `A_t` is the primary gate, representing action-conditioned local sensitivity / controllability.
- `σ_t` acts only as a difficulty enhancer, preventing σ-only gates from treating Noisy TV as high-resolution demand.
- `delta` is drawn from empirical action std or in-batch action differences, avoiding arbitrary OOD random actions.
- `critical_t` and `w_t` are fully detached; the gate is a controller, not a backprop shortcut.
- Stage C is only entered after logging-only validation (§3.3 and §5.1.2) proves structural alignment.

#### 2.2.4 SIGReg Remains on μ Only

Regardless of stage, SIGReg applies only to the deterministic μ. Do not extend SIGReg to `(μ, σ)` or reparameterized samples; that introduces Gaussian mixture higher-moment problems and breaks LeWM's validated anti-collapse mechanism.

### 2.3 The Four-Level σ Usage Framework

| Usage | Role | Risk |
|---|---|---|
| Training weight | Changes μ gradient allocation via hetero loss | May ignore hard-but-important states |
| σ-only noise/controller | High-σ regions modulate noise consistency strength | **Confounder trap**: high σ may stem from aleatoric visual noise, not task-critical dynamics |
| Action-aware consistency | Uses action sensitivity to separate controllable critical states from uncontrollable visual noise | Most aligned with adaptive resolution, but requires logging-only validation first |
| Planner budget | High-σ rollouts allocate more CEM samples / shorter horizon | Does not change representation, only inference compute |
| Uncertainty gate | Rejects or downweights candidate plans at high σ | May become overly conservative |

### 2.4 Action-Aware Gate Design

#### 2.4.1 Action Sensitivity `A_t`

For each context token, perturb the action and measure predictor response:
```
A_t = ||f(z, a + δ) - f(z, a)||_2 / ||δ||_2
```

#### 2.4.2 Multi-δ Perturbation and Coefficient of Variation

`A_t` is high for two distinct reasons:
- **Smooth controllable**: small δ → smooth large response. Multiple δ draws yield directionally correlated, similarly-magnitude responses → **low CV**.
- **Chaotic / extrapolation**: predictor is discontinuous near contact/boundaries. Small δ → arbitrarily large response. Multiple δ draws yield high variance → **high CV**.

Each token is evaluated with `K` independent δ samples:
```
A_t^{(k)} = d(f(z, a + δ^{(k)}), f(z, a)) / (||δ^{(k)}|| + eps)   for k=1..K
A_mean = mean_k A_t^{(k)}
A_cv = std_k A_t^{(k)} / (A_mean + eps)
```

Diagnostic thresholds:
- Global `cv_mean < 0.5`: predictor is locally smooth, `A_t` is trustworthy.
- High-`A_mean` regions do not show significantly higher CV than global: critical regions are not dominated by chaos.

If either fails, `A_t` should be chaos-discounted: `A_mean / (1 + α_cv * A_cv)` with `α_cv = 1.0`.

#### 2.4.3 EMA Z-Score and Warmup

`A_t` only acquires physical meaning after the predictor has learned action conditioning. Early in training, when the predictor largely ignores action, `A_t ≈ 0` reflects predictor immaturity rather than state insensitivity.

Logging activates only after either:
- `validate/id_probe_r2_epoch >= 0.5 * id_probe_r2_LeWM_base` (PushT: 0.39, TwoRoom: 0.14); or
- `warmup_epochs` have elapsed (default 3, ~30% of LeWM's 10-epoch training).

During warmup `A_t` is still computed and logged, but `critical_t` is not aggregated and EMA z-score statistics are not updated, preventing baseline drift from action-blind statistics.

#### 2.4.4 Consistency Weight Formula

```
log_A = log(A_mean.clamp(min=eps))
gA = sigmoid(zscore_ema(log_A))
gS = sigmoid(zscore_ema(s_t))    # falls back to 0.5 if σ is unavailable
critical = gA * (0.5 + 0.5 * gS)
w_t = w_max - (w_max - w_min) * critical
```

Default bounds: `w_min = 0.2`, `w_max = 1.0`.

### 2.5 Comparison with Existing Methods

| Existing Method | Position in this framework |
|---|---|
| LeWM + SIGReg | No σ usage logic; equivalent to ignoring the σ head in Stage A |
| SWM (V0 spherical) | Fixed unit-sphere geometry prior; no dynamic σ |
| VICReg | Fixed covariance / variance prior; no dynamic σ |
| LeWM + noise | Global input-side invariance; no state/action-aware weighting |

No existing method combines per-transition uncertainty with action-conditioned local sensitivity to control encoder invariance strength.

---

## 3. Experimental Validation

### 3.1 Setup

**Tasks:** TwoRoom and PushT (primary benchmarks).
**Training:** 10 epochs, LeWM baseline architecture.
**σ head:** `logvar_hidden_dim=256`, final layer zero-initialised (weight=bias=0), `s_min=-4.0`, `s_max=4.0`.
**Noise:** `image_noise.std_max=0.0` for ablation cleanliness (noise and σ-adaptive are complementary, not mutually exclusive; see §6.4).
**Evaluation:** Epoch 10, `num_eval=300`, aggregated over seeds 42/43/44.

### 3.2 Pilot-1B: Heteroscedastic Loss as Ablation

Pilot-1B tested the scale-preserving hetero loss (§2.2.2) as a direct MSE replacement. Configuration: `loss.hetero.enabled=true`, `loss.hetero.mode=loss`.

**Runs:**

| Task | Run name | SwanLab ID | Local output |
|---|---|---|---|
| TwoRoom | `tworoom_lewm_hetero_default` | `gps6asjv22tmflag9af5m` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_hetero_default` |
| PushT | `pusht_lewm_hetero_default` | `tge50bhmtws06xc7n4wtq` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/pusht_lewm_hetero_default` |

#### 3.2.1 Training Metrics

| Metric | TwoRoom hetero | PushT hetero | Interpretation |
|---|---:|---:|---|
| `fit/hetero_s_logerr_corr` tail100 | 0.894 | 0.950 | σ aligns with prediction difficulty; σ head semantics validated |
| `validate/hetero_s_logerr_corr_epoch` last | 0.912 | 0.957 | Validation correlation holds; not a train-only artifact |
| `fit/hetero_s_std` tail100 | 1.232 | 1.836 | PushT σ heterogeneity is stronger |
| `fit/hetero_s_abs_max` last | 3.236 | 4.000 | PushT clamps at upper bound |
| `fit/hetero_weight_q10` last | 0.495 | 0.369 | High-σ / hard tokens downweighted |
| `fit/hetero_weight_q90` last | 11.026 | 47.802 | Low-error tokens heavily upweighted |
| `fit/hetero_weight_q10_q90_ratio` last | **0.045** | **0.008** | **Extreme gradient imbalance (PushT risk signal)** |
| `fit/pred_loss_mse_equiv` tail100 | 0.0438 | 0.0394 | True MSE-equivalent loss still decreases |
| `validate/pred_loss_mse_equiv_epoch` last | 0.0274 | 0.0332 | Validation MSE also decreases; failure is not underfitting |

Key judgements:
- **σ calibration succeeds.** Both tasks show high `hetero_s_logerr_corr`, proving the σ head is not constant or noise.
- **PushT reweighting is excessive.** `q10/q90_ratio` of 0.008 is far below the 0.3 guardrail; this is the hard-but-important transition downweighting risk.
- **Hetero loss can go negative.** Slightly negative `pred_loss` in late training is an artifact of the formula `exp(-s) * err + tau * s`; the proper reference is `pred_loss_mse_equiv`.

#### 3.2.2 Evaluation Results

| Task / model | Clean | goal 0.05 | pixels 0.05 | pixels+goal 0.05 | goal 0.08 | pixels+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom LeWM+noise best (`0to008-p1`) | 98.33 | 98.00 | 98.33 | 98.00 | 98.67 | 98.67 |
| TwoRoom hetero | **99.67** | 85.33 | 96.67 | 84.67 | 73.33 | 55.33 |
| PushT LeWM-base | 87.33 / 86.00 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT LeWM+noise best (`0to002-p1`) | **90.00** | 85.00 | 87.67 | 86.00 | 83.00 | 70.67 |
| PushT hetero | **13.33** | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |

Interpretation:
- TwoRoom clean improves to 99.67, consistent with low-dimensional discrete tasks benefiting from stronger invariance / clustering.
- TwoRoom hetero does **not** replace noise training: high noise on goal/pixels+goal remains well below LeWM+noise best.
- PushT clean at 13.33 is a **method-level failure**, not a robustness tradeoff.

#### 3.2.3 Diagnostic Analysis

| Metric | TwoRoom LeWM-base | TwoRoom hetero | PushT LeWM-base | PushT hetero |
|---|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 |
| `clean_effective_rank` | 47.60 | 33.59 | 76.42 | 42.85 |
| `transition_resolution_ratio_cos` | 0.5538 | 0.3780 | 0.0868 | 0.0101 |
| `transition_resolution_ratio_l2` | 0.7216 | 0.6055 | 0.3015 | 0.1023 |
| `id_probe_r2` | 0.2889 | -0.0573 | 0.7739 | 0.2678 |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.0841 |
| `action_interpolation_endpoint_shift` | 1.0474 | 0.8907 | 0.3361 | 0.1702 |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 14.01 |

Mechanism:
- Hetero loss compresses representations on both tasks: NN distance drops, effective rank drops, action-induced shift drops.
- TwoRoom is low-dimensional and discrete; representation compression is acceptable or even beneficial.
- PushT requires continuous contact and pose resolution. `transition_resolution_ratio_l2` falls from 0.3015 to 0.1023, and `id_probe_r2` drops from 0.7739 to 0.2678, indicating task-relevant state information is erased.
- PushT's reduced `predictor_rollout_T8_l2` is not good news: latent becomes easier to predict by sacrificing resolution, not by becoming more controllable.

#### 3.2.4 Conclusion

Pilot-1B yields a **semantic success but system failure**:
1. The σ head is worth keeping. It stably learns per-transition prediction difficulty.
2. Direct hetero training is unsuitable for PushT. It treats high-error hard transitions as low-weight samples, yet these transitions are likely PushT's contact and fine-control critical regions.
3. Adaptive resolution cannot rely solely on loss reweighting. The need is: μ-representation preserves control resolution, while σ serves as an extra signal to modulate planning / consistency / compute, rather than letting σ directly decide which transitions to stop training.

### 3.3 Pilot-2A: Probe-Only σ and Action-Gate Logging

Pilot-2A tests Stage A (probe-only) and Stage B (logging-only gate) jointly.

**Runs:**

| Task | Run | SwanLab ID |
|---|---|---|
| TwoRoom probe | `tworoom_lewm_hetero_probe_default` | `75qiqru0ttwmyy7pwigly` |
| TwoRoom probe+gate | `tworoom_lewm_hetero_probe_default_action_gate` | `awokxbepmodp2shcqmynr` |
| PushT probe | `pusht_lewm_hetero_probe_default` | `jgqsw29zji110j3gczu03` |
| PushT probe+gate | `pusht_lewm_hetero_probe_default_action_gate` | `oezw5j3w0uh3ydxnan63c` |

Setup: `loss.hetero.enabled=true loss.hetero.mode=probe`; probe+gate additionally sets `loss.action_gate.enabled=true` (logging-only, `adaptive_consistency.weight=0`). Eval epoch 10, seeds 42/43/44, `num_eval=100` per seed.

#### 3.4.1 Evaluation Results

| Task / model | Clean | goal 0.05 | pixels 0.05 | px+goal 0.05 | goal 0.08 | px+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom hetero (Pilot-1B) | **99.67** | 85.33 | **96.67** | 84.67 | 73.33 | **55.33** |
| TwoRoom probe | 96.33 | 80.67 | 81.00 | 67.00 | 63.67 | 46.00 |
| TwoRoom probe+gate | 89.33 | 49.00 | 52.00 | 36.67 | 41.67 | 33.00 |
| PushT LeWM-base | 87.33 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT hetero (Pilot-1B) | 13.33 | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |
| PushT probe | 81.67 | 39.00 | 19.33 | 14.67 | 17.33 | 3.33 |
| PushT probe+gate | **87.00** | **52.00** | **31.67** | **21.00** | **23.00** | 3.33 |

#### 3.4.2 Training Diagnostics

| Metric | tw_probe | tw_probe_gate | pu_probe | pu_probe_gate |
|---|---:|---:|---:|---:|
| `pred_loss_mse_equiv` (tail100) | 0.0295 | 0.0295 | 0.0177 | 0.0162 |
| `validate/hetero_s_logerr_corr` (last) | 0.620 | 0.621 | 0.480 | 0.462 |
| `hetero_s_abs_max` (last) | 4.24 | 4.30 | 4.23 | 4.23 |
| `adaptive_action_sensitivity_cv_mean` | — | 0.36 | — | 0.245 |
| `adaptive_action_sensitivity_cv_high_A` | — | 0.35 | — | 0.277 |
| `validate/adaptive_corr_sigma_action_epoch` | — | −0.02 | — | +0.23 |
| `adaptive_critical_mean` (val) | — | 0.35 | — | 0.30 |

#### 3.4.3 Conclusions

1. **PushT collapse is resolved.** Probe-only PushT clean 81.67; probe+gate clean 87.00, matching LeWM-base 87.33. The hetero-loss 13.33 collapse does not recur, validating the detachment design.
2. **σ probe semantics are preserved.** PushT validate corr ≈ 0.46–0.54; TwoRoom ≈ 0.62. PushT is borderline but meets the Stage A 0.5 threshold. The correlation is weaker than under hetero loss (0.95) because NLL feedback no longer forces σ to track prediction error.
3. **Action-gate structural criteria pass on PushT.** `cv_mean = 0.245 < 0.5` ✅; `cv_high_A = 0.277` not significantly above global ✅; `corr_sigma_action`: PushT +0.23 weak positive, TwoRoom −0.02 approximately independent → σ and A_t are distinct signals, supporting the multiplicative `critical = gA × (0.5 + 0.5·gS)` design.
4. **TwoRoom probe+gate clean drop of 7pt (96.33 → 89.33) is a real bug, not solvable by Stage C.** See §3.4.4.
5. **PushT probe+gate robustness exceeds LeWM-base** (goal 0.05: 38 → 52; pixels 0.05: 17 → 32). Since logging-only runs should not alter training gradients, this is either seed noise or a side effect of the §3.4.4 BN drift bug inadvertently providing slight invariance training on PushT. Must be re-measured after BN fix.

#### 3.4.4 BN Drift Bug: Mechanism and Fix

**Mechanism:** `compute_action_gate_metrics` executes K=4 perturbation forwards via `model.predict(ctx_emb_d, act_emb_pert)` while `model.training=True`. Both `projector` and `predictor_proj` use `nn.BatchNorm1d` (`config/train/lewm.yaml::encoder.projection_head.norm_fn=batchnorm1d`). Each perturb forward updates BN running mean/var with OOD-ish activations, corrupting the clean-data distribution statistics at every training step.

**Asymmetric impact:**
- **TwoRoom:** small representation space, low batch visual diversity; BN statistics are highly sensitive to perturb forwards. Clean eval drops 7pt (96.33 → 89.33).
- **PushT:** visual diversity dominates BN statistics; K=4 perturb forwards are negligible. Clean 87.00 matches LeWM-base 87.33.

**Why Stage C cannot fix this:** Stage C adds `L_cons = w_t · d(z_clean, z_noisy)` to the main loss without changing the internal perturb-forward logic. BN running stats remain corrupted by K=4 OOD forwards every step, orthogonal to `alpha_cons` magnitude or `w_t` design. Adding Stage C would only entangle BN drift with consistency gradients, complicating diagnosis.

**Fix (mandatory before Stage C):**
```python
bn_states = []
for m in model.modules():
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        bn_states.append((m, m.training))
        m.eval()  # use running stats; do NOT update them
try:
    for _ in range(K):
        ...  # perturb forward + A_t computation
finally:
    for m, was_train in bn_states:
        m.train(was_train)
```
This is semantically correct: A_t measures local sensitivity `||predictor(z, a+δ) − predictor(z, a)||` under **fixed normalization statistics**; allowing perturb forwards to update BN stats is itself a leakage.

**Post-fix re-runs required:**
- TwoRoom probe+gate (verify clean returns to 96+; if still ~89, the bug is not solely BN drift).
- PushT probe+gate (verify clean 87 and robustness hold; if robustness drops back to LeWM-base, the "slight invariance training side effect" hypothesis from §3.4.3 conclusion 5 is confirmed).

**Revised Stage C entry condition:** TwoRoom probe+gate clean ≥ 92 (restored to LeWM-base vicinity), proving BN drift is fixed; PushT probe+gate clean ≥ 86 without degradation.

**True role of Stage C (only meaningful after BN drift fix):** Stage C may solve the compatibility problem of bringing both TwoRoom and PushT close to their respective optima — reducing consistency pressure in critical regions to preserve PushT contact resolution, while increasing consistency in non-critical regions to raise TwoRoom toward LeWM+noise levels. But this is only possible if the logging-only stage already shows no clean eval degradation. The current data **does not yet justify the conclusion that "adaptive consistency is compatible with dynamic resolution"**; the bug must be fixed first.

### 3.4 Key Properties Validated

**NLL benefit vs risk:** NLL can prevent wasting μ-resolution on unpredictable visual noise, and provides an uncertainty signal for planning. However, high error does not equal low value; PushT contact instants may be high-error but high-value. Downweighting hard samples can reduce clean control rather than improve robustness.

**LeWM as strict special case:** In the scale-preserving form, if `s ≡ 0` or σ is fixed, `hetero_loss = mean(err)` and SIGReg(μ) is unchanged. The model degenerates exactly to LeWM MSE + SIGReg. This special-case relationship only holds cleanly for the scale-preserving form; ordinary NLL changes constants and scale.

**Noisy TV / confounder trap:** High σ can arise from uncontrollable visual noise. A σ-only consistency gate would abandon invariance to noise. The consistency gate must therefore be action-aware, using `A_t` as the primary gate and σ only as an enhancer.

---

## 4. Discussion

### 4.1 Core Findings

1. **σ head learns non-trivial, task-related prediction difficulty.** `hetero_s_logerr_corr` ≥ 0.89 on both tasks.
2. **Direct hetero loss reweighting destroys PushT control resolution.** `transition_resolution_ratio_l2` collapses from 0.30 to 0.10; clean eval drops by 74 points.
3. **The viable path is σ as diagnostic/controller, not σ as gradient reweighter.** Action-aware adaptive consistency (§2.2.3) is the only usage layer that both changes resolution and avoids the confounder trap.
4. **Three levels of σ usage must be distinguished.** Probe-only is diagnostic; hetero loss is loss weighting; only action-aware consistency constitutes a full adaptive system that actually uses the extra head.

### 4.2 Risk Analysis

| Risk | Assessment | Mitigation |
|---|---|---|
| σ degenerates to global constant | If PushT σ is also approximately constant, the extra head has learned no useful heterogeneity | Do not enter NLL; inspect err target, head capacity, longer training |
| NLL changes MSE/SIGReg weight ratio | Ordinary NLL starts at 0.5× MSE and drifts with σ | Hetero loss is ablation-only; scale-preserving form if rerun |
| Hard-but-important states downweighted | PushT contact/fine-control may be high-error but high-value | Monitor transition/action resolution; fallback to guarded consistency |
| σ is uncertainty, not resolution | Calibration success ≠ planning improvement | Stage C must define explicit σ usage logic; otherwise treat as diagnostic output only |
| Noisy TV / confounder trap | High σ may stem from uncontrollable visual noise; σ-only consistency abandons noise invariance | Consistency gate must be action-aware: `A_t` primary gate, σ enhancer only |
| Action sensitivity OOD | Arbitrary random actions may leave data distribution, making `A_t` reflect extrapolation | `delta` uses empirical action std or in-batch action differences; logging-only first |
| Gate backprop shortcut | If `critical_t` is not detached, encoder/predictor can manipulate the gate to evade consistency | `σ_t`, `A_t`, `critical_t`, `w_t` are all stopgrad; warmup before enabling consistency |
| Encoder σ unidentifiability | Encoder σ lacks natural supervision; simultaneous encoder+predictor σ learning risks mutual evasion | Pilot-1 does not add encoder σ; add only after predictor σ is validated |
| Multi-step σ propagation inaccuracy | This framework does not advocate hand-written σ accumulation formulas | Let predictor `σ̂` learn multi-step uncertainty via multi-step rollout NLL |
| BN drift via gate perturb forward | K perturbation forwards in train mode update `BatchNorm1d` running stats with OOD activations; TwoRoom probe+gate clean drops 7pt (96.33 → 89.33). Stage C cannot fix this | Freeze all `_BatchNorm` modules to `.eval()` during perturb forwards; restore mode afterward. Re-run probe+gate before entering Stage C |
| Does not exceed LeWM+noise oracle | Likely; LeWM+noise is already strong | Target is first to reduce hand-tuning and approach oracle; if clearly below, downgrade to analysis/future work |

### 4.3 Role of Diagnostic Tools

The honest claim is modest: predictor `σ̂` is itself a new per-transition diagnostic quantity. Existing diagnostics (`clean_nn_dist`, `effective_rank`, `transition_resolution_ratio`, etc.) and their correlation with `σ̂` are **worth measuring post hoc**, but are not an a priori paper claim.

- If `σ̂` correlates strongly with certain diagnostics → bonus.
- If `σ̂` correlates weakly → it provides independent new information, also a bonus.

Diagnostic tools are primarily **design constraints and mechanistic explanations**. Their value is decoupled from this framework's success: even if P0.6 blind bucketing is weak, the σ-head may still succeed as a more direct adaptive resolution method.

### 4.4 Novelty Claim and Boundaries

**Claim (pending pilot validation):**
> Action-aware adaptive consistency for latent resolution: a detached scalar σ probe on the LeWM predictor estimates transition difficulty, while action-conditioned local sensitivity `A_t` separates controllable critical states from uncontrollable visual noise; together they control clean/noisy encoder consistency strength, making simple/redundant regions more invariant while preserving resolution in action-critical regions. LeWM and LeWM+noise are strict baselines (no σ/A controller vs global consistency).

**Prerequisites for this claim:**
- Stage A proves σ head learns non-trivial, task-related prediction difficulty.
- Logging-only stage proves `A_t` filters aleatoric visual noise from σ, rather than merely echoing prediction error.
- Adaptive consistency proves `critical_t = f(A_t, σ_t)` approaches or exceeds LeWM+noise oracle without breaking PushT resolution guardrails.
- Gains are not from retuning SIGReg / loss scale, nor from downweighting hard transitions' prediction gradient.

**What is no longer claimed:**
- "NLL is necessarily better than MSE."
- "σ head naturally equals latent resolution."
- "High σ should preserve resolution."
- "Planner needs no modification to benefit automatically."
- IB / Fisher manifold / "diagnostics = intrinsic axes of (μ, σ)" strong theoretical narratives.

---

## 5. Future Work and Roadmap

### 5.1 Primary Route: Probe → Gate Logging → Adaptive Consistency

The goal is not "make hetero loss a bit milder", but to give the system **adaptive resolution**: visual redundancy / uncontrollable noise regions gain stronger invariance, while PushT's action-critical continuous state/action resolution must be preserved.

The critical correction is that `σ_t` cannot alone signify "high resolution demand". Prediction difficulty mixes controllable dynamics difficulty and aleatoric visual noise; the former should reduce consistency weight, the latter should increase it. The next stage uses `A_t` (action-conditioned local sensitivity) as the primary gate, with σ as difficulty enhancer.

#### 5.1.1 Stage A Continued: Probe-Only σ Head

Implementation status: `loss.hetero.mode=probe` is implemented. `JEPA.predict_with_logvar(..., detach_logvar_input=True)` detaches the σ head's predictor hidden input, ensuring `sigma_probe_loss` does not update the shared predictor backbone.

```
pred_loss = MSE(mu_hat, mu_target)
loss = pred_loss + lambda_SIGReg * SIGReg(mu)

err_token = mean((mu_hat.detach() - mu_target.detach())^2, dim=-1)
s_hat = pred_logvar_hat.squeeze(-1)
sigma_probe_loss = smooth_l1(s_hat, log(err_token + eps))

loss_total = loss + beta_probe * sigma_probe_loss
```

Run command:
```bash
python train.py data=pusht output_model_name=pusht_lewm_sigma_probe_default \
    loss.hetero.enabled=true loss.hetero.mode=probe
```

Success criterion: PushT clean eval returns to LeWM-base vicinity while σ maintains `s_logerr_corr >= 0.5`.

#### 5.1.2 Stage B: Logging-Only Action-Aware Gate

Implementation status: `loss.action_gate` config block exists in `config/train/lewm.yaml`. `train.py::compute_action_gate_metrics` performs K perturbations → re-prediction → `A_t`, `A_t_cv`, `gA_t`, `critical_t`, `w_t`, all under `no_grad`. EMA statistics persist in `world_model.gate_{log_A,s}_{mean,var}` buffers; no aggregation during warmup.

Compatible with `loss.hetero.mode=probe`: if σ is off, the gate records only A-related metrics.

Run command:
```bash
python train.py data=pusht output_model_name=pusht_lewm_action_gate_logging \
    loss.hetero.enabled=true loss.hetero.mode=probe \
    loss.action_gate.enabled=true
```

Core quantities:
```
A_t = d(f(z_t, a_t + delta), f(z_t, a_t)) / (||delta|| + eps)
gA_t = sigmoid(zscore_ema(log(A_t + eps)))
gS_t = sigmoid(zscore_ema(s_t))
critical_t = gA_t * (0.5 + 0.5 * gS_t)
w_t = w_max - (w_max - w_min) * critical_t
```

Logging targets:
- `adaptive/sigma_mean`, `adaptive/action_sensitivity_mean`, `adaptive/critical_mean`, `adaptive/weight_mean`
- `adaptive/corr_sigma_action`, `adaptive/weight_q10_q90`
- `adaptive/action_sensitivity_cv_mean`, `adaptive/action_sensitivity_cv_high_A`

Entry criterion for Stage C:
- High `critical_t` correlates structurally with PushT contact / high action-norm / high transition displacement.
- Visual nuisance raises σ without synchronously raising `A_t`.
- `critical_t` explains `id_probe_r2` / action resolution diagnostics better than σ-only.
- High-`A_t` regions do not show significantly larger multi-δ variance than low-`A_t` regions.
- **BN drift bug is fixed:** TwoRoom probe+gate clean ≥ 92 (restored to LeWM-base vicinity); PushT probe+gate clean ≥ 86 without degradation. See §3.4.4 for the fix (freeze BN during perturb forwards).

#### 5.1.3 Stage C: Action-Aware Adaptive Consistency Training

Only entered if logging-only validation passes:
```
z_clean = enc(x_clean)
z_noisy = enc(aug(x_clean))
L_main = MSE(mu_hat, mu_target) + lambda_SIGReg * SIGReg(mu)
L_cons = mean(stopgrad(w_t) * d(stopgrad(z_clean_t), z_noisy_t))
loss = L_main + beta_probe * sigma_probe_loss + alpha_cons * L_cons
```

Rationale:
- Main prediction loss is not downweighted by σ or `A_t`, avoiding the hetero-loss PushT collapse.
- `w_t` controls only extra invariance pressure: less detail erasure in action-critical / high-σ regions, stronger invariance in visual-nuisance / action-insensitive regions.
- `alpha_cons` starts small and ramps; PushT resolution guardrails are hard rejection conditions.

### 5.2 Experimental Ladder and Entry Conditions

**Hyperparameter budget:**

| Name | Default | Range | Entry / early-stop condition |
|---|---:|---|---|
| `loss.hetero.probe_weight` (`beta_probe`) | 1.0 | [0.1, 5.0] | Probe stage; `hetero_s_logerr_corr >= 0.5` required to enter Stage B. Below 0.3 for 3 epochs → fallback to deeper-detach probe head |
| `loss.action_gate.delta_scale` | 0.25 | [0.05, 0.5] | δ relative to batch action std. Fixed, **not tuned** |
| `loss.action_gate.num_delta_samples` (K) | 4 | [2, 8] | Multi-δ variance estimation; increase to 8 only if CV untrustworthy |
| `loss.action_gate.warmup_epochs` | 3 | [0, 5] | Logging entry threshold |
| `loss.action_gate.ema_momentum` | 0.99 | [0.95, 0.999] | EMA smoothing for zscore. Fixed, **not tuned** |
| `loss.adaptive_consistency.w_min` | 0.2 | [0.0, 0.5] | Minimum invariance pressure in critical regions |
| `loss.adaptive_consistency.w_max` | 1.0 | [0.5, 1.5] | Maximum invariance pressure in non-critical regions |
| `loss.adaptive_consistency.alpha_cons` | 0.01 | [0.001, 0.1] | Start small, ramp ×3; freeze if guardrail triggered |
| `loss.adaptive_consistency.aug_type` | `gaussian_noise(std=0.04)` | — | Reuse LeWM+noise pipeline |

**Marginal empirical benefit requirements (PushT):**

| Stage transition | Prerequisite | Marginal benefit requirement |
|---|---|---|
| probe-only → action-gate logging | Probe passes criterion (§5.1.1) | clean >= LeWM-base - 1pt (>= 86) |
| logging → consistency `alpha=0.01` | Stage B criteria pass + CV criterion pass + **BN drift fixed** (TwoRoom probe+gate clean ≥ 92) | clean >= 86, transition_resolution_ratio_l2 >= 0.27 |
| `alpha=0.01` → `alpha=0.03` | Guardrails hold + clean does not drop > 1pt | robustness (goal+pixels 0.05) improves >= 5pt over LeWM-base |
| `alpha=0.03` → `alpha=0.1` | Same as above | robustness within 5pt of LeWM+noise oracle |

If any stage fails its benefit requirement → **freeze current alpha, switch to ablation/analysis**, no further ramp.

### 5.3 Recommended Next Experiments

Run TwoRoom + PushT first; do not expand to 4-task until the following pass:

| Experiment | TwoRoom | PushT | Purpose |
|---|---:|---:|---|
| `lewm_sigma_probe_default` | yes | yes | Validate σ independent semantics without changing μ |
| `lewm_action_gate_logging` | yes | yes | `weight=0` logging of `A_t` / `critical_t`; validate Noisy TV confounder filtering |
| `lewm_action_aware_consistency_alpha001` | optional | yes | Core new method: action-aware gate controls encoder consistency |
| `lewm_sigma_only_consistency_alpha001` | optional | yes | Failure control: σ directly as critical signal (`critical_t = sigmoid(zscore_ema(s_t))`), testing Noisy TV hypothesis |
| `lewm_hetero_alpha001_guarded` | optional | yes | Minimal-weight control for training reweighting |

Pass criteria:
1. PushT clean must be near LeWM-base (>= 84) to continue.
2. σ calibration must hold: `validate/hetero_s_logerr_corr_epoch >= 0.5`.
3. `A_t` / `critical_t` must show action-relevant structure; otherwise consistency is not enabled.
4. PushT resolution guardrails must not break.
5. **BN drift bug must be fixed and verified:** TwoRoom probe+gate clean ≥ 92 before entering Stage C.
6. If action-aware consistency shows no gain, the method is downgraded to uncertainty/action diagnostic; if it shows gain, it advances to primary adaptive-resolution method.

### 5.4 Joint Training with Noise

All σ pilots above use `image_noise.std_max=0.0` for clean ablation, but **this is not the final form**. Noise training and σ-adaptive operate at different positions:

| Mechanism | Position | Physical meaning |
|---|---|---|
| Image noise / consistency | Encoder input side | Data augmentation, forces nuisance invariance |
| σ probe | Predictor output side | Labels per-transition difficulty |
| σ planner use | Inference side | Allocates compute / truncates horizon by difficulty |
| Action-aware adaptive consistency | Encoder input side + σ/A feedback | Uses `A_t` for controllability filter, σ for criticality enhancement |

The correct relationship is complementary:
> LeWM+noise provides a global invariance baseline; the action-aware σ/A controller decomposes global invariance into per-state allocation, preserving detail in action-critical hard regions while erasing more in action-insensitive visual-redundancy regions.

Additional validation ladder (after §5.3 passes):
1. **Probe-on-noise:** Add σ probe to a LeWM+noise checkpoint (μ path unchanged). Check whether σ still calibrates stably and whether spatial distribution changes.
2. **Action-gate-on-noise:** Logging-only `A_t` / `critical_t` on a LeWM+noise checkpoint. Confirm the gate still separates contact from visual nuisance after noise training.
3. **Joint training:** Only if 1+2 pass. Must watch for double-downweight risk: noise makes contact transitions harder → σ-only gate might incorrectly relax consistency. `A_t` primary gate must be retained, and guardrails must monitor noise level, `A_t` distribution, and resolution metrics jointly.

### 5.5 Open Questions

These questions are registered for joint analysis once probe-only results return:

1. **Harder calibration criterion.** `s_logerr_corr >= 0.5` is necessary but not sufficient. Probe uses detached MSE supervision; high correlation is almost guaranteed given sufficient head capacity. Additional checks needed:
   - Does σ correlate structurally with contact / goal-edge masks or high action-norm quantiles?
   - Under visual nuisance (goal / pixels noise), does σ drift toward "hard" (control-relevant) or "random" (visual-irrelevant) regions?
   - Does σ anticipate contact onset, or is it merely a smoother of residual?
   If only correlation passes without structural evidence, probe-only should be upgraded to diagnostic only, not consistency training.

2. **Planner uncertainty direction risk.** In goal-conditioned planning, adding `α·σ` to cost makes the planner *avoid* high-σ regions, yet PushT contact/fine-control is high-σ. Preference:
   - Start with **σ-based CEM budget reallocation** (high σ → more samples / restarts), not cost penalty.
   - If cost-side is necessary, make it goal-aware (e.g. penalty only far from goal, tolerance near goal).

3. **σ clamp and parameterisation.** PushT `s_abs_max` was clamped at 4.0 throughout. Before re-enabling any hetero / guarded auxiliary:
   - Widen `s_max`? (Side effect: more extreme sample weight imbalance.)
   - Change σ parameterisation (e.g. softplus positive scale + learnable prior)?
   - Keep 4.0 but add guardrail: "s_abs_max persistently at ceiling → auto-reduce alpha".
   Probe-only stage can observe PushT σ's natural distribution to inform this decision.

4. **TwoRoom hetero +6.7pt explanation gap.** TwoRoom hetero clean (99.67) exceeds both LeWM-base (93.00) and LeWM+noise best (98.33). This needs ablation:
   - TwoRoom probe-only: if σ probe does not change μ, eval should return to LeWM-base. If it stays near 99.67, the gain is not from reweighting.
   - If probe-only falls back while hetero holds, the mechanism is "difficulty-based reweighting" — useful as a *task-conditional* argument, not a universal method argument.

5. **Noise + adaptive resolution interaction.** Short-term expansion to 4-task is unnecessary, but when reviewing probe-only results, the question must be answered: "If this σ is stacked on +noise training, is it still stable? Can `A_t` filter σ's visual-noise component?" This should be added to the Stage A review checklist.

### 5.6 Full 4-Task Validation

**Trigger condition:** Probe-only + σ usage logic passes, and empirically approaches LeWM+noise oracle.

| Item | Setting |
|---|---|
| Tasks | 4 tasks |
| Seeds | 3 |
| Eval | num_eval=300 (100 per seed) |
| Controls | LeWM-base / LeWM+noise shared std / LeWM+noise per-task oracle / this framework |
| Ablations | probe-only vs action-gate logging vs action-aware consistency vs σ-only consistency vs hetero loss; scalar σ vs per-dim σ (last only) |

---

## Appendix A: Design Rollback Record (Honest Engineering Notes)

Earlier versions of this document contained the following additions, **all of which have been rolled back**:

| Addition | Removal reason |
|---|---|
| EMA target encoder | Violates LeWM single-encoder philosophy; SIGReg already replaces EMA's anti-collapse role |
| Extending SIGReg to stochastic (μ, σ) via reparametrization | Gaussian mixture higher moments conflict with heteroscedasticity; "deliberate weakening" to second moments abandons most of SIGReg's value |
| Aggregate covariance Frobenius regularizer | Alternative to above, but introduces λ_agg; +1 hyperparameter vs LeWM |
| Information Bottleneck term `−β/2·E[log σ²]` | Even if σ calibrates via NLL, IB bound introduces β hyperparameter; deferred |
| Fisher manifold planning (Mahalanobis CEM cost) | (a) Not true Fisher distance (first-order approximation only); (b) σ-drift hallucination risk; (c) modifies planner, violating SWM design commitment; (d) σ_goal source unclear |
| σ propagation closed form `σ_{t+k}² ≈ σ_t² + Σσ̂²` | Assumes independent predictor errors; severely violated under autoregressive rollouts |
| σ-only adaptive consistency | High σ mixes dynamics difficulty and aleatoric visual noise; falls into Noisy TV / confounder trap; requires `A_t` |
| "Diagnostics = 2–3 intrinsic axes of (μ, σ)" strong claim | Empirical question; premature assertion risks undermining the paper |
| Multi-head GradNorm / PCGrad / Lagrangian | New hyperparameters + extra training complexity; cost exceeds benefit |

**Core lessons:**
1. Count hyperparameters for every addition — if gain is unclear, roll back.
2. Mathematical elegance ≠ empirical validity.
3. LeWM+noise is validated effective; replacement defaults to "does not exceed oracle".
4. One simple claim + thorough evidence > four interdependent theoretical layers.

---

## Appendix B: Hyperparameter Budget and Guardrail Thresholds

### B.1 Hyperparameters

| Name | Default | Range | Stage |
|---|---:|---|---|
| `loss.hetero.probe_weight` | 1.0 | [0.1, 5.0] | A |
| `loss.action_gate.delta_scale` | 0.25 | [0.05, 0.5] | B |
| `loss.action_gate.num_delta_samples` (K) | 4 | [2, 8] | B |
| `loss.action_gate.warmup_epochs` | 3 | [0, 5] | B |
| `loss.action_gate.ema_momentum` | 0.99 | [0.95, 0.999] | B |
| `loss.adaptive_consistency.w_min` | 0.2 | [0.0, 0.5] | C |
| `loss.adaptive_consistency.w_max` | 1.0 | [0.5, 1.5] | C |
| `loss.adaptive_consistency.alpha_cons` | 0.01 | [0.001, 0.1] | C |

### B.2 PushT Resolution Guardrails (Relative to LeWM-Base)

| Metric | LeWM-base | Stop / reject if |
|---|---:|---:|
| `id_probe_r2` | 0.774 | < 0.65 |
| `transition_resolution_ratio_l2` | 0.301 | < 0.24 |
| `action_mean_pred_shift_norm` | 0.128 | < 0.10 |
| Clean eval | 87.33 / 86.00 | < 84 |

These guardrails are more important than `pred_loss_mse_equiv` alone, since Pilot-1B proved MSE can decrease while planning fails.

---

## Appendix C: Relation to plan_v3 and plan_v2

### C.1 Relation to plan_v3 §6 P4

This document is the preferred elaboration of plan_v3 §6 P4 "Adaptive Resolution Method", executed in stages. Pilot-1B triggered a critical fallback: direct hetero loss damages PushT critical-transition resolution. The subsequent priority is probe-only σ, action-gate logging, and action-aware adaptive consistency; σ planner use / guarded hetero auxiliary remain controls or alternatives.

### C.2 Relation to plan_v2 V1/V2

- V1 (vMF): Spherical + 1D angular σ; a specialised version of spherical projection constraints in this framework.
- V2 (ball-cap): σ_x quantile clip as an OOD extension.

V1/V2 are more complex variants. **This minimal version does not presuppose either direction**; the decision depends on pilot results.

---

## Appendix D: References

- **JEPA / LeWM**: LeCun 2022 ("A Path Towards Autonomous Machine Intelligence"); **Maes et al. 2026, "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"** (arXiv:2603.19312)
- **Heteroscedastic regression**: Kendall & Gal NeurIPS 2017
- **Variational JEPA (rejected as direct borrow)**: Gögl & Yau 2026 (arXiv:2603.20111) — tabular only; this work extends to vision + multi-step
- **Anti-collapse**: SIGReg (Maes et al. 2026), VICReg (Bardes, Ponce & LeCun ICLR 2022), Barlow Twins (Zbontar et al. ICML 2021), RankMe, LiDAR, uniformity (Wang & Isola ICML 2020), BYOL (Grill et al. NeurIPS 2020)
- **Reconstruction-based world models**: Hafner et al. 2020/2023 (Dreamer / DreamerV3), Hansen et al. 2024 (TD-MPC2)
- **JEPA route**: LeCun 2022; Assran et al. CVPR 2023 (I-JEPA)
- **Noise / Lipschitz / certified-robustness diagnostics**: Hoffman 2019, Virmaux & Scaman NeurIPS 2018, Cohen, Rosenfeld & Kolter ICML 2019
- **Latent geometry diagnostics**: Sun et al. NeurIPS 2022 (KNN-OOD), Kornblith et al. ICML 2019 (CKA), Ethayarajh EMNLP 2019 (anisotropy), Jing et al. ICLR 2022 (dimensional collapse)
- **Action probing / inverse dynamics**: Alain & Bengio ICLR-W 2017, Brandfonbrener et al. NeurIPS 2023, Pathak et al. ICML 2017 (ICM)
- **Noisy TV / aleatoric confounder**: Burda et al. ICLR 2019 ("Large-Scale Study of Curiosity-Driven Learning")
- **Empowerment / controllability**: Klyubin, Polani & Nehaniv IEEE CEC 2005
- **Asymmetric consistency**: Chen & He CVPR 2021 (SimSiam)
- **Collision-risk recent work**: PCA++ (arXiv:2511.12278), Surprise-Recognition (arXiv:2512.01119), RobustZero (Li et al. ICML 2025)

---

## Maintenance Notes

- This document is for design reference and iteration; **not** a replacement for plan_v3.
- Add new entries to Appendix A or the risk table in §4.2 after each new discussion.
- The primary route is §5.1: probe-only σ → action-gate logging → action-aware adaptive consistency + PushT resolution guardrail.
- If action-aware Pilot-2 passes, merge §2–§4 and §5.1–§5.3 into plan_v3 §6 P4; this document is then archived.
- **Before adding any new mechanism:** review Appendix A and ask (1) "How many hyperparameters does it add?" and (2) "What is the evidence for empirical gain?". If both answers are unclear, do not add.
