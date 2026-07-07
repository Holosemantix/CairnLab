# Pushforward-Calibrated Latent Perturbations for Robust JEPA World-Model Control

**Status:** technical-report draft for Paper2 method development  
**Branch:** `ag/dev`  
**Date:** 2026-07-07  
**Intended role:** a structured bridge from Paper1 diagnostics and ACPC-Flow audits to a possible Paper2 method. This document is written so that large parts can later be moved into a paper draft.  
**Current claim level:** hypothesis + theory + completed diagnostic/audit evidence + next audit/training plan. No new method claim should be made until the planned pushforward replay and training MVE pass their gates.

---

## Abstract

Latent predictive world models such as LeWM avoid pixel reconstruction and plan in compact learned representations, but Paper1 shows that this alone does not guarantee closed-loop visual robustness. Under matched Gaussian observation noise, no-noise LeWM checkpoints can suffer large control failures, while input-side Gaussian noise training recovers broad task-dependent robustness plateaus. ACPC-family diagnostics localize this recovery to reduced same-action clean/noisy predictive drift and improved candidate-ranking stability, but several direct method conversions have failed: generic encoder consistency, one-step predictive consistency, paired clean/noisy no-aux controls, and planner-side robust CEM reranking are insufficient or weak.

The ACPC-Flow audits provide a sharper constraint. A core TwoRoom audit rejected post-projector clean-only local latent-noise repair: even weak pixel Gaussian corruption at `emb` produced shifts far outside the synthetic latent-noise tube. A later v2 four-level audit over `encoder_feat`, `emb`, `predictor_hidden`, and `pred_emb` found no-go fixed-checkpoint local repair across Gaussian, blur, and resize stressors; it also showed that candidate rankings are genuinely affected by corruption and that non-oracle time-conditioned FM calibration is not separable. However, the four-task origin-vs-noise v2 aligned summary shows that ordinary input-noise training **does** reshape the `P/R` path and planner-facing ranks: for matched Gaussian 0.08, ATR, SMPR, `amp_P`, wrong-neighbor rates, top-1 flips, and top-k overlaps all move strongly in the robust direction across TwoRoom, Reacher, PushT, and Cube.

This report proposes a new method hypothesis: pixel-space corruptions do not induce small isotropic Gaussian shifts in JEPA latent spaces. Instead, they induce layer-, task-, token-, and corruption-family-dependent **pushforward shifts** through the encoder, projector, predictor backbone, and predictor projection. A latent perturbation method should therefore match the measured pushforward geometry rather than inject scalar isotropic noise. We formalize coverage, non-crossing, and planner-facing relevance conditions for latent perturbation distributions; derive local covariance and rank-flip criteria; integrate the completed ACPC-Flow audit results as empirical constraints; and propose a staged audit-to-method pipeline. The next deliverable is not a frozen adapter, but a **Pushforward Noise Geometry and Replay Audit** that compares isotropic, diagonal, low-rank-plus-diagonal, mean-shifted, mixture, and token/state-conditioned latent perturbation families against measured pixel-induced representation shifts and the completed ACPC-Flow v2 metrics.

If successful, the contribution would be a theory-backed, diagnosis-guided latent perturbation method for robust JEPA world-model control. If the audit fails, it will still explain why naive latent-noise ACPC-Flow fails and will provide a principled no-go boundary for latent perturbation training.

---

## 1. Introduction

### 1.1 Background

Joint-Embedding Predictive Architectures (JEPAs) predict future representations rather than reconstructing pixels. This reduces pressure to model high-frequency visual detail and makes them attractive for world-model control. Paper1 reframes visual robustness for such models as **action-conditioned predictive consistency (ACPC)**: for a clean history and a visually perturbed history describing the same underlying state, the world model should produce consistent predicted futures under the same action sequence, while task- or action-relevant distinctions must remain separated.

Paper1's current role is diagnostic. It shows that input-side Gaussian noise training can recover matched-Gaussian observation-noise robustness, but the recovery is a broad, task-dependent plateau rather than a universal optimum. Diagnostics such as ATR/ACPC, PCC, CRA, and MAF are useful for mechanism localization and plateau triage, not as a standalone oracle.

### 1.2 Why a new method route is needed

The direct routes from diagnostic to method are now bounded by negative evidence:

- **Generic encoder-level latent consistency (GLC)** fails: clean/noisy encoder closeness does not recover the predictive plateau.
- **One-step SNAP-ACPC** fails: matching one-step noisy predictions to detached clean predictions does not reproduce ordinary noise training.
- **Paired no-aux control** fails: the paired clean/noisy in-forward path itself does not behave like ordinary `TransformDataset` noise training.
- **Planner-side Robust CEM** is weak/no-go: rank-vote and robust inner-loop scoring do not clearly beat compute-matched CEM and do not transfer beyond the small TwoRoom signal.
- **Frozen ACPC-Flow local repair** is no-go: post-hoc synthetic local-noise repair on a trained origin checkpoint fails coverage and non-crossing gates across audited representation levels.
- **Heteroscedastic loss reweighting** is unsafe: hard transitions can be action-relevant, so downweighting prediction difficulty can erase important control information.

These failures suggest that the next method should not simply add another consistency loss or final-stage planner selector. It should first answer a more basic geometric question:

> What directions, scales, and covariance structures do pixel perturbations actually induce in the representations consumed by the world model and planner, and which of those shifts are safely trainable?

### 1.3 What ACPC-Flow has already established

The updated `paper1/ACPC_FLOW_CODEX.md` should be treated as a completed empirical constraint on this report, not as an unrelated plan.

**Core64 audit.** On TwoRoom `baseline_seed3073`, post-projector `emb` synthetic local noise did not cover measured pixel-induced shifts:

```text
emb gaussian 0.03: no_go, ratio_q90 ~= 2.266, coverage@0.04/q95 = 0, wrong_nn ~= 0.137
emb gaussian 0.08: no_go, ratio_q90 ~= 7.752, coverage@0.04/q95 = 0, wrong_nn ~= 0.836
emb blur k7:       no_go, ratio_q90 ~= 17.871, coverage@0.04/q95 = 0, wrong_nn ~= 0.918
emb resize 0.5:    no_go, ratio_q90 ~= 16.304, coverage@0.04/q95 = 0, wrong_nn ~= 0.941
```

For `emb` Gaussian 0.03, the pixel shift `delta_q90` was about `10.27`, while synthetic std=0.04 had radius q95 about `0.70` and synthetic std=0.12 had radius q95 about `2.10`. This is not a small hyperparameter miss.

**V2 fixed-checkpoint audit.** The v2 audit used TwoRoom `baseline_seed3073`, 128 sequences, candidate-rank metrics, amplification metrics, and `t` calibration. It audited:

```text
encoder_feat
emb
predictor_hidden
pred_emb
```

It found no-go fixed-checkpoint synthetic repair for all audited stressors and levels:

```text
gaussian 0.03 / 0.05 / 0.08: no_go at encoder_feat, emb, predictor_hidden, pred_emb
blur k7: no_go at all four levels
resize 0.5: no_go at all four levels
```

Important readouts:

```text
amp_P_q90 is high across stressors (~5.1--6.0), so encoder projector P strongly amplifies pixel-induced shifts.
wrong_nn is already non-trivial at weak Gaussian 0.03 and severe for stronger stressors.
candidate top1 flip rises from ~=0.20 at Gaussian 0.03 to ~=0.66 at Gaussian 0.08, ~=0.83 for blur, and ~=0.79 for resize.
t calibration is not separable for any audited stressor/level.
```

**Four-task origin-vs-noise v2 aligned summary.** The same v2 audit was then used to compare origin and ordinary input-noise-trained seed3073 checkpoints on TwoRoom, Reacher, PushT, and Cube. The crucial distinction is:

```text
strict_gate_label / no_go = can we train frozen synthetic local repair here?
continuous movement      = did ordinary training reshape the P/R path and planner-facing ranks?
```

For matched Gaussian 0.08, ordinary noise training reshapes the P/R path in a way that aligns with behavior and Paper1 diagnostics:

| Task | eval px0.08 | ATR q90 | SMPR | amp_P q90 | emb wrongNN | pred wrongNN | top1 flip | top5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | 68.8 -> 97.1 | 1.509 -> 0.111 | 0.339 -> 0.989 | 5.192 -> 1.358 | 0.828 -> 0.003 | 0.906 -> 0.042 | 0.664 -> 0.016 | 0.534 -> 0.977 |
| Reacher | 18.2 -> 81.6 | 2.628 -> 0.082 | 0.733 -> 0.997 | 2.926 -> 1.031 | 0.930 -> 0.003 | 0.927 -> 0.000 | 0.797 -> 0.008 | 0.516 -> 0.980 |
| PushT | 7.2 -> 85.8 | 3.580 -> 0.247 | 0.439 -> 1.000 | 1.545 -> 1.140 | 0.732 -> 0.000 | 0.711 -> 0.003 | 0.867 -> 0.023 | 0.364 -> 0.970 |
| Cube | 43.1 -> 62.6 | 2.320 -> 0.100 | 0.453 -> 1.000 | 2.012 -> 1.207 | 0.932 -> 0.000 | 0.932 -> 0.000 | 0.859 -> 0.000 | 0.413 -> 0.988 |

Held-out blur/resize is task-dependent rather than globally negative. TwoRoom and Reacher align strongly with eval and diagnostics, PushT is positive especially on resize, and Cube remains the current boundary case.

**Interpretation for this report.** The live method hypothesis is no longer frozen post-hoc ACPC-Flow repair. It is **training-time P/R distribution migration**: can we identify the structured representation shifts that ordinary pixel-noise training implicitly induces, and reproduce or sharpen them with pushforward-calibrated latent perturbations plus discriminability guards?

### 1.4 Thesis of this report

The key hypothesis is:

> Pixel-space perturbations induce structured pushforward distributions in latent/predictive spaces. Isotropic scalar latent noise fails because it under-covers dominant pixel-induced directions while over-covering irrelevant dimensions and crossing task/action-distinct neighborhoods. A useful latent perturbation method should be pushforward-calibrated and guarded by ACPC, SMPR/non-crossing, and candidate-rank diagnostics.

This thesis changes the design target from:

```text
clean latent + Gaussian noise -> clean latent
```

to:

```text
measured pixel-pushforward geometry -> feasible structured latent perturbation -> training-time P/R distribution migration -> predictive plateau repair
```

### 1.5 Intended contributions if the route succeeds

1. **Theory.** Pixel corruptions induce anisotropic and possibly low-rank pushforward distributions in JEPA world-model representation chains. Isotropic latent noise has a coverage-vs-crossing conflict.
2. **Diagnostic audit.** A frozen, no-training audit measures pushforward covariance, layer amplification, non-crossing risk, and planner-facing candidate-rank sensitivity, extending the completed ACPC-Flow v2 metrics.
3. **Method.** A pushforward-calibrated structured latent perturbation family, combined with predictive plateau objectives and discriminability guards, aimed at training-time P/R distribution migration rather than frozen adapters.
4. **Evidence.** Structured latent perturbations mimic measured pixel-induced feature-shift directions better than isotropic noise and improve closed-loop robustness under matched Gaussian stress, with bounded unseen-stressor checks.
5. **Negative clarity.** Existing negative controls show why naive encoder invariance, one-step prediction matching, frozen ACPC-Flow repair, and planner-side reranking are insufficient.

---

## 2. Related Work

This section is intentionally close to Paper1's related-work framing so that the two documents can later be merged.

### 2.1 JEPA and latent predictive world models

I-JEPA learns image representations by predicting latent target-block representations from context blocks, avoiding pixel reconstruction and hand-crafted data augmentations. V-JEPA extends feature prediction to video and shows that latent feature prediction can produce broadly useful frozen video representations. LeWM applies the JEPA idea to pixel-based control: it trains an end-to-end latent world model with a prediction loss plus SIGReg-style latent regularization, then performs latent planning. Recent work on latent video prediction further studies robustness and discriminability of video latent predictors, showing that latent-prediction models can have distinctive robustness profiles under corruption and occlusion.

These works motivate latent prediction as a representation-learning and world-modeling principle, but they do not by themselves define control robustness under visual perturbation. Paper1 fills that gap by measuring whether clean and corrupted observations yield consistent action-conditioned predicted futures.

### 2.2 Robust visual control and world-model planning

Visual-control robustness is often improved by input-side augmentation. DrQ-v2 is a strong model-free baseline that uses data augmentation for visual continuous control. DreamerV3 and TD-MPC2 show that learned world models and latent planning can scale across many control tasks, but their robustness stories are not the same as ACPC-style paired clean/corrupted predictive consistency in a fixed JEPA checkpoint.

Paper1 observes that ordinary input-side Gaussian noise training is surprisingly strong for LeWM under matched Gaussian observation noise, but it is a coarse scalar pressure: the best training-noise level is task- and seed-dependent, and transfer to blur/resize is bounded. ACPC-Flow v2 adds that this training also reshapes projector/predictor paths and candidate ranks. This report asks whether we can explain and partially reproduce that robustness using a latent perturbation distribution calibrated to measured pixel-pushforward geometry.

### 2.3 Feature-space augmentation and latent perturbation

Feature-space augmentation is not new. Manifold Mixup regularizes hidden representations by interpolating hidden states and encourages smoother decision boundaries. A-FAN augments intermediate visual features adversarially and normalizes feature statistics for visual recognition tasks. Domain-generalization work such as XDomainMix augments feature components to improve invariance under domain shift. These approaches establish that perturbing intermediate representations can be useful and sometimes more efficient than input-space augmentation.

However, robust JEPA world-model control has a different constraint. The perturbation cannot merely improve classification invariance. It must preserve same-state action-conditioned predictive basins while keeping action-, transition-, and cost-relevant distinctions separable. This report therefore treats feature perturbation design as a **pushforward coverage plus non-crossing plus planner-facing relevance** problem.

### 2.4 Control-relevant abstractions and discriminability

Bisimulation, DeepMDP-style latent models, value-aware modeling, and decoder-free latent MPC all emphasize that representations should preserve downstream transition, reward, value, or planning consequences. Paper1's ACPC formulation is consistent with this view: same-state visual perturbations should be equivalent after the same action intervention, but action-relevant differences should not collapse.

The structured latent perturbation proposal inherits this guard. It is not enough to match clean and corrupted features; a candidate method must also preserve SMPR-like semantic margins, transition resolution, inverse-dynamics probes, effective rank, and candidate-ranking behavior.

### 2.5 Gap addressed by this report

Existing feature augmentation work usually chooses perturbation families directly in feature space, often as isotropic noise, adversarial directions, mixup directions, or domain-style components. Existing visual-control augmentation work often perturbs pixels. The missing middle layer is:

> measure how pixel perturbations are pushed forward through the world model, then design latent perturbations that cover those measured directions without crossing control-relevant basins.

The completed ACPC-Flow audits show why this gap matters: scalar synthetic local latent noise fails fixed-checkpoint repair, but ordinary input-noise training still moves the P/R path and planner-facing ranks in the robust direction. The method opportunity is therefore to make that migration explicit, structured, and testable.

---

## 3. Problem Setup

### 3.1 Representation chain

Let the model process observation histories and actions through the chain:

```text
pixels -> encoder H -> encoder projector P -> emb z -> predictor backbone B -> pred_proj R -> pred_emb y
```

We consider four representation levels:

```text
r_H(o) = encoder_feat h = H(o)
r_Z(o) = emb z = P(H(o))
r_U(o, a) = predictor_hidden u = B(z, a)
r_Y(o, a) = pred_emb y = R(B(z, a))
```

Here `o` denotes the visual history and `a` denotes the action context or candidate rollout prefix when predictor-facing levels are used.

### 3.2 Pixel perturbation pushforward

For a pixel perturbation family `tau`, define the level-`l` pushforward delta:

```text
Delta_l,tau(o) = r_l(tau(o)) - r_l(o)
```

The empirical pushforward set is:

```text
S_l(T) = { Delta_l,tau(o_i) : o_i ~ D, tau in T }
```

where `D` is the dataset distribution and `T` is a perturbation family or family set.

### 3.3 Candidate latent perturbation distribution

A latent perturbation method chooses a distribution:

```text
Q_l(o) over eps_l
```

and trains the model so that perturbing level `l` by `eps_l` does not change action-conditioned predictions in task-irrelevant ways:

```text
r_l_source = r_l(o) + eps_l,    eps_l ~ Q_l(o)
```

The central design question is whether `Q_l` can safely approximate the measured pushforward deltas `Delta_l,tau` and induce the same kind of P/R path migration that ordinary input-noise training produced in the ACPC-Flow v2 aligned summary.

---

## 4. Theory and Feasibility Criteria

### 4.1 Three necessary criteria

A candidate latent perturbation family is feasible only if it passes three criteria.

#### Criterion 1: coverage

`Q_l` must cover measured pixel-induced shifts:

```text
Delta_l,tau(o) lies in a high-probability tube of Q_l(o)
```

An empirical Mahalanobis version is:

```text
M_i = (Delta_i - mu_Q)^T Sigma_Q^{-1} (Delta_i - mu_Q)
coverage_q95 = Pr_i[ M_i <= empirical_q95(Q_l) ]
```

Coverage should be evaluated by severity and perturbation family, not only by aggregate norm. The ACPC-Flow core64 audit is a concrete failure case: `emb` Gaussian 0.03 had pixel `delta_q90 ~= 10.27`, while synthetic std=0.12 still had radius q95 only about `2.10`.

#### Criterion 2: non-crossing / basin safety

A perturbation distribution is invalid if it covers pixel shifts only by crossing into task- or action-distinct neighborhoods. Define a clean-neighborhood margin `m_l(o)` using state labels, semantic state proxies, or clean kNN distances. Then require:

```text
Pr_{eps ~ Q_l(o)}[ r_l(o) + eps crosses a task-distinct neighborhood ] <= crossing_max
```

Operational proxies:

```text
wrong_nn_rate
closer_to_wrong_than_pair_rate
same_label_topk_rate
SMPR_after_noise
```

The ACPC-Flow v2 fixed-checkpoint audit shows why this criterion is non-negotiable: even weak Gaussian 0.03 already produces non-trivial wrong-neighbor rates, and stronger Gaussian, blur, and resize are severe no-go cases across representation levels.

#### Criterion 3: planner-facing relevance

Large representation shift is not necessarily a control failure. The perturbation should be relevant to action-conditioned prediction or candidate ranking:

```text
ACPC_gap_l(o, tau, a)
candidate_rank_spearman
candidate_top1_flip_rate
candidate_topk_overlap
candidate_margin_clean_q10/q50
```

The ACPC-Flow v2 audit verifies planner-facing relevance: top-1 candidate flips are about `0.20` for Gaussian 0.03, about `0.66` for Gaussian 0.08, and about `0.83/0.79` for blur/resize. Training should therefore focus on directions that are both real pixel-pushforward directions and planner-facing failure directions.

### 4.2 Local pushforward covariance

For small pixel perturbations `xi`, local linearization gives:

```text
H(o + xi) - H(o) ~= J_H(o) xi
```

If:

```text
xi ~ N(0, Sigma_x)
```

then locally:

```text
Delta_H ~ N(0, J_H(o) Sigma_x J_H(o)^T)
```

For isotropic pixel Gaussian noise, `Sigma_x = sigma_x^2 I`, so:

```text
Cov(Delta_H) ~= sigma_x^2 J_H(o) J_H(o)^T
```

Thus even isotropic pixel noise generally becomes anisotropic in representation space. Blur, resize, compression, brightness, and occlusion are even less likely to be zero-mean isotropic after encoding.

### 4.3 Proposition: covariance domination for Gaussian coverage

Let `Q_l = N(mu_Q, Sigma_Q)` be a candidate Gaussian latent perturbation family, and let the pixel-pushforward deltas at level `l` have mean `mu_tau` and covariance `Sigma_tau`. A necessary condition for broad ellipsoidal coverage is that, in the dominant subspace of `Sigma_tau`,

```text
Sigma_Q approximately dominates c * Sigma_tau
```

for a quantile-dependent factor `c`, and that `mu_Q` accounts for any non-negligible mean drift:

```text
||mu_tau - mu_Q||_{Sigma_Q^{-1}} small.
```

If `Sigma_Q` fails to dominate the top eigendirections of `Sigma_tau`, coverage fails. If it dominates only by inflating irrelevant dimensions, non-crossing may fail.

### 4.4 Proposition: isotropic radius conflict

Suppose the pixel-pushforward covariance `Sigma_tau` has large top eigenvalue but low effective rank:

```text
effective_rank(Sigma_tau) << d.
```

An isotropic latent noise `N(0, sigma^2 I)` that covers the top direction needs `sigma^2` comparable to the top eigenvalue. Its expected radius scales as:

```text
E||eps|| ~= sigma sqrt(d).
```

The true pushforward radius scales with:

```text
sqrt(trace(Sigma_tau)).
```

When `Sigma_tau` is low-rank, isotropic noise wastes radius in irrelevant directions. This can make it simultaneously too small along relevant directions and too large in total radius, causing basin crossing.

### 4.5 Proposition: layerwise amplification

Let:

```text
Delta_H = H(tau(o)) - H(o)
Delta_Z = P(H(tau(o))) - P(H(o))
Delta_U = B(Z_tau, a) - B(Z_clean, a)
Delta_Y = R(U_tau) - R(U_clean)
```

Define:

```text
amp_P = ||Delta_Z|| / (||Delta_H|| + eps)
amp_B = ||Delta_U|| / (||Delta_Z|| + eps)
amp_R = ||Delta_Y|| / (||Delta_U|| + eps)
amp_total = ||Delta_Y|| / (||Delta_H|| + eps)
```

High `amp_P` suggests the encoder projector is a failure amplifier; high `amp_B` suggests the predictor backbone amplifies residual nuisance; high `amp_R` suggests the predictor projection is a planner-facing amplifier. This determines where structured perturbation or plateau training should be applied.

The ACPC-Flow v2 audit already gives a strong prior: in the TwoRoom origin baseline, `amp_P_q90` is high across stressors, about `5.1--6.0`, making `P` the dominant fixed-checkpoint amplifier. Yet fixed repair remains no-go because coverage and non-crossing fail. This motivates training-time `P/R` distribution migration rather than post-hoc adapters.

### 4.6 Proposition: rank-flip risk from covariance geometry

Let `J_j(z)` be the model-predicted cost for candidate action sequence `j`, and suppose:

```text
J_j(z + eps) ~= J_j(z) + grad J_j(z)^T eps.
```

For two candidates `i` and `j`, define clean margin:

```text
margin_ji = J_j(z) - J_i(z),   margin_ji > 0 if i is better.
```

Under `eps ~ N(0, Sigma_Q)`, the perturbation variance of the pairwise cost difference is:

```text
Var[(J_j - J_i)(z + eps)] = (grad J_j - grad J_i)^T Sigma_Q (grad J_j - grad J_i).
```

Rank flips are likely when:

```text
margin_ji <= c * sqrt((grad J_j - grad J_i)^T Sigma_Q (grad J_j - grad J_i)).
```

This connects latent perturbation covariance directly to candidate-rank instability. A perturbation distribution can be safe for representation distance but unsafe for candidate ranking, or vice versa.

### 4.7 Theoretical no-go conditions

A latent perturbation family is no-go at level `l` if any of the following hold:

1. It cannot cover measured `Delta_l,tau` without leaving the same-state basin.
2. Coverage requires a covariance that yields high wrong-neighbor or low SMPR rates.
3. Dominant pixel-pushforward directions are action/task-relevant and should not be contracted.
4. Candidate ranking changes are caused by upstream prediction errors that cannot be repaired by perturbing level `l`.
5. Oracle corruption-family labels or non-separable `t_start` labels are required for main-claim performance.

---

## 5. Method: Pushforward-Calibrated Latent Perturbation

The method is intentionally staged. The audit is part of the method, not a preliminary convenience.

### 5.1 Stage A: Completed ACPC-Flow constraints

Before adding a new audit, reuse the completed ACPC-Flow evidence as hard constraints:

1. **Do not train frozen post-hoc ACPC-Flow adapters.** Core64 and v2 reject fixed-checkpoint local synthetic repair.
2. **Do not use pure time-conditioned FM as the next method.** V2 `t` calibration is not separable.
3. **Do not interpret `no_go` labels as saying ordinary noise training failed.** They only gate frozen synthetic repair.
4. **Treat the four-task origin-vs-noise aligned summary as mechanism evidence.** Ordinary noise training strongly moves ATR, SMPR, `amp_P`, wrongNN, top1 flip, and top5 overlap for matched Gaussian across all four tasks.
5. **Treat blur/resize as task-dependent.** TwoRoom/Reacher and PushT show positive aligned movement, while Cube is a boundary.

### 5.2 Stage B: Pushforward Noise Geometry and Replay Audit

Create or extend:

```text
tools/acpc_flow/pushforward_noise_audit.py
```

This should extend the existing ACPC-Flow v2 outputs rather than duplicate them. It should add covariance geometry and synthetic structured replay on top of the already-computed levels, amplification, candidate rank, and `t` calibration.

Outputs:

```text
assets/paper2_data/latent_pushforward_audit_<task>_<checkpoint>_<date>.json
assets/paper2_data/latent_pushforward_audit_<task>_<checkpoint>_<date>.csv
assets/paper2_data/latent_pushforward_audit_summary_<date>.md
```

For each task, checkpoint, perturbation family, severity, and representation level, compute:

```text
# shift scale, already partly in ACPC-Flow v2
delta_norm_mean/median/q90/q95
ratio_to_clean_knn_q50/q90/q95

# covariance geometry, new
mean_shift_norm
cov_trace
cov_effective_rank
lambda_max_over_trace
top1/top5/top10_energy
diagonal_energy_ratio
offdiag_energy_ratio
family_subspace_overlap

# candidate Q coverage, new/expanded
coverage_isotropic_q95
coverage_diag_q95
coverage_lowrank_r{1,2,4,8,16}_q95
coverage_mixture_q95
mahalanobis_q50/q90/q95
required_isotropic_std
required_lowrank_rank_for_coverage

# safety, partly in ACPC-Flow v2
wrong_nn_rate
closer_to_wrong_than_pair_rate
crossing_rate_isotropic
crossing_rate_diag
crossing_rate_lowrank
safe_radius_q95
coverage_safe_conflict

# amplification, already in ACPC-Flow v2
amp_P
amp_B
amp_R
amp_total

# planner-facing relevance, partly in ACPC-Flow v2
ACPC_gap_q90/q95
candidate_rank_spearman
candidate_top1_flip_rate
candidate_topk_overlap
candidate_margin_clean_q10/q50
rank_flip_bound_mean/q90

# replay fidelity, new
delta_direction_cosine_vs_pixel
norm_ratio_vs_pixel
ACPC_gap_match_error
rank_spearman_match_error
topk_overlap_match_error

# decision
decision
recommended_next_action
```

Decision labels:

```text
no_go
isotropic_no_go
diagonal_candidate
lowrank_diag_candidate
family_mixture_candidate
projector_migration_candidate
predictor_projector_candidate
pixel_paired_upper_bound_only
needs_semantic_guard
training_time_only
```

### 5.3 Stage C: Candidate perturbation families

#### Family 1: isotropic scalar noise

```text
eps ~ N(0, sigma^2 I)
```

Role: baseline and negative control. The completed ACPC-Flow audits already make this unlikely for post-projector `emb` fixed repair.

#### Family 2: diagonal anisotropic noise

```text
eps ~ N(0, diag(s_1^2, ..., s_d^2))
```

Role: tests whether per-dimension scale fixes the isotropic mismatch.

#### Family 3: low-rank plus diagonal noise

```text
eps = U_r Lambda_r^{1/2} eta + sigma_floor delta
eta ~ N(0, I_r)
delta ~ N(0, I_d)
```

Role: primary candidate if measured deltas have low effective rank.

#### Family 4: mean-shifted family-specific noise

```text
eps = mu_tau + U_tau Lambda_tau^{1/2} eta + sigma_floor delta
```

Role: upper-bound or family-aware candidate for perturbations with non-zero drift.

#### Family 5: mixture covariance

```text
Q_l = sum_k pi_k Q_l,k
```

Role: covers multiple perturbation families or latent clusters without forcing one covariance to explain all shifts.

#### Family 6: token/state-conditioned covariance

```text
Q_l(o, t_token) = N(mu(o,t), Sigma(o,t))
```

Role: long-term candidate if global covariance is too coarse.

#### Family 7: pixel-paired source upper bound

```text
source = r_l(tau(o))
target = r_l(o)
```

Role: upper bound. It is not a clean-only latent perturbation method and must not be framed as corruption-agnostic unless the corruption process is available during training and not during evaluation.

### 5.4 Stage D: offline perturbation replay

Before training, replay synthetic structured perturbations at the chosen representation level:

```text
r_l_synth = r_l(o) + eps_structured
r_l_pixel = r_l(tau(o))
```

Compare structured latent noise with measured pixel-pushforward deltas:

```text
direction cosine: cos(r_l_synth - r_l_clean, r_l_pixel - r_l_clean)
norm ratio: ||synthetic_delta|| / ||pixel_delta||
ACPC gap match
candidate rank Spearman match
top-k overlap match
wrong-neighborhood rate
```

A training MVE is allowed only if structured latent replay matches pixel-induced predictive/rank effects better than isotropic and random low-rank controls.

### 5.5 Stage E: training objectives

#### Objective 1: encoder-feature perturbation through projector

Use when `encoder_feat` is safe and `amp_P` suggests the projector is trainably shapeable.

```text
h = H(o)
eps ~ Q_H
z_source = P(h + eps)
z_clean = sg(P(h))
L_z = ||z_source - z_clean||^2
```

#### Objective 2: predictor-facing plateau

Use when structured perturbations match pixel-induced predictive drift.

```text
z_source = z + eps_structured
pred_source = F(z_source, a)
pred_clean = sg(F(z, a))
L_pred = ||pred_source - pred_clean||^2
```

#### Objective 3: predictor-projector plateau

Use when `amp_R` or post-`pred_proj` rank instability is high.

```text
u_source = B(z_source, a)
y_source = R(u_source)
y_clean = sg(R(B(z, a)))
L_R = ||y_source - y_clean||^2
```

#### Objective 4: hybrid ACPC plateau

```text
L = alpha_z L_z + alpha_pred L_pred + alpha_R L_R
```

Start with one active component for attribution. Do not train `P` and `R` jointly until single-sided tests win.

### 5.6 Guard logging and optional guard losses

Always log:

```text
SMPR or task-grounded margin pass rate
transition resolution
inverse-dynamics probe
effective rank
local kNN / non-crossing
candidate rank metrics
clean behavior
```

Do not add explicit discriminability losses until isolated tests show collapse risk. If a guard loss is later needed, it should be a small margin-preservation term over task/action-distinct pairs rather than a global repulsion objective.

---

## 6. Experiments

### 6.1 Existing empirical evidence to integrate

#### Gaussian lockbox

Training seeds `3073/3074` reproduce the Paper1 pattern:

- TwoRoom: std0.08 endpoint improves obs0.08 by about +26.50 pp over baseline.
- PushT: std0.08 endpoint improves obs0.08 by about +75.50 pp.
- Reacher: std0.08 endpoint improves obs0.08 by about +63.17 pp.
- Cube: std0.08 endpoint improves obs0.08 by about +21.67 pp.

The best std is task- and seed-dependent, so this supports plateau language rather than a universal noise optimum.

#### Gaussian diagnostics

Recovered endpoints show large reductions in eight-step predictor rollout drift and high clean/noisy CKA. This supports the mechanism-localization view: robust checkpoints reduce action-conditioned predictive drift.

#### ACPC-Flow core/v2 fixed-repair audit

The core64 and v2 audits reject frozen local synthetic repair:

- post-projector `emb` Gaussian 0.03 already has zero coverage at synthetic std=0.04/q95 and a pixel shift far larger than synthetic radius;
- v2 no-go holds across `encoder_feat`, `emb`, `predictor_hidden`, and `pred_emb` for Gaussian 0.03/0.05/0.08, blur, and resize;
- `P` is the dominant fixed-checkpoint amplifier in the origin TwoRoom audit;
- candidate rankings are affected by corruption;
- non-oracle `t_start` is not separable.

#### ACPC-Flow four-task origin-vs-noise aligned summary

The same v2 metrics show that ordinary input-noise training reshapes the P/R path and candidate-rank behavior:

- matched Gaussian 0.08 improves eval, lowers ATR, raises SMPR, lowers `amp_P`, lowers wrong-neighbor rates, reduces top-1 flips, and raises top-5 overlap across all four tasks;
- blur/resize movement is task-dependent: TwoRoom/Reacher align strongly, PushT is positive especially on resize, and Cube is the boundary.

This evidence motivates training-time distribution migration rather than frozen adapter repair.

#### Unseen stressor boundary

Strongest-only blur/resize transfer is positive on TwoRoom/Reacher, weak/mixed on PushT, and neutral/slightly negative on Cube. This supports a bounded cross-stressor story but not a universal transfer claim.

#### Negative method controls

GLC, SNAP-ACPC, paired no-aux, and robust CEM are all negative or weak. These controls justify moving upstream to perturbation geometry rather than adding another naive consistency objective or planner rerank.

### 6.2 Experiment 1: pushforward covariance/replay audit, no training

**Goal.** Determine whether measured pixel-induced shifts are isotropic, diagonal, low-rank, mean-shifted, or family-specific, and identify safe representation levels. This should extend ACPC-Flow v2 rather than rerun a disconnected audit.

**Tasks.** First pass:

```text
TwoRoom, Reacher
```

Second pass:

```text
PushT, Cube
```

**Checkpoints.**

```text
origin baseline_seed3073
Gaussian std0.08 noise-trained seed3073
```

Optionally add seed3072/3074 once the audit runs.

**Perturbations.**

```text
gaussian_noise: 0.03, 0.05, 0.08
gaussian_blur: k=7 and/or k=15
resize: 0.5 and/or 0.25
```

**Levels.**

```text
encoder_feat
emb
predictor_hidden
pred_emb
```

**Baselines.**

```text
isotropic scalar noise
diagonal noise
low-rank+diag noise
random low-rank subspace
pixel-paired source upper bound
```

**Success condition.** At least one representation level and one structured noise family shows substantially better coverage/safety/replay trade-off than isotropic noise and random low-rank controls.

**Failure condition.** All noise families either fail coverage/replay or cross task-distinct neighborhoods.

### 6.3 Experiment 2: offline perturbation replay

**Goal.** Test whether structured latent perturbations reproduce pixel-induced predictive and ranking effects without training.

**Protocol.** For each sample, compare:

```text
pixel branch:      r_l(tau(o))
isotropic branch:  r_l(o) + eps_iso
diagonal branch:   r_l(o) + eps_diag
low-rank branch:   r_l(o) + eps_lr
random-LR branch:  r_l(o) + eps_random_lr
```

**Metrics.**

```text
delta direction cosine
delta norm ratio
ACPC gap error
candidate rank Spearman error
top-k overlap error
wrong-neighbor rate
```

**Success condition.** Structured perturbation matches pixel branch better than isotropic and random low-rank controls on both geometry and planner-facing metrics.

### 6.4 Experiment 3: one-task training MVE

**Goal.** Determine whether pushforward-calibrated latent perturbation training can improve closed-loop robustness under matched Gaussian stress.

**First target.**

```text
TwoRoom or Reacher
```

Do not begin with PushT; it is contact-heavy and likely requires stronger semantic guards.

**Models.**

```text
origin baseline
ordinary pixel Gaussian noise training
isotropic latent noise training
random low-rank latent noise training
pushforward-calibrated diagonal
pushforward-calibrated low-rank+diag
pixel-paired upper bound, optional
```

**Metrics.**

```text
clean success
pixels_std0.03 / 0.05 / 0.08 success
pixels_goal_std0.08, auxiliary
ATR / ACPC-H
SMPR / semantic margin pass
candidate rank flip / top-k overlap
rollout T8 drift
CKA / angle
transition resolution / inverse dynamics
```

**Success gate.**

- corrupted behavior improves over origin and isotropic latent noise;
- clean drop <= 5 pp;
- ACPC/ATR decreases;
- SMPR/discriminability does not collapse;
- candidate rank metrics improve;
- method is competitive with ordinary pixel noise training or provides a clear mechanism/transfer advantage.

### 6.5 Experiment 4: multi-seed matched Gaussian

Run only if Experiment 3 passes.

```text
training seeds: 3072/3073/3074
tasks: TwoRoom, Reacher, then PushT/Cube
```

Compare against ordinary pixel noise training at matched `std_max` and seed.

### 6.6 Experiment 5: bounded unseen perturbation transfer

Run only after matched Gaussian success.

```text
families: blur, resize, brightness/contrast, maybe compression/cutout
```

Claim only bounded transfer. Do not write universal cross-perturbation robustness.

---

## 7. Current Results and Expected Result Tables

### 7.1 Current result: Gaussian recovery and diagnostic alignment

Current Paper1 artifacts already support the following:

1. Input-side Gaussian noise training strongly recovers matched Gaussian observation-noise robustness across three LeWM training seeds.
2. Robust endpoints show large reductions in same-action clean/noisy predictor drift.
3. ACPC-family diagnostics retain modest residual association with reduced Gaussian drop after controlling for task, training seed, and noise level.
4. Plateau-localization rules are useful but not exact checkpoint selectors.

These results motivate the method but are not yet evidence for structured latent perturbation.

### 7.2 Current result: ACPC-Flow fixed repair is closed, training-time migration remains open

The completed ACPC-Flow audits support a two-part result:

1. **Closed:** fixed-checkpoint synthetic local repair is no-go. This includes post-projector `emb + epsilon -> clean emb`, analogous repair at `encoder_feat`, `predictor_hidden`, and `pred_emb`, and non-oracle `t_start` time-conditioned FM under the current calibration.
2. **Open:** training-time P/R distribution migration is plausible. Ordinary input-noise training changes `amp_P`, wrong-neighbor rates, ATR, SMPR, and candidate-rank metrics in a way that aligns with matched Gaussian behavior across all four tasks.

Therefore the next method should not train a frozen adapter. It should test whether structured latent perturbation can reproduce part of ordinary input-noise training's P/R path migration from scratch.

### 7.3 Current result: direct method conversions are insufficient

Existing negative controls imply:

1. Encoder-level latent consistency is too weak or mis-targeted.
2. One-step predictive consistency is insufficient for closed-loop planning.
3. Paired clean/noisy training infrastructure can itself change behavior.
4. Planner-side final reranking or inner-loop robust CEM does not repair frozen origin checkpoints.

This motivates an upstream geometry-aware method.

### 7.4 Current Table: ACPC-Flow matched Gaussian 0.08 origin-vs-noise movement

| Task | eval px0.08 | ATR q90 | SMPR | amp_P q90 | emb wrongNN | pred wrongNN | top1 flip | top5 overlap | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TwoRoom | 68.8 -> 97.1 | 1.509 -> 0.111 | 0.339 -> 0.989 | 5.192 -> 1.358 | 0.828 -> 0.003 | 0.906 -> 0.042 | 0.664 -> 0.016 | 0.534 -> 0.977 | aligned |
| Reacher | 18.2 -> 81.6 | 2.628 -> 0.082 | 0.733 -> 0.997 | 2.926 -> 1.031 | 0.930 -> 0.003 | 0.927 -> 0.000 | 0.797 -> 0.008 | 0.516 -> 0.980 | aligned |
| PushT | 7.2 -> 85.8 | 3.580 -> 0.247 | 0.439 -> 1.000 | 1.545 -> 1.140 | 0.732 -> 0.000 | 0.711 -> 0.003 | 0.867 -> 0.023 | 0.364 -> 0.970 | aligned |
| Cube | 43.1 -> 62.6 | 2.320 -> 0.100 | 0.453 -> 1.000 | 2.012 -> 1.207 | 0.932 -> 0.000 | 0.932 -> 0.000 | 0.859 -> 0.000 | 0.413 -> 0.988 | aligned, weaker eval gain |

### 7.5 Planned Table 1: pushforward covariance geometry

| Task | Checkpoint | Stressor | Level | eff. rank | top5 energy | mean shift | offdiag ratio | amp next | Reading |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| TwoRoom | origin | gauss0.08 | encoder_feat | TBD | TBD | TBD | TBD | TBD | TBD |
| TwoRoom | origin | gauss0.08 | emb | TBD | TBD | TBD | TBD | TBD | TBD |
| Reacher | origin | blur | encoder_feat | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.6 Planned Table 2: coverage vs crossing and replay fidelity

| Task | Stressor | Level | Noise family | coverage q95 | crossing | replay ACPC err | replay rank err | decision |
|---|---|---|---|---:|---:|---:|---:|---|
| TwoRoom | gauss0.05 | encoder_feat | isotropic | TBD | TBD | TBD | TBD | TBD |
| TwoRoom | gauss0.05 | encoder_feat | lowrank+diag | TBD | TBD | TBD | TBD | TBD |
| Reacher | resize | emb | lowrank+diag | TBD | TBD | TBD | TBD | TBD |

### 7.7 Planned Table 3: offline replay fidelity

| Task | Stressor | Level | Synthetic family | delta cosine | norm ratio err | ACPC-gap err | rank-Spearman err | wrong NN |
|---|---|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | isotropic | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | lowrank+diag | TBD | TBD | TBD | TBD | TBD |

### 7.8 Planned Table 4: training MVE

| Task | Model | Clean | px0.03 | px0.05 | px0.08 | ATR | SMPR | rank flip | T8 drift | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TwoRoom | origin | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline |
| TwoRoom | pixel noise | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | strong baseline |
| TwoRoom | isotropic latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | control |
| TwoRoom | lowrank+diag latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | candidate |

---

## 8. Discussion

### 8.1 Why this may be high contribution

If the route succeeds, it transforms Paper1 from a diagnostic study into a diagnosis-guided method paper. The method would not be “latent noise helps.” The actual contribution would be:

> measure the pixel-to-latent pushforward geometry, choose a structured perturbation family that covers real nuisance directions without crossing predictive basins, and train a predictive plateau at the diagnosed layer.

This is substantially more specific than generic feature augmentation and directly uses the completed ACPC-Flow result: frozen local repair fails, but ordinary training reshapes the P/R path.

### 8.2 Why this may fail

The route can fail for principled reasons:

1. Pixel-pushforward shifts may not be coverable by any safe latent distribution at the chosen layer.
2. The relevant shifts may be highly state-dependent, making global covariance too crude.
3. PushT/contact states may require semantic guards not available from current proxies.
4. Ordinary pixel noise training may work through data-path or optimization effects that latent perturbations do not reproduce.
5. Structured covariance estimated from test perturbations may become oracle-like and fail held-out perturbations.
6. ACPC-Flow v2 may have already shown that the relevant fixed-checkpoint geometry is too far gone; the method may need to operate during representation formation rather than as a local perturbation around a trained origin manifold.

Each failure is still informative if the audit is cleanly reported.

### 8.3 Claim boundaries

Do not claim:

- universal robustness to arbitrary visual perturbations;
- that ACPC diagnostics alone predict closed-loop success;
- that diagonal/low-rank latent noise is generally sufficient;
- that matching pixel-pushforward shifts always improves control;
- that oracle family-specific covariance is corruption-agnostic;
- that ACPC-Flow `no_go` labels mean ordinary input-noise training did not improve robustness.

Safe claims after successful audit/replay stage:

- pixel perturbations induce measurable anisotropic pushforward geometry;
- isotropic latent noise is a poor coverage/safety/replay match in tested settings;
- structured low-rank/diagonal families may provide a better candidate for training;
- completed ACPC-Flow v2 fixed-repair no-go motivates training-time distribution migration rather than frozen adapters.

Safe claims after successful training MVE:

- pushforward-calibrated structured latent perturbation improves matched-stressor robustness on tested tasks/seeds;
- gains are interpreted together with clean guard, SMPR, and candidate-rank metrics;
- if transfer to blur/resize occurs, it is bounded and task/stressor-specific unless broader evidence is collected.

### 8.4 Relationship to Paper1

Paper1 should remain a diagnostic paper. This report can become Paper2 or a later method section if the new experiments pass. If the final target is a single larger paper, Paper1's ACPC theory and diagnostic evidence can become Sections 2-4, while this report supplies the method/theory/experiment extension. The ACPC-Flow v2 tables should serve as the transition section: they close frozen repair and motivate training-time pushforward-calibrated migration.

---

## 9. Implementation Plan

### PR-A: audit/replay implementation only

- [ ] Add `tools/acpc_flow/pushforward_noise_audit.py`, or extend the existing v2 audit code if cleaner.
- [ ] Create `assets/paper2_data/` if needed.
- [ ] Ingest existing ACPC-Flow v2 artifacts where possible so the new audit does not recompute already-stable metrics unnecessarily.
- [ ] Reuse existing encoder/corruption/repr-analysis utilities.
- [ ] Expose or hook `predictor_hidden` before `pred_proj`.
- [ ] Compute delta covariance and anisotropy metrics.
- [ ] Compute isotropic/diagonal/low-rank+diag coverage.
- [ ] Compute clean-kNN and semantic-proxy non-crossing metrics.
- [ ] Reuse or recompute `amp_P`, `amp_B`, `amp_R`, `amp_total`.
- [ ] Reuse or recompute candidate-rank metrics.
- [ ] Add structured replay fidelity metrics: direction cosine, norm ratio, ACPC gap error, rank-Spearman error, top-k overlap error.
- [ ] Emit JSON/CSV/MD summary.
- [ ] Do not train.

### PR-B: report update from audit/replay artifacts

- [ ] Fill planned Tables 1-3 with actual pushforward covariance and replay results.
- [ ] Decide whether any latent perturbation family qualifies for training.
- [ ] If all structured families fail, record a no-go and stop.

### PR-C: training MVE only after PR-A/B pass

- [ ] Add one default-off structured perturbation family.
- [ ] Run one task, one seed, Gaussian first.
- [ ] Compare origin, ordinary pixel noise, isotropic latent, random low-rank, and pushforward-calibrated variants.
- [ ] Require behavior + ACPC + SMPR + rank improvement.

---

## 10. References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.** arXiv:2301.08243, 2023. https://arxiv.org/abs/2301.08243

[2] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. **Revisiting Feature Prediction for Learning Visual Representations from Video.** arXiv:2404.08471, 2024. https://arxiv.org/abs/2404.08471

[3] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** arXiv:2603.19312, 2026. https://arxiv.org/abs/2603.19312

[4] Ali J. Alrasheed, Aryan Yazdan Parast, Basim Azam, James Bailey, and Naveed Akhtar. **Latent Video Prediction Learns Better World Models.** arXiv:2605.15618, 2026. https://arxiv.org/abs/2605.15618

[5] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. **Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning.** arXiv:2107.09645, 2021. https://arxiv.org/abs/2107.09645

[6] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. **Mastering Diverse Domains through World Models.** arXiv:2301.04104, 2023. https://arxiv.org/abs/2301.04104

[7] Nicklas Hansen, Hao Su, and Xiaolong Wang. **TD-MPC2: Scalable, Robust World Models for Continuous Control.** arXiv:2310.16828, 2023. https://arxiv.org/abs/2310.16828

[8] Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, Aaron Courville, David Lopez-Paz, and Yoshua Bengio. **Manifold Mixup: Better Representations by Interpolating Hidden States.** arXiv:1806.05236, 2018. https://arxiv.org/abs/1806.05236

[9] Tianlong Chen, Yu Cheng, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zhangyang Wang, and Jingjing Liu. **Adversarial Feature Augmentation and Normalization for Visual Recognition.** arXiv:2103.12171, 2021. https://arxiv.org/abs/2103.12171

[10] Yingnan Liu, Yingtian Zou, Rui Qiao, Fusheng Liu, Mong Li Lee, and Wynne Hsu. **Cross-Domain Feature Augmentation for Domain Generalization.** arXiv:2405.08586, 2024. https://arxiv.org/abs/2405.08586

[11] Haoliang Wang, Chen Zhao, and Feng Chen. **Feature-Space Semantic Invariance: Enhanced OOD Detection for Open-Set Domain Generalization.** arXiv:2411.07392, 2024. https://arxiv.org/abs/2411.07392

[12] Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G. Bellemare. **DeepMDP: Learning Continuous Latent Space Models for Representation Learning.** arXiv:1906.02736, 2019. https://arxiv.org/abs/1906.02736

[13] Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. **Causal-JEPA: Learning World Models through Object-Level Latent Interventions.** arXiv:2602.11389, 2026. https://arxiv.org/abs/2602.11389

[14] Jingyang He, Guangrun Li, Jieyu Zhang, Chengkai Hou, Zhengping Che, and Shanghang Zhang. **Demo-JEPA: Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment Imitation.** arXiv:2605.20811, 2026. https://arxiv.org/abs/2605.20811

[15] Paper1 internal artifacts in this repository: `paper1/PLAN.md`, `paper1/LOCKBOX_RESULTS_20260703.md`, `paper1/ACPC_FLOW_CODEX.md`, `paper1/ROBUST_CEM_EVAL100X3_ITERATION_LOG_20260705.md`, `assets/paper1_data/acpc_flow_coverage_tworoom_baseline_seed3073_core64.json`, `assets/paper1_data/acpc_flow_coverage_v2_tworoom_baseline_seed3073_core128_fullstress.json`, `assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.md`, `assets/paper1_data/three_seed_diagnostic_validation.md`, `assets/paper1_data/selector_*_audit_20260704.md`, and `assets/paper1_data/residual_diagnostic_audit_20260704.md`.
