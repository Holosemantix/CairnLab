# ACPC-Flow / Predictive Plateau Transport: theory, t-conditioned FM audit, and execution plan

This document is the current working plan for the ACPC-Flow direction after the first core coverage audit. It is written for Codex execution.

The short conclusion is:

> Do not train post-projector clean-only latent-noise ACPC-Flow at scale. The core coverage audit shows that small synthetic noise in `emb` does not cover pixel-corruption-induced shifts. The next viable direction is a staged audit and small experiment around **projector-as-transport** and possibly **predictor-projector plateau**. A pure time-conditioned FM variant is only a candidate if a separate `t`-calibration audit shows that inference-time `t_start` can be chosen without clean/noisy labels.

---

## 0. Current empirical status from core coverage audit

Existing artifacts:

```text
assets/paper1_data/acpc_flow_coverage_tworoom_baseline_seed3073_core64.json
assets/paper1_data/acpc_flow_coverage_tworoom_baseline_seed3073_core64.csv
```

The core64 audit used TwoRoom `baseline_seed3073` and found:

### Post-projector `emb`

```text
gaussian 0.03: no_go, ratio_q90 ~= 2.266, coverage@0.04/q95 = 0, wrong_nn ~= 0.137
gaussian 0.08: no_go, ratio_q90 ~= 7.752, coverage@0.04/q95 = 0, wrong_nn ~= 0.836
blur k7:       no_go, ratio_q90 ~= 17.871, coverage@0.04/q95 = 0, wrong_nn ~= 0.918
resize 0.5:    no_go, ratio_q90 ~= 16.304, coverage@0.04/q95 = 0, wrong_nn ~= 0.941
```

For `emb` Gaussian 0.03:

```text
pixel shift delta_q90 ~= 10.27
synthetic std=0.04 radius_q95 ~= 0.70
synthetic std=0.12 radius_q95 ~= 2.10
```

Interpretation:

> `emb + epsilon` clean-only latent noise does not cover even weak Gaussian pixel corruption in the post-projector latent space. This is not a small hyperparameter issue.

### Pre-projector `encoder_feat`

`encoder_feat` is less hopeless for weak Gaussian but still weak:

```text
encoder_feat gaussian 0.03: low, ratio_q90 ~= 3.231, coverage@0.12/q95 ~= 0.824
encoder_feat gaussian 0.08: no_go
blur / resize: no_go
```

Interpretation:

> The current evidence does not support broad clean-only feature-noise generalization. At most it leaves a small opening for weak Gaussian/local-sensor-noise experiments through the original encoder projector.

### Immediate conclusion

Do not run large ACPC-Flow training yet. First expand the audit to answer:

1. Does the encoder projector `P` amplify pixel-induced shifts?
2. Does the predictor backbone or `pred_proj` amplify residual shifts?
3. Are candidate rankings actually affected in the same way as synthetic feature perturbations?
4. Can a time-conditioned FM model choose a useful inference-time `t_start` without an oracle clean/noisy flag?

---

## 1. Correct conceptual framing

ACPC does **not** say encoder geometry is unimportant. The correct framing is:

> Encoder geometry is a first-stage risk signal. Same-state perturbed views should remain in the same-state predictive basin and should not cross into task-distinct neighborhoods. ACPC then asks whether the remaining encoder/projector shift changes action-conditioned predicted futures, candidate costs, and rankings. SMPR checks that this contraction does not collapse action-relevant distinctions.

Thus there are two coupled requirements:

1. **Neighborhood consistency / non-crossing**: clean and perturbed representations for the same state should stay in the same predictive basin; they should not become closer to task-distinct states.
2. **Predictive plateau + anti-collapse**: after the same action rollout, transported perturbed histories should match clean histories in diagnostic space, while task-grounded different-state pairs remain separated.

Paper1's diagnostic target can be written as:

```text
z_t = E_theta(h_t),   z_tilde_t = E_theta(h_tilde_t)
zhat_{t+k}       = F_theta^k(z_t,       a_{0:k-1})
zhat_tilde_{t+k} = F_theta^k(z_tilde_t, a_{0:k-1})
ACPC-H = sum_k alpha_k * d(Pi(zhat_{t+k}), Pi(zhat_tilde_{t+k}))
```

ACPC-Flow inserts a transport/projection mechanism before the rollout comparison:

```text
eps_phi = d_H(Pi(F^{1:H}(T_phi(z_tilde), a)), Pi(F^{1:H}(z, a))).
```

The method is useful only if it reduces this planner-facing discrepancy while preserving SMPR/non-crossing.

---

## 2. Why pure large-noise FM is not automatically a solution

A tempting idea is to use standard Flow Matching with a full noise path:

```text
x_s = (1-s) * clean_latent + s * pure_noise,  s in [0,1]
```

and train a vector field that maps from noisy points back to clean/origin latent. This appears to solve coverage because pure noise has large radius.

This is not sufficient for control.

### 2.1 Radius coverage is not state-preserving coverage

Standard marginal FM learns:

```text
noise distribution -> clean latent distribution
```

or

```text
T_phi# p_noise ~= p_clean.
```

But control requires paired/conditional transport:

```text
same-state perturbed latent -> same-state clean predictive basin.
```

If the source is too noisy, two different states can produce overlapping intermediate points. A deterministic vector field cannot map the same input region to two different clean states. This creates identity ambiguity.

Therefore:

> Full-radius FM may cover the geometric magnitude of pixel corruption, but it may destroy the state identity needed for control.

### 2.2 Large-noise paths can cross task basins

The coverage audit already shows that pixel corruption can move `emb` several clean-neighborhood radii away. Pushing synthetic noise even larger may match the radius, but it moves samples into cross-neighborhood regions. This violates the non-crossing condition.

A pure FM model trained on such large paths risks learning an average or marginal clean latent manifold, not a state-preserving correction.

### 2.3 Acceptable FM variant: local conditional FM

The acceptable FM-style version is local and paired:

```text
source = z + sigma(s) * epsilon
target = z
condition = same state / same action rollout
```

where `sigma_max` is chosen from a coverage audit and must remain inside the same-state basin. This is better described as:

```text
local conditional ACPC-Flow
```

not unconditional marginal FM.

---

## 3. Inference-time `t_start`: feasibility and risks

If the projector/transport is time-conditioned, inference may choose a starting time/noise level `t_start`:

```text
T_phi(x, t_start)
```

This is potentially useful: clean/origin inputs can use `t_start=0`, while corrupted inputs can use larger `t_start`.

However, this raises a critical question:

> At inference time, how does the model know which `t_start` to use when it does not know whether the input is clean or perturbed?

### 3.1 Fixed `t_start` is weak

A fixed nonzero `t_start` applies correction to every input, including clean inputs. This can hurt clean control. A fixed zero `t_start` does nothing for corrupted inputs.

Therefore fixed `t_start` is only a baseline.

### 3.2 Oracle `t_start` is not allowed for main claims

If evaluation uses knowledge of the corruption type/severity to set `t_start`, then the method is no longer corruption-agnostic. It becomes a matched test-time intervention.

This can be an upper bound, but not the main method.

### 3.3 Learnable `t_start` estimator is possible but must be audited

A practical design is:

```text
t_hat = q_psi(x)
output = T_phi(x, t_hat)
```

where `x` may be `encoder_feat`, `emb`, predictor hidden, or another internal representation.

Possible signals for `q_psi`:

- distance to clean representation bank / kNN radius;
- SIGReg density / latent norm anomaly;
- predictor self-consistency under small local perturbations;
- candidate rank instability proxy;
- correction norm predicted by the transport head;
- learned source-noise labels from synthetic perturbation training.

But `q_psi` can fail if synthetic noise labels do not correspond to real pixel-corruption shifts. Therefore it requires a `t_calibration_audit` before training.

### 3.4 Multi-`t` self-selection is possible but expensive

At inference, one could evaluate multiple candidate `t_start` values:

```text
t in {0, 0.25, 0.5, 0.75, 1.0}
```

and choose the one minimizing a no-reference score, such as:

- predicted rollout stability under small local perturbations;
- small correction norm subject to low predictor disagreement;
- candidate cost/rank stability;
- distance to clean latent bank.

This is test-time selection. It may be useful, but it increases compute and risks becoming another weak planner-side trick unless it clearly beats compute-matched baselines.

### 3.5 Required `t_calibration_audit`

Before implementing time-conditioned FM training, add an audit to answer:

1. For each pixel corruption, what `t_star` would be needed to cover its representation shift?
2. Is `t_star` near 0 for clean/origin inputs?
3. Are `t_star` values separable between clean and corrupted inputs using non-oracle features?
4. Does the required `t_star` remain inside same-state neighborhoods, or does it imply cross-neighborhood transport?
5. Can a simple estimator predict `t_star` without knowing the corruption type?

Operational definition:

```text
t_star(o, tau) = smallest t in grid such that synthetic_radius_q95(t) >= ||Delta_tau(o)||
```

Also compute an ACPC version:

```text
t_star_acpc(o, tau) = smallest t such that synthetic_acpc_gap_q95(t) >= pixel_acpc_gap(o,tau)
```

Report:

```text
t_star_median
t_star_q90
t_star_q95
clean_false_positive_rate_at_t_threshold
wrong_label_rate_at_required_t
coverage_at_t_grid
```

Decision:

- If `t_star_q90` for the target stressor is large and wrong-label crossing is high, time-conditioned FM is no-go.
- If `t_star` is small for weak Gaussian and separable from clean, run a small local conditional FM experiment.
- Do not train full pure-noise FM unless a state-preserving conditioning mechanism is implemented and audited.

---

## 4. Two projector chain: encoder projector and predictor projector

LeWM has two relevant projection points:

```text
pixels -> encoder H -> encoder projector P -> emb z -> predictor backbone B -> pred_proj R -> predicted emb
```

The previous post-projector ACPC-Flow idea focused only on `emb` after `P`. That is incomplete.

### 4.1 Encoder projector `P` as transport / plateau projector

Primary candidate method:

```text
h = H(o)
z = P(h)
h_source = h + eps
z_source = P(h_source)
```

Training objectives:

```text
latent_z:   ||P(h+eps) - sg(P(h))||^2
predictor:  ||F(P(h+eps), a) - sg(F(P(h), a))||^2
diagnostic: ACPC_diag(F(P(h+eps), a), F(P(h), a))
```

This is stronger than adding a post-projector adapter because:

- it adds no new inference module;
- it makes the original LeWM projector learn a local predictive plateau;
- it directly targets the feature-shift coverage question from the audit.

### 4.2 Predictor projector `R = pred_proj` as predictive plateau map

A second candidate is to regularize predictor hidden/output projection:

```text
u_clean = B(z_clean, a)
y_clean = R(u_clean)
u_source = B(z_source, a)
y_source = R(u_source)
```

Possible objectives:

```text
pred_proj_z:   ||R(u_source) - sg(R(u_clean))||^2
pred_proj_diag: ACPC/diagnostic distance between y_source and y_clean
```

Motivation:

> ACPC is measured after action-conditioned prediction. If the predictor backbone or `pred_proj` amplifies residual nuisance shifts, only training the encoder projector may be insufficient.

### 4.3 Do not train both first

Do not start with both projectors enabled. That makes attribution impossible and increases collapse risk.

Required order:

1. Audit amplification at `P` and `R`.
2. Train encoder-projector-only small experiment if audit supports it.
3. Train predictor-projector-only small experiment if audit supports it.
4. Only then test two-sided training.

---

## 5. Feasibility theory for feature/latent perturbation coverage

Let `H(o)` be the pre-projector encoder feature and `P` the encoder projector. A pixel perturbation induces:

```text
Delta_H_tau(o) = H(tau(o)) - H(o)
Delta_z_tau(o) = P(H(tau(o))) - P(H(o))
```

Synthetic feature perturbation training can cover a pixel perturbation family only if these shifts lie inside, or near, the synthetic perturbation tube.

Sufficient condition:

```text
||Delta_H_tau(o) - eps_star|| <= kappa
```

and training achieves:

```text
d_H(Gbar_a(T_phi(H(o)+eps_star)), Gbar_a(H(o))) <= eps_train
```

with local Lipschitz constant `L`. Then:

```text
d_H(Gbar_a(T_phi(H(tau(o)))), Gbar_a(H(o))) <= eps_train + L*kappa.
```

If `kappa` is large, the method is extrapolating.

For small Gaussian pixel noise:

```text
H(o+xi)-H(o) ~= J_H(o) xi
Delta_H_tau ~ N(0, sigma^2 J_H J_H^T)
```

Isotropic feature noise covers it only if, roughly:

```text
sigma_H^2 I >= sigma^2 J_H J_H^T
```

This is why large structured shifts from blur/resize/compression cannot be assumed covered.

Impossibility condition:

If two task-distinct corrupted inputs collide in representation space, deterministic transport cannot recover both different states. Therefore neighborhood crossing is a hard no-go signal.

---

## 6. Coverage audit v2: must run before new training

Create or extend:

```text
tools/acpc_flow/coverage_audit.py
```

Existing core audit should be extended. Required outputs:

```text
assets/paper1_data/acpc_flow_coverage_v2_<task>_<checkpoint>_<date>.json
assets/paper1_data/acpc_flow_coverage_v2_<task>_<checkpoint>_<date>.csv
```

### 6.1 Increase sample size

Run at least:

```text
num_samples: 1000
```

Core64 is enough to reject the old post-projector direction, but not enough for final task/stressor decisions.

### 6.2 Required representation levels

Compute clean/corrupted shifts at:

```text
encoder_feat:      h = H(o)
emb:               z = P(h)
predictor_hidden:  u = B(z, a) before pred_proj
pred_emb:          y = R(u)
```

If `predictor_hidden` is not exposed, add a helper or hook to return the predictor backbone hidden before `pred_proj`.

### 6.3 Amplification metrics

Report:

```text
amp_P = ||Delta_emb|| / (||Delta_encoder_feat|| + eps)
amp_B = ||Delta_predictor_hidden|| / (||Delta_emb|| + eps)
amp_R = ||Delta_pred_emb|| / (||Delta_predictor_hidden|| + eps)
amp_total = ||Delta_pred_emb|| / (||Delta_encoder_feat|| + eps)
```

Interpretation:

- high `amp_P`: encoder projector is a failure amplifier; prioritize projector-as-transport.
- high `amp_R`: pred_proj is a failure amplifier; prioritize predictor-projector plateau.
- high `amp_B`: predictor backbone itself amplifies; simple projector-only method may be insufficient.

### 6.4 Candidate rank metrics

The previous audit had `candidate_rank_metrics_computed=false`. This must be fixed.

For a shared candidate pool, compute:

```text
candidate_rank_spearman
candidate_top1_flip_rate
candidate_topk_overlap_rate
candidate_margin_clean_q10
candidate_margin_clean_q50
```

Compute these for:

1. clean vs pixel-corrupted;
2. clean vs synthetic encoder-feature perturbation through `P`;
3. clean vs synthetic post-projector latent perturbation;
4. clean vs synthetic predictor-hidden perturbation through `R`, if implemented.

### 6.5 Time-conditioned FM calibration audit

Add a `t_grid`:

```yaml
t_grid: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
noise_schedule: linear_or_variance_preserving
```

For each representation level and stressor, compute:

```text
t_star_radius = min t such that synthetic_radius_q95(t) >= pixel_delta_norm
t_star_acpc   = min t such that synthetic_acpc_gap_q95(t) >= pixel_acpc_gap
```

Report:

```text
t_star_radius_median/q90/q95
t_star_acpc_median/q90/q95
wrong_label_rate_at_t_star
clean_false_positive_rate
is_t_start_separable_from_clean
```

If `t_star` is large and crossing is high, pure/t-conditioned FM is no-go.

### 6.6 Decision table

Output a concise table:

```text
task, stressor, level, decision, reason, recommended_next_action
```

Decision values:

```text
no_go
weak_local_only
encoder_projector_small_train
predictor_projector_small_train
t_conditioned_fm_upper_bound_only
t_conditioned_fm_candidate
pixel_paired_source_candidate
```

---

## 7. Training roadmap after audit v2

### Stage A: no training if audit remains no-go

If v2 audit says no-go for all levels except weak Gaussian, do not train broad ACPC-Flow.

### Stage B: encoder projector-as-transport small experiment

Run only if v2 audit supports `encoder_projector_small_train`.

Task:

```text
TwoRoom first
```

Stressors:

```text
clean
gaussian_std0.03
gaussian_std0.05
gaussian_std0.08 only as stress, not success target
```

Models:

```text
origin baseline
encoder_projector_latent_z
encoder_projector_predictor
encoder_projector_diagnostic
```

No blur/resize in this stage unless audit is medium/high for them.

### Stage C: predictor projector plateau small experiment

Run only if v2 audit shows high `amp_R` or candidate rank instability after predictor projection.

Models:

```text
predproj_latent/prediction
predproj_diagnostic
```

Keep encoder projector unchanged for attribution.

### Stage D: t-conditioned FM only as upper bound first

Before a learned `t_start` estimator, run an oracle upper bound:

```text
choose t_start using known corruption severity / audit-derived t_star
```

This is not a valid main method, but it tells whether time-conditioned correction could help at all.

If oracle t does not help, stop.

If oracle t helps, implement non-oracle `t_start` estimator and compare:

```text
fixed t
oracle t
estimated t
multi-t self-selection
```

### Stage E: two-sided training only after single-sided wins

Only if Stage B or C gives clear gains, test:

```text
encoder projector plateau + predictor projector plateau
```

Do not start here.

---

## 8. Method variants to keep

### Variant 1: Projector-as-transport, latent anchor

```text
L_z = ||P(H(o)+eps) - sg(P(H(o)))||^2
```

### Variant 2: Projector-as-transport, predictor matching

```text
L_pred = ||F(P(H(o)+eps), a) - sg(F(P(H(o)), a))||^2
```

### Variant 3: Projector-as-transport, diagnostic ACPC

```text
L_ACPC = D_diag(Pi(F(P(H(o)+eps), a)), Pi(F(P(H(o)), a)))
```

### Variant 4: Predictor-projector plateau

```text
u_clean = B(P(H(o)), a)
u_source = B(P(H(o)+eps), a)
L_predproj = ||R(u_source) - sg(R(u_clean))||^2
```

### Variant 5: Local t-conditioned FM

Only after t-calibration audit:

```text
source = z + sigma(t) eps
target = z
loss = ||v_phi(source, t) - (target-source)||^2 + ACPC loss
```

Must include clean/origin identity:

```text
t=0 -> correction near zero
```

Do not implement pure marginal noise-to-clean FM as the main method.

---

## 9. Success and no-go criteria

Promote any ACPC-Flow method only if:

1. v2 coverage audit supports the claimed stressor/level.
2. Offline ATR decreases and SMPR does not drop.
3. Candidate rank flip/top-k overlap improves.
4. Clean performance remains within 5 pp.
5. Small closed-loop eval improves over origin on the target stressor.
6. The method beats same-parameter identity/random controls.
7. For t-conditioned FM, estimated/non-oracle `t_start` works; oracle-only success is insufficient.

No-go if:

- target stressor remains no-go in v2 coverage audit;
- required `t_start` is large and causes neighborhood crossing;
- pure synthetic noise only covers radius by leaving same-state basin;
- M1/M2/M3 all match or underperform origin;
- clean success drops;
- candidate rank metrics do not improve.

---

## 10. Codex implementation checklist

### Immediate PR: audit only

- [ ] Extend `coverage_audit.py` to v2.
- [ ] Increase sample size option to 1000+.
- [ ] Expose `encoder_feat`, `emb`, `predictor_hidden`, and `pred_emb` if feasible.
- [ ] Compute `amp_P`, `amp_B`, `amp_R`, and `amp_total`.
- [ ] Compute candidate rank metrics.
- [ ] Add `t_grid` and `t_star` calibration metrics.
- [ ] Emit JSON/CSV artifacts and a printed decision table.
- [ ] Do not run training in this PR.

### Second PR: only if audit supports it

- [ ] Implement encoder projector-as-transport training.
- [ ] Add `project_features()` helper for `P(H(o)+eps)`.
- [ ] Add latent_z / predictor / diagnostic objective modes.
- [ ] Run only TwoRoom weak Gaussian first.

### Third PR: optional

- [ ] Implement predictor-projector plateau if v2 audit indicates `pred_proj` amplification.
- [ ] Implement t-conditioned FM only after oracle t upper bound looks useful.

Suggested commit message for immediate PR:

```text
Extend ACPC-Flow coverage audit for projector and t-calibration

- Add encoder/projector/predictor/pred_proj shift decomposition
- Add amplification and candidate-rank metrics
- Add t-grid calibration for time-conditioned FM feasibility
- Keep training disabled until audit supports a path
```
