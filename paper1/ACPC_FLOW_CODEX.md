# ACPC-Flow / Predictive Plateau Transport: theory, t-conditioned FM audit, and execution plan

This document is the current working plan for the ACPC-Flow direction after the first core coverage audit. It is written for Codex execution.

The short conclusion is:

> Do not train post-projector clean-only latent-noise ACPC-Flow at scale. The core64 and v2 audits reject frozen synthetic local-noise repair (`emb + epsilon -> clean emb`, or analogous repair at `encoder_feat`, `predictor_hidden`, `pred_emb`). That closes post-hoc transport on a trained baseline, but it does **not** close training-time projector distribution migration. The four-task origin-vs-noise v2 audit shows that ordinary input-noise training reshapes the `P/R` path and candidate rankings relative to origin, and those movements align with ATR/SMPR/eval for matched Gaussian and with task-dependent held-out blur/resize evidence. Treat `no_go` labels as strict method gates for frozen synthetic repair, not as paper-facing conclusions about whether noise training improved robustness.

> Do not route this document's next step to robust CEM, noisy-only in-forward control, or pixel/paired-source ablations. Robust CEM has a separate no-go record (`paper1/ROBUST_CEM_EVAL100X3_ITERATION_LOG_20260705.md`), and the Paper2 direct-regularization/data-path controls are recorded elsewhere (`paper1/ROBUSTNESS_TRIAGE_NEXT_STEP_PLAN_20260628.md`, `experiments.md`).

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

### v2 audit update: four-level fixed-checkpoint repair is no-go

New artifacts:

```text
assets/paper1_data/acpc_flow_coverage_v2_tworoom_baseline_seed3073_core128_fullstress.json
assets/paper1_data/acpc_flow_coverage_v2_tworoom_baseline_seed3073_core128_fullstress.csv
```

The v2 audit used TwoRoom `baseline_seed3073`, 128 sampled sequences, candidate-rank metrics, amplification metrics, and `t` calibration. It audited:

```text
encoder_feat:      h = H(o)
emb:               z = P(h)
predictor_hidden:  u = B(z, a) before pred_proj
pred_emb:          y = R(u)
```

Decision table summary:

```text
gaussian 0.03: no_go at encoder_feat, emb, predictor_hidden, pred_emb
gaussian 0.05: no_go at encoder_feat, emb, predictor_hidden, pred_emb
gaussian 0.08: no_go at encoder_feat, emb, predictor_hidden, pred_emb
blur k7:       no_go at encoder_feat, emb, predictor_hidden, pred_emb
resize 0.5:    no_go at encoder_feat, emb, predictor_hidden, pred_emb
```

Important readouts:

```text
amp_P_q90 is high across stressors (~5.1--6.0), so the encoder projector P
strongly amplifies pixel-induced shifts in this trained baseline.

This does not open an encoder-projector training gate, because the same audit
shows crossing and coverage failures before a local synthetic-noise repair can
be trusted. Gaussian 0.03 already has wrong_nn ~= 0.193 at encoder_feat and
wrong_nn ~= 0.497 at pred_emb. Gaussian 0.08, blur, and resize are severe
no-go cases across all levels.

Candidate rank is genuinely affected: top1 flip is ~=0.20 for Gaussian 0.03,
~=0.45 for Gaussian 0.05, ~=0.66 for Gaussian 0.08, ~=0.83 for blur, and
~=0.79 for resize.

`t` calibration is not separable for any audited stressor/level. Most
radius/acpc uncovered rates are near 1.0 outside the weakest encoder_feat
Gaussian 0.03 case.
```

### v2 cross-checkpoint update: noise training reshapes the P/R path

New artifacts:

```text
assets/paper1_data/acpc_flow_coverage_v2_tworoom_noise008_seed3073_core128_fullstress.json
assets/paper1_data/acpc_flow_coverage_v2_tworoom_noise008_seed3073_core128_fullstress.csv
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.json
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.md
```

The same v2 audit was run on origin and ordinary input-noise-trained seed3073
checkpoints for all four tasks. The aligned four-task summary uses stressors that
match the existing Paper1 eval artifacts:

```text
Tasks:       TwoRoom, Reacher, PushT, Cube
Checkpoints: origin baseline seed3073 and noise_0to008_p1 seed3073
Stressors:   Gaussian 0.03/0.05/0.08, blur ks15, resize factor0.25
Readout:     continuous origin -> noise movement, plus eval and diagnostics
```

Important interpretation rule:

```text
strict_gate_label / no_go = can we train frozen synthetic local repair here?
continuous movement      = did training reshape the P/R path and planner-facing ranks?
```

For paper planning, the second readout is the mechanism evidence. The strict gate
still blocks post-hoc synthetic repair, but it should not be used to deny
relative robustness improvements from ordinary training.

Matched Gaussian 0.08 is now aligned across all four tasks:

```text
Task      eval pixels_std0.08  ATR q90        SMPR          amp_P q90     emb wrongNN   pred wrongNN  top1 flip    top5 overlap
TwoRoom   68.8 -> 97.1         1.509 -> 0.111 0.339 -> 0.989 5.192 -> 1.358 0.828 -> 0.003 0.906 -> 0.042 0.664 -> 0.016 0.534 -> 0.977
Reacher   18.2 -> 81.6         2.628 -> 0.082 0.733 -> 0.997 2.926 -> 1.031 0.930 -> 0.003 0.927 -> 0.000 0.797 -> 0.008 0.516 -> 0.980
PushT      7.2 -> 85.8         3.580 -> 0.247 0.439 -> 1.000 1.545 -> 1.140 0.732 -> 0.000 0.711 -> 0.003 0.867 -> 0.023 0.364 -> 0.970
Cube      43.1 -> 62.6         2.320 -> 0.100 0.453 -> 1.000 2.012 -> 1.207 0.932 -> 0.000 0.932 -> 0.000 0.859 -> 0.000 0.413 -> 0.988
```

Held-out blur/resize is task-dependent, not globally negative:

```text
TwoRoom blur k15:    eval 47.7 -> 83.7, Phase-0 diagnostics 5/5 improve,
                     v2 primary metrics 6/6 improve.
TwoRoom resize 0.25: eval 44.7 -> 84.7, v2 primary metrics 6/6 improve.
Reacher blur k15:    eval 19.7 -> 72.0, Phase-0 diagnostics 5/5 improve,
                     v2 primary metrics 6/6 improve.
Reacher resize 0.25: eval 38.3 -> 78.3, v2 primary metrics 6/6 improve.
PushT blur k15:      eval 51.3 -> 58.7, v2 primary metrics 6/6 improve.
PushT resize 0.25:   eval 43.7 -> 68.0, Phase-0 diagnostics 5/5 improve,
                     v2 primary metrics 6/6 improve.
Cube blur k15:       eval 56.0 -> 58.3, but rank-side v2 movement is mixed/worse.
Cube resize 0.25:    eval 57.7 -> 56.3, Phase-0 diagnostics 0/5 improve,
                     rank-side v2 movement is mixed/worse.
```

Current reading:

> Ordinary noise training provides positive evidence that the `P/R` path is
> trainably shapeable. For matched Gaussian this is strong across all four tasks.
> For blur/resize, the scope must remain task/stressor-specific: TwoRoom and
> Reacher align with eval and diagnostics, PushT is positive especially on
> resize, and Cube is the boundary case. We should keep this as a shared analysis
> object rather than turning it into a broad go/no-go conclusion.

This update still does not reopen time-conditioned FM: the available `t_start`
calibration is not a usable non-oracle route for the current method.

### Immediate conclusion

Do not train a post-hoc ACPC-Flow adapter, fixed-checkpoint projector repair, or
time-conditioned FM from synthetic local latent/feature noise for the origin
baseline. The fixed-checkpoint audit answered the previous open questions:

1. `P` does amplify pixel-induced shifts in the origin baseline.
2. `R` can add mild amplification, but `P` is the dominant origin amplifier.
3. Candidate rankings are affected, especially at stronger Gaussian, blur, and resize stressors.
4. Non-oracle `t_start` is not supported by the calibration audit.

The four-task origin-vs-noise audit answers the next question more carefully:
from-scratch training can reshape the `P/R` path and planner-facing candidate
rankings. This is strongest and cleanest for matched Gaussian, and it is also
consistent with held-out blur/resize improvements on TwoRoom/Reacher and PushT
where eval and diagnostics move in the same direction. Cube remains a boundary
case. Therefore the live ACPC-Flow hypothesis is no longer frozen repair; it is
training-time P/R distribution migration with real corrupted-view pressure and
discriminability guards, to be analyzed by task/stressor rather than by a single
binary label.

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

### 4.4 Fixed-checkpoint repair vs. from-scratch projector migration

The v2 audit is a **fixed-checkpoint repair audit**. It asks whether a trained
baseline's corrupted representations are close enough to clean representations
that a synthetic local-noise transport can repair them after the fact. The answer
for TwoRoom `baseline_seed3073` is no.

This is not the same as asking whether training can shape the representation
path. A from-scratch projector-migration method would change the training
dynamics of `P` and/or `R` while the encoder/predictor are still forming their
basins. The plausible hypothesis is:

```text
clean/corrupted pixel views -> H -> P should learn a local same-state predictive plateau
B -> R should avoid turning residual nuisance shift into planner-facing candidate-rank flips
```

Therefore the current fixed-repair result should be read as:

```text
no:  post-projector synthetic-noise transport on a frozen baseline
no:  fixed-checkpoint local repair when the audit already shows crossing
maybe: training-time distribution migration through the original P/R projectors,
       using real corrupted-view pressure and discriminability guards
```

That analysis is now partially done in the four-task aligned summary. Ordinary
noise training lowers `amp_P`, same-state crossing proxies, and candidate-rank
flips for matched Gaussian across all four tasks; held-out blur/resize shows
aligned positive movement on TwoRoom/Reacher and PushT, with Cube as the current
boundary. The next decision is not whether to train a frozen adapter, but which
from-scratch `P` or `R` intervention would isolate this mechanism without merely
reproducing ordinary input-noise training.

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

`pixel_paired_source_candidate` is retained here as a legacy audit-schema label,
not as the current recommended backup route. Existing paired/pixel-source
training analyses are negative or out of scope for this ACPC-Flow plan; do not
reopen them without a separate positive artifact.

---

## 7. Training roadmap after audit v2

### Stage A: current fixed-checkpoint ACPC-Flow is closed

The v2 full-stress audit is no-go at all four levels. For TwoRoom
`baseline_seed3073`, do not train any of the following as post-hoc repair:

```text
post-projector residual transport
encoder_feat synthetic-noise repair
predictor_hidden synthetic-noise repair
time-conditioned FM with synthetic local noise
```

This closes the previous immediate training roadmap for this checkpoint.

### Stage B: projector-migration feasibility audit across checkpoints

Status: completed for seed3073 with aligned stressors.

```text
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.json
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.md
```

The comparison joins three signals, not only v2 geometry:

```text
1. v2 projector/rank movement:
   amp_P_q90, amp_R_q90, emb/pred wrong_nn, candidate_top1_flip, top-k overlap

2. paper-facing diagnostic trend:
   ATR and SMPR for matched Gaussian; Phase-0 ACPC-H/MAF/CRA/PCC directional
   checks for held-out cases where those diagnostics are available.

3. closed-loop eval trend:
   origin/noise success under the matched stressor from existing eval artifacts.
```

Current reading:

- Matched Gaussian: all four tasks show eval up, ATR down, SMPR up, and 6/6 v2
  primary metrics improving. This is strong evidence that training can shape the
  `P/R` path and reduce planner-facing rank instability.
- TwoRoom/Reacher blur and resize: eval improves strongly, available Phase-0
  diagnostics improve on blur, and v2 continuous metrics improve. This supports
  blur/resize transfer for these tasks even when strict synthetic-repair gates
  remain unfavorable.
- PushT: resize is aligned across eval, Phase-0 diagnostics, and v2; blur has
  mild eval gain and v2 improvement but lacks the Phase-0 diagnostic subset.
- Cube: resize is a boundary/negative control with eval slightly down, Phase-0
  diagnostics 0/5, and mixed rank-side v2 movement. Blur has slight eval gain but
  mixed/worse rank-side v2 movement, so it should remain uncertain.

Do not collapse this into a single blur/resize verdict. The useful conclusion is
that ordinary training can reshape projector/rank behavior, with transfer scope
that depends on task and stressor.

### Stage C: candidate from-scratch single-projector MVE, pending joint analysis

If we decide to test a method after analyzing Stage B, run a small from-scratch
MVE rather than a frozen-checkpoint adapter. Start with one side for attribution:

```text
P-only distribution migration: real clean/corrupted pixel pairs -> P plateau
R-only predictive plateau: real clean/corrupted branches -> R output stability
```

Do not start with both projectors unless one-sided evidence is already positive.
The objective must use real corrupted-view pressure or an audited source that
matches it; synthetic local latent noise alone is not enough.

Candidate slices to discuss before training:

```text
Matched Gaussian all-task slice:
  strongest evidence; best for proving P/R path is trainably shapeable.

TwoRoom/Reacher blur/resize slice:
  best for showing transfer beyond matched Gaussian, but claims must be scoped.

Cube boundary slice:
  useful as a falsification/control case; not the first method target.
```

Minimum gates for any MVE:

```text
clean drop <= 5 pp
corrupted behavior improves over origin on the target slice
v2 continuous metrics improve relative to origin
ATR/SMPR or Phase-0 diagnostics move in the same direction as eval
SMPR / task-discriminability guard does not degrade
beats ordinary noise training only if making a method-strength claim; otherwise
explain exactly what mechanism it isolates or simplifies
```

### Routes not reopened here

Do not list the following as current next steps in this ACPC-Flow document:

```text
robust CEM system eval
noisy-only in-forward control
pixel/paired-source ablation as an unqualified next step
```

Robust CEM has a separate no-go record in
`paper1/ROBUST_CEM_EVAL100X3_ITERATION_LOG_20260705.md`. The older Paper2
regularization/data-path controls are recorded in `experiments.md`,
`paper2/PLAN.md`, and `paper1/ROBUSTNESS_TRIAGE_NEXT_STEP_PLAN_20260628.md`;
they should not be mixed into the ACPC-Flow next-step plan.

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

## 9. Promotion and stop criteria

Promote any ACPC-Flow method only if:

1. V2 continuous movement supports the claimed task/stressor/level.
2. Offline diagnostics move in the same direction as eval: ATR down and SMPR not
   down for matched Gaussian, or ACPC-H/MAF/PCC down and CRA/elite overlap up
   for held-out Phase-0 cases.
3. Candidate rank flip/top-k overlap improves.
4. Clean performance remains within 5 pp.
5. Small closed-loop eval improves over origin on the target stressor.
6. The method beats same-parameter identity/random controls.
7. For t-conditioned FM, estimated/non-oracle `t_start` works; oracle-only
   success is insufficient.

Stop or hold a route if:

- eval, diagnostics, and v2 movement disagree and the mechanism cannot be
  isolated;
- pure synthetic noise only covers radius by leaving the same-state basin;
- required `t_start` is large and causes neighborhood crossing;
- clean success drops beyond the planned tolerance;
- candidate rank metrics do not improve;
- M1/M2/M3 all match or underperform origin.

Important caveat:

> A strict v2 `no_go` label stops frozen synthetic local repair. By itself, it
> does not stop a from-scratch P/R migration hypothesis when eval, diagnostics,
> and continuous v2 origin->noise movement show relative improvement.

## 10. Codex implementation checklist

### Completed audit step

- [x] Extend `coverage_audit.py` to v2.
- [x] Expose `encoder_feat`, `emb`, `predictor_hidden`, and `pred_emb`.
- [x] Compute `amp_P`, `amp_B`, `amp_R`, and `amp_total`.
- [x] Compute candidate rank metrics.
- [x] Add `t_grid` and `t_star` calibration metrics.
- [x] Emit JSON/CSV artifacts and a printed decision table.
- [x] Keep training disabled until audit supports a path.

Current key artifact:

```text
assets/paper1_data/acpc_flow_coverage_v2_tworoom_baseline_seed3073_core128_fullstress.json
assets/paper1_data/acpc_flow_coverage_v2_tworoom_baseline_seed3073_core128_fullstress.csv
```

### Current ACPC-Flow analysis step

- [x] Run the same v2 audit on ordinary noise-trained TwoRoom seed3073 std0.08.
- [x] Compare origin vs noise-trained `amp_P`, `amp_R`, crossing, candidate-rank
      flip, top-k overlap, and `t` calibration.
- [x] Decide whether from-scratch P/R distribution migration is plausible for
      matched Gaussian.
- [x] Run the aligned four-task origin-vs-noise v2 audit.
- [x] Join v2 movement with ATR/SMPR or corresponding Phase-0 diagnostics and
      closed-loop eval scores.
- [ ] Jointly inspect the four-task summary and choose whether the first MVE
      should target matched Gaussian, TwoRoom/Reacher held-out transfer, or a
      falsification/control slice.
- [ ] Optional before training: repeat on seed3074 or a std sweep to separate
      projector-path repair from checkpoint-specific variance.

Current summary artifacts:

```text
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.json
assets/paper1_data/acpc_flow_v2_four_task_origin_vs_noise008_aligned_summary.md
```

### Training only if the cross-checkpoint audit supports it

- [ ] Implement P-only from-scratch projector distribution migration.
- [ ] Implement R-only from-scratch predictive plateau.
- [ ] Do not implement a frozen post-projector adapter.
- [ ] Do not implement t-conditioned FM unless non-oracle `t_start` becomes
      separable in a later audit.
- [ ] Do not start two-projector training until one-sided evidence is positive.

Suggested commit message for the audit update:

```text
Record ACPC-Flow v2 audits and projector-migration evidence

- Add four-task origin-vs-noise aligned audit summary
- Join v2 projector/rank movement with eval and diagnostics
- Treat strict no_go labels as frozen-repair gates, not global conclusions
- Keep from-scratch P/R migration as the current hypothesis under joint review
```
