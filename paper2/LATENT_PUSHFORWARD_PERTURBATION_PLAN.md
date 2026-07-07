# Latent Pushforward Perturbation Plan

Status: Paper2 method/theory planning note.  
Branch: `ag/dev`.  
Date: 2026-07-07.  
Scope: diagnosis-guided latent perturbation design after Paper1 ACPC diagnostics and the negative train/planner controls.

This note records a new method hypothesis:

> Pixel-space corruptions do not push JEPA world-model representations like small isotropic Gaussian latent noise. They induce layer-, task-, token-, and corruption-family-dependent pushforward shifts. A useful latent perturbation method must match this pushforward geometry while preserving same-state predictive basins and task/action discriminability.

The immediate goal is not to train another loss. The immediate goal is to build a **pushforward noise geometry audit** that decides whether latent perturbation training is feasible, which layer should be perturbed, and which noise family is safe.

---

## 0. Current empirical context

### Paper1 status

Paper1 is now best treated as a diagnostic paper, not a method paper. The current arXiv-v1 identity is:

> a controlled diagnostic study of matched-Gaussian visual robustness in JEPA latent world-model checkpoints.

The next scientific step can become method work only if it establishes a causal diagnostic-to-behavior chain with proper baselines.

### Positive evidence available

The three-training-seed Gaussian lockbox is complete. It strongly reproduces the Paper1 pattern:

- no-noise LeWM checkpoints suffer large observation-noise cliffs;
- input-side Gaussian noise training recovers performance into broad task-dependent plateaus;
- recovered endpoints show large reductions in multi-step clean/noisy predictor drift;
- high clean/noisy CKA appears at the recovered endpoints;
- strongest-only blur/resize transfer is bounded: TwoRoom/Reacher positive, PushT weak/mixed, Cube neutral/negative.

The diagnostic validation artifacts show that ACPC/PCC/CRA/MAF readouts are useful as plateau localization / triage signals, especially for reduced Gaussian drop. They are not yet a standalone robustness oracle or exact checkpoint selector.

### Negative evidence available

Several direct conversions of the diagnostic into a method have failed or are not strong enough:

- generic encoder-level clean/noisy consistency (GLC) failed;
- one-step SNAP-ACPC failed;
- paired no-aux equivalence control failed;
- robust CEM / final rerank / rank vote / inner-loop robust scoring is weak/no-go;
- heteroscedastic error reweighting is unsafe because hard transitions can be action-relevant;
- fixed temporal hinge is not the right dynamics prior.

The main lesson is:

> Do not add another consistency loss or planner-side selector until the upstream perturbation geometry is understood.

---

## 1. Why the latent perturbation question is central

The old ACPC-Flow variant implicitly assumed:

```text
source = clean_latent + small isotropic-ish Gaussian noise
target = clean_latent
```

The coverage audit rejected this for post-projector `emb`: small synthetic latent noise did not cover even weak pixel Gaussian corruption, and larger noise risks crossing task-distinct neighborhoods.

This does **not** prove that latent perturbation training is impossible. It proves that the wrong latent noise family was being tested.

The missing variable is the **pushforward geometry** of pixel perturbations through the encoder/projector/predictor chain:

```text
pixels -> encoder H -> encoder projector P -> emb z -> predictor B -> pred_proj R -> pred_emb y
```

Different pixel perturbations can induce different latent shift structures:

- anisotropic covariance;
- low effective rank;
- non-zero mean drift;
- token-dependent scale;
- task-dependent safe radius;
- corruption-family-specific subspaces;
- predictor-facing amplification at `P`, `B`, or `R`.

Therefore the key question is:

> Can we design a latent perturbation distribution whose support covers measured pixel-induced feature-shift directions without crossing task/action-distinct basins and without breaking planner-facing candidate rankings?

If yes, this can become a high-contribution method. If no, the audit provides a principled no-go and prevents another expensive training dead end.

---

## 2. Core theoretical formulation

For a representation level `l`, define:

```text
r_l(o) in {encoder_feat h, emb z, predictor_hidden u, pred_emb y}
Delta_l,tau(o) = r_l(tau(o)) - r_l(o)
```

where `tau` is a pixel perturbation family such as Gaussian noise, blur, resize, brightness/contrast, compression, occlusion, or camera shift.

The empirical pushforward set is:

```text
S_l(T) = { Delta_l,tau(o_i) : o_i ~ data, tau in T }
```

A candidate latent perturbation distribution `Q_l(o)` is useful only if it satisfies three constraints.

### 2.1 Coverage

`Q_l` must cover the measured pixel-induced shifts:

```text
Delta_l,tau(o) should lie inside a high-probability tube of Q_l(o)
```

Operational example:

```text
mahalanobis_l,tau(o) = (Delta - mu_Q)^T Sigma_Q^{-1} (Delta - mu_Q)
coverage_q95 = Pr[ mahalanobis_l,tau(o) <= chi2_or_empirical_q95 ]
```

### 2.2 Non-crossing / basin safety

Perturbed representations sampled from `Q_l` must remain in the same-state predictive basin:

```text
crossing_rate = Pr_{eps ~ Q_l(o)}[ nearest_clean(r_l(o) + eps) is task-distinct ]
```

or, preferably:

```text
SMPR_after_noise remains high
```

A noise distribution that covers pixel shifts only by crossing task-distinct neighborhoods is not valid for control.

### 2.3 Planner-facing relevance

The covered directions must matter for action-conditioned prediction or candidate ranking. A large representation shift is not automatically a control failure.

Planner-facing tests:

```text
ACPC_gap_l(o, tau, a)
candidate_top1_flip_rate
candidate_topk_overlap
candidate_rank_spearman
candidate_margin_clean_q10/q50
```

Training should target only perturbation directions that are both:

1. induced by real pixel shifts;
2. relevant to predictive drift or candidate ranking.

---

## 3. Local theory: why isotropic Gaussian is usually wrong

Under local linearization,

```text
Delta_H(o) = H(o + xi) - H(o) ~= J_H(o) xi
```

For pixel Gaussian noise:

```text
xi ~ N(0, sigma_x^2 I)
Delta_H ~ N(0, sigma_x^2 J_H J_H^T)
```

So even pixel Gaussian noise becomes an anisotropic latent distribution. Blur, resize, compression, occlusion, and brightness changes are even less likely to be zero-mean isotropic after encoding.

An isotropic latent noise distribution:

```text
eps ~ N(0, sigma_z^2 I)
```

can cover a large shift direction only by injecting noise into all dimensions. In high dimension this causes large total radius:

```text
E||eps|| ~ sigma_z sqrt(d)
```

Thus isotropic noise can simultaneously under-cover real nuisance directions and over-cover irrelevant directions, increasing crossing risk.

### Candidate proposition 1: pushforward covariance condition

For local Gaussian pixel perturbations, a latent Gaussian `Q_l = N(0, Sigma_Q)` can cover the pixel pushforward only if approximately:

```text
Sigma_Q >= c * E_o[J_l(o) Sigma_x J_l(o)^T]
```

in positive-semidefinite order, for a coverage factor `c` determined by the desired quantile.

### Candidate proposition 2: isotropic radius conflict

If `Sigma_push` has top eigenvalue `lambda_max` and effective dimension `d_eff << d`, isotropic covariance large enough to cover `lambda_max` has radius scaling with `sqrt(d)`, while the true pushforward radius scales with `sqrt(trace(Sigma_push))`. When `d_eff << d`, isotropic coverage wastes radius in irrelevant directions and can violate the non-crossing margin.

### Candidate proposition 3: feasible perturbation criterion

Let `m_l(o)` be the clean same-state basin margin to nearest task-distinct clean state at level `l`. A candidate `Q_l` is feasible for stressor family `tau` only if:

```text
coverage_q95(Q_l, Delta_l,tau) >= coverage_min
and
Pr_{eps ~ Q_l}[ ||eps|| or semantic crossing exceeds m_l(o) ] <= crossing_max
```

If no `Q_l` in the candidate family satisfies both, that latent perturbation family is no-go at level `l`.

### Candidate proposition 4: candidate rank flip bound

For candidate costs `J_j(z)`, local linearization gives:

```text
J_j(z + eps) - J_i(z + eps)
  ~= margin_ji + (grad J_j - grad J_i)^T eps
```

If:

```text
margin_ji <= c * sqrt((grad J_j - grad J_i)^T Sigma_Q (grad J_j - grad J_i))
```

then rank flip risk is high under `Q_l`. This links covariance geometry to the Paper1 candidate-rank diagnostics.

---

## 4. Candidate latent perturbation families

Do not jump directly to training. Audit these families first.

### Family A: isotropic scalar noise

```text
eps ~ N(0, sigma^2 I)
```

Role: baseline and negative control.

Expected reading: likely no-go for post-projector `emb` except perhaps very weak Gaussian / local sensor noise.

### Family B: diagonal anisotropic noise

```text
eps ~ N(0, diag(s_1^2, ..., s_d^2))
```

Estimate `s_j` from empirical pixel-pushforward deltas or from a training-only perturbation bank.

Role: tests whether per-dimension scale fixes the isotropic mismatch.

Risk: cannot model correlated directions; can still waste radius if true shifts are low-rank correlated.

### Family C: low-rank plus diagonal noise

```text
eps = U_r Lambda_r^{1/2} eta + sigma_floor delta
eta ~ N(0, I_r)
delta ~ N(0, I_d)
```

where `U_r, Lambda_r` come from the top eigenvectors/eigenvalues of measured `Delta_l,tau` covariance.

Role: primary candidate if pushforward deltas have low effective rank.

Expected advantage: covers dominant corruption directions with much smaller total radius than isotropic noise.

### Family D: mean-shifted family-specific noise

```text
eps = mu_tau + U_tau Lambda_tau^{1/2} eta + sigma_floor delta
```

Role: needed if blur/resize/brightness induce non-zero mean drift.

Risk: if used as a main method, it becomes corruption-family-aware unless the family is inferred non-oracularly.

### Family E: mixture of latent perturbation families

```text
Q_l = sum_k pi_k Q_l,k
```

where `k` can index synthetic stressor families or latent clusters.

Role: covers multiple perturbation types without forcing one covariance to explain all shifts.

Risk: mixture component selection at inference/training must not use oracle test corruption labels for the main claim.

### Family F: token/state-conditioned covariance

```text
Q_l(o, t_token) = N(mu(o,t), Sigma(o,t))
```

Role: long-term candidate if global covariance is too coarse.

Risk: complexity; can overfit; requires strong no-collapse and clean guards.

### Family G: pixel-paired source upper bound

```text
source = r_l(tau(o))
target = r_l(o)
```

Role: upper bound / diagnostic. It tests whether paired transport can help at all when the exact pixel-induced shift is available during training.

Not a clean-only latent noise method. It must be framed separately.

---

## 5. Pushforward Noise Geometry Audit

Create:

```text
tools/acpc_flow/pushforward_noise_audit.py
```

Required artifacts:

```text
assets/paper2_data/latent_pushforward_audit_<task>_<checkpoint>_<date>.json
assets/paper2_data/latent_pushforward_audit_<task>_<checkpoint>_<date>.csv
assets/paper2_data/latent_pushforward_audit_summary_<date>.md
```

If `assets/paper2_data/` does not exist, create it.

### 5.1 Tasks and checkpoints

Minimum first pass:

```text
tasks: TwoRoom, Reacher
checkpoints:
  - origin baseline_seed3073
  - Gaussian noise-trained std0.08 seed3073
```

Second pass if the first pass is informative:

```text
tasks: PushT, Cube
training seeds: 3072/3073/3074 representative checkpoints
```

### 5.2 Perturbation families

First pass:

```text
gaussian_noise: std 0.03, 0.05, 0.08
gaussian_blur: k 7 or strongest k 15
resize: 0.5 or strongest 0.25
```

Later:

```text
brightness/contrast
compression
occlusion/cutout
camera/background shift if supported
```

### 5.3 Representation levels

Compute clean/corrupted deltas at:

```text
encoder_feat      h = H(o)
emb               z = P(h)
predictor_hidden  u = B(z, a) before pred_proj
pred_emb          y = R(u)
```

If `predictor_hidden` is not exposed, add a hook or helper.

### 5.4 Metrics

For each task/checkpoint/stressor/level:

```text
# shift scale
delta_norm_mean/median/q90/q95
ratio_to_clean_knn_q50/q90/q95

# covariance geometry
mean_shift_norm
cov_trace
cov_effective_rank
lambda_max_over_trace
top1/top5/top10_energy
diagonal_energy_ratio
offdiag_energy_ratio
family_subspace_overlap

# candidate Q coverage
coverage_isotropic_q95
coverage_diag_q95
coverage_lowrank_r{1,2,4,8,16}_q95
coverage_mixture_q95
mahalanobis_q50/q90/q95
required_isotropic_std
required_lowrank_rank_for_coverage

# safety
wrong_nn_rate
closer_to_wrong_than_pair_rate
crossing_rate_isotropic
crossing_rate_diag
crossing_rate_lowrank
safe_radius_q95
coverage_safe_conflict

# layer amplification
amp_P
amp_B
amp_R
amp_total

# planner-facing relevance
ACPC_gap_q90/q95
candidate_rank_spearman
candidate_top1_flip_rate
candidate_topk_overlap
candidate_margin_clean_q10/q50
rank_flip_bound_mean/q90

# decision
decision
recommended_next_action
```

### 5.5 Decision labels

```text
no_go
isotropic_no_go
diagonal_candidate
lowrank_diag_candidate
family_mixture_candidate
projector_plateau_candidate
predictor_projector_candidate
pixel_paired_upper_bound_only
needs_semantic_guard
```

### 5.6 Go/no-go thresholds for first pass

Initial thresholds, to be revised after scale inspection:

```text
coverage_min = 0.80 at q95 tube
crossing_max = 0.05 to 0.10
clean_false_positive_max = 0.05
candidate_top1_flip_improvement_required = measurable vs isotropic/no-noise baseline
clean_drop_allowed = <= 5 pp in later training MVE
```

Do not treat these as theory constants. They are audit gates.

---

## 6. Experiment stages after the audit

### Stage 0: audit only, no training

Run `pushforward_noise_audit.py` and produce the decision table.

Stop if all candidate noise families fail coverage/safety.

### Stage 1: offline perturbation replay

Before training, simulate perturbations at a chosen level and measure whether synthetic structured latent perturbations reproduce measured pixel-induced effects:

```text
r_l_synth = r_l_clean + eps_structured
compare against r_l_pixel = r_l(tau(o))
```

Readouts:

```text
Delta direction cosine
Delta norm ratio
ACPC gap match
candidate rank metric match
wrong-neighborhood rate
```

Goal: show that the proposed latent perturbation actually mimics pixel-pushforward directions better than isotropic noise.

### Stage 2: one-task training MVE

Run only if Stage 0/1 pass.

Candidate first MVE:

```text
task: TwoRoom or Reacher
checkpoint family: train from scratch under matched seed
perturbation target: Gaussian 0.03/0.05/0.08 first
noise family: lowrank+diag at encoder_feat or emb, depending on audit
objective: predictive plateau with discriminability logging
baseline: origin, ordinary pixel noise training, isotropic latent noise, random low-rank subspace
```

Do not start with PushT or blur/resize as the main success target.

### Stage 3: broaden within Gaussian

If Stage 2 works:

```text
training seeds: 3072/3073/3074
tasks: TwoRoom, Reacher, then PushT/Cube
metrics: behavior, ATR/ACPC, SMPR, candidate rank, clean guard
```

### Stage 4: bounded unseen perturbation transfer

Only after matched Gaussian succeeds:

```text
families: blur, resize, brightness/contrast
claim: bounded transfer, not universal perturbation robustness
```

---

## 7. Training objective candidates

Only after the audit supports a feasible noise family.

### Objective 1: encoder-feature structured perturbation through projector

```text
h = H(o)
eps ~ Q_H
z_source = P(h + eps)
z_clean = sg(P(h))
L_z = ||z_source - z_clean||^2
```

Use only if `encoder_feat` is safe and `amp_P` suggests the projector is an amplifier.

### Objective 2: predictor-facing plateau

```text
z_source = z + eps_structured
pred_source = F(z_source, a)
pred_clean = sg(F(z, a))
L_pred = ||pred_source - pred_clean||^2
```

Use only if structured perturbations match pixel-induced predictive drift and do not cross semantic margins.

### Objective 3: pred_proj plateau

```text
u_source = B(z_source, a)
y_source = R(u_source)
y_clean = sg(R(B(z, a)))
L_R = ||y_source - y_clean||^2
```

Use only if `amp_R` or rank instability appears after `pred_proj`.

### Objective 4: hybrid ACPC plateau with guard logging

```text
L = L_pred_or_R + lambda_z L_z
```

Log, but do not initially optimize, discriminability guards:

```text
SMPR
transition resolution
inverse dynamics probe
candidate margin/rank metrics
effective rank/local NN structure
```

Only add explicit guard losses after isolated tests show collapse risk.

---

## 8. Critical risks and objections

### 8.1 Matching pixel shift is not automatically good

Some pixel-induced latent shifts may carry task-relevant information, especially in contact or near-boundary states. A method that cancels all measured shift can destroy useful sensitivity.

Guard:

```text
same-state perturbation pairs -> consistency
task/action-distinct pairs -> separation
```

### 8.2 Structured covariance can overfit the evaluated corruption

If covariance is estimated from the exact test stressor, the method becomes corruption-aware. This is acceptable for an upper bound or analysis, but not for a corruption-agnostic main claim.

Mitigation:

- distinguish oracle-family covariance from non-oracle training covariance;
- train/evaluate on held-out severity or held-out perturbation family;
- report oracle vs non-oracle clearly.

### 8.3 Diagonal noise may be insufficient

If off-diagonal covariance is high or top-k eigenvectors explain most energy, diagonal noise may still waste radius and cross basins.

Required audit:

```text
offdiag_energy_ratio
topk_energy
coverage_vs_crossing for diagonal vs lowrank
```

### 8.4 Low-rank noise may miss rare but important directions

Low-rank coverage can improve average safety but miss tail cases. Tail ACPC/rank metrics must be reported.

### 8.5 Behavior still must beat ordinary noise training

The empirical bar is ordinary pixel noise training under matched settings. A latent perturbation method is interesting only if it:

1. matches or improves corrupted behavior at similar or lower cost;
2. preserves clean behavior;
3. gives a clearer mechanism or transfer story;
4. beats isotropic latent noise and random low-rank controls.

---

## 9. Contribution potential

If successful, this can be a high-contribution method because it is not simply “add latent noise.” The contribution would be:

1. **Theory:** pixel perturbations induce anisotropic pushforward distributions in latent/predictive spaces; isotropic latent noise has a coverage-vs-crossing conflict.
2. **Diagnostic audit:** a frozen method to measure pushforward covariance, layer amplification, non-crossing, and planner-facing rank sensitivity before training.
3. **Method:** pushforward-calibrated structured latent perturbation / predictive plateau training.
4. **Evidence:** structured latent perturbations mimic measured pixel-induced feature-shift directions better than isotropic noise, and improve behavior under matched Gaussian plus bounded unseen stressors.
5. **Negative clarity:** planner-side CEM reranking and naive consistency losses are insufficient, motivating upstream geometry-aware repair.

The contribution is high only if the method clears the matched baselines. If it only explains why isotropic latent noise fails, it is still valuable as an analysis/diagnostic contribution but not a full method paper.

---

## 10. Immediate Codex task list

### PR-A: audit implementation only

- [ ] Add `tools/acpc_flow/pushforward_noise_audit.py`.
- [ ] Add `assets/paper2_data/` output path if needed.
- [ ] Reuse existing encoder / corruption / representation-analysis utilities.
- [ ] Expose or hook `predictor_hidden` before `pred_proj` if feasible.
- [ ] Compute pushforward delta covariance and anisotropy metrics.
- [ ] Compute isotropic / diagonal / low-rank+diag coverage metrics.
- [ ] Compute non-crossing metrics using clean kNN and available state/semantic proxies.
- [ ] Compute layer amplification metrics `amp_P`, `amp_B`, `amp_R`, `amp_total`.
- [ ] Compute candidate-rank metrics where candidate cost code is available.
- [ ] Emit JSON/CSV plus compact markdown summary.
- [ ] Do not train in this PR.

### PR-B: offline perturbation replay

- [ ] Implement synthetic structured latent replay at selected level.
- [ ] Compare isotropic vs diagonal vs low-rank+diag vs measured pixel deltas.
- [ ] Report direction cosine, norm match, ACPC gap match, and candidate-rank match.
- [ ] Decide whether any noise family is eligible for training.

### PR-C: training MVE only if PR-A/B pass

- [ ] Implement one structured perturbation family behind default-off config.
- [ ] Run one task, one seed, weak-to-medium Gaussian first.
- [ ] Compare against origin, ordinary pixel noise training, isotropic latent noise, and random low-rank control.
- [ ] Require behavior + ACPC + SMPR + rank improvement.

---

## 11. Suggested artifact names

```text
assets/paper2_data/latent_pushforward_audit_tworoom_s3073_YYYYMMDD.json
assets/paper2_data/latent_pushforward_audit_tworoom_s3073_YYYYMMDD.csv
assets/paper2_data/latent_pushforward_audit_reacher_s3073_YYYYMMDD.json
assets/paper2_data/latent_pushforward_audit_summary_YYYYMMDD.md
```

---

## 12. Suggested commit message

```text
Add latent pushforward perturbation planning note

- Frame pixel-corruption pushforward geometry as the missing ACPC-Flow variable
- Define coverage, non-crossing, and planner-facing feasibility gates
- Propose isotropic/diagonal/low-rank/mixture latent perturbation audits
- Add staged audit -> replay -> training MVE roadmap
```
