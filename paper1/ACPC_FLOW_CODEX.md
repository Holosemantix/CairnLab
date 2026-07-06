# ACPC-Flow / Predictive Plateau Transport: method theory, feasibility audit, and Codex implementation plan

This document specifies the next-method direction that follows from Paper1's selective-ACPC diagnostic. It is written for Codex implementation. The goal is a **clean, testable method family**, not a loss soup.

## 0. One-sentence method idea

Paper1 shows that robust LeWM checkpoints occupy a **selective ACPC plateau**: same-state visual perturbations have low action-conditioned predictive tail risk (ATR), while task-grounded different-state pairs remain separated (SMPR). ACPC-Flow learns a **state-paired latent/feature transport map** that moves perturbed/off-manifold representations into the same action-conditioned predictive equivalence class as their clean representations.

Short form:

> Flow/transport supplies the mechanism; ACPC supplies the success criterion.

The main target is **not** marginal distribution matching and not only raw latent closeness. The main target is agreement **after the transported history is rolled out by the predictor under the same action sequence**, in the same diagnostic space that defines ATR.

## 1. Correct conceptual framing

### 1.1 ACPC does not say encoder geometry is unimportant

ACPC should be stated as an end-to-end diagnostic, not as a rejection of encoder analysis.

The correct interpretation is:

> Encoder geometry is a first-stage risk signal. Same-state perturbed views should remain in the same-state predictive basin and should not cross into task-distinct neighborhoods. ACPC then asks whether the remaining encoder/projection shift changes action-conditioned predicted futures, candidate costs, and rankings. SMPR checks that the contraction does not collapse action-relevant distinctions.

Thus there are two coupled dimensions:

1. **Neighborhood consistency / non-crossing**: clean and perturbed representations for the same state should stay in the same predictive basin; they should not become closer to task-distinct states.
2. **Predictive plateau + anti-collapse**: after the same action rollout, transported perturbed histories should match clean histories in diagnostic space, while task-grounded different-state pairs remain separated.

### 1.2 Paper1 diagnostic target

Paper1 defines same-state visual robustness as selective action-conditioned predictive consistency:

\[
z_t = E_\theta(h_t),\qquad \tilde z_t = E_\theta(\tilde h_t),
\]

\[
\hat z_{t+k}=F_\theta^k(z_t,\mathbf a_{0:k-1}),\qquad
\hat{\tilde z}_{t+k}=F_\theta^k(\tilde z_t,\mathbf a_{0:k-1}).
\]

ACPC-H measures the projected rollout discrepancy

\[
\sum_{k=1}^H \alpha_k d\big(\Pi(\hat z_{t+k}),\Pi(\hat{\tilde z}_{t+k})\big).
\]

ATR is the high-tail version of this same-state clean/noisy rollout disagreement. SMPR checks that task-grounded near-boundary pairs remain separated. Therefore a method should aim for:

\[
\text{low ATR} \quad + \quad \text{high SMPR}.
\]

### 1.3 Transport version of the diagnostic

Introduce a transport map \(T_\phi\) applied to a perturbed/off-manifold latent history:

\[
z_{t:t+C-1}^\phi = T_\phi(\tilde z_{t:t+C-1}),
\]

where \(C=\texttt{ctx_len}\) is the predictor history length. The diagnostic-space transport target is:

\[
\Pi(F_\theta^{1:H}(T_\phi(\tilde z_{t:t+C-1}),\mathbf a))
\approx
\Pi(F_\theta^{1:H}(z_{t:t+C-1},\mathbf a)).
\]

This is the key connection: transport success is measured **after action-conditioned rollout of the transported history**, not only in raw latent space.

### 1.4 Candidate-cost stability connection

Paper1's cost-drift and top-1 stability arguments say that if every candidate action sequence has clean/perturbed projected-rollout discrepancy at most \(\epsilon\), then candidate cost drift is at most \(L_J\epsilon\), and a clean top-1 margin \(\Delta>2L_J\epsilon\) preserves the top-1 candidate.

With transport, the relevant discrepancy becomes

\[
\epsilon_\phi
=
d_H\big(
\Pi(F^{1:H}(T_\phi(\tilde z),\mathbf a)),
\Pi(F^{1:H}(z,\mathbf a))
\big).
\]

ACPC-Flow tries to reduce \(\epsilon_\phi\), thereby reducing ATR, cost drift, and candidate-rank flip risk under the same fixed-candidate/margin caveats.

### 1.5 Local sensitivity connection

Paper1's local Gaussian sensitivity shows that the action-conditioned sensitivity is governed by the product

\[
J_{G_\mathbf a}(E(o))J_E(o),\qquad
G_\mathbf a(z)=\Pi(F^{1:H}(z,\mathbf a)).
\]

With a transport map, the effective map becomes

\[
G_\mathbf a(T_\phi(E(o))).
\]

The local sensitivity includes

\[
J_{G_\mathbf a}\,J_{T_\phi}\,J_E.
\]

Thus the desired transport contracts nuisance/corruption-induced representation directions before they reach the predictor, while avoiding contraction along task-relevant directions. This is exactly why SMPR and neighborhood-crossing diagnostics remain necessary.

## 2. What Flow Matching means here

This method should not claim generic image-generation Flow Matching unless the code implements a full time-conditioned ODE/vector field. The safer description is:

> flow-inspired state-paired latent transport.

The transport is **paired/conditional**, not marginal:

\[
\tilde z_i \mapsto z_i \quad \text{or} \quad \tilde z_i \mapsto \mathcal C_{ACPC}(z_i),
\]

where \(\tilde z_i\) and \(z_i\) represent the same underlying state, and \(\mathcal C_{ACPC}(z_i)\) is the clean action-conditioned predictive equivalence class.

Avoid this incorrect objective:

\[
T_\phi\#p_{pert}(z) \approx p_{clean}(z).
\]

Marginal matching can map a perturbed latent from state \(i\) to a clean latent from state \(j\). That may look distributionally clean but is wrong for control.

## 3. Theoretical feasibility of feature/latent perturbation coverage

This section is the precondition for the method. ACPC-Flow should **not** be trained at scale before this coverage audit is run.

### 3.1 Notation

Let \(H(o)\) denote the encoder output **before** the LeWM projection head, e.g. ViT CLS feature. Let \(P\) denote the LeWM projection head. The post-projector latent is:

\[
z=P(H(o)).
\]

A pixel perturbation \(\tau\) induces a pre-projector feature shift:

\[
\Delta^H_\tau(o)=H(\tau(o))-H(o),
\]

and a post-projector latent shift:

\[
\Delta^z_\tau(o)=P(H(\tau(o)))-P(H(o)).
\]

If ACPC-Flow trains with synthetic representation perturbations, then its source is one of:

Pre-projector version:

\[
\tilde h=H(o)+\epsilon_H,\qquad \tilde z=P(\tilde h).
\]

Post-projector version:

\[
\tilde z=z+\epsilon_z.
\]

The method can cover a pixel perturbation family only if the pixel-induced shifts \(\Delta^H_\tau\) or \(\Delta^z_\tau\) lie inside, or near, the synthetic perturbation tube used during training.

### 3.2 Sufficient coverage condition

Define

\[
G_\mathbf a(z)=\Pi(F^{1:H}(z,\mathbf a)).
\]

For pre-projector analysis, use

\[
\bar G_\mathbf a(h)=G_\mathbf a(P(h)).
\]

Assume:

1. **Coverage**: for each diagnostic pixel perturbation \(\tau\), there exists a synthetic perturbation \(\epsilon^\star\) used in training such that
   \[
   \|\Delta^H_\tau(o)-\epsilon^\star\|\le \kappa.
   \]
2. **Transport accuracy on the synthetic tube**:
   \[
   d_H\big(\bar G_\mathbf a(T_\phi(H(o)+\epsilon^\star)),\bar G_\mathbf a(H(o))\big)\le \varepsilon.
   \]
3. **Local Lipschitz continuity** of \(\bar G_\mathbf a\circ T_\phi\) on the tube with constant \(L\).
4. **No task-neighborhood crossing**: \(H(\tau(o))\) remains in the same-state predictive basin and does not enter a task-distinct basin.
5. **SMPR preservation**: task-grounded different-state margins are preserved after transport.

Then the pixel perturbation satisfies:

\[
d_H\big(\bar G_\mathbf a(T_\phi(H(\tau(o)))),\bar G_\mathbf a(H(o))\big)
\le \varepsilon + L\kappa.
\]

This is the main theoretical feasibility statement: representation perturbation training can cover pixel perturbations only up to the coverage error \(\kappa\). If \(\kappa\) is large, the method relies on extrapolation and has no guarantee.

### 3.3 Proof sketch for the sufficient condition

Since

\[
H(\tau(o))=H(o)+\Delta^H_\tau(o),
\]

and \(\Delta^H_\tau(o)\) is within \(\kappa\) of \(\epsilon^\star\), Lipschitz continuity gives:

\[
d_H\big(\bar G T(H(o)+\Delta^H_\tau),\bar G T(H(o)+\epsilon^\star)\big)
\le L\kappa.
\]

By triangle inequality:

\[
d_H\big(\bar G T(H(o)+\Delta^H_\tau),\bar G(H(o))\big)
\le
L\kappa+arepsilon.
\]

The post-projector version is identical after replacing \(H\) by \(z\) and \(\Delta^H_\tau\) by \(\Delta^z_\tau\).

### 3.4 Local Gaussian pixel noise analysis

For small Gaussian pixel noise

\[
\tau(o)=o+\xi,\qquad \xi\sim\mathcal N(0,\sigma^2 I),
\]

if \(H\) is locally differentiable:

\[
H(o+\xi)-H(o)=J_H(o)\xi+R_\xi.
\]

Ignoring higher-order terms:

\[
\Delta^H_\tau(o)\sim \mathcal N(0,\sigma^2 J_H(o)J_H(o)^\top).
\]

If training uses isotropic pre-projector feature noise

\[
\epsilon_H\sim \mathcal N(0,\sigma_H^2 I),
\]

then a covariance-dominance sufficient condition is:

\[
\sigma_H^2 I \succeq \sigma^2 J_H(o)J_H(o)^\top.
\]

Equivalently, a crude radius condition is:

\[
\sigma_H^2 \ge \sigma^2\lambda_{\max}(J_HJ_H^\top).
\]

This explains both why feature perturbation can cover small Gaussian pixel noise in principle, and why isotropic feature noise can be inefficient: if \(J_HJ_H^\top\) is anisotropic, covering the largest corruption direction may over-perturb many irrelevant directions.

### 3.5 Structured perturbations: blur, resize, compression

Blur, resize, JPEG, compression, brightness shifts, camera changes, and occlusion are not guaranteed to be small local Gaussian perturbations in feature space. For these stressors, coverage must be measured empirically:

\[
\Delta^H_\tau(o)=H(\tau(o))-H(o).
\]

The method is plausible only when these shifts remain local, same-state, and non-crossing. If they are large, structured, or aligned with task-neighbor directions, clean-only feature noise is unlikely to cover them.

### 3.6 Impossibility: task-neighborhood crossing

Suppose two task-distinct states \(i,j\) have different action-conditioned predictive targets:

\[
d_H(G_\mathbf a(z_i),G_\mathbf a(z_j))>m.
\]

If pixel corruptions make their encoder features collide:

\[
H(\tau_i(o_i))=H(\tau_j(o_j)),
\]

then any deterministic transport \(T_\phi\) produces the same output for both inputs. It cannot simultaneously recover the two different predictive targets. Therefore deterministic ACPC-Flow cannot solve representation crossing that has already erased task identity.

This is why coverage audit must include nearest-neighbor crossing and task-label crossing metrics.

### 3.7 Impossibility: outside-tube extrapolation

If training perturbations satisfy \(\|\epsilon\|\le r\), but a pixel stressor induces \(\|\Delta_\tau\|\gg r\), then the transport head is unconstrained on that region. Any success there is extrapolation, not supported by the ACPC-Flow objective. Do not claim coverage of such perturbations without empirical evidence.

## 4. Coverage audit: must run before training at scale

The coverage audit answers:

> Can synthetic pre-/post-projector perturbations plausibly cover the representation shifts induced by target pixel corruptions?

This is a cheap offline diagnostic and should be run before expensive training or CEM evaluation.

### 4.1 Required script

Create:

```text
tools/acpc_flow/coverage_audit.py
```

Required output:

```text
assets/paper1_data/acpc_flow_coverage_<task>_<checkpoint>_<date>.json
assets/paper1_data/acpc_flow_coverage_<task>_<checkpoint>_<date>.csv
```

The script must not train anything.

### 4.2 Inputs

Config arguments:

```yaml
checkpoint: <run>/lewm
task: tworoom | reacher | pusht | cube
num_samples: 1000             # start with 1000, scale to 5000 if cheap
history_size: ${wm.history_size}
feature_spaces:
  - encoder_feat              # pre-projector H(o), if exposed
  - emb                       # post-projector z=P(H(o))
corruptions:
  - {type: gaussian_noise, std: 0.03}
  - {type: gaussian_noise, std: 0.05}
  - {type: gaussian_noise, std: 0.08}
  - {type: gaussian_blur, kernel_size: 7}
  - {type: gaussian_blur, kernel_size: 15}
  - {type: resize, factor: 0.75}
  - {type: resize, factor: 0.50}
  - {type: resize, factor: 0.25}
synthetic_noise:
  std_grid: [0.01, 0.02, 0.04, 0.08, 0.12]
  mode: token_std             # token_std | rms | fixed
knn:
  k: 5
candidate_rank:
  num_candidates: 64
  horizon: 5
```

### 4.3 Required code preparation

Expose pre-projector encoder feature if possible:

```python
info["encoder_feat"] = rearrange(pixels_emb, "(b t) d -> b t d", b=b)
info["emb"] = self.projector(pixels_emb)
info["emb_trans"] = self.transport_emb(info["emb"])
```

If exposing `encoder_feat` slows Codex down, run the first coverage audit on `emb` only, but mark `encoder_feat_missing=true` in JSON.

### 4.4 Coverage metrics: magnitude

For each corruption \(\tau\) and feature space \(s\in\{H,z\}\), compute:

\[
\Delta^s_\tau = s(\tau(o))-s(o).
\]

Report:

```text
delta_norm_mean
delta_norm_median
delta_norm_q75
delta_norm_q90
delta_norm_q95
delta_norm_q99
```

For each synthetic noise scale \(\alpha\), compute synthetic noise radius quantiles and coverage rate:

\[
\mathrm{coverage}_{q}(\tau,\alpha)=
\Pr_o[\|\Delta_\tau(o)\|\le q_q(\|\epsilon_\alpha\|)].
\]

Report coverage for q90/q95/q99 synthetic radii.

### 4.5 Coverage metrics: clean-neighbor scale

Build a clean representation bank for the same task and feature space. For each clean representation, compute kNN distance to other clean states:

\[
d_{kNN}(o)=\frac1k\sum_{j\in kNN(o)}\|s(o)-s(o_j)\|.
\]

Report:

\[
r_\tau(o)=\frac{\|\Delta_\tau(o)\|}{d_{kNN}(o)+10^{-6}}.
\]

Quantiles:

```text
ratio_to_knn_median
ratio_to_knn_q75
ratio_to_knn_q90
ratio_to_knn_q95
```

Interpretation:

- `ratio_to_knn_q90 < 0.3`: high hope; corruption shift is mostly local.
- `0.3 <= ratio_to_knn_q90 < 0.8`: medium hope; may work for smoother tasks.
- `ratio_to_knn_q90 >= 0.8`: low hope; high risk of state-neighbor crossing.

### 4.6 Coverage metrics: neighborhood crossing

Using task proxy labels if available, compute:

1. `paired_clean_rank`: rank of the paired clean representation among nearest clean neighbors of corrupted representation.
2. `wrong_label_nn_rate`: nearest clean neighbor of corrupted representation has different task proxy label than paired clean.
3. `closer_to_wrong_than_pair_rate`: there exists a wrong-label clean neighbor closer than the paired clean representation.
4. `same_label_topk_rate`: proportion of top-k clean neighbors sharing the paired clean proxy label.

These metrics operationalize the non-crossing condition. If crossing is high, a deterministic transport cannot reliably recover state identity.

### 4.7 Coverage metrics: direction and anisotropy

Compute covariance of corruption shifts:

\[
\Sigma_\tau=\mathrm{Cov}(\Delta_\tau).
\]

Report:

```text
effective_rank
lambda_max_over_trace
top1_eigen_ratio
top5_eigen_ratio
```

Interpretation:

- Low effective rank / high top eigen ratio means pixel stressor induces structured feature shifts.
- If structured, isotropic synthetic noise may be inefficient.
- Consider covariance-shaped feature noise only after isotropic baseline is understood.

### 4.8 Coverage metrics: task-direction alignment

For each anchor, take nearest clean neighbors with different task proxy label. Compute maximum cosine alignment:

\[
\max_j \cos(\Delta_\tau(o_i), s(o_j)-s(o_i)).
\]

Report:

```text
task_alignment_mean
task_alignment_q90
task_alignment_q95
```

High alignment means the pixel perturbation moves along task-relevant directions, not nuisance directions. Treat this as a risk signal for over-contraction.

### 4.9 Coverage metrics: ACPC / rollout shift

For each clean/corrupted pair and a fixed recorded action sequence, compute one-step or short-horizon diagnostic gap:

\[
d_H(G_\mathbf a(s(\tau(o))),G_\mathbf a(s(o))).
\]

Report:

```text
acpc_gap_mean
acpc_gap_q90
acpc_gap_q95
```

Also compare synthetic perturbation gaps for each synthetic noise scale:

```text
synthetic_acpc_gap_mean[alpha]
synthetic_acpc_gap_q90[alpha]
synthetic_acpc_gap_q95[alpha]
```

If pixel-induced ACPC gaps are far outside the synthetic gap range, training on that synthetic perturbation scale is unlikely to cover the stressor.

### 4.10 Coverage metrics: candidate rank flip

Sample a shared candidate action set per anchor. Compute candidate costs or rollout proxies for clean and corrupted features. Report:

```text
candidate_rank_spearman
candidate_top1_flip_rate
candidate_topk_overlap_rate
candidate_margin_clean_q10
candidate_margin_clean_q50
```

These connect coverage directly to Paper1's fixed-candidate stability logic.

### 4.11 Summary decision rules

For each task/stressor/feature-space pair, output:

```json
"coverage_decision": "high" | "medium" | "low" | "no_go"
```

Suggested rules:

High:

```text
ratio_to_knn_q90 < 0.3
wrong_label_nn_rate < 0.05
closer_to_wrong_than_pair_rate < 0.10
coverage_q95_at_alpha_0.04 > 0.80
candidate_top1_flip_rate not much larger than synthetic alpha=0.04
```

Medium:

```text
ratio_to_knn_q90 < 0.8
wrong_label_nn_rate < 0.15
coverage_q95_at_alpha_0.08 > 0.70
```

Low:

```text
ratio_to_knn_q90 >= 0.8
or wrong_label_nn_rate >= 0.15
or coverage_q95_at_alpha_0.08 < 0.50
```

No-go:

```text
wrong_label_nn_rate >= 0.30
or closer_to_wrong_than_pair_rate >= 0.40
or pixel ACPC gap q90 > 2x largest synthetic ACPC gap q90
```

Do not train ACPC-Flow for a task/stressor as a generalization claim if coverage is no-go.

### 4.12 Required JSON schema

```json
{
  "schema_version": "acpc-flow-coverage-v1",
  "task": "tworoom",
  "checkpoint": "baseline_seed3073/lewm",
  "num_samples": 1000,
  "feature_spaces": ["encoder_feat", "emb"],
  "synthetic_noise_grid": [0.01, 0.02, 0.04, 0.08, 0.12],
  "results": {
    "emb": {
      "gaussian_std0.08": {
        "delta_norm_q90": 0.0,
        "ratio_to_knn_q90": 0.0,
        "wrong_label_nn_rate": 0.0,
        "closer_to_wrong_than_pair_rate": 0.0,
        "coverage_q95_by_alpha": {"0.04": 0.0, "0.08": 0.0},
        "effective_rank": 0.0,
        "top1_eigen_ratio": 0.0,
        "task_alignment_q90": 0.0,
        "acpc_gap_q90": 0.0,
        "synthetic_acpc_gap_q90_by_alpha": {"0.04": 0.0},
        "candidate_top1_flip_rate": 0.0,
        "coverage_decision": "medium"
      }
    }
  },
  "recommendation": {
    "train_clean_only_latent_noise": true,
    "preferred_feature_space": "emb",
    "suggested_noise_std_max": 0.04,
    "no_go_stressors": []
  }
}
```

## 5. Architecture and training design

### 5.1 Expose `info["emb_trans"]` in `encode()`

Make `encode()` expose both canonical LeWM latent and transported latent:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
info["encoder_feat"] = rearrange(pixels_emb, "(b t) d -> b t d", b=b)  # if feasible
emb = self.projector(pixels_emb)
emb = rearrange(emb, "(b t) d -> b t d", b=b)
info["emb"] = emb
info["emb_trans"] = self.transport_emb(info["emb"])
```

`transport_emb()` returns identity when no ACPC-Flow head is attached or transport is disabled.

### 5.2 Non-breaking contract

Do **not** replace `info["emb"]` by default. Existing LeWM behavior must remain unchanged when ACPC-Flow is disabled.

```python
class JEPA(nn.Module):
    def transport_emb(self, emb):
        head = getattr(self, "acpc_flow_head", None)
        enabled = bool(getattr(self, "acpc_flow_enabled", False))
        if head is None or not enabled:
            return emb
        return head(emb)
```

### 5.3 Shared transport for origin and perturbed inputs

The same transport is applied to origin and perturbed inputs. At inference the model usually does not know whether the current observation is clean or corrupted.

Intended behavior:

```text
origin input:    emb_trans ≈ emb
perturbed input: emb_trans moves toward same-state clean predictive basin
```

Therefore every ACPC-Flow mode should include clean identity:

\[
\mathcal L_{id}=\|T_\phi(z)-z\|^2.
\]

This is the origin zero-velocity condition.

### 5.4 Time-step and span design

The phrase “transport context tokens” means: **transport every latent token in the predictor history that the predictor consumes**, not just one state.

```text
ctx_emb = emb[:, :ctx_len]       # shape (B, ctx_len, D)
ctx_act = act_emb[:, :ctx_len]
```

Therefore ACPC-Flow first acts on the full predictor history:

\[
T_\phi(z_{t:t+C-1})=(T_\phi(z_t),\ldots,T_\phi(z_{t+C-1})).
\]

First implementation:

```yaml
loss:
  acpc_flow:
    horizon: 1
    apply_tokens: context
    source_time_policy: aligned
    rollout_mode: one_step
```

## 6. Objective hierarchy: anchors vs diagnostic target

Do **not** assume diagnostic-space loss will automatically dominate latent or predictor supervision.

- `latent_z` matching is the strongest state-preserving anchor. It directly discourages wrong-neighborhood transport.
- `predictor` matching is a strong dynamics-preserving baseline.
- `diagnostic` matching is the control-facing objective that directly matches Paper1's ATR/candidate-stability object, but it is looser and must be checked with SMPR and crossing metrics.
- `identity` matching on clean/origin latents is a safety constraint for the shared head, not an optional baseline.

First run pure variants separately. Only after those are understood, test a hybrid:

\[
\mathcal L_{hybrid}=\lambda_z\mathcal L_z+\lambda_{ACPC}\mathcal L_{ACPC}+\lambda_{id}\mathcal L_{id}.
\]

## 7. Three core objective variants

### Variant A: Latent-Z Transport Loss

\[
\mathcal L_z=\|T_\phi(\tilde z)-z\|^2.
\]

### Variant B: Predictor-Feature Transport Loss

\[
\mathcal L_{pred}
=
\sum_{k=1}^{H}
\|F^k(T_\phi(\tilde z),\mathbf a)-F^k(z,\mathbf a)\|^2.
\]

### Variant C: Diagnostic-Space ACPC Transport Loss

\[
\mathcal L_{ACPC}
=
D_{diag}
\left(
\Pi(F^{1:H}(T_\phi(\tilde z),\mathbf a)),
\Pi(F^{1:H}(z,\mathbf a))
\right).
\]

Recommended first implementation:

\[
\mathcal L_{ACPC-CVaR}
=
\mathrm{CVaR}_{q=0.90}
\left[
\frac{d_H(\Pi F^{1:H}(T_\phi(\tilde z),\mathbf a),\Pi F^{1:H}(z,\mathbf a))}{\text{clean transition scale}}
\right].
\]

Every variant includes clean identity unless deliberately ablated:

\[
\mathcal L = \mathcal L_{variant} + \lambda_{id}\mathcal L_{id}.
\]

## 8. Minimal experiment design

### 8.1 Models / training variants

| ID | Name | Pixel corruption aug during training? | Extra transport? | Objective |
|---|---|---:|---:|---|
| M0 | origin LeWM | no | no | baseline |
| M1 | Latent-Z Flow | no | yes/tiny | `L_z + L_id` |
| M2 | Predictor-Feature Flow | no | yes/tiny | `L_pred + L_id` |
| M3 | ACPC-Flow | no | yes/tiny | `L_ACPC + L_id` |
| M4 | Gaussian-aug LeWM | yes | no | strong matched baseline |
| M5 | ACPC-Flow + Gaussian aug | yes | yes/tiny | optional stacking |
| M6 | Hybrid Z+ACPC Flow | no | yes/tiny | optional after M1-M3 |

### 8.2 Required diagnostics

For each trained checkpoint:

1. ATR.
2. SMPR.
3. Encoder neighborhood crossing / basin preservation.
4. Candidate rank agreement / top-1 flip.
5. Clean prediction loss and clean closed-loop success.
6. Parameter count and training overhead.
7. Clean correction norm `||emb_trans-emb||`.
8. Source correction norm and transport-to-clean distance.

### 8.3 Success criteria

Promote ACPC-Flow only if:

1. Coverage audit is high or medium for the claimed stressor family.
2. M3 reduces ATR more than M1/M2 at comparable clean performance, **or** M6 clearly outperforms M1/M2 while preserving clean performance.
3. M3/M6 preserves or improves SMPR.
4. M3/M6 improves corrupted closed-loop success over M0 on at least two stressors or two tasks.
5. Clean correction norm remains small; clean success drop stays under 5 pp.
6. M3/M6 beats identity/random/same-param controls.

No-go if:

- coverage audit is no-go for the target stressor;
- ATR drops but SMPR drops;
- clean correction becomes large and clean success drops;
- M1/M2 perform the same as M3 and M6 adds no benefit;
- same-budget longer LeWM training matches the gains.

## 9. Codex implementation guide

### 9.1 Files to add/modify

```text
acpc_flow.py
tools/acpc_flow/coverage_audit.py
jepa.py
train.py
config/train/lewm.yaml
tests/...
```

Do not modify CEM for the first PR.

### 9.2 `acpc_flow.py`

Create:

```python
class ResidualTransportHead(nn.Module):
    def __init__(self, dim, hidden_dim=32, scale_init=0.0, norm="layernorm"):
        ...
    def forward(self, z):
        return z + alpha * residual(z)
```

Utilities:

```python
def sample_latent_noise(z, std_min, std_max, mode="token_std", relative=True): ...

def cvar_loss(values, q=0.90): ...

def token_mse(a, b): ...

def diagnostic_distance(pred_a, pred_b, *, normalize=None, tail_mode="mean", q=0.90): ...
```

### 9.3 Config block

Add to `config/train/lewm.yaml`:

```yaml
loss:
  acpc_flow:
    enabled: false
    mode: diagnostic        # latent_z | predictor | diagnostic | hybrid
    source: latent_noise
    weight: 0.1
    identity_weight: 0.1
    hidden_dim: 32
    scale_init: 0.0
    predictor_input_key: emb_trans
    detach_target: true
    stop_grad_clean_branch: true
    use_bounded_aux: true
    apply_tokens: context
    source_time_policy: aligned
    horizon: 1
    rollout_mode: one_step
    noise:
      std_min: 0.0
      std_max: 0.04
      mode: token_std
      relative: true
    pred_space: ${loss.pred.space}
    hybrid:
      latent_weight: 0.1
      acpc_weight: 1.0
    diagnostic:
      projection: identity
      normalize_by_transition_scale: true
      tail_mode: cvar
      q: 0.90
```

### 9.4 `jepa.py` patch

Add:

```python
def transport_emb(self, emb):
    head = getattr(self, "acpc_flow_head", None)
    if head is None or not bool(getattr(self, "acpc_flow_enabled", False)):
        return emb
    return head(emb)
```

Modify `encode()`:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
info["encoder_feat"] = rearrange(pixels_emb, "(b t) d -> b t d", b=b)
emb = self.projector(pixels_emb)
emb = rearrange(emb, "(b t) d -> b t d", b=b)
info["emb"] = emb
info["emb_trans"] = self.transport_emb(emb)
```

### 9.5 `train.py` integration sketch

After encoding:

```python
flow_cfg = cfg.loss.get("acpc_flow", {})
flow_enabled = bool(flow_cfg.get("enabled", False))
pred_input_key = flow_cfg.get("predictor_input_key", "emb_trans") if flow_enabled else "emb"

emb = output["emb"]
emb_trans = output.get("emb_trans", emb)
ctx_emb = output[pred_input_key][:, :ctx_len]
```

Target remains canonical:

```python
tgt_emb = output["emb"][:, n_preds:]
```

Auxiliary loss sketch:

```python
if flow_enabled:
    clean_ctx = output["emb"][:, :ctx_len]
    clean_ctx_trans = output.get("emb_trans", output["emb"])[:, :ctx_len]
    identity_raw = mse_token(clean_ctx_trans, clean_ctx.detach()).mean()

    noise = sample_latent_noise(clean_ctx, ...)
    source_ctx = clean_ctx + noise
    transported_ctx = self.model.transport_emb(source_ctx)

    latent_raw = mse_token(transported_ctx, clean_ctx.detach()).mean()

    if mode == "latent_z":
        variant_raw = latent_raw
    else:
        transported_pred = self.model.predict(transported_ctx, ctx_act)
        with torch.no_grad() if stop_grad_clean_branch else nullcontext():
            clean_pred = self.model.predict(clean_ctx_trans, ctx_act)

        pred_raw = mse_token(
            get_pred_loss_tensor(transported_pred, space=pred_space),
            get_pred_loss_tensor(clean_pred.detach(), space=pred_space),
        ).mean()

        if mode == "predictor":
            variant_raw = pred_raw
        else:
            per_token = mse_token(
                get_pred_loss_tensor(transported_pred, space=pred_space),
                get_pred_loss_tensor(clean_pred.detach(), space=pred_space),
            )
            if normalize_by_transition_scale:
                scale = mse_token(
                    get_pred_loss_tensor(tgt_emb.detach(), space=pred_space),
                    get_pred_loss_tensor(clean_ctx.detach(), space=pred_space),
                ).mean().sqrt().clamp_min(1e-6)
                per_token = per_token / scale.detach()
            diag_raw = cvar_loss(per_token.reshape(-1), q=q) if tail_mode == "cvar" else per_token.mean()
            variant_raw = diag_raw if mode == "diagnostic" else hybrid_latent_weight * latent_raw + hybrid_acpc_weight * diag_raw

    raw = variant_raw + identity_weight * identity_raw
    aux, aux_scale = self_bounded_aux_loss(pred_mse_loss, raw) if use_bounded_aux else (raw, 1.0)
    output["acpc_flow_raw"] = raw
    output["acpc_flow_identity_raw"] = identity_raw.detach()
    output["acpc_flow_latent_raw"] = latent_raw.detach()
    output["acpc_flow_clean_correction_norm"] = torch.linalg.vector_norm(clean_ctx_trans - clean_ctx, dim=-1).mean().detach()
    output["acpc_flow_source_correction_norm"] = torch.linalg.vector_norm(transported_ctx - source_ctx, dim=-1).mean().detach()
    output["acpc_flow_transport_to_clean_l2"] = torch.linalg.vector_norm(transported_ctx - clean_ctx, dim=-1).mean().detach()
    output["pred_loss"] = output["pred_loss"] + flow_weight * aux
```

### 9.6 Coverage audit script checklist

`tools/acpc_flow/coverage_audit.py` must:

- load checkpoint and dataset windows;
- compute clean `encoder_feat` and `emb`;
- compute corrupted `encoder_feat` and `emb` for each stressor;
- compute synthetic perturbation radii for each noise scale;
- compute magnitude, kNN ratio, crossing, anisotropy, task alignment, ACPC gap, and rank flip metrics;
- write JSON and CSV artifacts;
- print a concise per-task/stressor decision table.

### 9.7 Unit tests

Add tests without MuJoCo/data:

1. `ResidualTransportHead` with `scale_init=0` returns input near-exactly.
2. `JEPA.encode()` returns `encoder_feat`, `emb`, and `emb_trans`.
3. With no flow head or disabled flow, `emb_trans` equals `emb`.
4. With a nonzero dummy flow head, `emb_trans` changes and gradients flow.
5. `sample_latent_noise`, `cvar_loss`, and scalar loss modes are finite.
6. Identity loss is near zero when `scale_init=0`.

## 10. Implementation checklist for Codex

- [ ] Add `acpc_flow.py`.
- [ ] Add `tools/acpc_flow/coverage_audit.py` with JSON/CSV output.
- [ ] Expose `encoder_feat`, `emb`, and `emb_trans` in `JEPA.encode()`.
- [ ] Add explicit identity loss for origin near-zero correction.
- [ ] Add `loss.acpc_flow` config block.
- [ ] Add `latent_z`, `predictor`, `diagnostic`, and `hybrid` modes.
- [ ] Transport all context/history tokens by default.
- [ ] Add logging for correction norms and transport-to-clean distance.
- [ ] Add unit tests.
- [ ] Do not modify CEM in this PR.
- [ ] Do not run large CEM eval before coverage audit and offline ATR/SMPR improve.

Suggested commit message:

```text
Add ACPC-Flow feasibility audit and training plan

- Formalize coverage and non-crossing conditions for feature perturbations
- Add detailed coverage_audit.py specification before training
- Expose encoder_feat, emb, and emb_trans for diagnostics
- Keep latent, predictor, diagnostic, and hybrid objective modes
```
