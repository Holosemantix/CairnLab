# ACPC-Flow / Predictive Plateau Transport: method design and Codex implementation plan

This document specifies the next-method direction that follows from Paper1's selective-ACPC diagnostic. It is written for Codex implementation. The goal is a **clean, testable method family**, not a loss soup.

## 0. One-sentence method idea

Paper1 shows that robust LeWM checkpoints occupy a **selective ACPC plateau**: same-state visual perturbations have low action-conditioned predictive tail risk (ATR), while task-grounded different-state pairs remain separated (SMPR). ACPC-Flow learns a **state-paired latent transport map** that moves perturbed/off-manifold latents into the same action-conditioned predictive equivalence class as their clean latents.

Short form:

> Flow/transport supplies the mechanism; ACPC supplies the success criterion.

The main target is **not** marginal distribution matching and not only raw latent closeness. The main target is agreement **after the transported latent is rolled out by the predictor under the same action sequence**, in the same diagnostic space that defines ATR.

---

## 1. How this connects to Paper1 theory and diagnostics

### 1.1 Paper1 diagnostic target

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

### 1.2 Transport version of the diagnostic

Introduce a transport map \(T_\phi\) applied to a perturbed/off-manifold latent:

\[
z_t^\phi = T_\phi(\tilde z_t).
\]

The diagnostic-space transport target is:

\[
\Pi(F_\theta^{1:H}(T_\phi(\tilde z_t),\mathbf a))
\approx
\Pi(F_\theta^{1:H}(z_t,\mathbf a)).
\]

This is the key connection: transport success is measured **after action-conditioned rollout**, not only in raw latent space.

### 1.3 Candidate-cost stability connection

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

### 1.4 Local sensitivity connection

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

Thus the desired transport contracts nuisance/corruption-induced latent directions before they reach the predictor, while avoiding contraction along task-relevant directions. This is exactly why SMPR and neighborhood-crossing diagnostics remain necessary.

### 1.5 Encoder geometry nuance

Do **not** write that encoder geometry is unimportant. Better wording:

> Encoder geometry is a first-stage risk signal: perturbed views should remain in the same-state predictive basin and avoid crossing into task-distinct neighborhoods. However, raw encoder distance alone is not a complete robustness criterion; predictor sensitivity and planner margins determine whether the shift changes action-conditioned rollout and ranking.

Therefore ACPC-Flow should use raw latent matching as a baseline/anchor, but the main criterion must be diagnostic-space predictive matching.

---

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

### 2.1 One-step residual transport, recommended first

Use a small residual map:

\[
T_\phi(x)=x+r_\phi(x).
\]

Recommended architecture:

```text
LayerNorm(d)
Linear(d -> r)
GELU
Linear(r -> d)
residual scale alpha
```

Initialize alpha to 0 or a very small value so the transport starts as identity. This handles parameter-count criticism and protects clean performance.

### 2.2 Optional time-conditioned flow path, later only

If implementing actual flow-style training:

\[
z_s=(1-s)\tilde z+s z,\qquad s\sim U(0,1),
\]

\[
\mathcal L_{FM}=\|v_\phi(z_s,s)-(z-\tilde z)\|^2.
\]

Inference can still use one step:

\[
T_\phi(\tilde z)=\tilde z+v_\phi(\tilde z,0).
\]

Do **not** implement the time-conditioned version until the one-step residual version has working tests and offline diagnostics.

---

## 3. Three core objective variants that must be compared

The whole point is to show that ACPC diagnostic-space transport is not the same as raw latent transport.

### Variant A: Latent-Z Transport Loss, baseline

\[
\mathcal L_z=\|T_\phi(\tilde z)-z\|^2.
\]

Purpose: tests whether geometrically mapping source latent to clean latent is enough.

Expected limitation: raw latent closeness may not correspond to planner-facing predictive stability.

### Variant B: Predictor-Feature Transport Loss, stronger baseline

\[
\mathcal L_{pred}
=
\sum_{k=1}^{H}
\|F^k(T_\phi(\tilde z),\mathbf a)-F^k(z,\mathbf a)\|^2.
\]

Purpose: tests whether direct predictor rollout feature matching is enough.

Expected limitation: it may not use the exact diagnostic projection/normalization/tail metric that defines ATR.

### Variant C: Diagnostic-Space ACPC Transport Loss, main method

\[
\mathcal L_{ACPC}
=
D_{diag}
\left(
\Pi(F^{1:H}(T_\phi(\tilde z),\mathbf a)),
\Pi(F^{1:H}(z,\mathbf a))
\right).
\]

Use the same diagnostic conventions as Paper1 where possible:

- same action sequence on clean/source branches;
- horizon-weighted rollout distance;
- optional normalization by clean transition scale;
- tail/CVaR version to mimic ATR rather than only mean MSE.

Recommended first implementation:

\[
\mathcal L_{ACPC-CVaR}
=
\mathrm{CVaR}_{q=0.90}
\left[
\frac{d_H(\Pi F^{1:H}(T_\phi(\tilde z),\mathbf a),\Pi F^{1:H}(z,\mathbf a))}{\text{clean transition scale}}
\right].
\]

If CVaR is unstable in the first code pass, implement mean first but keep a config flag for `tail_mode=cvar`.

### Recommended training loss for first experiments

For each variant, train only one objective family at a time:

```text
loss = LeWM_base_loss + weight * bounded_aux_loss(variant_raw_loss)
```

Use the existing `self_bounded_aux_loss(base_loss, aux_raw)` pattern in `train.py` to prevent the auxiliary term from dominating the baseline prediction loss.

Do not combine all three variants in one run.

---

## 4. Source latent construction

### 4.1 Clean-only source, strongest claim

Use synthetic local latent perturbation:

\[
\tilde z=z+\epsilon.
\]

No pixel corruption is used during training. This is the strongest method claim:

> learn an ACPC plateau without hand-specified pixel corruption augmentation.

Recommended noise scale:

```yaml
noise:
  mode: token_std      # token_std | rms | fixed
  std_min: 0.0
  std_max: 0.04
  relative: true
  sample_per_token: true
```

Use local bounded noise. Do not claim coverage of arbitrary perturbations.

### 4.2 Pixel-paired source, baseline only

Use paired clean/pixel-corrupted views:

\[
\tilde z=E(T_{pixel}(o)),\qquad z=E(o).
\]

This is easier but weakens the claim because it uses a specified pixel corruption family. Use it only as a baseline or debugging mode.

### 4.3 Do not tune on target corruptions for generalization claims

If `std/blur/resize` corruptions are used to choose source noise or weights, do not call those corruptions held-out. For strongest claims, choose latent noise scale from clean latent statistics only, then evaluate on Gaussian/blur/resize/JPEG as held-out stressors.

---

## 5. Minimal experiment design

### 5.1 Models / training variants

Run at least these:

| ID | Name | Pixel corruption aug during training? | Extra transport? | Objective |
|---|---|---:|---:|---|
| M0 | origin LeWM | no | no | baseline |
| M1 | Latent-Z Flow | no | yes/tiny | `L_z` |
| M2 | Predictor-Feature Flow | no | yes/tiny | `L_pred` |
| M3 | ACPC-Flow | no | yes/tiny | `L_ACPC` |
| M4 | Gaussian-aug LeWM | yes | no | strong matched baseline |
| M5 | ACPC-Flow + Gaussian aug | yes | yes/tiny | stacking check, optional |

If parameter-count criticism matters, add:

| ID | Name | Purpose |
|---|---|---|
| P0 | identity transport head | confirms insertion does not change clean behavior |
| P1 | random frozen transport head | rules out architecture-only improvement |
| P2 | same-param transport trained with only identity loss | controls for extra parameters |

### 5.2 Tasks

Start with:

1. TwoRoom: smoother, high signal under Gaussian and blur.
2. Reacher: strong Gaussian recovery and clean-control lift in Paper1.
3. PushT: hard contact-heavy stress case.

Add Cube after offline diagnostics show signal.

### 5.3 Evaluation corruptions

Minimum:

- clean;
- Gaussian std=0.08;
- unseen Gaussian severity, e.g. std=0.05 if not selected;
- blur k=15;
- resize 0.25 or 0.5;
- optional JPEG/brightness/compression if already supported.

### 5.4 Required diagnostics

For each trained checkpoint:

1. ATR: q90 normalized same-state ACPC-H rollout disagreement.
2. SMPR: task-grounded selective margin pass rate.
3. Encoder neighborhood crossing / basin preservation:
   - nearest-neighbor label crossing rate;
   - noisy/source latent closer to wrong clean neighbor than paired clean latent;
   - transported latent crossing rate.
4. Candidate rank agreement / top-1 flip on shared candidates.
5. Clean prediction loss and clean closed-loop success.
6. Parameter count and training overhead.

### 5.5 Success criteria

Promote ACPC-Flow only if:

1. M3 reduces ATR more than M1/M2 at comparable clean performance.
2. M3 preserves or improves SMPR; low ATR with lower SMPR is failure.
3. M3 improves corrupted closed-loop success over M0 on at least two stressors or two tasks.
4. M3 is not simply matched Gaussian training: it uses no pixel corruption in the clean-only variant.
5. M3 beats identity/random/same-param controls.
6. M3 is competitive with Gaussian aug on at least some held-out stressors, or stacks with Gaussian aug.

No-go if:

- ATR drops but SMPR drops;
- clean success drops more than 5 pp;
- M1/M2 perform the same as M3;
- gains only appear in one task or one eval seed;
- same-budget longer LeWM training matches the gains.

---

## 6. Offline feasibility before expensive CEM eval

Before full closed-loop eval, run offline paired diagnostics.

### 6.1 Coverage analysis

For clean observations `o` and diagnostic-only corruptions `T(o)`, compute:

\[
z=E(o),\qquad z_T=E(T(o)),\qquad \Delta_T=z_T-z.
\]

Compare:

- norm of `Delta_T`;
- ratio to clean kNN distance;
- direction/covariance of `Delta_T`;
- rollout gap `d_H(G_a(z_T), G_a(z))`;
- rank flip / candidate cost correlation.

If pixel-corruption-induced shifts are much larger than planned latent perturbation radius or align with task-neighbor directions, clean-only latent perturbation may not work. Record this before training.

### 6.2 Transport repair check

After training M1/M2/M3, evaluate on clean/corrupt pairs not used in training:

\[
T_\phi(E(T(o)))
\]

should reduce:

- latent distance to paired clean latent;
- ACPC rollout gap;
- candidate rank flip;
- neighborhood crossing.

It must not reduce SMPR.

Only run expensive CEM eval if offline ATR/rank/crossing improves.

---

## 7. Codex implementation guide

### 7.1 Current code facts

`jepa.py` currently computes encoder CLS features and immediately applies `self.projector`:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
emb = self.projector(pixels_emb)
info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
```

`predict()` consumes `emb` and action embeddings. `rollout()` and `get_cost()` eventually use `emb` for planning.

`train.py` already has paired-view infrastructure, `mse_token`, `get_pred_loss_tensor`, and `self_bounded_aux_loss`. It also has `snap_acpc` and `generic_latent_consistency` code paths that are useful templates.

### 7.2 First implementation scope

Implement only post-projector latent transport first because `emb` already exists.

Do **not** modify CEM, eval policy, or data loaders.

Files to add/modify:

```text
acpc_flow.py                       # new helpers / transport head
train.py                           # integrate loss into lejepa_forward
config/train/lewm.yaml             # add loss.acpc_flow config block
scripts or tools optional          # offline diagnostic script if easy
```

Optional later:

```text
jepa.py                            # expose pre-projector encoder feature if needed
```

### 7.3 New module: `acpc_flow.py`

Create:

```python
class ResidualTransportHead(nn.Module):
    def __init__(self, dim, hidden_dim=32, scale_init=0.0, norm="layernorm"):
        ...
    def forward(self, z):
        # z: (B,T,D) or (...,D)
        return z + alpha * residual(z)
```

Utilities:

```python
def sample_latent_noise(z, std_min, std_max, mode="token_std", relative=True): ...

def cvar_loss(values, q=0.90): ...

def token_mse(a, b): ...

def diagnostic_distance(pred_a, pred_b, *, normalize=None, tail_mode="mean", q=0.90): ...
```

Keep these functions pure and unit-testable.

### 7.4 Attach transport head to model

Preferred minimal route:

- in `train.py` model construction, if `cfg.loss.acpc_flow.enabled`, attach:

```python
model.acpc_flow_head = ResidualTransportHead(
    dim=cfg.wm.embed_dim,
    hidden_dim=cfg.loss.acpc_flow.hidden_dim,
    scale_init=cfg.loss.acpc_flow.scale_init,
)
```

Make sure it is included in optimizer parameters automatically because it is a submodule of `model`.

If model construction is hard to edit, attach it in the Lightning module initialization path, but saving/loading object checkpoints is cleaner if it lives on `model`.

### 7.5 Config block

Add to `config/train/lewm.yaml`:

```yaml
loss:
  acpc_flow:
    enabled: false
    mode: diagnostic        # latent_z | predictor | diagnostic
    source: latent_noise    # latent_noise | pixel_paired
    weight: 0.1
    hidden_dim: 32
    scale_init: 0.0
    detach_target: true
    stop_grad_clean_branch: true
    use_bounded_aux: true
    noise:
      std_min: 0.0
      std_max: 0.04
      mode: token_std       # token_std | rms | fixed
      relative: true
    horizon: 1
    pred_space: ${loss.pred.space}
    diagnostic:
      projection: identity  # identity first; later cost/delta projection if available
      normalize_by_transition_scale: true
      tail_mode: cvar       # mean | cvar
      q: 0.90
```

### 7.6 Integrate into `lejepa_forward`

Add after `pred_emb`, `pred_loss_emb`, `tgt_loss_emb`, and `pred_mse_loss` are available, but before final `output["loss"]` is assembled.

Pseudo-code:

```python
flow_cfg = cfg.loss.get("acpc_flow", {})
if flow_cfg.get("enabled", False):
    ctx_clean = ctx_emb
    ctx_act = ctx_act

    # source branch
    if flow_cfg.get("source", "latent_noise") == "latent_noise":
        noise = sample_latent_noise(ctx_clean.detach() if detach_source else ctx_clean, ...)
        source_ctx = ctx_clean + noise
    elif flow_cfg.get("source") == "pixel_paired":
        # optional baseline only; reuse existing paired-view infrastructure if available
        source_ctx = noisy_or_perturbed_ctx
    else:
        raise ValueError(...)

    transported_ctx = self.model.acpc_flow_head(source_ctx)

    mode = flow_cfg.get("mode", "diagnostic")
    if mode == "latent_z":
        raw = mse_token(transported_ctx, ctx_clean.detach()).mean()
    else:
        transported_pred = self.model.predict(transported_ctx, ctx_act)
        with torch.no_grad() if stop_grad_clean_branch else nullcontext():
            clean_pred = self.model.predict(ctx_clean, ctx_act)
        if mode == "predictor":
            raw = mse_token(
                get_pred_loss_tensor(transported_pred, space=pred_space),
                get_pred_loss_tensor(clean_pred.detach(), space=pred_space),
            ).mean()
        elif mode == "diagnostic":
            # normalized ACPC-like discrepancy; start with one-step pred
            per_token = mse_token(
                get_pred_loss_tensor(transported_pred, space=pred_space),
                get_pred_loss_tensor(clean_pred.detach(), space=pred_space),
            )
            if normalize_by_transition_scale:
                # simple first scale: clean transition target scale
                scale = mse_token(
                    get_pred_loss_tensor(tgt_emb.detach(), space=pred_space),
                    get_pred_loss_tensor(ctx_clean.detach(), space=pred_space),
                ).mean().sqrt().clamp_min(1e-6)
                per_token = per_token / scale.detach()
            raw = cvar_loss(per_token.reshape(-1), q=q) if tail_mode == "cvar" else per_token.mean()
        else:
            raise ValueError(...)

    aux, aux_scale = self_bounded_aux_loss(pred_mse_loss, raw) if use_bounded_aux else (raw, 1.0)
    output["acpc_flow_raw"] = raw
    output["acpc_flow_loss"] = aux
    output["acpc_flow_scale"] = aux_scale
    output["pred_loss"] = output["pred_loss"] + flow_weight * aux
```

Notes:

- Start with horizon=1 because current `wm.num_preds=1` in the main config.
- Multi-step rollout can be added later by recursively feeding predictions, but do not block the first implementation on it.
- Do not train all modes at once.

### 7.7 Pre-projector version, later optional

If post-projector results are promising, expose encoder CLS features in `jepa.py`:

```python
info["encoder_feat"] = rearrange(pixels_emb, "(b t) d -> b t d", b=b)
```

Add helper:

```python
def project_features(self, feat):
    flat = rearrange(feat, "b t d -> (b t) d")
    emb = self.projector(flat)
    return rearrange(emb, "(b t) d -> b t d", b=feat.size(0))
```

Then compare:

- perturb post-projector `emb`;
- perturb pre-projector `encoder_feat` and pass through `projector`;
- transport post-projector `emb`.

This addresses the question of whether the method trains predictor robustness only or actually shapes encoder/projector geometry.

### 7.8 Unit tests

Add tests without MuJoCo/data:

1. `ResidualTransportHead` with `scale_init=0` returns input exactly or near-exactly.
2. `sample_latent_noise` returns correct shape and finite values.
3. `cvar_loss` equals top-tail mean on a known tensor.
4. `mode=latent_z`, `mode=predictor`, and `mode=diagnostic` produce scalar finite losses on dummy tensors.
5. `self_bounded_aux_loss` path does not increase aux above base loss scale when raw aux is large.

### 7.9 Training commands

Latent-Z baseline:

```bash
python train.py data=tworoom \
  output_model_name=acpcflow_latentz \
  loss.acpc_flow.enabled=true \
  loss.acpc_flow.mode=latent_z \
  loss.acpc_flow.weight=0.1 \
  loss.acpc_flow.noise.std_max=0.04 \
  image_noise.std_max=0.0
```

Predictor-feature baseline:

```bash
python train.py data=tworoom \
  output_model_name=acpcflow_pred \
  loss.acpc_flow.enabled=true \
  loss.acpc_flow.mode=predictor \
  loss.acpc_flow.weight=0.1 \
  loss.acpc_flow.noise.std_max=0.04 \
  image_noise.std_max=0.0
```

Diagnostic-space main method:

```bash
python train.py data=tworoom \
  output_model_name=acpcflow_diag \
  loss.acpc_flow.enabled=true \
  loss.acpc_flow.mode=diagnostic \
  loss.acpc_flow.weight=0.1 \
  loss.acpc_flow.noise.std_max=0.04 \
  loss.acpc_flow.diagnostic.tail_mode=cvar \
  loss.acpc_flow.diagnostic.q=0.90 \
  image_noise.std_max=0.0
```

Pixel Gaussian augmentation baseline remains:

```bash
python train.py data=tworoom \
  output_model_name=lewm_gauss_std008 \
  image_noise.std_max=0.08
```

### 7.10 Logging keys

Log at minimum:

```text
acpc_flow_raw
acpc_flow_loss
acpc_flow_scale
acpc_flow_noise_norm_mean
acpc_flow_correction_norm_mean
acpc_flow_transport_to_clean_l2
acpc_flow_mode_id
```

For diagnostic mode:

```text
acpc_flow_diag_tail_mode
acpc_flow_diag_q
acpc_flow_diag_scale
```

---

## 8. Analysis scripts to add after training works

Create `tools/acpc_flow/analyze_acpc_flow_offline.py` or extend existing diagnostic scripts.

Required outputs:

```text
assets/paper1_data/acpc_flow_offline_<task>_<run>.json
```

Fields:

```json
{
  "task": "tworoom",
  "run": "...",
  "mode": "diagnostic",
  "atr_before": ...,
  "atr_after": ...,
  "smpr_before": ...,
  "smpr_after": ...,
  "rank_flip_before": ...,
  "rank_flip_after": ...,
  "encoder_crossing_before": ...,
  "encoder_crossing_after": ...,
  "clean_distortion": ...
}
```

Do not run large CEM eval unless offline ATR/rank/crossing improve.

---

## 9. Paper-writing frame if successful

Do not frame this as generic Flow Matching. Frame it as:

> ACPC-guided latent transport into a selective predictive plateau.

Suggested paragraph:

```text
Paper1 showed that matched Gaussian input augmentation discovers checkpoints with low ATR and high SMPR, but does not tell us whether the plateau can be targeted directly. ACPC-Flow asks whether robustness can be induced without specifying a pixel corruption family: a small state-paired transport map moves perturbed latents toward the clean action-conditioned predictive equivalence class. We compare raw latent transport, predictor-feature transport, and diagnostic-space ACPC transport. Only the last uses the same object that Paper1 connects to candidate-cost drift and top-1 stability.
```

Main claim, only if data supports it:

> Diagnostic-space transport reduces ATR while preserving SMPR and improves corrupted closed-loop control more reliably than raw latent or predictor-feature transport.

Do not claim universal perturbation robustness.

---

## 10. Implementation checklist for Codex

- [ ] Add `acpc_flow.py` with residual transport head and pure helper losses.
- [ ] Add `loss.acpc_flow` config block to `config/train/lewm.yaml`, default disabled.
- [ ] Attach `model.acpc_flow_head` only when enabled.
- [ ] Integrate three mutually exclusive modes into `lejepa_forward`: `latent_z`, `predictor`, `diagnostic`.
- [ ] Use `self_bounded_aux_loss` by default.
- [ ] Keep `image_noise.std_max=0.0` for clean-only ACPC-Flow experiments.
- [ ] Add logging keys.
- [ ] Add unit tests for transport head, noise sampler, CVaR, and scalar loss outputs.
- [ ] Do not modify eval/CEM for this PR.
- [ ] Do not implement pre-projector feature transport until the post-projector version has tests and offline metrics.

Suggested commit message:

```text
Add ACPC-Flow latent transport training mode

- Add residual latent transport head and diagnostic-space loss helpers
- Add latent_z, predictor, and diagnostic ACPC-Flow objectives
- Wire optional loss.acpc_flow block into LeWM training
- Add unit tests and logging for offline ATR/SMPR follow-up
```
