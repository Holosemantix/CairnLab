# ACPC-Flow / Predictive Plateau Transport: method design and Codex implementation plan

This document specifies the next-method direction that follows from Paper1's selective-ACPC diagnostic. It is written for Codex implementation. The goal is a **clean, testable method family**, not a loss soup.

## 0. One-sentence method idea

Paper1 shows that robust LeWM checkpoints occupy a **selective ACPC plateau**: same-state visual perturbations have low action-conditioned predictive tail risk (ATR), while task-grounded different-state pairs remain separated (SMPR). ACPC-Flow learns a **state-paired latent transport map** that moves perturbed/off-manifold latents into the same action-conditioned predictive equivalence class as their clean latents.

Short form:

> Flow/transport supplies the mechanism; ACPC supplies the success criterion.

The main target is **not** marginal distribution matching and not only raw latent closeness. The main target is agreement **after the transported latent history is rolled out by the predictor under the same action sequence**, in the same diagnostic space that defines ATR.

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

## 3. Architecture: expose `info["emb_trans"]` in `encode()`

The preferred implementation is **not** to keep transport as an external temporary call only. Instead, make `encode()` expose both the canonical LeWM latent and the transported latent:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
emb = self.projector(pixels_emb)
emb = rearrange(emb, "(b t) d -> b t d", b=b)
info["emb"] = emb
info["emb_trans"] = self.transport_emb(info["emb"])
```

where `transport_emb()` returns identity when no ACPC-Flow head is attached or transport is disabled.

### 3.1 Why this interface matters

1. `info["emb"]` remains the exact original LeWM projected latent.
2. `info["emb_trans"]` becomes the stable interface for all future ACPC-Flow training/eval paths.
3. Training can compare `emb`, `emb_trans`, and transported noisy/source latents cleanly.
4. Inference can later switch predictor input from `emb` to `emb_trans` with a config flag, without rewriting all downstream code.
5. Diagnostic scripts can read both fields from the same encoded batch and compute before/after ATR, rank flip, and neighborhood crossing.

### 3.2 Non-breaking contract

Do **not** replace `info["emb"]` by default. Existing LeWM behavior must remain unchanged when ACPC-Flow is disabled.

Required behavior:

```python
class JEPA(nn.Module):
    def transport_emb(self, emb):
        head = getattr(self, "acpc_flow_head", None)
        enabled = bool(getattr(self, "acpc_flow_enabled", False))
        if head is None or not enabled:
            return emb
        return head(emb)
```

`emb_trans` may share storage with `emb` in the disabled path; no `.clone()` is required unless a later mutation requires it.

### 3.3 Shared transport for origin and perturbed inputs

The same `projector` and the same `transport_emb()` are applied to both origin and perturbed inputs. At inference time the model usually does **not** know whether the current observation is clean or corrupted. This is a feature, not a bug: `transport_emb()` must be near-identity on in-distribution origin latents and corrective only when the latent is off the clean predictive basin.

Intended behavior:

```text
origin input:    emb_trans ≈ emb
perturbed input: emb_trans moves toward same-state clean predictive basin
```

This intended origin behavior cannot be left implicit. Add an explicit identity/zero-correction term on clean origin latents:

\[
\mathcal L_{id}=\|T_\phi(z)-z\|^2.
\]

This term is not a new competing method; it is the safety condition that teaches the shared transport head that clean/origin inputs should have velocity/correction near zero. Without this term, there is no reliable way for a single shared head to infer that an origin-like input should be unchanged.

Required monitors:

```text
acpc_flow_clean_correction_norm = ||emb_trans - emb|| on clean train/val batches
acpc_flow_source_correction_norm = ||T(source_emb) - source_emb||
acpc_flow_transport_to_clean_l2 = ||T(source_emb) - clean_emb||
```

If clean correction grows large while ATR improves, treat it as over-transport risk. Clean eval and SMPR decide whether it is acceptable.

### 3.4 Predictor input selection

Add a model/config-level selector:

```yaml
loss:
  acpc_flow:
    predictor_input_key: emb_trans  # emb | emb_trans
```

During training, when ACPC-Flow is enabled, the main prediction context may use `emb_trans`:

```python
ctx_emb = output[predictor_input_key][:, :ctx_len]
```

Keep the clean target branch anchored to the canonical clean `emb` unless explicitly testing another ablation:

```python
tgt_emb = output["emb"][:, n_preds:]
```

This makes the method end-to-end: the flow head is actually in the predictor path, while the target remains the original clean latent/predictive reference.

## 4. Time-step and span design

The phrase “transport context tokens” means: **transport every latent token in the predictor history that the predictor consumes**, not just one state. In LeWM, predictor input is a history window:

```text
ctx_emb = emb[:, :ctx_len]       # shape (B, ctx_len, D)
ctx_act = act_emb[:, :ctx_len]
```

Therefore ACPC-Flow should first act on the full predictor history:

\[
T_\phi(z_{t:t+C-1})=(T_\phi(z_t),\ldots,T_\phi(z_{t+C-1})),
\]

where \(C=\texttt{ctx_len}\). The first implementation can use a tokenwise shared transport head, but it must be applied to all context tokens consumed by the predictor.

### 4.1 First implementation convention

Use an explicit config:

```yaml
loss:
  acpc_flow:
    horizon: 1
    apply_tokens: context       # context | last_context | all_available
    source_time_policy: aligned # aligned only for first PR
    rollout_mode: one_step
```

Meaning:

1. `apply_tokens=context` transports all history/context tokens `0:ctx_len`.
2. `horizon=1` uses the same one-step predictor output as the base LeWM loss. This keeps cost and code risk low.
3. `source_time_policy=aligned` means clean/source latents refer to the same batch window and same token indices. Do not introduce shifted-time source/target pairing in the first PR.
4. For `latent_z` loss, compare transported source context tokens to same-index clean context tokens.
5. For `predictor` and `diagnostic` losses, compare predictor outputs under the same action prefix. With `horizon=1`, use the existing `self.model.predict(ctx, ctx_act)` output.

### 4.2 Why explicit origin identity is needed

Because the transport head is shared across clean and perturbed inputs, time index alone cannot tell the model whether a token is origin or perturbed. If a clean token and a perturbed token have similar coordinates, the head receives only the latent vector. It cannot automatically know that “origin means velocity = 0” unless training supplies that condition.

Therefore every ACPC-Flow mode should include a clean identity component, either:

- as a separate raw term `identity_raw = mse_token(emb_trans, emb).mean()`, or
- as part of the `latent_z` / `hybrid` anchor.

Recommended first implementation:

```yaml
loss:
  acpc_flow:
    identity_weight: 0.1
```

Use:

```python
identity_raw = mse_token(output["emb_trans"], output["emb"].detach()).mean()
raw = variant_raw + identity_weight * identity_raw
```

This is the clean-origin zero-velocity constraint.

### 4.3 Later multi-step extension

Only add multi-step after horizon=1 works:

```yaml
loss:
  acpc_flow:
    horizon: 3
    rollout_mode: autoregressive # teacher_forced | autoregressive
```

For `autoregressive`, repeatedly feed the predicted token back exactly as `rollout()` does. For `teacher_forced`, use clean future latents as context after each step. Do not mix these modes without explicit ablation labels.

### 4.4 Inference selection, second step

Add later, after training losses pass tests:

```yaml
wm:
  inference:
    embedding_key: emb_trans  # default emb
```

In `rollout()` and `get_cost()`, use:

```python
key = getattr(self, "inference_embedding_key", "emb")
init_emb = _init.get(key, _init["emb"])
goal_emb = goal.get(key, goal["emb"])
```

Do not make this the first blocking change if it slows Codex down; but the `encode()` interface should be added now so training and offline diagnostics use the same fields.

## 5. Objective hierarchy: anchors vs diagnostic target

Do **not** assume diagnostic-space loss will automatically dominate latent or predictor supervision. The strict objectives are important.

- `latent_z` matching is the strongest state-preserving anchor. It directly discourages wrong-neighborhood transport.
- `predictor` matching is a strong dynamics-preserving baseline.
- `diagnostic` matching is the control-facing objective that directly matches Paper1's ATR/candidate-stability object, but it is looser and must be checked with SMPR and crossing metrics.
- `identity` matching on clean/origin latents is a safety constraint for the shared head, not an optional baseline.

The empirical question is:

> Is raw latent repair already enough, or does ACPC diagnostic-space transport provide additional planner-facing benefit?

First run pure variants separately. Only after those are understood, test a hybrid:

\[
\mathcal L_{hybrid}=\lambda_z\mathcal L_z+\lambda_{ACPC}\mathcal L_{ACPC}+\lambda_{id}\mathcal L_{id}.
\]

Hybrid interpretation:

- \(\mathcal L_z\) keeps the map in the correct same-state basin.
- \(\mathcal L_{ACPC}\) shapes the action-conditioned predictive plateau.
- \(\mathcal L_{id}\) makes origin inputs near-zero velocity/correction.

Do not start with the hybrid; it makes attribution unclear.

## 6. Three core objective variants that must be compared

### Variant A: Latent-Z Transport Loss, baseline

\[
\mathcal L_z=\|T_\phi(\tilde z)-z\|^2.
\]

Expected strength: best at restoring origin latent neighborhood.

Expected limitation: raw latent closeness may not correspond to planner-facing predictive stability.

### Variant B: Predictor-Feature Transport Loss, stronger baseline

\[
\mathcal L_{pred}
=
\sum_{k=1}^{H}
\|F^k(T_\phi(\tilde z),\mathbf a)-F^k(z,
\mathbf a)\|^2.
\]

Expected strength: stable dynamics-facing supervision.

Expected limitation: it may not use the exact diagnostic projection/normalization/tail metric that defines ATR.

### Variant C: Diagnostic-Space ACPC Transport Loss, main method candidate

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

### Shared identity component

Every variant should include clean identity unless deliberately ablated:

\[
\mathcal L = \mathcal L_{variant} + \lambda_{id}\mathcal L_{id}.
\]

This answers the origin/perturbed ambiguity: the same head sees both, and clean origin must be explicitly trained to have near-zero correction.

### Recommended training loss for first experiments

For each variant, train only one objective family at a time:

```text
loss = LeWM_base_loss + weight * bounded_aux_loss(variant_raw_loss + identity_weight * identity_raw)
```

Use the existing `self_bounded_aux_loss(base_loss, aux_raw)` pattern in `train.py` to prevent the auxiliary term from dominating the baseline prediction loss.

## 7. Source latent construction

### 7.1 Clean-only source, strongest claim

Use synthetic local latent perturbation around the canonical clean `emb`:

\[
\tilde z=z+\epsilon,
\qquad z=\texttt{info["emb"]}.
\]

Then transport it using the same head that defines `emb_trans`:

\[
T_\phi(\tilde z)=\texttt{model.transport_emb}(z+\epsilon).
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

### 7.2 Pixel-paired source, baseline only

Use paired clean/pixel-corrupted views:

\[
\tilde z=E(T_{pixel}(o)),\qquad z=E(o).
\]

This is easier but weakens the claim because it uses a specified pixel corruption family. Use it only as a baseline or debugging mode.

## 8. Minimal experiment design

### 8.1 Models / training variants

Run at least these:

| ID | Name | Pixel corruption aug during training? | Extra transport? | Objective |
|---|---|---:|---:|---|
| M0 | origin LeWM | no | no | baseline |
| M1 | Latent-Z Flow | no | yes/tiny | `L_z + L_id` |
| M2 | Predictor-Feature Flow | no | yes/tiny | `L_pred + L_id` |
| M3 | ACPC-Flow | no | yes/tiny | `L_ACPC + L_id` |
| M4 | Gaussian-aug LeWM | yes | no | strong matched baseline |
| M5 | ACPC-Flow + Gaussian aug | yes | yes/tiny | stacking check, optional |
| M6 | Hybrid Z+ACPC Flow | no | yes/tiny | optional after M1-M3 |

If parameter-count criticism matters, add:

| ID | Name | Purpose |
|---|---|---|
| P0 | identity transport head | confirms insertion does not change clean behavior |
| P1 | random frozen transport head | rules out architecture-only improvement |
| P2 | same-param transport trained with only identity loss | controls for extra parameters |

### 8.2 Required diagnostics

For each trained checkpoint:

1. ATR: q90 normalized same-state ACPC-H rollout disagreement.
2. SMPR: task-grounded selective margin pass rate.
3. Encoder neighborhood crossing / basin preservation.
4. Candidate rank agreement / top-1 flip on shared candidates.
5. Clean prediction loss and clean closed-loop success.
6. Parameter count and training overhead.
7. Clean correction norm `||emb_trans-emb||`.
8. Source correction norm and transport-to-clean distance.

### 8.3 Success criteria

Promote ACPC-Flow only if:

1. M3 reduces ATR more than M1/M2 at comparable clean performance, **or** M6 clearly outperforms M1/M2 while preserving clean performance.
2. M3/M6 preserves or improves SMPR; low ATR with lower SMPR is failure.
3. M3/M6 improves corrupted closed-loop success over M0 on at least two stressors or two tasks.
4. M3/M6 beats identity/random/same-param controls.
5. Clean correction norm remains small; clean success drop stays under 5 pp.
6. M3/M6 is competitive with Gaussian aug on at least some held-out stressors, or stacks with Gaussian aug.

No-go if:

- ATR drops but SMPR drops;
- clean correction becomes large and clean success drops;
- M1/M2 perform the same as M3 and M6 adds no benefit;
- gains only appear in one task or one eval seed;
- same-budget longer LeWM training matches the gains.

## 9. Codex implementation guide

### 9.1 Files to add/modify

```text
acpc_flow.py                       # new helpers / transport head
jepa.py                            # add transport_emb() and info["emb_trans"]
train.py                           # integrate loss into lejepa_forward
config/train/lewm.yaml             # add loss.acpc_flow config block
tests/...                          # unit tests
```

Do not modify CEM for this PR.

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
    source: latent_noise    # latent_noise | pixel_paired
    weight: 0.1
    identity_weight: 0.1
    hidden_dim: 32
    scale_init: 0.0
    predictor_input_key: emb_trans  # emb | emb_trans
    detach_target: true
    stop_grad_clean_branch: true
    use_bounded_aux: true
    apply_tokens: context           # context | last_context | all_available
    source_time_policy: aligned
    horizon: 1
    rollout_mode: one_step          # one_step first; later autoregressive | teacher_forced
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
emb = self.projector(pixels_emb)
emb = rearrange(emb, "(b t) d -> b t d", b=b)
info["emb"] = emb
info["emb_trans"] = self.transport_emb(emb)
```

### 9.5 `train.py` integration sketch

After `output = self.model.encode(...)`:

```python
flow_cfg = cfg.loss.get("acpc_flow", {})
flow_enabled = bool(flow_cfg.get("enabled", False))
pred_input_key = flow_cfg.get("predictor_input_key", "emb_trans") if flow_enabled else "emb"

emb = output["emb"]
emb_trans = output.get("emb_trans", emb)
ctx_emb = output[pred_input_key][:, :ctx_len]
```

Keep target anchored to canonical clean `emb`:

```python
tgt_emb = output["emb"][:, n_preds:]
```

After `pred_mse_loss` exists:

```python
if flow_enabled:
    clean_ctx = output["emb"][:, :ctx_len]
    clean_ctx_trans = output.get("emb_trans", output["emb"])[:, :ctx_len]

    identity_raw = mse_token(clean_ctx_trans, clean_ctx.detach()).mean()

    if source == "latent_noise":
        noise = sample_latent_noise(clean_ctx, ...)
        source_ctx = clean_ctx + noise
    elif source == "pixel_paired":
        source_ctx = paired_or_noisy_ctx
    else:
        raise ValueError(...)

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
            if mode == "diagnostic":
                variant_raw = diag_raw
            elif mode == "hybrid":
                variant_raw = hybrid_latent_weight * latent_raw + hybrid_acpc_weight * diag_raw
            else:
                raise ValueError(...)

    raw = variant_raw + identity_weight * identity_raw
    aux, aux_scale = self_bounded_aux_loss(pred_mse_loss, raw) if use_bounded_aux else (raw, 1.0)
    output["acpc_flow_raw"] = raw
    output["acpc_flow_identity_raw"] = identity_raw.detach()
    output["acpc_flow_latent_raw"] = latent_raw.detach()
    output["acpc_flow_loss"] = aux
    output["acpc_flow_scale"] = aux_scale
    output["acpc_flow_clean_correction_norm"] = torch.linalg.vector_norm(clean_ctx_trans - clean_ctx, dim=-1).mean().detach()
    output["acpc_flow_source_correction_norm"] = torch.linalg.vector_norm(transported_ctx - source_ctx, dim=-1).mean().detach()
    output["acpc_flow_transport_to_clean_l2"] = torch.linalg.vector_norm(transported_ctx - clean_ctx, dim=-1).mean().detach()
    output["pred_loss"] = output["pred_loss"] + flow_weight * aux
```

### 9.6 Unit tests

Add tests without MuJoCo/data:

1. `ResidualTransportHead` with `scale_init=0` returns input exactly or near-exactly.
2. `JEPA.encode()` returns both `emb` and `emb_trans`.
3. With no flow head or `acpc_flow_enabled=False`, `emb_trans is emb` or `allclose(emb_trans, emb)`.
4. With a nonzero dummy flow head, `emb_trans` changes and gradients flow.
5. `sample_latent_noise` returns correct shape and finite values.
6. `cvar_loss` equals top-tail mean on a known tensor.
7. `mode=latent_z`, `mode=predictor`, `mode=diagnostic`, and `mode=hybrid` produce scalar finite losses on dummy tensors.
8. Identity loss is near zero when `scale_init=0`.

## 10. Training commands

Latent-Z baseline:

```bash
python train.py data=tworoom \
  output_model_name=acpcflow_latentz \
  loss.acpc_flow.enabled=true \
  loss.acpc_flow.mode=latent_z \
  loss.acpc_flow.weight=0.1 \
  loss.acpc_flow.identity_weight=0.1 \
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
  loss.acpc_flow.identity_weight=0.1 \
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
  loss.acpc_flow.identity_weight=0.1 \
  loss.acpc_flow.noise.std_max=0.04 \
  loss.acpc_flow.diagnostic.tail_mode=cvar \
  loss.acpc_flow.diagnostic.q=0.90 \
  image_noise.std_max=0.0
```

Hybrid after pure variants:

```bash
python train.py data=tworoom \
  output_model_name=acpcflow_hybrid \
  loss.acpc_flow.enabled=true \
  loss.acpc_flow.mode=hybrid \
  loss.acpc_flow.weight=0.1 \
  loss.acpc_flow.identity_weight=0.1 \
  loss.acpc_flow.hybrid.latent_weight=0.1 \
  loss.acpc_flow.hybrid.acpc_weight=1.0 \
  loss.acpc_flow.noise.std_max=0.04 \
  loss.acpc_flow.diagnostic.tail_mode=cvar \
  image_noise.std_max=0.0
```

## 11. Implementation checklist for Codex

- [ ] Add `acpc_flow.py` with residual transport head and pure helper losses.
- [ ] Add `loss.acpc_flow` config block to `config/train/lewm.yaml`, default disabled.
- [ ] Attach `model.acpc_flow_head` only when enabled.
- [ ] Add `JEPA.transport_emb()` and make `JEPA.encode()` output both `emb` and `emb_trans`.
- [ ] Add `predictor_input_key`, `horizon`, `apply_tokens`, `source_time_policy`, and `identity_weight` config.
- [ ] Transport all context/history tokens by default (`apply_tokens=context`).
- [ ] Add explicit clean identity loss so origin-like inputs learn near-zero correction.
- [ ] Use `emb_trans` as predictor input when ACPC-Flow is enabled, but keep canonical `emb` as target anchor.
- [ ] Integrate four mutually exclusive modes into `lejepa_forward`: `latent_z`, `predictor`, `diagnostic`, `hybrid`.
- [ ] Use `self_bounded_aux_loss` by default.
- [ ] Keep `image_noise.std_max=0.0` for clean-only ACPC-Flow experiments.
- [ ] Add logging keys for clean/source correction norms.
- [ ] Add unit tests for transport head, `encode()` output, noise sampler, CVaR, scalar loss outputs, and identity behavior.
- [ ] Do not modify CEM for this PR.
- [ ] Make inference `embedding_key=emb_trans` a second PR unless it is straightforward.

Suggested commit message:

```text
Clarify ACPC-Flow history transport and identity behavior

- Transport all predictor history tokens through emb_trans
- Add explicit identity loss for origin near-zero correction
- Define one-step aligned time policy for first implementation
- Keep latent, predictor, diagnostic, and hybrid objective modes
```
