# Codex task: ACPC-inspired Robust CEM wrapper

This document is written as an implementation brief for Codex. It should be treated as a **surgical, testable code change** rather than a broad refactor.

## 0. Research stance

The goal is not to claim that test-time robust CEM magically solves visual robustness. The goal is to test the strongest intervention naturally implied by the diagnostic paper:

> If perturbations break control because the world model's action-conditioned predicted costs and candidate rankings are unstable, then CEM should select action sequences whose predicted cost is both low and stable under a small observation-perturbation ensemble.

A successful result would be surprising and valuable because it is **planner-side**, **training-free**, and can be applied to frozen checkpoints. A weak result is still useful if it falsifies the hypothesis cleanly.

The implementation must therefore include compute-matched and TTA baselines. Without those, positive numbers will not be scientifically convincing.

## 1. Why this is plausible, and why it may fail

### Plausibility

The current evaluation path already instantiates a solver through Hydra and passes it into `stable_worldmodel.policy.WorldModelPolicy` in `eval.py`. The policy then calls `solver(sliced, init_action=sliced_init)` at replanning time. This is an ideal insertion point because the model/checkpoint and environment loop do not need to change.

The upstream `CEMSolver` samples candidates, evaluates `cost.get_cost(expanded_infos, candidates)`, selects top-k candidates, and updates the CEM Gaussian from those elites. A robust wrapper can change only the cost used for elite selection.

The method is also aligned with the ACPC diagnostic: compare fixed action candidates under perturbed observations and prefer candidates with stable predicted costs/rankings.

### Main failure modes

1. **Information has already been destroyed.** If the actual observation is blurred/resized/occluded, local TTA around that already-corrupted image cannot recover missing information. This method is more likely to help additive or mild sensor noise than severe low-pass corruption.

2. **Variance penalties may become conservative.** PushT/Cube may require contact-sensitive, high-gain actions whose predicted costs are naturally sensitive near decision boundaries. Penalizing instability too hard may avoid the very actions needed for success.

3. **Compute can explain the gain.** Robust CEM uses multiple observation views. If `K` views help only because they spend `K` times more model calls, then `standard CEM with K*num_samples` is the real baseline.

4. **Mean-only TTA may explain the gain.** If `mean(J)` over views matches `mean(J) + beta*std(J)`, then the contribution is generic test-time augmentation, not ACPC/ranking stability.

5. **Matched corruption can overstate significance.** If planner views use Gaussian noise and the test corruption is Gaussian noise, the method may be a test-time matched-noise trick. Held-out blur/resize/compression/mixed corruption must be tested before claiming broad robustness.

6. **The model cost function mutates `info_dict`.** `jepa.get_cost()` adds keys such as `goal_emb`, `predicted_emb`, and `action`. The robust wrapper must avoid accidental cross-view or cross-step contamination by creating fresh dicts for each cost call or by carefully reconstructing robust info dicts.

## 2. Minimal implementation target

Add a new local solver wrapper:

```text
robust_cem.py
config/eval/solver/robust_cem.yaml
tests/test_robust_cem.py
```

Do not modify training code. Avoid touching `eval.py` unless absolutely necessary. It already writes `policy.solver.last_robust_stats` and `policy.solver.robust_history` if those attributes exist.

## 3. API design

The class should be Hydra-instantiable and compatible with both possible upstream solver constructor conventions (`model=` in this repo's config, `cost=` in newer upstream docs).

```python
class RobustCEMSolver:
    def __init__(
        self,
        model=None,
        cost=None,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cuda",
        seed: int = 1234,
        callbacks: list | None = None,
        # robust-scoring knobs
        enabled: bool = True,
        num_views: int = 4,
        include_identity: bool = True,
        view_type: str = "gaussian_noise",      # gaussian_noise | gaussian_blur | resize
        view_std: float = 0.04,                  # gaussian_noise only, pixel-space std
        view_kernel_size: int = 7,               # gaussian_blur only
        view_resize_factor: float = 0.75,        # resize only
        perturb_pixels: bool = True,
        perturb_goal: bool = False,
        score_mode: str = "mean_std",           # base | mean | mean_std | worst | quantile
        beta: float = 0.5,
        quantile: float = 0.8,
        robust_rescore: str = "all",            # all | elite
        elite_rescore_multiplier: int = 2,
        max_view_batch: int | None = None,
        record_history: bool = True,
    ): ...
```

Important defaults:

- `enabled=True` for `solver=robust_cem`, but if `num_views <= 1` or `score_mode == "base"`, it should reduce exactly to standard CEM scoring.
- `include_identity=True` so the current observation always participates as one view. This avoids selecting candidates that are stable but bad on the actual observation.
- `perturb_goal=False` by default because current eval configs usually corrupt `pixels` but not `goal`. Add this knob for future ablations.
- `robust_rescore="all"` is scientifically cleaner: robust score directly drives elite selection. `elite` is a cheaper ablation, not the main method.

## 4. Implementation details

### 4.1 Use the upstream CEM logic as the template

Copy the upstream `CEMSolver.solve()` structure rather than trying to monkey-patch internals. Preserve these behaviors:

- same `configure(action_space, n_envs, config)` API;
- same `action_dim`, `horizon`, `dtype`, `__call__` properties;
- same warm-start behavior via `prepare_init_action`;
- same candidate sampling, mean/variance update, callbacks, and output keys.

Reason: the robust score must enter before `torch.topk(costs, ...)`, otherwise it will not influence CEM distribution updates.

### 4.2 Cost aggregation

For candidates with shape:

```python
candidates: (B, N, H, D)
expanded_infos[k]: (B, N, ...)
```

construct robust views along the sample axis:

```python
robust_candidates: (B, N * K, H, D)
robust_infos[k]:   (B, N * K, ...)
view_costs_flat = model.get_cost(robust_infos, robust_candidates)  # (B, N*K)
view_costs = view_costs_flat.reshape(B, N, K)
```

Aggregate:

```python
base_cost = view_costs[..., 0] if include_identity else view_costs.mean(dim=-1)
mean_cost = view_costs.mean(dim=-1)
std_cost = view_costs.std(dim=-1, unbiased=False)

if score_mode == "base":
    score = base_cost
elif score_mode == "mean":
    score = mean_cost
elif score_mode == "mean_std":
    score = mean_cost + beta * std_cost
elif score_mode == "worst":
    score = view_costs.max(dim=-1).values
elif score_mode == "quantile":
    score = torch.quantile(view_costs, q=quantile, dim=-1)
else:
    raise ValueError(...)
```

Use `score` for top-k elite selection. Keep base/view stats for logging.

### 4.3 View construction

Only perturb transformed tensors already inside planning. Do not use any clean/oracle observation branch.

Keys:

- perturb `pixels` when `perturb_pixels=True`;
- perturb `goal` only when `perturb_goal=True`;
- never perturb `action`, `proprio`, `state`, `qpos`, `qvel`, `terminated`, or non-image columns.

Suggested helper:

```python
def _repeat_along_sample_axis(x, num_views):
    # x: (B, N, ...)
    B, N = x.shape[:2]
    return x.unsqueeze(2).expand(B, N, num_views, *x.shape[2:]).clone()
```

Then apply corruption to `x_views[:, :, start_view:]`, where `start_view = 1` if `include_identity` else `0`.

Use existing transforms from `utils.py` if practical:

- `AddNormalizedGaussianNoise(view_std, view_std)`;
- `AddGaussianBlur(view_kernel_size, view_kernel_size)`;
- `AddResize(view_resize_factor, view_resize_factor)`.

These transforms already operate on ImageNet-normalized tensors and support leading dimensions, so they can be applied to tensors shaped `(B, N, K, T, C, H, W)` or `(B, N, K, C, H, W)`.

Flatten back:

```python
x_flat = x_views.reshape(B, N * num_views, *x.shape[2:])
```

### 4.4 Avoid mutation bugs

Because `model.get_cost()` may mutate `info_dict`, do not call it on a dict that will be reused in another score path. Use fresh dicts.

At minimum:

```python
def _fresh_info_dict(info):
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out
```

For the standard/base path in `score_mode="base"`, use a fresh copy too. Scientific correctness matters more than micro-optimizing here.

### 4.5 Logging

Expose:

```python
self.last_robust_stats: dict
self.robust_history: list[dict]
```

`eval.py` already writes these when present.

Record per solve:

- `enabled`, `num_views`, `include_identity`, `view_type`, `score_mode`, `beta`, `quantile`;
- `mean_base_cost`, `mean_robust_score`, `mean_view_std`, `mean_view_range` at the final step;
- `top1_flip_rate`: fraction of non-identity views whose best candidate differs from the identity-view best candidate;
- `robust_changed_top1_rate`: fraction of envs where aggregate-score best differs from identity best;
- `solve_time_sec`;
- `effective_model_call_multiplier` approx `num_views` for `all`, or less for `elite`.

For `robust_history`, append one compact dict per CEM iteration per batch, not full tensors.

### 4.6 Config file

Create `config/eval/solver/robust_cem.yaml`:

```yaml
_target_: robust_cem.RobustCEMSolver
model: ???
batch_size: 1
num_samples: 300
var_scale: 1.0
n_steps: 30
topk: 30
device: "cuda"
seed: ${seed}

enabled: true
num_views: 4
include_identity: true
view_type: gaussian_noise
view_std: 0.04
view_kernel_size: 7
view_resize_factor: 0.75
perturb_pixels: true
perturb_goal: false
score_mode: mean_std
beta: 0.5
quantile: 0.8
robust_rescore: all
elite_rescore_multiplier: 2
max_view_batch: null
record_history: true
```

Usage:

```bash
python eval.py --config-name=pusht.yaml \
  policy=<run>/lewm \
  solver=robust_cem \
  solver.num_views=4 \
  solver.view_std=0.04 \
  solver.score_mode=mean_std \
  solver.beta=0.5 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08
```

Compute-matched baseline:

```bash
python eval.py --config-name=pusht.yaml \
  policy=<run>/lewm \
  solver=cem \
  solver.num_samples=1200 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08
```

Mean-only TTA baseline:

```bash
python eval.py --config-name=pusht.yaml \
  policy=<run>/lewm \
  solver=robust_cem \
  solver.num_views=4 \
  solver.score_mode=mean \
  solver.view_std=0.04 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08
```

Worst-case robust baseline:

```bash
python eval.py --config-name=pusht.yaml \
  policy=<run>/lewm \
  solver=robust_cem \
  solver.num_views=4 \
  solver.score_mode=worst \
  solver.view_std=0.04 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08
```

## 5. Unit tests

Add `tests/test_robust_cem.py` with synthetic costs. Do not require MuJoCo or datasets.

### Test 1: standard-mode shape and protocol

- Build a fake continuous action space.
- Build a dummy model with `get_cost(info_dict, action_candidates)` returning `(B, S)`.
- Instantiate `RobustCEMSolver(enabled=False or score_mode="base", num_views=1)`.
- Call `configure()` and `solve()`.
- Assert output has `actions` shape `(B, horizon, action_dim)` and `costs` length `B`.

### Test 2: robust cost penalizes instability

Create a dummy model whose `get_cost()` reads a scalar `pixels` value and makes one candidate good under identity but unstable under perturbed views. Verify `mean_std` or `worst` selects the stable candidate when `beta` is high enough.

Implementation hint: simplify by setting `n_steps=1`, `num_samples=2`, `topk=1`, and manually controlling candidates if needed by monkeypatching candidate sampling or by testing `_aggregate_view_costs()` directly.

### Test 3: no mutation leakage

Use a dummy model that mutates `info_dict["goal_emb"] = ...` inside `get_cost()`. Verify repeated robust scoring calls do not accumulate unexpected keys in the original input dict passed to `solve()`.

### Test 4: identity reduction

With `num_views=1`, `include_identity=True`, and `score_mode="mean"`, robust scores should equal base scores for the same candidates.

## 6. Pilot experiment plan

Start small. Do not run the full paper grid until this passes a cheap sanity check.

### Stage A: two-task smoke test

Tasks:

- Reacher: smoother dynamics, likely higher chance of success.
- PushT: contact-heavy, higher chance of failure due to conservative robust scoring.

Checkpoints:

- clean LeWM checkpoint;
- Gaussian-augmented endpoint if available.

Eval budget:

- `eval.num_eval=20` for the first sweep;
- one eval seed only;
- compare policies on the same episode list/seed.

Conditions:

1. clean eval;
2. Gaussian `std=0.08`;
3. blur `kernel_size=7` or `15`;
4. resize `factor=0.75` or `0.5`.

Solvers:

1. standard CEM, `num_samples=300`;
2. compute-matched CEM, `num_samples=1200` for `num_views=4`;
3. robust CEM mean-only, `K=4`, `view_std=0.04`;
4. robust CEM mean+std, `K=4`, `view_std=0.04`, `beta in {0.25, 0.5, 1.0}`;
5. robust CEM worst, `K=4`, `view_std=0.04`.

### Stage B: grid if Stage A has signal

Grid:

- `num_views in {2, 4, 8}`;
- `view_std in {0.02, 0.04, 0.08}`;
- `score_mode in {mean, mean_std, worst}`;
- `beta in {0.25, 0.5, 1.0}` for mean_std.

Stop early if clean success drops by more than 5 percentage points or if compute-matched CEM matches the gains.

### Stage C: full validation

Only if Stage B shows signal:

- 4 tasks: PushT, TwoRoom, Reacher, Cube;
- clean and Gaussian-augmented checkpoints;
- 3 eval seeds;
- 100 trajectories per seed if matching the existing paper standard;
- report wall-clock evaluation time.

## 7. What counts as a real win

This should be the go/no-go standard before changing the paper's main claim.

### Strong positive

Use as a main paper contribution only if most of the following hold:

1. Frozen clean checkpoint + robust CEM improves corrupted eval by at least 10 percentage points over standard CEM on at least two tasks.
2. Robust CEM does not reduce clean eval by more than 5 percentage points.
3. Robust CEM improves or stacks on Gaussian-augmented checkpoints.
4. Gains persist on at least one held-out corruption family not matching the planner view type.
5. Compute-matched standard CEM does not close the gap.
6. `mean_std` or `worst` beats mean-only TTA in at least some settings.
7. Logged top-1 flip / view-std diagnostics decrease or align with success improvements.

If these hold, the paper can become: diagnostic analysis reveals planner-side ranking instability, and a training-free robust CEM intervention directly mitigates it.

### Medium positive

Use as appendix or secondary result if:

- robust CEM helps only clean checkpoints but not augmented checkpoints;
- robust CEM helps only Gaussian corruption;
- mean-only TTA explains most gains;
- gains are task-specific.

### Negative

Do not promote as a main contribution if:

- clean performance drops heavily;
- compute-matched CEM catches up;
- mean-only TTA matches mean+std/worst;
- improvements require planner view type to match test corruption;
- only one task improves.

A negative result is still publishable as a diagnostic failure analysis if reported honestly.

## 8. Paper-writing implications if successful

If robust CEM works, do **not** oversell it as replacing perturbation training. The stronger, safer framing is:

> Perturbation training is a training-time solution. Robust CEM is a planner-time intervention derived from action-conditioned diagnostics. It can be applied to frozen checkpoints and can stack with augmentation.

Required paper table:

| Checkpoint | Planner | Clean | Gaussian | Blur | Resize | Eval time |
|---|---|---:|---:|---:|---:|---:|
| clean | CEM | | | | | |
| clean | CEM, Kx samples | | | | | |
| clean | TTA mean | | | | | |
| clean | robust CEM mean+std/worst | | | | | |
| aug | CEM | | | | | |
| aug | robust CEM | | | | | |

Required ablation table:

| Mode | K | view std | beta | Gaussian | held-out corruption | clean drop |
|---|---:|---:|---:|---:|---:|---:|
| mean | 4 | 0.04 | - | | | |
| mean_std | 4 | 0.04 | 0.5 | | | |
| worst | 4 | 0.04 | - | | | |
| Kx CEM | - | - | - | | | |

## 9. Implementation acceptance checklist

Codex should finish only when all of these are true:

- [ ] `robust_cem.py` exists and is Hydra-instantiable.
- [ ] `config/eval/solver/robust_cem.yaml` exists.
- [ ] `solver=robust_cem` runs without changing `eval.py`.
- [ ] `score_mode=base` or `num_views=1` reduces to normal CEM logic.
- [ ] The wrapper does not require a clean observation branch.
- [ ] The wrapper exposes `last_robust_stats` and optionally `robust_history`.
- [ ] Unit tests cover shape/protocol, aggregation, mutation safety, and identity reduction.
- [ ] The README or this document includes exact pilot commands.
- [ ] No training code is changed.

## 10. Suggested commit message

```text
Add robust CEM solver wrapper

- Add Hydra-instantiable planner-side robust CEM solver
- Score CEM candidates under perturbation ensembles
- Log robust cost/rank stability diagnostics
- Add solver config and synthetic unit tests
```
