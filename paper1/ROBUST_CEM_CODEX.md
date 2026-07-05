# Codex task: final sanity check for full inner-loop Robust CEM

This file supersedes the earlier broad Robust CEM implementation brief. The 2026-07-05 `eval100x3` iteration log shows that the **final-rerank / rank-vote / elite-mean** family is not strong enough for a Paper1 main-method claim. Do not keep tuning that family.

## 0. Current empirical status

Primary artifact:

```text
paper1/ROBUST_CEM_EVAL100X3_ITERATION_LOG_20260705.md
assets/paper1_data/robust_cem_eval100x3_iteration_summary_20260705.json
```

Observed result on frozen origin `baseline_seed3073` LeWM, TwoRoom, Gaussian pixel corruption `std=0.08`, `num_eval=100`, seeds `42/43/44`:

| Planner | Success | Interpretation |
|---|---:|---|
| standard `cem48_n4` | 29.7% | base low-budget CEM |
| compute-matched `cem192_n4_compute` | 33.3% | stronger baseline |
| old selected-candidate `rank_mean_std` final rerank | 30.0% | no clear gain |
| `rank_vote + elite_mean` final rerank | 34.0% | small/borderline |
| `rank_vote + elite_mean`, `num_views=8`, pool32 | 33.0% | no improvement |

Conclusion: best final-rerank result is only **+4.3 pp over standard CEM** and **+0.7 pp over compute-matched CEM**. Reacher seed42 is negative for the best variant. PushT remains near floor. This should be treated as **weak/no-go**, not as a method result.

## 1. Research interpretation

The failed final-rerank result is informative:

1. ACPC/ranking diagnostics correctly suggest that candidate rankings can be unstable under perturbed observations.
2. However, **selecting or reranking action sequences by stability at the end of CEM is not enough** to repair a frozen origin checkpoint under corrupted observations.
3. The main bottleneck is likely upstream representation/prediction shift, not just final candidate selection.
4. Stable candidates are not necessarily correct candidates. In contact or near-boundary states, the useful action may be sensitive rather than stable.
5. Extra views also increase compute. Any planner-side positive result must beat a compute-matched CEM baseline.

Therefore, do not place this route in the paper unless the final sanity check below gives a clear, compute-matched win.

## 2. Only remaining CEM question

The only remaining untested possibility is:

> Did we only test final-stage robust reranking, while full inner-loop robust CEM could still matter because it changes the CEM search distribution at every iteration?

This is the only reason to run one more sanity check. It should be a **strict, low-budget go/no-go test**, not a new tuning campaign.

## 3. Required implementation: full inner-loop robust elite selection

Implement or verify a solver mode where robust scoring is used **inside every CEM iteration before `topk`**, not only for final candidate reranking.

Expected behavior per CEM step:

```python
# Standard CEM step
candidates = sample_from_current_gaussian(mean, var)        # (B, N, H, D)
base_cost = model.get_cost(expanded_infos, candidates)      # (B, N)

# Full inner-loop Robust CEM replacement
view_costs = evaluate_same_candidates_under_views(
    info=expanded_infos,
    candidates=candidates,
    num_views=K,
    include_identity=True,
)                                                           # (B, N, K)
robust_score = aggregate(view_costs)                        # (B, N)

topk_vals, topk_inds = torch.topk(
    robust_score, k=topk, dim=1, largest=False
)
topk_candidates = candidates[batch_indices, topk_inds]
mean = topk_candidates.mean(dim=1)
var = topk_candidates.std(dim=1)
```

The important distinction is that **the robust score updates the CEM mean/variance at each iteration**. A final-stage rerank is not sufficient for this sanity check.

## 4. Minimal modes to implement

Only implement the minimal modes needed for the sanity check.

```yaml
solver.inner_loop_robust: true
solver.num_views: 4
solver.include_identity: true
solver.view_type: gaussian_noise
solver.view_std: 0.04
solver.perturb_pixels: true
solver.perturb_goal: false
solver.score_mode: rank_vote_elitemean   # preferred if already available
# or score_mode: mean_std                # fallback if rank_vote is not wired into inner loop
solver.beta: 0.5
```

Do **not** tune many modes. If both are cheap, test only:

1. inner-loop `mean_std`, `K=4`, `view_std=0.04`, `beta=0.5`;
2. inner-loop `rank_vote`, `K=4`, `view_std=0.04`, CEM-consistent `elite_mean` output.

Keep `include_identity=True` so candidates must still be good on the actual current observation.

## 5. Mutation safety

`jepa.get_cost()` and `SphericalJEPA.get_cost()` mutate `info_dict` by adding rollout/goal keys. Robust scoring must use fresh cloned dicts when expanding views.

Required helper behavior:

```python
def fresh_info_dict(info):
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

Do not reuse a dict after a model cost call when evaluating another view or another scoring path.

## 6. Strict sanity-check protocol

Run exactly this before doing anything larger.

### Task/checkpoint/eval

- Checkpoint: frozen origin `baseline_seed3073` LeWM.
- Corruption: pixel Gaussian `std=0.08`.
- Eval: `num_eval=100`, seed `42` only for the first sanity pass.
- Low-concurrency guard: `world.num_envs=1`, one eval process at a time.
- Planner budget: match the iteration log unless current code requires otherwise:
  - standard: `num_samples=48`, `n_steps=4`, `topk=8`;
  - compute baseline: `num_samples=192`, `n_steps=4`, `topk=24`.

### Tasks

Run only:

1. TwoRoom — because it had the only positive final-rerank signal;
2. Reacher — because the previous best final-rerank method was negative on seed42.

### Solvers

For each task:

1. standard `cem48_n4`;
2. compute-matched `cem192_n4_compute`;
3. best final-rerank method from the log, as a reference only;
4. full inner-loop robust CEM, `K=4`.

## 7. Go/no-go rule

Stop immediately unless the seed42 sanity check satisfies all of these:

1. TwoRoom inner-loop robust CEM beats compute-matched CEM by **at least +5 pp**.
2. Reacher inner-loop robust CEM is **not worse** than standard CEM.
3. Clean eval drop, if checked, is less than **5 pp**.
4. The result is not explained by simply increasing CEM samples.

If these pass, run `num_eval=100` with seeds `42/43/44` on TwoRoom and Reacher. If the 3-seed mean gap over compute-matched is less than +5 pp, declare no-go.

If any seed42 condition fails, do not continue. Record the route as a bounded negative result in lab notes only.

## 8. Do not add this to Paper1 unless it wins clearly

Do not include failed Robust CEM in the main paper or appendix. The current paper is already diagnostic-heavy. A weak planner-side no-go would distract from the main ACPC analysis unless it produces a clean positive intervention.

Promote Robust CEM only if it becomes a clear result:

- frozen origin checkpoint improves corrupted eval by a large margin;
- compute-matched CEM cannot explain it;
- at least TwoRoom and Reacher are non-negative;
- clean performance is preserved.

Otherwise, the next method direction should move upstream: representation-level repair, clean-only robustness regularization, or training-time objectives that push the model into the ACPC robustness plateau without relying on explicit corruption augmentation.

## 9. Suggested commit message for implementation

```text
Add inner-loop robust CEM sanity mode

- Apply robust candidate scoring before CEM top-k at every iteration
- Preserve CEM elite-mean output protocol
- Add strict TwoRoom/Reacher sanity-check configs
- Keep final-rerank family marked as weak/no-go
```
