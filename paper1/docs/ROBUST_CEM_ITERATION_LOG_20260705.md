# Robust CEM Iteration Log 2026-07-05

Goal: use a frozen origin-trained LeWM checkpoint and change only the eval-time planner so visual-perturbation eval improves. The implementation is tied to the Paper 1 ACPC/ranking diagnostic: perturbations make action-conditioned candidate costs and rankings unstable, so the planner should prefer candidates whose rank is stable across local perturbed views.

## Implementation Iterations

1. `mean` / `mean_std` / `worst` robust CEM over all CEM updates: negative or unstable. Reacher seed42+43 at 50 episodes each was effectively tied with standard CEM.
2. `margin_std`: ACPC-margin inspired cost penalty. It was too conservative on Reacher and did not improve over standard.
3. `final` robust rerank: keep CEM proposal updates on the identity/base observation, then robustly select only from the final base-cost elite pool. This matches the fixed-candidate theorem more closely than changing every CEM update.
4. `rank_mean_std`: convert each view cost vector to candidate ranks and aggregate rank mean + rank std. This targets ranking instability directly rather than raw cost scale drift.

Selected config: `config/eval/solver/robust_cem_rank.yaml`.

## Main Pilot Result

| Task | Checkpoint | Corruption | Planner | Seeds | Success |
|---|---|---|---|---|---:|
| TwoRoom | origin `baseline_seed3073` | pixels Gaussian std=0.08 | `cem48_n4` | seed42,seed43,seed44 | 24.0% (36/150) |
| TwoRoom | origin `baseline_seed3073` | pixels Gaussian std=0.08 | `cem192_n4_compute` | seed42,seed43,seed44 | 28.7% (43/150) |
| TwoRoom | origin `baseline_seed3073` | pixels Gaussian std=0.08 | `rcem_final_rankmeanstd` | seed42,seed43,seed44 | 32.7% (49/150) |

The rank-based final robust CEM improves TwoRoom corrupted eval by +8.7 pp over standard CEM and +4.0 pp over compute-matched CEM at this reduced budget.

## Clean Check

| Task | Planner | Seed | Clean Success |
|---|---|---|---:|
| TwoRoom | `cem48_n4` | seed42 | 40.0% (20/50) |
| TwoRoom | `rcem_final_rankmeanstd` | seed42 | 32.0% (16/50) |

Clean seed42 drops by 8 pp (40% -> 32%). This means the current method is a promising corrupted-eval intervention, but not yet a strong main-method result under the strict no-clean-drop criterion.

## Paper-Facing Interpretation

- Positive story: the version that aligns with ACPC ranking instability, not raw cost variance, is the first one that beats both standard and compute-matched CEM on TwoRoom corrupted eval.
- Boundary: the evidence is task-specific so far, and clean drop still needs mitigation or honest framing.
- Recommended framing if used: planner-side diagnostic intervention / appendix method, not a replacement for perturbation training.

## Exact Command Pattern

```bash
PYTHONPATH=. STABLEWM_HOME=/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll CUDA_VISIBLE_DEVICES=0 python eval.py --config-name=tworoom.yaml \
  cache_dir=/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll \
  policy=lewm-tworooms/ckpt/tworoom_lewm_baseline_seed3073 \
  seed=42 eval.num_eval=50 world.num_envs=1 eval.eval_budget=25 \
  eval.corruption.std=0.08 solver=robust_cem_rank
```
