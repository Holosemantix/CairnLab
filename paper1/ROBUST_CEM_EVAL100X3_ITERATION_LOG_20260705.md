# Robust CEM 100x3 Iteration Log 2026-07-05

Goal: validate whether frozen origin LeWM checkpoints can gain corrupted-eval score by changing only the eval-time planner. The user-facing decision rule for this pass is strict: if the three-seed mean gap is only around 3-5 pp, treat it as no clear improvement and keep iterating or record the route as weak/no-go.

## Protocol

- Checkpoint: origin `baseline_seed3073` LeWM.
- Corruption: pixel Gaussian `std=0.08`.
- Eval: `num_eval=100`, seeds `42/43/44` when a method has signal.
- Low-concurrency guard: `world.num_envs=1`, one eval process at a time.
- Planner budget: `num_samples=48`, `n_steps=4`, `topk=8` unless stated; compute baseline is `num_samples=192`, `n_steps=4`, `topk=24`.

## Main 100x3 Result: TwoRoom

| Planner | Seeds | Success | Interpretation |
|---|---:|---:|---|
| `cem48_n4` | 42/43/44 | 29.7% (89/300) | standard baseline |
| `cem192_n4_compute` | 42/43/44 | 33.3% (100/300) | compute-matched baseline |
| old `rank_mean_std` selected-candidate final rerank | 42/43/44 | 30.0% (90/300) | no clear gain |
| `rank_vote + elite_mean` | 42/43/44 | 34.0% (102/300) | best, but borderline |
| `rank_vote + elite_mean`, `num_views=8`, pool32 | 42/43/44 | 33.0% (99/300) | worse than default vote |

Best delta: `rank_vote + elite_mean` is +4.3 pp over standard and +0.7 pp over compute-matched. This is inside the borderline band and is not a strong planning-side win.

## Iteration Findings

1. The previous final-rerank implementation selected a single robust-best candidate as the output action sequence. That differs from standard CEM, which outputs the elite mean. This explains why gated fallbacks could perform worse than standard even when they rejected robust switching.
2. `final_output_mode=elite_mean` fixes the protocol mismatch: robust scoring changes the final elite set, but output remains the top-k mean. This produced the only positive TwoRoom signal.
3. `rank_vote` is better than rank mean/std for this planner-side use: it requires cross-view top-1 consensus and preserves base ordering on ties.
4. Increasing views to 8 or expanding the final pool to 32 did not improve the three-seed result.
5. Reacher seed42 was negative for the best method: standard 7%, compute 7%, `rank_vote + elite_mean` 6%. PushT seed42 remained near floor under the older selected-output rank method: standard 0%, compute 1%, robust 3%.

## Decision

This route is not strong enough for a Paper1 main method claim. It can be written as a planner-side diagnostic intervention / no-go result:

- ACPC/ranking diagnostics correctly suggested that CEM candidate rankings are unstable under perturbed views.
- A CEM-consistent rank-vote final elite rerank gives a small TwoRoom gain.
- The gain is not clearly above compute-matched CEM and does not transfer to Reacher in the seed42 check.

Recommended framing: keep this as an appendix/bounded negative result unless future work moves beyond final CEM reranking, such as a learned standard-vs-robust selector, representation-level denoising, or training-time ACPC objectives.

Primary machine-readable artifact: `assets/paper1_data/robust_cem_eval100x3_iteration_summary_20260705.json`.
