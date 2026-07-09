# Paper1 Experiment Remediation Status

Implemented from `codex_paper1_experiment_remediation_plan.md`:

- M1 full-sweep diagnostic dynamics from retained evaluation and diagnostic summaries.
- M2 held-out diagnostic-region validation for leave-one-seed-out and leave-one-task-out splits.
- M3 retained-summary fixed-pool top-1/proxy audit over the full sweep.
- M3 full-sweep sample-level fixed-pool event-rate audit recomputed from checkpoints for all 108 task--seed--stdmax rows.
- M3 endpoint sample-level fixed-pool certificate audit retained as a compact base/std0.08 view.
- M4 finite-difference Gaussian sensitivity audit over base/onset/endpoint checkpoints, using 100 sampled sequences and 5 noise draws per small sigma.
- M4 exact-autograd JVP/Hutchinson Gaussian sensitivity decomposition over base/onset/endpoint checkpoints, using 100 sampled sequences and 8 Rademacher probes per checkpoint.
- M5 joint ATR plus guard-side validation: SMPR and fixed-pool top-1 flip are reported only with ATR, not as standalone robustness metrics.
- M6 recovery-threshold and clean-tolerance sensitivity, with unavailable ATR/SMPR tail/quantile variants explicitly marked.

Not implemented in this training-free pass:

- Full Jacobian materialization, SVD, or exact semantic attribution of encoder-side repair versus rollout-side contraction; the JVP/Hutchinson audit estimates local Frobenius traces only.
- Stronger hand-labeled or simulator-derived SMPR semantics; the implemented M5 audit is a joint guard-side validation, not a new oracle-label metric.

The paper text should only claim the implemented retained-summary, finite-difference, and exact-JVP/Hutchinson local-trace evidence.

## Review-sync status (2026-07-09)

- Conclusion wording is synchronized with the current evidence stack: full-sweep low-ATR/high-SMPR regions, held-out seed/task diagnostic separation, full-sweep fixed-pool event-rate recomputation, and complementary finite-difference plus exact-JVP/Hutchinson sensitivity evidence.
- The radius--margin overlay caption is synchronized: the retained-summary overlay remains q50/q90, while the separate full-sweep sample-level recomputation reports q10/q95 gaps and event rates. The q10/q95 gaps remain negative, so the audit supports the mechanism rather than a calibrated certificate.
- Event-rate uncertainty is included through Wilson intervals for cert-pass, top-1 flip, and top-1 flip conditional on cert-pass.
- M4 decomposition is implemented as an exact-autograd JVP/Hutchinson local-trace audit over base/onset/endpoint checkpoints for all four tasks and training seeds 3072/3073/3074, using 100 sampled sequences and 8 Rademacher probes per checkpoint.
- SMPR semantics remains the real limitation. The current paper uses programmatic task-state proxy labels and a joint fixed-pool top-1 flip guard; it does not claim hand-labeled or simulator-derived oracle semantic labels, and it does not claim a standalone action-distinct semantic guard.

## Feasibility note for stronger SMPR/action guards

The lowest-risk extension is a task-independent action-distinct fixed-pool guard, not a new oracle-semantic SMPR. The existing fixed 65-candidate pool machinery already recomputes clean/noisy candidate costs and top-1 flips, so a follow-up audit could select action/cost-distinct candidate pairs within each sampled state and test whether projected rollouts or costs preserve those separations under perturbation. This would directly align with the radius--margin theory, but it would still be a fixed-pool planning guard rather than an oracle semantic label.

Simulator-derived semantic labels are possible only if each task's logged state exposes stable task semantics. TwoRoom and Reacher are relatively straightforward from position/target geometry. PushT and Cube are heavier because contact, topology, object-goal relation, and action-value semantics require task-specific thresholds or simulator contact signals; without those signals, they remain programmatic proxy labels. Hand-labeled semantic guards are therefore outside the current training-free pass.
