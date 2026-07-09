# Paper1 Experiment Remediation Status

Implemented from `codex_paper1_experiment_remediation_plan.md`:

- M1 full-sweep diagnostic dynamics from retained evaluation and diagnostic summaries.
- M2 held-out diagnostic-region validation for leave-one-seed-out and leave-one-task-out splits.
- M3 retained-summary fixed-pool top-1/proxy audit over the full sweep.
- M3 full-sweep sample-level fixed-pool event-rate audit recomputed from checkpoints for all 108 task--seed--stdmax rows.
- M3 endpoint sample-level fixed-pool certificate audit retained as a compact base/std0.08 view.
- M4 finite-difference Gaussian sensitivity audit over base/onset/endpoint checkpoints.
- M5 joint ATR plus guard-side validation: SMPR and fixed-pool top-1 flip are reported only with ATR, not as standalone robustness metrics.
- M6 recovery-threshold and clean-tolerance sensitivity, with unavailable ATR/SMPR tail/quantile variants explicitly marked.

Not implemented in this training-free pass:

- Exact JVP/Hutchinson Gaussian sensitivity decomposition; the implemented M4 audit is finite-difference only.
- Stronger hand-labeled or simulator-derived SMPR semantics; the implemented M5 audit is a joint guard-side validation, not a new oracle-label metric.

The paper text should only claim the implemented retained-summary evidence.
