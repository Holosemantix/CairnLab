# Paper1 Experiment Remediation Status

Implemented from `codex_paper1_experiment_remediation_plan.md`:

- M1 full-sweep diagnostic dynamics from retained evaluation and diagnostic summaries.
- M2 held-out diagnostic-region validation for leave-one-seed-out and leave-one-task-out splits.
- M3 retained-summary fixed-pool top-1/proxy audit over the full sweep.
- M3 endpoint sample-level fixed-pool certificate audit recomputed from checkpoints for base and std0.08 rows.
- M6 recovery-threshold and clean-tolerance sensitivity, with unavailable tail/quantile variants explicitly marked.

Not implemented in this training-free pass:

- M3 full-sweep sample-level pool cert-pass rates; the endpoint base/std0.08 recomputation is implemented, but full 108-row sample-level recomputation remains optional due runtime.
- M4 Gaussian finite-difference/JVP sensitivity audit, because it requires checkpoint-level forward/JVP computation.
- M5 stronger SMPR/action-distinct guard, because it requires stronger labels or action-distinct trace computation.

The paper text should only claim the implemented retained-summary evidence.
