# Top-Conference Remediation Execution Matrix

Source plan: `paper1/TOP_CONF_REMEDIATION_PLAN.md`.

This file separates remediation work completed from existing checkpoints/artifacts from work that still needs new diagnostic runs or task-semantic construction. It is intentionally conservative: completed Gaussian training-seed statistics are promoted to main evidence, while task-semantic margins are only specified until their state-margin runs exist.

## A. Completed without retraining

| Plan item | Action taken | Files/artifacts | Claim status |
|---|---|---|---|
| P0.2 independent training seeds as main Gaussian statistics | Promoted the completed Gaussian sweeps for canonical seed 3072 plus lockbox seeds 3073/3074 to a main training-seed mean/std table. | `tools/paper1_training_seed_lockbox.py`, `assets/paper1_data/training_seed_gaussian_lockbox.*`, `paper1/main.tex`, `DATA_MANIFEST.md` | Main Gaussian behavior no longer relies only on evaluation-seed variance. |
| P0.1 prospective validation protocol, current slice | Added a frozen validation ledger with a three-training-seed unseen score aggregate over seeds 3072/3073/3074 and a matched diagnostic slice over seeds 3073/3074 and blur/resize. The composite signed-rank rule hits the top 4/4 rows for stress-success delta and drop improvement in the matched diagnostic slice. | `tools/paper1_validation_remediation.py`, `assets/paper1_data/prospective_validation_summary.*`, `paper1/main.tex` | Upgrades evidence beyond representative-row explanation; full held-out checkpoint-grid diagnostics remain next work. |
| P0.3 semantic discriminability protocol | Added a task-semantic state-margin protocol table for PushT, TwoRoom, Reacher, and Cube; proxy rank/transition/ID checks are no longer presented as semantic margins. | `paper1/main.tex`, `assets/paper1_data/prospective_validation_summary.*` | Protocol frozen; result table requires state-margin runs. |
| P1.2 ACPC vs simpler metrics | Added a no-retraining frozen-rule diagnostic audit over existing LeWM/PLDM Phase-0 full-grid artifacts. The fixed rule ranks nonzero-noise checkpoints by ACPC-H/transition, PCC, CRA, and MAF without using closed-loop success. | `tools/paper1_no_retrain_remediation.py`, `assets/paper1_data/no_retrain_diagnostic_audit.*`, `paper1/main.tex` | Supports triage/localization value and exposes boundary cases. |
| P1.1 representative-row clarity | Main text now reports both a fixed high-noise endpoint/plateau framing and the frozen-rule selected row against the closed-loop best row. | `paper1/main.tex`, `assets/paper1_data/no_retrain_diagnostic_audit.*`, `assets/paper1_data/training_seed_gaussian_lockbox.*` | Reduces cherry-picking risk; exact point-best rows are treated as plateau context. |
| P1.4/P2.3 metric-load and tone cleanup | Main text now organizes diagnostics around three questions: seed-level behavior, rollout/cost consistency, and selective discriminability. Caveats are consolidated in Discussion instead of repeated throughout the contribution story. | `paper1/main.tex` | Clearer main contribution with honest boundaries. |
| P2.1/P2.2 title/abstract/contribution framing | Abstract and C3 now foreground the three-training-seed Gaussian result, three-seed unseen score check, validation ledger, matched held-out diagnostic slice, and semantic state-margin protocol. | `paper1/main.tex` | Stronger positive contribution while preserving scope. |
| Figure wording cleanup | Paper-facing Figure 1/3/5 generated labels now use `ACPC rollout readout R_F`, and unused legacy figures are removed from arXiv copy lists. | `tools/paper1_selective_contraction.py`, regenerated PNGs, `paper1/README.md`, `paper1/check_arxiv_ready.sh` | Removes stale H=8/predictor wording from paper-facing assets. |
| Artifact mapping | Added the no-retraining audit, training-seed Gaussian lockbox, audited unseen score artifacts, and validation remediation artifacts/scripts to the reproducibility map and consistency checks. | `paper1/main.tex`, `DATA_MANIFEST.md`, `tools/check_paper1_consistency.py` | Reproducible from existing JSON/Markdown; no model loading. |

## B. Directly doable next from existing artifacts/checkpoints

| Plan item | Concrete next step | Inputs | Expected manuscript change |
|---|---|---|---|
| P1.2 appendix scatter/table | Extend `paper1_no_retrain_remediation.py` to write a small CSV/LaTeX table or scatter plot for encoder radius vs ACPC/PCC/CRA/MAF correlations. | Existing Phase-0 artifacts | Stronger incremental-value appendix evidence; still not a full held-out study. |
| P1.1 final row policy polish | Keep main text on fixed `std_max=0.08` endpoint plus plateau summaries; ensure appendix captions use the same wording. | Existing sweep / Phase-0 / ACPC-basin artifacts | Less representative-row drift across captions. |
| P1.4 appendix pruning | Move lower-priority ADM/SPRR/full five-layer tables deeper into appendix if page budget is tight. | Existing `main.tex` tables | Shorter main narrative without deleting release artifacts. |

## C. Requires new diagnostic runs, evaluation, or semantic construction

| Plan item | Why existing artifacts are insufficient | Required work |
|---|---|---|
| Full held-out checkpoint-grid prospective validation | The current unseen score aggregate covers all three training seeds, and the matched diagnostic slice covers held-out seeds and unseen perturbations, but not the full ACPC/PCC/CRA/MAF grid for seeds 3073/3074. | Run fixed ACPC/PCC/CRA/MAF diagnostics over the completed seed-3073/3074 Gaussian grids and evaluate ranking/drop prediction on held-out checkpoints. |
| Task-semantic discriminability result table | Current ADM/SPRR are action-distance proxy guards; main text now freezes semantic factors but does not report state-margin pass rates. | Construct task-specific semantic pairs from simulator/state metadata and report same-state radius vs different-state semantic margin pass rates. |
| P1.3 stronger baseline comparison | Existing target-view and heteroscedastic branches are negative checks, not a robust baseline suite. | Add an explicit baseline such as encoder consistency, latent consistency, or standard augmentation control; otherwise keep baseline comparisons scoped as context. |

## D. Recommended submission path

1. Use the current revision as the no-retraining top-conference cleanup: three training seeds are now in the main Gaussian result and the unseen score check, while matched diagnostics are explicitly scoped to seeds 3073/3074.
2. If time remains before submission, prioritize full held-out checkpoint-grid diagnostics for seeds 3073/3074.
3. Next priority is the task-semantic state-margin result table.
4. Treat method objectives and stronger baseline suites as follow-up unless the target venue demands them for acceptance.
