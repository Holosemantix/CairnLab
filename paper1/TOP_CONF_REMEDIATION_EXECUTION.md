# Top-Conference Remediation Execution Matrix

Source plan: `paper1/TOP_CONF_REMEDIATION_PLAN.md`.

This file separates remediation work that can be completed from existing checkpoints/artifacts from work that requires new training, new held-out evaluation, or new task-semantic annotations. It is intentionally conservative: no result is upgraded to a prospective or semantic-margin claim unless the artifact actually supports that reading.

## A. Completed without retraining

| Plan item | Action taken | Files/artifacts | Claim status |
|---|---|---|---|
| P1.2 ACPC vs simpler metrics | Added a no-retraining frozen-rule diagnostic audit over existing LeWM/PLDM Phase-0 full-grid artifacts. The fixed rule ranks nonzero-noise checkpoints by ACPC-H/transition, PCC, CRA, and MAF without using closed-loop success. | `tools/paper1_no_retrain_remediation.py`, `assets/paper1_data/no_retrain_diagnostic_audit.json`, `assets/paper1_data/no_retrain_diagnostic_audit.md`, `paper1/main.tex` | Supports triage/localization value; not true held-out prospective validation. |
| P1.1 representative-row clarity | Main text now explicitly reports the frozen-rule selected row against the closed-loop best row, exposing gaps instead of hiding point selection. | `paper1/main.tex`, `assets/paper1_data/no_retrain_diagnostic_audit.*` | Reduces cherry-picking risk; full unification still needs a final row policy decision before submission. |
| P2.1/P2.2 title/abstract/contribution framing | Abstract and C3 now state the no-retraining audit and keep held-out prospective validation / semantic guards out of scope. | `paper1/main.tex` | Stronger positive contribution while preserving scope. |
| Figure wording cleanup | Paper-facing Figure 1/3/5 generated labels now use `ACPC rollout readout R_F`, and unused legacy figures are removed from arXiv copy lists. | `tools/paper1_selective_contraction.py`, regenerated PNGs, `paper1/README.md`, `paper1/check_arxiv_ready.sh` | Removes stale H=8/predictor wording from paper-facing assets. |
| Artifact mapping | Added the new no-retraining audit artifact and script to the reproducibility map. | `paper1/main.tex` | Reproducible from existing JSON; no model loading. |

## B. Directly doable next from existing artifacts/checkpoints

| Plan item | Concrete next step | Inputs | Expected manuscript change |
|---|---|---|---|
| P1.4 reduce metric load | Move ADM/SPRR and some proxy diagnostics deeper into appendix, keeping closed-loop success, ACPC/R_F, PCC/CRA, and collapse guard in the main text. | Existing `main.tex` tables and artifacts | Shorter main narrative; no new computation. |
| P1.1 final row policy | Choose one final policy: fixed `std_max=0.08`, frozen diagnostic rule, or plateau summary. Then align captions and compact tables. | Existing sweep / Phase-0 / ACPC-basin artifacts | Less reviewer concern about representative-row drift. |
| P1.2 appendix scatter/table | Extend `paper1_no_retrain_remediation.py` to write a small CSV/LaTeX table or scatter plot for encoder radius vs ACPC/PCC/CRA/MAF correlations. | Existing Phase-0 artifacts | Stronger incremental-value evidence; still not held-out. |
| P0.1 quasi-prospective boundary | Recast existing frozen-rule audit as a development-only diagnostic audit and add a short paragraph specifying what would constitute the real held-out split. | Current no-retrain audit | Cleaner boundary between done-now evidence and required future validation. |
| P2.3 limitations tone | Consolidate repeated “not predictor” caveats into the limitations section now that the frozen-rule audit exists. | `main.tex` | Less self-undermining framing. |

## C. Requires new training/evaluation or new semantic construction

| Plan item | Why existing artifacts are insufficient | Required work |
|---|---|---|
| P0.2 independent training seeds as main Gaussian statistics | Existing main Gaussian grid is one training seed with three eval seeds. The 3073/3074 artifacts are unseen-stressor checks, not full matched Gaussian training-seed grids. | Train/evaluate at least base and fixed/high-noise rows for >=3 independent training seeds per task, or complete a multi-seed sweep. |
| P0.1 true prospective validation | Existing Phase-0 audit uses artifacts generated after the sweep was known. It is a fixed-rule sanity check, not a dev/held-out protocol. | Freeze metric/rule on a development split; evaluate on held-out training seeds/checkpoints/perturbation families. |
| P0.3 task-semantic discriminability guard | Current ADM is an action-distance latent proxy; rank/transition/ID probes are not task-semantic margins. | Construct task-specific semantic pairs from simulator/state metadata: PushT pose/contact, TwoRoom room/door topology, Reacher joint/target geometry, Cube object/gripper pose. |
| P1.3 stronger baseline comparison | Existing target-view and heteroscedastic branches are negative checks, not a robust baseline suite. | Add at least one explicit baseline such as encoder consistency / latent consistency / standard augmentation control, or keep baseline comparisons out of scope. |

## D. Recommended submission path

1. Finish the no-retraining manuscript cleanup in section B.
2. Decide whether the target venue requires the P0 items as completed main evidence. If yes, do not submit until P0.1/P0.2/P0.3 are actually run.
3. If no new training budget is available, submit as a bounded diagnostic paper with the frozen-rule audit clearly labeled as non-held-out evidence.
4. If training budget is available, prioritize matched Gaussian independent training seeds first, then true held-out diagnostic validation, then semantic discriminability guards.
