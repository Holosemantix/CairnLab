# Paper 1 variance-aware five-reviewer audit — 2026-06-24

This audit focuses on a subtle but important issue: single-grid, single-training-run maxima under three evaluation seeds must not be written as statistically unique optima. This affects Figure 2, sweep tables, ACPC-basin representative rows, PLDM appendix rows, heteroscedastic comparison rows, and Phase-0 rows.

## Overall verdict

The theory section remains sound and useful. The remaining major writing risk is not theory hallucination; it is over-reading noisy sweep maxima. The manuscript should consistently distinguish:

- **highest observed grid mean**: the maximum mean in the released grid;
- **plateau/reference marker**: a visual marker for a high-performing region;
- **representative high-corruption checkpoint**: a row chosen for compact comparison;
- **statistically unique optimum**: not supported by the current design.

The current canonical grid has three evaluation seeds per checkpoint and one trained epoch-10 checkpoint per `std_max` cell. Therefore, terms like `optimum`, `robust-eval optimum`, and unqualified `best` should be avoided unless immediately qualified as highest observed mean in the grid.

A patch script has been added:

```bash
python paper1/docs/apply_plateau_uncertainty_patch.py
```

It updates `paper1/main.tex` and `tools/paper1_figs.py` wording and figure labels to remove the strongest optimum-like language. After running it, regenerate Figure 2:

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
```

Then rebuild the paper.

---

## Reviewer 1 — Statistical reviewer

**Finding.** Figure 2 currently marks a single dashed green line labeled by the renderer as `Robust-eval optimum sigma*`. This is not statistically justified. Many nearby `std_max` values are within evaluation variability, especially on PushT, Reacher, and TwoRoom plateaus.

**Required fix.** Treat the dashed line as a reference marker for the highest observed corrupted-eval mean in the grid, not as an optimum. The patch changes the renderer label to:

```text
Highest observed robust eval (not unique)
```

and panel title to:

```text
(ref. sigma=...)
```

**Related fixes.** Main-text table captions should say parenthesised values are highest observed means / plateau reference points, not point-best optima. Sweep narrative should explicitly say that high-performing ranges overlap and the robust pattern is plateau/task dependence rather than a unique scalar optimum.

---

## Reviewer 2 — Theory reviewer

**Finding.** The ACPC theory section remains mathematically correct. The claims are fixed-candidate-set sufficient conditions under Lipschitz cost readout and margin assumptions. The theory does not depend on the uniqueness of the empirical `std_max` marker.

**Required caution.** Do not use the theory to justify selecting a unique training-noise level. The theory connects ACPC to candidate-cost/ranking stability, not to hyperparameter optimality.

**Safe wording.**

- `fixed-candidate-set stability condition`
- `candidate-cost drift bound`
- `highest observed grid point`
- `plateau/reference marker`

**Unsafe wording.**

- `robust-eval optimum`
- `optimal sigma`
- `unique best noise level`
- `ACPC selects the checkpoint`

---

## Reviewer 3 — RL/world-model reviewer

**Finding.** The task-level story is stronger if it emphasizes plateaus rather than exact point optima. For control, robust performance is often flat over a range of augmentation strengths; claiming a precise `sigma*` makes the paper look overfit to noise in the evaluation estimate.

**Required fix.** Use `representative high-corruption checkpoint` for compact ACPC-basin tables and `highest observed grid point` for appendix sweep markers.

**Interpretation to keep.** It is still valid to say noise augmentation recovers performance and that the high-performing range differs by task. The evidence supports task-dependent plateau structure, not a unique scalar optimum.

---

## Reviewer 4 — Writing / AI-feel reviewer

**Finding.** The paper is much less defensive than before, but `point-best`, `optimum`, and `selected endpoint` language still creates a brittle impression. A human reviewer will immediately ask whether the single maximum is meaningful under the reported evaluation std.

**Required fix.** Replace strong maxima language with:

- `highest observed mean in this grid`
- `reference marker`
- `representative high-corruption checkpoint`
- `plateau region`

Avoid repeated reminders in every paragraph; one caption/paragraph note per table/figure is enough.

---

## Reviewer 5 — Artifact/release reviewer

**Finding.** Patching the manuscript text is not enough. Figure 2 is script-generated, so the label inside the PNG must be regenerated after changing `tools/paper1_figs.py`.

**Required commands.**

```bash
python paper1/docs/apply_plateau_uncertainty_patch.py
python -m tools.paper1_figs --out-dir assets/paper1_figs
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

If `main.tex` still contains the author placeholder, `paper1/check_arxiv_ready.sh` should fail. That is correct until the real author list is filled.

---

## Similar issues to check after patch

Run:

```bash
rg -n "optimum|optimal sigma|sigma\\^\\ast|point-best|noise-best|best checkpoint|selected endpoint|Robust-eval optimum|unique optimum" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

Expected state:

- No `Robust-eval optimum` in `tools/paper1_figs.py`.
- No unqualified `optimum` or `optimal sigma` in `main.tex`.
- `point-best` should be gone from main text and reduced in appendix captions; if it remains in historical tooling docs, it must be clearly framed as highest observed grid mean, not statistical optimum.

---

## Bottom line

The paper's theoretical core is now strong enough for the diagnostic contribution. The next crucial polish is variance-aware language around sweep maxima. Once this patch is applied and Figure 2 is regenerated, the paper will read as a careful plateau/diagnostic study rather than as a hyperparameter-optimization claim.
