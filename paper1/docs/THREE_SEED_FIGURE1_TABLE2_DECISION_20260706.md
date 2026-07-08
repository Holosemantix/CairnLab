# Three-seed Figure 1 / main Table 2 decision

Date: 2026-07-06  
Branch: `ag/dev`

This note records the presentation decision made after the post-update review.

## Decision

Use the main sweep figure, not a main endpoint table, as the three-training-seed evidence.

Specifically:

1. Regenerate the main Gaussian sweep figure so that each point is the mean across LeWM training seeds 3072/3073/3074.
2. Compute each training-seed point by first averaging evaluation seeds 42/43/44.
3. Use population std across the three training seeds for error bars.
4. Show at least unperturbed evaluation and observation-only Gaussian evaluation at sigma=0.08 with a clean goal image.
5. Delete the current main three-seed endpoint table.
6. Move exact all-seed Gaussian sweep/evaluation values to appendix and generated artifacts.

## Rationale

This is cleaner than keeping a small endpoint table because the updated Figure 1 will already show the full plateau story across training seeds. Removing the main table avoids point-best / leaderboard wording such as `best obs`, `gain`, `gap`, or `regret`, while appendix artifacts preserve exact numeric reproducibility.

## Required artifact targets

Generate:

```text
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.md
```

These artifacts should include task, training stdmax, per-training-seed values for each evaluation condition, mean/std across training seeds, and source manifests. Do not invent or interpolate missing evaluation severities.

## Main-text wording target

After the updated figure, use prose like:

```tex
Because \Cref{fig:sweep} aggregates the full Gaussian sweep across three training seeds, we do not report a separate point-best or endpoint leaderboard in the main text. Exact all-seed sweep values are reported in Appendix~\ref{sec:appendix-gaussian-evals} for reproducibility.
```
