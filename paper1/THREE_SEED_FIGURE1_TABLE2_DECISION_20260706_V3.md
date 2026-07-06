# Three-seed Figure 1 and main Table 2 decision

Decision: update the main Gaussian sweep figure to aggregate LeWM training seeds 3072, 3073, and 3074, then delete the current main endpoint table.

The updated figure should compute each training-seed point by first averaging evaluation seeds 42/43/44, then plot mean plus population standard deviation across the three training seeds. It should show at least clean evaluation and observation-only Gaussian evaluation at sigma 0.08 with a clean goal image over stdmax 0.0 through 0.08.

Exact all-seed Gaussian sweep and evaluation values should move to appendix and generated artifacts. The required artifact targets are assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json and assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.md. Do not invent or interpolate missing evaluation severities.

Rationale: the three-seed sweep figure will carry the plateau evidence directly. Removing the main endpoint table avoids point-best or leaderboard wording such as best obs, gain, gap, and regret, while appendix artifacts preserve numeric reproducibility.
