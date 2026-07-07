# Paper1 Unseen ATR/SMPR Summary

Values are population mean over training seeds 3072/3073/3074. ATR drop is no-noise minus noise-trained ATR under the same unseen stressor; SMPR gain is noise-trained minus no-noise SMPR under the same unseen stressor.

| Task | stressor | no-noise score | noise-trained score | ATR drop | SMPR gain |
|---|---:|---:|---:|---:|---:|
| TwoRoom | gaussian_blur 15 | 47.67 +/- 5.44 | 90.78 +/- 5.38 | 0.37 | 0.61 |
| Reacher | gaussian_blur 15 | 22.00 +/- 3.78 | 71.22 +/- 1.10 | 2.28 | 0.38 |
| PushT | resize 0.25 | 63.44 +/- 14.05 | 66.33 +/- 8.38 | 0.24 | 0.03 |
| Cube | resize 0.25 | 57.00 +/- 1.96 | 56.11 +/- 0.57 | -0.24 | -0.03 |

Seed-row correlations are descriptive for this bounded scope check, not a formal transfer theorem.

```json
{
  "pearson_stress_delta_vs_ATR_drop": 0.768142082157261,
  "pearson_stress_delta_vs_SMPR_gain": 0.8705024599334041,
  "seed_rows_n": 12,
  "spearman_stress_delta_vs_ATR_drop": 0.8391608391608392,
  "spearman_stress_delta_vs_SMPR_gain": 0.8721554530812411
}
```
