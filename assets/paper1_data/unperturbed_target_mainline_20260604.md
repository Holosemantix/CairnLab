# Perturbed-Input -> Original-Target Mainline Sweep

Scope: LeWM target-view ablation, 4 tasks x 8 nonzero train-time perturbation levels.

| Task | branch | best std | origin eval | pixels 0.08 | delta vs full-seq px0.08 | delta vs base px0.08 |
|---|---|---:|---:|---:|---:|---:|
| TwoRoom | perturbed input -> original target | 0.04 | 93.00 | 73.67 | -24.00 | 8.00 |
| PushT | perturbed input -> original target | 0.02 | 88.33 | 8.33 | -80.67 | 4.00 |
| Reacher | perturbed input -> original target | 0.07 | 59.67 | 30.00 | -54.67 | 11.67 |
| Cube | perturbed input -> original target | 0.02 | 66.67 | 42.67 | -25.67 | -4.33 |

Comparison rows use the best pixels 0.08 checkpoint within each branch. The full-sequence branch is read from the canonical LeWM sweep JSON.
