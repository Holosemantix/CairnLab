# Selector plateau audit

- Tolerance: 5.0 pp
- Unit: task-training-seed block over the eight nonzero Gaussian candidates.
- Reading: plateau localization and bad-checkpoint triage, not point-optimal std selection.

## Selection summaries

| Selector | plateau hit | bad picks | point regret mean±std | regret-to-plateau mean±std |
|---|---:|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 10/12 | 2/12 | 2.25±2.51 | 0.33±0.92 |
| Fixed std=0.08 | 10/12 | 2/12 | 2.14±2.18 | 0.17±0.40 |
| MAF only | 10/12 | 2/12 | 1.89±2.80 | 0.44±1.29 |
| Random nonzero std (exact expectation) | 8.5/12 | 3.5/12 | 7.02±11.72 | 4.33±10.67 |

## Decisive-pair ranking summaries

| Ranker | decisive pairs | accuracy |
|---|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 141 | 88.3% |
| ACPC only | 141 | 87.2% |
| PCC only | 141 | 87.9% |
| CRA only | 141 | 87.2% |
| MAF only | 141 | 92.6% |
| Monotone high-std baseline | 141 | 92.9% |
