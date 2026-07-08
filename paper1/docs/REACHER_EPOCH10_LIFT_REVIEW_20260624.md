# Reacher epoch-10 lift review — Paper 1

This note addresses an important Reacher-specific phenomenon: under the fixed epoch-10 protocol, LeWM noise training improves not only observation-noise robustness but also unperturbed evaluation. This should be explicitly analysed. Otherwise readers may think the paper is mixing robustness recovery with a training-regularization or undertraining effect.

## What the paper currently shows

Current main text already contains the relevant facts:

- Reacher no-noise LeWM baseline: unperturbed `58.67`, observation-noise 0.08 `18.33`.
- Reacher representative/high grid rows: unperturbed around `86.00` at `std_max=0.06`, observation-noise 0.08 `84.67` at `std_max=0.07`.
- Reacher diagnostics: effective rank increases slightly (`61.04 -> 65.92`), transition-resolution L2 slightly improves (`0.3704 -> 0.3791`), and rollout T8 drift collapses (`15.17 -> 0.44`).

So Reacher is not just a robustness recovery case. It is also a **fixed-epoch clean-control lift**.

## Why this matters

If left unexplained, a strict reviewer will ask:

1. Is noise augmentation improving robustness, or simply improving training/optimization at epoch 10?
2. Is the no-noise Reacher baseline undertrained or unstable at epoch 10?
3. Does the paper’s ACPC framing explain the clean lift, or only the noisy lift?
4. Is this effect method-specific, since PLDM Reacher is already strong without noise augmentation?

The paper should answer these questions conservatively.

## Safe interpretation

The safe interpretation is:

> On Reacher, input-side noise acts as a fixed-epoch representation/predictor regularizer, not only as a test-time corruption match. The key diagnostic movement is not resolution collapse: rank, transition resolution, and controllability proxies stay flat or improve. The dominant movement is a large reduction in multi-step rollout drift. This is consistent with a smoother, more stable action-conditioned predictor that benefits both clean and noisy MPC at epoch 10. Because the sweep lacks independent training seeds and later-epoch learning curves, this should be treated as a fixed-protocol diagnostic finding rather than a claim about asymptotic performance.

This interpretation fits the multi-stage diagnostic mainline:

```text
noise intervention -> encoder geometry remains/resolution preserved -> rollout drift collapses -> clean and noisy CEM planning improve
```

It does not overclaim causality.

## Required manuscript addition

### Add after the sweep paragraph in `sec:exp-sweep`

Suggested text:

```tex
Reacher shows an additional fixed-epoch effect. Noise training improves not only observation-noise robustness but also unperturbed control: the representative clean row rises from $58.67\%$ at the no-noise baseline to about $86\%$ under noise training. We interpret this as a fixed-protocol regularisation/stabilisation effect rather than as pure robustness recovery. The diagnostic table supports this reading: Reacher does not show a resolution collapse (effective rank and transition-resolution stay flat or slightly improve), while multi-step rollout drift drops sharply ($15.17\to0.44$). Thus the same intervention that contracts same-state noisy rollouts appears to stabilise the clean action-conditioned predictor at epoch 10. This is a diagnostic observation, not a claim about asymptotic training behaviour; independent training seeds and longer training curves would be needed to separate regularisation from optimisation timing.
```

This paragraph should be placed before the ACPC-basin subsection or at the end of the sweep subsection.

### Adjust the Reacher bullet in `sec:exp-diag`

Current bullet already says Reacher rank/resolution/probe remain flat and rollout drift collapses. Add one sentence:

```tex
This also explains why Reacher's unperturbed score rises under the fixed epoch-10 protocol: the noise-trained checkpoint appears to stabilise rollout prediction without erasing the low-dimensional control state.
```

But keep the caveat:

```tex
Because this is one trained checkpoint per noise level, we do not interpret the clean lift as an asymptotic training advantage.
```

## PLDM cross-check

Add a short sentence in the PLDM appendix or main PLDM mention:

```tex
PLDM is a useful boundary case for this interpretation: its Reacher baseline is already high under no-noise training, so the LeWM Reacher clean lift should be read as a LeWM fixed-epoch stabilisation phenomenon, not as a universal property of noise training.
```

This prevents overgeneralization.

## Things not to claim

Do not write:

- `Noise training improves Reacher asymptotic performance`.
- `Gaussian noise is a better Reacher training objective`.
- `The Reacher clean lift proves ACPC causes better control`.
- `Noise training always regularizes low-dimensional control tasks`.

Use:

- `fixed-epoch effect`;
- `regularisation/stabilisation under this protocol`;
- `consistent with reduced rollout drift`;
- `not a training-seed or asymptotic claim`.

## Why this strengthens the paper

This point actually supports the representation-diagnostic mainline. It shows that the intervention is not merely matching the evaluation corruption. It can also reshape the learned model in a way that improves clean planning at a fixed training budget. The diagnostics explain where this happens: not by erasing Reacher state resolution, but by strongly stabilising action-conditioned rollout dynamics.
