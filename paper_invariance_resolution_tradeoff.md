# Understanding the Invariance–Resolution Trade-off in Latent Predictive World Models: A Diagnostic Study of JEPA-Based Control

*Chinese version: [paper_invariance_resolution_tradeoff_zh.md](paper_invariance_resolution_tradeoff_zh.md).*

---

## Abstract

Joint-Embedding Predictive Architectures (JEPAs) are commonly *believed* to learn abstract, invariant world representations: by predicting in latent space rather than reconstructing pixels, the encoder is expected to discard visual redundancy and noise. This expectation is a *community heuristic* rather than a published guarantee — to our knowledge no JEPA work formally claims pixel-noise robustness for control. We test the implicit assumption on **LeWorldModel (LeWM)**, a published JEPA world model, across four manipulation and navigation tasks (PushT, TwoRoom, Reacher, Cube) and eight levels of train-time pixel-noise augmentation. We find three things:

1. **Visual OOD collapse in JEPA + CEM control.** Without noise-aware training, LeWM collapses under mild pixel noise: PushT control success drops from 87.33% (clean) to 3.67% (Gaussian std = 0.08, near-random); TwoRoom drops from 93.00% to 44.33%.

2. **No global-optimal noise level exists.** Tasks respond very differently to noise augmentation. Visually redundant navigation (TwoRoom) benefits from heavy noise (best at std = 0.008), whereas contact-heavy control (PushT) reaches peak *clean* at std = 0.002 but peak *robustness* at std = 0.006 — clean and robust optima dissociate within a single task.

3. **A five-layer diagnostic protocol explains the underlying compression mechanism.** Instrumenting encoder shift, encoder geometry, predictor sensitivity, latent-noise response, and task resolution traces noise-induced control failure to a representational chain: representation compression (drop in effective rank) → loss of transition-key resolution (drop in `transition_resolution_ratio`) → loss of controllability (drop in `id_probe_r²`). When used as a **cross-checkpoint predictor**, the strongest single diagnostic (`predictor_target_to_nn_cos_ratio_at_max_std`) tracks **overall ckpt quality** (clean success rate ρ ≈ −0.73 on the n = 9 LeWM PushT sweep, and within-protocol px+g 0.08 ρ ≈ −0.67 after conditioning on training-noise level). The metric does *not* predict OOD drop beyond what training std_max already explains: the apparent ρ(metric, drop) = −0.77 is reduced to ≈ 0 once std_max is partialled out. The diagnostic toolkit usefully ranks checkpoints within a fixed training protocol, but does *not* substitute for actually training with noise when the goal is OOD robustness.

This paper does not propose a new training algorithm. Its contribution is (i) a systematic empirical study of JEPA + CEM visual OOD failure, (ii) a reproducible diagnostic toolkit, and (iii) an honest delineation of what cross-ckpt diagnostics can and cannot predict.

**Keywords**: world models; JEPA; visual robustness; representation diagnostics; invariance–resolution trade-off.

---

## 1 Introduction

### 1.1 The implicit invariance heuristic and where it breaks

Since Yann LeCun proposed the Joint-Embedding Predictive Architecture (JEPA) [1], this paradigm has been advanced as a direction for self-supervised learning. Unlike generative models (VAEs, diffusion), JEPA does not reconstruct pixels; it predicts future *representations* in latent space. The motivating intuition — that predicting "what is invariant" rather than "what pixels look like" should yield abstract representations that discard visual redundancy and noise — is now part of the field's *informal vocabulary* in talks, blog posts, and survey articles [2,3].

We emphasise that this is a heuristic rather than a published guarantee. To our knowledge, no JEPA paper has *formally claimed* visual-OOD robustness for control-relevant downstream tasks. I-JEPA [2] and V-JEPA [3,4] established strong visual representations on ImageNet and video via masked prediction; LeWorldModel (LeWM) [5] extended the framework to end-to-end stable world-model training across four robotic control tasks. Existing robustness studies have probed JEPA only on image classification (N-JEPA [8]), synthetic 1D distractors (VJEPA [9]), or medical ultrasound (US-JEPA [10]); none on **JEPA-based control** under realistic pixel noise.

This leaves a basic operational question open: **if the input image is degraded by sensor noise, lighting change, or camera jitter, does a JEPA + CEM world model still plan and act reliably?**

The data say no. On PushT (2D pushing), the untrained-with-noise LeWM achieves 87.33% on clean images but falls to 3.67% under Gaussian pixel noise of std = 0.08 — essentially random. TwoRoom (2D navigation) drops from 93.00% to 44.33%. Latent prediction alone does not, in this regime, confer the visual robustness the community heuristic would predict.

### 1.2 The core tension: no globally optimal noise level

A natural remedy for the fragility above is input-side noise augmentation during training, a technique long-validated in supervised and contrastive learning [6,7]. But a deeper question arises: **does there exist a single, universal noise level that is optimal across tasks?**

A systematic sweep across four tasks at eight levels of `std_max ∈ {0.001, …, 0.008}` answers no:

- **TwoRoom** (visually redundant navigation): clean success rises monotonically with noise, peaking at std = 0.008 (98.33% / 98.67%).
- **PushT** (contact-heavy manipulation): clean peaks at std = 0.002 (90.00%), but robustness at px+goal 0.08 peaks at std = 0.006 (87.00%) — clean and robust optima dissociate.
- **Reacher** (continuous reaching): best at std = 0.006 (86.00% / 84.67%); very low noise (0.001) is statistically indistinguishable from base (55.67% vs 57.67%, within binomial sampling noise of ~2.9 pts at this protocol), with the inflection at std = 0.002 (jump to 80.33%). Reacher appears to need a *minimum* noise threshold before benefiting.
- **Cube** (structured manipulation): noise sweep is weakest; no monotone trend on clean.

This finding exposes a fundamental tension: **global noise augmentation cannot distinguish "background visual redundancy that should be made invariant" from "control-relevant features that should retain resolution".**

### 1.3 Contributions

The paper makes the following contributions:

**Contribution 1: A systematic quantification of visual-OOD fragility of JEPA + CEM control across four representative tasks.** We sweep 4 tasks × 8 noise levels with a unified sample budget of 300 trajectories per condition (either 1 seed × 300 episodes or 3 seeds × 100 episodes).

**Contribution 2: The "invariance–resolution trade-off" concept and a five-layer diagnostic protocol that operationalises it.** We define five complementary diagnostic layers (encoder shift, encoder geometry, predictor sensitivity, latent-noise response, task resolution) with 17+ concrete metrics, validated by Spearman correlations on the canonical n = 8 LeWM checkpoint set and a 9-level within-method sweep, with partial correlations conditioned on training noise to separate ckpt-quality signal from confounding-by-protocol.

**Contribution 3: A mechanistic account of why global noise augmentation has limited returns.** Through the diagnostic layers we show that on TwoRoom the gains come from desirable representation compression (drop in effective rank) — a low-dimensional discrete task does not need high resolution — whereas in PushT the same amount of compression at heavy noise drives `transition_resolution_ratio` from 0.30 to 0.10 and `id_probe_r²` from 0.77 to 0.27, erasing task-relevant state information.

**Contribution 4: A clean negative result on automated transition reweighting.** Direct heteroscedastic-NLL training (using predicted uncertainty σ to weight transitions) on PushT collapses clean success from 87.33% to 13.33%, showing that "hard" transitions and "unimportant" transitions are not interchangeable in contact-heavy control.

### 1.4 Organisation

§2 reviews related work; §3 defines the LeWM background, the noise protocol, and the diagnostic framework; §4 reports the experimental findings; §5 discusses mechanism and implications; §6 concludes.

---

## 2 Related Work

### 2.1 JEPA and latent world models

JEPA [1] predicts in latent space rather than reconstructing pixels. I-JEPA [2] predicts target representations from a masked context; V-JEPA [3,4] extends the framework to video understanding and video-driven world modelling; LeWM [5] is the first end-to-end stable JEPA world model. It uses **SIGReg** (Sketch Isotropic Gaussian Regularizer) — random projections plus Epps–Pulley empirical-characteristic-function matching [19] — to prevent representation collapse without requiring batch normalisation, and it validates latent-space planning on PushT, TwoRoom, Reacher, and Cube.

**Relation to this paper.** LeWM is the baseline system in our experiments. The original LeWM paper reports a Violation-of-Expectation experiment showing the model is sensitive to physical perturbations (object teleportation) but not to visual perturbations (colour change). Note however that (i) VoE measures prediction error (surprise), not control success rate, and (ii) colour change and pixel-level Gaussian noise are distinct corruptions. We give the first picture, to our knowledge, of LeWM's control success rate under pixel-noise corruption.

### 2.2 Robustness studies of JEPA

N-JEPA [8] introduces diffusion-noise augmentation into I-JEPA via noise-to-teacher and context-to-noise losses, improving linear-probing robustness on ImageNet. VJEPA [9] tests a "Noisy TV" distractor on synthetic 1D signals and reports JEPA retains R² > 0.84 under high noise. US-JEPA [10] tests Gaussian blur, contrast reduction, and speckle noise on medical ultrasound.

**Relation.** These works study image classification (N-JEPA), synthetic signals (VJEPA), or medical-image analysis (US-JEPA). **None studies the pixel-noise robustness of a JEPA world model on robotic control tasks.** Furthermore, VJEPA's optimistic conclusion (R² > 0.84 in 1D) contrasts with our control-time observation (success rate → 3.67%) — suggesting that the "natural robustness" of JEPA may be an artefact of evaluation modality.

### 2.3 World models and input augmentation

In RL world-model literature, DreamerV3 [11] and TD-MPC2 [12] rely on convolutional inductive bias for some implicit noise tolerance. ViGMO [13] tests Gaussian noise and blur on DMC tasks, finds "sensor noise is a fundamentally different distribution shift", and proposes a latent-consistency loss.

**Relation.** ViGMO addresses model-based RL (DrQ-v2, DreamerV3), not JEPA architectures. Its conclusion that "sensor noise is special" is directionally aligned with ours; our contribution adds *mechanistic decomposition* via the five-layer diagnostic, not just performance reporting.

### 2.4 The invariance–resolution tension

Tamkin et al. [14] argue, in the contrastive-learning context, that "label-destroying augmentations can be useful" and that augmentations act as feature dropout rather than pure invariance inducers. Zhang et al. [15] note that overly strong augmentation imposes excess invariance and erases fine-grained downstream information.

**Relation.** These insights are well-established in contrastive learning for classification, but their manifestation in *latent predictive world models* — where the downstream task is planning rather than discrimination — has not been systematically studied. The present paper extends the discussion from classification to control, with a quantitative framework.

### 2.5 Representation diagnostics and collapse analysis

Self-supervised learning broadly uses effective rank [16], condition number, and participation ratio to diagnose dimensional collapse [17]. Next-Latent Prediction [18] uses effective latent rank to assess world-model compactness. VICReg [20] provides an anti-collapse regularizer distinct from SIGReg.

**Relation.** Individual metrics (effective rank, NN distance, CKA) are not new. Our contribution is to **combine them into a coherent five-layer protocol with per-token noise sensitivity, cross-ckpt correlation, and a partial-correlation validation scheme conditioning on training noise** that ties representation properties to control performance.

---

## 3 Background and Diagnostic Framework

### 3.1 LeWorldModel baseline

LeWM [5] is an end-to-end JEPA world model trained with two terms only:

$$
\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{SIGReg}}
$$

where $\mathcal{L}_{\text{pred}}$ is the latent-space MSE between predicted and target representations. **SIGReg (Sketch Isotropic Gaussian Regularizer)** projects each latent onto $M$ unit-norm random directions, computes the Epps–Pulley empirical-characteristic-function distance [19] between each projection and $\mathcal{N}(0,1)$, and aggregates with the Gauss-window weights — preventing collapse without explicit BatchNorm. (The Cramér–Wold theorem motivates the construction: equality of high-dimensional distributions reduces to equality of all one-dimensional projections of their characteristic functions.) Inference uses the Cross-Entropy Method (CEM) for latent-space MPC.

### 3.2 Input-side noise augmentation

We add per-frame Gaussian noise to the LeWM input pipeline via `utils.py::AddNormalizedGaussianNoise`. Each frame is independently noised: Bernoulli$(p)$ decides whether the frame is corrupted, and if so the standard deviation is drawn from Uniform$(0, \text{std\_max})$. We fix $p = 1.0$ and sweep $\text{std\_max} \in \{0.001, \dots, 0.008\}$ (eight levels).

Evaluation comprises clean and noised conditions. Noised conditions use two intensities:

- **pixels+goal 0.05**: Gaussian noise of std = 0.05 on both the pixels and the goal image.
- **pixels+goal 0.08**: same, with std = 0.08.

### 3.3 Five-layer diagnostic framework

To understand how noise augmentation affects the latent representation, we define a five-layer diagnostic protocol:

**Layer 1 — Encoder shift.** Quantifies the direction and magnitude of latent displacement induced by input noise. Key metrics:
- `noise_angle_deg` — angle between clean and noisy latents.
- `noise_l2` — L2 displacement.
- `noise_to_nn_cos_ratio` — noise displacement relative to local NN cosine distance.
- `noise_angle_slope` — slope of angle vs noise std.

**Layer 2 — Encoder geometry.** Quantifies global structure of the latent space. Key metrics:
- `clean_nn_cos_dist` — local cosine distance to nearest neighbour.
- `clean_effective_rank` — effective rank of the latent covariance.
- `cka_linear_at_max_std` — Centered Kernel Alignment between clean and noisy latents.

**Layer 3 — Predictor sensitivity.** Quantifies how the predictor amplifies input noise. Key metrics:
- `predictor_target_to_nn_cos_ratio_at_max_std` — single-step predictor target shift normalised by clean NN distance. **The strongest cross-ckpt diagnostic we find.**
- `predictor_rollout_drift_T(T)` — autoregressive drift over T steps.

**Layer 4 — Latent-noise response.** Inject noise directly in the latent `z` (bypassing the encoder) to isolate predictor and cost contributions:
- `latent_cost_surface_slope_z` — slope of planning cost under perturbations of the goal latent.
- `latent_robust_radius_z` — empirical robust radius in latent space.

**Layer 5 — Task resolution.** Quantifies how much control-relevant information the latent retains:
- `transition_resolution_ratio_cos` / `transition_resolution_ratio_l2` — distinguishability of consecutive latents.
- `id_probe_r²` — R² of a linear probe predicting state ID from the latent (a controllability proxy).

### 3.4 Cross-checkpoint validation protocol

To ensure the diagnostic signals are not spurious training artefacts, we adopt two complementary analyses on the same LeWM PushT sweep:

- **Canonical n = 8.** Eight LeWM checkpoints (1 base + 7 representative noise levels). Compute Pearson and Spearman correlations of each diagnostic with the clean success rate and the OOD success rate.
- **n = 9 LeWM sweep with partial correlation.** All 9 LeWM PushT checkpoints (base + std_max ∈ {0.001, …, 0.008}). Compute Spearman ρ and *partial Spearman ρ conditioned on `std_max`*. The partialling step matters: many diagnostics correlate with control performance only because both quantities co-vary with the training noise level along the sweep.

We report **both** the raw Spearman and the partial-on-`std_max` quantities. The raw ρ tells you "which checkpoint over the entire sweep behaves better"; the partial ρ tells you "which checkpoint *within a fixed training protocol* behaves better". The two questions are distinct, and we will see in §4.5 that they have markedly different answers for the same metric.

> A larger cross-architecture protocol (varying the latent geometry of the world model itself) is left to future work; we will add a non-JEPA baseline (DreamerV3 or TD-MPC2 on the same tasks) in a follow-up version.

---

## 4 Experiments

### 4.1 Setup

**Tasks.** PushT (2D pushing), TwoRoom (2D navigation), Reacher (2D arm), Cube (3D manipulation).

**Baselines.** LeWM-base (no noise) and LeWM+noise (8-level sweep).

**Training.** Each configuration is trained with 3 random seeds (42 / 43 / 44); each seed is evaluated on 100 trajectories.

**Clean-metric definition.** All "clean" success rates reported in this paper use a unified `clean_300` baseline: the model is run for 300 trajectories without any input noise. For ckpts trained at single-seed × 300 we use the directly logged `clean_300` block in `summary.txt`; for ckpts trained at 3 seeds × 100 we average the three `clean_seed{42,43,44}` blocks (total 300 trajectories). PushT LeWM-base clean = **87.33%** in this paper follows this convention; the **86.00%** number reported in some earlier write-ups corresponds to the same ckpt at `num_eval = 150` and is not used in this paper. We retain the larger 300-trajectory budget throughout for consistency with the noise-eval protocol.

**Hardware.** Single NVIDIA A100 (80 GB) GPU; training takes 2–4 hours per task per configuration.

**Main figures.** This paper has 6 main figures rendered by `tools/paper1_figs.py` and stored in `assets/paper1_figs/`. Figure 1 (the hero) summarises the OOD cliff and per-task recovery; Figure 2 shows the per-task noise sweep; Figure 3 shows the LeWM PushT scatter of the strongest cross-ckpt diagnostic against clean vs OOD-drop; Figure 4 shows the per-task diagnostic radar; Figure 5 shows the three-layer mechanism attribution; Figure 6 shows the per-task Pareto trajectory of (clean, OOD) under the noise sweep.

### 4.2 JEPA control fragility under visual OOD

Table 1 reports LeWM-base success rates under clean and noised eval (mean ± std across 3 seeds × 100 evaluations).

**Table 1. LeWM-base under visual OOD (mean ± std; 3 seeds × 100 evaluations).**

| Task | clean | px+goal 0.05 | px+goal 0.08 | clean → 0.08 drop |
|---|---:|---:|---:|---:|
| TwoRoom | 93.00 ± 2.52 | 62.33 ± 4.04 | 44.33 ± 5.51 | **−48.67** |
| PushT   | 87.33 ± 2.31 | 15.00 ± 3.46 |  3.67 ± 1.53 | **−83.66** |
| Reacher | 57.67 ± 3.51 | 25.33 ± 4.16 | 14.67 ± 3.51 | **−43.00** |
| Cube    | 72.33 ± 3.06 | 61.33 ± 4.16 | 52.33 ± 4.51 | **−20.00** |

![Fig 1 — Visual OOD cliff in LeWM and recovery by noise training](assets/paper1_figs/fig1_hero.png)

LeWM-base is strong on clean images (especially TwoRoom and PushT), but visual std = 0.05 applied to pixels and goal jointly already produces large drops on all tasks. PushT loses 70+ pts (down to near-random 3.67%), TwoRoom 30+ pts, Reacher 30+ pts, Cube 10+ pts. **This is not a marginal phenomenon**: a JEPA + CEM world model without noise-aware training has essentially no resistance to visual corruption. The drop pattern across tasks is informative: Cube degrades least (−20 pt) — structured manipulation has some natural robustness to pixel noise — whereas PushT degrades most catastrophically (−83.66 pt), confirming that contact-heavy continuous control is most sensitive to visual precision.

### 4.3 Noise augmentation closes the gap — at the cost of task-specific tuning

Table 2 reports the complete 8-level sweep across tasks.

**Table 2. LeWM+noise sweep (4 tasks × {clean, px+g 0.08}).** All values are success rate ± standard deviation (in pts). For 3-seed × 100 runs the std is the across-seed std-of-mean; for single-seed × 300 runs it is the binomial std `sqrt(p(1-p)/300)`. Each row's total sample budget is 300 trajectories.

**(a) Clean success rate (%).**

| std_max | TwoRoom | PushT | Reacher | Cube |
|---|---:|---:|---:|---:|
| 0 (base) | 93.00 ± 1.5 | 87.33 ± 1.9 | 57.67 ± 2.9 | 72.33 ± 2.6 |
| 0.001 | 92.00 ± 1.6 | 89.67 ± 1.8 | 55.67 ± 2.9 | 73.00 ± 2.6 |
| 0.002 | 94.33 ± 1.3 | **90.00 ± 1.7** | 80.33 ± 2.3 | 64.67 ± 2.8 |
| 0.003 | 96.33 ± 2.3 | 89.67 ± 1.2 | 78.67 ± 0.9 | 65.00 ± 1.2 |
| 0.004 | 96.33 ± 1.5 | 89.33 ± 1.5 | 84.00 ± 2.1 | 69.00 ± 2.7 |
| 0.005 | 94.00 ± 1.4 | 82.00 ± 2.2 | 73.33 ± 2.6 | 61.33 ± 2.8 |
| **0.006** | 96.67 ± 1.5 | 89.33 ± 1.5 | **86.00 ± 2.1** | 66.67 ± 1.5 |
| 0.007 | 96.00 ± 1.2 | 85.67 ± 2.2 | 83.67 ± 2.3 | 67.67 ± 0.7 |
| **0.008** | **98.33 ± 0.3** | 88.33 ± 2.0 | 84.00 ± 0.6 | 62.33 ± 0.9 |

**(b) Pixels+goal noise std = 0.08 success rate (%).**

| std_max | TwoRoom | PushT | Reacher | Cube |
|---|---:|---:|---:|---:|
| 0 (base) | 44.33 ± 2.9 |  3.67 ± 1.1 | 14.67 ± 2.0 | 52.33 ± 2.9 |
| 0.001 | 84.67 ± 2.1 | 46.33 ± 2.9 | 45.33 ± 2.9 | 53.33 ± 2.9 |
| 0.002 | 91.00 ± 1.7 | 70.67 ± 2.6 | 80.67 ± 2.3 | 63.00 ± 2.8 |
| 0.003 | 94.67 ± 2.0 | 83.00 ± 2.7 | 73.67 ± 0.3 | 67.33 ± 1.3 |
| 0.004 | 95.00 ± 1.7 | 81.33 ± 2.0 | 80.00 ± 1.0 | 67.00 ± 2.5 |
| 0.005 | 94.00 ± 1.4 | 78.00 ± 2.4 | 71.33 ± 2.6 | 60.67 ± 2.8 |
| **0.006** | 96.67 ± 1.8 | **87.00 ± 2.7** | **84.67 ± 2.9** | 65.00 ± 2.1 |
| 0.007 | 96.33 ± 1.5 | 82.33 ± 3.3 | 81.33 ± 0.9 | 68.00 ± 1.0 |
| **0.008** | **98.67 ± 0.7** | 85.33 ± 1.9 | 83.00 ± 3.1 | 60.33 ± 0.7 |

Reading guidance: differences of ≤ 3 pts within a column are within ~1–2 std and should not be interpreted as meaningful; the **per-task optima** (bolded) sit ≥ 5 pts above their nearest neighbour in all four task columns.

![Fig 2 — Noise-training sweep: clean vs OOD per task; no single std_max is jointly optimal](assets/paper1_figs/fig2_sweep.png)

The same sweep, plotted as a (clean, OOD) trajectory per task, makes the trade-off visually explicit (Figure 6): each task's sweep curve moves from base far below the y = x diagonal toward the upper-right, with per-task curvature determined by how much OOD gain a task can buy from a given drop in clean. TwoRoom moves nearly along the diagonal up to (98, 98); PushT moves vertically (clean stays around 87–90 while OOD rises from 4 to 87); Reacher makes a big diagonal jump from (58, 15) to (86, 85); Cube barely moves.

![Fig 6 — Per-task Pareto trajectory of (clean, OOD) under noise sweep](assets/paper1_figs/fig6_pareto.png)

**Three observations.**

**(1) No single std_max is jointly optimal across tasks, and within a single task, clean and robustness optima can dissociate.**
- TwoRoom peaks globally at std = 0.008 (98.33 / 98.67); clean rises monotonically with noise — visually redundant tasks benefit from heavy noise.
- PushT peaks on clean at std = 0.002 (90.00), but on robustness (px+g 0.08) at std = 0.006 (87.00 vs. 0.002's 70.67; +16.33 pt). **Clean and robust optima dissociate within the task.**
- Reacher peaks at std = 0.006 (86.00 / 84.67); the apparent dip at std = 0.001 (clean 55.67 vs base 57.67) is within the binomial sampling noise (≈2.9 pts) for this protocol, so the data only support **"low noise is statistically equivalent to base"**, not "low noise hurts". The inflection occurs at std = 0.002 (jump to 80.33), suggesting Reacher needs a *minimum* invariance threshold rather than gradient improvement.
- Cube responds least to noise: clean is non-monotonic (peaks at std = 0.001 with 73.00), and px+g 0.08 improves only in the 0.003–0.007 range (67.33 vs. base 52.33). Structured manipulation is largely insensitive to global input noise.

**(2) Per-task tuning is necessary, not optional.** Optimal std_max varies substantially across tasks: TwoRoom 0.008 (heavy), PushT clean 0.002 / robust 0.006, Reacher 0.006, Cube no clear optimum (~0.001). This delineates the boundary of global input-side noise: **it is the strongest "global" form of invariance pressure, but closing the OOD gap requires per-task tuning cost.**

**(3) The four tasks form a clear sensitivity gradient.** PushT (−83.66 base drop) > Reacher (−43.00) ≈ TwoRoom (−48.67) > Cube (−20.00). However the recovery effect of noise training does not scale with sensitivity — TwoRoom recovers most fully (+54.34 pt), Cube recovers least (+15.67 pt). This indicates that input-side global noise is most effective on "visually redundant" tasks and offers limited returns on "structured manipulation".

### 4.4 Diagnostic analysis: why global noise is not a silver bullet

Table 3 compares core diagnostic metrics on LeWM-base versus LeWM+noise (per-task best).

**Table 3. Representation diagnostics: LeWM-base vs. per-task noise-best configurations.**

| Metric | TwoRoom base | TwoRoom best (0.008) | PushT base | PushT best (0.002) | Reacher base | Reacher best (0.006) | Cube base | Cube best (0.001) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 | 0.0633 | 0.0676 | 0.1856 | 0.1879 |
| `clean_effective_rank`     | 47.60  | 33.59  | 76.42  | 42.85  | 61.04  | 65.92  | 73.25  | 71.83  |
| `transition_resolution_ratio_l2`  | 0.7216 | 0.6055 | 0.3015 | 0.2800 | 0.3704 | 0.3791 | 0.4847 | 0.4629 |
| `transition_resolution_ratio_cos` | 0.5538 | 0.3780 | 0.0868 | 0.0800 | 0.1351 | 0.1399 | 0.2347 | 0.2168 |
| `id_probe_r2`              | 0.2889 | −0.0573 | 0.7739 | 0.7500 | 0.1621 | 0.1729 | 0.6657 | 0.6720 |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.1200 | 0.2518 | 0.2585 | 0.2364 | 0.2320 |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 16.50 | 15.17 | 0.44 | 20.20 | 19.25 |

**Notes.** (i) `transition_resolution_ratio_l2` and `_cos` values for TwoRoom are taken directly from `geometry_summary.json` and `task_resolution.json` and corrected against an earlier transcription. (ii) Reacher/Cube best diagnostics are pulled from the corresponding ckpt's `eval_results/diagnostics/{geometry_summary, task_resolution, predictor_sensitivity}.json` (max-std = 0.1, history-only noise). (iii) Cube base `predictor_rollout_T8_l2 = 20.20` and Reacher base `15.17` are of similar magnitude; Cube best = 19.25 and Reacher best = 0.44 show that noise training's effect on long-horizon rollout drift is highly **task-dependent** (Reacher: 35× reduction; Cube: nearly unchanged).

![Fig 4 — Per-task diagnostic radar: base vs noise-best on 6 metrics](assets/paper1_figs/fig4_radar.png)

**Mechanistic reading.**

- **TwoRoom.** Low-dimensional, discrete, visually redundant. Compressing the representation (effective rank 47.6 → 33.6) is acceptable and even beneficial. Smaller NN distances mean a more compact latent space that planning navigates more easily.
- **PushT.** Continuous contact requires fine-grained pose resolution. Even at the optimal light noise (std = 0.002), `transition_resolution_ratio_l2` already trends slightly downward. At heavier noise (e.g. std = 0.006) this metric would drop further, erasing the contact-transition keyframes.
- **A predictor-rollout caveat.** A drop in `predictor_rollout_T8_l2` is not unambiguously good news. It can also mean the latent has become more *predictable* without being more *controllable* — predictor stability can be bought by sacrificing resolution.

### 4.5 Cross-checkpoint correlation analysis

Table 4 reports task-specific cross-ckpt correlations for the strongest candidate metrics on the canonical n = 8 set.

**Table 4. Diagnostic metric ↔ eval correlation (canonical n = 8).**

| Metric | TwoRoom (r / ρ) | PushT (r / ρ) | Reacher (r / ρ) | Cube (r / ρ) |
|---|---:|---:|---:|---:|
| `predictor_target_to_nn_cos_ratio_at_max_std` | −0.96 / −0.43 | **−0.80 / −0.93** | −0.56 / −0.58 | +0.17 / +0.04 |
| `latent_cost_surface_slope_z`                 | +0.47 / +0.61 | **+0.74 / +0.93** | −0.20 / −0.14 | −0.28 / −0.37 |
| `predictor_rollout_T8_l2`                     | +0.22 / +0.23 | +0.68 / +0.79 | **−0.71 / −0.83** | +0.41 / +0.76 |
| `cka_linear_at_max_std`                       | +0.58 / +0.29 | −0.08 / −0.02 | +0.92 / +0.68 | **−0.85 / −0.96** |

**n = 9 LeWM PushT sweep, with partial correlation conditioned on `std_max`.**

To strengthen the analysis we replace eval-only correlation with partial correlation conditioned on the training noise level. On the n = 9 LeWM PushT sweep (canonical ckpts including the 3-seed retrained 0to006), the strongest single per-token diagnostic, `predictor_target_to_nn_cos_ratio_at_max_std`, gives the following picture:

| Quantity | Spearman ρ |
|---|---:|
| ρ(std_max, metric) | +0.83 |
| ρ(std_max, clean success rate) | −0.43 |
| ρ(std_max, px+goal 0.08 success rate) | +0.82 |
| ρ(std_max, OOD drop) | −0.93 |
| ρ(metric, clean) — unconditional | **−0.73** |
| ρ(metric, clean) ∣ std_max — partial | **−0.73** |
| ρ(metric, px+goal 0.08) — unconditional | +0.55 |
| ρ(metric, px+goal 0.08) ∣ std_max — partial | **−0.67** |
| ρ(metric, OOD drop) — unconditional | −0.77 |
| ρ(metric, OOD drop) ∣ std_max — partial | +0.13 |

This is the table that decides the honest interpretation of the diagnostic toolkit. The key reads:

1. **Within a fixed training protocol the metric strongly predicts both clean and OOD performance.** Partial Spearman ρ on clean is **−0.73** and on px+goal 0.08 is **−0.67** — both meaningful negative correlations. That is, among models trained at the *same* `std_max`, the one with the lower fragility ratio is the one that achieves better clean *and* better noisy success.
2. **The metric does NOT predict OOD drop beyond training protocol.** The unconditional ρ(metric, drop) = −0.77 looks impressive, but partialling out `std_max` flips the sign and collapses the magnitude to +0.13. The drop's strong correlation with the metric is a mediated effect: `std_max` drives both the metric (ρ = +0.83) and the drop (ρ = −0.93). Once `std_max` is fixed, the metric tells you nothing new about how much the *gap* between clean and noisy performance will be.
3. **Practical reading.** The toolkit is a model-selection tool that ranks checkpoints within a fixed training protocol. It is *not* a substitute for actually training with noise when the OOD gap is the quantity of interest.

#### 4.5.5 What this diagnostic actually predicts: clean vs OOD

Older versions of this analysis reported only the unconditional ρ between the fragility metric and "eval". Earlier internal write-ups described that quantity as predicting OOD failure; the partial-correlation analysis above shows the more nuanced reality:

- The metric is a **strong ckpt-quality signal** — within any fixed training protocol, lower fragility ratio means better control on *both* clean and noisy evaluation (partial ρ ≈ −0.7 on both).
- The metric **does not isolate noise-robustness** — its apparent correlation with the *gap* between clean and noisy success is fully mediated by `std_max`.

The PushT scatter (Figure 3) makes both halves of this visible. Panel (a) plots metric × clean and shows the strong negative trend (Spearman ρ = −0.73). Panel (b) plots metric × OOD drop and shows the same trend at unconditional ρ = −0.77, but the colour-bar (which encodes `std_max`) reveals the structure: low-`std_max` ckpts (light blue) live at high drop, high-`std_max` ckpts (dark blue) live at low drop, and the metric tracks `std_max` itself. Once `std_max` is held constant, the metric does not separate small-drop from large-drop ckpts.

![Fig 3 — PushT n = 9 LeWM sweep: fragility metric is a ckpt-quality predictor (a), the apparent OOD-drop correlation in (b) is mediated by std_max](assets/paper1_figs/fig3_scatter.png)

### 4.6 Mechanism attribution: where in the pipeline does noise cause failure?

§4.4 reports *what* is compressed and §4.5 reports *which diagnostics* predict eval drop across checkpoints, but neither answers **"is the failure in the encoder, the predictor, or the cost surface?"** We address this with two complementary experiments.

#### 4.6.1 Eval-only cost swap: cost surface is not the main culprit

If the failure were primarily caused by the planning-time cost function (for instance, cosine cost saturating under noise), changing the cost should produce a substantial recovery. We hold a TwoRoom checkpoint fixed and only swap the CEM cost type at eval time (variant A is the canonical inference configuration; variant B is an alternative-cost ablation):

| Variant | cost type | cost space | std = 0.03, pix+goal success |
|---|---|---|---:|
| A (default) | cosine | normalized | 36.0 |
| B (swap) | mse | raw | 42.0 |
| *Reference: same checkpoint, clean eval (num_eval = 300)* | — | — | 69.7 |

Swapping cost recovers only +6 pt (36 → 42), far below the clean baseline (69.7). **Conclusion: cost surface is not the main cause.** Upstream noisy-goal embedding corruption sets the ceiling.

#### 4.6.2 Latent-noise probing: encoder is the principal bottleneck

Directly injecting noise into the latent `z` (skipping the encoder) decouples encoder contributions from predictor + cost. We compare diagnostic signals across two injection points:

| Metric | Injection point | What it measures |
|---|---|---|
| `predictor_rollout_T8_l2_history` | pixels (history-only) | encoder + predictor multi-step drift |
| `latent_predictor_rollout_T8_l2_history` | latent `z` (history-only) | predictor amplification of latent perturbations |
| `cost_surface_slope_z` | latent `z` (goal-only) | local smoothness of cost vs. goal latent |

**Key findings** (canonical n = 8 correlation analysis):

- **TwoRoom.** `latent_predictor_rollout_T8_l2_history` ↔ eval ρ = +0.738, slightly stronger than the input-space `predictor_rollout_T8_l2` (ρ = +0.667) — encoder remains dominant but predictor adds an independent contribution.
- **PushT.** Input-space and latent-only signals are nearly collinear (+0.627 / +0.636); the single-step `predictor_target_to_nn_cos_ratio_at_max_std` (ρ = −0.791) is the strongest. **Encoder + single-step predictor jointly dominate; cost surface (`latent_cost_surface_slope_z`, ρ = +0.93) is collinear with the latent rollout.**
- **Reacher / Cube.** `latent_cost_surface_slope_z` |ρ| < 0.4 — cost surface is not an explanatory variable.

**Three-layer attribution summary.**

| Task | Primary cause | Secondary |
|---|---|---|
| TwoRoom | encoder dominant | independent predictor contribution |
| PushT   | encoder + single-step predictor | cost surface (collinear with latent rollout) |
| Reacher | encoder + multi-step rollout | — |
| Cube    | encoder | — |

The common primary cause across the four tasks is **encoder shift transduced by the predictor**, and **cost surface is not a principal explanatory variable on any task**. This is also the root reason that Layer-5 task-resolution metrics (`transition_resolution_ratio`, `id_probe_r²`) carry strong signal in §4.4: once the encoder's latent neighbourhood structure is corrupted beyond the NN distance scale, downstream predictor and planner are already operating on the wrong neighbourhood.

![Fig 5 — Mechanism attribution: encoder shift transduced by predictor dominates; cost surface is not the bottleneck](assets/paper1_figs/fig5_mechanism.png)

---

## 5 Discussion

### 5.1 Rethinking the "JEPA invariance" narrative

The data here challenge an implicit assumption in the JEPA community: latent prediction alone is not sufficient to confer invariance to visual noise. The invariance a JEPA encoder acquires is **in-distribution invariance** — invariance to visual patterns present in the training distribution. When test-time inputs carry high-frequency pixel noise not seen during training, the topology of the latent space breaks down, the predictor outputs incorrect future states, and the planner fails.

This stands in contrast to VJEPA [9], which reports R² > 0.84 under a Noisy-TV distractor — but that result is on 1D synthetic signals evaluated by linear probe. **Control-task evaluation (success rate) is a strictly stricter standard than linear-probe R²**: linear probes only require enough latent information for a linear classifier to extract, whereas control tasks demand a representation that supports precise latent-space planning and action optimisation.

### 5.2 The invariance–resolution trade-off — manifestation and scope

In this paper the **invariance–resolution trade-off** has the following four-task signature:

- **TwoRoom.** Background wall colour / texture is pure visual redundancy; discarding it does not impair navigation.
- **PushT.** The pixel changes during T-block / arm contact carry critical force and pose information; discarding them blinds the planner.
- **Reacher.** Low-dimensional continuous control; visual encoding of joint angle requires moderate resolution.
- **Cube.** Object pose and grasp-point spatial relations need moderate resolution, but Cube itself is largely insensitive to global noise (action sequence is structured; visual–action coupling is predictable).

Task-specificity explains why there is no single optimal noise level.

**Scope.** Whether the trade-off generalises to other latent world-model architectures is an **open question we do not address**:

- **Reconstruction-based world models** (DreamerV3 / TD-MPC2). The reconstruction loss explicitly forces preservation of pixel information; the trade-off may manifest differently. ViGMO [13] observes related task-specific noise sensitivity on DMC; the qualitative direction agrees but the quantitative regime differs.
- **EMA-target JEPA** (I-JEPA / V-JEPA lineage). Different encoder update dynamics may modulate SIGReg's anti-collapse behaviour under noise.
- **Variational / information-bottleneck JEPA** (VJEPA [9]). An explicit KL term provides a second invariance pressure; whether it is complementary or orthogonal to input-side noise training is unknown.

This paper's scope is LeWM + CEM. Universalising the trade-off to "all latent compression world models" exceeds the present evidence.

### 5.3 When the diagnostic toolkit works — and when it does not

Three boundaries should be reported up-front so practitioners know when to use the toolkit and when to fall back to direct evaluation.

**Boundary 1: the toolkit ranks ckpts by clean-control quality, not by OOD-specific robustness.** §4.5.5 shows the strongest cross-ckpt diagnostic (`predictor_target_to_nn_cos_ratio_at_max_std`) correlates strongly with clean success (ρ ≈ −0.8) but only weakly with OOD drop (ρ ≈ −0.3). Use the toolkit when you want to pick the *best-trained* checkpoint from a sweep; do *not* use it as a substitute for actually running an OOD eval if your downstream question is robustness.

**Boundary 2: tasks with weak per-state controllability variance fall outside the toolkit's reliable regime.** Reacher (low-dimensional continuous reaching) and TwoRoom (visually redundant discrete navigation) do not produce diagnostic-vs-eval Spearman ρ that survives our partial-correlation criterion. Both tasks lack the within-method spread to distinguish "good" from "bad" ckpts via a label-free metric. The toolkit can describe what the model has compressed on these tasks (Table 3) but cannot predict relative checkpoint quality.

**Boundary 3: cross-ckpt diagnostics cannot recover what training did not provide.** The largest determinant of OOD drop is whether the model saw noise during training — not which fragility metric it happens to score on. This is the structural reason ρ(metric, drop) is weak even when ρ(metric, clean) is strong: noise vs no-noise training puts each ckpt on a completely different (clean, OOD) curve (cf. Fig 6), and *no static cross-ckpt diagnostic* can substitute for that training-time choice.

The toolkit therefore has a precise scope: it is a **clean-evaluation auxiliary** that lets you select among ckpts on tasks with strong per-state controllability variance (PushT, Cube), assuming the training protocol is already fixed. It is not an OOD prediction oracle.

### 5.4 Practical guidance: how to choose `std_max` on a new task

Our sweep data suggest a simple operational recipe:

1. **First, inspect the clean baseline's `predictor_target_to_nn_cos_ratio_at_max_std`.** If below 1e-5 (PushT / Cube regime), the task is pixel-noise sensitive; start sweeping at `std_max ∈ [0.001, 0.003]` with clean performance as the primary constraint.
2. **Then check `clean_effective_rank` and `transition_resolution_ratio`.** High rank and high ratio (PushT: 76 / 0.30) → resource-rich task; cap the sweep at 0.005 to avoid destroying resolution. Low rank and low ratio (TwoRoom: 47 / 0.72) → visually redundant; sweep safely up to 0.008+.
3. **`noise_prob` and `std_min`.** We fix `noise_prob = 1.0` and `std_min = 0`. Softening the training distribution via `noise_prob ∈ [0.5, 1.0]` is future work.
4. **Use two endpoints in eval.** Clean and max-noise; checking only one misses one of the two optima (PushT's clean optimum at 0.002 vs. robustness optimum at 0.006 is the clearest example).
5. **Under compute budget,** a 4-level sweep (`{0.001, 0.003, 0.005, 0.007}`) already locates the optimum within ±0.001.

### 5.5 Limitations and future directions

**Limitation 1 — Single backbone family.** We validate on LeWM. Other JEPA variants (EMA-target I-JEPA / V-JEPA lineage; variational JEPA) may exhibit different noise responses.

**Limitation 2 — Gaussian pixel noise only.** Real-world visual corruption includes motion blur, contrast variation, occlusion, and lighting change; whether the trade-off transfers to these regimes is open.

**Limitation 3 — Diagnostic framework is empirical, not theoretical.** Our metrics are selected by cross-ckpt correlation. Establishing a formal causal chain "effective rank ↓ → resolution ratio ↓ → control failure" is future work. Reacher and TwoRoom fail the partial-correlation criterion for *all* metrics — directly exposing where the empirical framework breaks down.

**Limitation 4 — Mixed statistical protocol.** Some rows use single-seed × 300 trajectories and others 3-seed × 100 trajectories (total sample size 300 in both cases, but across-seed variance estimation differs). A unified 5-seed × 100 protocol upgrade is planned for the next version.

**Limitation 5 — Automated transition reweighting is not a substitute for noise training.** As a sanity check we also tested a scale-preserving heteroscedastic-NLL formulation in which a learned per-transition σ-head downweights high-error transitions. The result is informative as a negative finding: TwoRoom clean reaches 99.67% but high-noise robustness lags noise training; PushT clean *collapses* from 87.33 to 13.33% because contact-control transitions have high prediction error and are exactly the transitions the σ-weighting would discard. Full data are reported in Appendix D. The lesson — "hard ≠ unimportant" in contact control — connects the trade-off documented here to the broader question of per-token controllers, which we leave to follow-up work.

**Future direction 1 — Per-token adaptive consistency.** Our strongest diagnostic, `predictor_target_to_nn_cos_ratio_at_max_std`, is a ckpt-level scalar. Whether its per-token variant can serve as a per-token consistency controller is a separate methodological question (an investigation we have ongoing).

**Future direction 2 — Cross-architecture replication.** Running the sweep and diagnostic protocol on DreamerV3 / TD-MPC2 would reveal whether the trade-off is JEPA-specific or shared by all latent-compression world models.

**Future direction 3 — Theoretical grounding.** Reformulating the trade-off in an information-bottleneck / rate-distortion language is an attractive direction; we did not pursue it because the empirical phenomenology may not yet be stable enough for formal modelling.

---

## 6 Conclusion

This paper provides a systematic diagnostic study of the visual-OOD control robustness of JEPA + CEM world models, using LeWorldModel as the representative system. Three findings:

1. **The "JEPA invariance" assumption does not hold in control.** Without noise-aware training LeWM collapses under pixel noise; latent prediction alone does not confer visual robustness.

2. **Global noise augmentation has a boundary.** It effectively closes the OOD gap but has no globally optimal level; task differences are large, and within a single task clean and robustness optima can dissociate.

3. **A five-layer diagnostic protocol reveals the underlying mechanism.** Noise-augmentation gains arise from representation compression, but excessive compression destroys the resolution required for control — this is the *invariance–resolution trade-off*.

The paper does not propose a new training algorithm. Its contribution is a systematic body of empirical evidence and a reusable diagnostic toolkit. We believe that, before proposing more elegant mathematical controllers, understanding the behavioural boundary of existing systems — as done here — is the responsible scientific stance.

---

## References

[1] Y. LeCun, "A path towards autonomous machine intelligence," *Open Review*, 2022.

[2] M. Assran et al., "Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA)," *CVPR*, 2023.

[3] A. Bardes et al., "Revisiting feature prediction for learning visual representations from video (V-JEPA)," *Trans. Machine Learning Research / arXiv:2404.08471*, 2024.

[4] M. Assran et al., "V-JEPA 2: Self-supervised video models enable understanding, prediction, and planning," *arXiv:2506.09985*, 2025.

[5] L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, R. Balestriero, "LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels," *arXiv:2603.19312*, 2026. *(LeWM; SIGReg defined therein.)*

[6] T. Chen et al., "A simple framework for contrastive learning of visual representations (SimCLR)," *ICML*, 2020.

[7] E. D. Cubuk et al., "RandAugment: Practical automated data augmentation with a reduced search space," *NeurIPS*, 2020.

[8] Anonymous, "Improving joint embedding predictive architecture with diffusion noise (N-JEPA)," *arXiv:2507.15216*, 2025.

[9] Y. Huang, "VJEPA: Variational joint embedding predictive architectures as probabilistic world models," *arXiv:2602.19322*, 2026.

[10] Anonymous, "US-JEPA: A joint embedding predictive architecture for medical ultrasound," *arXiv preprint*, 2025–2026. *(Anonymous; cite by title.)*

[11] D. Hafner et al., "Mastering diverse domains through world models (DreamerV3)," *Nature*, 2024.

[12] N. Hansen et al., "TD-MPC2: Scalable, robust world models for continuous control," *ICLR*, 2024.

[13] Anonymous, "Zero-shot visual generalization in model-based reinforcement learning (ViGMO)," *OpenReview submission*, 2024–2025.

[14] A. Tamkin et al., "Feature dropout: Revisiting the role of augmentations in contrastive learning," *NeurIPS*, 2022.

[15] J. Zhang et al., "Rethinking the augmentation module in contrastive learning," *ECCV*, 2022.

[16] O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," *EUSIPCO*, 2007.

[17] L. Jing, P. Vincent, Y. LeCun, Y. Tian, "Understanding dimensional collapse in contrastive self-supervised learning," *ICLR*, 2022.

[18] Z. Teoh et al., "Next-latent prediction transformers learn compact world models," *NeurIPS*, 2025.

[19] T. W. Epps, K. J. Pulley, "A test for normality based on the empirical characteristic function," *Biometrika*, 1983. *(Statistical foundation of SIGReg in [5].)*

[20] A. Bardes, J. Ponce, Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," *ICLR*, 2022. *(Anti-collapse baseline.)*

---

## Appendix A — Experimental details

### A.1 Environments

- **PushT.** 2D continuous pushing; 20,000 expert episodes; ~196 steps/episode; action dim 2 (direction + magnitude); 224×224 RGB.
- **TwoRoom.** 2D continuous navigation; 10,000 episodes; ~92 steps; action dim 2; 224×224 RGB.
- **Reacher.** 2D arm control (DeepMind Control Suite); 10,000 episodes; 200 steps; action dim 2.
- **Cube.** 3D cube manipulation (OGBench); 10,000 episodes; 200 steps; action dim 7.

### A.2 Noise augmentation implementation

The actual implementation is `utils.py::AddNormalizedGaussianNoise`. Key points: (i) noise is added to ImageNet-normalized tensors and must be divided by channel std to land in "pixel-space std" units; (ii) sampling is **per-frame independent** over the leading frame dims, not per-batch; (iii) in all sweeps we fix `noise_prob = 1.0` and `std_min = 0` and only vary `std_max`.

```python
class AddNormalizedGaussianNoise:
    """
    Per-frame independent: each frame draws Bernoulli(noise_prob) then
    std ~ Uniform(std_min, std_max). Pixel-space std is converted to
    normalized space by dividing by channel std before adding.
    """
    def __init__(self, std_min: float, std_max: float, noise_prob: float = 1.0):
        self.std_low, self.std_high, self.noise_prob = std_min, std_max, noise_prob
        self.channel_std = torch.as_tensor(IMAGENET_STD)  # (C,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W), normalized; leading dims = frame dims.
        if self.std_high <= 0 or self.noise_prob <= 0:
            return x
        leading = x.shape[:-3]
        stds = torch.empty(leading, device=x.device, dtype=x.dtype).uniform_(
            self.std_low, self.std_high
        )
        if self.noise_prob < 1.0:
            mask = (torch.rand(leading, device=x.device) < self.noise_prob).to(x.dtype)
            stds = stds * mask
        per_frame_scale = stds.view(*leading, 1, 1, 1)
        channel_factor = (1.0 / self.channel_std.to(x.device, x.dtype)).view(
            *([1] * len(leading)), -1, 1, 1
        )
        scale = per_frame_scale * channel_factor   # pixel-space std → normalized
        return x + torch.randn_like(x) * scale
```

### A.3 Evaluation protocol

- **Clean eval.** No noise; 100 trajectories × 3 seeds.
- **Noisy eval.** Gaussian noise added to both pixels and goal images at std ∈ {0.05, 0.08}.
- **Success criterion.** Task-specific (PushT: T-block pose match; TwoRoom: target region; Reacher: joint-angle match; Cube: cube position match).

### A.4 Diagnostic-metric computation

Implementations are in `tools/repr_analysis/`.

### A.5 Compute

All experiments run on a single NVIDIA A100 (80 GB). Training: ~2 hours/task for LeWM-base, ~2.5 hours/task/configuration for LeWM+noise. Diagnostic analysis: ~30 minutes/task/checkpoint.

### A.6 Main-figure rendering recipe

The 5 main figures are produced by `tools/paper1_figs.py`. Run:

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
```

| Fig | Layout | Data source | Notes |
|---|---|---|---|
| **1 (hero)** | Grouped bars per task: clean / px+g 0.08 base / px+g 0.08 best | Tables 1 + 2 | Annotates per-task σ* and Δ |
| **2 (sweep)** | 4 panels (tasks) × 2 curves (clean / px+g 0.08) | Table 2 | Dashed vertical at per-task σ* |
| **3 (scatter)** | Two-panel scatter of `predictor_target_to_nn_cos_ratio_at_max_std` vs PushT clean (a) and OOD drop (b) | n = 9 LeWM PushT checkpoint dirs under `lewm-pusht/ckpt/`; `predictor_sensitivity.json` (max-std, history-only) + `summary.txt`; canonical `_20260507` paths preferred for the 0to006 retrain | colour by std_max; Spearman annotated |
| **4 (radar)** | 2×2 grid; 6-axis polar per task; base vs noise-best overlay | Table 3 | Per-metric min-max normalization across tasks |
| **5 (mechanism)** | Left: schematic 3-layer flow; right: per-task |ρ| per layer | §4.6 numbers | Encoder shift dominates everywhere; cost surface low on Reacher / Cube |

Auxiliary figures from `assets/diagnostics/` (`p0_correlation_*.png`, `predictor_drift_eval_correlation.png`, `geometry_tradeoff_goal.png`, etc.) are suitable as supplementary material.

---

## Appendix B — Full diagnostic profile of LeWM-base on four tasks

The table below summarises every core diagnostic on the four LeWM-base checkpoints. Data are taken from each checkpoint's `eval_results/diagnostics/{geometry_summary, noise_sensitivity, task_resolution, predictor_sensitivity, latent_noise_sensitivity, action_effect}.json`.

| Layer | Metric | TwoRoom | PushT | Reacher | Cube | Unit / note |
|---|---:|---:|---:|---:|---:|---|
| **Encoder Geometry** | `clean_nn_cos_dist_median` | 0.0449 | 0.2360 | 0.0633 | 0.1856 | cosine distance |
| | `clean_pair_cos_dist_median` | 0.9904 | 1.0228 | 1.0252 | 1.0193 | pair-wise cos distance |
| | `clean_effective_rank` | 47.60 | 76.42 | 61.04 | 73.25 | effective rank |
| **Noise Sensitivity** | `noise_angle_deg_median` (@ std = 0.005) | 5.51° | 1.33° | 3.22° | 1.40° | median angular shift |
| | `noise_to_nn_cos_ratio_median` | 0.1031 | 0.0011 | 0.0249 | 0.0016 | noise/NN cos ratio |
| | `robust_radius_std` | 0.0142 | 0.0537 | 0.0142 | 0.0356 | critical noise std |
| | `first_risk_std` | >0.08 | >0.08 | >0.08 | >0.08 | first high-risk std |
| | `noise_angle_slope_deg_per_std` | 1085.8 | 284.8 | 831.7 | 327.0 | °/std angular gain |
| | `geometry_flag` | balanced | robust | balanced | robust | geometry label |
| **Task Resolution** | `transition_resolution_ratio_l2`  | 0.7216 | 0.3015 | 0.3704 | 0.4847 | L2 resolution ratio |
| | `transition_resolution_ratio_cos` | 0.5538 | 0.0868 | 0.1351 | 0.2347 | cos resolution ratio |
| | `id_probe_r2` | 0.2889 | 0.7739 | 0.1621 | 0.6657 | linear probe R² (action) |
| | `id_probe_r2_min` | 0.2599 | 0.6786 | 0.1366 | 0.0972 | min probe R² |
| | `lidar_rank` | 46.06 | 13.95 | 45.90 | 42.46 | LiDAR rank proxy |
| **Action Effect** | `action_mean_pred_shift_norm` | 0.5329 | 0.1283 | 0.2518 | 0.2364 | action-perturb mean shift |
| | `action_perturb_pred_shift_corr` | 0.2847 | 0.2873 | 0.4042 | 0.2559 | shift × action norm corr |
| **Predictor Stability** | `predictor_rollout_T8_l2` | 18.62 | 18.65 | 15.17 | 20.20 | T=8 rollout L2 drift (history-only @ max std) |
| | `predictor_target_to_nn_cos_ratio_at_max_std` | 1.51e-4 | 3.54e-6 | 2.67e-5 | 3.39e-6 | max std target/NN ratio |
| **Latent Noise** | `cka_linear_at_max_std` | 0.1986 | 0.5536 | 0.3085 | 0.1814 | CKA clean vs noisy |
| | `latent_cost_surface_slope_z` | 635.31 | 1.3886 | 599.45 | 0.6208 | goal-latent perturb cost slope |

The full per-noise-level diagnostic values for the 9-level LeWM PushT sweep live in `canonical_evals_20260508.json` / `canonical_correlations_20260508.json` (generated locally; not checked into git).

---

## Appendix C — Heteroscedastic-loss formulation

The scale-preserving heteroscedastic NLL referenced in §5.5 (Limitation 5) and detailed in Appendix D is:

$$
\mathcal{L}_{\text{hetero}} = \tfrac{1}{2}\,\exp(-s_t)\,\|z_{t+1} - \hat z_{t+1}\|^2 + \tfrac{1}{2}\,s_t
$$

where $s_t$ is the σ-head-predicted log-variance. During training $\exp(-s_t)$ acts as an automatic weight: high-error transitions are down-weighted and low-error ones are up-weighted. When $s_t \equiv 0$ the loss reduces to plain MSE.

---

## Appendix D — Heteroscedastic-loss negative result (data)

This appendix records the full data behind §5.5 Limitation 5. The σ-head is trained jointly with the predictor; gradient is detached on the σ path so the mean prediction path is exactly LeWM MSE when σ is constant.

**Table D.1. Heteroscedastic-loss eval.**

| Task / model | Clean | goal 0.05 | pixels 0.05 | px+goal 0.05 | goal 0.08 | px+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base       | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom LeWM+noise best | 98.33 | 98.00 | 98.33 | 98.00 | 98.67 | 98.67 |
| TwoRoom hetero          | **99.67** | 85.33 | 96.67 | 84.67 | 73.33 | 55.33 |
| PushT LeWM-base         | 87.33 | 38.00 | 17.33 | 15.00 | 15.00 |  3.67 |
| PushT LeWM+noise best   | **90.00** | 85.00 | 87.67 | 86.00 | 83.00 | 70.67 |
| **PushT hetero**        | **13.33** |  7.67 |  7.67 |  7.67 |  9.67 |  6.00 |

**Reading.** TwoRoom hetero reaches 99.67% clean (consistent with the prior that low-dimensional discrete tasks benefit from stronger invariance / clustering) but lags noise training on high-noise robustness. **PushT hetero clean is 13.33% — a method-level failure**, not a robustness trade-off.

**Table D.2. Representation diagnostics of the failure.**

| Metric | TwoRoom base | TwoRoom hetero | PushT base | PushT hetero |
|---|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 |
| `clean_effective_rank`     | 47.60  | 33.59  | 76.42  | 42.85  |
| `transition_resolution_ratio_l2` | 0.5538 | 0.3780 | 0.3015 | **0.1023** |
| `id_probe_r2`              | 0.2889 | −0.0573 | 0.7739 | **0.2678** |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.0841 |
| `predictor_rollout_T8_l2`     | 18.62  | 17.90  | 18.65  | 14.01  |

Hetero-loss compresses the representation in both tasks. For TwoRoom (low-dimensional, discrete, redundant), compression is acceptable. For PushT, `transition_resolution_ratio_l2` collapses from 0.3015 to 0.1023 and `id_probe_r²` from 0.7739 to 0.2678 — task-relevant state information is erased. The drop in `predictor_rollout_T8_l2` is not good news either: the latent has become more *predictable* without being more *controllable*.

**The mechanism.** The σ-head correctly learns per-transition difficulty (high σ on contact frames in PushT, on doorway crossings in TwoRoom, etc.), but using σ as an automatic weight to *down*weight high-error transitions misclassifies the contact-and-control-critical states of PushT as "unimportant" and erases them. This is a direct demonstration of the broader trade-off lesson: in contact-heavy control, **hard ≠ unimportant**.

This finding motivates a follow-up direction we are currently exploring: per-token *consistency* (not loss-reweighting) controllers driven by detached difficulty signals, which preserve the mean prediction path's gradient distribution. That work is independent of this paper's diagnostic study and will be reported separately.

---

*Code and complete data: https://github.com/qun-team/wm_exp.*
