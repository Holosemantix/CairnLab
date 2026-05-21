# Paper 1 故事线大纲

## 一句话版本

LeWM 这类 JEPA + CEM 世界模型在 clean 视觉输入上能规划，但未经噪声训练时会在 control-time pixel/goal noise 下严重崩溃；简单的全局 input-side noise augmentation 能大幅恢复 OOD 成功率，但没有全局最优剂量，因为它同时带来有益的 invariance 和有害的 task-resolution compression。本文用 4 个任务、36 个 LeWM checkpoint、统一 3 evaluation seeds × 100 trajectories 协议，以及五层表征诊断，把这个现象定义为 invariance-resolution trade-off。

## 读者应带走的核心判断

- 这不是一个新算法 paper，而是一个诊断型 empirical paper。
- 主实验结论严格限定在 LeWM 上：4 tasks × 9 configs，即 base + `std_max=0.001..0.008`。
- PLDM 只作为 PushT clean-trained external sanity check：它支持 visual OOD cliff 不只发生在 LeWM，但不支撑跨架构 trade-off 结论。
- 自适应分辨率 / per-token controller 是后续方法线，不放入 Paper 1 的主线。
- 最强 label-free 诊断指标不是 OOD oracle；它更多是控制 `std_max` sweep trend 后的 residual checkpoint-quality signal。

## 论文问题设定

### 背景直觉

JEPA 社区常有一个隐含直觉：既然模型在 latent space 预测而不是重建像素，它可能自然忽略 irrelevant pixel details，因此对视觉扰动更鲁棒。

### 本文要挑战的点

这个直觉在 control setting 下不成立。控制不是分类或线性 probing；CEM planner 依赖 encoder 和 predictor 保持局部状态拓扑。一旦 test-time pixel noise 改变 latent neighborhood structure，planner 会在错误的未来状态上优化。

### 我们的具体问题

1. LeWM 在 clean vs visual OOD 下到底掉多少？
2. input-side noise training 能否恢复？
3. 如果能恢复，为什么不是越多越好？
4. 哪些表征诊断能解释 task-specific trade-off？
5. 哪些诊断只能选 checkpoint，不能预测 OOD robustness？

## 核心贡献结构

### C1：统一协议量化 JEPA + CEM visual OOD fragility

实验覆盖 4 tasks：TwoRoom、PushT、Reacher、Cube。

LeWM sweep 覆盖 36 checkpoint：每个任务 9 个 config，即 base + 8 个 noise levels。

所有 success rate 使用统一协议：3 evaluation seeds（42/43/44）× 每 seed 100 trajectories，每个 condition 共 300 trajectories。

### C2：提出 invariance-resolution trade-off 和五层诊断

五层诊断不是新指标本身，而是把指标组织成机制链：

- Layer 1：encoder shift，看 pixel noise 如何移动 latent。
- Layer 2：encoder geometry，看 rank / nearest-neighbor scale / collapse。
- Layer 3：predictor sensitivity，看 encoder shift 是否被 predictor 放大。
- Layer 4：latent-noise response，看 latent perturbation 下 cost / rollout 的响应。
- Layer 5：task resolution，看 latent 是否仍保留控制所需的 transition separability 和 action/state probe signal。

### C3：解释为什么 global noise augmentation 有边界

TwoRoom 视觉冗余、低维、离散，表征压缩通常有益；PushT contact-heavy，需要高分辨率和高 controllability，过度压缩会伤 clean control。

因此同一个 `std_max` 不可能全局最优。

### C4：诚实限定 cross-checkpoint diagnostics 的使用范围

`predictor_target_to_nn_cos_ratio_at_max_std` 在 PushT 上对 clean 和 px+goal 0.08 success 有 residual checkpoint-quality signal。

但它不能隔离 OOD-specific robustness：表面上与 OOD drop 强相关，主要是因为 `std_max` 同时驱动 metric 和 drop；partial out `std_max` 后，metric 与 clean-to-OOD drop 的关系基本消失。

## 主结果故事线

### Step 1：先证明问题真实存在

对应位置：§4.2、Table 1、Figure 1。

LeWM-base 在 clean 上表现不差，但 visual OOD 下大幅掉点：

- PushT：86.33 → 4.67，drop 81.67pt。
- TwoRoom：94.00 → 50.00，drop 44.00pt。
- Reacher：58.67 → 15.00，drop 43.67pt。
- Cube：66.67 → 46.33，drop 20.33pt。

这里的重点不是 “某个任务 bad”，而是 clean planning success 不能说明 visual robustness。

### Step 2：证明简单 noise training 可以显著恢复

对应位置：§4.3、Table 2、Figure 1、Figure 2、Figure 6。

Noise training 基本能关闭 high-noise OOD gap，但 optimum strongly task-dependent：

- TwoRoom：heavy noise 有利，clean / OOD point-best 都在 0.008。
- PushT：clean point-best 是 0.003，px+goal 0.08 point-best 是 0.006。
- Reacher：clean point-best 是 0.006，px+goal 0.08 point-best 是 0.002，整体是 plateau。
- Cube：px+goal 0.08 point-best 是 0.007，clean 是 shallow plateau。

这一步引出核心 tension：noise 是有用的，但不是一个可无脑调大的 hyperparameter。

### Step 3：用诊断解释 task-specific response

对应位置：§4.4、Table 3、Figure 4。

Table 3 不要读成 “best checkpoint 对比”。它对比的是 base vs representative diagnostic checkpoint。

关键语义：

- clean point-best：clean 成功率最高的 config。
- px+goal 0.08 point-best：高噪声 OOD 成功率最高的 config。
- representative diagnostic checkpoint：完整诊断 suite 实际跑过的代表性 noise-trained checkpoint。

诊断解释：

- TwoRoom：压缩 rank 后仍保持 transition resolution，甚至更适合低维导航。
- PushT：base 本来需要高 rank 和高 action/state controllability；noise-induced compression 只能很谨慎。
- Cube / Reacher：介于中间，且部分指标与 eval 的关系较弱。

### Step 4：用相关性告诉读者什么能预测、什么不能预测

对应位置：§4.5、Figure 3、Table 4、Table 4b、Table 5。

Figure 3 是 PushT n=9 LeWM sweep 的核心诊断图。

读法：

- raw Spearman 告诉我们整个 sweep 上谁和谁一起变。
- partial Spearman 告诉我们移除 `std_max` 单调趋势后，metric 是否还有 residual signal。

关键结论：

- fragility metric 对 clean success 和 px+goal 0.08 success 有一定 residual checkpoint-quality signal。
- 但对 clean-to-OOD drop 没有 OOD-specific explanatory power。
- 因此它是 model-selection / checkpoint-quality aid，不是 OOD robustness oracle。

### Step 5：机制归因要收窄

对应位置：§4.6、Figure 5、Appendix E。

机制主张是：

pixels → encoder shift → predictor transduction → CEM planner failure。

Figure 5 是 schematic，不再画旧的 cross-method cost-surface attribution bar。

Cost-swap 只保留为 TwoRoom one-off sanity check：换 CEM cost 只从 36 到 42，远低于 clean reference 69.7。因此只能说 cost function alone unlikely to explain collapse，不能说 cost surface 在所有任务都不是主因。

## 外部 baseline 的当前位置

对应位置：§2.1、§4.2、Appendix F。

PLDM 引用链：

- Sobal et al. 2022：Joint Embedding Predictive Architectures Focus on Slow Features。
- Sobal et al. 2025：Stress-Testing Offline Reward-Free Reinforcement Learning，即具体 PLDM latent-dynamics planning baseline。
- stable-worldmodel-v1：本文实际使用的 PLDM implementation 来源。

PLDM 当前数据只覆盖 PushT clean-trained baseline：

- PLDM clean：75.33 ± 3.68。
- PLDM px+goal 0.05：43.67 ± 4.64。
- PLDM px+goal 0.08：10.00 ± 2.16。
- clean → px+goal 0.08 drop：65.33pt。

它支持的窄结论：

- clean-trained visual world model 的 control-time pixel+goal noise cliff 不只发生在 LeWM。

它不支持的强结论：

- 不能证明 invariance-resolution trade-off 已跨架构成立。
- 不能证明 PLDM noise training 也有同样 clean/OOD optimum dissociation。
- 不能替代 DINO-WM 或 PLDM sweep。

## 图表功能地图

### 主图

- Figure 1：hero figure，展示 base OOD cliff 和 noise-training recovery。
- Figure 2：四任务 sweep curves，展示 task-specific `std_max`。
- Figure 3：PushT fragility metric scatter，展示 clean signal vs OOD-drop non-oracle。
- Figure 4：diagnostic radar，展示 base vs representative diagnostic checkpoint 的机制差异。
- Figure 5：mechanism schematic，展示 pixels → encoder → predictor → CEM。
- Figure 6：clean vs px+goal 0.08 Pareto trajectory，展示 clean/robustness optimum dissociation。

### 主表

- Table 1：LeWM-base visual OOD cliff。
- Table 2：完整 LeWM noise sweep，标 clean point-best 和 px+goal 0.08 point-best。
- Table 3：base vs representative diagnostic checkpoint。
- Table 4：per-task raw Pearson/Spearman vs OOD drop。
- Table 4b：partial Spearman vs OOD drop conditioned on `std_max`。
- Table 5：PushT fragility metric 的 raw/partial correlation 拆解。

### 附录

- Appendix A：实验细节、noise implementation、evaluation protocol、figure rendering。
- Appendix B：LeWM-base 四任务完整诊断。
- Appendix C：heteroscedastic loss 公式。
- Appendix D：heteroscedastic loss negative result。
- Appendix E：cost-swap sanity check。
- Appendix F：PLDM PushT external sanity check。

## 正文立场边界

### 可以强说

- LeWM-base under visual OOD collapses across all four tasks。
- Noise training can recover much of the OOD performance。
- No single global `std_max` is optimal across tasks。
- The trade-off is task-dependent and tied to representation compression vs task-resolution needs。
- PushT PLDM sanity check indicates the visual-noise cliff is not isolated to LeWM。

### 必须保守说

- PLDM / DINO-WM 跨架构泛化还在后续实验中。
- Cross-checkpoint diagnostic 是 residual checkpoint-quality aid，不是 OOD oracle。
- Cost-swap 只是 one-off sanity check，不是四任务归因。
- Table 2 的 optima 应按 plateau 理解，而不是 reviewer 可抓的唯一 best。

### 不应在 Paper 1 中说

- 自适应分辨率方法已经解决问题。
- PLDM / DINO-WM 已证明同样 trade-off。
- cost surface 不是任何任务的主因。
- 同一 `std_max` 下 metric 直接比较出 robustness；当前是 residualization，不是 repeated same-std comparison。

## 合作者讨论入口

### 故事线是否足够清楚

- 读者是否能从 Figure 1 直接理解 “clean success 不等于 visual robustness”？
- Figure 2 / Figure 6 是否足够说明 “noise useful but task-specific”？
- Table 3 / Figure 4 是否清楚区分 representative diagnostic checkpoint 与 point-best？

### 立场强度是否合适

- PLDM sanity check 是否放在 Appendix F + §4.2 一句话足够，还是应在 Limitations 中再强调一次？
- “not isolated to LeWM” 是否过强？当前证据是 PushT-only PLDM clean-trained eval。
- 对 JEPA invariance narrative 的措辞是否应再弱化为 “under this control-time pixel-noise setting”？

### 证据链是否完整

- Reviewer 会不会要求 PLDM/DINO-WM 至少 TwoRoom + PushT？
- 是否需要把 PLDM noise-training sweep 留到 v1，而 arXiv v0 只放 sanity check？
- 是否需要补充 exact eval command / seed manifest 到 DATA_MANIFEST？

### 下一步实验如何接入

- 第一优先级：PushT + TwoRoom 的 PLDM / DINO-WM clean-trained eval-only OOD。
- 第二优先级：少量 noise-training points，而不是完整 4 tasks × 8 sweep。
- 最小跨架构问题：visual OOD cliff 是否普遍？
- 更强跨架构问题：invariance-resolution trade-off 是否普遍？

## 数据 source-of-truth

- LeWM canonical eval：`assets/paper1_data/canonical_evals_20260517.json`。
- LeWM canonical diagnostics：`assets/paper1_data/canonical_diagnostics_20260517.json`。
- PLDM external sanity check：`assets/paper1_data/canonical_external_baselines_20260520.json`。
- Release consistency checker：`tools/check_paper1_consistency.py`。
- LaTeX upload source：`paper1/main.tex` + `paper1/references.bib`。
- Current compiled PDF：`paper1/main.pdf`。

## 推荐给合作者的阅读顺序

1. 先看 Abstract + Introduction，确认 paper 定位不是新算法，而是诊断研究。
2. 看 Figure 1 和 Table 1，确认视觉 OOD cliff。
3. 看 Figure 2 / Table 2 / Figure 6，确认 global noise 没有统一最优。
4. 看 Table 3 / Figure 4，理解 invariance-resolution trade-off。
5. 看 Figure 3 / Table 5，理解 diagnostic 的能力边界。
6. 看 Appendix F，理解 PLDM 只作为 external sanity check。
7. 最后看 Discussion，讨论是否需要调整 claims 的强度。
