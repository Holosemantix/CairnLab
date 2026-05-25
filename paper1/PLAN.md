# Paper 1 — 故事线与研究路线图

> Source of truth: `paper1/main.tex`. 数值、表格、图和 artifact 以论文正文与 `assets/paper1_data/` 为准。
> Last updated: 2026-05-25.

---

## 1. 一分钟版本（讨论入口）

JEPA 这类世界模型在 latent 空间预测而不是重建像素，因此一种常见 informal 直觉是：latent prediction 会降低保留 observation-level 细节的压力，从而更偏向保留对未来 target representation 有用的结构。Paper 1 不把这句话写成“JEPA 天然鲁棒性神话”，而是系统刻画它在 closed-loop control 下的 visual-corruption robustness boundary。

我们在 4 个机器人控制任务（PushT、TwoRoom、Reacher、Cube）上系统地测了一下：unperturbed evaluation images 上 86% 成功的 PushT，在 observation+goal image 加 Gaussian pixel noise（std=0.08）后跌到 5%；TwoRoom 从 94% 跌到 50%。给训练数据加同类噪声能基本恢复——但在当前 sweep grid 上没有观察到一个跨任务共同最优的噪声量：任务结构决定最佳剂量，unperturbed 最优和 robustness 最优甚至在同一任务上还会错开。

我们把这个现象命名为 **invariance-resolution trade-off**——加噪声同时压平无关像素变化（有益 invariance）和任务相关细节（有害 resolution loss）。一个 5 层诊断协议把机制拆开来读；第二个方法家族 PLDM 复现了 task-level signature，但 full diagnostic profile 与 LeWM compression chain 不同。我们也专门测了"label-free 诊断指标能否直接预测 corruption robustness"，结论是 partial-correlation null 在 LeWM、PLDM、joint n=18 三处稳定复现：诊断工具能做 mechanism localization 和 checkpoint selection，但不能替代真实 corruption evaluation。

本文是 **diagnostic paper 不是 method paper**。它的价值在于把这个 trade-off 命名、量化，并划清诊断工具的边界，为后续 method paper（plan-side robust CEM、adaptive resolution、spherical world model — §7.3）建立 baseline 与诊断框架。

---

## 2. 故事线（2–3 分钟版本，6 步逻辑链）

每一步独立可读；前一步推出下一步的必要性。

### Step 1 — 一个需要在 closed-loop control 中检验的直觉

JEPA（Joint-Embedding Predictive Architecture）把训练目标从"重建像素"换成"在 latent 空间预测未来表征"。更稳的表述是：latent prediction 可能降低保留 observation-level 细节的压力，鼓励模型保留对预测未来 target representation 有用的结构。已有工作研究了 augmentation、JEPA robustness、slow-feature distractors 等相邻问题；我们的缺口是 **JEPA-style latent world model + CEM closed-loop control + success-rate evaluation + cross-checkpoint diagnostics**。

### Step 2 — 实证：unperturbed performance 不蕴含 visual-corruption robustness

我们用统一的 4 task × 36 LeWM checkpoint × 3 evaluation seeds × 100 trajectories 协议测了一遍。**unperturbed 上 86% 的 PushT，在像素+goal 加 Gaussian noise std=0.08 后跌到 5%；TwoRoom 94% → 50%；Reacher、Cube 也有 20–44pt 的 drop。** 这不是表征空间的小毛病，是部署级失效。**Unperturbed success 和 visual-corruption robustness 是两条曲线，unperturbed 看好的模型不意味着 robust**。文中保留 `clean` 只是 artifact/condition name，正文定义为 unperturbed/original evaluation images。

### Step 3 — 直觉性解药"加噪声训练"碰到边界

把同类 Gaussian noise 加进训练能基本关闭这个 gap。但完整的 8-level noise sweep 揭示：**在当前 sweep grid 上没有单一 `std_max` 跨任务共同最优**。视觉冗余强的 TwoRoom 越加越好（最优 std=0.08）；接触/精细控制的 PushT 的 **unperturbed 最优在 std=0.03，robustness 最优在 std=0.06，两者错开**。"加噪声就行了"这种简单解答被这组数据关掉。

### Step 4 — 命名并解释这个边界：invariance-resolution trade-off

为什么没有 universal 最优？因为噪声训练同时产生两种压缩：把无关像素变化压平（有益的 **invariance**），同时把任务相关的细节也压平（有害的 **resolution 损失**）。任务结构决定二者权重——视觉冗余强的任务能吸收更多压缩，接触/精细控制任务不行。我们把这个张力命名为 **invariance-resolution trade-off**，作为后续机制分析的概念锚点。

### Step 5 — 机制：5 层诊断 + 跨方法验证

我们设计了一个 5 层诊断协议（encoder shift → encoder geometry → predictor sensitivity → latent-noise response → task resolution），把 control pipeline 拆成 5 个独立可测的阶段。**LeWM 上的证据支持 compression-chain reading：表征 effective rank 压缩 → 状态间分辨率丢失 → 可控性（inverse-dynamics R²）下降。** PLDM（第二个方法家族）复现 task-level signature，但 full diagnostic 显示它的 diagnostic profile 更一致地伴随 multi-step predictor drift 下降，rank/resolution 大体保留。因为我们没有对 PLDM 做 causal intervention，所以这里写成 **mechanism boundary / architecture-specific profile**，不写成“PLDM 因果证明动力学预测出问题”。

### Step 6 — 诊断工具的边界：能定位机制和选 checkpoint，不能替代 corruption evaluation

最后我们专门测了 model selection 实践中的实用问题："label-free 诊断指标能否直接预测 corruption robustness"。在 LeWM PushT n=9 sweep 上，最强单一指标（**fragility ratio**）对 corruption drop 的 unconditional 相关性看似很强（ρ=−0.77）。但 partial correlation 控制掉 std_max 后，**相关性塌到 +0.06（95% bootstrap CI [−0.00, +0.25]，含 0）**。这个 null 在 PLDM PushT 上复现（partial=−0.14，CI [−1.00, +0.87]），在 joint LeWM+PLDM n=18 上也复现（partial=+0.11，CI [−0.54, +0.71]）。**诊断工具的关键作用不是当 oracle，而是定位机制、筛 checkpoint、告诉后续方法该修哪一层**。这反而增强了后续 method paper 的落点。

---

## 3. 贡献写法 C1–C4

- **C1：系统化暴露问题。** 4 task × 8 noise level × 2 method × 3 eval seeds × 100 traj 的统一协议下，量化 latent predictive control 的 visual-corruption cliff，确认现象不是单一 ckpt / 单一架构的偶然。
- **C2：提出 invariance-resolution trade-off + 5 层诊断 toolkit。** 不只看 success rate，把 failure 拆到 encoder geometry / predictor sensitivity / latent-noise response / task resolution，并给出 partial-correlation 验证方案。
- **C3：机制解释 LeWM-centred + PLDM mechanism boundary 显式。** Noise augmentation gain 在 LeWM 上对应 "compression chain"；PLDM 复现 task-level signature，但 diagnostic profile 更一致地伴随 predictor drift 下降，机制层面 architecture-aware。
- **C4：诊断指标的范围明确。** 最强 cross-checkpoint diagnostic 是 checkpoint quality probe；partial-correlation 控制 std_max 后对 corruption drop 的 residual association null 在 LeWM/PLDM/joint 三处复现，95% bootstrap CI 全部含 0。

## 4. 写作立场

**应该坚持的强说法**：

- Latent prediction alone does not guarantee visual-corruption robustness for control.
- Visual-corruption failure is a real closed-loop control issue, not a representation-space curiosity.
- Noise training creates a task-dependent invariance-resolution trade-off.
- LeWM 的 mechanism evidence 支持 compression-chain reading.
- PLDM 支持现象跨方法，但 diagnostic profile 不等同于 LeWM compression chain.

**需要避免的过强说法**：

- 不要说所有 JEPA 都会同样崩溃.
- 不要说某个 diagnostic universally predicts robustness.
- 不要说 cost surface 已被排除为所有任务的主因.
- 不要把 blur eval-only 写成 blur training conclusion.
- 不要说 Gaussian-noise sweep 的 per-task signature 整体泛化到 blur；更稳的说法是 blur collapse 主要集中在 TwoRoom，其他任务整体更稳定，说明 visual fragility 能跨 Gaussian-noise axis 出现，但 task ordering 和 recovery profile 是 corruption-specific.
- 不要把 PLDM mechanism 写成 LeWM mechanism 的简单复制.

## 5. 当前 submit-readiness

- 主文 story 已闭环：failure → recovery → trade-off → mechanism → boundary.
- LeWM 是主 microscope，PLDM 是 second-family replication，blur 是 cross-corruption sanity check.
- 95% checkpoint-row bootstrap CI 已加入 Table 7 / Appendix F / 正文，用来约束 partial-correlation 结论强度.
- Success-rate tables 的 uncertainty 是 3 evaluation seeds 的 population std；correlation intervals 是 checkpoint-level bootstrap CI，二者口径已在正文区分.
- `tools/check_paper1_consistency.py` 已覆盖核心 artifact 与关键数值一致性.
- `paper1/main.pdf` 可 clean build（32 pages, 0 Overfull, 0 errors）.

**仍需要人工完成的一项**：

- **References final manual source audit**。机器辅助核对已更新：VJEPA 的 Noisy-TV / R² 表述、Alain-Bengio OpenReview/arXiv 口径、DrQ/DrQ-v2 conference year、seq-JEPA NeurIPS 2025 poster、DrQ author order 均已校正。提交前仍建议人工逐条打开最终 bibliography 页面做最后确认。

## 6. 讨论时常见问题

- 题目是否应该更强调 "latent prediction is not visual robustness"，还是更强调 "invariance-resolution trade-off"？
- PLDM 放在主文还是 appendix 的分量是否合适？
- Blur eval-only 是否足够作为 sanity check，还是需要后续 blur training v1？
- 五层诊断公式是否已足够清楚，还是需要把更多 metric definition 移到主文？
- 当前 paper 是投 empirical diagnostics 方向，还是后续补 algorithm 后转 method paper？

---

## 7. 研究路线图（多篇 paper 视角）

### 7.1 Paper 1 v0 — 当前状态

完成度 95%。主线 story、PLDM 复制、blur sanity check、bootstrap CI 都已就位。差一项人工 arXiv 核对（§5）就可提交。

### 7.2 Paper 1 v1 — 可选增强（不阻塞 v0）

| 项 | 工作量 | 价值 |
|---|---|---|
| arXiv 9 条 ID 人工核对 | ~1 hr | **v0 提交前必做** |
| BCa 加偏校正 bootstrap | ~半天 | percentile CI 升级；当前 null 已稳健，BCa 是 reviewer-friendly polish |
| Blur training sweep | ~1 周 | 把 App G 从 eval-only sanity check 升级为 blur-training recovery 实验 |
| I-JEPA / V-JEPA EMA 变体复制 | ~2–4 周 | 弥补 §5.5 Limitation 1 |
| DMC-Suite task 扩展 | ~1 周 | 弱化 "4 task cherry-picked" 质疑 |
| TD-MPC2 / DreamerV3 cross-arch | ~1 周 each | 测 5 层诊断是否能扩到 reconstruction-based world model |

### 7.3 Paper 2 候选 — 由 Paper 1 motivating 的 3 条独立方向

每条都是独立的方法贡献，可以并行推进。

#### 7.3.a Plan-side robust CEM

- **核心问题**：world model 已经 noise-fragile，CEM 又只在 single point estimate 上优化——能否只改 inference 阶段（不重训）就显著恢复 visual-corruption control？
- **由 Paper 1 哪里 motivating**：§4.6 mechanism attribution 表明 encoder 是主因但 cost surface 也有贡献；§4.6.1 cost-swap 只是单点 sanity check，证据强度不够。"训练侧 mitigation 已用 noise training 试过；推理侧 CEM 还没动过" 是 Paper 1 自然引出的对偶问题。
- **当前状态**：`config/eval/solver/robust_cem.yaml` 实装完成（top-K reranking、TTA-empirical belief、CVaR risk）；详细 plan 在 `planner_side_robustification_experiment_plan.md`；**系统 eval 未开始**。
- **相对 Paper 1 的 delta**：不重训、零 model 改动，只换 solver；把 Paper 1 mechanism 里 cost surface 的角色单独抽出来 quantify。

#### 7.3.b Adaptive resolution / AAAC

- **核心问题**：能否在 controller 端通过 per-token consistency routing，突破 input-side noise training 的 per-task tuning 边界？
- **由 Paper 1 哪里 motivating**：§5.5 "Additional ablation" 已验证 σ-head 学到了 prediction difficulty，但作为 loss reweighter 在 PushT 上 collapse（86% → 13%）；§5.5 Future direction 1 明示该方向。Paper 1 trade-off 是 input-side full picture，Paper 2b 要回答 controller-side 是否能进一步榨出增益。
- **当前状态**：完整 4 任务 sweep + 4 件套因果干预（constant_w、random_gate、shuffle_σ、shuffle_A）已完成。**关键数据**：PushT 强视觉扰动（px+goal 0.08）C1+C2 联用 = 85.33 vs C1 单独 75.75（**+9.58pt**）；causal claim "per-token routing 本身是主导项"（constant_w −28.67pt）经 ablation 证实。详细 plan 在 `plan_adaptive_resolution.md`；**paper 写作未开始**。
- **相对 Paper 1 的 delta**：Paper 1 诊断现象 + input-side fix 的边界；Paper 2b 提供 controller-side instantiation，与 input-side noise 正交可叠加。

#### 7.3.c Spherical world model / Field-JEPA

- **核心问题**：LeWM 用 SIGReg 强制 latent 分布为各向同性 Gaussian。对低内在维度任务（如 TwoRoom）这是否过约束？换成球面表征 + uniformity 损失能否在保留 anti-collapse 的同时更好对齐任务流形？
- **由 Paper 1 哪里 motivating**：与 Paper 1 互补——Paper 1 把现象诊断清楚，SWM 改 representation geometry 本身。也是 §5.5 future direction 2 "broader cross-architecture replication" 的另一种实现：换 representation regularizer 而不是换整个架构。
- **当前状态**：三阶段路线 V0/V1/V2（V0 spherical encoder + cosine pred loss + uniformity reg / V1 vMF per-observation concentration / V2 learnable ball-cap for OOD detection）；V0 实装完成（`jepa.py::SphericalJEPA`, `train_swm.py`）；详细 plan 在 `plan_v2.md` + `plan.md`；**系统 eval 未开始**。
- **相对 Paper 1 的 delta**：Paper 1 不动 LeWM 架构；SWM 把 representation geometry 当 first-order design choice。

### 7.4 长程方向

- **IB / rate-distortion 理论 framing**：把 invariance-resolution trade-off 形式化为 information bottleneck 或 rate-distortion 优化问题。Paper 1 §5.5 future direction 3 明示但当前认为现象未 stable 到值得形式化。
- **Sim-to-real corruption**：把 Gaussian noise / blur 替换成真相机噪声、光照变化、运动模糊等更接近 deployment 的 visual shifts。
- **Cross-architecture extension**：把 5 层诊断协议测在 reconstruction-based world model（DreamerV3）和 decoder-free latent MPC（TD-MPC2）上，看现象与机制是否跨更大架构空间复现。

---

## 8. 提交命令清单

```bash
# 一致性检查
python -m tools.check_paper1_consistency

# PDF 构建
cd paper1 && bash build.sh --clean
```

数据有变化时:

```bash
# 主图重生成
python -m tools.paper1_figs --out-dir assets/paper1_figs

# 跨方法相关性
python -m tools.pldm_correlation_analysis \
  --out assets/paper1_data/cross_method_corr_pldm_20260522.json

# Bootstrap CI
python -m tools.build_partial_corr_bootstrap \
  --out assets/paper1_data/partial_corr_bootstrap_20260523.json
```
