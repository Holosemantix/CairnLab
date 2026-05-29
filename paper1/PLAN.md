# Paper 1 — 故事线与研究路线图

> Source of truth: `paper1/main.tex`. 数值、表格、图和 artifact 以论文正文与 `assets/paper1_data/` 为准。
> Reframing 执行依据：`paper1/paper1_acpc_rewrite_execution_plan.md`。
> Last updated: 2026-05-29（reframed: latent invariance shorthand → action-conditioned predictive consistency）。

---

## 1. 一分钟版本（讨论入口）

JEPA 这类 latent predictive world model 在 latent 空间预测而非重建像素，常被期待能 abstract 掉 nuisance 视觉细节。但对 control 而言，视觉鲁棒性**不应**定义成 clean/corrupted 图像的 encoder 输出越接近越好。corrupted observation 完全可以 encode 成不同 latent；真正要求的是：在**同一历史、同一动作干预**下，世界模型预测的下一状态与多步 rollout 在任务相关坐标上保持一致。我们把这个性质命名为 **action-conditioned predictive consistency (ACPC)**，并配一个 discriminability countercondition：会改变 action-conditioned transition、cost 或最优行为的状态差异必须保持可区分。encoder-level latent closeness 既不充分也不必要。

Paper 1 用受控 Gaussian pixel corruption 作为**探针**（不是新 benchmark）来检验这个性质：在 4 个控制任务（PushT、TwoRoom、Reacher、Cube）上，无噪声训练的 LeWM 在 observation+goal 加噪（std=0.08）后，PushT 从 86% 掉到 5%、TwoRoom 从 94% 掉到 50%。给训练加同类噪声能恢复大部分性能，但它是一个 **coarse global scalar pressure**，呈 broad task-dependent plateaus，而非 principled 解（不再写成”无 universal std_max”定理）。

两个负结果指向新指标：(i) 控制掉训练噪声后，single-step encoder/predictor fragility 不能解释 corruption gap（partial-correlation null 在 LeWM、PLDM、joint n=18 三处复现），而 multi-step predictor drift 在部分任务（Reacher partial ρ=+0.79）保留残差信号；(ii) heteroscedastic σ-head 用 prediction error 下采样 hard transitions，让 PushT clean 从 86% 崩到 13%——说明 **hard ≠ nuisance**。诊断工具只做 mechanism localization 与 checkpoint selection，**不预测** robustness。

本文是 **reframing + diagnostic paper**，不提出新训练算法。中心贡献是把鲁棒性重新定义在 action-conditioned predictive dynamics 层，给出 ACPC 系列 paired-diagnostic protocol（目前尚未作为正文结果计算），并据此指出方法方向 adaptive predictive-dynamics consistency（plan-side robust CEM、adaptive resolution、spherical world model — §7.3 仍为后续独立方向）。CEM 只是 evaluation 阶段的 action optimizer，不属于 thesis。

---

## 2. 故事线（reframed 逻辑链）

每一步独立可读；前一步推出下一步的必要性。CEM 只在 setup / evaluation 出现，不进 thesis。

### Step 1 — 承认 JEPA 优势但限定其作用域

相比 reconstruction，joint-embedding prediction 降低了重建 high-magnitude nuisance 特征的压力，但仍需 augmentation 或 inductive bias 才能判断”什么是 irrelevant”（Joint-Embedding vs Reconstruction）。在 control 里，irrelevance 不是 image-level / label-level，而是 **action / transition / cost dependent**。所以不能写”latent prediction 不重建像素，所以 robust”，只能写”它减少了一种压力，但没回答 control 该 contract 哪些视觉变化”。

### Step 2 — 吸收 LeJEPA world-model 理论但不撞车

*When Does LeJEPA Learn a World Model?* 说明在一定假设下 LeJEPA 能线性恢复世界 latent 变量并支持 latent-space planning，主要解决 state-side identifiability，并指出 action-conditioned transition 是自然扩展。我们的问题是 **complementary**：给定一个学好的 visual world model，clean / corrupted observation 在同一动作干预下是否产生一致预测。我们**不**声称证明或加强任何 identifiability / planning-equivalence 结论。

### Step 3 — 重新定义视觉鲁棒性：ACPC 而非 latent invariance

旧代理 d(z_t, z̃_t) 小被换成：在同一动作序列下，F^k(z_t, a) 与 F^k(z̃_t, a) 在任务相关读数 Π 上一致（k=1…H）。允许 z≠z̃。配 discriminability countercondition：会改变 future transition / cost / 最优行为的 state 差异必须保留。central tension 不是抽象 invariance-vs-resolution，而是”same-state perturbation pairs → 高 predictive consistency；action-distinct pairs → 高 predictive discriminability”。形式化见 main.tex §3（`sec:acpc`）。

### Step 4 — 现有实验作为现象锚点 / 探针，而非最终故事

统一 4 tasks × 9 configs（base + 8 noise levels）× 3 seeds × 100 traj 协议下：无噪训练 corruption cliff（PushT 86→5、TwoRoom 94→50、Reacher/Cube 20–44pt drop），noise training 大幅恢复但是 **coarse global scalar pressure**（broad task-dependent plateaus）。重点写法：失效的本质不是”像素有噪声”，而是 visual perturbation 把模型推进了不同的 action-conditioned predictive neighborhood。PLDM 复现 task-level signature（method-family 证据）。任务面板必须写成 heterogeneous stress panel：TwoRoom / PushT / Reacher / Cube 分别覆盖 discrete redundancy、contact-heavy manipulation、low-dimensional continuous control、structured 3D coupling，不声称代表 all control tasks。

### Step 5 — partial-correlation 负结果 → 新指标动机

控制 std_max 后，最强 single-step fragility ratio 对 corruption drop 的 partial ρ 塌到 **+0.06**（95% bootstrap CI [−0.00, +0.25]，含 0），在 PLDM（−0.14）、joint n=18（+0.11）复现。Reacher 的 multi-step rollout drift partial ρ=**+0.79** 保留残差信号。结论：label-free pointwise / single-step 不是 control robustness 的正确抽象；更接近问题本质的是 **multi-step action-conditioned predictive consistency（ACPC-H）**。

### Step 6 — heteroscedastic σ-head 负结果放核心

σ-head 学到了 prediction difficulty，但把 hard transitions downweight 后 PushT clean 从 86% 崩到 13%。这说明 **hard ≠ nuisance**：contact-sensitive transition 预测难，恰因为它们 action-relevant，不能被 global compression / uncertainty downweighting / naive invariance 丢掉。故方法应是 **sensitivity-aware predictive consistency**，不是 error-based downweighting。

### Step 7 — 方法引子：Adaptive Predictive-Dynamics Consistency

落点：在同一动作干预下 regularize clean/corrupted **predictions**（不是 encoder outputs），按 action / transition sensitivity gating；low sensitivity → 更强 consistency，high sensitivity → 更强 discriminability guard，high prediction difficulty 本身不 downweight。方法名暂定 APDC / Selective Predictive Consistency，**本文只写成 design implication / future direction**，不当成已完成结果。

---

## 3. 贡献写法 C1–C4

- **C1 — Problem reframing。** 视觉鲁棒性应定义为 action-conditioned predictive consistency + discriminability countercondition，而非 encoder-level latent invariance。main.tex §3（`sec:acpc`）给出形式化与 downstream readout 的边界。
- **C2 — Diagnostic evidence。** 统一 4 task × 9 configs（base + 8 noise levels）× 3 seeds × 100 traj（PLDM 复现）下：visual perturbation 造成 closed-loop failure；noise augmentation 只是 coarse global pressure；pointwise single-step fragility 不够（控 std_max 后 partial ρ=+0.06，PLDM/joint 复现），multi-step predictor drift 在部分任务保留残差（Reacher +0.79）。
- **C3 — Selective-consistency diagnostics。** 定义 ACPC-1 / ACPC-H / PCC / CRA / MAF / ADM / SPRR，比较同一动作序列下 clean/corrupted predictions 并单独度量 action-relevant discriminability。**这些 paired ACPC quantities 是下一步 diagnostic pass 的 protocol definitions，目前不是正文结果。**
- **C4 — Method-design implication。** 据上指出 adaptive predictive-dynamics consistency：在 predictor 之后做 consistency，按 action sensitivity gating，保留 action-sensitive transitions。hetero σ-head 负结果（hard ≠ nuisance）说明为何 error-based gate 是错的。无方法实验时只写成 design implication / future direction。

## 4. 写作立场

**应该坚持的强说法**：

- 视觉鲁棒性应定义为 action-conditioned predictive consistency，而非 encoder-level latent invariance；latent closeness 既不充分也不必要。
- 必须保留 discriminability countercondition：改变 action-conditioned transition / cost / 最优行为的差异要可区分。
- Latent prediction alone 不保证 control 的 visual-corruption robustness；visual-corruption failure 是真实 closed-loop 控制问题，不是 representation-space curiosity。
- noise training 是 coarse global scalar pressure（broad task-dependent plateaus），不是 principled 解。
- pointwise single-step fragility 不预测 corruption gap；multi-step predictive consistency 更接近问题本质。
- hard ≠ nuisance：difficulty-based downweighting 会丢掉 action-relevant transitions。
- visual perturbation 是 controlled probe，不是新 benchmark。

**需要避免的过强说法**：

- 不要把 "JEPA + rollout + CEM planning" 当 novelty；CEM 只是 evaluation-time action optimizer。
- 不要把中心概念叫成 "planning equivalence"；planning / cost / action 是 downstream readout。
- 不要把 robustness 定义成 z_clean ≈ z_corrupted。
- 不要写 "no universal std_max" 强定理；用 coarse global scalar pressure / broad task-dependent plateaus。
- 不要说任何 diagnostic universally predicts robustness（含 ACPC-H / SPRR）；它们 localize mechanism、motivate method target。
- 不要把 ACPC 系列 paired 指标写成已计算结果——它们是下一步 diagnostic pass 的 protocol definitions。
- 不要声称证明或加强 LeJEPA identifiability。
- 不要说所有 JEPA 都会同样崩溃；不要把 PLDM mechanism 写成 LeWM 的简单复制；不要把 blur eval-only 写成 blur training conclusion（blur collapse 主要集中在 TwoRoom，task ordering 是 corruption-specific）。

## 5. 当前 submit-readiness

**状态：not submit-ready；正在 reframe 到 predictive-consistency diagnostics，可能加一个 lightweight method（execution plan §12）。**

- 框架已 reframe：title / abstract / intro / related work / 新增 §3 ACPC 概念+诊断 / discussion / conclusion 已围绕 action-conditioned predictive consistency 重写；实验数值、表格、artifact **未改**（仍是 corruption cliff → noise recovery → diagnostics → partial-corr null → PLDM → blur）。
- ACPC 系列指标（ACPC-1/H、PCC、CRA、MAF、ADM、SPRR）在正文改为 **paired-diagnostic protocol definitions**；它们尚未作为 empirical evidence 报告。
- Phase 0 runner 已加：`tools/paper1_phase0_acpc.py` 会从 canonical eval manifest 解析 checkpoints，并在存在 loadable model object 时计算 ACPC-1/H、PCC、CRA、MAF、ADM action-distance proxy、SPRR；`--dry-run` 可先检查本机路径覆盖率。当前本地 canonical LeWM `path` 多数只有 `eval_results`，缺少可直接 `torch.load` 的 model object，实际出数前需要补齐 checkpoint root 或恢复 model object files。
- 最小可立项版本需先完成 **Phase 0**（execution plan §8）：用现有 checkpoints 计算 ACPC-H / PCC / ranking / ADM，证明比 old fragility ratio 更贴近 closed-loop failure，或至少解释 heteroscedastic negative result；related work 已明确 ViGMO / Bisim-JEPA / LeJEPA theory 的边界。Phase 0 成立后再进 Phase 1 方法实验。
- `paper1/main.pdf` 可 clean build（37 pages）；`tools/check_paper1_consistency.py` 仍通过（数值/artifact 未动）。

**仍需要人工完成**：

- **References final source audit**。reframe 新增条目已做首轮核验并移除 `TODO verify`：`bsmpc` 已替换为 Shimizu--Tomizuka, ICLR 2025 / arXiv:2410.04553；`voelcker2025calibratedvalueaware` 已改为 ICML 2025 PMLR 267 的正式题名 *Calibrated Value-Aware Model Learning with Probabilistic Environment Models*；`dupuis2023vibr`、`gelada2019deepmdp` 已补 PMLR volume/pages。提交前仍需做最终人工 bib audit，尤其是 2026 arXiv 条目是否已有新版 metadata。

## 6. 讨论时常见问题

- ACPC 系列指标在 Phase 0 上能否比 old fragility ratio 更贴近 corrupted success / corruption gap？（失败判据见 execution plan §8.7）
- candidate action sequence 来源如何固定（CEM 采样 / 固定 random / dataset actions），保证 clean/noisy 同一 candidate set？
- Π（task-relevant predictive readout）与 Ψ（discriminability readout）具体取什么？full latent / transition delta / cost feature / learned projection？
- 是否补 Phase 1 方法实验后转成 method paper，还是先以 reframing + diagnostic 形态投出？
- 新增 related work 边界（Bisim-JEPA / ViGMO / value equivalence / LeJEPA theory）是否足够区分，避免被 reviewer 当作 condition stacking？

---

## 7. 研究路线图（多篇 paper 视角）

### 7.1 Paper 1 v0 — 当前状态

已重构为 **action-conditioned predictive consistency** 的 reframing + diagnostic paper。实验证据（corruption cliff、noise sweep、PLDM、blur、partial-corr null、hetero 负结果）保持不变，改作 ACPC 的现象锚点 / 探针。**不再按旧版本直接挂出**：最小可立项需先完成 Phase 0（计算 ACPC-H / PCC / ranking / ADM 并证明解释力，execution plan §8）以及 references 人工核对（§5）。注意 §7.3.b（adaptive resolution / per-token consistency）已与本文 motivate 的 APDC 高度重合——若补 Phase 1 方法实验，应明确 Paper 1 与 Paper 2b 的边界。

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
- **由 Paper 1 哪里 motivating**：§5.5 "Additional ablation" 已验证 σ-head 学到了 prediction difficulty，但作为 loss reweighter 在 PushT 上 collapse（86% → 13%）；§5.5 Future direction 1 明示该方向。Paper 1 给出 input-side selective-consistency 诊断图景，Paper 2b 要回答 controller-side 是否能进一步榨出增益。
- **当前状态**：完整 4 任务 sweep + 4 件套因果干预（constant_w、random_gate、shuffle_σ、shuffle_A）已完成。**关键数据**：PushT 强视觉扰动（px+goal 0.08）C1+C2 联用 = 85.33 vs C1 单独 75.75（**+9.58pt**）；causal claim "per-token routing 本身是主导项"（constant_w −28.67pt）经 ablation 证实。详细 plan 在 `plan_adaptive_resolution.md`；**paper 写作未开始**。
- **相对 Paper 1 的 delta**：Paper 1 诊断现象 + input-side fix 的边界；Paper 2b 提供 controller-side instantiation，与 input-side noise 正交可叠加。

#### 7.3.c Spherical world model / Field-JEPA

- **核心问题**：LeWM 用 SIGReg 强制 latent 分布为各向同性 Gaussian。对低内在维度任务（如 TwoRoom）这是否过约束？换成球面表征 + uniformity 损失能否在保留 anti-collapse 的同时更好对齐任务流形？
- **由 Paper 1 哪里 motivating**：与 Paper 1 互补——Paper 1 把现象诊断清楚，SWM 改 representation geometry 本身。也是 §5.5 future direction 2 "broader cross-architecture replication" 的另一种实现：换 representation regularizer 而不是换整个架构。
- **当前状态**：三阶段路线 V0/V1/V2（V0 spherical encoder + cosine pred loss + uniformity reg / V1 vMF per-observation concentration / V2 learnable ball-cap for OOD detection）；V0 实装完成（`jepa.py::SphericalJEPA`, `train_swm.py`）；详细 plan 在 `plan_v2.md` + `plan.md`；**系统 eval 未开始**。
- **相对 Paper 1 的 delta**：Paper 1 不动 LeWM 架构；SWM 把 representation geometry 当 first-order design choice。

### 7.4 长程方向

- **IB / rate-distortion 理论 framing**：把 selective-consistency tension 形式化为 information bottleneck 或 rate-distortion 优化问题。Paper 1 §5.5 future direction 3 明示但当前认为现象未 stable 到值得形式化。
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
