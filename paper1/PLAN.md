# Paper 1 — 故事线与研究路线图

> Source of truth: `paper1/main.tex`. 数值、表格、图和 artifact 以论文正文与 `assets/paper1_data/` 为准。
> Reframing 执行依据：`paper1/paper1_acpc_rewrite_execution_plan.md`。
> Last updated: 2026-06-08（synced fixed-eval target-view reruns, full-sequence mainline, figure pruning, and PDF/check status）。

---

## 1. 一分钟版本（讨论入口）

JEPA 这类 latent predictive world model 在 latent 空间预测而非重建像素，常被期待能 abstract 掉 nuisance 视觉细节。但对 control 而言，视觉鲁棒性**不应**定义成 clean/corrupted 图像的 encoder 输出越接近越好。corrupted observation 完全可以 encode 成不同 latent；真正要求的是：在**同一历史、同一动作干预**下，世界模型预测的下一状态与多步 rollout 在任务相关坐标上保持一致。我们把这个性质命名为 **action-conditioned predictive consistency (ACPC)**，并配一个 discriminability countercondition：会改变 action-conditioned transition、cost 或最优行为的状态差异必须保持可区分。encoder-level latent closeness 既不充分也不必要。

Paper 1 用受控 Gaussian pixel corruption 作为**探针**（不是新 benchmark）来检验这个性质：在 4 个控制任务（PushT、TwoRoom、Reacher、Cube）上，无噪声训练的 LeWM 在 observation pixels 加噪、goal 保持 clean（std=0.08）后，PushT 从 86% 掉到 4%、TwoRoom 从 94% 掉到 66%；observation+goal 同时加噪作为更强 stress condition 单独报告。给训练加同类噪声能恢复大部分性能，但它是一个 **coarse global scalar pressure**，呈 broad task-dependent plateaus，而非 principled 解（不再写成”无 universal std_max”定理）。

三个负结果限定 claim 边界：(i) 控制掉训练噪声后，single-step encoder/predictor fragility 不能稳定解释 corruption gap（PushT LeWM partial ρ=+0.19，CI 含 0；PLDM −0.05；joint n=18 +0.22），multi-step predictor drift 在 clean-goal observation-noise endpoint 下也只剩弱残差信号（Reacher partial ρ=+0.37，CI 很宽）；(ii) heteroscedastic σ-head 用 prediction error 下采样 hard transitions，让 PushT clean 从 86% 崩到 13%——说明 **hard ≠ nuisance**；(iii) completed target-view ablation 显示 perturbed-history → original-future one-step denoising 并不 closed-loop rollout consistent，fixed eval 后 PushT pixels 0.08 仍只有 6.75%，而 matched full-sequence perturbed-target branch 是 72.83%。诊断工具只做 mechanism localization 与 checkpoint selection，**不预测** robustness。

本文是 **reframing + diagnostic paper**，不提出新训练算法。中心贡献是把鲁棒性重新定义在 action-conditioned predictive dynamics 层，并补充 dense Gaussian-noise ACPC basin diagnostic：在同一 state、同一 action sequence 下比较 clean/noised views 的 encoder radius 与 rollout prediction radius。该诊断是主文 paired evidence，只使用 Gaussian noise eval grid（0.01..0.08），与 noise sweep 训练 family 一致；blur/resize 不混入 ACPC 主证据。Phase 0 full-sweep ACPC artifact 保留为 Appendix H exploratory sanity check，说明 ACPC-H / PCC / CRA / MAF 等 paired probes 可计算且 face-valid，但不写成 universal robustness predictor。target-view ablation 作为 Appendix I negative result，支持把 **full-sequence perturbed-target training** 保留为当前主线。CEM 只是 evaluation 阶段的 action optimizer，不属于 thesis。

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

统一 4 tasks × 9 configs（base + 8 noise levels）× 3 seeds × 100 traj 协议下：无噪训练在 clean-goal observation-noise endpoint 上出现 corruption cliff（PushT 86→4、TwoRoom 94→66、Reacher 59→18、Cube 67→47），noise training 大幅恢复但是 **coarse global scalar pressure**（broad task-dependent plateaus）。重点写法：失效的本质不是”像素有噪声”，而是 visual perturbation 把模型推进了不同的 action-conditioned predictive neighborhood。PLDM 复现 task-level signature（method-family 证据）。任务面板必须写成 heterogeneous stress panel：TwoRoom / PushT / Reacher / Cube 分别覆盖 discrete redundancy、contact-heavy manipulation、low-dimensional continuous control、structured 3D coupling，不声称代表 all control tasks。

### Step 5 — partial-correlation 负结果 → 新指标动机

控制 std_max 后，最强 single-step fragility ratio 对 corruption drop 的 PushT partial ρ 是 **+0.19**（95% bootstrap CI [−0.00, +0.70]，含 0），PLDM 为 **−0.05**，joint n=18 为 **+0.22**。Reacher 的 multi-step rollout drift partial ρ=**+0.37**，CI [−0.35, +0.99] 很宽。结论：label-free pointwise / single-step 不是 control robustness 的正确抽象；更接近问题本质的是 **multi-step action-conditioned predictive consistency（ACPC-H）**，但现有诊断仍是 mechanism localization，不是 robust oracle。

### Step 6 — heteroscedastic σ-head 负结果放核心

σ-head 学到了 prediction difficulty，但把 hard transitions downweight 后 PushT clean 从 86% 崩到 13%。这说明 **hard ≠ nuisance**：contact-sensitive transition 预测难，恰因为它们 action-relevant，不能被 global compression / uncertainty downweighting / naive invariance 丢掉。故方法应是 **sensitivity-aware predictive consistency**，不是 error-based downweighting。

### Step 7 — 方法引子：Adaptive Predictive-Dynamics Consistency

completed target-view ablation 已排除一个简单方向：perturbed-history → original-future one-step denoising 不是充分修复，原因是训练 teacher-forced one-step，而 CEM eval 是 latent autoregressive closed-loop rollout，会把 predicted latent 作为下一步输入。因此当前主线改回 full-sequence perturbed-target augmentation；真正的方法落点是在同一动作干预下 regularize clean/corrupted **predictions**（不是 encoder outputs），按 action / transition sensitivity gating；low sensitivity → 更强 consistency，high sensitivity → 更强 discriminability guard，high prediction difficulty 本身不 downweight。方法名暂定 APDC / Selective Predictive Consistency，**本文只写成 design implication / future direction**，不当成已完成结果。

---

## 3. 贡献写法 C1–C4

- **C1 — Problem reframing。** 视觉鲁棒性应定义为 action-conditioned predictive consistency + discriminability countercondition，而非 encoder-level latent invariance。main.tex §3（`sec:acpc`）给出形式化与 downstream readout 的边界。
- **C2 — Diagnostic evidence。** 统一 4 task × 9 configs（base + 8 noise levels）× 3 seeds × 100 traj（PLDM 复现）下：visual perturbation 造成 closed-loop failure；noise augmentation 只是 coarse global pressure；pointwise single-step fragility 不够（控 std_max 后 PushT partial ρ=+0.19，PLDM −0.05，joint +0.22，CI 均不支持稳定 predictor），multi-step predictor drift 在 clean-goal observation-noise endpoint 下也只剩弱残差（Reacher +0.37）。
- **C3 — Selective-consistency diagnostics。** 定义 ACPC-1 / ACPC-H / PCC / CRA / MAF / ADM / SPRR，比较同一动作序列下 clean/corrupted predictions 并单独度量 action-relevant discriminability。当前正文已报告 Gaussian-noise ACPC basin diagnostic（`assets/paper1_data/acpc_basin_diagnostics.json`）：LeWM 36 ckpts、epoch-10、clean + noise std 0.01..0.08 views，输出 encoder radius / prediction radius / contraction。Phase 0 也已对 LeWM+PLDM full sweep 出数（`assets/paper1_data/acpc_phase0_diagnostics.json`，72/72 rows ok），作为 Appendix H face-validity / mechanism-localization evidence；不能写成单指标预测 robustness。
- **C4 — Method-design implication。** completed target-view ablation 排除 simple original-future denoising target，并把 full-sequence perturbed-target training 定位为 rollout-consistent baseline。据上指出 adaptive predictive-dynamics consistency：在 predictor 之后做 consistency，按 action sensitivity gating，保留 action-sensitive transitions。hetero σ-head 负结果（hard ≠ nuisance）说明为何 error-based gate 是错的。无方法实验时只写成 design implication / future direction。

## 4. 写作立场

**应该坚持的强说法**：

- 视觉鲁棒性应定义为 action-conditioned predictive consistency，而非 encoder-level latent invariance；latent closeness 既不充分也不必要。
- 必须保留 discriminability countercondition：改变 action-conditioned transition / cost / 最优行为的差异要可区分。
- Latent prediction alone 不保证 control 的 visual-corruption robustness；visual-corruption failure 是真实 closed-loop 控制问题，不是 representation-space curiosity。
- noise training 是 coarse global scalar pressure（broad task-dependent plateaus），不是 principled 解。
- full-sequence perturbed-target training 是当前证据下的 rollout-consistent mainline；perturbed-history → original-future target-view ablation 是 negative result。
- pointwise single-step fragility 不预测 corruption gap；multi-step predictive consistency 更接近问题本质。
- hard ≠ nuisance：difficulty-based downweighting 会丢掉 action-relevant transitions。
- visual perturbation 是 controlled probe，不是新 benchmark。

**需要避免的过强说法**：

- 不要把 "JEPA + rollout + CEM planning" 当 novelty；CEM 只是 evaluation-time action optimizer。
- 不要把中心概念叫成 "planning equivalence"；planning / cost / action 是 downstream readout。
- 不要把 robustness 定义成 z_clean ≈ z_corrupted。
- 不要写 "no universal std_max" 强定理；用 coarse global scalar pressure / broad task-dependent plateaus。
- 不要说任何 diagnostic universally predicts robustness（含 ACPC-H / PCC / CRA / MAF / SPRR / basin contraction）；它们 localize mechanism、motivate method target。
- 不要把 perturbed-history → original-future 写成改进方法；它在 completed target-view ablation 中失败，主要价值是暴露 closed-loop train/eval consistency 边界。
- 不要把 Phase 0 ACPC 写成方法结果或因果证据；它是 post-hoc checkpoint diagnostic，且 ADM 目前是 action-distance proxy，不是 oracle task-state margin。
- 不要把 ACPC basin 写成跨 corruption-family 结果：主证据只覆盖 Gaussian noise eval，且与 Gaussian-noise training sweep 匹配。blur/resize 仍只是 Appendix G 的 eval-only stress test，不能和 ACPC basin 混写。
- 不要过强声称“encoder 仍明显分散但 predict 聚合”普遍成立；dense noise diagnostic 显示 robust checkpoints 通常是 encoder basin 与 prediction basin 同时缩小，prediction radius 是 control-facing readout。
- 不要声称证明或加强 LeJEPA identifiability。
- 不要说所有 JEPA 都会同样崩溃；不要把 PLDM mechanism 写成 LeWM 的简单复制；不要把 blur eval-only 写成 blur training conclusion（blur collapse 主要集中在 TwoRoom，task ordering 是 corruption-specific）。

## 5. 当前 submit-readiness

**状态：near submit-ready after final source audit；ACPC dense Gaussian-noise basin diagnostic、Phase 0 exploratory appendix、completed target-view negative ablation 已补，PDF 与 consistency checks 已通过。**

- 框架已 reframe：title / abstract / intro / related work / §3 ACPC 概念+诊断 / discussion / conclusion 已围绕 action-conditioned predictive consistency 重写；新增 §4.x Gaussian-noise ACPC basin table；main-text radar / mechanism schematic 已移除，避免证据弱图压低严谨性。
- ACPC 主文 empirical evidence 现在是 **Gaussian-noise basin radius**：`tools/paper1_acpc_basin.py` 从 canonical eval manifest 解析 LeWM 36 个 epoch-10 checkpoints，在 clean + Gaussian noise std 0.01..0.08 views 上计算 encoder radius / prediction radius / contraction，输出 `assets/paper1_data/acpc_basin_diagnostics.json`。
- ACPC 系列 paired probes（ACPC-1/H、PCC、CRA、MAF、ADM、SPRR）已有 full-sweep Phase 0 artifact：`assets/paper1_data/acpc_phase0_diagnostics.json`，72 rows = 2 methods × 4 tasks × 9 std levels，全部 `status=ok`。当前只作为 Appendix H face-validity / mechanism-localization evidence，不能声称预测 robustness。
- target-view ablation 已完成并作为 Appendix I negative result 纳入：`assets/paper1_data/target_view_closed_loop_summary.json` 记录四任务八个 target-noise checkpoints，支持 full-sequence perturbed-target branch 作为当前主线。
- related work 已明确 ViGMO / Bisim-JEPA / LeJEPA theory 的边界。提交前仍需 references final source audit。
- `paper1/main.pdf` 可 clean build（33 pages）；`tools/check_paper1_consistency.py` 通过；`git diff --check` 通过。

**仍需要人工完成**：

- **References final source audit**。reframe 新增条目已做首轮核验并移除 `TODO verify`：`bsmpc` 已替换为 Shimizu--Tomizuka, ICLR 2025 / arXiv:2410.04553；`voelcker2025calibratedvalueaware` 已改为 ICML 2025 PMLR 267 的正式题名 *Calibrated Value-Aware Model Learning with Probabilistic Environment Models*；`dupuis2023vibr`、`gelada2019deepmdp` 已补 PMLR volume/pages。提交前仍需做最终人工 bib audit，尤其是 2026 arXiv 条目是否已有新版 metadata。
- **Optional final rerun audit**。若要把 camera-ready 表格完全绑定到已修正 eval 入口，建议用当前代码重新跑 32 个 eval/diagnostics；这属于最终归档准确性，不是当前 mainline claim 的逻辑前提。

## 6. 讨论时常见问题

- Dense Gaussian-noise ACPC basin 是否足够作为 C3 的正文证据，还是还需要 PLDM 同款 basin replication？
- Phase 0 ACPC artifact 是否足以作为 Appendix H exploratory evidence，还是必须补 Phase 1 training objective？
- target-view negative result 是否需要后续 rollout-consistent denoising / multi-step objective 复核，才能从 scope boundary 升级为 causal method claim？
- candidate action sequence 来源如何固定（CEM 采样 / 固定 random / dataset actions），保证 clean/noisy 同一 candidate set？
- Π（task-relevant predictive readout）与 Ψ（discriminability readout）具体取什么？full latent / transition delta / cost feature / learned projection？
- 是否补 Phase 1 方法实验后转成 method paper，还是先以 reframing + diagnostic 形态投出？
- 新增 related work 边界（Bisim-JEPA / ViGMO / value equivalence / LeJEPA theory）是否足够区分，避免被 reviewer 当作 condition stacking？

---

## 7. 研究路线图（多篇 paper 视角）

### 7.1 Paper 1 v0 — 当前状态

已重构为 **action-conditioned predictive consistency** 的 reframing + diagnostic paper。实验证据包括 corruption cliff、noise sweep、dense Gaussian-noise ACPC basin、PLDM、blur eval-only sanity、partial-corr null、hetero 负结果、completed target-view negative ablation；Phase 0 ACPC artifact 已完成并作为 exploratory Appendix H 纳入正文。当前最小可立项版本不再需要“Phase 0 尚未完成”作为阻塞项；但若要把 PCC/CRA/MAF/ADM/SPRR 也写成主文证据，仍需额外扩展和 claim audit。提交前仍需 references 人工核对（§5）；build/checker 当前已通过。注意 §7.3.b（adaptive resolution / per-token consistency）已与本文 motivate 的 APDC 高度重合——若补 Phase 1 方法实验，应明确 Paper 1 与 Paper 2b 的边界。

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
