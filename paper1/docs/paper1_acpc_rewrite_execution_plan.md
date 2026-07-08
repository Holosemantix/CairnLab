# Paper 1 新路线执行文档：从视觉不变性诊断转向 Action-Conditioned Predictive Consistency

> 用途：给 Codex / Claude Code / 人工改稿使用。  
> 当前目标：先重构 Paper 1 的故事线、claim、related work 边界和实验/诊断路线；暂不在本文档中最终定具体新方法实现。  
> 核心修正：不要再把论文建立在“JEPA + rollout + CEM planning”这种堆条件式表述上，也不要把中心概念叫成含混的“planning equivalence”。新的主轴应落在世界模型本身的**动作条件预测动力学**上。

---

## 0. 不可违反的改稿原则

### 0.1 禁止堆条件式 claim

禁用以下风格：

> “our problem is latent predictive world model + rollout + CEM planning”  
> “JEPA + CEM closed-loop planning is the novelty”

问题在于：latent predictive model、rollout、CEM 并不在同一抽象维度上。latent predictive model 是模型类别；rollout 是使用 dynamics model 的计算过程；CEM 是一种 action optimization / MPC solver。把三者堆在一起说成“我们的不同点”，会显得人为缩小问题设定。

建议全篇统一为：

- **visual world models for control**
- **latent predictive world-model control**
- **JEPA world-model control under standard evaluation**
- **action-conditioned predictive dynamics under visual perturbations**

CEM 只在 setup / evaluation / implementation 中出现：

> We follow the standard evaluation protocol of the baseline, which uses CEM for action optimization.

不要把 CEM 写成 thesis 的组成部分。

### 0.2 不把中心概念叫成 “planning equivalence”

“规划等价性”不够自明，容易被问：planning 和 prediction 的边界是什么？世界模型实际学的是从历史状态和动作预测未来状态，planning 只是用这些预测来选择动作。因此主概念应放在 **predictive dynamics** 层，而不是 planner 层。

推荐中心术语：

> **Action-Conditioned Predictive Consistency under Visual Perturbations**

可简称 **ACPC**，但正文第一次出现时不要急于用缩写。中文可写为：

> **视觉扰动下的动作条件预测一致性**

一句话定义：

> 对同一真实状态的 clean / corrupted observation，encoder 输出可以不同；但在同一历史、目标和动作干预下，世界模型预测的下一状态或多步未来应在任务相关动力学上保持一致。真正会改变 action-conditioned transition、cost 或 optimal behavior 的状态差异必须保留。

### 0.3 planning / cost / action 是下游读数，不是中心定义

可以保留 cost ranking、elite set、action choice、regret 等指标，但它们应作为**预测一致性的下游控制读数**，而不是论文的核心概念。

推荐表述：

> Planning metrics are downstream readouts of predictive consistency. They test whether prediction mismatch matters for control, but the object we want to regularize is the action-conditioned predictive dynamics.

### 0.4 不要再把 clean/corrupted latent close 当作鲁棒性定义

禁止把视觉鲁棒性定义成：

\[
\|z(o)-z(\tilde{o})\| \to 0
\]

新的定义允许：

\[
z(o) \neq z(\tilde{o})
\]

但要求：

\[
F_\theta^H(z(o), a_{0:H-1}) \approx F_\theta^H(z(\tilde{o}), a_{0:H-1})
\]

其中近似只应发生在任务/动力学相关的预测空间，而不是完整 latent 的每一维。

---

## 1. 新中心 thesis

### 1.1 推荐英文 thesis

> Visual robustness for world-model control should be defined at the level of action-conditioned predictive dynamics, not encoder-level latent invariance. Clean and corrupted views of the same underlying state may encode differently, but under the same history and action intervention they should induce consistent next-state and rollout predictions in task-relevant coordinates. Conversely, state differences that change action-conditioned transitions, costs, or optimal behavior must remain distinguishable.

### 1.2 推荐中文 thesis

> 世界模型控制中的视觉鲁棒性不应定义为 clean/corrupted 图像的 latent 越接近越好，而应定义在动作条件预测动力学层面：同一真实状态的 clean/noisy observation 可以 encode 成不同 latent，但在同一历史和动作干预下，应预测到一致的任务相关未来；而真正改变 transition、cost 或最优行为的状态差异必须被保留。

### 1.3 一句话 pitch

> Robust visual world models need **selective predictive consistency**: contract nuisance perturbations only after they pass through action-conditioned dynamics, while preserving the distinctions that change future transitions.

### 1.4 题目候选

优先候选：

1. **Action-Conditioned Predictive Consistency for Robust Visual World Models**
2. **Robust Visual World Models Need Predictive Consistency, Not Latent Invariance**
3. **Diagnosing Predictive Inconsistency under Visual Perturbations in JEPA World Models**
4. **Selective Predictive Consistency in Latent World Models for Visual Control**

不推荐：

- “Planning-Equivalent Visual Robustness ...” —— planning / prediction 边界含混。
- “JEPA + CEM ...” —— 堆条件，不是有价值的差异点。
- “Invariance–Resolution Trade-off ...” —— resolution 不自明，且当前故事线不够强。

---

## 2. 新故事线：7 步逻辑链

### Step 1 — 先承认 JEPA 理论优势，但限定其作用域

JEPA / joint-embedding 的优势不是“天然视觉鲁棒”，而是：相比 reconstruction，它减少了保留 input-level irrelevant variance 的压力。`Joint-Embedding vs Reconstruction` 给出的理论支点是：SSL 需要 augmentation 与 irrelevant features 对齐；在高幅无关噪声时，joint-embedding 的 alignment requirement 比 reconstruction 更弱，但仍然需要知道或逼近“什么是 irrelevant”。

对我们的启发：

> 在控制中，irrelevance 不是 image-level 或 label-level，而是 action / transition / cost dependent。

因此，不能只说“latent prediction 不重建像素，所以会 robust”。应改成：

> Latent prediction reduces one source of pressure to reconstruct nuisance pixels, but it does not identify which visual variations are irrelevant for action-conditioned dynamics.

### Step 2 — 吸收 LeJEPA world model 理论，但不要撞车

`When Does LeJEPA Learn a World Model?` 的关键价值是理论支撑：LeJEPA 在 Gaussian latent world + stationary additive-noise transitions 等假设下可线性恢复真实 latent variables，并且 linear/orthogonal identifiability 能支持 optimal latent-space planning。

我们的差异不是“我们也证明 planning equivalence”。不要这么写。

应写成：

> Recent LeJEPA theory studies when the learned representation recovers the latent state structure of the world and thereby supports latent-space planning. Our question is complementary: given a learned visual world model, when do visually perturbed observations of the same underlying state induce consistent action-conditioned predictions, and when do perturbations push the model into different predictive neighborhoods?

尤其注意该论文自己也指出 action-conditioned transition 仍是需要学习和扩展的方向；我们正好把问题落在 corrupted observation 下的 action-conditioned predictive dynamics。

### Step 3 — 重新定义视觉鲁棒性：不是 latent invariance，而是动作条件预测一致性

旧代理：

\[
d(z_t, \tilde{z}_t) \text{ small}
\]

新代理：

\[
d_\mathrm{pred}\left(F_\theta^k(z_t, a_{t:t+k-1}), F_\theta^k(\tilde{z}_t, a_{t:t+k-1})\right) \text{ small}, \quad k=1,\dots,H
\]

这里：

- \(z_t = E(h_t)\) 是 clean 历史的 latent；
- \(\tilde z_t = E(\tilde h_t)\) 是 corrupted 历史的 latent；
- \(a_{t:t+k-1}\) 是同一条动作序列；
- \(d_\mathrm{pred}\) 不一定是完整 latent L2，可以是 target latent、transition delta、cost-relevant projection、goal distance 或其他任务相关读数。

更直观的图示：

```text
clean observation    -> encoder -> z_clean  --same action--> predicted future z'_clean
corrupted observation -> encoder -> z_noise  --same action--> predicted future z'_noise

允许: z_clean != z_noise
要求: task-relevant part of z'_clean ≈ task-relevant part of z'_noise
禁止: action-sensitive state differences 被压到同一个 future prediction
```

### Step 4 — 用现有实验作为现象锚点，而不是最终故事线

现有实验强支撑三件事：

1. 无噪声训练下，视觉 corruption 会造成明显 closed-loop failure；例如 PushT 从 86.33% unperturbed 掉到 4.67% px+goal std=0.08，TwoRoom 从 94.00% 掉到 50.00%。
2. Gaussian noise training 能恢复大量 performance，但固定 scalar `std_max` 是粗粒度压力，不是 principled solution。
3. 旧的 pointwise / single-step fragility 指标不足以解释 robustness gap；multi-step rollout drift 在部分任务上更接近失败机制。

写法重点：

> These results motivate a new diagnostic target rather than serving as the final contribution. The important failure is not that pixels are noisy, but that visual perturbations can move the model into different action-conditioned predictive neighborhoods.

### Step 5 — 把 partial-correlation 负结果变成新指标动机

当前 strongest fragility ratio 在控制 `std_max` 后不能预测 corruption gap；Reacher 的 multi-step rollout drift 反而保留 residual signal。这说明：

> encoder-level 或 single-step predictor shift 不是 control robustness 的充分定义；更接近问题本质的是 multi-step action-conditioned predictive consistency。

建议新增段落：

> The null result is not a weakness of the study; it shows that label-free pointwise fragility is the wrong abstraction for visual robustness. Robustness should be assessed after conditioning on the actions that the model is asked to imagine.

### Step 6 — 把 heteroscedastic σ-head 负结果放到核心位置

σ-head / heteroscedastic NLL 的负结果非常有价值：它学会了 prediction difficulty，但把 hard transitions downweight 后，PushT clean performance 崩掉。这说明：

> hard does not mean nuisance.

这应该成为新故事线的关键反例：contact-sensitive transitions 预测难，恰恰因为它们是 action-relevant；不能被 global compression、uncertainty downweighting 或 naive invariance pressure 丢掉。

推荐写法：

> A difficulty-aware objective is not enough. In contact-heavy control, the most prediction-difficult transitions can be exactly the transitions that determine the next action. This motivates sensitivity-aware predictive consistency rather than error-based downweighting.

### Step 7 — 方法引子：Adaptive Predictive-Dynamics Consistency，而不是再调 std

故事线最终落点：

> The next algorithmic step is to regularize clean/corrupted predictions under the same action intervention while gating the regularization by action/transition sensitivity.

方法方向暂定名：

- **Adaptive Predictive-Dynamics Consistency (APDC)**
- 或 **Selective Predictive Consistency (SPC)**

不要在当前文档中锁死最终方法名，后续方法讨论再定。

---

## 3. 形式化定义：ACPC 与 discriminability

### 3.1 基本对象

令：

- \(o_t\)：clean observation；
- \(\tilde{o}_t = \tau(o_t)\)：同一真实状态下的 corrupted observation；
- \(h_t\)：历史 observation/action；
- \(E_\theta\)：encoder；
- \(F_\theta\)：action-conditioned latent dynamics predictor；
- \(\mathbf a = a_{t:t+H-1}\)：固定动作序列；
- \(G\)：目标或 cost 读数；
- \(\Pi\)：任务相关预测读数，可以是完整 latent、transition delta、goal-distance features、cost features 或 learned projection。

定义：

\[
z_t = E_\theta(h_t), \quad \tilde z_t = E_\theta(\tilde h_t)
\]

\[
\hat z_{t+k} = F_\theta^k(z_t, \mathbf a_{0:k-1}), \quad
\hat{\tilde z}_{t+k} = F_\theta^k(\tilde z_t, \mathbf a_{0:k-1})
\]

### 3.2 动作条件预测一致性

\[
\mathrm{ACPC}_H(o_t, \tilde o_t, \mathbf a)
=
\sum_{k=1}^{H} \alpha_k \cdot
 d\left(\Pi(\hat z_{t+k}), \Pi(\hat{\tilde z}_{t+k})\right)
\]

鲁棒性不是要求 \(d(z_t,\tilde z_t)\) 小，而是要求 \(\mathrm{ACPC}_H\) 小。

### 3.3 动作相关可辨性

对两个不同真实状态 \(s_i, s_j\)，如果存在动作序列 \(\mathbf a\) 使它们产生不同未来：

\[
\Delta^H_\mathrm{dyn}(s_i, s_j, \mathbf a)
=
 d\left(\Pi(F^H(z_i,\mathbf a)), \Pi(F^H(z_j,\mathbf a))\right)
> m
\]

则模型表示和预测不应把它们 collapse：

\[
d\left(\Psi(z_i), \Psi(z_j)\right) > m'
\]

其中 \(\Psi\) 可以是 latent、predicted delta、inverse-dynamics feature、cost feature 或 rollout embedding。

### 3.4 关键张力

真正的核心不是 invariance vs resolution，而是：

```text
same-state visual perturbation pairs       -> predictive consistency should be high
state/action/transition-distinct pairs     -> predictive discriminability should be high
```

这比“latent close / latent far”更准确，因为它把区分标准放在 action-conditioned dynamics 上。

---

## 4. 新诊断指标：从 pointwise fragility 到 predictive consistency

> 目标：先用现有 checkpoints 做 Phase 0 诊断，不需要立刻重训。

### 4.1 Encoder Perturbation Difference，EPD

保留旧 encoder shift，但降级为 descriptive metric：

\[
\mathrm{EPD}=d(E(o),E(\tilde o))
\]

解释：EPD 大不一定坏，EPD 小也不一定好。它只说明 perturbation 是否进入 latent。

### 4.2 One-step Action-Conditioned Predictive Consistency，ACPC-1

\[
\mathrm{ACPC}_1 = d\left(F(E(o), a), F(E(\tilde o), a)\right)
\]

建议归一化：

\[
\mathrm{nACPC}_1 =
\frac{d(F(E(o),a),F(E(\tilde o),a))}
{\mathrm{median}\ d(F(E(o_i),a_i),F(E(o_j),a_j)) + \epsilon}
\]

或用 natural transition magnitude 归一化：

\[
\frac{d(\hat z'_{clean}, \hat z'_{noise})}{d(z_t, z_{t+1})+\epsilon}
\]

### 4.3 Multi-step ACPC-H

\[
\mathrm{ACPC}_H = \sum_{k=1}^{H} \alpha_k d(\Pi(\hat z_{t+k}), \Pi(\hat{\tilde z}_{t+k}))
\]

这是现有 `predictor_rollout_T8_l2` 的更原则化版本，但必须明确它是**同一 action sequence 下 clean/noisy rollout 的一致性**，不是泛泛的 rollout drift。

### 4.4 Predictive Cost Consistency，PCC

如果已有 cost / goal 读数：

\[
\mathrm{PCC} = |J(F^H(E(o),\mathbf a), g)-J(F^H(E(\tilde o),\mathbf a), g)|
\]

PCC 是下游指标，不是主定义。它用于验证 ACPC 是否真的影响 control。

### 4.5 Candidate Ranking Agreement，CRA

对同一候选动作集 \(\mathcal A=\{\mathbf a^1,\dots,\mathbf a^K\}\)，计算 clean/corrupted cost vectors 的 Spearman / Kendall / pairwise ranking agreement。

使用方式：

- 作为 downstream readout；
- 不叫 planning equivalence；
- 不作为唯一 robustness metric；
- 需要 margin conditioning。

### 4.6 Margin-conditioned Action Flip，MAF

如果 clean top-1 和 top-2 cost margin 很小，action flip 不一定是 failure。建议只统计大 margin 下的 flip：

\[
\mathrm{MAF}=\mathbb{1}\left[u^*_{clean}\neq u^*_{noise},\ C_{(2)}-C_{(1)}>\delta\right]
\]

### 4.7 Action-Relevant Discriminability Margin，ADM

构造 action/transition distinct pairs：

- inverse dynamics label 不同；
- clean latent transition magnitude 高；
- contact / keyframe transition；
- oracle 或 dataset action 差异大；
- cost-to-go / goal-distance change 大。

指标：

\[
\mathrm{ADM}=\mathrm{median}_{(i,j)\in P_\mathrm{disc}} d(\Psi_i,\Psi_j)
\]

对方法评估，要求 ACPC 降低时 ADM 不明显下降。

### 4.8 Selective Predictive Robustness Ratio，SPRR

一个总括性诊断：

\[
\mathrm{SPRR}=\frac{\mathrm{median}_{(i,j)\in P_\mathrm{disc}} d(\Psi_i,\Psi_j)}
{\mathrm{median}_{(o,\tilde o,\mathbf a)\in P_\mathrm{pert}} \mathrm{ACPC}_H(o,\tilde o,\mathbf a)+\epsilon}
\]

但不要过度 claim 它能预测 success。它是 diagnostic summary，不是 oracle。

---

## 5. 相关工作吸收与差异边界

### 5.1 Joint-Embedding vs Reconstruction

核心结论：

- reconstruction 和 joint-embedding 都依赖 augmentation 与 irrelevant features 的 alignment；
- SSL 不能只靠样本数自动克服 augmentation-noise misalignment；
- 高幅 irrelevant noise 下，joint-embedding alignment requirement 更弱；低幅 irrelevant noise 且缺少有效 augmentation prior 时，reconstruction 可能更稳。

对我们的作用：理论背景。

不能写成：

> joint-embedding is robust, so JEPA should be robust.

应写成：

> Joint-embedding reduces the burden of reconstructing irrelevant high-magnitude features, but it still needs augmentation or inductive bias to identify which variations are irrelevant. In control, irrelevance is action-conditioned rather than purely visual.

### 5.2 When Does LeJEPA Learn a World Model?

核心结论：

- LeJEPA = alignment + Gaussian regularization；
- 在 Gaussian latent world 和 stationary additive-noise transitions 等假设下，LeJEPA 可线性恢复世界 latent variables；
- linear / orthogonal identifiability 支持 latent-space planning；
- 该理论主要解决 state-side latent identifiability；action-conditioned transition 的扩展仍是自然方向。

对我们的作用：理论支撑与 scope boundary。

不能写成：

> we propose planning equivalence after LeJEPA.

应写成：

> LeJEPA theory explains when a representation can recover latent state structure. We study a complementary robustness question: whether clean and visually corrupted observations of the same state induce consistent action-conditioned predictions in a learned visual world model.

### 5.3 Bisim-JEPA / Learning Invariant Visual Representations for Planning with JEPA World Models

核心结论：

- 该工作把 bisimulation encoder 加入 JEPA-style visual world models；
- 目标是 control-relevant state equivalence；
- 主要处理 slow features，如 background changes / distractors；
- 使 transition-dynamics 相似的 states 在 latent 中接近，并减少 slow feature 影响。

这是最强 collision risk 之一。

我们的差异不能写成“我们用了 CEM / Gaussian noise / 四个任务”这种低价值差异。真正差异应是：

1. **paired clean-corrupted predictive consistency**：我们关注同一 underlying state 的 clean / corrupted observation 是否在同一动作干预下产生一致预测；
2. **encoder 不必 collapse**：我们允许 corrupted latent 与 clean latent 不同，只要求 action-conditioned future prediction 在任务相关空间一致；
3. **discriminability guard**：我们显式保护 action-sensitive / transition-sensitive pairs，避免把 hard contact transitions 当成 nuisance；
4. **diagnostic-to-objective route**：先证明旧 pointwise diagnostics 不够，再提出 ACPC / ADM 这种预测动力学指标；
5. **claim 不是 invariant representation alone**：我们的中心是 predictive dynamics consistency，不是只学一个更 invariant 的 state embedding。

如果最后方法只做普通 latent consistency，那么会与 Bisim-JEPA / ViGMO 撞得很严重。必须加入 action-conditioned candidate sequence、sensitivity gating 和 discriminability preservation。

### 5.4 ViGMO

核心结论：

- visual MBRL 在 unseen distractions 下会 degrade；
- ViGMO 使用 mixed weak-to-strong augmentation、latent-consistency learning 和 encoder regularization；
- latent consistency 用来稳定 distribution shift 下的 transition predictions。

这是第二个强 collision risk。

我们的差异边界：

- 不能只说“他们不是 JEPA”。这是弱差异。
- 真正差异应是：我们的 consistency 是 **action-conditioned and discriminability-gated**，评估 clean/corrupted pair 在同一动作序列下的 one-step / multi-step predictive consistency，并同时约束 action-relevant transition 不被 collapse。
- 如果加入 cost/ranking 指标，也应写成下游 readout，而不是“我们多了 CEM”。

### 5.5 Value Equivalence / Value-Aware Model Learning

核心结论：

- 模型不一定要完整拟合 state-to-state dynamics；它应服务于 value / planning consequence；
- value-equivalent models 对一组 policy/function 产生相同 Bellman updates；
- value-aware model learning 主张模型 loss 应考虑 downstream value estimation。

对我们的作用：理论背景。

差异：

- 它们主要讨论 model learning 的 value/planning objective；
- 我们讨论 visual perturbation pair 下的 action-conditioned prediction consistency；
- 我们不把全部问题降成 value equivalence，因为世界模型仍需保留 contact-sensitive predictive details。

推荐相关工作写法：

> Value-equivalent and value-aware model learning motivate evaluating models by their downstream use rather than raw reconstruction. Our setting adds a visual robustness constraint: two observations of the same state should remain equivalent after action-conditioned prediction, while genuinely different transitions must remain distinguishable.

### 5.6 Bisimulation / DeepMDP / DBC / BS-MPC

核心结论：

- bisimulation 提供 state abstraction：reward 和 transition behavior 相同的 states 可以合并；
- DBC 学习对 distractors 更鲁棒的控制表示；
- BS-MPC 把 bisimulation metric 引入 MPC encoder，提高稳定性和 noise robustness。

对我们的作用：机制基础与 related work。

差异：

- bisimulation 是 state equivalence / abstraction；
- ACPC 是 paired visual perturbation 下的 action-conditioned predictive consistency；
- 可以借用 bisimulation 思想定义 action-relevant discriminability，但不应把方法写成“又一个 bisimulation encoder”。

### 5.7 VIBR

核心结论：

- model-free visual control 中，不一定要 representation invariant；value function invariant 可能更合适；
- 这支持“robustness target 不应停在 representation close”。

差异：

- VIBR 是 value-function / model-free setting；
- 我们是 visual world model 的 action-conditioned predictive dynamics；
- 不要把差异写成 solver 或 CEM，而是写成预测动力学 vs value function。

### 5.8 stable-worldmodel

影响：

- visual/physical controllable variations 已经成为 world model evaluation ecosystem 的一部分；
- 因此 Paper 1 不能以 visual perturbation benchmark 新颖性立项；
- 但可以用它支持我们的定位：视觉扰动是 controlled probe，用来诊断并设计 robust visual world models。

---

## 6. 新 contribution 写法

### C1. Problem reframing

> We argue that visual robustness for world-model control should be formulated as action-conditioned predictive consistency rather than encoder-level latent invariance.

### C2. Diagnostic evidence

> Using existing Gaussian corruption experiments, we show that visual perturbations cause closed-loop failure and that pointwise fragility metrics are insufficient; multi-step predictive drift provides a more relevant diagnostic signal in some tasks.

### C3. Selective consistency diagnostics

> We introduce diagnostics that compare clean/corrupted predictions under the same action sequence while separately measuring action-relevant discriminability.

### C4. Method-design implication

> The diagnostics motivate adaptive predictive-dynamics consistency: enforce consistency for same-state perturbations but preserve distinctions for action-sensitive transitions.

如果还没有新方法实验，C4 只能写成 design implication / future direction。若补方法实验，C4 可以升级成 method contribution。

---

## 7. main.tex / PLAN.md 大规模修改指令

### 7.1 全局替换策略

#### 应弱化或删除

- “invariance-resolution trade-off” 作为 title / thesis；保留时只作为早期 intuition。
- “no single jointly optimal std_max” 的强 claim。
- “diagnostics predict robustness”。
- “JEPA + CEM” 作为贡献名。
- “planning equivalence” 作为中心概念。

#### 应新增或强化

- action-conditioned predictive consistency；
- visual perturbation as paired probe；
- encoder difference allowed, prediction consistency required；
- hard transitions can be important；
- downstream planning/cost/action metrics are readouts；
- related work boundaries with Joint-vs-Reconstruction and LeJEPA theory。

### 7.2 Title

替换为：

```latex
\title{Action-Conditioned Predictive Consistency for Robust Visual World Models}
```

备选：

```latex
\title{Robust Visual World Models Need Predictive Consistency, Not Latent Invariance}
```

### 7.3 Abstract 草稿

```text
Latent predictive world models are often expected to abstract away nuisance visual detail because they predict future representations rather than reconstructing pixels. For control, however, visual robustness should not be defined as encoder-level invariance between clean and corrupted images. A corrupted observation may legitimately map to a different latent code; what matters is whether, under the same history and action intervention, the world model predicts the same task-relevant future. We formulate this requirement as action-conditioned predictive consistency under visual perturbations, coupled with a discriminability constraint: state differences that change transitions, costs, or optimal behavior must remain separable.

We revisit Gaussian visual-corruption experiments on JEPA world-model control through this lens. Without noise-aware training, visual corruptions induce severe closed-loop failures, while noise augmentation recovers much of the performance but acts as a coarse global pressure rather than a mechanism-aware solution. Existing pointwise diagnostics are insufficient: after controlling for training noise, single-step fragility does not explain the corruption gap, whereas multi-step predictive drift carries residual signal in some tasks. A negative heteroscedastic reweighting result further shows that hard transitions are not necessarily nuisance transitions; in contact-heavy control, they may be exactly the action-relevant events that must be preserved.

These findings motivate a new diagnostic and method-design target: enforce clean/corrupted consistency after action-conditioned prediction, not necessarily before it, while preserving action-relevant discriminability. The resulting view connects SSL augmentation-alignment theory, LeJEPA identifiability theory, bisimulation-style state abstraction, and value-aware model learning, but differs by focusing on paired visual perturbations and action-conditioned predictive dynamics in learned visual world models.
```

### 7.4 Introduction 新结构

建议改成四小节：

1. **Latent prediction is not a robustness definition**
2. **From visual invariance to action-conditioned predictive consistency**
3. **Existing corruption results as a diagnostic probe**
4. **Contributions**

第一节核心句：

```text
For a visual world model, the relevant question is not whether two observations have identical latents, but whether they induce consistent predictions under the same action intervention.
```

第二节加入 formal intuition：

```text
We therefore allow z_clean and z_corrupted to differ. The consistency target is placed after the action-conditioned predictor: F(z_clean, a) should agree with F(z_corrupted, a) in task-relevant coordinates.
```

### 7.5 Related Work 新结构

建议 related work 至少包含：

1. **Joint-embedding, reconstruction, and augmentation alignment**
   - cite Joint-Embedding vs Reconstruction。
   - 关键：augmentation must align with irrelevant features；control 中 irrelevant 是 action-conditioned。

2. **LeJEPA identifiability and latent world-model theory**
   - cite When Does LeJEPA Learn a World Model?。
   - 关键：state-side identifiability 支撑 latent planning；我们的 focus 是 corrupted observations 下 action-conditioned predictive consistency。

3. **Bisimulation and control-relevant representations**
   - cite DBC / DeepMDP / BS-MPC / Bisim-JEPA。
   - 关键：state equivalence foundation；我们的 focus 是 paired clean/corrupted prediction consistency + discriminability guard。

4. **Visual robustness in model-based RL**
   - cite ViGMO / stable-worldmodel。
   - 关键：不要 claim visual perturbation novelty；把我们定位成 diagnostic + predictive consistency framing。

5. **Value-aware and value-equivalent model learning**
   - cite Value Equivalence / VAML。
   - 关键：model accuracy should serve downstream use；我们把这个原则具体化到 visual perturbation pairs and predictive dynamics。

### 7.6 Diagnostic section 新增小节

新增：

```latex
\subsection{Action-conditioned predictive consistency diagnostics}
```

包括：

- ACPC-1；
- ACPC-H；
- PCC / cost consistency as downstream readout；
- CRA / ranking agreement as downstream readout；
- ADM / action-relevant discriminability；
- margin-conditioned action flip。

如果这些指标尚未算出，不能写成结果。写成 “we propose / future diagnostic target” 或 “Phase 0 TODO”。如果要进入 main contribution，必须补实验。

### 7.7 Discussion 新结构

新增或替换为：

```latex
\subsection{From latent invariance to predictive consistency}
```

要点：

- latent closeness is neither necessary nor sufficient；
- same-state corrupted observations should be predictive-consistent；
- action-sensitive state differences should remain discriminable；
- hard transitions should not be downweighted blindly。

```latex
\subsection{Implications for adaptive predictive-dynamics objectives}
```

要点：

- fixed Gaussian augmentation = coarse pressure；
- need action/transition sensitivity gating；
- method should regularize predictions, not merely encoder outputs。

---

## 8. Phase 0 实验：先证明新指标解释力

### 8.1 目标

在不重训的情况下，用现有 checkpoints 重算 ACPC 系列指标，验证它们是否比旧 fragility ratio 更贴近 corrupted success / corruption gap。

### 8.2 数据

- 已有 LeWM checkpoints：4 tasks × base + 8 noise levels。
- 已有 PLDM replication 如可访问则加入。
- evaluation states / goals / histories 与当前 protocol 对齐。

### 8.3 每个 checkpoint 要计算

1. `encoder_shift_clean_noise`：旧指标，保留为对照。
2. `acpc_1_l2`：one-step same-action predictive distance。
3. `acpc_h_l2`：multi-step same-action predictive distance，H 可取 4/8。
4. `acpc_h_norm_by_transition`：按 natural transition magnitude 归一化。
5. `cost_consistency_abs`：同一 action sequence 的 clean/noisy cost 差。
6. `candidate_ranking_spearman`：同一 candidate action set 的 ranking agreement。
7. `elite_overlap`：top-K candidate overlap。
8. `margin_conditioned_flip_rate`：大 margin 下的 action flip。
9. `action_discriminability_margin`：action/transition distinct pairs 的 margin。
10. `sprr`：selective predictive robustness ratio。

### 8.4 候选动作序列来源

优先顺序：

1. 使用 evaluation 中 CEM 采样的 candidate sequences，如日志可取。
2. 使用固定 random action sequences，保证 clean/noisy 完全相同 candidate set。
3. 使用 dataset actions / replay buffer actions。
4. 对 PushT 额外采样 contact-sensitive local actions。

注意：candidate sequence 来源必须在 clean/noisy 间固定，否则无法归因。

### 8.5 分析

与旧 Section 4.5 对齐：

- Spearman / partial Spearman vs unperturbed success；
- Spearman / partial Spearman vs corrupted success；
- Spearman / partial Spearman vs corruption drop；
- 控制 `std_max`；
- 对 PLDM 加 method dummy；
- bootstrap CI。

### 8.6 成功标准

新指标至少满足其中两项：

- ACPC-H 或 cost consistency 在 partial correlation 上比 old fragility ratio 更稳定；
- margin-conditioned flip 能解释 PushT catastrophic failures；
- Reacher 的 rollout drift residual signal 可被 ACPC-H 更清楚地复现；
- TwoRoom 的高鲁棒性对应 low ACPC-H + sufficient topology discriminability；
- heteroscedastic failure 对应 ADM collapse 或 action-sensitive transition collapse。

### 8.7 失败判据

如果 ACPC-H / PCC / ranking metrics 都不能比旧指标解释更多，那新故事线仍可作为 theoretical reframing，但不够支撑方法 paper。需要转向：

- 更好的 action-sensitive pair construction；
- oracle state / keypoint / simulator state metrics；
- cost function mismatch 分析；
- dataset / planner candidate distribution 问题。

---

## 9. Phase 1 方法占位：Adaptive Predictive-Dynamics Consistency

> 后续再细化。当前只给最小 skeleton，避免 Codex 把它写成已完成结果。

### 9.1 核心假设

> If visual perturbations are nuisance for a state-action transition, clean/corrupted branches should produce consistent action-conditioned predictions. If a transition is action-sensitive, the objective should preserve discriminability rather than enforce unconditional invariance.

### 9.2 Loss skeleton

\[
\mathcal L = \mathcal L_\mathrm{base}
+ \lambda_p \mathcal L_\mathrm{ACPC}
+ \lambda_c \mathcal L_\mathrm{cost}
+ \lambda_d \mathcal L_\mathrm{disc}
\]

其中：

\[
\mathcal L_\mathrm{ACPC}=\sum_{k=1}^{H} w_\mathrm{inv}(t,k)\cdot d(\Pi(\hat z_{t+k}),\Pi(\hat{\tilde z}_{t+k}))
\]

\[
\mathcal L_\mathrm{disc}=w_\mathrm{disc}(i,j)\cdot [m-d(\Psi_i,\Psi_j)]_+
\]

### 9.3 gating 信号

候选：

- clean latent transition magnitude；
- inverse-dynamics confidence / error；
- action difference；
- local action variance；
- contact detector / keyframe heuristic；
- cost gradient or goal-distance change；
- rollout disagreement under action perturbation。

原则：

- low action sensitivity -> stronger consistency；
- high action sensitivity -> weaker invariance, stronger discriminability guard；
- high prediction difficulty alone 不应 downweight。

### 9.4 Baseline / ablation

最小 ablation：

| Variant | 作用 |
|---|---|
| base no-noise | 原始 failure |
| fixed Gaussian augmentation | 现有强 baseline |
| fixed encoder consistency | 测 latent closeness 是否足够 |
| fixed predictor consistency | 测 ACPC 不带 gating 是否有效 |
| adaptive predictor consistency | 核心 consistency |
| adaptive predictor consistency + discriminability | 完整方法 |
| without discriminability | 验证是否伤 PushT/contact |
| shuffled gate | 验证 gate 不是噪声 |
| random gate | 验证 action-sensitive signal 的必要性 |
| difficulty-only gate | 验证 hard != nuisance |

### 9.5 成功标准

- PushT px+goal 0.08 提升，同时 clean 不明显掉；
- TwoRoom high-noise robustness 不低于 fixed Gaussian；
- Reacher/Cube 不退化；
- ACPC-H 下降；
- ADM 不下降或下降小于固定 consistency；
- 不出现 heteroscedastic σ-head 那种 contact collapse。

---

## 10. References.bib 待补条目草案

> 下列条目为改稿辅助草案，最终提交前必须人工核验 venue、author spelling、version 和 DOI。

```bibtex
@misc{vanassel2025jointembeddingreconstruction,
  title         = {Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning},
  author        = {Van Assel, Hugues and Ibrahim, Mark and Biancalani, Tommaso and Regev, Aviv and Balestriero, Randall},
  year          = {2025},
  eprint        = {2505.12477},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2505.12477}
}

@misc{klindt2026lejepaworldmodel,
  title         = {When Does LeJEPA Learn a World Model?},
  author        = {Klindt, David and LeCun, Yann and Balestriero, Randall},
  year          = {2026},
  eprint        = {2605.26379},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@misc{toso2026invariantvisualplanningjepa,
  title         = {Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models},
  author        = {Toso, Leonardo F. and Shadunts, Davit and Lu, Yunyang and Sharma, Nihal and Zhan, Donglin and Nguyen, Nam H. and Anderson, James},
  year          = {2026},
  eprint        = {2602.18639},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2602.18639}
}

@misc{maes2026stableworldmodel,
  title         = {stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation},
  author        = {Maes, Lucas and Le Lidec, Quentin and Haramati, Dan and Massaudi, Nassim and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  year          = {2026},
  eprint        = {2602.08968},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {ICLR 2026 Workshop on World Models}
}

@misc{park2026vigmo,
  title         = {Zero-Shot Visual Generalization in Model-Based Reinforcement Learning via Latent Consistency},
  author        = {Park, Mingyu and Noh, Samyeul and Myung, Hyun and Lee, Donghwan},
  year          = {2026},
  note          = {OpenReview, submitted to ICLR 2026. Verify final status before camera-ready.}
}

@inproceedings{dupuis2023vibr,
  title     = {VIBR: Learning View-Invariant Value Functions for Robust Visual Control},
  author    = {Dupuis, Tom and Rabarisoa, Jaonary and Pham, Quoc-Cuong and Filliat, David},
  booktitle = {Proceedings of The 2nd Conference on Lifelong Learning Agents},
  pages     = {658--682},
  year      = {2023},
  volume    = {232},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

@inproceedings{grimm2020valueequivalence,
  title     = {The Value Equivalence Principle for Model-Based Reinforcement Learning},
  author    = {Grimm, Christopher and Barreto, Andre and Singh, Satinder and Silver, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@misc{voelcker2025calibratedvalueaware,
  title         = {Calibrated Value-Aware Model Learning with Stochastic Environment Models},
  author        = {Voelcker, Claas and Pedan, Anastasiia and Ahmadian, Arash and Abachi, Romina and Gilitschenski, Igor and Farahmand, Amir-massoud},
  year          = {2025},
  eprint        = {2505.22772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## 11. Codex / Claude Code 具体执行 prompt

把下面这段直接给代码助手：

```text
You are editing Paper 1. Do not change experimental numbers, tables, artifacts, or claims unless explicitly instructed. Your job is to rewrite the framing from “invariance-resolution trade-off / planning equivalence / JEPA+CEM failure” to “action-conditioned predictive consistency under visual perturbations”.

Hard constraints:
1. Do not use “JEPA + rollout + CEM planning” as the novelty claim. CEM is only an evaluation/action-optimization implementation detail.
2. Do not make “planning equivalence” the central concept. The central concept is action-conditioned predictive consistency: clean/corrupted observations of the same underlying state may encode differently, but under the same history and action sequence they should induce consistent task-relevant next-state/rollout predictions.
3. Do not define robustness as z_clean ≈ z_corrupted. Encoder-level latent closeness is neither necessary nor sufficient.
4. Preserve the countercondition: states/transitions that change action-conditioned dynamics, cost, or optimal behavior must remain distinguishable.
5. Do not claim visual perturbation stress testing is new. Present it as a controlled probe.
6. Do not claim a strict “no universal std_max” theorem. Use “coarse global scalar pressure / broad task-dependent plateaus”.
7. Do not claim diagnostics predict robustness. They localize mechanisms and motivate method targets.
8. Integrate related work: Joint-Embedding vs Reconstruction, When Does LeJEPA Learn a World Model?, Bisim-JEPA / invariant visual representations for JEPA planning, ViGMO, value equivalence / value-aware model learning, bisimulation/DBC/DeepMDP/BS-MPC, VIBR, stable-worldmodel.

Files to edit:
- PLAN.md: rewrite one-minute summary, six-step story, contributions, roadmap.
- paper1/main.tex or main.tex: rewrite title, abstract, intro, related work, diagnostic framing, discussion, limitations/future work.
- references.bib: add missing references with TODO verify markers where venue/status is uncertain.

Expected new title candidate:
Action-Conditioned Predictive Consistency for Robust Visual World Models

Expected new thesis:
Visual robustness for world-model control should be defined at the level of action-conditioned predictive dynamics, not encoder-level latent invariance. Clean and corrupted views of the same underlying state may encode differently, but under the same history and action intervention they should induce consistent next-state and rollout predictions in task-relevant coordinates. Conversely, state differences that change action-conditioned transitions, costs, or optimal behavior must remain distinguishable.

Add a new conceptual/diagnostic section defining:
- ACPC-1: one-step action-conditioned predictive consistency.
- ACPC-H: multi-step predictive consistency under the same action sequence.
- PCC / cost consistency as downstream readout.
- Candidate ranking agreement and margin-conditioned action flip as downstream control readouts, not central definitions.
- ADM: action-relevant discriminability margin.
- SPRR: ratio of action-distinct margin to clean/corrupted predictive inconsistency.

If the metrics have not been computed, mark them explicitly as proposed diagnostics / Phase 0 TODO, not results.

Rewrite related work to make true differences, not condition stacking:
- Compared with Bisim-JEPA: not simply another invariant representation; allow clean/noisy latents to differ and regularize action-conditioned predictions while preserving sensitive transitions.
- Compared with ViGMO: not generic latent consistency; require action-conditioned clean/corrupted rollout consistency plus discriminability guard.
- Compared with value equivalence: use downstream consequence principle, but focus on paired visual perturbations and predictive dynamics.
- Compared with LeJEPA theory: complementary robustness question under corrupted observations; do not claim to prove LeJEPA identifiability.

Run consistency checks after edits and leave TODOs for uncomputed metrics.
```

---

## 12. 改稿后的 submit-readiness 判断

当前 Paper 1 不建议按旧版本直接挂出。建议状态改成：

> not submit-ready; under reframing toward predictive-consistency diagnostics and possibly a lightweight method.

最小可立项版本需要至少完成 Phase 0：

- 用现有 checkpoints 计算 ACPC-H / PCC / ranking / ADM；
- 证明新指标确实比 old fragility ratio 更贴近 closed-loop failure 或至少解释 heteroscedastic negative result；
- related work 中明确 ViGMO / Bisim-JEPA / LeJEPA theory 的边界。

如果 Phase 0 成立，再进入 Phase 1 方法实验。

---

## 13. Coverage / Novelty / Execution Audit

### Coverage

已覆盖 direct work：JEPA world models、LeWM/PLDM、Bisim-JEPA、ViGMO、stable-worldmodel。  
已覆盖 theory support：Joint-Embedding vs Reconstruction、When Does LeJEPA Learn a World Model、value equivalence / value-aware model learning、bisimulation。  
仍需补查：2026 年 ICLR/ICML/OpenReview 是否有新的 “predictive consistency / latent consistency / robust world model” concurrent work。

### Novelty risk

最大风险：

1. 如果方法只做 latent consistency，会被 ViGMO 覆盖；
2. 如果方法只做 invariant representation，会被 Bisim-JEPA / DBC / BS-MPC 覆盖；
3. 如果只讲 cost/action 等价，会被 value equivalence 视角覆盖；
4. 如果只讲 LeJEPA learns world model，会撞 LeJEPA theory。

真正可守的增量：

> paired clean/corrupted action-conditioned predictive consistency + discriminability guard + diagnostics showing why pointwise latent invariance is the wrong robustness target.

### Execution

最先做的 sanity check：

1. 从一个 PushT checkpoint 抽 100 states；
2. 对每个 state 构造 clean/noisy observation；
3. 固定 64 条 action sequences；
4. 计算 ACPC-H、PCC、ranking agreement、ADM；
5. 看 catastrophic px+goal failures 是否对应 high ACPC-H 或 high-margin action flip。

失败后定位：

- 如果 ACPC-H 高但 success 不掉：检查 metric projection / cost function；
- 如果 ACPC-H 低但 success 掉：检查 action optimizer / candidate distribution；
- 如果 ADM 下降但 ACPC-H 也下降：说明 consistency 可能通过 collapse 实现；需要 discriminability guard；
- 如果所有指标无信号：需要 oracle state/keypoint/contact probe，而不是 latent-only probe。
