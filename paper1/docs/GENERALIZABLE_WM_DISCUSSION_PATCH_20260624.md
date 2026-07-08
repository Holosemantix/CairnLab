# Paper 1 patch plan: from diagnostics to generalizable world models — 2026-06-24

目标：把论文的长期意义从“Gaussian noise robustness / noise augmentation”提升为“诊断牵引可泛化世界模型基础能力”，同时避免过 claim。本文不应声称已经实现类人泛化或上下文学习；它应声称：**视觉扰动实验提供了一个 controlled stressor，用来诊断世界模型的两个基础能力：control-relevant representation 和 action-conditioned predictive dynamics。**

这份 patch 给 Codex 执行。不要新增实验，不要伪造 context-learning 结果，不要写 human-like claims 为本文贡献。

---

## 1. 总体定位

当前 paper 的主线应是：

> Gaussian visual corruption is a controlled diagnostic probe. Noise training is a coarse intervention. The paper studies how the intervention changes the encoder--predictor--planner chain: encoder geometry, rollout geometry, candidate-cost/ranking stability, and discriminability. These diagnostics point to two foundation capabilities for more generalizable world models: control-relevant representation and action-conditioned predictive dynamics.

不要把文章写成：

> Noise hurts; Gaussian noise training helps.

也不要写成：

> We build human-like generalizable world models.

安全表述是：

> These diagnostics expose prerequisites for more generalizable world models.

---

## 2. Abstract 末尾替换

### 当前需要替换的方向

当前 abstract 末尾如果仍是类似：

```tex
The paper contributes a diagnostic framing, release artifacts, and negative ablations for this controlled setting.
```

或者只强调 artifacts，应替换为更强但安全的 multi-stage diagnostic account。

### 推荐替换文本

把 abstract 最后 1--2 句改为：

```tex
We use the corruption sweep as a stressor for representation analysis rather than as a hyperparameter search. Paired ACPC-basin probes localise the recovery in rollout space (PushT $R_F$: $1.543\to0.088$), while representation diagnostics check whether encoder contraction preserves action-relevant information. The result is a multi-stage diagnostic account: encoder geometry remains informative, but robustness for control depends on whether perturbation-induced latent changes survive action-conditioned rollout into candidate costs while action-relevant distinctions remain separable.
```

如果 abstract 太长，可压缩为：

```tex
These results position Gaussian visual corruption as a controlled probe for two broader world-model capabilities: control-relevant representation and action-conditioned predictive dynamics. Encoder geometry remains informative, but robustness for control depends on whether perturbation-induced latent changes survive action-conditioned rollout into candidate costs while action-relevant distinctions remain separable.
```

---

## 3. Introduction 贡献 C1/C2 微调

### C1 推荐版本

```tex
\paragraph{C1 --- Multi-stage diagnostic principle.}
We formulate visual robustness for JEPA world-model control as a multi-stage diagnostic rather than encoder invariance alone. Encoder perturbation geometry measures whether visual noise enters and reshapes the latent state; action-conditioned predictive consistency measures whether that shift changes predicted futures and candidate costs; the discriminability countercondition prevents contraction from erasing action-relevant distinctions.
```

### C2 推荐版本

```tex
\paragraph{C2 --- Planner link and controlled evidence.}
Under a Lipschitz cost readout, bounded rollout disagreement bounds candidate-cost drift on a fixed candidate set, and a sufficiently large clean action margin preserves the selected candidate within that set. We use a fixed Gaussian-noise protocol to probe this encoder--predictor--planner chain on LeWM, with PLDM as a second-family replication check.
```

### C3 推荐版本

```tex
\paragraph{C3 --- Diagnostic artifacts and negative checks.}
We release paired ACPC-basin artifacts, the full LeWM basin grid, cross-checkpoint diagnostic tables, PLDM replication, a target-view ablation, and a heteroscedastic-loss failure case. These artifacts localise where the intervention changes the model and mark where stronger readings fail.
```

---

## 4. ACPC / diagnostics section: ensure encoder is not sidelined

在 `Operational consistency diagnostics` 后加入：

```tex
Thus $R_E$ and $R_F$ answer different questions. $R_E$ measures whether the visual perturbation changes the encoded neighbourhood; $R_F$ measures whether that change survives action-conditioned rollout in coordinates used for planning. Robust checkpoints in this Gaussian-noise study often reduce both, so the claim is not predictor-only contraction. The claim is that encoder contraction must be interpreted through its downstream predictive effect and checked against discriminability.
```

这段是必须的。它防止读者误解为“encoder 表征不重要”。

---

## 5. Theory section: add encoder-to-rollout bridge

在 `\subsection{ACPC as a fixed-candidate stability condition}` 中，在 `ACPC controls candidate-cost drift` proposition 之前插入：

```tex
\begin{proposition}[Encoder shift as one route to rollout disagreement]\label{prop:encoder-route}
Fix an action sequence $\mathbf a$ and write
$G_{\mathbf a}(z)=\Pi(F_\theta^{1:H}(z,\mathbf a))$ for the rollout-readout map. If $G_{\mathbf a}$ is locally $L_G$-Lipschitz between $E_\theta(h)$ and $E_\theta(\tilde h)$ under metrics $d_Z$ and $d_H$, then
\[
 d_H\!\left(G_{\mathbf a}(E_\theta(h)),G_{\mathbf a}(E_\theta(\tilde h))\right)
 \le L_G\, d_Z\!\left(E_\theta(h),E_\theta(\tilde h)\right).
\]
\end{proposition}

\begin{proof}
This is the Lipschitz condition applied to the two encoded histories.
\end{proof}

This bound makes encoder geometry a first-stage risk signal. A small encoder shift is sufficient for small rollout disagreement only when the action-conditioned predictor is locally insensitive in that direction. It is not necessary, because a large encoder shift can lie in a nuisance direction contracted by $G_{\mathbf a}$. It is not sufficient without the local-sensitivity condition, because a small encoder shift can be amplified by the predictor or cost readout. This is why the empirical analysis reports both encoder radii $R_E$ and rollout radii $R_F$.
```

这条 proposition 是安全的：它只是 local Lipschitz sufficient condition，不声称全局 Lipschitz、不证明 closed-loop robustness。

---

## 6. Results opening: reinforce representation-diagnostic mainline

在 `\section{Experiments}` 开头，替换或追加：

```tex
The experiments use Gaussian corruption as a controlled stressor for a representation-diagnostic question. We first show that the stressor changes closed-loop behaviour. We then ask where the recovery appears in the model: encoder geometry ($R_E$, rank, transition resolution, ID probe), action-conditioned rollout geometry ($R_F$, rollout drift), and downstream candidate/cost readouts. The goal is not to tune a unique noise level, but to trace how visual perturbations move through the encoder--predictor--planner chain.
```

---

## 7. Reacher fixed-epoch lift paragraph

在 sweep subsection 结束、ACPC-basin subsection 之前加入：

```tex
Reacher shows an additional fixed-epoch effect. Noise training improves not only observation-noise robustness but also unperturbed control: the representative clean row rises from $58.67\%$ at the no-noise baseline to about $86\%$ under noise training. We interpret this as a fixed-protocol regularisation/stabilisation effect rather than as pure robustness recovery. The diagnostic table supports this reading: Reacher does not show a resolution collapse (effective rank and transition-resolution stay flat or slightly improve), while multi-step rollout drift drops sharply ($15.17\to0.44$). Thus the same intervention that contracts same-state noisy rollouts appears to stabilise the clean action-conditioned predictor at epoch 10. This is a diagnostic observation, not a claim about asymptotic training behaviour; independent training seeds and longer training curves would be needed to separate regularisation from optimisation timing.
```

这段能解释为什么 Reacher clean eval 大幅上升，避免读者以为文章把 robustness 和 undertraining/regularization 混在一起。

---

## 8. Discussion 新增 subsection

在 `Discussion and limitations` 的 `Negative checks and next steps` 后、`Conclusion` 前加入：

```tex
\paragraph{From diagnostics to more generalizable world models.}
The corruption study should be read as a diagnostic probe, not as an endpoint. It points to two foundation capabilities for more generalizable latent world models. First, the encoder should filter visual nuisance variation while preserving geometry, contact, goal relations, and other action-relevant state differences needed for planning. Second, the predictor should make these representations useful under intervention: clean and perturbed views of the same state should produce consistent action-conditioned rollouts and cost-relevant predictions under the same action sequence, while states or actions that imply different futures remain separable.

These requirements are broader than Gaussian robustness. They are prerequisites for world models that can adapt predictions and plans under new appearances, goals, or dynamics from context rather than memorising a fixed training distribution. This paper does not test such context adaptation directly; it provides diagnostics that expose where the current encoder--predictor--planner chain succeeds, where it collapses nuisance variation safely, and where stronger objectives must protect action-relevant distinctions.
```

不要写 “human-like” 在正文里。可以放在 talk slides，但 paper 中建议保持 `context adaptation` / `generalizable world models`。

---

## 9. Figure 2 no-marker requirement

必须执行之前的 no-marker 审查：Figure 2 不画任何单点 marker。只显示 curves + error bars。

修改 `tools/paper1_figs.py`：

- 删除 `best_std = ...`。
- 删除 `ax.axvline(...)`。
- 删除 `Robust-eval optimum`、`Highest observed robust eval`、`sigma*`。
- 使用 `ax.set_title(t, fontsize=11)`。
- legend 只保留两条曲线，`ncol=2`。

重渲：

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
```

---

## 10. Forbidden / risky claims

不要写：

- `human-like generalization` as achieved result；
- `context learning is demonstrated`；
- `ACPC enables generalization`；
- `noise training learns generalizable world models`；
- `optimal sigma`；
- `robust-eval optimum`；
- `best checkpoint` as statistical claim。

安全写法：

- `diagnostics expose prerequisites`；
- `points toward foundation capabilities`；
- `controlled probe`；
- `fixed-epoch diagnostic observation`；
- `not a claim about asymptotic training behaviour`；
- `not tested directly in this paper`。

---

## 11. Final grep/build commands

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

Then:

```bash
rg -n "human-like|like a human|context learning is demonstrated|ACPC enables generalization|optimal sigma|robust-eval optimum|highest observed robust|sigma\\*|sigma\\^\\*|green dashed|point-best|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

Expected: no main-text hits except clearly safe `context adaptation` / `more generalizable world models` phrasing.

---

## Final intended story

The paper should read as:

> This is a representation-diagnostic study. Visual corruption is the stressor; noise training is the intervention. The diagnostic traces how perturbations and interventions affect encoder geometry, action-conditioned rollout geometry, candidate costs/rankings, and discriminability. The fixed-candidate theory explains why rollout consistency matters for planning. The long-term implication is not that this paper solves generalization, but that these diagnostics expose two prerequisites for generalizable world models: control-relevant representation and action-conditioned predictive dynamics.
