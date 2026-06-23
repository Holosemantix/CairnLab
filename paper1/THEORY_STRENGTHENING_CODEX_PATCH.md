# Paper 1 theory strengthening patch for Codex

目标：在不夸大成新方法或 full CEM theorem 的前提下，把 ACPC 理论段从“短桥接”升级成能支撑文章分量的核心理论骨架。

当前 `main.tex` 已经完成了三件重要修正：

1. `Proposition~\ref{prop:cost-drift}` 后已经说明它对应 L2 rollout-token distance / $R_F$。
2. `Proposition~\ref{prop:top1-stability}` 已经写明 lower cost is preferred。
3. top-1 stability proof 已经写出关键不等式。

因此本 patch 只做两件事：

- 新增一个 corollary，把 `ACPC -> cost drift -> fixed-candidate top-1 stability` 合成一条清楚结论。
- 把 “ACPC without discriminability admits collapse” 从一句话升级成正式 proposition，证明 discriminability guard 是必要条件，不是 defensive caveat。

不要新增实验、不要增强到 closed-loop/CEM guarantee、不要写 ACPC guarantees robustness。

---

## 1. 修改 C1 contribution

### 当前文本

```tex
\paragraph{C1 --- Problem reframing.} We formulate a diagnostic lens for visual robustness in world-model control: action-conditioned predictive consistency plus an action-relevant discriminability countercondition, rather than encoder-level invariance.
```

### 替换为

```tex
\paragraph{C1 --- Diagnostic principle and planner link.} We formulate visual robustness for world-model control as action-conditioned predictive consistency with an action-relevant discriminability countercondition. On a fixed candidate set, bounded ACPC implies bounded candidate-cost drift and preserves the selected candidate under a clean action-margin condition.
```

理由：这不是过 claim。它只说 fixed candidate set 和 margin condition，正好对应 formal section。它能显著增加文章分量，让贡献不只是“定义一个 lens”。

---

## 2. 修改 theory section 标题

### 当前文本

```tex
\subsection{A planner-stability view}\label{sec:acpc-planner-stability}
```

### 替换为

```tex
\subsection{ACPC as a fixed-candidate stability condition}\label{sec:acpc-planner-stability}
```

理由：标题更明确、更有理论感，但仍然不夸大到 full CEM 或 closed-loop theorem。

---

## 3. 在 Proposition 2 proof 后加入 Corollary

### 定位

找到当前片段：

```tex
\begin{proof}
For any $j\neq 1$,
\[
C_{\tilde h}(\mathbf a^j,g)-C_{\tilde h}(\mathbf a^{(1)},g)
\ge
C_h(\mathbf a^j,g)-C_h(\mathbf a^{(1)},g)-2\eta
\ge \Delta-2\eta >0 .
\]
Thus every non-top candidate remains more costly than the clean top-1 candidate on the corrupted branch.
\end{proof}
```

### 在这段后面立即插入

```tex
\begin{corollary}[ACPC-margin condition for candidate stability]\label{cor:acpc-margin}
Let $\mathcal A$ be a shared candidate set and assume the cost readout $J$ is $L_J$-Lipschitz as in Proposition~\ref{prop:cost-drift}. If every candidate in $\mathcal A$ has rollout-readout discrepancy at most $\epsilon$ between the clean and corrupted branches, and the clean branch has top-1/top-2 margin $\Delta > 2L_J\epsilon$, then the two branches select the same top-1 candidate from $\mathcal A$.
\end{corollary}

\begin{proof}
Proposition~\ref{prop:cost-drift} gives $|C_h(\mathbf a^j,g)-C_{\tilde h}(\mathbf a^j,g)|\le L_J\epsilon$ for every candidate. Applying Proposition~\ref{prop:top1-stability} with $\eta=L_J\epsilon$ gives the result.
\end{proof}
```

理由：这个 corollary 是文章理论核心。它把 ACPC、cost readout、candidate margin 直接连成一句可引用结论。

---

## 4. 替换 Proposition 后的解释段

### 当前文本

```tex
These propositions explain the roles of the downstream diagnostics. ACPC controls candidate-cost drift; if the drift is small relative to the clean action margin, the selected candidate is stable. Candidate-ranking agreement and margin-conditioned action flips are therefore readouts of the same chain. The chain can fail in two simple ways. Encoder closeness is neither necessary nor sufficient: an invertible nuisance-subspace change can move $E_\theta(h)$ and $E_\theta(\tilde h)$ far apart while leaving $\Pi(F_\theta^{1:H}(\cdot,\mathbf a))$ and the cost unchanged, whereas a small encoder displacement can still flip a low-margin cost ranking if the predictor or cost readout has high local sensitivity. ACPC without discriminability also admits collapse: a constant encoder and predictor give zero same-state ACPC for every perturbation pair, but merge states requiring different actions, costs, or transitions. Low ACPC is meaningful only with a guard on action-relevant separation.
```

### 替换为

```tex
The corollary is the formal reason that ACPC is a planning diagnostic rather than only a representation-distance measurement. It links paired predictive agreement to candidate-cost stability and then to action selection under a margin condition. Candidate-ranking agreement and margin-conditioned action flips are downstream readouts of this chain. The result is deliberately limited to a fixed candidate set; CEM resampling, repeated replanning, and environment feedback can still change the closed-loop trajectory.

Encoder closeness is not the right substitute for this condition. It is not necessary: a nuisance-subspace change can move $E_\theta(h)$ and $E_\theta(\tilde h)$ far apart while leaving $\Pi(F_\theta^{1:H}(\cdot,\mathbf a))$ and the cost unchanged. It is not sufficient either: a small encoder displacement can flip a low-margin cost ranking when the predictor or cost readout has high local sensitivity.

\begin{proposition}[ACPC alone permits collapse]\label{prop:acpc-collapse}
There exist encoder--predictor pairs for which $\mathrm{ACPC}_H=0$ for every same-state perturbation pair and every action sequence, while action-distinct states are mapped to identical rollout readouts.
\end{proposition}

\begin{proof}
Let $E_\theta(h)=z_0$ for every history and let $F_\theta^k(z_0,\mathbf a)=u_0$ for every horizon $k$ and action sequence $\mathbf a$. Then clean and corrupted branches always have identical rollout readouts, so same-state ACPC is zero. However, any two states that require different actions, costs, or transitions also have identical readouts, violating the discriminability condition in \Cref{eq:acpc-disc}.
\end{proof}

Thus low ACPC is meaningful only with a guard on action-relevant separation. This is the role of the transition-resolution and inverse-dynamics probes in the main diagnostics, and it is the failure exposed by the heteroscedastic-loss ablation in Appendix~\ref{sec:appendix-D}.
```

理由：

- 把 collapse 反例变成正式 proposition。
- 明确 discriminability guard 是必要的理论条件，而不是“我们怕被说 collapse”才加的 caveat。
- 保持边界：只 fixed candidate set，不说 full CEM/closed-loop guarantee。

---

## 5. 软化 TwoRoom 机制措辞

### 当前文本

```tex
\item \textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. Compressing the representation (effective rank $47.6 \to 37.7$) is acceptable and even beneficial: a more compact latent space is easier to plan in.
```

### 替换为

```tex
\item \textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. The observed compression (effective rank $47.6 \to 37.7$) is compatible with high success in this task; the data do not show that compactness itself causes the improvement.
```

理由：避免把相关现象写成因果机制。

---

## 6. 替换 “arXiv version” 内部措辞

### 当前文本

```tex
\paragraph{Scope of this arXiv version.}
This version is a controlled diagnostic study.
```

### 替换为

```tex
\paragraph{Scope.}
This paper is a controlled diagnostic study.
```

理由：最终论文不要像内部 release note。

---

## 7. grep 检查

改完后运行：

```bash
rg -n "ACPC guarantees|proves CEM|closed-loop guarantee|full CEM stability|Scope of this arXiv version|acceptable and even beneficial|optima dissociate|robustness oracle|paper-facing|method-facing" paper1/main.tex || true
```

期望：没有命中，或者只有历史 appendix/plan 文件之外的可解释命中。

---

## 8. build / readiness checks

改完后运行：

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

如果真实作者和最终公开 URL 已经填好，再运行：

```bash
bash paper1/check_arxiv_ready.sh
```

注意：如果 `\arxivauthors` placeholder 仍在，`check_arxiv_ready.sh` 应该失败，这是正确行为。

---

## 9. 完成后的预期效果

改完后，文章的中心 claim 应变成：

> Visual robustness for JEPA world-model control should be diagnosed after action-conditioned prediction, not at the encoder alone. On a fixed candidate set, bounded ACPC yields bounded candidate-cost drift and preserves the selected candidate under a margin condition. Because ACPC alone admits collapse, this consistency must be paired with an action-relevant discriminability guard. The LeWM/PLDM Gaussian-noise sweeps, ACPC-basin grid, and negative ablations support this diagnostic principle without claiming a new training algorithm or full closed-loop guarantee.

这比“我们只是做了几个受限实验”更有分量，同时不会产生理论幻觉或过度 claim。
