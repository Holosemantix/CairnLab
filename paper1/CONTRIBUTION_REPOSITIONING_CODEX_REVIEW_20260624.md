# Paper 1 contribution repositioning review — 2026-06-24

目标：解决一个核心心理和审稿风险：读者可能觉得“视觉噪声会伤模型、加噪训练会恢复”是常识，从而认为文章只是普通 Gaussian augmentation sweep。本文需要把贡献中心重新钉牢：**不是忽略 encoder 表征，也不是只说 predictor；而是把视觉鲁棒性诊断放在一个 encoder → action-conditioned predictor → cost/ranking 的多阶段链条里。**

当前最稳的中心命题应改成：

> Visual robustness in JEPA latent world-model control should be diagnosed as a multi-stage property. Encoder perturbation geometry tells whether visual noise enters and reshapes the latent state; action-conditioned rollout consistency tells whether that shift changes predicted futures and candidate costs; discriminability diagnostics check whether contraction has erased action-relevant distinctions. Encoder invariance is therefore useful but insufficient as a sole target.

这比“不要看 encoder，只看 predictor”更准确，也能回应：加噪训练通常让 encoder 和 predictor 后的空间都变好；真正的 claim 是 **不能只用 encoder invariance 作为鲁棒性定义**。

---

## 0. 立即要修的两件事

### 0.1 Figure 2 不要有任何单点 marker

用户指出得对：即使标成 `highest observed` 也会把一个 noisy grid maximum 放大。Figure 2 应只显示两条曲线和误差条，不画竖线，不写 `sigma*`，不写 `highest observed robust eval`。

必须修改：

- `tools/paper1_figs.py`：删除 `best_std`、`ax.axvline(...)`、`Robust-eval optimum`、`Highest observed robust eval`、`sigma*` 或 `ref. sigma` title。
- `paper1/main.tex`：Figure 2 caption 删除 `green dashed`、`point-best`、`highest observed`、`sigma*` 语义。
- 重渲：`python -m tools.paper1_figs --out-dir assets/paper1_figs`。

推荐 Figure 2 caption：

```tex
\caption{Noise-training sweep across four tasks. Blue: unperturbed evaluation; red: observation-only Gaussian noise $\sigma=0.08$ with an unperturbed goal. Error bars show the population standard deviation across the three evaluation seeds. The curves show task-dependent high-performing regions rather than a statistically unique training-noise optimum.}
```

### 0.2 不要把 `best` 语义换成 `highest observed` 后继续强化单点

全文里需要统一口径：

- 主文：讲 **regions / plateaus / high-performing ranges**。
- compact tables：若必须保留代表行，称 **representative high-corruption row** 或 **compact reference row**。
- appendix：可以说明某行是 grid maximum，但必须同时说不是统计唯一 optimum。

---

## 1. 重新定位文章贡献：不要把贡献说小

### 当前风险

如果文章读成下面这样，就会显得普通：

> LeWM/PLDM under Gaussian noise drops; Gaussian-noise training recovers; tasks differ.

这些都不是足够强的贡献。

### 应该改成下面这样

文章应正面 claim：

> The contribution is a diagnostic principle for latent world-model control: robustness is a property of the full encoder-to-predictor-to-planner chain. Encoder geometry remains part of the diagnosis, but the decisive question is whether visual perturbations change action-conditioned predicted futures and candidate costs while preserving action-relevant distinctions.

### 建议替换 Introduction 里的贡献 C1

当前 C1 如果已经写了 planner link，可以进一步纳入 encoder 位置：

```tex
\paragraph{C1 --- Multi-stage diagnostic principle.}
We formulate visual robustness for JEPA world-model control as a multi-stage diagnostic rather than encoder invariance alone. Encoder perturbation geometry measures whether visual noise enters and reshapes the latent state; action-conditioned predictive consistency measures whether that shift changes predicted futures and candidate costs; the discriminability countercondition prevents contraction from erasing action-relevant distinctions. Under a Lipschitz cost readout, bounded rollout disagreement bounds candidate-cost drift on a fixed candidate set, and a sufficiently large clean action margin preserves the selected candidate within that set.
```

这段比“只看 predictor”更准确，也更有分量。

---

## 2. 理论部分需要补一个 encoder-to-rollout 桥

当前 theory section 已经有：

1. ACPC controls candidate-cost drift。
2. Candidate top-1 stability。
3. ACPC-margin corollary。
4. ACPC alone permits collapse。

这些是正确的，但还缺一小段把 encoder 重新纳入理论链条。否则读者会误解为“encoder 不重要”。建议新增一个 short observation/proposition，放在 cost-drift proposition 前或后。

### 建议新增 Proposition

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
```

### 紧跟解释段

```tex
This bound makes encoder geometry a first-stage risk signal: a small encoder shift is sufficient for small rollout disagreement only when the action-conditioned predictor is locally insensitive in that direction. It is not necessary, because a large encoder shift can lie in a nuisance direction contracted by $G_{\mathbf a}$. It is not sufficient without the local-sensitivity condition, because a small encoder shift can be amplified by the predictor or cost readout. This is why the empirical analysis reports both encoder radii $R_E$ and rollout radii $R_F$.
```

这段非常关键：

- 不否定 encoder；
- 解释为什么加噪训练让 encoder 也变好是合理现象；
- 解释为什么文章仍要把最终鲁棒性读到 predictor/cost 层。

### 为什么这没有理论幻觉

这是最基础的 Lipschitz bound。只要写清楚“if locally Lipschitz”，就没有错。它不声称实际模型一定满足全局 Lipschitz，也不声称能证明 closed-loop robustness。

---

## 3. ACPC section 的叙述要从“替代 encoder”改成“组织 encoder 与 predictor”

### 当前风险句型

避免读起来像：

```text
Encoder invariance is not the target; ACPC is the target.
```

这会让读者觉得你忽略 encoder。

### 推荐句型

```text
Encoder invariance is a useful first-stage diagnostic, but it is not a complete robustness definition for control. The planner acts on predicted futures and costs, so the diagnostic must continue through the action-conditioned predictor and retain a guard on action-relevant separation.
```

### 在 Operational diagnostics 后加一句

```tex
Thus $R_E$ and $R_F$ answer different questions. $R_E$ measures whether the visual perturbation changes the encoded neighbourhood; $R_F$ measures whether that change survives action-conditioned rollout in the coordinates used for planning. Robust checkpoints in this Gaussian-noise study often reduce both, so the claim is not predictor-only contraction. The claim is that encoder contraction is interpreted through its downstream predictive effect and checked against discriminability.
```

这能直接解决“encoder 也变好，为什么你说不看 encoder”的问题。

---

## 4. 摘要需要更明确贡献，不要只像实验记录

当前 abstract 末尾偏弱：

```text
The paper contributes a diagnostic framing, release artifacts, and negative ablations for this controlled setting.
```

建议改成更有贡献感但不过 claim：

```tex
The result is a multi-stage diagnostic account: encoder geometry remains informative, but robustness for control is decided by whether perturbation-induced latent changes survive action-conditioned rollout into candidate costs while action-relevant distinctions remain separable. The paper contributes this diagnostic framing, a fixed-candidate planner-stability argument, release artifacts, and negative ablations for the controlled Gaussian-noise setting.
```

这样读者一开始就知道：不是普通 sweep，而是 multi-stage diagnostic account。

---

## 5. Figure 2 和 sweep narrative 的最终安全写法

### Figure 2 不标点

只保留 curves + error bars。

### Sweep paragraph 推荐替换

```tex
The sweep has two readings. First, input-side noise can be effective in this protocol: PushT and Reacher recover much of the observation-noise loss. Second, recovery is range-based rather than point-based. TwoRoom reaches high robust success across high-noise settings, PushT recovers across moderate-to-high noise levels, Reacher has a broad plateau, and Cube responds weakly and non-monotonically. The claim is therefore not that a particular $\stdmax{}$ is optimal, but that a single scalar Gaussian-noise strength is only a coarse control over the nuisance/discriminability tradeoff.
```

不要再写“point-best differ”。如果要讲 PushT unperturbed/noisy tension，用 appendix 或一句弱表达：

```tex
PushT also shows that high unperturbed and high noisy performance need not align perfectly across the grid, but neighbouring high-noise values are close under the reported evaluation variability.
```

---

## 6. Compact ACPC basin table 的安全解释

保留 compact table 可以，但不要让它像基于 noisy maximum 的 post-hoc cherry-pick。

### 表 caption 推荐

```tex
\caption{LeWM Gaussian-noise ACPC-basin proxy: no-noise baseline vs. representative high-corruption grid rows. $R_E$ is normalised encoder-view spread and $R_F$ is normalised action-conditioned rollout-view spread under the same action sequence. The rows summarise the high-corruption regime; the full grid is reported in Appendix~\ref{sec:appendix-acpc-basin-grid}.}
```

### 正文推荐

```tex
The compact rows show the characteristic basin movement in the high-corruption regime: $R_F$ is much smaller than at the no-noise baseline on the tasks with large recovery. The full-grid appendix is the evidence against treating any row as a model-selection rule.
```

这比“direct-eval-selected endpoint”更稳。

---

## 7. 五个顶会审稿人式最终判断

### Reviewer A — 理论洁癖型

理论现在有实质，但需要补 encoder-to-rollout proposition。否则 ACPC theory 只从 rollout discrepancy 开始，没解释 encoder 表征诊断如何进入理论链。补后结构完整：

```text
encoder shift + local predictor sensitivity -> rollout disagreement -> cost drift -> action ranking stability
```

这条链足够作为 diagnostic theory，不是 hallucination。

### Reviewer B — 统计洁癖型

Figure 2 不能有任何 single-point marker。所有 `best / optimum / highest observed` 都要降级成 plateau/range。当前设计无法支持唯一最优噪声强度。Full-grid appendix 是必要的；compact rows 只能是 summaries。

### Reviewer C — RL/world-model 型

文章有意义，但要把故事讲成 world-model control pipeline diagnosis，而不是 augmentation sweep。encoder diagnostics 不能删，predictor diagnostics 也不能单独神化。贡献是把二者组织到 action-conditioned planning chain 中。

### Reviewer D — 写作/AI 感型

避免“我们不 claim X，不 claim Y”反复出现。正向主线：

1. Visual noise reshapes encoder neighbourhoods。
2. Predictor may contract or amplify those shifts。
3. Planner sees costs/rankings。
4. Discriminability prevents collapse。
5. Experiments show where this chain moves。

这比防御性 caveat 更像成熟论文。

### Reviewer E — Artifact 型

必须改图生成脚本并重渲 `fig2_sweep.png`。只改 LaTeX 不够。最终 grep 要查：

```bash
rg -n "robust-eval optimum|highest observed robust|sigma\\*|sigma\\^\\*|green dashed|unique optimum|optimal sigma|point-best|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

主文和 Figure 2 generator 不应命中。

---

## 8. 给 Codex 的执行清单

### Step 1: 删除 Figure 2 marker

修改 `tools/paper1_figs.py`：

- 删除 `best_std = tables[...]`。
- 删除 `ax.axvline(...)`。
- 删除 marker legend。
- `ax.set_title(t, fontsize=11)`。
- legend `ncol=2`。

重渲：

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
```

### Step 2: 改 main.tex 中的 sweep wording

替换：

- abstract 的 `best recovery point`；
- Figure 2 caption；
- sweep table headers；
- sweep paragraph；
- ACPC basin compact-row explanation；
- PLDM/hetero/Phase-0 中的 point-best/selected endpoint。

### Step 3: 加 encoder-to-rollout proposition

在 `sec:acpc-planner-stability` 中加入 `Encoder shift as one route to rollout disagreement` proposition 和解释段。

### Step 4: 强化 abstract 贡献感

把 abstract 末尾改成 multi-stage diagnostic account，而不是只说 release artifacts。

### Step 5: build and grep

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "robust-eval optimum|highest observed robust|sigma\\*|sigma\\^\\*|green dashed|unique optimum|optimal sigma|point-best|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

---

## 9. 最终定位句

建议整个文章围绕这句话收束：

> This paper is not an augmentation sweep. It is a multi-stage diagnostic study of visual robustness in JEPA world-model control: encoder geometry shows how visual perturbations enter the latent state, action-conditioned rollout consistency shows whether those perturbations affect predicted futures and candidate costs, and discriminability checks whether robustness is bought by erasing action-relevant distinctions.

这句话能让读者感觉这个工作不是“大家都知道”，而是把一个常见现象放到了正确的 world-model control 诊断结构里。
