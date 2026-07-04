# Paper1 理论加深与 unseen perturbation 整改方案 — Codex 执行版

Date: 2026-07-04  
Branch: `qun-team/wm_exp@ag/dev`  
Target paper: `paper1/main.tex`  
Scope: 本文件只给 Codex 执行整改的计划与建议文本；不要在执行时手改 released numeric JSON。数值应从现有 artifacts / tables 读取或复核后同步。

---

## 0. 总体判断

当前 Paper1 的主线已经适合定位为 **controlled Gaussian-noise diagnostic paper**：

- ACPC 把鲁棒性从 encoder invariance 推到 action-conditioned rollout readout；
- fixed-candidate / sampled-pool stability 结果能支撑 diagnostic interpretation；
- 三训练种子 Gaussian lockbox 和 held-out fixed-rule validation 已经比单 seed 解释强很多；
- unseen stressor 结果显示 TwoRoom / Reacher 有明显跨扰动收益，PushT / Cube 没有 clear gain，且 paired diagnostics 与这个谨慎结论一致。

但如果按顶会主会审稿标准，当前仍有两个短板：

1. **理论部分正确但偏浅**：Lipschitz drift、margin stability、union bound、local Gaussian linearization 都合理，但更像 formal diagnostic justification，不是强理论贡献。
2. **unseen perturbation 证据被埋在 appendix**：它不能升级成 universal transfer claim，但非常适合在主文作为 bounded scope check，回应 “matched Gaussian only?” 的 reviewer 攻击。

本整改方案目标是：**加深理论连接但不越界；把 unseen perturbation 作为有限主文 scope check；不把论文包装成新方法或通用鲁棒性论文。**

---

## 1. 不要做的改动

执行时避免以下过度 claim：

- 不要写 “ACPC guarantees robustness”。
- 不要写 “ACPC proves CEM / MPC closed-loop stability”。
- 不要写 “Gaussian noise training universally improves unseen perturbation robustness”。
- 不要写 “PushT / Cube fail under unseen perturbations”。正确表述是：**no clear behavioral gain at this evaluation scale / within sampling variability**。
- 不要把 existing unseen Phase-0 diagnostic slice 叫成 `R_E/R_F ACPC basin`，除非重新生成了同一 basin protocol 的 unseen version。当前 existing unseen 表更准确的名字是 **paired ACPC/PCC/CRA diagnostic slice**。
- 不要把 appendix 中所有 unseen 内容搬到主文。主文只放一个 compact scope check。

---

## 2. 理论部分加深方案

### T1. 明确理论角色：diagnostic calibration，不是 guarantee

**位置建议：** `main.tex` 的 `\section{Action-Conditioned Predictive Consistency}` 末尾，或 `\subsection{Sampled-pool stability and Gaussian sensitivity}` 末尾，在进入 experiments 前。

**建议新增文字：**

```tex
The theoretical statements in this section are intended as diagnostic calibration rather than as closed-loop control guarantees. They identify the failure modes that a paired clean/noisy diagnostic should measure---rollout-disagreement tails and clean candidate-margin tails---and explain why the Gaussian basin radius is an action-conditioned encoder--predictor sensitivity measurement. They do not assert that a low empirical median radius is a uniform certificate for adaptive CEM, repeated replanning, or environment-feedback stability.
```

**目的：** 让 theory reviewer 明确看到作者知道 theorem 与 empirical proxy 的距离，降低 “overclaimed theorem” 风险。

---

### T2. 增加一个 finite-sample tail calibration proposition

当前 sampled-pool theorem 给出：flip probability 受 `Kδ + margin-tail` 控制。但 empirical table 里 ACPC/PCC/MAF 是有限样本估计。建议加一个很轻量但实用的 finite-sample calibration proposition，把理论和 diagnostics table 接上。

**位置建议：** 放在 `\Cref{thm:sampled-pool-acpc}` 后，或 appendix proof section 中。主文若篇幅紧，可主文一句引用，完整 proposition 放 appendix。

**建议 LaTeX：**

```tex
\begin{proposition}[Finite-sample tail calibration for paired diagnostics]
\label{prop:finite-sample-tail-calibration}
Fix the diagnostic sampling distribution used to draw paired clean/noisy histories and candidate action sequences. Let
\[
  X_i = \mathbf 1[D_i > \epsilon], \qquad i=1,\ldots,n,
\]
where $D_i$ is the paired ACPC rollout-readout discrepancy for the sampled diagnostic item, and let $p_\epsilon=\Pr[D>\epsilon]$ and $\hat p_\epsilon=n^{-1}\sum_i X_i$. Then, with probability at least $1-\delta$ over the diagnostic sample,
\[
  p_\epsilon \le \hat p_\epsilon + \sqrt{\frac{\log(1/\delta)}{2n}}.
\]
The same one-sided calibration applies to any bounded Bernoulli readout computed from the paired candidate costs, such as a margin-conditioned action-flip indicator.
\end{proposition}

\begin{proof}
This is the one-sided Hoeffding inequality applied to the Bernoulli variables $X_i$.
\end{proof}
```

**必须加的边界句：**

```tex
This calibration is with respect to the diagnostic sampling distribution and the released finite candidate construction. It does not convert the empirical diagnostic into a uniform bound over all CEM samples or all closed-loop states.
```

**为什么有用：** 这不是强理论，但能把 sampled-pool theorem 中的 ACPC-tail 与实际 finite diagnostic estimates 接起来，比单纯 deterministic theorem 更像顶会论文中的 principled diagnostic section。

---

### T3. 固定规则 selector 需要 baseline comparator，否则 triage claim 会被问住

当前主文有 fixed ACPC/PCC/CRA/MAF aggregate rule，在 held-out seeds 上 7/8 within 5pp、held-out regret `2.21 ± 1.83` pp。这个结果不错，但 reviewer 会问：

- fixed `std=0.08` 是否同样好？
- 单独 ACPC 是否已经够？
- 单独 PCC / CRA / MAF 呢？
- 随机选 nonzero std 的 regret 是多少？

**建议 Codex 新增一个小表或 appendix 表，不一定放主文。**

**数据源：**

- `assets/paper1_data/three_seed_diagnostic_validation.json`
- `assets/paper1_data/acpc_phase0_lewm_three_seed.json`
- `assets/paper1_data/training_seed_eval_manifests/lewm_seed3072_evals.json`
- `assets/paper1_data/training_seed_eval_manifests/lewm_seed3073_evals.json`
- `assets/paper1_data/training_seed_eval_manifests/lewm_seed3074_evals.json`

**需要比较的 selectors：**

1. `aggregate_rank_acpc_pcc_cra_maf`：当前主规则。
2. `fixed_std_0.08`：固定选 high-noise endpoint。
3. `best_acpc_only`：非零 std 中 ACPC-H/trans 最低。
4. `best_pcc_only`：PCC 最低。
5. `best_cra_only`：CRA 最高。
6. `best_maf_only`：MAF 最低。
7. `random_nonzero_std`：8 个非零 std 的 expected regret；可以 exact average over rows，不需要 Monte Carlo。
8. `oracle_best`：只作为 lower bound，不作为 selector。

**建议表格列：**

```tex
Selector & split & within 5pp & regret to best & note
```

**如果主规则没有明显优于 `fixed_std_0.08`：** 不要硬说 “ACPC selector is better”。改成：

```tex
The aggregate rule is best read as a no-retraining diagnostic triage rule that localizes the plateau and explains exceptions, not as evidence that the diagnostic dominates a fixed high-noise endpoint in every task.
```

**如果主规则明显优于简单指标：** 可以在主文保留一句：

```tex
A selector-baseline audit in Appendix~X shows that the aggregate rule reduces held-out regret relative to single-metric and random nonzero-std selectors, while remaining comparable to the strong fixed-0.08 endpoint on the broad Gaussian plateaus.
```

**验收标准：** reviewer 不会再说 fixed-rule validation 缺少 obvious baseline。

---

### T4. 可选：增加 selector regret decomposition，但不要伪装成强理论

如果需要再加一点理论深度，可在 appendix 加一个 simple decomposition，帮助解释 diagnostic selector 的边界。

**建议文字：**

```tex
For a task--training-seed block $b$, let $S_b(c)$ be the closed-loop observation-noise score of checkpoint $c$, $c_b^\star=\arg\max_c S_b(c)$ the closed-loop oracle checkpoint, and $\hat c_b$ the diagnostic selector. The selector regret is
\[
  \mathrm{Regret}_b = S_b(c_b^\star)-S_b(\hat c_b).
\]
This quantity can be small either because the diagnostic identifies the same high-performing row or because the closed-loop sweep has a broad plateau. We therefore report both within-5pp counts and regret to best, and we audit simple selectors such as fixed $\sigma_{\max}=0.08$ to avoid attributing plateau structure to the diagnostic rule alone.
```

**目的：** 这可以直接解释为什么 “exact best 2/12” 不致命，但 “within 5pp 10/12” 有意义。

---

## 3. Unseen perturbation 主文整改方案

### U1. 结论定位

主文中应加入一段 **bounded unseen-stressor scope check**，但不要升级为主结果。

推荐结论句：

```tex
The unseen-stressor check is therefore a specificity test rather than a transfer benchmark: when TwoRoom and Reacher show clear out-of-family score gains, the paired diagnostics move in the expected direction; when PushT and Cube show no clear behavioral gain at this evaluation scale, the diagnostics do not support a positive or negative transfer claim.
```

---

### U2. 放置位置

**优先位置：** `\section{Experiments}` 中，在 `\subsection{Selective consistency and failure checks}` 后、`\section{Discussion and limitations}` 前，新增一个短 subsection：

```tex
\subsection{Bounded unseen-stressor scope check}
\label{sec:exp-unseen-scope}
```

**理由：**

- 放在 main Gaussian story 之后，不打断核心证据链；
- 放在 Discussion 前，可以主动回应 scope；
- 不需要进入 abstract 或 contributions，避免过度强调。

如果页数太紧，就不要开 subsection，改为 Discussion 的一个 paragraph + compact table。

---

### U3. 主文 compact table 建议

**数据来源：** appendix 中已有 strongest-severity unseen score aggregate 与 matched diagnostic slice。

注意：score side 是三训练种子 aggregate；diagnostic side 是 matched diagnostic slice，当前为 seeds 3073/3074 的 selected rows。caption 必须说清楚。

**建议 LaTeX：**

```tex
\begin{table}[H]
\centering
\caption{Bounded unseen-stressor scope check. Score $\Delta$ is the fixed $\sigma_{\max}=0.08$ checkpoint minus the no-noise baseline under the strongest unseen stress, averaged over training seeds 3072/3073/3074. Diagnostic deltas are from the matched paired-diagnostic slice on independent seeds 3073/3074; negative is better for ACPC-$H$/transition and PCC, positive is better for CRA. The table is a scope check, not a universal transfer claim.}
\label{tab:unseen-scope-main}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{llrrrr>{\raggedright\arraybackslash}X}
\toprule
Task & stress & score $\Delta$ & $\Delta$ACPC & $\Delta$PCC & $\Delta$CRA & reading \\
\midrule
TwoRoom & blur $k=15$ & $+43.11 \pm 8.90$ & $-0.590$ & $-42.7$ & $+0.567$ & clear gain; diagnostics aligned \\
Reacher & blur $k=15$ & $+49.22 \pm 3.29$ & $-1.770$ & $-47.0$ & $+0.568$ & clear gain; diagnostics aligned \\
PushT & resize $0.25$ & $+2.89 \pm 17.98$ & $-0.249$ & $-6.4$ & $+0.009$ & no clear gain at this scale \\
Cube & resize $0.25$ & $-0.89 \pm 1.40$ & $+0.088$ & $-0.1$ & $-0.042$ & no clear gain at this scale \\
\bottomrule
\end{tabularx}
\end{table}
```

**解释段建议：**

```tex
The main claim remains matched Gaussian-noise robustness. After freezing the Gaussian grid, we also ran a strongest-severity unseen-stressor check to test whether the diagnostic direction is specific outside the training perturbation family. \Cref{tab:unseen-scope-main} shows a split pattern. TwoRoom and Reacher exhibit clear positive score deltas under strongest blur, and the paired ACPC/PCC/CRA diagnostics move in the same direction. PushT and Cube do not show a clear score gain at the scale of training-seed and evaluation variance, and their diagnostic deltas are correspondingly weak or mixed. We therefore use this table only as bounded scope evidence: ACPC-style paired diagnostics localize clear out-of-family improvements when present, but the current evidence does not justify a universal cross-perturbation robustness claim.
```

**如果担心只放 blur for TwoRoom/Reacher、resize for PushT/Cube 显得 cherry-pick：**

可以在 caption 或 prose 加一句：

```tex
The selected rows follow the appendix convention: TwoRoom/Reacher blur are clear positive-transfer endpoints, while PushT/Cube resize are small-effect boundary endpoints. The full strongest-severity unseen score aggregate is reported in Appendix~\ref{sec:appendix-unseen-transfer}.
```

更保守的替代是主文只放 score aggregate table，不放 diagnostic deltas；但我建议保留 diagnostic deltas，因为它正好支撑 “有行为收益时 diagnostics 同向，无清晰收益时 diagnostics 不过度表态”。

---

### U4. 术语修正：不要误称为 ACPC basin，除非重跑 basin protocol

现有主文 `ACPC-basin` 是 Gaussian same-state repeated view 的 `R_E/R_F` basin protocol，且主文写明 diagnostic intentionally excludes blur/resize。unseen appendix 当前是 Phase-0 paired ACPC/PCC/CRA diagnostic slice，不是同一个 `R_E/R_F` basin table。

因此主文请写：

- `paired ACPC/PCC/CRA diagnostics`
- `matched paired-diagnostic slice`
- `ACPC-style paired diagnostics`

不要写：

- `unseen ACPC basin proves ...`
- `R_F supports unseen transfer ...`

如果作者确实想用 “ACPC basin” 说法，需要新增 unseen basin artifact：同一 `R_E/R_F` protocol，在 blur/resize observation-history perturbations 下，对 selected endpoints 重跑。否则不要混用术语。

---

## 4. Abstract / Introduction 是否需要改？

### 建议：不要改 abstract

当前 abstract 已经很满，而且主 claim 是 Gaussian diagnostic。把 unseen 加进 abstract 会把 reviewer 的期待拉高，反而引来 cross-perturbation benchmark 质疑。

### Introduction 可选加一句

如果想在 contribution C3 里轻微提示 scope check，可加非常弱的一句：

```tex
A bounded unseen-stressor appendix check further shows diagnostic specificity on selected clear-effect blur endpoints, while leaving cross-perturbation robustness as future benchmark work.
```

但我的建议是：**不加到 contributions，只放 Experiments/Discussion。**

---

## 5. Appendix 同步

如果主文新增 `tab:unseen-scope-main`，appendix `\section{Unseen-stressor score and matched diagnostic check}` 需要轻微同步：

1. 第一段加一句：

```tex
A compact four-row summary of this appendix appears in \Cref{tab:unseen-scope-main}; the appendix gives the full score aggregate and matched diagnostic slice.
```

2. 确保 appendix 仍清楚说明：
   - score aggregate covers seeds 3072/3073/3074；
   - matched diagnostic slice covers 3073/3074；
   - PushT/Cube are no-clear-effect rows, not negative transfer proof。

---

## 6. Codex 执行 checklist

### 修改文件

- `paper1/main.tex`
- 可选：新增或更新一个 appendix selector-baseline artifact / table generator。
- 可选：新增 `assets/paper1_data/selector_baseline_audit_YYYYMMDD.json` 与 `.md`，但不要手写数值 JSON。

### 必须运行

```bash
bash paper1/build.sh --clean
python tools/check_paper1_consistency.py
```

如果 blind / arXiv readiness scripts 当前依赖 author metadata，可按现有流程判断是否运行：

```bash
bash paper1/check_blind_ready.sh
bash paper1/check_arxiv_ready.sh
```

### 需要 grep 的危险词

```bash
grep -R "universal cross-perturbation\|guarantees robustness\|CEM stability\|closed-loop guarantee\|PushT.*fail\|Cube.*fail" -n paper1/main.tex
```

这些词如果出现，要确认是 negative statement，而不是 claim。

---

## 7. 预期审稿收益

整改后，顶会 reviewer 的观感会更稳：

- 理论不再只是 deterministic margin theorem，而是和 finite diagnostic estimation 接上；
- fixed-rule validation 不再缺少 simple selector baseline；
- unseen perturbation 不再被埋没，能主动回应 “matched Gaussian only”；
- PushT/Cube 的边界结果被诚实解释，不会被 reviewer 认为 cherry-pick；
- 主 claim 仍保持克制：Gaussian diagnostic + bounded scope check，而不是 universal robustness method。

---

## 8. 最终推荐执行顺序

1. 先加 U1--U4：主文 unseen compact scope check。这是最直接提升 reviewer perception 的改动。
2. 再加 T1--T2：理论 calibration paragraph + finite-sample proposition。
3. 然后做 T3：selector baseline audit。如果结果支持主规则，就放 appendix 小表；如果不支持，也诚实写成 plateau/context audit。
4. 最后跑 build 和 consistency checks，统一 terminology。