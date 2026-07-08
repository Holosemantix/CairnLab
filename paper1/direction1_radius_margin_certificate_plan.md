# Paper1 方向一整改执行文档：ACPC Radius--Margin Certificate 理论统一与轻量诊断验证

> 目标读者：Codex / 后续执行者  
> 适用范围：仅限 `paper1`，尤其是 `paper1/main.tex` 及生成 paper1 图表/表格所需的脚本和结果文件。  
> 分支：`ag/dev`  
> 重要限制：不要基于其他文档推断 paper1 的贡献；不要把本次整改扩成新的 robust training method paper；不要伪造实验结果、阈值或图表。

---

## 0. 本次整改的核心目标

当前 paper1 已经提出了 selective action-conditioned predictive consistency (ACPC)，并用 ATR 衡量 same-state clean/noisy rollout tail risk，用 SMPR 衡量 task-grounded anti-collapse margin。现有理论包括 fixed-candidate stability、sampled-pool tail motivation、local Gaussian sensitivity。方向一的整改目标不是新增训练方法，而是把这些理论组件统一成一个更强的诊断理论：

> A checkpoint is in a matched-perturbation robust interval when the high-probability same-state predictive radius is below the relevant planner/discriminability margin.

建议贡献名称：

**ACPC radius--margin certificate for fixed-checkpoint Gaussian robustness diagnostics.**

整改后 paper1 的理论叙事应从：

> ATR/SMPR 是两个合理的诊断指标。

升级为：

> ATR/SMPR 是同一个 ACPC radius--margin certificate 的两个 empirical estimators：ATR 估计 same-state predictive radius upper tail，SMPR 估计 task-grounded margin preservation / anti-collapse guard。二者共同定位 fixed checkpoint 是否落入 matched-Gaussian robust interval。

---

## 1. Scope guard：必须坚持的边界

### 1.1 本次可以做的 claim

可以 claim：

1. 我们给出一个 fixed-candidate / sampled-pool 层面的 **diagnostic certificate**。
2. ATR 和 SMPR 可以被解释为 certificate 的两个 empirical sides：radius side 与 margin side。
3. 在 matched Gaussian setting 下，certificate 可以用于解释/定位已有 Gaussian robustness plateau。
4. 轻量新增实验可以验证该 certificate 是否和已有 closed-loop plateau 对齐。

### 1.2 本次不能做的 claim

不要 claim：

1. full CEM adaptive sampling guarantee；
2. repeated replanning / full closed-loop trajectory guarantee；
3. universal checkpoint ranker；
4. broad perturbation-family transfer theorem；
5. 新 robust training method；
6. ATR-only 就足以证明 robustness。

### 1.3 实验范围

不需要重新训练大量模型。优先使用现有 checkpoint、现有 rollout/diagnostic logs、已有三训练种子 sweep。新增分析应是 **training-free diagnostic analysis**。

推荐新增分析：

1. full-sweep ATR/SMPR or radius--margin diagnostic overlay；
2. radius--margin overlap plot；
3. ATR-only vs SMPR-only vs ATR+SMPR gate analysis；
4. optional：candidate flip / empirical cost drift；
5. optional：Gaussian sensitivity/Jacobian proxy。

---

## 2. Paper1 当前理论结构与需要补强的位置

执行前先打开 `paper1/main.tex`，只以该文件为准。

当前已有结构大致是：

1. Introduction：强调 JEPA control robustness 不是 encoder invariance，而是 action-conditioned rollout consistency；同时要保留 task-relevant separability。
2. `Action-Conditioned Predictive Consistency`：定义 ACPC-H、discriminability guard、ATR/SMPR。
3. `ACPC as a fixed-candidate stability condition`：从 ACPC discrepancy 推到 candidate cost drift，再到 fixed candidate top-1/top-2 stability。
4. `Sampled-pool tail motivation and Gaussian sensitivity`：说明为什么 ATR 用 upper tail，并给 local Gaussian sensitivity proposition。
5. Appendix proofs：已有 sampled-pool ACPC tail bound。

当前缺口：

1. ATR/SMPR 仍像两个并列 diagnostics，理论上还没有被统一成一个 explicit certificate。
2. robust interval 还没有正式定义。
3. planner margin 与 discriminability margin 还没有被放进同一个 radius--margin risk decomposition。
4. Gaussian sensitivity 目前主要是 expectation-level，尚未和 tail quantile / max tolerable noise scale 连接。
5. 实验中 ATR/SMPR 目前主要是 base vs std0.08 endpoint 对比，若 claim “可定位 robust interval”，最好新增轻量 full-sweep or margin-overlap validation。

---

## 3. 整改后的主线结构

建议将 Section 3 的逻辑整理为：

1. **ACPC definitions**：保留当前 clean/noisy same-state action-conditioned rollout distance。
2. **Predictive tube and task margin view**：新增本次理论动机，把 robustness 解释成 same-state perturbations stay inside predictive tube, while task-distinct states stay outside margin。
3. **ATR/SMPR as empirical certificate components**：重写 diagnostics 小节，让 ATR/SMPR 成为 radius side 和 margin side 的估计器。
4. **ACPC radius--margin certificate**：新增 theorem，统一 fixed-candidate stability、sampled-pool tail、collapse guard。
5. **Local Gaussian quantile sensitivity**：扩展 Gaussian proposition，给出 weighted chi-square quantile 与 local noise threshold `sigma_star`。
6. **Diagnostic validation**：实验新增一小节，验证 radius--margin certificate 是否覆盖已有 robustness plateau。

---

## 4. 需要修改的正文内容

### 4.1 Abstract / Contributions

如果 abstract 或 contribution list 篇幅允许，加入一句理论贡献。不要写成 method。

建议 contribution 改写为：

```latex
This paper makes three contributions. First, it formulates visual robustness for JEPA world-model control as selective action-conditioned predictive consistency rather than encoder invariance. Second, it derives an ACPC radius--margin diagnostic certificate: high-probability same-state predictive radius, planner candidate margins, and task-grounded discriminability margins jointly bound fixed-pool candidate instability and collapse risk. ATR and SMPR are empirical estimates of the two sides of this certificate. Third, it provides a fixed-checkpoint Gaussian robustness study across four tasks and three LeWM training seeds, with lightweight certificate validation and bounded unseen-stressor checks that delimit the perturbation scope.
```

如担心贡献过强，可弱化为：

```latex
Second, it organizes ATR and SMPR under an ACPC radius--margin diagnostic theory...
```

### 4.2 ACPC definition 后新增 subsection

在 `\subsection{Setup and definitions}` 或 `\subsection{Operational consistency diagnostics}` 前后新增：

```latex
\subsection{Predictive tubes and task margins}\label{sec:acpc-radius-margin}
```

核心文本目标：

- 定义 same-state predictive radius：

```latex
R_\sigma(h,\mathbf a)
= d_H\!\left(G_{\mathbf a}(E_\theta(h)),G_{\mathbf a}(E_\theta(\tilde h))\right),
\quad
G_{\mathbf a}(z)=\Pi(F_\theta^{1:H}(z,\mathbf a)).
```

- 定义 high-probability radius：

```latex
r_{1-\alpha}(\sigma)=Q_{1-\alpha}\big(R_\sigma(h,\mathbf a)\big).
```

- 定义 planner margin：

```latex
\Delta_{\mathcal A}(h,g)
= C_h(\mathbf a^{(2)},g)-C_h(\mathbf a^{(1)},g).
```

- 定义 discriminability margin：

```latex
M_{\mathrm{diff}}(i,j,\mathbf a)
= d\big(\Psi(z_i),\Psi(z_j)\big).
```

- 给出直觉：

```latex
The radius side asks whether a visual perturbation remains inside the same-state predictive tube after the action intervention. The margin side asks whether task-distinct states remain outside that tube. Robustness requires both: small radius without margin is collapse, while large margin without small radius is visually unstable planning.
```

### 4.3 重写 ATR/SMPR diagnostics 解释

当前 `Operational consistency diagnostics` 已有 ATR 和 SMPR。需要将其改为 certificate language：

建议加入：

```latex
ATR estimates the upper quantile of the same-state predictive radius. SMPR estimates the probability that task-grounded different-state rollout separations exceed the same-state noisy radius. Therefore the diagnostic gate is not ATR alone, but a radius--margin event: the perturbation tube is small and task-grounded separations remain outside it.
```

并明确：

```latex
Low ATR without high SMPR is not a robustness certificate, because a collapsed encoder--predictor can make all same-state pairs consistent while erasing the distinctions needed for planning.
```

### 4.4 新增核心 theorem：ACPC radius--margin certificate

建议放在当前 `ACPC as a fixed-candidate stability condition` 后，或者替换当前 sampled-pool theorem 的正文版本，appendix 给 proof。

#### Theorem skeleton

```latex
\begin{theorem}[ACPC radius--margin certificate]\label{thm:radius-margin-certificate}
Let $\mathcal A=\{\mathbf a^1,\ldots,\mathbf a^K\}\sim q^K$ be a once-sampled candidate pool with deterministic tie-breaking. For a clean history $h$ and matched perturbed history $\tilde h$, define
\[
D_j=d_H\!\left(G_{\mathbf a^j}(E_\theta(h)),G_{\mathbf a^j}(E_\theta(\tilde h))\right).
\]
Assume the planner cost $J$ is locally $L_J$-Lipschitz on the evaluated projected-rollout neighborhood and that for a single sampled candidate
\[
\Pr_{\mathbf a\sim q}[D(\mathbf a)>\epsilon]\le \alpha.
\]
Let $\Delta_{\mathcal A}=C_h(\mathbf a^{(2)},g)-C_h(\mathbf a^{(1)},g)$ be the clean top-1/top-2 candidate margin. Then
\[
\Pr_{\mathcal A}\!\left[
\arg\min_j C_h(\mathbf a^j,g)
\ne
\arg\min_j C_{\tilde h}(\mathbf a^j,g)
\right]
\le
K\alpha + \Pr_{\mathcal A}[\Delta_{\mathcal A}\le 2L_J\epsilon].
\]
If additionally the task-grounded discriminability failure event satisfies
\[
\Pr\!\left[M_{\mathrm{diff}}\le R_\sigma+\delta_m\right]\le \gamma,
\]
then the selective-ACPC diagnostic failure risk is controlled by the union of planning-instability risk and discriminability-failure risk,
\[
\Pr[\mathrm{selective\ failure}]
\le
K\alpha + \Pr_{\mathcal A}[\Delta_{\mathcal A}\le 2L_J\epsilon] + \gamma.
\]
\end{theorem}
```

#### Proof skeleton

Appendix proof should be short:

1. If all `D_j <= epsilon`, Lipschitz gives all candidate cost drifts `<= L_J epsilon`.
2. If clean top margin `Delta_A > 2 L_J epsilon`, then no non-top candidate can overtake the clean top candidate under perturbation.
3. Probability at least one candidate violates radius bound is at most `K alpha` by union bound.
4. Remaining failure is clean margin tail.
5. Add discriminability failure event by another union bound.

### 4.5 新增 corollary：robust interval definition

新增一个 corollary 或 definition，避免 “robust interval” 只停留在口头。

```latex
\begin{definition}[Matched-perturbation robust interval]\label{def:robust-interval}
For a checkpoint family indexed by training noise level $\rho$ and evaluated at observation noise $\sigma$, define the empirical radius--margin pass event
\[
\widehat{\mathcal G}_{\rho,\sigma}
=
\left\{
\widehat Q_{1-\alpha}(R_\sigma) \le \widehat r_{\mathrm{margin}}
\right\}
\cap
\left\{
\widehat{\mathrm{SMPR}}_{\rho,\sigma}\ge 1-\widehat\gamma
\right\}.
\]
The robust interval is the set of checkpoint indices $\rho$ for which $\widehat{\mathcal G}_{\rho,\sigma}$ holds. When planner margin traces are unavailable, $\widehat r_{\mathrm{margin}}$ is reported as a rollout-space proxy and not as a planner-margin certificate.
\end{definition}
```

更 concrete 的 empirical version：

```latex
\widehat\Gamma_{\rho,\sigma}
=\widehat Q_{\beta}(\Delta_{\mathcal A})
-2\widehat Q_{1-\alpha}(|C_h-C_{\tilde h}|),
```

其中 `Gamma > 0` 表示 empirical cost-drift certificate pass。这个版本不需要估计 `L_J`，更适合实际分析。

### 4.6 扩展 Gaussian sensitivity：从 expectation 到 quantile/noise threshold

当前已有 expectation-level proposition。新增一段 local tail/quantile interpretation。

建议文本：

```latex
Under the same local linearization, let
\[
A_{h,\mathbf a}=J_{G_{\mathbf a}}(E_\theta(o))J_E(o).
\]
Then
\[
R_\sigma^2(h,\mathbf a) \approx \|A_{h,\mathbf a}\xi\|_2^2,
\quad \xi\sim\mathcal N(0,\sigma^2 I).
\]
If $A_{h,\mathbf a}A_{h,\mathbf a}^\top$ has eigenvalues $\lambda_i$, then
\[
R_\sigma^2/\sigma^2 \approx \sum_i \lambda_i \chi_i^2.
\]
Therefore the local radius quantile is approximately
\[
Q_{1-\alpha}(R_\sigma)
\approx
\sigma\sqrt{Q_{1-\alpha}\!\left(\sum_i\lambda_i\chi_i^2\right)}.
\]
Combining this with the fixed-pool margin condition gives a local tolerable noise scale
\[
\sigma^*(h,\mathbf a,\mathcal A)
=
\frac{\Delta_{\mathcal A}}
{2L_J\sqrt{Q_{1-\alpha}(\sum_i\lambda_i\chi_i^2)}}.
\]
```

Important wording:

```latex
This expression is a local diagnostic approximation, not a global robustness theorem. It clarifies why matched Gaussian training can lower ATR: it may reduce encoder-side neighborhood crossing, rollout-side amplification, or both.
```

### 4.7 Update limitations paragraph

在 limitation/caveat 段落中增加：

```latex
The radius--margin certificate is a fixed-pool diagnostic statement. It does not guarantee the adaptive CEM sampling distribution, repeated replanning, or closed-loop environment feedback. The Gaussian quantile expression is local and depends on the validity of the linearization. Empirical robust intervals are therefore diagnostic predictions to be checked against closed-loop endpoints, not replacements for behavioral evaluation.
```

---

## 5. 新增实验分析总览

新增实验不应变成大规模 method comparison。目标是佐证 direction 1 的理论统一：certificate 是否能定位已有 matched-Gaussian robustness plateau。

建议新增一个 subsection：

```latex
\subsection{Diagnostic validation of the radius--margin certificate}\label{sec:exp-radius-margin}
```

该 subsection 放在当前 `ATR/SMPR selective-ACPC diagnostics` 后面。

核心问题：

1. ATR/SMPR 是否在 full sweep 上形成和 closed-loop recovery plateau 对齐的 interval？
2. same-state noise radius tail 是否被 planner/discriminability margins 压住？
3. ATR-only / SMPR-only 是否不如 joint radius--margin gate 稳定？

---

## 6. 新增实验 A：Full-sweep robust interval overlay

### 实验目标

验证 radius--margin diagnostic gate 是否能在 `stdmax in {0.00,...,0.08}` sweep 上定位 closed-loop Gaussian recovery plateau。

### 核心假设

Recovered checkpoints 应满足：

```text
ATR / same-state radius tail low
AND
SMPR / task-grounded margin pass high
```

而 base checkpoint 或 fragile checkpoint 至少违反其中一侧。

### 最小可行版本

如果当前只计算了 base 和 std0.08 的 ATR/SMPR，需要补算所有 training stdmax checkpoint：

```text
task in {TwoRoom, PushT, Reacher, Cube}
train_seed in {3072, 3073, 3074}
train_stdmax in {0.00, 0.01, ..., 0.08}
eval_noise_sigma = 0.08 primary endpoint
```

### 指标

每个 `(task, train_seed, train_stdmax)` 输出：

1. `score_clean`
2. `score_obs_sigma_0p08`
3. `ATR_q90`
4. `SMPR_margin0`
5. optional `radius_q90`, `radius_q95`
6. optional `certificate_pass = ATR_pass AND SMPR_pass`

### 推荐 robust plateau 标注

为了不把 metric gate 和 behavior gate 混在一起，报告两个对象：

1. **Behavioral plateau**：只由 closed-loop score 定义，用作对照。
2. **Diagnostic interval**：只由 ATR/SMPR 或 radius--margin gap 定义，用作预测/解释。

Behavioral plateau 可以定义为：

```text
score_obs_sigma_0p08(stdmax) >= score_base_sigma_0p08 + 0.8 * (best_score_obs_sigma_0p08 - score_base_sigma_0p08)
```

并且 clean score 不应比 base clean score 下降超过一个小 tolerance，例如 5 points。若使用其他阈值，必须在 caption 中说明。

Diagnostic gate 推荐优先使用：

```text
SMPR >= 0.95
AND
ATR_q90 below task-calibrated or margin-derived threshold
```

更推荐的 margin-derived gate：

```text
certificate_gap = q_beta(clean_top1_top2_margin) - 2 * q_{1-alpha}(abs_cost_drift)
certificate_pass = certificate_gap > 0 AND SMPR >= 0.95
```

其中建议 `alpha=0.10`，`beta=0.10` 或 `0.20`；caption 中明确这些只是 empirical reporting choices。

### 图表输出

新增 figure：

```text
paper1/figures/fig_radius_margin_interval_overlay.png
```

建议图内容：

- x-axis: training `stdmax`
- line 1: closed-loop score under observation noise sigma=0.08
- line 2 or secondary panel: ATR / radius quantile
- line 3 or secondary panel: SMPR
- shaded region: diagnostic pass interval
- markers: behavioral plateau points

如果一张图太挤，可以四个 task 分 panel。

### 论文中要写的解释

```latex
The overlay is not used to tune a new checkpoint selector. It asks whether the radius--margin event derived from the theory aligns with the broad Gaussian recovery plateau already observed in closed-loop evaluation.
```

---

## 7. 新增实验 B：Radius--margin overlap plot

### 实验目标

直接可视化 theorem 中的核心条件：same-state noise radius/cost drift tail 是否小于 planner or discriminability margin。

### 优先版本：planner cost margin available

如果可以从 MPC candidate pool 或 diagnostic rollout 中取到 candidate costs，计算：

```text
cost_drift_j = |C_h(a_j,g) - C_tilde_h(a_j,g)|
clean_margin = C_h(a_top2,g) - C_h(a_top1,g)
```

聚合：

```text
q95_cost_drift = Q_0.95(cost_drift_j)
q10_clean_margin = Q_0.10(clean_margin)
certificate_gap = q10_clean_margin - 2 * q95_cost_drift
```

若 `certificate_gap > 0`，表示 empirical cost-drift margin pass。

### fallback 版本：planner costs unavailable

如果没有 candidate cost trace，使用 rollout-space proxy，必须标注为 proxy：

```text
same_radius = d_H(G_a(E(h)), G_a(E(tilde_h)))
diff_margin = d(Psi(z_i), Psi(z_j)) for task-grounded near-boundary different-state pairs
proxy_gap = Q_0.10(diff_margin) - Q_0.90(same_radius)
```

不能把 proxy_gap 写成 planner-margin certificate，只能写成 discriminability-side certificate evidence。

### 图表输出

新增 figure：

```text
paper1/figures/fig_radius_margin_overlap.png
```

建议每个 task 至少展示 base vs std0.08：

- distribution of `2 * cost_drift` 或 `same_radius`
- distribution of `clean_margin` 或 `diff_margin`
- 标出 quantile lines

预期现象：

- base：radius/cost drift tail 与 margin 明显重叠；
- robust endpoint：radius/cost drift tail 被 margin 分布压住；
- Cube 可能是 boundary case：diagnostic repair strong，但 closed-loop recovery weaker，应诚实解释。

---

## 8. 新增实验 C：ATR-only vs SMPR-only vs joint gate

### 实验目标

证明方向一理论统一的必要性：robustness 不是 same-state consistency alone，而是 radius + margin joint event。

### 指标定义

对每个 `(task, seed, stdmax)`：

```text
ATR_pass = ATR_q90 <= threshold_R
SMPR_pass = SMPR >= threshold_M
Joint_pass = ATR_pass AND SMPR_pass
```

若有 certificate_gap：

```text
RadiusMargin_pass = certificate_gap > 0 AND SMPR >= threshold_M
```

### 对照对象

Behavioral robust label 使用 closed-loop endpoint 的 high-score plateau 标注。注意这只是验证 alignment，不要把 label 用来调参后再 claim prediction。

### 表格输出

新增 table：

```latex
\begin{table}[H]
\centering
\caption{Diagnostic gate alignment with the Gaussian recovery plateau. ATR-only can pass collapsed or non-discriminative cases, SMPR-only can miss visually unstable rollouts, while the joint radius--margin gate matches the selective-ACPC theory.}
\label{tab:radius-margin-gate-ablation}
...
\end{table}
```

表格字段建议：

```text
Task
Criterion: ATR-only / SMPR-only / ATR+SMPR / cost-margin+SMPR if available
Predicted robust stdmax range
Behavioral high-score range
False positives
False negatives
```

如果数据不够支持 precision/recall，就报告 interval overlap 和 qualitative alignment，不要强行造 F1。

---

## 9. Optional 实验 D：candidate flip / empirical cost drift

### 目标

更直接地验证 theorem 的 fixed-pool instability link。

### 计算

对同一候选池 `A`，分别在 clean 和 noisy branch 下计算候选 cost 和 top candidate：

```text
top_clean = argmin_j C_h(a_j,g)
top_noisy = argmin_j C_tilde_h(a_j,g)
flip = int(top_clean != top_noisy)
```

同时记录：

```text
clean_margin = C_h(a_top2,g)-C_h(a_top1,g)
max_cost_drift = max_j |C_h(a_j,g)-C_tilde_h(a_j,g)|
```

理论预测：

```text
flip can occur when clean_margin <= 2 * max_cost_drift
```

### 输出

可以新增 appendix table：

```text
Task | Checkpoint | flip_rate | Pr[margin <= 2 drift] | ATR_q90 | SMPR
```

注意：如果 candidate pool 与真实 CEM 过程不一致，必须说是 fixed sampled-pool diagnostic，不是 full MPC guarantee。

---

## 10. Optional 实验 E：Gaussian sensitivity / Jacobian proxy

### 目标

支撑 Gaussian quantile theory：ATR 下降是否对应 encoder--predictor composition sensitivity 下降。

### 最小版本

对少量 sampled states/actions，用 Hutchinson/JVP 估计：

```text
|| J_{G_a}(E(o)) J_E(o) ||_F^2
```

或者估计方向导数：

```text
E_v || (G(E(o + eps v)) - G(E(o))) / eps ||^2
```

其中 `v` 是标准高斯方向。

### 输出

appendix scatter：

```text
x = estimated local sensitivity
 y = ATR_q90 or same_radius_q90
```

预期：robust endpoint 的 local sensitivity 低于 base；若不成立，要说明 ATR 可能主要来自 nonlocal neighborhood crossing repair，而不是 local Jacobian shrinkage。

---

## 11. 建议新增/修改的文件

优先修改：

```text
paper1/main.tex
```

可能新增：

```text
paper1/figures/fig_radius_margin_interval_overlay.png
paper1/figures/fig_radius_margin_overlap.png
paper1/results/radius_margin_certificate_summary.csv
paper1/results/radius_margin_gate_ablation.csv
```

如果已有脚本目录，请扩展已有脚本；如果没有，可新增：

```text
paper1/scripts/compute_radius_margin_certificate.py
paper1/scripts/plot_radius_margin_certificate.py
```

脚本必须支持：

```bash
python paper1/scripts/compute_radius_margin_certificate.py --task all --seeds 3072 3073 3074 --stdmax all
python paper1/scripts/plot_radius_margin_certificate.py --input paper1/results/radius_margin_certificate_summary.csv
```

如果真实 repo 中已有不同路径/命名，以现有结构为准，不要强行重复造目录。

---

## 12. 数据字段规范

`radius_margin_certificate_summary.csv` 建议包含：

```text
task
train_seed
train_stdmax
checkpoint_path_or_id
eval_sigma
score_clean
score_obs_sigma_0p08
atr_q90
smpr_margin0
same_radius_q90
same_radius_q95
cost_drift_q90
cost_drift_q95
clean_margin_q10
clean_margin_q20
certificate_gap_q10_q95
certificate_pass
behavioral_plateau_label
notes
```

如果某字段无法计算，写空值并在 `notes` 说明原因，不要填假值。

`radius_margin_gate_ablation.csv` 建议包含：

```text
task
criterion
thresholds
predicted_robust_stdmax_range
behavioral_plateau_range
false_positive_stdmax
false_negative_stdmax
notes
```

---

## 13. 论文新增 subsection 的推荐写法

可在实验中新增：

```latex
\subsection{Diagnostic validation of the radius--margin certificate}\label{sec:exp-radius-margin}
```

推荐正文 skeleton：

```latex
The radius--margin theory predicts that a checkpoint should be diagnostically robust when the high-tail same-state predictive radius is contained within the relevant planner or task-discriminability margin. We therefore evaluate the full Gaussian training sweep using two training-free checks. First, we overlay the metric-derived pass interval on the closed-loop Gaussian recovery plateau. Second, we compare the distribution of same-state cost drift or rollout radius against clean planner or task-grounded margins. These analyses do not introduce a new checkpoint-selection method; they test whether the theory-derived diagnostic event explains the broad recovery interval already visible in closed-loop evaluation.
```

If using cost margins:

```latex
When candidate-cost traces are available, we use the empirical gap
\[
\widehat\Gamma
=\widehat Q_{0.10}(\Delta_{\mathcal A})
-2\widehat Q_{0.95}(|C_h-C_{\tilde h}|).
\]
A positive gap is the empirical counterpart of the fixed-pool margin condition. When candidate costs are unavailable, we report the rollout-space proxy gap separately and do not call it a planner-margin certificate.
```

For gate comparison:

```latex
The ablation compares ATR-only, SMPR-only, and the joint radius--margin gate. ATR-only is insufficient because collapse can reduce same-state disagreement. SMPR-only is insufficient because a model may keep task-grounded pairs separated while still amplifying same-state visual perturbations through the action-conditioned rollout. The joint event is the diagnostic object implied by selective ACPC.
```

---

## 14. Caption requirements

Every new figure/table must make the claim boundary explicit.

### Overlay figure caption

Must say:

```text
The shaded diagnostic interval is derived from ATR/SMPR or radius--margin statistics only; the behavioral plateau is shown after the fact for alignment. The plot is a diagnostic validation, not a new checkpoint-selection algorithm.
```

### Overlap figure caption

Must say:

```text
A reduced overlap between same-state radius/cost-drift tails and margin distributions supports the radius--margin condition. If planner costs are unavailable, this is a rollout-space discriminability proxy rather than a planner-margin certificate.
```

### Gate ablation table caption

Must say:

```text
The joint gate reflects the selective-ACPC theory. ATR-only and SMPR-only are reported to show why the two diagnostic sides are not interchangeable.
```

---

## 15. Acceptance criteria

Codex execution is complete only if all items below are satisfied:

### Theory

- [ ] `paper1/main.tex` defines same-state predictive radius `R_sigma`.
- [ ] `paper1/main.tex` defines planner margin `Delta_A` and discriminability margin/failure event.
- [ ] A theorem named similar to `ACPC radius--margin certificate` is added.
- [ ] Proof is included in appendix or near theorem.
- [ ] Robust interval is explicitly defined.
- [ ] Gaussian sensitivity is extended from expectation to quantile / `sigma_star` interpretation.
- [ ] Limitations clearly state fixed-pool diagnostic only, not full closed-loop guarantee.

### Experiments / analysis

- [ ] Full sweep ATR/SMPR or radius-margin summary is computed if raw data/checkpoints are available.
- [ ] If candidate-cost traces are available, empirical cost-drift margin gap is computed.
- [ ] If candidate-cost traces are not available, rollout-space proxy gap is computed and explicitly labeled proxy.
- [ ] At least one overlay figure or table shows diagnostic interval vs behavioral plateau.
- [ ] At least one radius--margin overlap plot/table is added, or the absence of raw traces is documented.
- [ ] ATR-only / SMPR-only / joint gate comparison is added if full sweep diagnostics are available.
- [ ] No new training is required.

### Reproducibility

- [ ] New scripts have deterministic seeds where sampling is involved.
- [ ] New CSV outputs are saved under `paper1/results/` or existing results path.
- [ ] New figures are saved under `paper1/figures/` or existing figure path.
- [ ] `paper1/main.tex` compiles.
- [ ] Generated auxiliary LaTeX files are not committed.
- [ ] If data are missing, the paper does not contain fabricated numbers; instead, add a clear TODO or omit that analysis.

---

## 16. Suggested implementation order

1. Inspect `paper1/main.tex` and locate current ACPC/theory/experiment sections.
2. Add theoretical definitions and theorem first; compile.
3. Add Gaussian quantile sensitivity extension; compile.
4. Add robust interval definition; compile.
5. Search within repo for existing ATR/SMPR scripts/logs/checkpoint result CSVs.
6. Implement full-sweep diagnostic summary if data exist.
7. Implement overlay and overlap plots.
8. Add experimental subsection and captions.
9. Add gate ablation table only if enough full-sweep diagnostics exist.
10. Recompile paper and fix references.
11. Keep final claims conservative.

---

## 17. Reviewer-risk checklist

Before final commit, check these likely reviewer questions:

1. **Is this just post-hoc explanation?**  
   Response: include metric-only diagnostic interval overlay; if possible, seed 3072 calibration and held-out seed 3073/3074 validation.

2. **Does low ATR imply collapse?**  
   Response: emphasize SMPR and show ATR-only vs joint gate.

3. **Does theorem guarantee closed-loop CEM?**  
   Response: no; fixed-pool only. State this in theorem text, caption, and limitations.

4. **Why Gaussian?**  
   Response: Gaussian gives local sensitivity / weighted chi-square quantile interpretation; unseen blur/resize remain scope checks only.

5. **Why not train with the certificate?**  
   Response: out of scope. This paper is a fixed-checkpoint diagnostic study; training objectives are future method work.

6. **Is robust interval universal?**  
   Response: no. It is task/checkpoint/noise setting dependent and empirically calibrated.

---

## 18. Recommended final contribution sentence

Use a sentence close to this in introduction or conclusion:

```latex
The resulting view is a radius--margin diagnostic theory for JEPA world-model robustness: matched visual perturbations should remain inside a small action-conditioned predictive tube, while task-grounded distinctions should remain outside that tube. ATR estimates the tube radius tail; SMPR estimates the margin-preservation side. Their conjunction, not either metric alone, defines the empirical robust interval studied here.
```

---

## 19. Non-goals to explicitly preserve

Do not add claims that require new baselines or method comparison. In particular, do not compare against DrQ/DrQ-v2/SODA/Dreamer/TD-MPC unless the paper already has those experiments. The current paper can mention them as future method-comparison axes, but direction one is theory/diagnostic deepening, not algorithmic competition.

---

## 20. Minimal fallback if experiment data are insufficient

If full-sweep ATR/SMPR or candidate-cost traces are unavailable and cannot be recomputed quickly, still do the theory rewrite and add a smaller diagnostic validation using existing base vs std0.08 ATR/SMPR table:

1. State that full interval validation is future work or appendix TODO.
2. Add only a conceptual radius--margin interpretation of the existing endpoint table.
3. Do not claim predictive robust interval.
4. Claim only: existing recovered endpoints satisfy the expected radius--margin direction.

Preferred fallback wording:

```latex
Because the available diagnostic table covers the base and recovered endpoint rather than every sweep checkpoint, we treat this as endpoint evidence for the radius--margin mechanism rather than a prospective interval-identification result.
```

Do not overclaim if only endpoint diagnostics are available.
