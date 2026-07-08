# paper1 Codex 执行说明：整改建议与闭式 Gaussian ACPC sensitivity 加深

目标仓库：`qun-team/wm_exp`  
目标分支：`ag/dev`  
目标范围：只处理 `paper1` 相关内容，尤其是 `paper1/main.tex`、`paper1` 下的图表/脚本/附录；不要关联仓库中其他论文或文档来扩展叙事。  
建议文档路径：`paper1/docs/codex_rectification_and_gaussian_sensitivity_plan.md`

---

## 0. 总体目标

当前 `paper1` 已经补上了两块关键内容：

1. `radius--margin certificate`：把 ATR/SMPR 从诊断指标升级为固定候选池、匹配扰动分布下的 diagnostic certificate。
2. `prospective robust interval validation`：用训练外的 diagnostic interval 与 closed-loop Gaussian recovery plateau 做对齐验证。

下一步修改不要把论文从 diagnostic paper 改成 method paper，也不要把现有结果包装成 calibrated closed-loop guarantee。建议目标是：

- 重新整理整改建议：避免把 F1/IoU 当成跨任务、跨敏感度的“通用优劣判据”；
- 加强方向二：把已有的 local Gaussian ACPC sensitivity 从理论动机段升级为可检验的机制分析；
- 让 radius--margin certificate 与 Gaussian sensitivity 形成两层理论：
  - Layer 1：什么时候进入 matched-Gaussian diagnostic robust interval；
  - Layer 2：为什么某些 checkpoint 的 same-state predictive radius 会收缩。

---

## 1. 需要修正的判断口径：不要把 F1/IoU 当作跨任务通用判据

### 1.1 背景问题

当前 `paper1` 里报告了一个 held-out ATR/SMPR interval audit，使用 seed 3072 校准阈值，并在 seeds 3073/3074 上报告 per-task precision、recall、F1、mean interval IoU 等结果。

这个结果可以保留，但不能作为“哪个任务更好”“哪个诊断更通用”“是否存在通用 robust interval 判据”的主要证据。原因是：

1. 四个任务的视觉敏感度和动力学敏感度不同。
   - PushT 接触/几何精度敏感；
   - Reacher 对目标/末端关系和局部视觉扰动敏感；
   - TwoRoom 更偏拓扑/门区状态；
   - Cube 存在 3D 姿态和目标关系的混合敏感性。
2. 相同间隔的 noise sweep，例如 `stdmax = 0.01, ..., 0.08`，在不同任务上不是同等语义分辨率。
   - 对高敏感任务，一个 0.01 step 可能已经跨过行为边界；
   - 对低敏感或宽 plateau 任务，多个相邻 step 可能行为等价；
   - 因此 interval start/end 的离散边界会带来量化误差。
3. F1 和 IoU 默认把每个 grid point 当成同等可比较的离散分类点，但这里每个点背后的任务敏感度、margin 分布、closed-loop nonlinear response 不同。
4. 如果把 F1/IoU 作为主指标，容易给 reviewer 一个错误印象：论文在追求一个跨任务通用 checkpoint selector 或 universal threshold。当前 paper1 的定位应是 diagnostic certificate，而不是 checkpoint-ranking method。

### 1.2 新的建议口径

把 F1/IoU 降级为“离散 sweep 下的辅助一致性描述”，不要作为核心判断。主文和附录建议改成以下口径：

> Because the Gaussian sweep is discretized at uniform `stdmax` increments while tasks have different visual and dynamical sensitivities, we do not interpret F1 or interval IoU as task-universal selection quality. They are reported only as coarse descriptive summaries. The primary validation is boundary-aware, task-conditioned alignment: whether the diagnostic interval crosses near the behavioral plateau onset, remains stable through the plateau, and fails in interpretable ways when task sensitivity or fixed-pool margin proxies are conservative.

中文理解：

> 因为 noise sweep 是等间隔离散网格，但不同任务对噪声的敏感度不一样，所以 F1/IoU 不应被当作通用优劣指标。它们只能作为离散网格上的粗粒度一致性描述。核心应看每个任务内部的 boundary-aware alignment：诊断区间是否在行为 plateau 开始附近越过阈值，是否在 plateau 内保持稳定，错误是否能由任务敏感度或 fixed-pool proxy 保守性解释。

---

## 2. 整改建议一：把 interval validation 改成 boundary-aware task-conditioned audit

### 2.1 需要修改的位置

建议修改：

- `paper1/main.tex` 的 `Diagnostic validation of the radius--margin certificate` 小节；
- `Discussion and limitations` 中关于 empirical diagnostic interval 的措辞；
- Appendix 的 `Radius--margin parameter interpretation` 表格和说明。

### 2.2 主文中需要保留的核心 claim

保留：

- behavioral plateau 由 closed-loop observation-only `sigma=0.08` score 定义；
- diagnostic proxy 独立由 fixed-pool margin summaries 定义；
- radius--margin proxy 与 behavioral plateau 大体对齐，但存在 task-specific mismatch；
- 这些 mismatch 是 diagnostic limitation，不是失败，需要解释。

删除或弱化：

- 不要把 `precision 1.00 / recall 0.92 / F1 0.96 / IoU 0.92` 放在主文当作核心总结。
- 不要说“ATR/SMPR threshold 识别 held-out robust intervals”时显得像 checkpoint selector。
- 不要把 F1/IoU 解释成通用 predictive ability。

建议替换为：

> As a separate no-retraining audit, ATR/SMPR thresholds calibrated on seed 3072 produce task-conditioned held-out interval predictions on seeds 3073/3074. We treat this as a boundary-aware diagnostic sanity check rather than a universal interval-classification score: the sweep grid is discrete and equally spaced in `stdmax`, while task sensitivity to visual noise is not equally spaced. Therefore, aggregate F1/IoU are reported only as coarse summaries in the appendix. The main evidence is per-task boundary alignment and failure interpretation.

### 2.3 新增/修改附录表：boundary-aware interval alignment

新增一个 appendix 表，替代或补充单句 F1/IoU。表格不需要证明哪个指标更好，而是解释每个任务的诊断边界与行为边界关系。

建议表格字段：

| Task | Behavioral plateau | Diagnostic interval | Start boundary error | End boundary error | Within one-grid tolerance? | Interpretation |
|---|---|---|---|---|---|---|
| TwoRoom | `0.01--0.08` | `0.02--0.08` | +1 grid step | 0 | yes | proxy misses first plateau point; acceptable under discrete sweep uncertainty |
| PushT | `0.03--0.08` | `0.02--0.08` | -1 grid step | 0 | yes | proxy fires one step early; acceptable early warning |
| Reacher | `0.02--0.08` | `0.04--0.08` | +2 grid steps | 0 | partial/no | conservative fixed-pool proxy; Reacher behavior recovers earlier than cost-margin proxy |
| Cube | `0.03--0.08` | `0.03--0.08` | 0 | 0 | yes | proxy and plateau align |

说明：

- `+1` 表示 diagnostic interval 比 behavioral plateau 晚一个 grid step；
- `-1` 表示 diagnostic interval 比 behavioral plateau 早一个 grid step；
- 一个 grid step 对应 `stdmax = 0.01`；
- 对于等间隔 noise sweep，这种 one-step miss 不应被解释为强错误；
- Reacher 的 +2 step 应被明确解释为 conservative proxy 或任务敏感度差异，而不是掩盖。

### 2.4 新增 tolerance 规则

在 appendix 里定义一个 `one-grid boundary tolerance`：

\[
\tau_{\mathrm{grid}} = 0.01.
\]

对于 interval onset：

\[
\mathrm{BoundaryError}_{\mathrm{start}}
= \rho^{\mathrm{diag}}_{\mathrm{start}}
- \rho^{\mathrm{beh}}_{\mathrm{start}}.
\]

解释规则：

- `|BoundaryError_start| <= 0.01`：boundary-aligned under grid uncertainty；
- `BoundaryError_start < -0.01`：diagnostic fires materially earlier；
- `BoundaryError_start > 0.01`：diagnostic is conservative / late；
- end boundary 同理，但当前多数任务 end 都到 0.08，不应过度分析 end error。

这比 F1/IoU 更符合 paper1 的任务异质性设定。

---

## 3. 整改建议二：把 F1/IoU 放到 appendix，并明确其用途

### 3.1 如果保留 F1/IoU，必须加 caveat

可以保留 F1/IoU，但只作为附录描述性 summary。建议文字：

> We include grid-point F1 and interval IoU only as descriptive summaries of the discretized sweep. They should not be interpreted as task-universal calibration metrics, because equal `stdmax` increments do not imply equal behavioral or diagnostic sensitivity across tasks. The primary interpretation is the boundary-aware per-task table.

### 3.2 更推荐的指标层级

建议把诊断验证指标分成三层：

1. **Primary：per-task boundary alignment**
   - diagnostic onset 是否接近 behavioral plateau onset；
   - 是否在 plateau 内持续 positive；
   - mismatch 是否可解释。
2. **Secondary：within-task stability**
   - interval 内 diagnostic proxy 是否稳定；
   - ATR 是否在 plateau 内保持低值；
   - SMPR 是否保持 near-one。
3. **Tertiary：grid-point F1/IoU**
   - 只做 compact summary；
   - 不用于跨任务优劣判断。

---

## 4. 整改建议三：增加 fixed-pool top-1 agreement audit，直接支撑 theorem 中间命题

### 4.1 动机

radius--margin theorem 的核心不是 F1/IoU，而是：当 same-state predictive radius 或 cost drift 小于 planner margin 时，固定候选池的 top-1 candidate 应该稳定。

当前实验用了 cost-margin proxy：

\[
\widehat\Gamma_{\rho,0.08}
= \widehat Q_{0.50}(\Delta_{\mathcal A})
-2\widehat Q_{0.90}(|C_h-C_{\tilde h}|).
\]

这很有用，但还是 proxy。更直接的中间量是：

\[
\mathrm{Top1Agree}
=
\Pr\left[
\arg\min_j C_h(\mathbf a^j,g)
=
\arg\min_j C_{\tilde h}(\mathbf a^j,g)
\right].
\]

### 4.2 建议新增表/图

新增 appendix 表或小图：

| Task | stdmax | q90 cost drift | q50 clean margin | proxy gap | empirical Top1Agree | behavioral plateau? |
|---|---:|---:|---:|---:|---:|---|

最小版本只做四个任务的 base、plateau onset、std0.08 三个点即可。更完整版本做 full sweep。

### 4.3 预期解释

- robust plateau 内：Top1Agree 应整体更高；
- proxy-positive 区间：Top1Agree 应与 proxy gap 同向；
- Reacher 若行为 recovery 早于 proxy-positive，可检查 Top1Agree 是否也早恢复：
  - 如果 Top1Agree 早恢复，说明 q50/q90 proxy 过保守；
  - 如果 Top1Agree 也晚恢复，说明 Reacher behavior 对 top-1 fixed-pool proxy 不完全敏感，需在 limitation 里写明。

### 4.4 注意事项

不要 claim 这覆盖 adaptive CEM 或 closed-loop trajectory。它只验证 fixed-pool theorem 的中间机制。

建议措辞：

> This audit is closer to the fixed-pool theorem than closed-loop score, but it still does not cover adaptive CEM resampling or repeated replanning.

---

## 5. 整改建议四：补 q90/q95/q99 sensitivity，但不要转成概率证书

### 5.1 动机

现有 theorem 中有 `K alpha`，而 appendix 已经指出 `K=65` 时，直接把 q90 的 10% exceedance 当作 alpha 会导致 vacuous bound。这个 caveat 是正确的。

因此 q90/q95/q99 sensitivity 的目的不是生成 calibrated probability bound，而是说明 diagnostic interval 对 quantile choice 是否稳定。

### 5.2 建议新增表

| Task | quantile | diagnostic interval | boundary error | interpretation |
|---|---:|---|---:|---|
| TwoRoom | q90 | ... | ... | ... |
| TwoRoom | q95 | ... | ... | ... |
| TwoRoom | q99 | ... | ... | ... |

### 5.3 解释规则

- 若 q90/q95 给出相近 boundary，说明诊断对 reporting quantile 稳定；
- q99 可能更保守，不要视为失败；
- 若不同 quantile 差异大，说明 tail distribution 较重，正好支持“ATR 应作为 tail diagnostic，而非 mean score”。

---

## 6. 方向二：把闭式 Gaussian ACPC sensitivity 升级为机制理论

当前 main.tex 已经有 local Gaussian ACPC sensitivity 的核心公式：

\[
G_{\mathbf a}(E(o+\xi))-G_{\mathbf a}(E(o))
=
J_{G_{\mathbf a}}(E(o))J_E(o)\xi + R_\xi,
\]

以及：

\[
\mathbb E_\xi\left[
\|G_{\mathbf a}(E(o+\xi))-G_{\mathbf a}(E(o))\|_2^2
\right]
=
\sigma^2\|J_{G_{\mathbf a}}(E(o))J_E(o)\|_F^2+O(\sigma^3).
\]

但现在它主要是理论动机。建议将其升级为第二层理论贡献：

> Gaussian ACPC sensitivity explains why the radius side of the radius--margin certificate contracts under matched Gaussian training, and whether the contraction comes from encoder-side repair, rollout-side contraction, or alignment between encoder nuisance directions and rollout-amplifying directions.

---

## 7. 方向二理论新增内容

### 7.1 新增 Corollary：Local Gaussian radius quantile

在 proposition 后加一个 corollary：

```latex
\begin{corollary}[Local Gaussian ACPC radius quantile]
Under the local linearization in Proposition~\ref{prop:gaussian-sensitivity}, let
\[
A_{h,\mathbf a}=J_{G_{\mathbf a}}(E_\theta(o))J_E(o),
\]
and let \(\lambda_1,\ldots,\lambda_r\) be the nonzero eigenvalues of
\(A_{h,\mathbf a}A_{h,\mathbf a}^{\top}\). For isotropic pixel noise
\(\xi\sim\mathcal N(0,\sigma^2 I)\),
\[
R_\sigma^2(h,\mathbf a)
\approx
\sigma^2\sum_{i=1}^r \lambda_i \chi_i^2,
\]
and therefore
\[
r_{1-\alpha}(\sigma;h,\mathbf a)
=Q_{1-\alpha}(R_\sigma)
\approx
\sigma\sqrt{Q_{1-\alpha}\left(\sum_i\lambda_i\chi_i^2\right)}.
\]
\end{corollary}
```

解释文字：

> This corollary connects the empirical ATR quantile to the singular spectrum of the action-conditioned encoder--predictor Jacobian. ATR decreases when the spectrum of the composed map contracts along noise directions, not necessarily when the raw encoder distance is minimized.

### 7.2 新增 decomposition：encoder-side、rollout-side、alignment-side

定义：

\[
S_{\mathrm{comp}}(h,\mathbf a)
=\|J_{G_{\mathbf a}}(E(o))J_E(o)\|_F^2
=\operatorname{tr}(J_E^\top J_G^\top J_G J_E).
\]

以及两个辅助量：

\[
S_E(o)=\|J_E(o)\|_F^2,
\]

\[
S_G(z,\mathbf a)=\|J_{G_{\mathbf a}}(z)\|_F^2
\quad \text{or} \quad
\|J_{G_{\mathbf a}}(z)\|_2^2.
\]

加入解释：

- `encoder-side repair`：Gaussian training 降低 `S_E`，减少 pixel nuisance 进入 latent 的幅度；
- `rollout-side contraction`：Gaussian training 降低 `S_G` 或降低 predictor 对 residual nuisance latent directions 的放大；
- `alignment repair`：即使 `S_E` 或 `S_G` 单独不大变，`J_E` 的高敏方向也可能不再落入 `J_G` 的高增益方向，因此 `S_comp` 下降。

可以写一个 normalized alignment ratio：

\[
\kappa(h,\mathbf a)
=
\frac{\|J_GJ_E\|_F^2}
{\|J_G\|_F^2\|J_E\|_F^2+\varepsilon}.
\]

注意：`kappa` 只作诊断比例，不作严格角度证书。

### 7.3 新增 diagnostic noise tolerance proxy

已有公式：

\[
\sigma^*(h,\mathbf a,\mathcal A)
=
\frac{\Delta_{\mathcal A}}
{2L_J\sqrt{Q_{1-\alpha}(\sum_i\lambda_i\chi_i^2)}}.
\]

建议保留，但强调：

- 这是 local diagnostic approximation；
- 不要当作 global robustness theorem；
- 实验里如果无法估计 `L_J`，用 paired cost drift slope 作为 proxy：

\[
\widehat s_C
=Q_q(|C_h-C_{\tilde h}|)/\sigma,
\]

然后：

\[
\widehat\sigma^*_{\mathrm{cost}}
=
\widehat Q_{0.50}(\Delta_{\mathcal A})/(2\widehat s_C).
\]

这个 proxy 可以和 behavioral plateau onset 做 task-conditioned boundary comparison。

---

## 8. 方向二实验新增内容

以下实验都应是 training-free，不需要重新训练。

### 8.1 实验 A：JVP/Hutchinson composed sensitivity audit

**实验目标**  
验证 ATR contraction 是否由 composed action-conditioned encoder--predictor sensitivity 的下降解释。

**核心假设**  
noise-trained recovered checkpoints 的

\[
S_{\mathrm{comp}}=\mathbb E_v\|J_{G_{\mathbf a}}J_E v\|_2^2
\]

低于 no-noise base，并与 ATR 下降同向。

**最小可行版本**

- 任务：TwoRoom、PushT、Reacher、Cube；
- checkpoint：base 和 `stdmax=0.08`；
- seeds：优先 3072/3073/3074；如果资源紧，先做 seed 3072 proof-of-concept；
- 样本：每个 task/seed/checkpoint 取 128 或 256 个 clean history/action sequence；
- Hutchinson probe：每个样本 4 或 8 个 Gaussian/Rademacher probe；
- 输出：mean、median、q90 的 `S_comp`，并与 ATR base/std0.08 对照。

**计算方式**

对输入 observation/history `o` 采样 probe `v ~ N(0, I)` 或 Rademacher。用 JVP 计算：

\[
J_E(o)v,
\]

再通过 rollout map 的 JVP 得到：

\[
J_G(E(o))(J_E(o)v).
\]

记录平方范数：

\[
\|J_GJ_Ev\|_2^2.
\]

归一化建议：

- 使用 clean transition scale，与 ATR normalization 保持一致；
- 或同时报告 raw 和 normalized 两版。

**建议输出表**

| Task | Checkpoint | ATR q90 | `S_comp` mean | `S_comp` q90 | relative drop | interpretation |
|---|---|---:|---:|---:|---:|---|

**预期现象**

- base：ATR 高，`S_comp` 高；
- std0.08：ATR 低，`S_comp` 低；
- Cube 可能出现 `S_comp`/ATR 修复强但 closed-loop recovery 较弱，用来支持“radius 修复不等于完整 closed-loop guarantee”。

**失败判据**

- 如果 `S_comp` 与 ATR 完全不同向，说明当前 JVP 估计的 projection/map 与 ATR 计算不一致，或局部线性化不适用于当前 noise scale。

**失败定位**

1. 检查 `G_a` 是否与 ATR 使用同一 projection/horizon；
2. 检查 normalization 是否一致；
3. 对更小 sigma 做 small-sigma slope check；
4. 检查 JVP 是否只覆盖 encoder 而没有覆盖 predictor rollout。

---

### 8.2 实验 B：small-sigma slope check

**实验目标**  
验证 local Gaussian linearization 的适用范围。

理论预测：

\[
\mathbb E[R_\sigma^2]\propto \sigma^2.
\]

因此：

\[
\mathbb E[R_\sigma^2]/\sigma^2
\]

在小 sigma 区间应近似稳定。

**最小可行版本**

对 base 和 std0.08 checkpoint，使用：

\[
\sigma\in\{0.005,0.01,0.02,0.03\}
\]

或若代码中只方便使用已有评估：

\[
\sigma\in\{0.01,0.02,0.03\}.
\]

计算：

- mean `R_sigma^2 / sigma^2`；
- q90 `R_sigma / sigma`；
- ATR-like normalized q90。

**建议图**

- `fig_gaussian_small_sigma_slope.png`：x-axis 为 sigma，y-axis 为 `E[R^2]/sigma^2` 或 `q90(R)/sigma`；
- 每个任务单独一条 base/std0.08 对比曲线，或四任务 panel。

**预期现象**

- std0.08 checkpoint 曲线整体低于 base；
- 小 sigma 区间相对平坦；
- 若 sigma=0.03 已开始偏离，说明 local approximation 只在更小噪声下成立，应如实写 limitation。

---

### 8.3 实验 C：encoder-only vs rollout-composed sensitivity decomposition

**实验目标**  
强化 paper1 的核心论点：encoder invariance alone is incomplete；planning-relevant quantity 是 rollout-composed sensitivity。

**计算量**

1. Encoder finite/noise shift：

\[
D_E=\|E(o+\xi)-E(o)\|.
\]

2. Rollout-composed shift：

\[
D_G=\|G_a(E(o+\xi))-G_a(E(o))\|.
\]

3. Cost drift：

\[
D_C=|C_h-C_{\tilde h}|.
\]

如果能做 JVP，则同时报告：

\[
S_E=\mathbb E_v\|J_Ev\|^2,
\]

\[
S_{\mathrm{comp}}=\mathbb E_v\|J_GJ_Ev\|^2.
\]

**建议分析**

- 比较 `D_E` 与 ATR/behavior 的相关性；
- 比较 `D_G` 或 `S_comp` 与 ATR/behavior 的相关性；
- 预期 `D_G/S_comp` 比 raw encoder distance 更贴近 ATR 和 cost drift。

**输出表**

| Task | Checkpoint | encoder shift q90 | rollout shift q90 / ATR | cost drift q90 | behavior score | interpretation |
|---|---|---:|---:|---:|---:|---|

---

### 8.4 实验 D：local noise tolerance proxy `sigma*`

**实验目标**  
把 Gaussian sensitivity 接回 radius--margin interval：估计每个 checkpoint 的 local tolerable noise scale 是否跨过 evaluation sigma=0.08 附近。

**最小实现**

如果估计 weighted chi-square tail 太复杂，可以先用 cost drift slope proxy：

\[
\widehat s_C(\rho)
=\widehat Q_{0.90}(|C_h-C_{\tilde h}|)/0.08.
\]

然后：

\[
\widehat\sigma^*_{\mathrm{cost}}(\rho)
=
\widehat Q_{0.50}(\Delta_{\mathcal A})/(2\widehat s_C(\rho)).
\]

判断：

- 如果 `sigma*_cost >= 0.08`，说明 fixed-pool margin proxy 支持 robust endpoint；
- 如果低于 0.08，说明 proxy 不支持，可能对应 fragile 或 conservative mismatch。

**注意**

这仍然是 diagnostic proxy，不是 closed-loop guarantee。

---

## 9. 建议新增/修改文件

### 9.1 建议新增脚本

1. `paper1/scripts/diagnostics/summarize_interval_alignment.py`
   - 输入：现有 sweep score summary、diagnostic interval summary；
   - 输出：boundary-aware interval table、grid-step boundary error、optional F1/IoU appendix summary。

2. `paper1/scripts/diagnostics/compute_fixed_pool_top1_agreement.py`
   - 输入：fixed-pool clean/noisy candidate costs；
   - 输出：Top1Agree、q90 cost drift、q50/q10 clean margin、proxy gap。

3. `paper1/scripts/diagnostics/compute_gaussian_sensitivity_jvp.py`
   - 输入：checkpoint、task trajectories、recorded action sequences；
   - 输出：JVP/Hutchinson estimates for `S_comp` and optionally `S_E`。

4. `paper1/scripts/diagnostics/summarize_small_sigma_slope.py`
   - 输入：multi-sigma ACPC radius results；
   - 输出：`E[R^2]/sigma^2` 和 `q90(R)/sigma` 图表。

### 9.2 建议新增图表

1. `fig_radius_margin_boundary_alignment.png` 或 appendix table；
2. `fig_fixed_pool_top1_agreement.png`；
3. `fig_gaussian_sensitivity_vs_atr.png`；
4. `fig_small_sigma_slope.png`。

### 9.3 建议新增 LaTeX 表

1. `tab:appendix-boundary-aware-intervals`；
2. `tab:appendix-top1-agreement`；
3. `tab:appendix-gaussian-sensitivity-audit`。

---

## 10. 建议修改 main.tex 的具体写法

### 10.1 Introduction contribution 可微调

当前 contribution 2 已经很好。可以轻微扩充：

> Second, it derives an ACPC radius--margin diagnostic certificate and a local Gaussian sensitivity interpretation. The certificate links same-state predictive radius, planner candidate margins, and task-grounded discriminability margins; the sensitivity view explains radius contraction through the composed action-conditioned encoder--predictor Jacobian.

但如果版面紧，可以不在 introduction 加方向二，保留在 theory section。

### 10.2 Radius--margin validation 小节替换重点

把原先一句强 aggregate summary 改成：

> We report aggregate grid-point summaries only in the appendix because they are sensitive to the discretized sweep grid. The main validation is task-conditioned boundary alignment: whether the diagnostic interval enters within one grid step of the behavioral plateau onset and whether mismatches are interpretable under task-specific sensitivity or fixed-pool proxy conservatism.

### 10.3 Discussion 增加 limitation

建议加：

> The equal-spacing of the Gaussian training sweep should not be confused with equal task sensitivity. A one-step interval mismatch can reflect grid quantization rather than diagnostic failure, especially near plateau onset. We therefore use F1/IoU only as descriptive sweep summaries and interpret diagnostic validity through per-task boundary alignment and mechanism audits.

### 10.4 Gaussian sensitivity 小节增加机制解释

在现有 proposition 后加入：

> The product `J_G J_E` creates a mechanism-level distinction that encoder-only audits miss. Noise training may reduce encoder sensitivity, reduce rollout amplification, or change the alignment between encoder-sensitive pixel directions and rollout-amplifying latent directions. The empirical sensitivity audit in Appendix X estimates this composed quantity using JVP/Hutchinson probes.

---

## 11. Codex 执行优先级

### Priority 1：重写 interval validation 解释，不再把 F1/IoU 当主判据

必须完成：

- 修改主文对 held-out interval audit 的文字；
- 新增 boundary-aware interval table；
- F1/IoU 若保留，移到 appendix 并加 caveat。

验收标准：

- 文中没有把 F1/IoU 称为 universal 或 primary validation metric；
- 每个任务的 mismatch 都有解释；
- 明确写出 uniform noise sweep + task-specific sensitivity 会导致 boundary quantization error。

### Priority 2：增加 fixed-pool top-1 agreement audit

建议完成：

- 如果现有 fixed-pool cost traces 能复用，新增 Top1Agree 表；
- 如果没有逐候选 cost，只保留 TODO 并解释当前只能用 q50/q90 proxy。

验收标准：

- 能直接对应 theorem 的 top-1 stability statement；
- 不声称覆盖 adaptive CEM / repeated replanning。

### Priority 3：方向二理论补强

必须完成：

- 加 corollary：local Gaussian radius quantile；
- 加 decomposition：encoder-side / rollout-side / alignment-side；
- 明确 `sigma*` 是 local diagnostic approximation。

验收标准：

- 理论公式与 ATR/radius--margin certificate 有明确连接；
- 没有把 local linearization 写成 global robustness theorem。

### Priority 4：方向二最小实验

建议优先做：

1. JVP/Hutchinson `S_comp` vs ATR；
2. small-sigma slope check。

如果资源不足，先做 seed 3072 proof-of-concept，并把三训练 seed 结果列为 TODO。

验收标准：

- 所有新增实验 training-free；
- 输出至少一个表或图；
- 若结果不完全支持，要写成 limitation，不要隐藏。

---

## 12. 不要做的事

1. 不要把 paper1 改成 robust training method paper。
2. 不要 claim ATR/SMPR 是 universal checkpoint ranker。
3. 不要 claim radius--margin certificate 是 closed-loop guarantee。
4. 不要把 F1/IoU 当作跨任务通用基础或主要优劣判断。
5. 不要把 q90 ATR 直接代入 `K alpha` 生成概率 bound；`K=65` 时这会是 vacuous。
6. 不要只看 encoder invariance；方向二必须围绕 `J_G J_E` 的 action-conditioned composed sensitivity。

---

## 13. 最终期望论文叙事

修改后的 paper1 应形成如下叙事：

1. JEPA latent prediction alone does not guarantee visual robustness for control。
2. Robustness should be diagnosed after action-conditioned rollout, not only at encoder level。
3. Selective ACPC requires same-state predictive radius contraction and task-grounded margin preservation。
4. Radius--margin certificate explains when a fixed checkpoint is in a matched-Gaussian diagnostic robust interval。
5. Because tasks have different sensitivity and the sweep grid is discrete, interval validation should be boundary-aware and task-conditioned, not judged primarily by aggregate F1/IoU。
6. Local Gaussian ACPC sensitivity explains why the radius contracts: the key quantity is the composed Jacobian `J_G J_E`, with encoder-side, rollout-side, and alignment-side mechanisms。
7. Experiments remain diagnostic and training-free: closed-loop behavior is the authority, while ATR/SMPR, fixed-pool margin proxy, top-1 agreement, and Gaussian sensitivity audits explain mechanisms and limits。
