# Paper 1 理论部分整改规划

> 目标：把当前理论从“固定候选集上的 Lipschitz / margin sanity proof”升级为一个更完整的诊断理论链条：**ACPC 如何控制候选池稳定性、为什么 Gaussian 扰动下 ACPC 是 action-conditioned sensitivity 的估计、以及 semantic margin 如何作为 selective consistency 的有限样本证据**。
>
> 适用范围：`paper1/main.tex` 当前 `ag/dev` 版本。本文档给 Codex 执行用，优先做低风险、可快速合入的理论增强，不把论文扩展成完整方法论文。

## 0. 当前问题诊断

当前理论部分是正确的，但审稿观感偏薄：

1. **Planner link 太理想化。** 当前主要是固定候选集上：ACPC 小 → candidate-cost drift 小 → clean top-1/top-2 margin 足够大则 top-1 不变。这个链条基本是 Lipschitz + margin argument，容易被认为是常识性引理。
2. **与实际 CEM/MPC 存在缝隙。** LeWM 实际 inference 用 CEM / latent-space MPC，但当前理论明确不覆盖 CEM resampling、repeated replanning、closed-loop trajectory。
3. **Gaussian stressor 的机制解释不足。** 论文主轴是 Gaussian pixel noise，但理论没有解释为什么 ACPC basin contraction 对应 action-conditioned noise sensitivity 的下降。
4. **semantic margin 结果还缺一个理论证书表述。** 新版已经补了 task-semantic margin pass-rate，但可加有限样本下界或 calibration，让它更像“selective consistency certificate”。
5. **训练目标与 ACPC 的关系还未形式化。** 可加一个小 lemma 解释 prediction loss / target consistency / ACPC 的三角关系，但必须避免 claim target-view ablation 已解决。

整改原则：

- 不声称完整 CEM 收敛或 closed-loop robustness theorem。
- 不声称 ACPC rule 可以替代 closed-loop evaluation。
- 不把 semantic margin 说成 oracle semantic proof；用 “state-proxy semantic margin”。
- 主文只加 1.5–2 页理论增强，完整证明放 appendix。

## 1. 推荐的主文理论结构

建议把当前 `Action-Conditioned Predictive Consistency` section 调整为以下结构。

### 1.1 ACPC definition and selective pseudo-metric

保留当前定义、fixed-candidate pseudo-metric 和 collapse counterexample。只做少量压缩，给新定理腾空间。

保留重点：

- same-state clean/noisy views should agree after action-conditioned rollout；
- different action-/transition-/cost-relevant states must remain separable；
- low ACPC alone permits collapse。

### 1.2 From fixed candidates to sampled candidate-pool stability

新增一个 sampled-pool theorem，把固定候选集稳定性扩展到“一次 sampled candidate pool”的概率界。

这是最优先新增的理论内容。

#### Theorem 1: Sampled-pool ACPC stability

设一次 MPC/CEM 评估中从 proposal distribution `q` 采样了 \(K\) 个 candidate action sequences：

\[
\mathcal A = \{\mathbf a^1,\ldots,\mathbf a^K\}\sim q^K.
\]

对每个 candidate 定义 rollout discrepancy：

\[
D_j = d_H\!\left(
\Pi(F^{1:H}(E(h),\mathbf a^j)),
\Pi(F^{1:H}(E(\tilde h),\mathbf a^j))
\right).
\]

假设 cost readout \(J\) 在该 neighborhood 上是 \(L_J\)-Lipschitz，并且对单个 sampled candidate 有 tail bound：

\[
\Pr_{\mathbf a\sim q}[D(\mathbf a)>\epsilon] \le \delta.
\]

令 sampled pool 上 clean branch 的 top-1/top-2 margin 为：

\[
\Delta_{\mathcal A}
= C_h(\mathbf a^{(2)},g)-C_h(\mathbf a^{(1)},g).
\]

则一次 sampled candidate-pool top-1 选择在 clean/noisy branches 间发生 flip 的概率满足：

\[
\Pr_{\mathcal A}\!\left[
\arg\min_j C_h(\mathbf a^j,g)
\neq
\arg\min_j C_{\tilde h}(\mathbf a^j,g)
\right]
\le
K\delta + \Pr_{\mathcal A}[\Delta_{\mathcal A}\le 2L_J\epsilon].
\]

证明思路：

1. 若所有 \(D_j\le\epsilon\)，则由 Lipschitz 得所有 candidate cost drift 都不超过 \(L_J\epsilon\)。
2. 若 clean margin \(\Delta_{\mathcal A}>2L_J\epsilon\)，则固定候选集 top-1 不变。
3. flip 只能来自两个坏事件：至少一个 candidate 的 ACPC tail event，或 sampled clean margin 太小。
4. 用 union bound 得到 \(K\delta + \Pr[\Delta_{\mathcal A}\le2L_J\epsilon]\)。

主文解释：

- ACPC/PCC 估计 candidate-cost drift；
- CRA 估计 ranking stability；
- MAF 估计 large-margin flip failure；
- fixed-rule validation 中的 ACPC/PCC/CRA/MAF 不是任意指标组合，而是对应这个 bound 的经验 readouts。

注意措辞：

> This is not a guarantee for adaptive multi-round CEM; it controls the instability of a sampled candidate pool and exposes the two missing terms: ACPC tails and clean margin tails.

### 1.3 Receding-horizon calibration corollary

可加一个短 corollary，但要保守。目的不是证明完整 closed-loop robustness，而是显示 repeated replanning 需要额外控制。

#### Corollary 1: Replanning union bound

如果第 \(t\) 次 replanning 的 sampled-pool flip probability 上界为

\[
r_t = K_t\delta_t + \Pr[\Delta_{\mathcal A_t}\le 2L_J\epsilon_t],
\]

则在 \(T\) 次 replanning 中至少一次 action selection flip 的概率满足：

\[
\Pr[\exists t\le T: \text{flip at }t] \le \sum_{t=1}^T r_t.
\]

如果想再加环境动力学 deviation bound，可放 appendix，不建议主文展开。

主文解释：

- 该 corollary 不 claim CEM closed-loop guarantee；
- 它说明为什么 repeated replanning 仍需要 closed-loop evaluation；
- 它也解释为何本文坚持 closed-loop score 是最终 authority。

### 1.4 Gaussian local sensitivity decomposition

新增一个 proposition 解释 Gaussian ACPC 的机制意义。

#### Proposition 2: Local Gaussian ACPC sensitivity

令

\[
G_{\mathbf a}(z)=\Pi(F^{1:H}(z,\mathbf a)).
\]

对 pixel noise \(\xi\sim\mathcal N(0,\sigma^2 I)\)，设 \(\tilde o=o+\xi\)。若 \(E\) 和 \(G_{\mathbf a}\) 在局部可微且二阶项有界，则小噪声下：

\[
G_{\mathbf a}(E(o+\xi))-G_{\mathbf a}(E(o))
\approx
J_{G_{\mathbf a}}(E(o))J_E(o)\xi.
\]

因此：

\[
\mathbb E_{\xi}\left[
\left\|G_{\mathbf a}(E(o+\xi))-G_{\mathbf a}(E(o))\right\|_2^2
\right]
=
\sigma^2
\left\|J_{G_{\mathbf a}}(E(o))J_E(o)\right\|_F^2
+ O(\sigma^3).
\]

主文解释：

- Gaussian ACPC measures **action-conditioned encoder–predictor sensitivity**；
- encoder invariance alone is insufficient，因为关键对象是 \(J_{G_{\mathbf a}}J_E\)，不是单独 \(J_E\)；
- robust checkpoints can reduce nuisance-direction sensitivity after rollout even if encoder latents are not exactly invariant；
- small encoder shifts can still matter if \(J_{G_{\mathbf a}}\) has high gain。

这条 proposition 可以直接连接：

- ACPC basin 中 \(R_F\) 下降；
- same-state noisy radius 下降；
- semantic margin pass-rate 提升。

### 1.5 Selective sensitivity / semantic margin

在 local Gaussian proposition 后加一段 selective interpretation。

令 nuisance perturbation directions 为 \(\mathcal N\)，semantic/action-relevant directions 为 \(\mathcal S\)。理想选择性目标不是让全部 sensitivity 下降，而是：

\[
\|J_{G_{\mathbf a}}J_E v\| \text{ small for } v\in\mathcal N,
\]

同时对 action-relevant state difference \(u\in\mathcal S\)：

\[
\|G_{\mathbf a}(E(s+u))-G_{\mathbf a}(E(s))\| > m.
\]

将其连接到 empirical pass event：

\[
M = \mathbf 1\left[
 d_{\mathrm{semantic\ diff}}
 > d_{\mathrm{same\ state\ noise}} + \delta_m
\right].
\]

当前主文表格用 \(\delta_m=0\)。建议在文中明确说这是 **state-proxy semantic margin**，不是 oracle semantic proof。

### 1.6 Finite-sample semantic margin certificate

给 semantic margin pass-rate 加一个简单有限样本 calibration。

若 \(n\) 个 sampled pairs 的 empirical pass-rate 为 \(\hat p\)，独立抽样近似下 Hoeffding 给出：

\[
p \ge \hat p - \sqrt{\frac{\log(1/\alpha)}{2n}}
\quad \text{with probability } 1-\alpha.
\]

如果每 task/endpoint 聚合三训练种子，每种子 100 pairs，则 \(n=300\)。当 \(\hat p=1.00\)、\(\alpha=0.05\) 时：

\[
1-\sqrt{\frac{\log 20}{600}} \approx 0.929.
\]

推荐写法：

> Treating sampled windows as independent for calibration, a 300-pair pass-rate of 1.00 gives a Hoeffding lower bound of about 0.93 at 95% confidence. Because nearby dataset windows may be dependent, we use this as a finite-sample calibration rather than a dataset-wide formal guarantee.

这样不会过度 claim，但理论味更强。

### 1.7 Optional: prediction-loss to ACPC triangle lemma

可放 appendix，不一定主文。

One-step 情况下，定义：

\[
\hat z_{t+1}=F(E(h_t),a_t), \quad
\hat{\tilde z}_{t+1}=F(E(\tilde h_t),a_t),
\]

clean/noisy target latents：

\[
z_{t+1}=E(h_{t+1}), \quad
\tilde z_{t+1}=E(\tilde h_{t+1}).
\]

由三角不等式：

\[
d(\hat z_{t+1},\hat{\tilde z}_{t+1})
\le
 d(\hat z_{t+1},z_{t+1})
 + d(z_{t+1},\tilde z_{t+1})
 + d(\tilde z_{t+1},\hat{\tilde z}_{t+1}).
\]

解释：

- clean branch prediction error 小；
- noisy branch prediction error 小；
- target future readout 对 nuisance perturbation 稳定；
- 则 one-step ACPC 小。

必须加限制：

> This teacher-forced one-step bound does not guarantee closed-loop rollout stability; the target-view ablation shows that perturbed-history to original-future denoising is not sufficient.

## 2. 推荐修改位置

### 2.1 `paper1/main.tex`

建议在 `\section{Action-Conditioned Predictive Consistency}` 中调整：

1. 保留 setup / definitions。
2. 压缩当前 fixed-candidate propositions 的重复表述。
3. 新增 subsection：
   - `\subsection{Sampled-candidate stability}`
   - `\subsection{Gaussian sensitivity and selective margins}`
4. 将较长证明放入 appendix：
   - `\section{Proofs for ACPC stability and sensitivity}`

### 2.2 摘要 / Contributions

摘要不要塞太多公式，只加一句理论升级：

> We further extend the fixed-candidate link to a sampled-candidate flip-probability bound and show by local Gaussian linearization that ACPC estimates action-conditioned encoder–predictor sensitivity.

Contributions 中 C2 可改为：

> Under a Lipschitz cost readout, ACPC controls fixed-candidate cost drift and yields a sampled-candidate top-1 flip bound with two explicit terms: ACPC tails and clean margin tails. A local Gaussian linearization shows that the measured basin radius estimates action-conditioned sensitivity \(\|J_GJ_E\|_F\), explaining why encoder invariance alone is incomplete.

### 2.3 Main experiments text

在 fixed-rule validation 段落后加一句理论连接：

> The four-rule readout matches the sampled-pool theorem: ACPC-H and PCC estimate drift, CRA estimates rank stability, and MAF estimates the high-margin flip term.

在 semantic margin 表后加一句有限样本 calibration：

> With 100 sampled pairs per seed and three seeds per task/endpoint, a pass-rate near 1.00 corresponds to an approximate Hoeffding lower confidence bound of 0.93 under independent-pair calibration.

注意加依赖性免责声明。

## 3. 建议新增 appendix 证明结构

新增 appendix：`\section{Proofs and calibration for ACPC diagnostics}`。

内容顺序：

1. Proof of sampled-pool ACPC stability theorem。
2. Proof of replanning union bound。
3. Proof / derivation of local Gaussian sensitivity proposition。
4. Hoeffding calibration for semantic margin pass-rate。
5. Optional triangle lemma from prediction loss to one-step ACPC。

证明都要短，不要引入过多 notation。

## 4. 建议新增或调整实验补充

这些不是必须，但如果 Codex 能从现有 artifacts 快速生成，建议做。

### 4.1 Margin-conditioned flip curve

从 paired candidate artifact 中生成：

- 横轴：clean top-1/top-2 margin bin；
- 纵轴：action flip rate 或 MAF；
- 分 baseline vs std0.08 endpoint；
- 可按四任务 aggregate，也可每任务一条。

作用：直接验证 sampled-pool theorem 中的 margin-tail 项。

### 4.2 ACPC tail vs MAF table

按 checkpoint 统计：

- fraction of candidate pools with ACPC-H/trans > threshold；
- PCC median；
- MAF；
- obs0.08 success。

作用：让 theorem 中 \(K\delta\) 和 empirical MAF 更贴近。

### 4.3 Semantic margin bootstrap CI

给 `semantic_margin_passrate_lewm_three_seed.json` 生成 bootstrap CI 或 Wilson CI。若实现麻烦，先用 Hoeffding calibration 文本即可。

## 5. 最终论文应避免的 claim

不要写：

- “We prove CEM robustness.”
- “ACPC guarantees closed-loop robustness.”
- “The fixed rule replaces closed-loop evaluation.”
- “Semantic margin proves all task semantics are preserved.”
- “Gaussian-noise training gives general visual robustness.”

推荐写：

- “The theorem controls sampled-pool instability under explicit tail and margin conditions.”
- “The bound exposes the terms measured by ACPC/PCC/CRA/MAF.”
- “The semantic margin is a state-proxy selective-consistency certificate.”
- “Closed-loop evaluation remains the final authority.”
- “Blur/resize are scope checks, not a general transfer claim.”

## 6. 建议 Codex 执行 checklist

### 必做

- [ ] 在 main theory section 中新增 sampled-candidate theorem。
- [ ] 新增 local Gaussian sensitivity proposition。
- [ ] 新增 semantic margin finite-sample calibration。
- [ ] 在 appendix 中补完整证明。
- [ ] 更新 abstract / contributions / discussion 的理论表述。
- [ ] 确保所有 claim 都维持 diagnostic scope。

### 推荐做

- [ ] 从现有 artifacts 生成 margin-conditioned flip curve 或 table。
- [ ] 在 fixed-rule validation 段落显式连接 ACPC/PCC/CRA/MAF 与 theorem terms。
- [ ] 将 “semantic margin” 全文统一为 “state-proxy semantic margin” 或首次出现时这样定义。

### 暂不做

- [ ] 不新增 ACPC training objective。
- [ ] 不新增大规模 baseline comparison。
- [ ] 不把 blur/resize 提升为主结果。
- [ ] 不写完整 CEM convergence proof。

## 7. 建议最终新增理论贡献表述

可在 introduction/contributions 或 theory section 末尾使用：

> The resulting theory is deliberately diagnostic rather than algorithmic. Fixed-candidate ACPC bounds cost drift; sampled-candidate ACPC bounds expose the ACPC-tail and clean-margin failure modes relevant to MPC candidate pools; local Gaussian linearization shows that the measured basin radius estimates action-conditioned encoder–predictor sensitivity rather than encoder invariance alone; and the state-proxy semantic margin pass-rate checks the selective half of the condition.

中文理解：

> 这套理论不试图证明完整闭环鲁棒性，而是解释诊断为什么合理：ACPC 控制候选池 cost/rank 稳定性，Gaussian ACPC 测的是 encoder–predictor 的动作条件敏感度，semantic margin 检查这种收缩是否没有吞掉动作相关差异。

## 8. 推荐优先级

若时间有限，按以下顺序执行：

1. **Sampled-pool theorem**：最能修复 fixed-candidate 薄弱点。
2. **Gaussian sensitivity proposition**：最能解释 Gaussian stressor 与 ACPC basin。
3. **Semantic finite-sample calibration**：最能增强 selective margin 表格可信度。
4. **Triangle lemma**：锦上添花，可放 appendix。
5. **Margin-conditioned flip curve**：如果 artifact 支持，审稿观感会更强。
