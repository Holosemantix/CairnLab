# Paper 1 post-整改 strict multi-reviewer audit — Codex 执行版

Date: 2026-06-26  
Branch audited: `Holosemantix/le-wm@ag/dev`  
Key files inspected: `paper1/main.tex`, `paper1/references.bib`, `paper1/docs/reference_audit.md`, `paper1/check_arxiv_ready.sh`, public release state in `Anguo-star/le-wm`.

## 0. 总体判断

当前版本已经比上一轮明显更接近投稿稿件：

- 标题已收窄为 **Gaussian Visual Robustness / Diagnostic**，不会再被读成“通用 JEPA 鲁棒性定理”。
- 摘要已明确说是 diagnostic framework / release package，不是新训练目标或 closed-loop guarantee。
- `drop` 主文表格的符号方向已经统一为 `clean success - observation-noise 0.08 success`。
- 理论部分已经补上：
  - horizon metric `d_H`；
  - fixed-candidate cost drift；
  - top-1 candidate margin stability；
  - candidate-family ACPC pseudo-metric；
  - collapse counterexample；
  - discriminability guard。
- 理论相关文献已经补进来：`vanassel2025jointembeddingreconstruction` 和 `littwin2024jepaavoidsnoisyfeatures`。

但我不建议立刻提交。还有几个会被严格审稿人或 arXiv release gate 抓住的问题，尤其是：

1. **作者占位符仍在，readiness script 会失败。**
2. **公开仓库 URL 仍需确认 default branch 是否真的包含 release artifacts/scripts。**
3. **理论定理与主文实测 ACPC-basin 之间还需要一句关键边界说明：主文 `R_F` 是 median / recorded-action diagnostic，不是 fixed-candidate uniform certificate。**
4. **主文 discriminability guard 证据表没有展示 ID probe，但正文用 ID probe 支撑“不是 collapse”。**
5. **blur appendix 的 `worst drop` 仍是负号，和主文 drop 方向冲突。**
6. **Phase-0 shared-candidate evidence 太靠后，建议主文补 1 句代表性数值，以支撑理论分量和 planner link。**

下面按 7 位严格审稿人角色给出整改项。

---

## 1. Reviewer A — Theory / math rigor

### A1. P0/P1：主文 `R_F` 不是 fixed-candidate theorem 的 `epsilon`，必须显式写清楚

**问题。**  
理论 corollary 要求：

\[
D_{\mathrm{ACPC}}^{\mathcal A}(h,\tilde h) = \max_{\mathbf a \in \mathcal A} d_H(\cdots) \le \epsilon
\]

这是一个 fixed candidate set 上的 **uniform max bound**。但主文 ACPC-basin 的 `R_F` 是：

- 100 dataset windows；
- 每个 window 用 recorded action sequence；
- original/noised views 的 median spread；
- normalised by unperturbed transition reference。

它是 localization diagnostic，不是 uniform candidate-set certificate。当前稿件虽然说了 diagnostic，但还没有把 theorem 和 `R_F` 的形式差异钉死。严格审稿人可能会说：你用一个 median single-sequence proxy 去支撑 fixed-candidate theorem。

**建议修改位置。**  
`main.tex` 中 `\subsection{Operational consistency diagnostics}` 或 `\subsection{Paired Gaussian-noise ACPC basin diagnostic}`。

**建议插入文本。**

在 `Operational consistency diagnostics` 中当前这句后面：

```tex
$R_F/R_E$ is a descriptive contraction ratio.
```

插入：

```tex
This empirical $R_F$ is a distributional, recorded-action localization proxy. It should not be read as an estimate of the uniform $\epsilon$ in the fixed-candidate stability corollary, which requires a bound over every candidate in a shared set $\mathcal A$. The shared-candidate PCC/CRA/MAF readouts in Appendix~\ref{sec:appendix-phase0} are included only as a face-validity check of that downstream link.
```

**验收标准。**  
读者不会把 `R_F` 误认为 theorem certificate。

---

### A2. P1：`D_ACPC^A` pseudo-metric 需要补 “nonempty candidate family” 和 `d_H` 条件

**问题。**  
当前写法：

```tex
When $d_H$ is a metric, $D_{\mathrm{ACPC}}^{\mathcal A}$ is a pseudo-metric...
```

数学上基本对，但严格写法应补：candidate family 非空；如果 `\alpha_k` 可为 0，`d_H` 本身可能只是 pseudo-metric。你当前已经用 “When d_H is a metric” 回避了大问题，但可以更严谨。

**建议替换。**

把：

```tex
For a fixed candidate family $\mathcal A$, the stability condition can be summarized by
```

改成：

```tex
For a nonempty fixed candidate family $\mathcal A$, the stability condition can be summarized by
```

把：

```tex
When $d_H$ is a metric, $D_{\mathrm{ACPC}}^{\mathcal A}$ is a pseudo-metric on histories...
```

改成：

```tex
When $d_H$ is a metric (or a pseudo-metric, if some horizon weights vanish), $D_{\mathrm{ACPC}}^{\mathcal A}$ is a pseudo-metric on histories...
```

**验收标准。**  
不会被理论审稿人用 “max over empty set / zero weights” 这种小洞攻击。

---

### A3. P1：discriminability readout 建议显式 action-conditioned

**问题。**  
当前 discriminability condition 写：

```tex
d(\Psi(z_i), \Psi(z_j)) > m'
```

但前文说 `s_i, s_j` 是在某个 action sequence 下 external future readout diverges。严格来说，如果区分性是 action/transition/cost-dependent，`\Psi` 最好允许依赖 action sequence 或 rollout，而不只是 raw latent。

**建议最小修改。**

把定义段中：

```tex
where $\Psi$ is a discriminability readout (latent, predicted delta, inverse-dynamics feature, cost feature, or rollout embedding)
```

改成：

```tex
where $\Psi$ is a discriminability readout, possibly action-conditioned (latent, predicted delta under $\mathbf a$, inverse-dynamics feature, cost feature, or rollout embedding)
```

如果愿意更严谨，可把式子改为：

```tex
d\!\big(\Psi(z_i,\mathbf a),\, \Psi(z_j,\mathbf a)\big) > m',
```

但这需要同步全文变量说明。最小修改用文字补充即可。

**验收标准。**  
定义和 “action-conditioned” 主题完全一致。

---

### A4. P1：理论部分已经具备自己的东西，但主文还应明确“不是借鉴那两篇理论文献”

**当前状态。**  
你已经引用并正确定位了：

- `vanassel2025jointembeddingreconstruction`：closed-form SSL solutions / view generation / irrelevant large-magnitude features / joint embedding weaker alignment condition。
- `littwin2024jepaavoidsnoisyfeatures`：deep-linear self-distillation dynamics / JEPA bias toward high-influence predictive features rather than high-variance/noisy features。

这是正确的，不能删。

**仍需补强的点。**  
这两篇解释的是 representation-level nuisance/noisy-feature behavior；你们的理论独有点是：

1. 把一致性要求放到 **action-conditioned rollout readout** 后；
2. 给出 **fixed candidate cost-drift / top-1 candidate stability**；
3. 明确 **ACPC alone permits collapse**；
4. 用 discriminability guard 把 nuisance contraction 与 action-distinct separation 绑在一起。

建议在 ACPC 理论段结尾加 1 句，避免读者觉得只是套用了已有 JEPA noisy-feature theory。

**建议插入位置。**  
`ACPC as a fixed-candidate stability condition` 末尾、collapse proposition 后。

**建议文本。**

```tex
This is the point at which ACPC differs from existing nuisance-feature analyses of latent prediction: those results help explain why latent prediction may suppress irrelevant or noisy features at the representation level, whereas the condition above asks whether the remaining perturbation changes action-conditioned rollout readouts and candidate costs while preserving action-relevant separations.
```

**验收标准。**  
审稿人能清楚看到你们的理论贡献不是“引用已有视觉扰动理论”，而是把它推进到 control-time predictive dynamics。

---

### A5. 不要做的理论改动

不要把当前结果升级成：

- “ACPC guarantees robustness”
- “CEM stability theorem”
- “closed-loop guarantee”
- “noise augmentation optimizes ACPC”
- “R_F predicts success”

当前 fixed-candidate theorem 的边界是正确的；不要为了显得理论更强而越界。

---

## 2. Reviewer B — Empirical evidence / statistics

### B1. P1：把 Phase-0 shared-candidate evidence 提前一点，增强故事分量

**问题。**  
理论强调 candidate-cost drift / top-1 stability，但主文主要报告 `R_F`，而 `R_F` 是 recorded-action basin proxy。真正更贴近 candidate-cost / ranking 的 PCC/CRA/MAF 在 appendix Phase-0。现在主文只一句 “Appendix adds exploratory check”，分量有点埋了。

**建议。**  
在 `Paired Gaussian-noise ACPC basin diagnostic` 末尾，也就是当前：

```tex
The basin contraction is not treated as sufficient evidence of robustness...
```

后面补 1 句代表性结果，不需要加表。

**建议文本。**

```tex
A separate shared-candidate sanity check in Appendix~\ref{sec:appendix-phase0} follows the same direction on the auxiliary observation+goal stress endpoint: for LeWM PushT, ACPC-$H$/transition drops from $2.740$ to $0.149$, PCC from $67.3$ to $4.6$, candidate-ranking agreement rises from $0.323$ to $0.986$, and margin-conditioned action flips drop from $0.96$ to $0.08$. Because this check uses the auxiliary observation+goal endpoint and random candidate sets, we keep it as face-validity evidence rather than as the main robustness claim.
```

**理由。**  
这会显著增加理论-实验闭环的重量，同时仍然不越界。Phase-0 的原表已经在 appendix，数值可直接引用。

---

### B2. P1：主文 discriminability guard 表格最好展示 ID probe，或降低正文依赖

**问题。**  
正文说：

```tex
rank, transition-resolution, inverse-dynamics diagnostics check that contraction is not merely collapse
```

并在 PushT bullet 中说 ID probe stays flat (`0.774 -> 0.765`)。但主文 `tab:diag-base-vs-repr` 只有：

- NN cos dist
- effective rank
- trans L2
- rollout T8 L2

没有 ID probe。严格审稿人会说：你最关键的 anti-collapse proxy 没放进主表，只在 prose 里说。

**两种修法。**

优先修法：把 `NN cos dist` 换成 `ID probe`。  
理由：NN cos dist 对 anti-collapse guard 的解释价值弱于 ID probe；ID probe 是 action-relevant proxy。

建议表头变成：

```tex
Task & \stdmax{} & eff. rank & trans. L2 & ID probe $R^2$ & rollout T8 L2
```

并从 canonical diagnostics / diagnostics summary 中填四个任务的 base -> 0.08 ID probe 值。

保守修法：如果暂时不想改表，就把表 caption / notes 改成：

```tex
The compact table shows rank, transition-resolution, and rollout drift; inverse-dynamics probe values used in the PushT collapse check are reported in Appendix~...
```

但我建议采用优先修法。

**验收标准。**  
读者不需要跳 appendix 才能看到 “不是 collapse” 的核心证据。

---

### B3. P1：`R_F` / rollout drift “near-collapse” 用词不够好

当前 note 写：

```tex
the near-collapse of multi-step rollout drift...
```

“collapse” 在本文中已经有 representation collapse 的技术含义，这里会混淆。建议改成：

```tex
the sharp reduction of multi-step rollout drift...
```

并把后文 “drop in rollout drift is not unambiguously good news” 保留。

**验收标准。**  
避免同一个词 collapse 同时指好现象和坏现象。

---

### B4. P1：blur appendix 的 `worst drop` 符号与主文 drop 方向冲突

**问题。**  
主文已统一 drop = clean - obs0.08，正数代表退化。但 blur appendix 表里 `worst drop` 仍为负数，例如 LeWM TwoRoom `-63.67`。这会重新制造符号混乱。

**建议修法。**

二选一：

1. 改表头为 `worst change`，caption 明确 `corrupted - clean`，保留负号。
2. 更推荐：改成正数 `worst drop`，即 clean - worst corrupted success。

推荐改法：

```tex
worst drop
```

列值全部取正：

- LeWM TwoRoom: `63.67`
- LeWM PushT: `27.67`
- LeWM Reacher: `38.33`
- LeWM Cube: `12.67`
- PLDM TwoRoom: `68.33`
- PLDM PushT: `13.00`
- PLDM Reacher: `14.67`
- PLDM Cube: `2.00`

同步改 reading 中的 “lose at most about $15$ pt” 不受影响。

**验收标准。**  
全篇 drop 方向一致。

---

### B5. P2：no independent training seeds 的限制已经写清楚，不要再过度 caveat

当前 scope 已明确：

```tex
three evaluation seeds per checkpoint, not independent training seeds
```

这足够了。不要再到处重复，否则会削弱故事分量。保留 abstract 的 diagnostic framing 和 discussion 的 scope 即可。

---

## 3. Reviewer C — Claims / story weight

### C1. 当前 story 分量是够的，不要继续削弱标题和摘要

当前标题和摘要已经安全，但不是没分量。核心故事现在是：

- Latent prediction 的 representation-level nuisance suppression 理论已有；
- 控制中鲁棒性不应只看 encoder；
- ACPC 把判据放到 action-conditioned rollout 后；
- fixed-candidate theorem 连接到 candidate costs；
- collapse proposition 强制 discriminability guard；
- LeWM/PLDM Gaussian sweeps + ACPC basin + negative ablations 支撑这个 diagnostic principle。

这个分量对于 **diagnostic empirical study** 是够的。不要再把标题降成 “notes on” 或 “preliminary observations”。

---

### C2. P1：C2 contribution 仍可再精确一点

当前 C2：

```tex
preserves the selected candidate within that set
```

建议与 abstract 和 corollary 完全一致：

```tex
preserves the top-ranked candidate within that fixed set
```

**替换句。**

```tex
Under a Lipschitz cost readout, bounded rollout disagreement bounds candidate-cost drift on a fixed candidate set, and a sufficiently large clean action margin preserves the top-ranked candidate within that fixed set.
```

**验收标准。**  
贡献、摘要、理论命题措辞一致。

---

### C3. P2：个别 “mechanistic” wording 仍偏强

以下句子可略软化：

```tex
the same intervention that contracts same-state noisy rollouts appears to stabilize the clean action-conditioned predictor at epoch 10
```

建议：

```tex
the same intervention coincides with a more stable clean action-conditioned predictor at epoch 10
```

原因：只有 one trained checkpoint per noise level，`appears to stabilize` 虽然已经谨慎，但 `intervention that ... stabilise` 仍有因果味。

---

### C4. P2：表格 reading 中 “redundant visual state” 可改为 “visually redundant regime”

当前 `Main reading`：

```tex
large recovery; redundant visual state
```

建议：

```tex
large recovery; visually redundant regime
```

原因：state 是否 redundant 是任务解释，不是直接测量。

---

## 4. Reviewer D — Related work / citation accuracy

### D1. 已修好：Van Assel + Littwin 都应保留

当前相关段落已正确补入：

- Van Assel et al. for closed-form SSL solutions / view generation / large-magnitude irrelevant features / joint embedding weaker alignment.
- Littwin et al. for deep-linear self-distillation dynamics / high-influence predictive features vs high-variance/noisy features.

BibTeX 里也已有对应条目：

```bibtex
@misc{vanassel2025jointembeddingreconstruction,...}
@misc{littwin2024jepaavoidsnoisyfeatures,...}
```

Reference audit 也记录了 43 entries 和 Littwin add。这个整改通过。

### D2. 重要边界：不要把 Littwin 写成 Gaussian pixel corruption theory

Littwin 是 noisy features / implicit bias / deep-linear self-distillation，不是 closed-loop Gaussian pixel noise theorem。当前正文写的是 “nuisance or noisy features”，这个边界是对的。不要改成 “Gaussian visual perturbation theory”。

### D3. P2：`reference_audit.md` 顶部 Date 仍是 2026-06-22

文件已加入 2026-06-26 记录，但顶部：

```md
Date: 2026-06-22
```

建议改为：

```md
Date: 2026-06-26
```

这不是论文内容问题，但 release audit 看起来更干净。

### D4. P2：public reproduction guide 标题仍是旧题目

`Anguo-star/le-wm@ag/dev` 的 `PAPER1_REPRODUCTION.md` 中标题仍写：

```md
A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models
```

当前论文题目是：

```tex
Action-Conditioned Predictive Consistency as a Diagnostic for Gaussian Visual Robustness in JEPA World Models
```

建议同步 public release docs，至少在 reproduction guide 中加：

```md
Current title: Action-Conditioned Predictive Consistency as a Diagnostic for Gaussian Visual Robustness in JEPA World Models
```

或直接替换旧标题。

---

## 5. Reviewer E — Figures / tables / PDF layout

### E1. P0/P1：必须本地渲染 PDF 检查，尤其是 `[H]` floats 和 landscape pages

我无法从 GitHub connector 直接视觉检查二进制 PDF/PNG。Codex 必须本地执行：

```bash
cd paper1
bash build.sh --clean
```

然后人工或脚本检查：

- main text 是否出现大空白页；
- `[H]` 表/图是否挤压到奇怪位置；
- landscape page 是否在 PDF viewer 中正确旋转；
- figure text 是否可读；
- t-SNE 小 panel summary 是否没有糊；
- figure captions 是否和图中颜色/markers 一致；
- `fig2_sweep.png` 中 blue circles / red squares 是否真实存在；
- appendix large tables 是否超出页边距；
- main `fig1_concept.png` 是否没有过多文字或小字。

**特别注意。**  
主文现在有多个 `[H]`，例如 `tab:corruption-cliff`、`tab:sweep-summary`、`fig:sweep`。这有利于顺序，但会增加排版风险。如果 PDF 有明显空白，优先把主文 float 改回 `[tbp]`，appendix 保留 `[H]`。

---

### E2. P1：`tab:diag-base-vs-repr` 最好改成真正支撑 collapse guard 的表

见 B2。这个既是 empirical 问题，也是表格设计问题。现在表里 `NN cos dist` 不是最能支撑 discriminability guard 的列；建议换成 ID probe。

---

### E3. P1：blur table drop 符号必须修

见 B4。图表一致性问题，不要漏。

---

### E4. P2：Phase-0 table 很宽，检查 landscape 是否真的可读

Phase-0 表有 9 列并且 values dense。它在 appendix，用 landscape 可以接受，但 PDF 必须检查：

- column labels 是否重叠；
- `ACPC-H/trans.` 是否够清楚；
- `top-8` 是否在 caption 里定义；
- `obs+goal` 是 success rate 还是 condition，建议 caption 或 table header 写清楚为 `obs+goal success`。

建议把表头：

```tex
obs+goal
```

改成：

```tex
obs+goal success
```

---

## 6. Reviewer F — Reproducibility / release gate

### F1. P0：作者占位符仍在，readiness script 会失败

当前：

```tex
\newcommand{\arxivauthors}{Author names to be supplied for arXiv v1}
\author{\arxivauthors}
```

`check_arxiv_ready.sh` 明确将其设为 hard blocker：

```bash
if grep -q "Author names to be supplied" main.tex && [[ "$ALLOW_AUTHOR_PLACEHOLDER" != "1" ]]; then
  fail ...
fi
```

**必须修。**

如果是 arXiv v1：填真实作者。  
如果是匿名会议稿：不要走这个 arXiv readiness script；另建匿名版本。

---

### F2. P0：公开 URL 需要和 default branch 内容一致

当前 acknowledgements：

```tex
https://github.com/Anguo-star/le-wm
```

`check_arxiv_ready.sh` 也强制这个 URL 存在。问题是本轮实际论文源码审在 `Holosemantix/le-wm@ag/dev`，而 `Anguo-star/le-wm` 的 default `main` 目前还不是 paper1 release 状态；`Anguo-star/le-wm@ag/dev` 有 release artifacts/scripts，但 root URL 默认展示的是 `main`。

**整改选项。**

优先选项：

1. 将 `Anguo-star/le-wm@ag/dev` merge 到 `Anguo-star/le-wm@main`。
2. 确认 root URL 打开即可看到:
   - `PAPER1_REPRODUCTION.md`
   - `DATA_MANIFEST.md`
   - `assets/paper1_data/*`
   - `assets/paper1_figs/*`
   - `tools/paper1_*`
3. 再保留当前 acknowledgements root URL。

如果不 merge main：

- 不要在论文里写 root URL；
- 用 release tag / DOI / Zenodo；
- 同步修改 `check_arxiv_ready.sh` 的 URL gate。

**验收标准。**  
论文里给的 URL 不需要读者切 branch 就能找到 release artifacts。

---

### F3. P1：readiness script 很好，但还应增加 public release content check

可在 `check_arxiv_ready.sh` 里增加本地检查很难验证远程 default branch。最实际做法是在 release checklist 加人工项：

```bash
# Manual release check:
# 1. Open https://github.com/Anguo-star/le-wm in an incognito browser.
# 2. Confirm PAPER1_REPRODUCTION.md and assets/paper1_data are visible on default branch.
# 3. Confirm README points to Paper 1 reproduction guide.
```

---

## 7. Reviewer G — Writing / wording polish

### G1. P1：abstract 已经安全，不建议大改

当前 abstract 已经兼顾分量与边界：

- 有 problem；
- 有 ACPC definition；
- 有 fixed-candidate planner link；
- 有 collapse guard；
- 有 strong numbers；
- 有 PLDM bounded replication；
- 有 negative ablations；
- 有 “not a new training objective or closed-loop guarantee”。

不建议再压缩太多，否则故事会变轻。

### G2. P2：避免 “recovered checkpoints” 被理解为独立训练稳定性

当前 abstract 说：

```tex
Full-sequence input-side noise training supplies recovered checkpoints for diagnosis
```

这是可以接受的。但如果要更严格，可改为：

```tex
Full-sequence input-side noise training supplies point-recovered checkpoints for diagnosis within this fixed epoch-10 protocol
```

我不强制，因为这会让 abstract 变笨重。Discussion 已经说明没有 independent training seeds。

### G3. P2：美式/英式拼写已基本统一为美式

当前主要是 `behavior`, `localized`, `regularization`, `neighbor`，基本统一。后续避免再引入 `behaviour/colour`。

---

## 8. 建议 Codex 执行顺序

### Step 1 — hard blockers

1. 替换 `\arxivauthors` 为真实作者。
2. 确认 `Anguo-star/le-wm` default branch 是否是 release branch；若不是，merge `ag/dev` 或修改 URL/checker。

### Step 2 — theory precision patch

1. 在 ACPC operational diagnostics 增加 `R_F is not uniform epsilon` 说明。
2. 给 `D_ACPC^A` 加 `nonempty` 和 `metric/pseudometric` 条件。
3. 把 C2 的 `selected candidate` 改为 `top-ranked candidate within that fixed set`。
4. 给 discriminability readout 增加 “possibly action-conditioned”。
5. 可选：collapse proposition 后加“区别于 existing nuisance-feature theory”的一句话。

### Step 3 — evidence/story patch

1. 主文 ACPC-basin 后加一条 Phase-0 PushT shared-candidate sanity-check sentence。
2. `tab:diag-base-vs-repr` 加 ID probe 或替换 NN cos dist。
3. 将 “near-collapse of multi-step rollout drift” 改为 “sharp reduction”。

### Step 4 — table/layout patch

1. blur table `worst drop` 改成正数，或改名 `worst change` 并解释负号。推荐正数。
2. Phase-0 table header `obs+goal` 改成 `obs+goal success`。
3. 本地 render PDF，检查 `[H]` float、landscape、figure text。

### Step 5 — release doc patch

1. `reference_audit.md` 顶部 Date 改为 2026-06-26。
2. `PAPER1_REPRODUCTION.md` 更新到当前标题。
3. README 如有旧题目或旧 paper link，更新或加 Paper 1 reproduction pointer。

---

## 9. 最终应通过的命令 / 检查

```bash
python -m tools.check_paper1_consistency
cd paper1
bash build.sh --clean
bash check_arxiv_ready.sh
```

若作者占位符还没填，`check_arxiv_ready.sh` 失败是正确的；最终提交前必须不失败。

额外 grep：

```bash
rg -n "Author names to be supplied|Scope of this arXiv version|paper-facing|method-facing|complete code and data|tree/ag/dev|tree/main" paper1/main.tex
```

期望：无命中。

理论 claim grep：

```bash
rg -n "guarantee|proves|CEM stability|closed-loop guarantee|predicts robustness|oracle" paper1/main.tex
```

期望：只有明确否定或 limitation 语境。

drop 符号 grep / 人工检查：

```bash
rg -n "drop|worst drop|worst change" paper1/main.tex
```

期望：所有 `drop` 方向一致；如果 blur 用负号，必须命名为 `change` 而不是 `drop`。

---

## 10. 最终 verdict

整改后这篇文章的理论部分已经不是“偏薄弱”的状态。它有自己的贡献：

- existing JEPA/noisy-feature theory 解释 representation-level nuisance suppression；
- 本文提出 control-time ACPC；
- fixed-candidate margin theorem 把 predictive consistency 接到 candidate cost / top-1 candidate stability；
- collapse proposition 证明 discriminability guard 是必要的；
- empirical ACPC basin + Phase-0 + negative ablations 支撑 selective-consistency story。

当前剩余问题不是“理论没有东西”，而是 **理论对象与主文实测 proxy 的边界必须再写清楚**，并且 **主文要露出一小块 candidate-cost evidence**，否则理论看起来比实验多走了一步。按上面的 P0/P1 修完后，我认为作为 diagnostic paper 的故事分量是足够的。
