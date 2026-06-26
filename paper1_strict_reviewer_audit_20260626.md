# Paper1 提交前严格审稿整改文档（2026-06-26）

对象：`Holosemantix/le-wm` 仓库 `ag/dev` 分支的 `paper1/`。

目标：以 7 位严格审稿人视角重新审视 paper1 的投稿风险，并给 Codex 一份可执行的整改清单。本文档优先列真正会被审稿人挑战、会影响 arXiv/投稿就绪、会削弱故事分量的问题；避免为了找问题而把正确边界改坏。

本轮依据：`paper1/main.tex`、`paper1/references.bib`、`paper1/README.md`、`paper1/check_arxiv_ready.sh`、`DATA_MANIFEST.md`、现有 audit/patch 文档，以及公开论文元数据抽查。由于当前 GitHub connector 不能直接读取二进制 PDF/PNG，PDF 版式和图像细节没有被我直接截图验证；因此所有“图/版式”项都写成 Codex 必须本地渲染确认的 blocking preflight，而不是盲目声称已视觉检查通过。

---

## 0. 总体结论

**当前稿件已经从早期 artifact report 进化成一个可投稿的 diagnostic empirical/theory paper 雏形，但仍不应直接提交。** 最主要原因不是实验不够，而是若不修正几个表述和提交 gate，审稿人会从“过宽标题/摘要、drop 符号不一致、作者/URL 未准备、理论看似只是 Lipschitz 小 lemma、代表点选择看似 cherry-picking、图表/PDF 未最终 preflight”这些点切入。

修完 P0 与 P1 后，建议投稿定位保持为：

> A controlled diagnostic study of Gaussian visual robustness in JEPA latent world-model control, centered on LeWM and bounded by PLDM replication, using ACPC plus a discriminability guard to connect representation perturbations to action-conditioned rollout and fixed-candidate planning stability.

不要改成“新鲁棒训练方法论文”，也不要把 ACPC 写成 closed-loop/CEM guarantee。本文真正的分量应来自三件事的组合：

1. **理论骨架**：ACPC + fixed-candidate cost/action stability + collapse counterexample + discriminability guard。
2. **实验证据链**：Gaussian noise cliff/recovery + ACPC basin localization + proxy discriminability + PLDM boundary replication。
3. **负结果边界**：target-view denoising 失败、heteroscedastic loss 抹掉 PushT control-critical transitions、partial correlation 不支持 metric-as-oracle。

---

## 1. 七位严格审稿人画像与核心质疑

### R1：Area Chair / Story & Contribution Reviewer

会问：这到底是方法、理论、还是经验诊断？标题和摘要是否让读者期待一个通用 JEPA robustness theory？贡献是否因为过度保守而没有分量？

核心风险：当前标题 `Diagnosing Visual Robustness in JEPA World Models through Action-Conditioned Predictive Consistency` 没有明确 `Gaussian`、`diagnostic study`、`controlled LeWM/PLDM sweeps`。摘要信息量太大，一段里塞了任务、36 checkpoints、PLDM、多个指标、generalizable world models，很容易让人读完仍不知道主线。

整改方向：收窄标题但增强理论主线；摘要保留 2-3 个强数字，明确 ACPC 的理论桥接和边界。

### R2：Theory Reviewer

会问：ACPC 是不是只是重新命名的 consistency？Proposition 是否只是显然 Lipschitz？你们和 LeJEPA / joint embedding theory / bisimulation / group-action world model 的理论边界是什么？

核心风险：当前 theory 已有 fixed-candidate cost drift、top-1 margin stability、ACPC-collapse proposition，方向正确；但如果不把它们组织成“selective predictive stability”核心命题，审稿人仍可能认为理论薄。

整改方向：新增一个小而清楚的 “ACPC pseudo-metric / selective stability theorem” 或 corollary，把已有 pieces 合并成可引用核心结论；增加与现有理论的关系段，说明我们借鉴但不重复：latent prediction theory 解释为什么像素重构不是目标，LeJEPA 讨论 latent recovery/planning，bisimulation讨论 control abstraction，group actions讨论 action-faithfulness；本文独有的是 same-state visual corruption 的 action-conditioned rollout stability + discriminability guard。

### R3：Empirical / Statistics Reviewer

会问：3 seeds 是训练种子还是 evaluation seeds？每个 grid cell 是不是一个 checkpoint？drop 的定义是否一致？点最佳是否 cherry-picking？partial correlation 的小 n 是否过读？

核心风险：文中已经正确说明是 evaluation seeds，不是 independent training seeds；但表格中的 `drop` 符号存在不一致，且 `representative` 列可能被理解成同一 checkpoint，其实有些任务 clean best 与 noisy best 不同。

整改方向：统一 drop 定义；明确 point-best / representative selection；把 cross-checkpoint correlation 降温，主文只保留 partial Spearman 或把 unconditional Pearson/Spearman 挪 appendix。

### R4：Figures / Tables / Layout Reviewer

会问：图中文字能否读清？是否只靠蓝/红颜色？t-SNE 是否误导？landscape table 是否旋转/裁切？arXiv source 是否包含 symlink 或缺图？

核心风险：源文件中主图 caption 有 color-only 表述；`fig:selective-contraction` 是 t-SNE qualitative；`paper1/figures` 是 symlink，arXiv 上传要复制真实图；当前环境无法直接视觉确认 PDF/PNG。

整改方向：Codex 必须本地 render PDF，并逐页检查图/表；图 2 用 markers/linestyle，不靠颜色；t-SNE caption 保留 caveat；source tarball 使用真实 PNG，不用 symlink。

### R5：References / Reproducibility Reviewer

会问：2025/2026 新文献是否准确？所有 citation 是否真的支持 text claim？公开 code/data URL 是否有效？reference audit 是否与 bib 一致？

核心风险：`references.bib` 大体已 audit，但我抽查发现 DrQ 作者顺序与官方记录不一致；Acknowledgements 的 URL 指向 `Anguo-star/le-wm/tree/ag/dev`，而本轮实际读取的是 `Holosemantix/le-wm/ag/dev`，并且我对 `Anguo-star/le-wm` 的 `paper1/main.tex@ag/dev` 读取返回 404。若最终公开仓库确实是 Anguo-star，必须同步；否则 ACK URL 应改。

整改方向：修 DrQ BibTeX；更新 `reference_audit.md`；修公开 URL 并让 `check_arxiv_ready.sh` 加 URL grep gate。

### R6：Writing / Style Reviewer

会问：是不是太像内部报告？是否有 AI caveat overload？术语是否统一？标题/小标题是否自然？英式/美式拼写是否混杂？

核心风险：稿件为了避免过 claim，很多段落反复 “not X / not Y / bounded / diagnostic”，会削弱叙事力度。另有 `repr.`, `obs-high`, `population std`, `localisation/visualization` 等混杂。

整改方向：只保留必要 caveats，把重复 caveat 合并到 Scope/Limitations；正文用更自然的科学写法；统一 spelling 和缩写。

### R7：Skeptical Control / World-Model Reviewer

会问：成功率恢复是否只是数据增强常识？ACPC 是否真的比 encoder invariance 多？“generalizable world models” 是否超出证据？

核心风险：若只说“noise augmentation improves robustness”，分量不足；若说“ACPC explains/generalizes robustness”，又过 claim。

整改方向：把故事写成“latent prediction alone is not a robustness definition; robustness for control is decided after action-conditioned rollout and cost/action margin, while discriminability prevents collapse”。这样既有方法论分量，也不夸大。

---

## 2. P0：不修不应提交的 blocking issues

### P0.1 替换 arXiv 作者占位符

**问题**：`main.tex` 仍有：

```tex
\newcommand{\arxivauthors}{Author names to be supplied for arXiv v1}
```

`check_arxiv_ready.sh` 已把该占位符作为 hard blocker。这是正确 gate，但当前状态仍会失败。

**Codex 修改**：

- 在 `paper1/main.tex` 中替换 `\arxivauthors` 为真实作者列表。
- 若目标会议需要匿名版，则另建匿名投稿分支/flag，不要把 arXiv v1 与匿名投稿混在一个 main.tex 默认状态里。
- 若只是 arXiv v1，`\author{}` 不可为空。

**验收**：

```bash
rg -n "Author names to be supplied|\\author\{\}" paper1/main.tex
bash paper1/check_arxiv_ready.sh
```

期望：第一条无命中；第二条不因作者占位失败。

---

### P0.2 修公开 code/data URL：当前 ACK URL 与实际分支不一致

**问题**：Acknowledgements 当前写：

```tex
The complete code and data for this revision are available at
\url{https://github.com/Anguo-star/le-wm/tree/ag/dev}
```

但本轮实际可读取的 paper1 是 `Holosemantix/le-wm@ag/dev`。我尝试读取 `Anguo-star/le-wm@ag/dev:paper1/main.tex` 得到 404。若最终提交仍指向 Anguo-star，会被 reproducibility reviewer 或读者直接发现。

**Codex 修改二选一**：

1. **推荐**：如果公开 release 就是当前组织仓库，改为：

   ```tex
   \url{https://github.com/Holosemantix/le-wm/tree/ag/dev}
   ```

2. 如果必须用 Anguo-star 个人仓库，则先同步 `ag/dev` 分支和 `paper1/`、assets、data manifest，再保留该 URL。

同时把 `check_arxiv_ready.sh` 加一个 URL gate，例如：

```bash
if grep -q "Anguo-star/le-wm/tree/ag/dev" main.tex; then
  fail "main.tex still points to the old Anguo-star URL; verify/sync or replace with the public release URL."
fi
```

**验收**：

```bash
rg -n "Anguo-star/le-wm|Holosemantix/le-wm|complete code and data" paper1/main.tex paper1/README.md DATA_MANIFEST.md
```

确认最终 URL 可匿名访问，且 README / acknowledgement / data manifest 不互相矛盾。

---

### P0.3 统一 `drop` 的定义和符号

**问题**：当前表格里 `drop` 符号不一致。主文 corruption cliff 表中 `drop` 是负数，例如 PushT `-82.00`；但 ACPC-basin / correlation caption 使用的是 `unperturbed - observation-noise 0.08`，PushT base 是正 `82.00`。同一篇论文里同一词一会儿为负、一会儿为正，审稿人会认为数据表不可信。

**Codex 修改**：全篇统一为一个定义。建议使用：

```text
corruption drop = clean success - observation-noise 0.08 success
```

这样“drop”是正数表示性能下降；若 noisy eval 略高于 clean，则为负数，并在 caption 中说明 `negative values mean no degradation within evaluation variability`。

需要改的地方：

- `tab:corruption-cliff`：TwoRoom `28.33`，PushT `82.00`，Reacher `40.33`，Cube `19.67`，不要带负号。
- `tab:pldm-base-cliff`：TwoRoom `29.67`，PushT `57.00`，Reacher `2.33`，Cube `4.33`。
- 所有 `drop` caption、正文“loses / drops / drop”表述。
- `tab:corr-n9`、`tab:partial-corr-4tasks` 与 appendix 中 correlation 的 outcome 名称。

**验收**：

```bash
rg -n "drop|Drop|loses|falls|degradation" paper1/main.tex
```

人工逐个确认符号一致。

---

### P0.4 标题与摘要必须重写：收窄 scope，但增强故事分量

**问题**：当前标题缺少 `Gaussian` / `diagnostic`，容易让人期待通用 JEPA robustness theory。摘要过长且信息堆叠，读者不容易抓住：理论定义、实验发现、边界各是什么。

**推荐标题**（二选一）：

1. `Action-Conditioned Predictive Consistency as a Diagnostic for Gaussian Visual Robustness in JEPA World Models`
2. `A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models`

如果想保留更强 ACPC 分量，用 1；如果目标是更稳的 empirical study，用 2。

**摘要改写目标**：170-220 words；结构为：问题 → ACPC → theory bridge → strongest empirical evidence → boundary。

**可直接替换的摘要草稿**（Codex 需要按最终标题和 drop 符号微调）：

```tex
\begin{abstract}
Latent predictive world models such as JEPAs avoid pixel reconstruction, but latent prediction alone does not define robustness for closed-loop control. We study Gaussian visual corruption as a controlled diagnostic stressor for LeWM and PLDM. We formalize action-conditioned predictive consistency (ACPC): clean and corrupted views of the same state should induce similar rollout readouts under the same action sequence, while action-relevant state differences remain separable. On a fixed candidate set, bounded ACPC implies bounded candidate-cost drift and preserves the selected action under a clean margin condition; ACPC alone admits collapse, motivating a discriminability guard.

Across PushT, TwoRoom, Reacher, and Cube, no-noise LeWM is fragile under observation-only Gaussian noise, e.g., PushT falls from $86.33\%$ to $4.33\%$ and Reacher from $58.67\%$ to $18.33\%$. Full-sequence input-noise training recovers high-corruption checkpoints, with bounded replication in PLDM. Paired ACPC-basin probes localize recovery in rollout space (PushT $R_F$: $1.543\to0.088$), while rank, transition-resolution, and inverse-dynamics probes check that contraction is not merely collapse. Negative target-view and heteroscedastic-loss ablations expose failure modes. The result is a diagnostic framework and release package, not a new training objective or a closed-loop guarantee.
\end{abstract}
```

**注意**：摘要里不要写 `generalizable world models` 太重；可以在最后一句 discussion 写成 “suggests design requirements for more generalizable world models”。

---

### P0.5 统计语言和 seed 语义必须全篇一致

**问题**：当前 paper 已写清楚 evaluation seeds，不是 independent training seeds，这是好的；但摘要/表格/正文必须避免任何让人误以为每个 cell 有 3 training seeds 的表达。

**Codex 修改**：在 `Study protocol` 第一段或 `Evaluation and diagnostics` 中保留/强化如下句子：

```tex
Each training-noise cell is a single epoch-10 trained checkpoint; the reported standard deviation is the population standard deviation across three evaluation seeds, not training-run variability.
```

**禁用词**：

- `significant`，除非有正式检验。
- `reliably predicts`, `robust predictor`, `oracle`, `selection rule`。
- `optimal sigma`，改为 `point-best under this evaluation protocol` 或 `representative high-corruption row`。

**验收**：

```bash
rg -n "significant|reliably predicts|robust predictor|oracle|selection rule|optimal sigma|independent training" paper1/main.tex
```

---

### P0.6 提交 gate 必须实际通过

**Codex 执行顺序**：

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
cd .. && bash paper1/check_arxiv_ready.sh
```

**如果失败**：不要绕过。除非明确是作者匿名版与 arXiv 版的冲突，否则修 source。

**建议增强 `check_arxiv_ready.sh`**：

- grep 旧 URL。
- grep `drop` caption 定义是否出现。
- 检查 `\includegraphics` 列表与 arXiv tarball copy list 一致，避免漏图或复制未用图。

---

## 3. P1：强烈建议提交前修复的问题

### P1.1 理论分析再增强：不要只停在 Lipschitz lemma

**当前状态**：已经有局部 Lipschitz encoder-route、cost drift、fixed-candidate top-1 stability、ACPC-margin corollary、collapse proposition。这是对的，但还可以更像论文核心。

**建议新增一个短 subsection 或 corollary：`Selective predictive stability`**

放在 `ACPC as a fixed-candidate stability condition` 后，核心是把已有 pieces 合成一个定义/命题：

```tex
\paragraph{Selective ACPC pseudo-metric.}
For a candidate family $\mathcal A$, define
\[
  d^{\mathcal A}_{\mathrm{ACPC}}(h,\tilde h)
  = \sup_{\mathbf a\in\mathcal A}
    \sum_{k=1}^H \alpha_k d\!\left(
      \Pi(F^k(E(h),\mathbf a_{0:k-1})),
      \Pi(F^k(E(\tilde h),\mathbf a_{0:k-1}))
    \right).
\]
```

然后给一个小命题：

```tex
\begin{proposition}[Selective predictive stability]
Assume $d^{\mathcal A}_{\mathrm{ACPC}}(h,\tilde h)\le \epsilon$ for same-state corruption pairs, and $J$ is $L_J$-Lipschitz. For any fixed candidate set $\mathcal A$ whose clean top-1/top-2 margin exceeds $2L_J\epsilon$, clean and corrupted branches select the same top-1 candidate. If, in addition, action-distinct state pairs satisfying the external future gap condition preserve a predictive/readout margin $m'$, then the stability is selective rather than collapse-induced.
\end{proposition}
```

这不是新 theorem 夸大，而是把当前 corollary + discriminability guard 组织成文章可引用的核心理论骨架。

**必须保留边界**：

```tex
This proposition is a fixed-candidate diagnostic condition. It does not prove stability of CEM resampling, repeated replanning, or environment feedback.
```

---

### P1.2 借鉴现有理论文献，但突出本文独有点

在 Related Work 或 theory 后新增一段 `Relation to existing theory`，建议写法：

- Van Assel et al.：latent prediction can reduce pressure to model high-magnitude irrelevant pixels, but does not decide which variations are action-relevant in control。
- LeJEPA theory：studies latent-variable recovery and latent planning under assumptions; this paper does not claim identifiability, instead evaluates whether learned checkpoints give stable action-conditioned predictions under visual perturbation。
- Bisimulation / Bisim-JEPA：control-equivalence abstractions merge states by reward/transition behavior; ACPC is same-state corruption consistency after action-conditioned rollout, plus a discriminability guard against merging action-distinct states。
- Group-action world models：identity/inverse/composition action-faithfulness; ACPC is same-action paired visual-corruption stability, not action group consistency。

**本文独有的理论分析一句话**：

```text
Our theoretical contribution is not a new identifiability theorem; it is a selective stability criterion that connects same-state visual perturbation consistency to fixed-candidate planning through rollout-readout cost drift, and proves why a discriminability guard is necessary because ACPC alone permits collapse.
```

这能解决“安全 claim 导致没分量”的问题：不是退成“只是实验”，而是把贡献精确定义为 diagnostic stability criterion。

---

### P1.3 把 discriminability guard 的证据锚点提前并显眼化

**问题**：ACPC 的一半是 same-state contraction，另一半是 action-relevant distinctions preserved。当前主文有 PushT rank/ID probe 和 heteroscedastic failure 的段落，这是对的，但还不够醒目。

**最低修改**：在 ACPC basin table 后立即加一个小段：

```tex
The contraction is not read as success by itself. PushT is the guard case: the input-noise checkpoint reduces $R_F$ while keeping effective rank and inverse-dynamics probe nearly flat, whereas the heteroscedastic-loss ablation compresses PushT rank, transition resolution, and ID-probe performance and fails in control. Thus the main claim is proxy-level selective contraction, not an oracle discriminability proof.
```

**更强修改**：加一张小表，标题类似：

```tex
Selective-contraction guard on PushT: same-state basin contraction versus action-relevant proxy preservation.
```

可列：base、input-noise row、hetero failure；列包含 `R_F`, effective rank, transition L2, ID probe, success endpoint。注意 hetero 的 success endpoint 与主 sweep 不完全相同，caption 必须说明。

---

### P1.4 代表点/point-best 选择要透明

**问题**：`tab:sweep-summary` 的 `unpert. repr.` 与 `obs 0.08 repr.` 在 PushT/Reacher/Cube 可能是不同 `std_max`。如果叫 `representative`，审稿人可能以为是同一 checkpoint 或认为 cherry-picking。

**Codex 修改二选一**：

1. **最透明**：改列名为：

   ```text
   best clean row (σmax)
   best obs0.08 row (σmax)
   ```

   并在 caption 写：

   ```text
   The two point-best columns may refer to different training-noise levels; they summarize the sweep envelope rather than a single selected checkpoint.
   ```

2. **更稳健**：每个任务选一个 `diagnostic row σmax`，同一 checkpoint 同时报 clean / obs0.08，然后把 point-best 放 appendix。

如果篇幅允许，推荐 2；如果想保留主文冲击力，推荐 1 但必须透明。

---

### P1.5 Cross-checkpoint correlation 表要降温

**问题**：主文 `tab:corr-n9` 同时给 Pearson / Spearman unconditional，数值很大，但随后又说它们主要由 `std_max` 驱动。审稿人可能只看表，以为作者先卖强相关，再 disclaim。

**建议**：

- 主文只保留 partial Spearman table。
- unconditional Pearson/Spearman 移到 appendix，标题改成：

  ```tex
  Sweep-trend correlations before conditioning on training-noise level.
  ```

- 主文一句话：

  ```tex
  Unconditioned correlations are reported in Appendix~X and are dominated by the monotone training-noise sweep; therefore the main diagnostic check is partial Spearman conditioned on \stdmax{}.
  ```

这样更诚实，也更不容易被统计 reviewer 打。

---

### P1.6 Figure/Table/PDF 本地视觉 preflight 必须做

本轮没有直接读取二进制 PDF/PNG，因此 Codex 必须执行以下视觉检查并根据结果修图：

```bash
cd paper1
bash build.sh --clean
mkdir -p /tmp/paper1_pages
pdftoppm -r 200 -png main.pdf /tmp/paper1_pages/page
pdfinfo main.pdf
rg -n "Overfull|Underfull|Float too large|LaTeX Warning|undefined|Citation .* undefined|Reference .* undefined" main.log || true
```

人工检查：

- 首页标题、作者、abstract 不拥挤。
- 所有主文表格没有超页宽、没有过小字号、没有奇怪 float 空白。
- `landscape` appendix 页在 PDF viewer 中正确旋转，不被裁切。
- `fig2_sweep.png` 不只依赖蓝/红；增加 markers/linestyles 或 caption 写明 markers。
- `fig:selective-contraction` 的 t-SNE 文本、legend、panel summary 可读；caption 明确 t-SNE qualitative。
- `fig:selective-contraction-atlas` 圆/点/局部 PCA 说明与图中标注一致。
- corruption visualization 每个 panel 确实是 caption 所说的 `2 x 4` grid，顺序是 Original / sigma / blur kernels。
- PDF 中没有黑块、缺字、figure clipping。

如果 `paper1/figures` 是 symlink，arXiv source tarball 必须复制真实 PNG，而不是上传 symlink。

---

### P1.7 修正 DrQ 参考文献作者顺序并更新 audit

**问题**：`references.bib` 当前：

```bibtex
@inproceedings{kostrikov2020drq,
  title = {Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author = {Yarats, Denis and Kostrikov, Ilya and Fergus, Rob},
  ...
}
```

官方记录作者顺序应为 `Kostrikov, Ilya and Yarats, Denis and Fergus, Rob`。这不是大科学问题，但 reference audit 声称已全量检查，若被发现会伤 credibility。

**Codex 修改**：

```bibtex
author = {Kostrikov, Ilya and Yarats, Denis and Fergus, Rob},
```

然后更新 `paper1/reference_audit.md` 增加 2026-06-26 targeted fix 说明。

**验收**：

```bash
cd paper1 && bash build.sh --clean
```

并检查 PDF bibliography 中 DrQ 作者顺序。

---

### P1.8 Acknowledgement 的 “complete code and data” 不能过度承诺

如果最终 release 只包含代码、aggregate JSON、manifests、fig scripts，而 raw datasets/checkpoints 在 HuggingFace/Drive 或另一个位置，那么 `complete code and data` 可能过强。

建议改成更准确：

```tex
Code, aggregate evaluation artifacts, rendering scripts, and data/checkpoint pointers for this revision are available at ...; artifact hashes are listed in the data manifest.
```

如果确实所有 raw data/checkpoints 都公开，则保留 `complete` 也可以，但要在 README 里给出可点击路径。

---

## 4. P2：写作、用词、小标题与风格 polish

### P2.1 统一英式/美式拼写

当前有 `localise / regularisation / behaviour` 与 `visualization` 混用。建议全篇统一为美式 ML 写法：

- localize, localization
- regularization
- behavior
- analyze
- visualizations

若保留英式，也要全篇一致。

### P2.2 少用缩写 `repr.` / `obs-high` / 内部口吻

- `unpert. repr.` 改成 `best clean row` 或 `representative clean row`。
- `obs-high` 改成 `representative obs0.08 row`。
- `paper-facing`, `method-facing`, `arXiv version`, `release-note` 不应出现在 `main.tex`。

### P2.3 减少重复 caveat，保留关键边界

该保留：

- 不证明 CEM / closed-loop。
- 不是 independent training seeds。
- ACPC basin 是 post-hoc localization，不是 predictor。
- discriminability 是 proxy-level。

可压缩：

- 每节重复 `diagnostic not predictor`。
- 每个结果段都说 `not universal`。

建议集中在 Scope / Discussion 里讲边界，结果段更直接讲发现。

### P2.4 小标题建议

当前整体可读，但可略调：

- `Latent prediction is not a robustness definition` 很好，保留。
- `Existing corruption results as a controlled probe` 可改为 `Gaussian corruption as a controlled diagnostic probe`。
- `Noise augmentation supplies recovered checkpoints for diagnosis` 可改为 `Input noise creates a recovery grid for diagnosis`。
- `Diagnostic analysis: what changes under the intervention` 可改为 `What changes: rollout contraction with discriminability checks`。
- `Cross-checkpoint correlation analysis` 可改为 `Diagnostic metrics are local explanations, not selection rules`。

---

## 5. 建议后的主文结构

如果不换模板，建议主文组织为：

1. **Introduction**（1.25-1.5 页）
   - Cliff motivation。
   - ACPC intuition。
   - Contributions，三条即可。
2. **Action-Conditioned Predictive Consistency**（1.5 页）
   - Definition。
   - Main instantiation: identity rollout latent, L2, H=8。
   - Fixed-candidate stability theorem/corollary。
   - Collapse proposition / discriminability guard。
3. **Related Work**（0.75-1 页）
   - JEPA/latent prediction。
   - robustness/augmentation。
   - control-relevant representations / theory boundary。
4. **Study Protocol**（0.5 页）
   - tasks, LeWM/PLDM, Gaussian noise, eval seeds。
5. **Results**（3-4 页）
   - corruption cliff。
   - recovery grid。
   - ACPC basin localization。
   - discriminability and cross-checkpoint boundary。
6. **Discussion and Limitations**（1 页）
7. **Conclusion**（短，不重复 abstract）

Appendix 保留：full sweeps、full ACPC grid、fig rendering details、artifact mapping、five-layer diagnostics、PushT partial, hetero, PLDM, blur, Phase-0, target-view。

---

## 6. 不要改坏的点

这些点是正确的，不要为了“增强”而改坏：

1. **不要删除 claim 边界**。fixed-candidate / no CEM / no closed-loop guarantee 是必要保护。
2. **不要把 ACPC 写成 robustness predictor**。partial correlation 明确不支持 metric-as-oracle。
3. **不要隐藏 PLDM Reacher/Cube boundary cases**。它们让文章更可信。
4. **不要删除 heteroscedastic / target-view 负结果**。它们是故事分量，不是累赘。
5. **不要手工改 JSON artifact**。任何数据变更必须由 builder/checker 产生。
6. **不要把 proxy discriminability 写成 oracle state margin proof**。
7. **不要新增没有 artifact 支撑的 SOTA/baseline claim**。

---

## 7. Codex 执行 patch 顺序

### Step 1：提交硬阻断

- 替换作者占位符。
- 修公开 URL / ACK wording。
- 统一 drop 符号。
- 重写标题和摘要。

### Step 2：理论增强

- 新增 `Selective ACPC pseudo-metric` 与 selective predictive stability proposition/corollary。
- 新增 `Relation to existing theory` 段。
- 保留 fixed-candidate/CEM 边界。

### Step 3：表格与代表点透明化

- 改 `tab:sweep-summary` 列名或改为同一 checkpoint diagnostic row。
- 主文 correlation 降温。
- ACPC basin table caption 明确 post-hoc localization。

### Step 4：discriminability guard 锚点

- 加 PushT guard paragraph 或 mini-table。
- 检查 hetero endpoint caption 不混用 obs-only 与 obs+goal。

### Step 5：参考文献与 audit

- 修 DrQ 作者顺序。
- 更新 `reference_audit.md`。
- 跑 cite/bib consistency check。

### Step 6：PDF/figure/arXiv preflight

- 本地 render PDF pages。
- 检查图表可读、无裁切。
- 生成 arXiv source tarball，确认没有内部文件/缺图/symlink。

---

## 8. 最终验收命令

```bash
# 0. Search for known blockers
rg -n "Author names to be supplied|Anguo-star/le-wm|significant|reliably predicts|robust predictor|robustness oracle|full CEM|closed-loop guarantee|optimal sigma|method-facing|paper-facing|Scope of this arXiv version" paper1/main.tex paper1/README.md || true

# 1. Check drop wording manually
rg -n "drop|Drop|drops|loses|falls|degradation" paper1/main.tex

# 2. Citation key consistency
python - <<'PY'
import re, pathlib
tex = pathlib.Path('paper1/main.tex').read_text()
bib = pathlib.Path('paper1/references.bib').read_text()
cited = set(k.strip() for m in re.finditer(r'\\cite\{([^}]*)\}', tex) for k in m.group(1).split(','))
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
print('missing:', sorted(cited - keys))
print('unused:', sorted(keys - cited))
PY

# 3. Artifact consistency
python -m tools.check_paper1_consistency

# 4. Build paper
cd paper1 && bash build.sh --clean

# 5. Visual render preflight
mkdir -p /tmp/paper1_pages
pdftoppm -r 200 -png main.pdf /tmp/paper1_pages/page
pdfinfo main.pdf
rg -n "Overfull|Underfull|Float too large|LaTeX Warning|undefined|Citation .* undefined|Reference .* undefined" main.log || true

# 6. arXiv readiness
cd .. && bash paper1/check_arxiv_ready.sh

# 7. Check source bundle contents
if [ -f /tmp/paper1_arxiv_v1_src.tar.gz ]; then
  tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | sort
fi
```

---

## 9. 最终投稿前人工 checklist

- [ ] 标题明确 `diagnostic` / `Gaussian`，不会被误读为通用 JEPA robustness theory。
- [ ] 摘要 170-220 words，包含 ACPC、fixed-candidate theory、2-3 个强结果、边界。
- [ ] 作者列表真实；ACK URL 可访问；data/checkpoint pointers 清楚。
- [ ] `drop` 全文符号一致。
- [ ] 每个表格 caption 写明 std 是 across evaluation seeds。
- [ ] 每个 representative/point-best 选择透明。
- [ ] ACPC basin 不被写成 predictor/model-selection rule。
- [ ] Discriminability guard 在主文有证据锚点。
- [ ] Theory section 有独立可引用的 selective stability criterion。
- [ ] Related work 正确区分 Van Assel / LeJEPA / Bisim-JEPA / group action / ACPC。
- [ ] DrQ 作者顺序已修；reference audit 已更新。
- [ ] PDF 每页视觉检查通过；图中文字可读；图 2 不只靠颜色。
- [ ] arXiv source tarball 不含 internal planning docs、build logs、main.pdf、symlink。
- [ ] `check_arxiv_ready.sh` 通过。

---

## 10. 建议最终中心 claim

投稿时建议在 intro/conclusion 中把中心 claim 固定为：

> Latent prediction alone is not a robustness definition for world-model control. Robustness should be diagnosed after action-conditioned prediction: same-state visual perturbations should induce stable rollout/cost readouts under the same action sequence, while action-relevant state differences remain separable. Under a fixed candidate set, this ACPC condition bounds candidate-cost drift and preserves the selected candidate under a margin condition; because ACPC alone permits collapse, discriminability guards are necessary. The LeWM/PLDM Gaussian-noise sweeps, paired ACPC-basin artifacts, and negative ablations support this diagnostic principle without claiming a new training objective or full closed-loop guarantee.

这句话既有分量，也基本不会被当前证据挑战。
