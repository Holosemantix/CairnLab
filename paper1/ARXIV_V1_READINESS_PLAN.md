# Paper 1 arXiv v1 整改规划（交给 Codex 执行）

目标：把 `paper1` 改成可以先公开挂 arXiv 的 v1。这个版本不追求“主会方法论文”完成度，而追求：**公开表述稳、理论动机够、主文收敛、源码包可编译、没有明显夸大或 AI 感**。

本文最安全的 arXiv v1 定位：

> A controlled diagnostic study of Gaussian visual robustness in LeWM/PLDM-style JEPA latent world models, using action-conditioned predictive consistency as the organising diagnostic lens.

不要把 v1 写成：

- 新训练算法论文；
- 一般 JEPA 鲁棒性理论；
- robustness benchmark；
- ACPC 指标能预测 robustness 的选择规则；
- independent training-seed study。

---

## 0. arXiv v1 必须满足的硬标准

### 0.1 内容硬标准

1. **标题和摘要收窄。** 读者第一眼必须知道这是 controlled diagnostic study，不是 claim 解决 JEPA visual robustness。
2. **补一个轻量 formal section。** 不需要复杂 theorem，但要把 ACPC 和 planner/ranking stability 形式化连起来。
3. **主文压缩。** 主文保留四个结果：corruption cliff、noise sweep、ACPC basin、diagnostic limitation。长定义和长表移 appendix。
4. **边界集中说明。** 不要全文反复 caveat；在 Limitations 或一个 arXiv v1 scope box 中集中说明。
5. **杜绝 AI 感。** 少用排比式自我辩护，少用抽象口号，少用 “not X but Y” 连环句。
6. **release gate 可信。** 本地 consistency checker、LaTeX build、citation/reference grep、arXiv tarball audit 都要通过。

### 0.2 arXiv 官方提交硬标准

Codex 不要凭记忆改；以 arXiv 官方说明为准。当前需要遵守：

- arXiv metadata 中 `Title`、`Authors`、`Abstract` 是 required fields。
- arXiv 不接受匿名提交，作者信息必须完整准确；生成式 AI 工具不得列为作者。
- TeX/LaTeX source package 不要包含无关文件、备份文件、旧输出、unused figures、referee letters、内部计划文档。
- PDFLaTeX 场景下图像使用 PDF/PNG/JPG，不要依赖 arXiv 运行图像格式转换。
- 若使用 BibTeX/Biber，可以上传 `.bib` 或预生成 `.bbl`；如果上传 `.bbl`，文件名必须和主 `.tex` 匹配，例如 `main.tex` 对 `main.bbl`。
- 不建议使用 `\today` 作为公开版本日期，因为 arXiv 重编译时日期可能变化。

官方参考：

- https://info.arxiv.org/help/prep.html
- https://info.arxiv.org/help/submit_tex.html
- https://arxiv.org/category_taxonomy

---

## 1. 推荐 arXiv v1 标题、分类和 metadata

### 1.1 标题备选

优先使用更稳的诊断标题。

**首选：**

```text
A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models
```

**若要保留 ACPC 亮点：**

```text
Action-Conditioned Predictive Consistency as a Diagnostic for Visual Robustness in JEPA World Models
```

**若要保留原题风格但收窄：**

```text
Diagnosing Gaussian Visual Robustness in JEPA World Models through Action-Conditioned Predictive Consistency
```

不要使用暗示 general solution 的标题，例如：

```text
Solving Visual Robustness in JEPA World Models
Robust JEPA World Models through Action-Conditioned Predictive Consistency
```

### 1.2 arXiv 分类建议

建议：

```text
Primary: cs.LG
Cross-list candidates: cs.RO, eess.SY
```

理由：

- `cs.LG` 明确覆盖 machine learning 的 robustness、reinforcement learning、methodology。
- `cs.RO` 可作为 robotics/control task cross-list，但不建议 primary，除非论文更强调机器人任务。
- `eess.SY` / `cs.SY` 覆盖 automatic control、robust control、reinforcement learning、robotics、modeling/simulation/optimization，可作为控制侧 cross-list。

不建议 primary 放 `cs.CV`，因为本文核心不是视觉识别或视觉模型 benchmark，而是 latent world-model control robustness。

### 1.3 author/date/comments

必须处理：

- `\author{}` 不能空。
- 若之前有匿名会议版，arXiv 版要单独切换到实名作者。
- `\date{}` 使用固定日期，例如 `June 2026` 或具体 arXiv v1 日期，不用 `\today`。
- arXiv `Comments` 字段建议写短句：

```text
Technical report; diagnostic study; code and data manifest included in repository
```

不要在 `Comments` 里写过长 rebuttal 式说明。

---

## 2. 摘要重写任务

### 2.1 摘要长度和结构

目标：170--210 words。

结构：

1. 一句话说明问题：latent prediction does not by itself define visual robustness for control。
2. 一句话定义 ACPC diagnostic：same-state corrupted/clean views should agree after action-conditioned prediction while preserving action-relevant distinctions。
3. 一句话说明实验：LeWM 36 checkpoints + PLDM replication under Gaussian observation noise。
4. 一句话给最强数字：PushT cliff、Reacher/TwoRoom recovery、PushT `R_F` contraction。
5. 一句话说明边界：diagnostic framing, no new training algorithm, no independent training-seed claim。

### 2.2 摘要禁忌

不要写：

- `we establish`；
- `we prove`，除非指轻量 proposition；
- `robustness oracle`；
- `method-invariant`；
- `benchmark`；
- `optimal noise level`；
- `predicts robustness`。

### 2.3 摘要草稿模板

Codex 可以据此改，不要逐字照搬；数值需与当前 `main.tex` 和 artifact 保持一致。

```text
Latent predictive world models such as JEPAs predict future representations rather than pixels, but this objective alone does not specify visual robustness for closed-loop control. We study this gap as a diagnostic problem through action-conditioned predictive consistency (ACPC): corrupted and clean views of the same state may encode differently, but under the same action sequence their task-relevant predicted futures should agree, while action-relevant state distinctions remain separable. In a controlled Gaussian-noise study on PushT, TwoRoom, Reacher, and Cube, we evaluate 36 epoch-10 LeWorldModel checkpoints with three evaluation seeds per cell and replicate the main signatures with PLDM. No-noise LeWM drops sharply under observation-only noise, for example from 86.33% to 4.33% on PushT and from 58.67% to 18.33% on Reacher. Full-sequence input-side noise augmentation recovers much of this loss, but the best recovery point and plateau structure vary by task. Post-hoc paired ACPC-basin probes localise the recovery: direct-evaluation-selected robust endpoints have smaller same-action prediction radii, e.g. PushT RF decreases from 1.543 to 0.088. The paper contributes a diagnostic framing, release artifacts, and negative ablations; it does not propose a new training algorithm or estimate independent training-run variability.
```

注意：如果用这个模板，把 `RF` 改成 LaTeX 里的 `$R_F$`，并检查 word count。

---

## 3. 轻量理论补强：新增 formal section

### 3.1 放置位置

建议放在 ACPC 定义之后、Experiments 之前：

```tex
\subsection{A simple planner-stability view}\label{sec:acpc-planner-stability}
```

目标长度：0.75--1.25 页。

目的：不是把论文改成理论论文，而是回答读者的核心疑问：**ACPC 为什么和 closed-loop planning 有关？为什么还需要 discriminability guard？**

### 3.2 必须包含的三个元素

1. Proposition 1：ACPC bounds candidate-cost drift。
2. Proposition 2：large action margin prevents top-1 candidate flip。
3. Two counterexamples：encoder closeness neither necessary nor sufficient；ACPC alone admits collapse。

### 3.3 Proposition 1 模板：ACPC 到 cost drift

需要定义候选 action sequence 成本：

```tex
C_h(\mathbf a, g)
= J\!\left(\Pi(\hat z_{t+1}), \ldots, \Pi(\hat z_{t+H}), g\right),
```

其中：

- `h` 是 clean history；
- `\tilde h` 是 corrupted history；
- 两边使用同一个 candidate action sequence `\mathbf a`；
- `J` 是 planning cost readout；
- `\Pi` 是 paper-facing rollout readout。

形式化陈述：

```tex
\begin{proposition}[ACPC controls cost drift]
Assume that the planning readout $J$ is $L_J$-Lipschitz in the rollout-readout sequence under metric $d_H$. If
\[
 d_H\!\left(\Pi(F^{1:H}(E(h),\mathbf a)),
             \Pi(F^{1:H}(E(\tilde h),\mathbf a))\right) \le \epsilon,
\]
then
\[
 |C_h(\mathbf a,g)-C_{\tilde h}(\mathbf a,g)| \le L_J\epsilon.
\]
\end{proposition}
```

证明一句即可：由 Lipschitz continuity 直接得到。不要装成深理论。

### 3.4 Proposition 2 模板：margin 到 ranking stability

形式化陈述：

```tex
\begin{proposition}[Candidate top-1 stability]
Let $\mathcal A=\{\mathbf a^1,\ldots,\mathbf a^K\}$ be the shared CEM candidate set, and suppose that for every candidate $j$,
\[
 |C_h(\mathbf a^j,g)-C_{\tilde h}(\mathbf a^j,g)|\le \eta.
\]
If the clean branch has top-1/top-2 margin
\[
\Delta = C_h(\mathbf a^{(2)},g)-C_h(\mathbf a^{(1)},g) > 2\eta,
\]
then the corrupted branch selects the same top-1 candidate.
\end{proposition}
```

证明：对 clean top-1 和任意其他 candidate 用 triangle inequality；如果 margin 大于两倍扰动，排序不会翻转。

解释：这把 ACPC、PCC、CRA、MAF 串起来：ACPC 控制 candidate cost drift；cost drift 小且 margin 大时 action flip 少。

### 3.5 Counterexample 1：encoder closeness 既非必要也非充分

简短写法：

- 非必要：存在 invertible or nuisance subspace transformation，使 `E(h)` 与 `E(\tilde h)` 远，但 `\Pi(F^k(E(h),a)) = \Pi(F^k(E(\tilde h),a))`，planning cost 不变。
- 非充分：`E(h)` 与 `E(\tilde h)` 近，但 predictor/cost 在该方向 Lipschitz 常数大，或者 clean margin 很小，导致 top candidate 翻转。

不要展开成复杂数学；这段是为了支撑“encoder invariance 不是正确目标”。

### 3.6 Counterexample 2：ACPC without discriminability collapses

写法：

```tex
A constant encoder and predictor can make same-state ACPC zero for all perturbation pairs, because every rollout readout is identical. Such a model violates the discriminability condition: states requiring different actions, costs, or transitions are merged. Therefore a low ACPC score is meaningful only together with a guard on action-relevant separation.
```

这段要和 heteroscedastic negative result 呼应：PushT 中错误的 downweighting 可以让 rollout drift 看起来更小，却抹掉 controllability。

### 3.7 Formal section 验收标准

- 没有复杂新符号泛滥。
- 不声称 theorem 解释所有实验。
- 明确说 propositions are sufficient conditions for candidate-set stability, not guarantees for full CEM distributional stability。
- 读者读完后能明白为什么 ACPC 和 planner 有关。

---

## 4. 主文压缩计划

### 4.1 推荐主文结构

```text
1 Introduction                         1.25--1.50 pages
2 Related Work                         0.75--1.00 page
3 ACPC Diagnostic and Planner View     1.25--1.75 pages
4 Experimental Setup                   0.50 page
5 Results                              3.00--3.75 pages
6 Discussion and Limitations           1.00--1.25 pages
7 Conclusion                           <=0.30 page
```

目标：主文 9--11 页左右；appendix 可以长。

### 4.2 主文保留四个结果

1. **Corruption cliff**：base LeWM under observation-only Gaussian noise。
2. **Noise sweep**：input-side full-sequence augmentation recovers but with task-dependent plateau/point-best。
3. **ACPC basin**：direct-eval-selected endpoints have smaller paired prediction radii。
4. **Diagnostic limitation**：partial correlation after conditioning on `\stdmax{}` is weak; diagnostic is not a model-selection oracle。

### 4.3 移到 appendix 的内容

移动而非删除：

- Related-work boundary table。
- Full five-layer diagnostic framework definitions。
- PushT detailed partial-correlation table。
- Full LeWM-base diagnostics landscape table。
- PCC/CRA/MAF/ADM/SPRR long definitions and Phase-0 table。
- Blur stress long table。
- Target-view ablation long discussion。
- Reference audit details；主文只留 code/data/reproducibility note。

### 4.4 应删除或合并的重复内容

Codex grep 后人工处理：

```bash
rg -n "not a|does not|do not claim|scope boundary|robustness oracle|method-invariant|diagnostic framing|mechanism localisation|coarse pressure|deliberately|crucially|paper-facing|control-facing" paper1/main.tex
```

原则：

- 每个 limitation 在主文最多出现一次。
- 技术限制集中放到 `Limitations`。
- 不要每节末尾都自我降级。

---

## 5. ACPC basin 全网格补充分析

### 5.1 为什么要补

当前 base-vs-point-best 很容易被读成 post-hoc cherry-picking。即使主文说不是 predictor，arXiv 读者也会问：full 9-level grid 怎么样？

### 5.2 最低实现

如果已有 `assets/paper1_data/acpc_basin_diagnostics.json` 和 PLDM basin artifact，优先使用现有数据生成：

1. Appendix table：每个 task 的 9 个 `std_max`，列出 `obs 0.08 success`、`drop`、`R_E`、`R_F`。
2. Optional figure：`R_F` vs `obs 0.08 success` 或 `R_F` vs `drop`，按 task 分 facet 或小 multiples。

### 5.3 主文写法

主文只写：

```text
The base-vs-selected endpoint table is a compact summary. The full 9-level basin grid is reported in Appendix X. We do not use the basin score to select checkpoints; the score is used to localise the change after closed-loop evaluation.
```

如果 full-grid correlation 弱，不要隐藏。写成：

```text
Across the full grid, basin radius is not a reliable standalone predictor of success, consistent with our use of the probe as mechanism localisation rather than model selection.
```

---

## 6. Limitations box

新增一个短 subsection 或 boxed paragraph：

```tex
\paragraph{Scope of this arXiv version.}
This version is a controlled diagnostic study. The canonical LeWM grid uses three evaluation seeds per checkpoint, not independent training seeds. Gaussian pixel noise is the main training and evaluation axis; blur is an eval-only sanity check. The ACPC basin probe is computed after closed-loop endpoint selection and is not a model-selection rule. The discriminability condition is tested through proxy diagnostics rather than oracle state margins. The main mechanism analysis is LeWM-centred, with PLDM used as a replication and boundary check.
```

放置建议：Discussion 开头或 Limitations 开头。这样可以删除散落在全文里的重复 caveat。

---

## 7. 去 AI 感写作任务

### 7.1 风格规则

Codex 修改时遵守：

- 一段只讲一个点。
- 首句直接给技术结论。
- 少用 “This is why / Importantly / Crucially / Deliberately”。
- 少用排比式三连 caveat。
- 不要过度使用 em dash。
- 减少抽象名词堆叠：`control-facing reading`、`method-facing probes`、`scope-boundary evidence`。
- 多用具体对象：`the paired rollout radius`、`the cost ranking`、`the PushT contact states`。

### 7.2 替换例子

**原：**

```text
The reading is deliberately narrow.
```

**改：**

```text
This comparison has one purpose: to localise the change at checkpoints already selected by closed-loop evaluation.
```

**原：**

```text
The toolkit is therefore an unperturbed-evaluation auxiliary for mechanism localisation and checkpoint triage within a fixed protocol, not a robustness oracle.
```

**改：**

```text
The diagnostics can help triage checkpoints within this protocol, but final model choice still requires closed-loop corruption evaluation.
```

**原：**

```text
This pattern is consistent with a coarse global pressure, not with a selective objective that knows which visual variation is nuisance and which controls future dynamics.
```

**改：**

```text
A single Gaussian-noise strength recovers performance unevenly across tasks, suggesting that the augmentation does not encode the task-specific nuisance/action distinction.
```

### 7.3 Grep 清单

```bash
rg -n "crucially|deliberately|importantly|scope boundary|robustness oracle|method-invariant|control-facing|paper-facing|mainline|coarse pressure|not a .* proof|not a .* theorem|does not claim|we do not claim|This is why" paper1/main.tex
```

验收标准：主文命中显著减少，剩余命中都是真必要。

---

## 8. arXiv source package 准备

### 8.1 不要直接上传整个 repo

arXiv source tarball 只放论文编译必需文件。不要包含：

- `PLAN.md`；
- `CODEX_SUBMISSION_READINESS.md`；
- `ARXIV_V1_READINESS_PLAN.md`；
- `paper1_acpc_rewrite_execution_plan.md`；
- checker logs；
- old PDFs or intermediate outputs；
- unused figures；
- raw experiment JSON，除非作为 ancillary files 单独提交；
- private absolute paths；
- `.git`、`.DS_Store`、临时文件。

### 8.2 必需文件

建议 source package 包含：

```text
main.tex
references.bib
main.bbl
figures/<only used figures>
```

如果 `main.tex` 依赖其他 local style/macro 文件，也必须包含。当前似乎是 standalone article class，不应额外需要 style file。

### 8.3 打包命令建议

当前 `figures` 是 symlink 到 `../assets/paper1_figs/`。打包时必须展开成真实目录。

建议在 `paper1/` 下执行：

```bash
bash build.sh --clean
mkdir -p /tmp/paper1_arxiv_src
rsync -avL main.tex references.bib main.bbl figures/ /tmp/paper1_arxiv_src/
cd /tmp/paper1_arxiv_src
find . -type f | sort
cd /tmp

tar -czf paper1_arxiv_v1_src.tar.gz -C /tmp/paper1_arxiv_src .
```

如果不用 `rsync -L`，也可以：

```bash
tar -czhf paper1_arxiv_v1_src.tar.gz main.tex references.bib main.bbl figures/
```

但要确认 `figures/` symlink 被展开。

### 8.4 arXiv source audit

打包前运行：

```bash
cd paper1
bash build.sh --clean
rg -n "undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence|No file main.bbl" main.log || true
```

打包后检查：

```bash
tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | sort
```

确认没有：

- `.aux`, `.log`, `.out`, `.toc`, `.fls`, `.fdb_latexmk`, `.synctex.gz`；
- `main.pdf`，除非你决定 PDF-only 另走流程；TeX source submission 通常不需要放 output PDF；
- Codex/plan/internal markdown；
- unused figures；
- hidden files。

---

## 9. Release gate 修改

### 9.1 `build.sh` 不要吞 BibTeX 错误

当前若有 `bibtex main || true`，需要改。建议逻辑：

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

若担心无 citation 的场景，先检测 `.aux` 中是否有 `\citation`，但 Paper 1 有 bibliography，所以不需要吞错误。

### 9.2 README 和 reference audit 一致

- 不要在 README 写死 `41 entries` 或 `42 keys`。
- 改成：`references.bib # bibliography; final source audit is tracked in reference_audit.md`。
- 若 citation keys 变化，更新 `reference_audit.md` 和 `references.bib`。

### 9.3 final code URL

主文 acknowledgement 里的 code/data URL 必须是最终公开仓库。若最终公开地址不是当前 `Holosemantix/le-wm`，需要明确：

- arXiv v1 用哪个公开 URL；
- artifacts 是否在该 URL；
- manifest hash 是否匹配公开内容。

不要写临时私有仓库或未同步地址。

---

## 10. Codex 执行顺序

### Step 1：基线检查

```bash
python -m tools.check_paper1_consistency | tee /tmp/paper1_check_before.log
cd paper1 && bash build.sh --clean | tee /tmp/paper1_build_before.log
```

记录是否通过，不要隐瞒失败。

### Step 2：结构改写

按顺序改：

1. title/authors/date/hypersetup metadata；
2. abstract；
3. contributions；
4. ACPC section；
5. new planner-stability formal subsection；
6. experiments claim wording；
7. discussion/limitations；
8. appendix relocation。

### Step 3：全网格 ACPC appendix

如果现有 artifact 支持，生成 appendix table/figure；若脚本已有则复用，不新造数字。

### Step 4：release 文件

修改：

- `paper1/build.sh`；
- `paper1/README.md`；
- `DATA_MANIFEST.md` only if artifact/hash changes；
- `paper1/reference_audit.md` only if bibliography changes。

### Step 5：去 AI 感 pass

运行 grep 清单，人工改写主文重复 caveat。

### Step 6：构建和打包

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence|No file main.bbl" main.log || true
```

准备 `/tmp/paper1_arxiv_v1_src.tar.gz` 并列出内容。

### Step 7：Codex 最终报告

Codex 完成后必须报告：

- changed files；
- title/abstract 是否已改；
- formal section 加在哪里；
- 哪些内容移到 appendix；
- commands run and results；
- arXiv tarball contents；
- unresolved items, especially whether independent training seeds were not added。

---

## 11. arXiv v1 验收 checklist

### Paper claim

- [ ] 标题不暗示 solved robustness 或 universal JEPA theory。
- [ ] 摘要 170--210 words，边界清楚，不像 rebuttal。
- [ ] Contributions 是 diagnostic study，不是 method paper。
- [ ] 明确没有 independent training-seed claim。
- [ ] ACPC basin 是 post-hoc diagnostic，不是 predictor。
- [ ] Discriminability guard 有形式化 counterexample 和实验 proxy evidence。

### Theory/formal section

- [ ] 有 ACPC bounds cost drift proposition。
- [ ] 有 margin-based top-1 stability proposition。
- [ ] 有 encoder closeness neither necessary nor sufficient counterexample。
- [ ] 有 ACPC-without-discriminability collapse counterexample。
- [ ] 不声称 propositions 证明 full CEM closed-loop stability。

### Empirical presentation

- [ ] 主文只保留四个核心结果。
- [ ] Full-grid ACPC basin 至少在 appendix 透明展示，若 artifact 支持。
- [ ] Partial correlation `n=9` 被写成 small-sample diagnostic check。
- [ ] PLDM 是 replication/boundary，不是 leaderboard。
- [ ] Blur 是 eval-only sanity check，除非新增 blur training。

### Writing

- [ ] 主文重复 caveat 明显减少。
- [ ] 每段只讲一个技术点。
- [ ] 不过度使用 `deliberately / crucially / scope boundary / robustness oracle`。
- [ ] 术语表不压在主文；长工具定义进 appendix。

### arXiv technical

- [ ] `\author{}` 非空且是真实作者。
- [ ] `\date{}` 不用 `\today`。
- [ ] `main.bbl` 存在且与 `main.tex` 匹配。
- [ ] `bash build.sh --clean` 通过。
- [ ] 无 undefined citation/reference。
- [ ] final code/data URL 正确。
- [ ] source tarball 只含编译必需文件。
- [ ] `figures/` symlink 已展开。
- [ ] 没有 Codex/internal planning docs 进入 tarball。

---

## 12. 非目标

arXiv v1 不要求完成以下事项，但必须如实写入 limitations 或 future work：

- 不要求补完整 independent training-seed grid。
- 不要求提出并训练新的 ACPC objective。
- 不要求证明 method-invariant theorem。
- 不要求把 ACPC 变成 model-selection rule。
- 不要求 Gaussian-noise training 泛化到 blur 或其他 corruption，除非新增实验支持。

如果 Codex 发现没有对应 artifact，不要把这些写成已经完成。
