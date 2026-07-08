# Paper 1 投稿就绪整改清单（交给 Codex 执行）

目标：把 `paper1` 从“可读的 diagnostic 技术报告”改到“可以直接投稿的、claim 收敛、写作自然、release gate 可信”的状态。

当前最安全的投稿定位是 **diagnostic empirical study / robustness analysis**，不是方法论文。若目标是 NeurIPS/ICLR/ICML 主会，独立训练种子、held-out diagnostic validation 和更多 baseline 是强烈建议项；若没有新增算力，则必须通过标题、摘要、贡献和 limitations 把 claim 收紧到 controlled LeWM/PLDM Gaussian-noise diagnostic study。

---

## 0. 执行原则

1. **不伪造实验。** 没有跑出的新 artifact 不写成结果；没有独立训练种子就不能暗示 training-run 稳定性。
2. **不手工改 canonical JSON。** 任何数据变动必须由对应 builder 产生，并更新 manifest/hash/checker。
3. **主文只保留核心证据。** 其余术语、Phase-0 readouts、长表、历史修复说明放到 appendix 或 reproducibility note。
4. **杜绝“AI 感”。** 少用排比式 caveat、反复的 “not X but Y”、过多 “crucially / deliberately / importantly / robust oracle / scope boundary”。每段只说一个技术点。
5. **投稿状态以 release gate 为准。** 最终必须能通过 consistency checker、LaTeX build、citation/reference grep、artifact path audit。

---

## 1. P0：不做就不应投稿的修改

### 1.1 收窄标题、摘要和贡献

**问题。** 当前题目和摘要容易被读成“JEPA world models 的一般鲁棒性理论”。实际证据是 LeWM-centered，PLDM replication，Gaussian pixel noise 为主轴，blur 只是 eval-only sanity check。

**建议标题二选一：**

- `A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models`
- `Action-Conditioned Predictive Consistency as a Diagnostic for Visual Robustness in JEPA World Models`

若保留原题，副标题或 abstract 第一段必须明确：`controlled LeWM/PLDM sweeps under Gaussian pixel noise`。

**摘要改写要求：**

- 目标长度 170--210 words。
- 结构：问题 → ACPC diagnostic → 关键实验现象 → 边界。
- 摘要只保留 2--3 个最强数值：PushT base cliff、Reacher or TwoRoom recovery、PushT `R_F` contraction。
- 只出现一次“不是新算法/不是训练种子研究”的 caveat。
- 不要把 ACPC basin 写成 predictor；写成 `post-hoc paired diagnostic` 或 `mechanism-localisation probe`。

**贡献改写要求：**

主文贡献只保留三条：

1. Diagnostic framing: ACPC + discriminability guard。
2. Controlled evidence: LeWM Gaussian-noise cliff/recovery + PLDM replication as boundary。
3. Release package: paired ACPC basin + negative target-view/heteroscedastic ablations + reproducibility artifacts。

避免使用 `we demonstrate`、`we establish`、`universal`、`method-invariant` 等强动词。优先用 `we find`、`we report`、`we use as evidence for`、`within this protocol`。

---

### 1.2 固定 ACPC 的 paper-facing 实例

**问题。** 目前 ACPC 定义里的 `Pi`、`Psi`、PCC、CRA、MAF、ADM、SPRR 太自由，容易被认为不可证伪。

**改法。** 在 `sec:acpc` 中新增一个短段落：

> In the main paper we instantiate ACPC with identity readout on the model inference/cost latent rollout and L2 distance over horizon H=8. PCC/CRA/MAF/ADM/SPRR are exploratory downstream readouts in the appendix and are not used for model selection or for the main robustness claim.

**主文指标最小化：**

- 主文保留：`ACPC-H / R_F`、`R_E`、closed-loop success、一个 discriminability proxy。
- PCC、CRA、MAF、ADM、SPRR 的详细定义移动到 appendix；主文一句话概括即可。
- `five-layer diagnostic` 保留为简化表或一段，不要在主文铺成工具箱论文。

**验收标准。** 主文读者应能在 1 页内知道：ACPC 是什么、本文实际怎么算、它不能证明什么。

---

### 1.3 修正 ACPC basin 的 claim 边界

**问题。** 当前表格比较 base vs direct-eval-selected point-best，属于 post-hoc endpoint comparison。不能写成 ACPC 能预测 robustness。

**必须改写的表述：**

- 把 `robust endpoints selected by direct closed-loop evaluation` 后的解释改为：
  - `This comparison is diagnostic, not predictive: the endpoint is selected by closed-loop evaluation, and the basin radius is used to localise what changed.`
- 不使用 `basin boundary`、`proof`、`predictor`、`selection rule`。
- `R_F/R_E` 明确为 descriptive ratio，不进入主要结论。

**强烈建议新增一个全网格小图或 appendix 表。**

- `4 tasks × 9 LeWM configs` 的 `R_F` vs `obs 0.08 success/drop` scatter。
- 如果没有图，就加 appendix 表，并在主文明确 “base-vs-best is a compact summary; full grid is released”。
- 若 full-grid correlation 很弱，不要隐藏；这反而支持 diagnostic-not-oracle 的边界。

---

### 1.4 给 discriminability guard 一个主文证据锚点

**问题。** ACPC 的一半是 same-state perturbation contraction，另一半是 action-relevant discriminability preservation。当前主文对后者主要靠 rank、transition resolution、ID probe 的 proxy，证据不够显眼。

**最低改法（不加新实验）：**

- 在 ACPC basin 或 sweep 后新增一个 compact paragraph：
  - PushT robust/noise representative rank and ID probe remain nearly flat。
  - Heteroscedastic ablation shows what failure looks like: PushT rank/transition/ID probe collapse and success collapses。
  - 因此本文只 claim proxy-level guard，不 claim oracle discriminability proof。

**更强改法（有数据/能跑）：**

- 对 PushT 加 contact/keyframe 或 action-distance-conditioned margin。
- 对 TwoRoom 加 room/doorway/topology margin。
- 对 Cube/Reacher 至少给 inverse-dynamics or transition-magnitude stratified margins。
- 加一张 `same-state contraction vs action-distinct separation` 的小表：base、robust、hetero failure。

**验收标准。** Reviewer 不能再轻易说“你只证明了 collapse，不证明 selective”。

---

### 1.5 统计语言降温并统一

**必须保留的事实：**

- `3 evaluation seeds × 100 trajectories` 不是 independent training seeds。
- 每个 `std_max` cell 是一个 trained epoch-10 checkpoint。
- `std` 是 across evaluation seeds 的 population std。
- Partial Spearman 每任务 `n=9`，bootstrap over checkpoint rows，CI 很宽。

**必须避免：**

- `significant`，除非有正式统计检验。
- `reliably predicts` / `robust predictor`。
- `optimal sigma`，除非写成 `point-best under this evaluation protocol` 或 `plateau region`。
- 把 `CI includes/rounds to zero` 写成强 null proof。

**建议固定措辞：**

- `The sweep identifies point-best checkpoints under a fixed evaluation protocol; it does not estimate training-run variability.`
- `The partial-correlation analysis is a mechanism-localisation check with small n, not a standalone selection rule.`
- `Evaluation-seed variability is reported for transparency; independent training-seed variability remains a limitation.`

---

### 1.6 主文压缩和段落重排

**目标。** 主文应像投稿论文，不像 artifact report。建议主文压到 8--10 pages（不含 appendix，具体按目标会议模板调整）。

**建议结构：**

1. Introduction：1.25--1.5 页。
2. Related Work：0.75--1 页；删除或 appendix 化 related-boundary 长表。
3. ACPC Diagnostic：1 页；正式定义 + paper-facing instantiation + discriminability guard。
4. Experimental Setup：0.5 页；任务、checkpoint、eval protocol。
5. Results：3--3.5 页：
   - corruption cliff；
   - noise sweep；
   - ACPC basin；
   - diagnostic boundary / partial correlation compact result。
6. Discussion and Limitations：1--1.25 页；合并重复 caveat。
7. Conclusion：短段，不重复 abstract。

**建议移动到 appendix 的内容：**

- Related-work boundary table。
- `five-layer diagnostic framework` 的完整 layer-by-layer 定义。
- PushT detailed partial-correlation table。
- Phase-0 PCC/CRA/MAF/ADM/SPRR 长定义和表。
- Full LeWM-base diagnostics landscape table。
- Blur stress 长表。
- Target-view ablation长解释可保留 appendix，主文只留一句和关键数字。

**删除/合并重点：**

- 重复出现的 `not a benchmark / not oracle / not method-invariant / scope boundary`。每个 caveat 只出现一次。
- Discussion 中与 limitations 重复的任务解释。
- 过细实现说明，例如 SIGReg 的 Cramér--Wold 解释可移 appendix。

---

### 1.7 去“AI 感”写作 pass

**全局规则：**

- 每段首句直接说结论，不用“where/while/although”堆条件。
- 少用抽象名词串：`selective-consistency tension`, `control-facing reading`, `mechanism-localisation evidence`。保留核心术语 ACPC，其余尽量用普通技术句。
- 避免自我辩护式句子连发。限制型句子集中放在每节末尾或 Limitations。
- 数值句不要堆太多括号；关键数值进表，正文只解释趋势。
- 不要频繁用 em dash。自然写法优先。

**建议 grep：**

```bash
rg -n "crucially|deliberately|scope boundary|robustness oracle|method-invariant|control-facing|paper-facing|mainline|coarse pressure|not a .* proof|not a .* theorem|does not claim|we do not claim|This is why" paper1/main.tex
```

不是所有命中都要删除，但主文命中数量应明显下降。

**替换风格示例：**

- 原：`The reading is deliberately narrow.`
- 改：`This comparison has one purpose: to localise the change at checkpoints already selected by closed-loop evaluation.`

- 原：`The toolkit is therefore an unperturbed-evaluation auxiliary for mechanism localisation and checkpoint triage within a fixed protocol, not a robustness oracle.`
- 改：`The diagnostics are useful for triage within this protocol, but final model choice still requires closed-loop corruption evaluation.`

- 原：`This pattern is consistent with a coarse global pressure, not with a selective objective...`
- 改：`A single Gaussian-noise strength recovers performance unevenly across tasks, suggesting that the augmentation does not encode the task-specific nuisance/action distinction.`

---

### 1.8 Release gate 修复

**必须修：**

1. `paper1/build.sh` 不要吞 BibTeX 错误。把 `bibtex main || true` 改成可审计逻辑：
   - 若 `references.bib` 存在且 `.aux` 需要 bibtex，则 bibtex 失败应 fail；
   - 或 build 后强制 grep undefined citations/references 并 fail。
2. `paper1/docs/README.md` 中 `references.bib # 41 entries` 与 `reference_audit.md` 的 `42 citation keys` 口径要统一。改成 `references.bib # bibliography entries; audited in reference_audit.md`，不要写死数量。
3. `\author{}` 为空要处理：
   - 匿名投稿：切换到目标会议模板/匿名宏；
   - arXiv：填真实作者或明确 placeholder 不进 release。
4. Acknowledgements 里的代码 URL 与当前仓库/最终公开仓库必须一致。不要在 paper 中写无法访问或非最终路径。
5. `reference_audit.md` 的日期和 key 数与 `references.bib` 实际状态一致。
6. `DATA_MANIFEST.md` 若 artifact 有任何变化，更新 SHA-256 和 provenance note。

**最终必须运行：**

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "Overfull|undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence" paper1/main.log || true
```

验收标准：无 fatal、无 undefined citation/reference。Overfull 若不能全消，必须确认不是正文大面积溢出；长 code key 应使用 `\codebrk{}`。

---

## 2. P1：有算力/数据时强烈建议补的实验

这些不是无算力情况下的硬门槛，但会显著提高主会投稿胜率。

### 2.1 独立训练种子

**最小设计：**

- 每个任务至少 3 个 `std_max`：base、primary robust point-best、一个中间点。
- 每个 cell 跑 3 independent training seeds。
- 保持 evaluation seeds 42/43/44 × 100 trajectories。

**主文新增：**

- 一个小表：training-seed mean/std for unperturbed and obs 0.08。
- 改写 sweep 结论：区分 training variance 和 eval variance。

### 2.2 Held-out ACPC validation

**最小设计：**

- 用一组 dataset windows 算 ACPC basin，另一组 windows 或另一 corruption severity 验证 relation。
- 不用 direct-eval point-best 再回头解释同一个 endpoint。

**可报告指标：**

- `R_F` vs obs 0.08 success 的 rank relation。
- `R_F` 是否区分 base / robust / known failure（hetero）。
- 若不能预测，写成 negative diagnostic boundary。

### 2.3 简单 baseline

至少补一种：

- Encoder latent consistency regularizer。
- DrQ-style random shift/crop/color jitter。
- Gaussian-noise full-sequence vs history-only vs target-only cleanly matched。
- Latent perturbation consistency loss。

目的不是 SOTA，而是防止 reviewer 说“只是普通 augmentation 故事”。

### 2.4 Out-of-family corruption

当前 blur 是 eval-only sanity check。若能补：

- Gaussian-trained checkpoint 在 blur 下表现。
- Blur-trained checkpoint 或 mixed corruption training。
- 至少说明 Gaussian recovery 是否转移到 blur。

---

## 3. P2：投稿材料和 rebuttal 预案

### 3.1 Reviewer-facing limitations box

在 limitations 开头放一个短 box 或 compact paragraph：

- No independent training seeds in canonical grid。
- Gaussian pixel noise is the controlled training axis。
- ACPC basin is diagnostic, not a model-selection oracle。
- Discriminability guard is proxy-level in this paper。
- Main mechanism claim is LeWM-centered; PLDM is replication/boundary。

这样比在全文反复插入 caveat 更自然。

### 3.2 Reproducibility appendix

增加一个短 appendix subsection：

- canonical artifacts and hashes；
- checker command；
- build command；
- known 2026-06-10 representative-diagnostics contamination and fix；
- statement that JSON artifacts are generated, not hand edited。

这能把之前的 contamination audit 转化成可信度加分项。

### 3.3 Reviewer FAQ 草稿

准备但不一定放正文：

- Q: Is ACPC a new objective? A: No, diagnostic framing; objective is future work。
- Q: Why no training seeds? A: Controlled intervention sweep; evaluation seeds reported; training variability is limitation / P1 extension。
- Q: Does smaller `R_F` predict robustness? A: Not claimed; selected endpoints show localised mechanism; full-grid diagnostics are released。
- Q: Why not encoder invariance? A: same-state views need predictive agreement, but action-distinct states must remain separable。
- Q: Why full-sequence noise target? A: target-view ablation shows original-future denoising alone fails closed-loop recovery。

---

## 4. Suggested Codex execution order

1. Create a working branch from `ag/dev`.
2. Run current gates and save logs:

```bash
python -m tools.check_paper1_consistency | tee /tmp/paper1_check_before.log
cd paper1 && bash build.sh --clean | tee /tmp/paper1_build_before.log
```

3. Rewrite `main.tex` in this order:
   - title/abstract/contributions；
   - ACPC section simplification；
   - Experiments section claim boundary；
   - Discussion/Limitations merge；
   - appendix migration for long definitions/tables。
4. Fix release files:
   - `paper1/build.sh`；
   - `paper1/docs/README.md`；
   - `paper1/docs/reference_audit.md` if key count/date changed；
   - `DATA_MANIFEST.md` only if artifacts changed。
5. Run grep for AI-ish/repetitive phrasing and manually reduce。
6. Rebuild and check logs。
7. Produce final summary with:
   - changed files；
   - removed/relocated sections；
   - final paper positioning；
   - commands run and pass/fail；
   - unresolved compute-dependent P1 items。

---

## 5. Final submission acceptance checklist

Before calling the paper submission-ready, all boxes below should be checked.

### Claim and writing

- [ ] Title and abstract no longer imply universal JEPA robustness theory.
- [ ] Abstract is under ~210 words and contains only essential numbers.
- [ ] Contributions fit the diagnostic-study framing.
- [ ] ACPC paper-facing readout is fixed and easy to find.
- [ ] PCC/CRA/MAF/ADM/SPRR are appendix/exploratory, not main claim.
- [ ] Point-best selection is described as direct-eval-selected, not diagnostic-selected.
- [ ] No claim that ACPC basin predicts robustness unless held-out validation is added.
- [ ] Discriminability guard has at least one main-text proxy evidence paragraph.
- [ ] Independent training seed limitation is stated once clearly, not defensively repeated.
- [ ] Main text avoids repetitive “not X / not Y / scope boundary” phrasing.

### Statistics and experiments

- [ ] Every success-rate std is described as evaluation-seed population std.
- [ ] Partial Spearman `n=9` limitations are explicit.
- [ ] Point-best/plateau wording is used instead of universal optimum wording.
- [ ] PLDM is framed as replication/boundary, not leaderboard.
- [ ] Blur is framed as eval-only sanity check unless blur training is added.

### Release

- [ ] `python -m tools.check_paper1_consistency` passes.
- [ ] `cd paper1 && bash build.sh --clean` passes.
- [ ] No undefined citations/references in `main.log`.
- [ ] `build.sh` does not silently ignore BibTeX failure.
- [ ] README/reference audit bibliography counts are consistent or not hard-coded.
- [ ] Final code/data URL is correct and accessible.
- [ ] Author/anonymity mode matches target venue.
- [ ] Artifact hashes are updated if any generated artifact changed.

---

## 6. Non-goals / things not to do

- Do not invent a new algorithmic contribution in prose.
- Do not present target-view or heteroscedastic ablations as proof of ACPC objective superiority.
- Do not hide weak/zero partial correlations; use them to justify diagnostic boundary.
- Do not move negative results out of sight; they are part of the paper’s strength.
- Do not over-polish into generic “AI paper” prose. Prefer short technical sentences tied to exact tables.
