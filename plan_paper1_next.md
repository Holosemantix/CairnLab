# Paper 1 — 下一步执行计划

> 配套 `paper_invariance_resolution_tradeoff.md` / `paper_invariance_resolution_tradeoff_zh.md` / `paper1/main.tex`。
> 本文档汇总：故事强度评估、查重/新颖性结论、reviewer 视角写作整改清单、PLDM 数据补齐后的执行步骤。
> 最后更新：2026-05-21（commit `bae6259` 已包含本文档的主要写作整改）。

---

## 1. 故事强度评估（reviewer 视角）

### 1.1 优势

- **统一概念清晰**：`invariance–resolution trade-off` 配套可操作定义（5 层诊断协议）和四任务 task-specific 证据（PushT 接触 vs TwoRoom 冗余）
- **诚实的负向结论**（partial-corr → 0.06）提高 credibility，在 benchmarking 论文里少见
- **可复现工具链**：36 ckpt × 300 traj/condition，canonical JSON release，paper figures by single script
- **每个 contribution 都对应可数 evidence**（C1=4×8 sweep；C2=17 metrics；C3=表 4；C4=表 5/6/7）

### 1.2 审稿人最可能攻击的薄弱面（按可能性排序）

| # | 薄弱点 | 当前 mitigation | 行动建议 |
|---|---|---|---|
| W1 | trade-off 只在 LeWM 上验证；PLDM 只是一个 ckpt sanity check | §5.5 Limitation 1 已写明 | **PLDM 数据补齐后这是最大升级**（见 §4） |
| W2 | 仅 Gaussian pixel noise | §5.5 Limitation 2 已写明 | 可加一个 blur 或 contrast 单档（200 traj 就够）作为辅证 |
| W3 | Reacher / TwoRoom partial-corr 失效 | §5.3 Scope 2 已明确划界 | 当前 framing OK；可再补充"这是 scope 而非 bug" |
| W4 | 无形式化理论（IB / rate–distortion） | §5.5 Limitation 3 已写明 | 不致命；可在 §2.4 加一个 IB-style 段落预埋未来工作 |
| W5 | 四任务体量偏小 | 已经为代表性 | 取决于场地（ICLR/NeurIPS OK；CoRL/RSS 可能想加一两个 manipulation） |
| W6 | `fragility ratio` 单指标承担 C4 | partial-corr 自带 sensitivity（同样在 `predictor_rollout_T8_l2` 上归零） | 在 §5.3 加一句显式声明这不是 cherry-picked |

---

## 2. 查重 / 新颖性结论

### 2.1 总判定

**Novelty 完整，没有撞车。** 本会话已经把 6 个最关键的近邻工作加进 bib + §2 并明确画出区分线。

### 2.2 关键近邻文献定位表

| 论文 | 关系 | 我们的差异化定位 | 在 paper 中的位置 |
|---|---|---|---|
| **seq-JEPA**（Ghaemi 2025，arXiv:2505.03176） | JEPA + invariance-**equivariance** trade-off | trade-off 维度不同；他们做架构 disentanglement，我们做 task-resolution loss | §2.4 |
| **Bisim-JEPA**（Toso 2026，arXiv:2602.18639） | JEPA + planning + 视觉鲁棒性 | 不同 corruption family（slow-feature distractors vs pixel noise）+ 不同 remedy（bisimulation encoder vs 诊断 toolkit） | §2.2 |
| **RankMe**（Garrido 2023, ICML） | label-free effective-rank ↔ SSL 下游迁移 | 直接 motivate 我们"label-free 诊断能否预测控制成功"的问题，明确说我们 ask the analogous question for control | §2.5 |
| **DrQ / DrQ-v2**（Yarats 2020/2022） | input-side augmentation for pixel RL | 经典先例，但 model-free + 奖励驱动；我们是 JEPA + CEM 规划 | §2.3 |
| **SODA / DMC-GB**（Hansen & Wang 2021） | DMC 视觉泛化基准 | 不同 corruption family（distractor 背景 vs pixel noise） | §2.3 |
| **Wang & Isola 2020**（alignment-uniformity） | SSL 两个竞争属性 | 概念先例（two competing properties），我们扩展到 control | §2.4 |

### 2.3 没有任何一个工作同时做了

JEPA 世界模型 + CEM 控制 + pixel noise sweep + 5 层诊断 + partial-correlation cross-checkpoint 验证。**novelty 安全。**

---

## 3. 写作整改：已 done + 留给作者权衡

### 3.1 已经在 commit `bae6259` 整改的高优先项

| 项 | 整改 |
|---|---|
| 摘要里 code-field 字段名 `\fragmetric{}` 显式出现 | 摘要现在只用 "fragility ratio (Section 3.3)"；35 字符长字段名挪进表格和首定义 |
| C2 vague "17+ concrete metrics" | 改成 "17 concrete metrics" |
| C3 数字过细（`0.7739 / 0.7500 / 0.0573 / 0.3015 → 0.2800`） | 用 prose："rank drops by ~44%", "R² ≈ 0.77" + 表 4 引用 |
| §4.4 mechanistic reading 内嵌 `transition_resolution_ratio_l2` / `id_probe_r2` 等代码标识 | 改为 "L2 transition-resolution metric"、"controllability probe" |
| §4.5.1 子节标题 "What this diagnostic actually predicts: clean vs OOD"（colon + 口语） | 改为 "Clean control fidelity and OOD robustness as separable signals" |
| §5.3 "Boundary 1/2/3"（非学术词） | 统一改成 "Scope 1/2/3"；正文 intro 也改为 "Three scope conditions..." |
| §4.3 "Reading guidance" | 改成 "Notation" |
| Cube 弱信号像 hole | §1.2 显式 reframe："we read this as the trade-off's mildest manifestation rather than an absence" |
| Reacher 35× rollout 减少的"惊吓" | tab:diag-base-vs-best Note (iii) 改成 "not a recording error: ...this is the dissociation that motivates §4.5" |
| 残余 "honestly" 等 AI 语气 | 全部去掉或替换为 "delineate / explicit" |
| §4.5 paragraph 1 重复展开长 code 名 | 改成 "defined in Section 3.3" + 短自然名 |

### 3.2 留给作者权衡的可选改动

- **C1/C2/C3/C4 `\paragraph{}` 段标题**：保留还是改成 enumerate？两种 NeurIPS / ICLR 都有；建议保留（更醒目）
- **"Cross-Entropy Method (CEM)" 在摘要第 1 条 bullet 内括号展开**：略显字数密集，但首次出现必须展开；当前是 reviewer-safe 的做法

---

## 4. PLDM 数据补齐后的执行清单

> 数据期望落到 `/opt/.../lewm-{pusht,tworooms,reacher,cube}/ckpt/<dataset>_pldm_{baseline,noise_*}/eval_results/...`。
> 配置和 LeWM 完全一样：4 tasks × 9 std_max × 3 eval seeds × 100 traj。

### 4.1 Step A — 数据聚合（半天）

```bash
# 1. 复用 LeWM 的 canonical generation script，给 PLDM 单独跑一遍
python -m tools.repr_analysis.build_canonical_evals \
    --method pldm \
    --stds 0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 \
    --out assets/paper1_data/canonical_evals_pldm_<DATE>.json

python -m tools.repr_analysis.build_canonical_diagnostics \
    --method pldm \
    --out assets/paper1_data/canonical_diagnostics_pldm_<DATE>.json

# 2. 一致性检查
python tools/check_paper1_consistency.py --include-pldm
```

### 4.2 Step B — 跨方法相关性分析（半天）

```bash
# n=18 joint Spearman + 双重 partial（条件于 std_max 与 method dummy）
python -m tools.repr_analysis.cross_check_correlations \
    --method-set lewm,pldm \
    --partial-on std_max method \
    --out assets/paper1_data/cross_check_corr_n18_<DATE>.json
```

预期产出每任务 4 列：`ρ_n18 / partial|std / partial|method / partial|std+method`。

### 4.3 Step C — 论文更新（1–2 天）

需要改的位置：

1. **§2.1 PLDM 段**：把 "For one external-baseline sanity check we also evaluate PLDM" 改成 "We replicate the full sweep on PLDM..."
2. **§4.2 OOD cliff**：把 single-ckpt PLDM 数字从 Appendix F 拉到正文；加一行到 `tab:ood-cliff`
3. **§4.3 sweep**：加一对 PLDM 的 `tab:sweep-clean-pldm` + `tab:sweep-px08-pldm`（或合并成"LeWM/PLDM 对照"的双列宽表）
4. **§4.5 cross-checkpoint correlation**：
   - 新增 `tab:corr-n9-pldm`（mirror `tab:corr-n9`）
   - 新增 `tab:partial-corr-pldm`（mirror `tab:partial-corr-4tasks`）
   - **核心新表 `tab:joint-n18`**：LeWM+PLDM 联合分析，partial conditioning on (std_max, method)
5. **§4.6 mechanism attribution**：加一段 "... and the same pattern replicates on PLDM"
6. **新增 Figure 7**：LeWM-vs-PLDM Pareto 对比，或在 fig5_scatter 拼 PLDM 的 panel (a)(b)
7. **§5.5 Limitation 1**："Single backbone family" → "Two backbone families validated; broader JEPA variants (I-JEPA / V-JEPA lineage; variational JEPA) remain open"
8. **Appendix F**：从 PushT-only 升级成完整 4-task × 9-level PLDM sweep

### 4.4 Step D — 摘要与 Contributions 升级（半天）

- **C1**："JEPA + CEM control" → "JEPA-family (LeWM and PLDM) + CEM control"
- **C3**：补充 "We replicate the same compression pattern on PLDM (Appendix F)" 一句
- **C4**：补充 "and the partial-correlation finding generalizes across the two methods (Table N)"

### 4.5 Step E — 编译 + push（10 分钟）

```bash
export PATH=/home/ag/texlive/2026/bin/x86_64-linux:$PATH
bash paper1/build.sh --clean
# 目标：0 Overfull, 0 errors
```

---

## 5. 投稿成色路线图

| 状态 | 目标场地 |
|---|---|
| **当前**（commit `bae6259`） | ICLR workshop / 工作研讨会够格；正会需要 W1+W2 |
| **PLDM 数据补齐 + §4 Step C 完成** | 投 ICLR / NeurIPS 正会的最低门槛达成 |
| **再补一个 blur / contrast 单档** | 投 CoRL / 顶会强投稿 |

---

## 6. 一次性参考清单（新增的关键引用，已在 `references.bib` 中）

| BibTeX key | 出处 | 用途 |
|---|---|---|
| `wang2020alignuniform` | Wang & Isola, ICML 2020 | §2.4 alignment-uniformity 概念先例 |
| `garrido2023rankme` | Garrido et al., ICML 2023 | §2.5 RankMe 作为 cross-checkpoint 诊断的 motivating precedent |
| `kostrikov2020drq` | Kostrikov et al., ICLR 2021 | §2.3 DrQ |
| `yarats2022drqv2` | Yarats et al., ICLR 2022 | §2.3 DrQ-v2 |
| `hansen2021soda` | Hansen & Wang, ICRA 2021 | §2.3 SODA / DMC-GB |
| `ghaemi2025seqjepa` | Ghaemi et al., arXiv:2505.03176 | §2.4 seq-JEPA |
| `toso2026bisimjepa` | Toso et al., arXiv:2602.18639 | §2.2 Bisim-JEPA |

---

## 7. 已知遗留 / 监控项

- LaTeX 编译目标：0 Overfull（当前满足），0 errors（当前满足），少量 Underfull 是 `\emergencystretch` 的预期 trade-off
- TeX Live 路径：`/home/ag/texlive/2026/bin/x86_64-linux`（已写入 `~/.zshrc` 和 `~/.bashrc`）
- 数据 source-of-truth：`assets/paper1_data/canonical_evals_20260517.json` + `assets/paper1_data/canonical_diagnostics_20260517.json`（PLDM 来了后会出现 `canonical_*_pldm_<DATE>.json`）
- 一致性检查：`python tools/check_paper1_consistency.py`
