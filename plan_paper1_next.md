# Paper 1 — 下一步执行计划

> 配套 `paper_invariance_resolution_tradeoff.md` / `paper_invariance_resolution_tradeoff_zh.md` / `paper1/main.tex`。
> 本文档汇总：故事强度评估、查重/新颖性结论、reviewer 视角写作整改清单、PLDM 数据补齐后的状态与下一步实验。
> 最后更新：2026-05-22 — PLDM 36-ckpt sweep（4 tasks × 9 configs）已整合进 Appendix F；clean 模型 blur eval 正在运行中。

---

## 1. 故事强度评估（reviewer 视角）

### 1.1 优势

- **统一概念清晰**：`invariance–resolution trade-off` 配套可操作定义（5 层诊断协议）和四任务 task-specific 证据（PushT 接触 vs TwoRoom 冗余）
- **诚实的负向结论**（partial-corr → 0.06）提高 credibility，在 benchmarking 论文里少见
- **可复现工具链**：LeWM 36 ckpt + PLDM 36 ckpt，均为 3 eval seeds × 100 traj/condition，canonical JSON release，paper figures by single script
- **每个 contribution 都对应可数 evidence**（C1=LeWM/PLDM 4×8 sweep；C2=17 metrics；C3=表 4；C4=表 5/6/7 + PLDM PushT cross-check）

### 1.2 审稿人最可能攻击的薄弱面（按可能性排序）

| # | 薄弱点 | 当前 mitigation | 行动建议 |
|---|---|---|---|
| W1 | trade-off 只在 LeWM 上验证 | Appendix F 已补完整 PLDM 36-ckpt sweep；Limitation 1 改为 broader JEPA variants remain open | 当前已缓解；不要再把结论写成 PLDM 全指标机制归因 |
| W2 | 仅 Gaussian pixel noise | §5.5 Limitation 2 已写明 | clean 模型 blur eval 正在跑，完成后作为 appendix sanity check |
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

## 4. PLDM 数据集成状态

> 数据落在 `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-{pusht,tworooms,reacher,cube}/ckpt/<dataset>_pldm_{baseline,noise_*}/eval_results/...`。
> 配置和 LeWM 完全一样：3 eval seeds × 100 traj。

### 4.0 已完成（2026-05-22）

✅ Step A — 数据聚合
- `assets/paper1_data/canonical_evals_pldm_20260522.json` — 36 ckpts（4 tasks × 9 configs）
- `assets/paper1_data/canonical_diagnostics_pldm_20260522.json` — 同 grid，两项 full-coverage predictor metrics
- 聚合脚本：`tools/build_canonical_evals_pldm.py` + `tools/build_canonical_diagnostics_pldm.py`

✅ Step B — 跨方法相关性分析
- `assets/paper1_data/cross_method_corr_pldm_20260522.json` — within-LeWM / within-PLDM / joint (LeWM+PLDM) partial Spearman
- 脚本：`tools/pldm_correlation_analysis.py`

✅ Step C — 论文升级
- §1.3 Contributions C1/C4 加入 PLDM replication 证据
- §2.1 Related-work PLDM 段：从"sanity check"升级为"second method family"
- §4.2 OOD cliff 段：加 PLDM PushT / TwoRoom cliff，同时明确 Reacher/Cube clean-trained PLDM gap 较小
- §5.5 Limitation 1：从"Single backbone family"软化为"Two backbone families validated; broader JEPA variants remain open"
- Appendix F：完整 4-task PLDM sweep，含 clean baseline table、clean/px+goal 0.08 sweep table、PLDM all-task partial-corr table
- Abstract/C4：只说 PushT null 跨方法复现；TwoRoom/Reacher/Cube residual correlations 作为 scope boundary 公开报告
- 摘要 + Contributions 措辞保持保守：什么 replicate 了 / 什么是 method-specific 都明示

### 4.1 PLDM 结果读法

- C1 strengthened：PLDM 在 TwoRoom/PushT/Cube 上复现 task-level signature。TwoRoom 从 px+goal 0.08 baseline 51.67 recovery 到 98.33；PushT 从 10.00 recovery 到 72.00；Cube 响应弱。
- Reacher 是 low-cliff external baseline：clean-trained PLDM 已有 79.33 px+goal 0.08，不应写成 universal noise-training gain。
- C4 strengthened only on PushT：PLDM PushT OOD-drop partial 为 -0.14，joint LeWM+PLDM PushT 为 +0.11。TwoRoom/Reacher/Cube 的 residual correlations 保留为 scope boundary。

### 4.2 仍然是限制项

- broader JEPA variants（I-JEPA / V-JEPA lineage; variational JEPA）仍未跑。
- Gaussian pixel noise 是主实验 corruption；clean 模型 blur eval 已在跑，完成后可以作为 Appendix G / sanity check。
- PLDM 目前只承担 cross-method eval/sweep replication；完整 five-layer mechanism 主体仍以 LeWM 为 source of truth。

---

## 5. 投稿成色路线图

| 状态 | 目标场地 |
|---|---|
| **当前**（LeWM + PLDM 36-ckpt 完整 sweep） | ICLR / NeurIPS 正会叙事最低门槛达成；仍需控制措辞，避免把 PLDM 解释成完整机制归因 |
| **再补一个 blur / contrast 单档** | 对 Gaussian-only reviewer 攻击有直接缓解 |
| **后续 DINO-WM / V-JEPA lineage** | 第二篇或 v1 扩展，不应阻塞 arXiv v0 |

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
- 数据 source-of-truth：LeWM 使用 `assets/paper1_data/canonical_evals_20260517.json` + `assets/paper1_data/canonical_diagnostics_20260517.json`；PLDM 使用 `assets/paper1_data/canonical_evals_pldm_20260522.json` + `assets/paper1_data/canonical_diagnostics_pldm_20260522.json` + `assets/paper1_data/cross_method_corr_pldm_20260522.json`
- 一致性检查：`python tools/check_paper1_consistency.py`
