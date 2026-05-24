# Paper 1 — 故事线 + 计划

> Source of truth: `paper1/main.tex`. 数据 source-of-truth 见 §7.
> Last updated: 2026-05-23（reviewer 视角整改 commit `fd7242d` 后）。

---

## 1. 一句话定位

LeWM 这类 JEPA + CEM 世界模型在 clean 视觉输入上能规划，但未经噪声训练时会在 control-time pixel/goal noise 下严重崩溃。全局 input-side noise augmentation 可大幅恢复 OOD 成功率，但没有全局最优剂量——它同时带来有益 invariance 和有害 task-resolution compression。本文用 4 task × 2 method × 36+36 ckpt × 3 eval seed × 100 traj 的统一协议，加上五层表征诊断，把这个现象命名为 **invariance–resolution trade-off**。

**这不是新算法 paper，是诊断 + empirical paper。** Contribution 是 (i) systematic study of JEPA+CEM visual OOD failure, (ii) 可复现 diagnostic toolkit, (iii) cross-checkpoint 诊断可预测什么、不可预测什么的明确范围。

## 2. 故事线（5 step + 4 contribution）

### Storyline

- **Step 1（§4.2）— cliff 真实存在.** LeWM-base PushT clean 86.33 → px+goal 0.08 4.67 (−81.67pt)；TwoRoom 94 → 50；Reacher 58.67 → 15；Cube 66.67 → 46.33.
- **Step 2（§4.3）— noise training 能恢复，但 optimum task-dependent.** TwoRoom 在 0.08 plateau；PushT clean 0.03 vs OOD 0.06 解耦；Reacher 0.02–0.06 plateau；Cube 0.07. **不存在 universal σ\***.
- **Step 3（§4.4）— 5 层诊断解释 task-specific response.** TwoRoom rank compression 有益（47.6→33.6）；PushT base 本来就 high-rank high-controllability（rank 76.42, ID-probe R²=0.77），噪声训练 rank 砍 44% 也连带砍 transition resolution & controllability，因此不能再无脑压。
- **Step 4（§4.5）— partial Spearman 划清"能预测什么".** fragility ratio 是 checkpoint-quality 信号（partial ρ = −0.59 on clean, −0.41 on px+goal 0.08）；但不是 OOD oracle（partial ρ = +0.06 on clean-to-OOD drop）. PLDM PushT 复现同样的 null（−0.14 partial），joint n=18 也是 +0.11 partial.
- **Step 5（§4.6 + §F + §G）— mechanism 主张窄但稳.** pixels → encoder → predictor → CEM 失败. PLDM 复现 task-level recovery signature，但 internal route 不同（rollout drift dominated, rank/resolution 几乎不动）. Blur 也能破坏控制（TwoRoom 最敏感, −64/−68 pt vs Gaussian noise 的 −44/−45）, task ordering 与 Gaussian noise 不同。

### Contributions

- **C1** — Systematic OOD fragility 量化：n = 72 ckpts under unified 3-seed × 100 protocol.
- **C2** — invariance–resolution trade-off 概念 + 5 层诊断协议（17 metrics, partial-correlation validation）.
- **C3** — Mechanism 解释：LeWM compression chain；PLDM via predictor-drift reduction 不同路由（mechanism 是 method-specific，不是 universal）.
- **C4** — Cross-checkpoint diagnostic 范围：model-selection tool, **不是** OOD oracle. PushT partial-ρ null 在 LeWM / PLDM / joint 三处复现。

## 3. 当前状态（2026-05-23）

### 数据完整性 ✅

| Asset | LeWM | PLDM |
|---|---|---|
| Eval sweep canonical JSON | 36 ckpt (4×9) | 36 ckpt (4×9) |
| Predictor diagnostic summary | 36 ckpt | 36 ckpt |
| Full 5-layer diagnostics | ✓ | ✓ |
| Cross-method partial-corr | joint n=18 PushT released | same |
| Blur stress test | clean-trained 4 task × 4 kernel | clean-trained 4 task × 4 kernel |

### Paper 状态 ✅

- LaTeX: 32 页, 0 Overfull, 0 errors.
- `tools/check_paper1_consistency.py`: 全绿.
- TeX Live 2026 at `/home/ag/texlive/2026/bin/x86_64-linux` (已写入 `~/.zshrc`).

### 写作完成度（commit `fd7242d` 后）

- Abstract / §1.3 / §6 mechanism chain 加 "on LeWM" 限定 + PLDM 不同路由声明 ✅
- §5.5 Lim 2 blur task-ordering 数字化反 Gaussian-only 质疑 ✅
- §F Table 16 TwoRoom +0.87 / Reacher −0.86 saturation 解释 ✅
- §5.5 Lim 4 → "Additional ablation" (hetero-loss 是 ablation 不是 limitation) ✅
- §3.3 ε 定义 / §4.6.1 cost-swap n=300 注 / §F 77→76.67 精确化 ✅
- §4.4 Notes 过时项删除；§5.5 Lim 3 扩成 3 句；§1.1 "blog posts" 描述修正 ✅
- 命名空间统一：`gaussian_noise` / `gaussian_blur`（代码 + 配置 + paper 三处对齐）✅

## 4. Reviewer 视角薄弱点与 mitigation

| # | 薄弱点 | 已 mitigation | 提交前是否再加固 |
|---|---|---|---|
| W1 | trade-off 是否跨方法 | App F PLDM 完整 sweep + §5.5 Lim 1 LeWM-centred 措辞 | 已缓解 |
| W2 | 仅 Gaussian noise 训练轴 | App G clean-trained blur eval + §5.5 Lim 2 数字化 | 已缓解 |
| W3 | Reacher/TwoRoom partial-corr 失效 | §5.3 Scope 2 + §5.5 Lim 3 三句扩展 | 已缓解 |
| W4 | n=9 partial-corr 统计稳健性 | n=18 joint 已加；尚无 CI | **加 bootstrap 95% CI**（见 §5.A.2） |
| W5 | 无形式化理论（IB/rate-distortion） | §5.5 future direction 3 已声明 | 不致命 |
| W6 | 无 method 贡献 | Framing 明示 "empirical + toolkit + delineation" + hetero-loss negative ablation | 不致命 |
| W7 | 部分 2026 arXiv ID 未人工核对 | — | **核对 9 条**（见 §5.A.1） |

## 5. 待办与优先级

### 5.A 提交前必做（约 1.5 天）

**5.A.1 arXiv 引用核对**（~1 hr 手工）

下面 9 条逐条 open URL，对照 (a) 作者+标题 (b) 我引用的 claim 是否在原文 abstract 能查到。任何对不上 → 找替代引用或删该 claim。

```
[3]  V-JEPA 2          https://arxiv.org/abs/2506.09985
[11] seq-JEPA          https://arxiv.org/abs/2505.03176
[15] VJEPA (Huang)     https://arxiv.org/abs/2601.14354    ← 最高风险（"R² > 0.84 under Noisy-TV"）
[22] LeWM              https://arxiv.org/abs/2603.19312
[23] ViGMO             OpenReview ICLR 2026 submission
[24] N-JEPA            https://arxiv.org/abs/2507.15216
[25] US-JEPA           https://arxiv.org/abs/2602.19322
[31] Next-Latent       https://arxiv.org/abs/2511.05963
[32] Bisim-JEPA        https://arxiv.org/abs/2602.18639
```

如果 [15] 不存在，§2.2 "VJEPA tests Noisy-TV..." 和 §5.1 "stands in contrast to VJEPA..." 都要重写。

**5.A.2 Bootstrap 95% CI on partial correlations**（~半天）

```
tools/build_partial_corr_bootstrap.py
  input:  canonical_evals / diagnostics / cross_method_corr
  output: assets/paper1_data/partial_corr_bootstrap_<date>.json
  method: 1000-iter resample with replacement (stratified by std_max bin)
  paper:  Table 6 / 7 / 16 加 95% CI 列；正文一句 "with CI [a, b]"
```

预期：PushT joint n = 18 partial = +0.11 的 CI 应明显包含 0，把 null claim 钉死。Reacher partial = +0.79 的 CI 若不含 0，反而支持现有 "only non-trivial residual" 表述。

### 5.B v1 / 第二阶段（不阻塞 arXiv v0）

- **Blur training sweep**（~1 周）：blur augmentation 是否 recover、Gaussian noise training 是否 transfer 到 blur。
- **Plan-side robustness CEM**（~1–2 周）：robust CEM solver 已剥出 `config/eval/solver/robust_cem.yaml`。
- **Broader JEPA variants**（~2–4 周）：I-JEPA / V-JEPA lineage / variational JEPA.
- **TD-MPC2 / DreamerV3 cross-arch**（~1 周）：reconstruction-based world model 对比.
- **DMC-Suite task 扩展**（~1 周）：1 个 DMC task 削弱"4 task cherry-picked"质疑.
- **IB / rate–distortion 理论 framing**（开放期）.

## 6. 投稿路线图

| 状态 | 成色 |
|---|---|
| **当前 v0**（LeWM+PLDM 36+36 ckpt + full diagnostics + clean blur eval + reviewer 整改） | arXiv preprint readiness 已达成；ICLR / NeurIPS 正会叙事最低门槛满足 |
| **+ arXiv 核对 + bootstrap CI** | submission-ready |
| **+ blur training（v1）** | 加深 robustness 结论可控范围 |
| **+ 跨架构 baseline** | 第二篇或 v1 扩展 |

## 7. 数据 source-of-truth 与工具

| 用途 | 文件 |
|---|---|
| LeWM eval sweep | `assets/paper1_data/canonical_evals_20260517.json` |
| LeWM predictor diagnostics | `assets/paper1_data/canonical_diagnostics_20260517.json` |
| PLDM eval sweep | `assets/paper1_data/canonical_evals_pldm_20260522.json` |
| PLDM predictor diagnostics | `assets/paper1_data/canonical_diagnostics_pldm_20260522.json` |
| PLDM full 5-layer diagnostics | `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` |
| LeWM+PLDM cross-method partial-corr | `assets/paper1_data/cross_method_corr_pldm_20260522.json` |
| Clean-trained blur baselines | `assets/paper1_data/canonical_blur_baselines_20260523.json` |
| External baselines | `assets/paper1_data/canonical_external_baselines_20260520.json` |

```bash
# 主图重生成
python -m tools.paper1_figs --out-dir assets/paper1_figs

# 一致性检查
python tools/check_paper1_consistency.py

# PDF 构建
export PATH=/home/ag/texlive/2026/bin/x86_64-linux:$PATH
cd paper1 && bash build.sh --clean
```

## 8. 近邻文献 novelty 表

| 论文 | 关系 | 差异化定位 | 位置 |
|---|---|---|---|
| seq-JEPA [Ghaemi 2025] | JEPA + invariance-**equivariance** 张力 | 不同 trade-off 维度；他们做架构 disentanglement | §2.4 |
| Bisim-JEPA [Toso 2026] | JEPA + planning + 视觉鲁棒性 | 不同 corruption family（slow-feature vs pixel noise）+ 不同 remedy | §2.2 |
| RankMe [Garrido 2023] | label-free effective-rank ↔ SSL 下游 | 我们 ask the analogous question for control | §2.5 |
| DrQ / DrQ-v2 | input-side aug for pixel RL | model-free + reward-driven，不是 JEPA+CEM | §2.3 |
| SODA / DMC-GB | DMC 视觉泛化 benchmark | 不同 corruption family（distractor 背景） | §2.3 |
| Wang & Isola 2020 | SSL alignment-uniformity | 概念先例，我们扩展到 control | §2.4 |
| ViGMO | model-based RL 视觉鲁棒性 | 不同架构 + 我们多了 mechanistic decomposition | §2.3 |

**没有任何一个工作同时做了** JEPA 世界模型 + CEM 控制 + pixel noise sweep + 5 层诊断 + partial-correlation cross-checkpoint 验证。**Novelty 安全。**
