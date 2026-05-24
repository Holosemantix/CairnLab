# Paper 1 — 故事线与提交计划

> Source of truth: `paper1/main.tex`. 数值、表格、图和 artifact 以论文正文与 `assets/paper1_data/` 为准。
> Last updated: 2026-05-24。

---

## 1. 一句话故事

JEPA world model 的 latent prediction 常被理解为会自然学到更抽象、更 invariant 的表征；但在 control 里，invariance 不是免费午餐。我们发现 LeWM/PLDM 这类 latent predictive world model 在 clean input 上能规划，却会在 control-time visual corruption 下显著崩溃；input-side noise training 能恢复鲁棒性，但会按任务不同程度压缩 task-relevant resolution。Paper 1 的核心是把这个现象系统化地命名、量化、诊断，并划清“诊断指标能预测什么、不能预测什么”的边界。

这不是新算法 paper，而是 empirical + diagnostic paper。

## 2. 读者应该记住的逻辑链

1. **动机：latent prediction 不等于 visual robustness。**
   JEPA 避免 pixel reconstruction，不代表 controller 面对 pixel/goal corruption 时自动鲁棒。这个假设需要在 closed-loop control 里被实证检验。

2. **现象：clean performance 和 visual OOD performance 可以严重脱钩。**
   Clean 上能规划的模型，在 pixel/goal noise 或 blur 下可能突然失效；这说明仅报告 clean success 会高估 world model controller 的可部署性。

3. **干预：noise training 有用，但不存在 universal noise dose。**
   视觉冗余强的任务更容易从 heavy noise training 受益；接触/精细控制任务则更容易受到 representation compression 的副作用影响。关键不是“加噪声一定好”，而是“noise dose 在 invariance 和 resolution 之间重新分配容量”。

4. **机制：LeWM 的主要路径是 compression chain。**
   在 LeWM 上，noise training 的收益/代价可以通过五层诊断读出来：encoder shift、geometry compression、predictor response、latent-noise response、task resolution。主线机制是表征压缩影响 transition-key resolution，再影响 controllability。

5. **边界：PLDM 复现 task-level signature，但 mechanism route 不同。**
   PLDM 支持“visual OOD fragility + noise recovery 不是 LeWM 单点偶然”，但它的内部变化更偏 predictor-drift route，而不是完全复刻 LeWM 的 compression chain。因此 paper 的强结论是跨方法的现象，机制结论则保持 architecture-aware。

6. **诊断指标的正确用法：model-selection signal，不是 OOD oracle。**
   Cross-checkpoint diagnostic 能帮助判断 checkpoint quality，但不能替代真实 OOD evaluation。特别是 partial correlation after conditioning on `std_max` 后，fragility ratio 对 clean/OOD performance 有 checkpoint-quality 信息，却不能稳定解释 clean-to-OOD gap。95% bootstrap CI 包含 0 时，应解读为当前小样本 sweep 下没有稳定非零 residual association 的证据，而不是“相关性严格等于 0”。

## 3. 贡献写法

- **C1：系统化暴露问题。** 统一协议下比较多任务、多 noise level、多模型家族，展示 latent predictive control 的 visual OOD cliff。
- **C2：提出 invariance--resolution trade-off 作为解释框架。** Noise augmentation 同时带来有益 invariance 和潜在 resolution loss，任务结构决定二者平衡。
- **C3：给出五层诊断 toolkit。** 不是只看 success rate，而是把 failure 拆到 encoder、geometry、predictor、latent perturbation、task resolution。
- **C4：明确诊断边界。** Diagnostic 可以作为 checkpoint-quality probe，但不能被夸大成 OOD robustness predictor。

## 4. 文章立场

应该坚持的强说法：

- Latent prediction alone does not guarantee visual robustness for control.
- Visual OOD failure is a real closed-loop control issue, not just representation-space curiosity.
- Noise training creates a task-dependent invariance--resolution trade-off.
- LeWM 的 mechanism evidence 支持 compression-chain reading。
- PLDM 支持现象跨方法，但机制路径不完全相同。

需要避免的过强说法：

- 不要说所有 JEPA 都会同样崩溃。
- 不要说某个 diagnostic universally predicts robustness。
- 不要说 cost surface 已被排除为所有任务的主因。
- 不要把 blur eval-only 写成 blur training conclusion。
- 不要说 Gaussian-noise sweep 的 per-task signature 整体泛化到 blur；更稳的说法是 visual fragility 能跨 Gaussian-noise axis 出现，但 task ordering 和 recovery profile 是 corruption-specific。
- 不要把 PLDM mechanism 写成 LeWM mechanism 的简单复制。

## 5. 当前 submit-readiness

当前版本已经具备 arXiv / submission draft 的主体条件：

- 主文 story 已闭环：failure → recovery → trade-off → mechanism → boundary。
- LeWM 是主 microscope，PLDM 是 second-family replication。
- Blur 是 cross-corruption sanity check，不阻塞主线。
- 95% checkpoint-row bootstrap CI 已加入，用来约束 partial-correlation 结论强度；CI 宽且包含 0 的地方不声明稳定非零 residual association。
- Success-rate tables 的 uncertainty 是 3 evaluation seeds 的 population std；correlation intervals 是 checkpoint-level bootstrap CI，二者口径已在正文区分。
- `tools/check_paper1_consistency.py` 已覆盖核心 artifact 和关键数值一致性。
- `paper1/main.pdf` 可 clean build。

仍需要人工完成的一项：

- **References final manual source audit。** 机器辅助核对已完成并修正了 metadata / naming 问题；提交前仍建议人工逐条打开 2025/2026 arXiv、OpenReview、LeWM/PLDM 相关页面，确认作者、标题、年份、claim 与正文描述一致。若某条引用无法确认，就删弱相关 claim 或换更稳来源。

## 6. 合作者讨论时的核心问题

- 题目是否应该更强调 “latent prediction is not visual robustness”，还是更强调 “invariance--resolution trade-off”？
- PLDM 放在主文还是 appendix 的分量是否合适？
- Blur eval-only 是否足够作为 sanity check，还是需要后续 blur training v1？
- 五层诊断公式是否已足够清楚，还是需要把更多 metric definition 移到主文？
- 当前 paper 是投 empirical diagnostics 方向，还是后续补 algorithm 后转 method paper？

## 7. 下一步

提交前：

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

提交后 / v1：

- 如需增强 robustness 轴，优先做 blur training sweep。
- 如需增强跨架构说服力，再扩 DreamerV3 / TD-MPC2 / V-JEPA-like baseline。
- 如需转 method paper，再引入 adaptive resolution 或 planner-side robustification；不要让这些阻塞 Paper 1 v0。
