# Paper 1 整改后校验与理论部分二次审稿整改清单（2026-06-26）

> 审核对象：`Holosemantix/le-wm@ag/dev` 当前 `paper1/main.tex` 与 `paper1/references.bib`。  
> 结论级别：**比上一版明显更接近投稿，但理论段仍有 3 个必须修的小口子；参考文献还缺一篇非常相关的 JEPA noisy-features 理论文献；arXiv 作者占位符和公开 URL 仍是提交前 hard blocker。**

---

## 0. 总体 verdict

当前版本已经解决了上一轮几个主要问题：

- 标题已经收窄为 `Gaussian Visual Robustness`，不再像泛化大论文。
- 摘要明确说这是 diagnostic framework/release package，不是新训练目标，也不是 closed-loop guarantee。
- `drop` 已统一为 `clean success - obs-noise 0.08 success`，主文和 appendix 表含义一致。
- ACPC 已经固定到主文实例：identity readout、L2 rollout tokens、`H=8`、`R_E/R_F`。
- 理论段现在有 fixed-candidate stability、pseudo-metric、collapse counterexample，方向正确。

但**还不能直接交**，原因如下。

---

## 1. P0 hard blockers

### 1.1 作者占位符仍在，会直接导致 readiness fail

当前 `main.tex` 仍有：

```tex
\newcommand{\arxivauthors}{Author names to be supplied for arXiv v1}
\author{\arxivauthors}
```

这和 `paper1/check_arxiv_ready.sh` 的 hard blocker 冲突。提交前必须替换为真实作者列表。

**Codex 执行：**

```bash
rg -n "Author names to be supplied|\\arxivauthors" paper1/main.tex paper1/check_arxiv_ready.sh
```

预期：`main.tex` 无命中；`check_arxiv_ready.sh` 可保留检查逻辑。

---

### 1.2 Acknowledgements 的公开仓库 URL 仍需确认

当前 acknowledgement 写：

```tex
Code, aggregate evaluation artifacts, rendering scripts, and data/checkpoint pointers for this revision are available at \url{https://github.com/Anguo-star/le-wm};
```

本轮实际审的是 `Holosemantix/le-wm@ag/dev`。我尝试直接 fetch `Anguo-star/le-wm` 的 `paper1/main.tex` 没有拿到文件，因此不能确认这个公开 URL 已包含 paper1 source/artifacts。

**两种安全做法：**

1. 如果最终公开仓库是 `Anguo-star/le-wm`，必须先把当前 `paper1/`、`assets/paper1_data/`、`assets/paper1_figs/`、tools、manifest 同步过去，并确认公开可访问。
2. 如果最终公开仓库是组织仓库或 release tag，则把 URL 改成最终 tag/commit，而不是裸 main branch。

**建议最终写法：**

```tex
Code, aggregate evaluation artifacts, rendering scripts, and data/checkpoint pointers are available at \url{<final-public-repo-or-release-tag>}; JSON artifact hashes are listed in the data manifest.
```

---

## 2. 理论分析：现在方向正确，但还需三处修补才能完全站得住脚

### 2.1 ACPC 标量定义和 Proposition 里的 `d_H` 还没有完全对齐

当前 ACPC 定义是：

```tex
ACPC_H = \sum_k \alpha_k d( ... )
```

但后面 cost-drift proposition 使用的是一个 rollout-sequence metric `d_H(...) <= epsilon`。这二者直觉上一致，但形式上没有写死。严格审稿人会问：你说 “ACPC controls candidate-cost drift”，到底是 `ACPC_H` 这个标量，还是另一个 `d_H` metric？

**必须修。** 在 ACPC 定义处加一句，把 `d_H` 定义成同一个 aggregate metric：

```tex
For the theoretical statements below, we use the induced horizon metric
\[
  d_H\big((u_1,\ldots,u_H),(v_1,\ldots,v_H)\big)
  = \sum_{k=1}^{H}\alpha_k d(u_k,v_k),
\]
so that Equation~\eqref{eq:acpc-h} is exactly the clean/corrupted rollout-readout distance under the shared action sequence. In experiments we instantiate this with the L2 distance over rollout tokens used by $R_F$.
```

如果不加这句，理论段不是错，但会显得“ACPC”和“用于证明的 metric”之间有缝。

---

### 2.2 `Selective predictive stability` 现在不该写成一个混合 corollary

当前 `Selective predictive stability` 的前半句是严格结论：`D_ACPC^A <= epsilon + Lipschitz + margin => fixed-set top-1 candidate stable`。这没问题。

但后半句：

```tex
If, in addition, action-, transition-, or cost-distinct pairs preserve a diagnostic margin ... the stability is selective under that proxy; without this second condition, the statement does not rule out collapse.
```

这更像**定义/解释/guard condition**，不是由前面数学推出的 corollary。严格审稿人可能会说：你把 empirical proxy 写进 corollary 里，混淆了 theorem statement 和 diagnostic assumption。

**建议替换为：**

```tex
\begin{corollary}[Fixed-candidate predictive stability]\label{cor:selective-stability}
Assume $D_{\mathrm{ACPC}}^{\mathcal A}(h,\tilde h)\le\epsilon$ for a same-state visual-corruption pair and that $J$ is $L_J$-Lipschitz on the rollout readout. If the clean candidate margin satisfies $\Delta>2L_J\epsilon$, then the clean and corrupted branches select the same top-1 candidate from the fixed set $\mathcal A$.
\end{corollary}

\begin{proof}
By definition of $D_{\mathrm{ACPC}}^{\mathcal A}$, every candidate in $\mathcal A$ has rollout-readout discrepancy at most $\epsilon$. Proposition~\ref{prop:cost-drift} bounds the cost drift of every candidate by $L_J\epsilon$, and Proposition~\ref{prop:top1-stability} gives top-1 stability when $\Delta>2L_J\epsilon$.
\end{proof}

The statement becomes selective only when it is paired with a discriminability condition such as \Cref{eq:acpc-disc}. Without that second condition, fixed-candidate predictive stability can still be achieved by a collapsed encoder--predictor, as shown next.
```

这样更干净：数学结论归数学，selective/collapse guard 归解释和下一条 proposition。

---

### 2.3 pseudo-metric 这句是对的，但最好加一句证明或降级为 remark

当前写：

```tex
When d_H is a metric, D_ACPC^A is a pseudo-metric on histories ...
```

这个命题是成立的：每个 candidate action sequence 拉回一个 pseudometric，有限/任意 pointwise maximum 仍是 pseudometric。但论文里直接一笔带过，容易让理论审稿人觉得“又加了一个术语但没证明”。

**二选一：**

- 如果保留术语，补一句：

```tex
This follows because each map $h\mapsto \Pi(F^{1:H}_\theta(E_\theta(h),\mathbf a))$ pulls back $d_H$ to a pseudo-metric, and the pointwise maximum of pseudo-metrics is again a pseudo-metric.
```

- 如果想更稳，直接把标题改为 `Fixed-candidate rollout discrepancy`，不用 pseudo-metric 这个术语。

我建议保留并补一句，因为这是你们理论部分“自己的东西”的一部分。

---

### 2.4 摘要中的 “selected action” 应改成 “selected candidate/action sequence”

摘要现在写：

```tex
preserves the selected action under a clean margin condition
```

理论只证明 fixed candidate set 里的 top-1 candidate/action sequence 稳定，不证明 CEM 真实采样过程、不证明 repeated replanning、不证明 closed-loop action trajectory。

**改成：**

```tex
preserves the selected candidate action sequence within that fixed set under a clean margin condition
```

或者更短：

```tex
preserves the top-ranked candidate within that fixed set under a clean margin condition
```

---

### 2.5 `J` 的 Lipschitz assumption 建议写成 local，而不是全局

当前 proposition 写：

```tex
Assume that J is L_J-Lipschitz ...
```

全局 Lipschitz 对 neural/cost readout 可能能成立但没必要承担。更稳的写法：

```tex
Assume that $J$ is locally $L_J$-Lipschitz on the clean/corrupted rollout-readout neighborhood evaluated by the fixed candidate set.
```

这样更贴近 diagnostic proof，也避免被问“全局 Lipschitz 常数从哪来”。

---

## 3. 理论参考文献：已有一篇，但还缺一篇不能忽略的 JEPA noisy-features 理论文献

### 3.1 `vanassel2025jointembeddingreconstruction` 已在 bib 和正文中

当前 bib 已有：

```bibtex
@misc{vanassel2025jointembeddingreconstruction,
  title = {Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self-Supervised Learning},
  ...
}
```

正文也已引用它来支撑 “latent prediction can reduce pressure to encode high-magnitude irrelevant features, but still needs augmentation/bias”。这篇确实是很相关的理论文献：它用 closed-form solutions 比较 reconstruction 与 joint embedding，并明确分析 augmentation/noise alignment 与 high-magnitude irrelevant features。

**问题：** 当前对它的使用偏“背景一句话”，没有和我们理论段建立边界。建议在 Related Work 或 ACPC 开头加一个 paragraph，把它和我们的理论贡献区分清楚。

---

### 3.2 建议新增 `littwin2024jepaavoidsnoisyfeatures`

如果你说的“视觉扰动或高斯噪声下的理论分析文章”是 **How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks**，当前 references.bib 里没有它，需要加。

**BibTeX 建议：**

```bibtex
@misc{littwin2024jepaavoidsnoisyfeatures,
  title         = {How {JEPA} Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks},
  author        = {Littwin, Etai and Saremi, Omid and Advani, Madhu and Thilak, Vimal and Nakkiran, Preetum and Huang, Chen and Susskind, Joshua},
  year          = {2024},
  eprint        = {2407.03475},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2407.03475},
  url           = {https://arxiv.org/abs/2407.03475}
}
```

**正文建议插入：** 在 `Invariance, augmentation, and visual robustness` 小节，把 Van Assel 句子扩成：

```tex
Recent theory clarifies why latent prediction can help with nuisance or noisy features but also why it is not enough for control. Van Assel et al.~\cite{vanassel2025jointembeddingreconstruction} derive closed-form SSL solutions showing that joint embedding imposes weaker alignment requirements than reconstruction when irrelevant features have large magnitude, while Littwin et al.~\cite{littwin2024jepaavoidsnoisyfeatures} analyze deep-linear self-distillation dynamics and show a JEPA bias toward high-influence predictive features rather than merely high-variance features. These analyses are representation-level and do not address closed-loop action-conditioned rollout, candidate-cost stability, or the need to keep action-distinct states separable. ACPC is complementary: it asks whether same-state visual perturbations become equivalent after a fixed action intervention while action-, transition-, or cost-distinct cases remain discriminable.
```

这样写有三个好处：

1. 不会显得你们忽略已有理论。
2. 不会把已有理论误写成你们的贡献。
3. 清楚说明你们“自己的东西”在哪里：**action-conditioned rollout + fixed-candidate cost/margin stability + discriminability guard**。

---

### 3.3 加引用后必须更新 reference audit

当前 `reference_audit.md` 写的是 42 entries。如果新增 Littwin，需要更新：

- entry count：42 -> 43。
- 新增一行 official source check。
- 检查 `references.bib` 每个 entry 都被 main.tex cite。

**Codex 执行：**

```bash
python - <<'PY'
import re, pathlib
tex = pathlib.Path('paper1/main.tex').read_text()
bib = pathlib.Path('paper1/references.bib').read_text()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cites = set(k.strip() for m in re.findall(r'\\cite\{([^}]+)\}', tex) for k in m.split(','))
print('unused:', sorted(keys-cites))
print('missing:', sorted(cites-keys))
print('n_bib:', len(keys), 'n_cites:', len(cites))
PY
```

预期：`unused=[]`, `missing=[]`。

---

## 4. 我们自己的理论分量是否足够？

### 4.1 够，但前提是把 claim 定位成 diagnostic theory，不是 robustness theorem

当前理论分量的核心可以站住：

1. **定义层：** ACPC 把 same-state visual perturbation 的一致性放到 action-conditioned rollout 后，而不是 encoder latent 距离上。
2. **规划层：** fixed candidate set 下，rollout-readout disagreement 通过 Lipschitz cost readout 控制 candidate-cost drift；若 clean top-1/top-2 margin 足够大，则 top candidate 不变。
3. **选择性层：** ACPC alone admits collapse，因此必须配 discriminability guard。
4. **实证实例层：** `R_E/R_F` 作为 ACPC-basin proxy，rank/transition-resolution/ID probe/hetero failure 作为 guard evidence。

这不是很深的理论，但作为 empirical diagnostic paper 的 theory skeleton 是够的。它的独立性在于：已有理论解释“为什么 JEPA/latent prediction可能避开 noisy/high-variance features”；你们解释“为什么 control robustness 必须在 action-conditioned predictive rollout 和 candidate cost 之后诊断”。

### 4.2 不要把理论包装成以下内容

必须避免：

- full CEM stability theorem；
- closed-loop trajectory robustness；
- learned representation identifiability；
- Gaussian-noise augmentation必然恢复 robustness；
- ACPC basin可以预测成功率；
- discriminability proxy 等于 oracle margin proof。

当前主文基本避开了这些，但摘要里的 selected action、小 corollary 里的混合 proxy、`d_H` 未对齐仍需修。

---

## 5. 其他提交前校验发现

### 5.1 `drop` 已修好

主文 corruption table caption 已说明 drop 是 clean minus obs0.08，表里数值为正；ACPC basin 表和 full grid 也说明 negative drop means no degradation。这个整改通过。

### 5.2 DrQ 作者顺序已修好

当前 bib 中：

```bibtex
author = {Kostrikov, Ilya and Yarats, Denis and Fergus, Rob}
```

这已修复。

### 5.3 主文有较多 `[H]` figure/table placement，需本地 PDF 视觉检查

为了压版面，当前主文和 appendix 多处使用 `[H]`。这本身不一定错，但容易造成：

- 大块空白；
- 图表堆积；
- landscape 前后分页怪；
- arXiv 编译后位置变化。

**Codex 必跑：**

```bash
cd paper1
bash build.sh --clean
python - <<'PY'
from pathlib import Path
log = Path('main.log').read_text(errors='ignore')
for pat in ['Overfull \\hbox', 'Underfull \\vbox', 'Float too large', 'LaTeX Warning: Float']:
    if pat in log:
        print('\n---', pat, '---')
        for line in log.splitlines():
            if pat in line:
                print(line)
PY
```

并人工打开 PDF 检查：Fig.2 曲线颜色/marker 是否可区分，Fig. selective contraction 小字是否可读，landscape 表是否旋转正常。

---

## 6. 最小 patch 顺序

按这个顺序交给 Codex：

1. 替换真实作者；确认/修复公开 URL。
2. 理论段：定义 `d_H` 与 ACPC 对齐。
3. 理论段：把 `Selective predictive stability` 改为 clean corollary + proof；把 discriminability proxy 移到解释段。
4. 理论段：pseudo-metric 补一句证明，或降级为 rollout discrepancy。
5. 摘要：`selected action` -> `top-ranked candidate within the fixed set`。
6. Lipschitz：global -> local on evaluated rollout neighborhood。
7. 新增 `littwin2024jepaavoidsnoisyfeatures`，扩写 related-work 理论边界段。
8. 更新 `reference_audit.md` count 和新增 entry。
9. 跑 consistency/build/arxiv-ready。

---

## 7. 推荐最终中心 claim

建议全文最终对齐到这句话：

> Existing JEPA/noisy-feature theory explains why latent prediction can reduce pressure to encode nuisance or high-variance features under appropriate augmentation assumptions. Our contribution is complementary: for world-model control, visual robustness must be diagnosed after the action-conditioned predictor. Under a fixed candidate set, bounded clean/corrupted rollout disagreement bounds candidate-cost drift and preserves the top-ranked candidate under a margin condition; because this condition alone admits collapse, it must be paired with an action-relevant discriminability guard. The LeWM/PLDM Gaussian-noise sweeps instantiate this diagnostic principle empirically without claiming a new training algorithm or full closed-loop guarantee.

这句话有分量，也不会被理论审稿人轻易打穿。
