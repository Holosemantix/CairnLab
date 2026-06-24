# Paper 1 no-marker sweep audit — 2026-06-24

This audit responds to the remaining Figure 2 issue: marking a single train-time noise level, even as the highest observed robust-eval mean, gives one grid point too much visual importance. Under the current protocol, the correct message is the shape of the sweep and the plateau/high-performing ranges, not a selected `sigma`.

## Overall verdict

The theory section is still sound and useful. The remaining submission risk is statistical and rhetorical: any single-line marker in Figure 2 invites reviewers to ask whether that point is a statistically meaningful optimum. The safest fix is to remove the vertical marker entirely and let the curves with error bars show the plateau.

## Required patch

### Figure 2 generator

In `tools/paper1_figs.py`, edit `fig2_sweep` so that it only plots:

- unperturbed evaluation curve;
- observation-noise 0.08 curve;
- evaluation-seed error bars.

Remove:

- `best_std = ...`;
- `ax.axvline(...)`;
- `label=...Robust-eval...` marker;
- title text containing `sigma*` or reference `sigma`.

Use a plain task title:

```python
ax.set_title(t, fontsize=11)
```

The shared legend should have two columns, not three.

### Figure 2 caption

Replace any caption phrase about `green dashed`, `highest observed`, `point-best`, `sigma*`, or reference marker. Suggested caption:

```tex
\caption{Noise-training sweep across four tasks. Blue: unperturbed evaluation; red: observation-only Gaussian noise $\sigma=0.08$ with an unperturbed goal. Error bars show the population standard deviation across the three evaluation seeds. The curves show broad task-dependent high-performing regions rather than a statistically unique training-noise optimum.}
```

### Sweep table and prose

Do not call any row an optimum. If a compact table still lists a representative row, call it:

- `representative high grid row`, or
- `highest observed mean in this grid`, only in table caption, with a clear warning that it is not unique.

Main prose should emphasize ranges:

```text
The sweep shows broad task-dependent high-performing regions. TwoRoom and Reacher have plateau-like behaviour, PushT recovers over a range of moderate-to-high noise levels, and Cube has a weak/non-monotone response. The claim is not that a particular sigma is optimal, but that one scalar Gaussian-noise strength gives only a coarse task-dependent control.
```

## Similar issues to remove

Search and remove or rephrase in `main.tex`, `tools/paper1_figs.py`, and paper-specific docs:

```text
robust-eval optimum
sigma*
sigma^*
green dashed
unique optimum
optimal sigma
best checkpoint
selected endpoint
noise-best
```

`point-best` should not appear in main text. If it appears in appendix tables inherited from artifact names, it must be clearly explained as a grid maximum used only for compact display, not as statistical evidence of a unique optimum.

## Five strict reviewer perspectives

### Statistical reviewer

A vertical marker is not justified under three evaluation seeds and one training run per grid point. Remove it entirely. Error bars and full sweep curves are the evidence.

### Theory reviewer

The fixed-candidate ACPC theory is unaffected. It links rollout consistency to candidate ranking, not training-noise hyperparameter selection. Do not use the theory to justify a unique augmentation strength.

### RL/world-model reviewer

Plateau behaviour is more plausible and more useful than exact sigma selection. The paper should claim task-dependent recovery regions, not point optima.

### Writing reviewer

Single-point markers create the impression of overfit hyperparameter storytelling. Curves plus concise plateau prose read more mature and less AI-like.

### Artifact reviewer

After editing `tools/paper1_figs.py`, regenerate `assets/paper1_figs/fig2_sweep.png`; otherwise the old PNG can still contain the wrong label even if `main.tex` is fixed.

## Required commands after patch

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

Then grep:

```bash
rg -n "robust-eval optimum|sigma\\*|sigma\\^\\*|green dashed|unique optimum|optimal sigma|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

Expected: no main-text or Figure 2 generator hits.
