# paper1 — LaTeX source

Self-contained LaTeX source for the arXiv preprint.

> *Action-Conditioned Predictive Consistency as a Diagnostic for Gaussian-Noise Robustness in JEPA World Models*

Companion plan / storyline doc: `PLAN.md` (next to this README).

## Layout

```
paper1/
├── main.tex          # the paper (article class)
├── references.bib    # bibliography entries; final source audit is tracked in reference_audit.md
├── figures/          # symlink → ../assets/paper1_figs/
├── build.sh          # `bash build.sh` (uses latexmk if available, else pdflatex + bibtex)
├── .gitignore        # ignores LaTeX intermediates; main.pdf is tracked intentionally
└── README.md         # this file
```

## Build

Requires `texlive-latex-recommended` + `texlive-bibtex-extra` (or any TeX distribution with `pdflatex`, `bibtex`, and the packages listed in `main.tex`).

```bash
bash build.sh           # builds main.pdf
bash build.sh --clean   # remove intermediates first
```

To regenerate the script-generated main figures before building:

```bash
cd .. && python -m tools.paper1_figs --out-dir assets/paper1_figs
cd .. && python tools/paper1_atr_smpr_figure.py
cd .. && python tools/paper1_feature_neighborhood_figure.py
```

The feature-neighborhood figure is a qualitative illustration. It reuses the cached PushT feature arrays in `/tmp/paper1_selective_contraction_cache`; if that cache is absent, regenerate it with the optional selective-contraction tooling documented in `../tools/README_paper1.md`.

Paper-specific tool usage is documented in `../tools/README_paper1.md`.

## Submitting to arXiv

The current source is configured as an arXiv-style non-anonymous draft. Before submitting, replace the `\arxivauthors` placeholder in `main.tex` with the real author list and verify the public code/data URL printed after the abstract.

Current public companion repository: `https://github.com/Anguo-star/lewm-acpc-diagnostics`.

For a double-blind conference variant, use `main_blind.tex` and run `bash paper1/check_blind_ready.sh` from the repository root. The blind path compiles the same paper with anonymous authors, hides the public code URL and acknowledgements, and creates `/tmp/paper1_blind_src.tar.gz` without `arxiv_metadata.tex` or `arxiv_release_notes.tex`.

arXiv source upload should contain only files required to compile the paper. Do not upload `PLAN.md`, `CODEX_SUBMISSION_READINESS.md`, `ARXIV_V1_READINESS_PLAN.md`, checker logs, old PDFs, raw experiment JSON, or other internal planning files.

To prepare a source tarball with only the figures referenced by `main.tex`:

```bash
cd paper1
bash build.sh --clean
rm -rf /tmp/paper1_arxiv_src
mkdir -p /tmp/paper1_arxiv_src/figures
cp main.tex arxiv_metadata.tex arxiv_release_notes.tex references.bib main.bbl /tmp/paper1_arxiv_src/
cp ../assets/paper1_figs/fig2_sweep.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_atr_smpr_plane.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_feature_neighborhood_atr_smpr.png /tmp/paper1_arxiv_src/figures/
tar -czf /tmp/paper1_arxiv_v1_src.tar.gz -C /tmp/paper1_arxiv_src .
tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | sort
```

The source package intentionally excludes `main.pdf`, unused figures, and local build products; the TeX source path includes `main.bbl`, whose basename matches `main.tex`.
