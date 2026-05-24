# paper1 — LaTeX source

Self-contained LaTeX source for the arXiv preprint.

> *Latent Prediction Is Not Visual Robustness: Diagnosing the Invariance--Resolution Trade-off in JEPA World Models for Control*

Companion plan / storyline doc: `PLAN.md` (next to this README).

## Layout

```
paper1/
├── main.tex          # the paper (article class, ~970 lines)
├── references.bib    # 35 entries; final source audit is tracked in PLAN.md
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

To regenerate the 6 figures before building:

```bash
cd .. && python -m tools.paper1_figs --out-dir assets/paper1_figs
```

Paper-specific tool usage is documented in `../tools/README_paper1.md`.

## Submitting to arXiv

arXiv requires a single self-contained tarball. To prepare:

```bash
bash build.sh           # build once, ensure no errors
tar -czf paper1.tar.gz main.tex references.bib main.bbl figures/
```

The figures symlink expands into a real directory inside the tarball (use `-h` if your `tar` does not follow symlinks by default: `tar -czhf paper1.tar.gz ...`).

Note: arXiv does not run BibTeX during build, so the `.bbl` file must be included alongside `main.tex`.
