#!/usr/bin/env bash
# Build and audit the double-blind Paper 1 variant.
# Run from repository root: bash paper1/check_blind_ready.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper1"
cd "$PAPER"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

rm -f main_blind.aux main_blind.bbl main_blind.blg main_blind.log \
      main_blind.out main_blind.toc main_blind.fdb_latexmk \
      main_blind.fls main_blind.synctex.gz main_blind.pdf

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_blind.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main_blind.tex
  bibtex main_blind
  pdflatex -interaction=nonstopmode -halt-on-error main_blind.tex
  pdflatex -interaction=nonstopmode -halt-on-error main_blind.tex
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main_blind.bbl" main_blind.log >/tmp/paper1_blind_build_grep.log 2>/dev/null; then
    cat /tmp/paper1_blind_build_grep.log
    fail "unresolved LaTeX references/citations or fatal build diagnostics in blind build"
  fi
else
  if grep -En "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main_blind.bbl" main_blind.log >/tmp/paper1_blind_build_grep.log 2>/dev/null; then
    cat /tmp/paper1_blind_build_grep.log
    fail "unresolved LaTeX references/citations or fatal build diagnostics in blind build"
  fi
fi

[[ -f main_blind.bbl ]] || fail "main_blind.bbl was not generated."

PDF_TEXT=/tmp/paper1_blind_pdf_text.txt
if command -v pdftotext >/dev/null 2>&1; then
  pdftotext main_blind.pdf "$PDF_TEXT"
else
  strings main_blind.pdf > "$PDF_TEXT"
fi

if grep -E -i "Anguo-star|github\.com|Author names to be supplied|Code and data release|Acknowledgements|public repository|LeWM authors" "$PDF_TEXT"; then
  fail "blind PDF still contains self-identifying arXiv/source wording"
fi

rm -rf /tmp/paper1_blind_src
mkdir -p /tmp/paper1_blind_src/figures/corruption
cp main_blind.tex main.tex references.bib main_blind.bbl /tmp/paper1_blind_src/
cp figures/fig2_sweep.png /tmp/paper1_blind_src/figures/
cp figures/pusht_fullseq_selective_contraction_clusters.png /tmp/paper1_blind_src/figures/
cp figures/pusht_fullseq_selective_contraction_atlas.png /tmp/paper1_blind_src/figures/
cp figures/pusht_pldm_noise_selective_contraction_clusters.png /tmp/paper1_blind_src/figures/
cp figures/corruption/pusht_corruption_visualization.png /tmp/paper1_blind_src/figures/corruption/
cp figures/corruption/tworoom_corruption_visualization.png /tmp/paper1_blind_src/figures/corruption/
cp figures/corruption/reacher_corruption_visualization.png /tmp/paper1_blind_src/figures/corruption/
cp figures/corruption/cube_corruption_visualization.png /tmp/paper1_blind_src/figures/corruption/

if grep -R -n -E -i "Anguo-star|github\.com|Author names to be supplied|Acknowledgements|public repository|LeWM authors" /tmp/paper1_blind_src/*.tex; then
  fail "blind source bundle contains self-identifying arXiv/source wording"
fi

tar -czf /tmp/paper1_blind_src.tar.gz -C /tmp/paper1_blind_src .

if tar -tzf /tmp/paper1_blind_src.tar.gz | grep -E '(^|/)(arxiv_metadata\.tex|arxiv_acknowledgements\.tex|PLAN|CODEX|ARXIV_V1|FINAL_SUBMISSION_AUDIT|\.git|.*\.log|.*\.aux|.*\.out|.*\.toc|.*\.fls|.*\.fdb_latexmk|main\.pdf|main_blind\.pdf)$'; then
  fail "blind source tarball contains arXiv metadata, acknowledgements, internal planning/build files, or PDFs."
fi

echo "OK: Paper 1 double-blind readiness checks passed."
echo "Source bundle: /tmp/paper1_blind_src.tar.gz"
