#!/usr/bin/env bash
# Lightweight arXiv submission readiness checks for Paper 1.
# Run from repository root: bash paper1/check_arxiv_ready.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper1"
ALLOW_AUTHOR_PLACEHOLDER="${ALLOW_AUTHOR_PLACEHOLDER:-0}"
cd "$PAPER"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# Hard blockers that should not reach arXiv.
if grep -q "Author names to be supplied" arxiv_metadata.tex && [[ "$ALLOW_AUTHOR_PLACEHOLDER" != "1" ]]; then
  fail "main.tex still contains the arXiv author placeholder. Replace \\arxivauthors with the real author list."
fi

if grep -q "Author names to be supplied" arxiv_metadata.tex; then
  echo "WARN: author placeholder allowed because ALLOW_AUTHOR_PLACEHOLDER=1; replace it before final arXiv upload." >&2
fi

if grep -q "\\\\author{}" main.tex; then
  fail "main.tex contains an empty \\author{} field. arXiv v1 must be non-anonymous."
fi

if grep -q "Scope of this arXiv version" main.tex; then
  fail "main.tex still contains internal release-note wording: 'Scope of this arXiv version'. Use 'Scope' / 'This paper' instead."
fi

if grep -q "paper-facing\|method-facing" main.tex; then
  fail "main.tex still contains paper-facing/method-facing internal wording. Move such wording to tooling notes or rewrite for readers."
fi

if grep -q "tree/ag/dev\|tree/main" main.tex arxiv_metadata.tex; then
  fail "main.tex points to a branch-specific GitHub tree. Use the public repository root URL for arXiv."
fi

if grep -q "complete code and data" main.tex arxiv_metadata.tex arxiv_release_notes.tex; then
  fail "main.tex over-claims the release package as 'complete code and data'. Use code/artifacts/scripts/pointers wording."
fi

if ! grep -q "https://github.com/Anguo-star/lewm-acpc-diagnostics" arxiv_metadata.tex; then
  fail "main.tex does not contain the intended public repository URL https://github.com/Anguo-star/lewm-acpc-diagnostics."
fi

if ! grep -q "Drop is clean success minus observation-noise 0.08 success" main.tex; then
  fail "main.tex should define the corruption-table drop direction explicitly."
fi

# Build first; build.sh also greps undefined refs/cites/fatal diagnostics.
bash build.sh --clean

[[ -f main.bbl ]] || fail "main.bbl was not generated; arXiv source package should include main.bbl matching main.tex."

# Prepare a minimal arXiv source bundle in /tmp and audit obvious internal files.
rm -rf /tmp/paper1_arxiv_src
mkdir -p /tmp/paper1_arxiv_src/figures/corruption
cp main.tex arxiv_metadata.tex arxiv_release_notes.tex references.bib main.bbl /tmp/paper1_arxiv_src/

# Keep this list aligned with figure inclusions in main.tex.
cp figures/fig2_sweep.png /tmp/paper1_arxiv_src/figures/
cp figures/pusht_fullseq_selective_contraction_clusters.png /tmp/paper1_arxiv_src/figures/
cp figures/pusht_fullseq_selective_contraction_atlas.png /tmp/paper1_arxiv_src/figures/
cp figures/pusht_pldm_noise_selective_contraction_clusters.png /tmp/paper1_arxiv_src/figures/
cp figures/corruption/pusht_corruption_visualization.png /tmp/paper1_arxiv_src/figures/corruption/
cp figures/corruption/tworoom_corruption_visualization.png /tmp/paper1_arxiv_src/figures/corruption/
cp figures/corruption/reacher_corruption_visualization.png /tmp/paper1_arxiv_src/figures/corruption/
cp figures/corruption/cube_corruption_visualization.png /tmp/paper1_arxiv_src/figures/corruption/

tar -czf /tmp/paper1_arxiv_v1_src.tar.gz -C /tmp/paper1_arxiv_src .

if tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | grep -E '(^|/)(PLAN|CODEX|ARXIV_V1|FINAL_SUBMISSION_AUDIT|\.git|.*\.log|.*\.aux|.*\.out|.*\.toc|.*\.fls|.*\.fdb_latexmk|.*\.synctex\.gz|main\.pdf)$'; then
  fail "arXiv source tarball contains internal planning/build/output files."
fi

echo "OK: Paper 1 arXiv readiness checks passed."
echo "Source bundle: /tmp/paper1_arxiv_v1_src.tar.gz"
