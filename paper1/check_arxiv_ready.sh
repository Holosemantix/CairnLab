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

if grep -q "tab:theory-metric-map\|tab:sweep-summary\|fig:atr-smpr-plane\|fig_atr_smpr_plane\|fig_feature_neighborhood_atr_smpr" main.tex; then
  fail "main.tex still references a removed table or figure from the pre-convergence draft."
fi

if ! grep -q "fig_acpc_basin_tsne.png" main.tex; then
  fail "main.tex should reference the canonical qualitative ACPC t-SNE figure."
fi

for figure in fig1_concept.png fig_endpoint_atr_smpr.png fig_full_sweep_diagnostics.png fig_fixed_pool_event_rates.png fig_gaussian_sensitivity_mechanism.png fig_radius_margin_overlap.png; do
  if ! grep -q "$figure" main.tex; then
    fail "main.tex should reference $figure for the radius-margin diagnostic validation."
  fi
done

# Build first; build.sh also greps undefined refs/cites/fatal diagnostics.
bash build.sh --clean

[[ -f main.bbl ]] || fail "main.bbl was not generated; arXiv source package should include main.bbl matching main.tex."

# Prepare a minimal arXiv source bundle in /tmp and audit obvious internal files.
rm -rf /tmp/paper1_arxiv_src
mkdir -p /tmp/paper1_arxiv_src/figures /tmp/paper1_arxiv_src/tables
cp main.tex arxiv_metadata.tex arxiv_release_notes.tex references.bib main.bbl /tmp/paper1_arxiv_src/

# Keep this list aligned with figure inclusions in main.tex.
cp ../assets/paper1_figs/fig1_concept.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig2_sweep.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_endpoint_atr_smpr.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_acpc_basin_tsne.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_full_sweep_diagnostics.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_fixed_pool_event_rates.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_gaussian_sensitivity_mechanism.png /tmp/paper1_arxiv_src/figures/
cp ../assets/paper1_figs/fig_radius_margin_overlap.png /tmp/paper1_arxiv_src/figures/
cp tables/table_heldout_diagnostic_validation.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_endpoint_atr_smpr.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_theory_evidence_map.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_fixed_pool_tail_audit.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_sample_level_certificate_full_sweep.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_sample_level_event_rate_ci.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_sample_level_certificate_endpoint.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_joint_guard_side_validation.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_gaussian_sensitivity_audit.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_jvp_hutchinson_sensitivity_audit.tex /tmp/paper1_arxiv_src/tables/
cp tables/table_threshold_quantile_sensitivity.tex /tmp/paper1_arxiv_src/tables/

tar -czf /tmp/paper1_arxiv_v1_src.tar.gz -C /tmp/paper1_arxiv_src .

if tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | grep -E '(^|/)(PLAN|CODEX|ARXIV_V1|FINAL_SUBMISSION_AUDIT|\.git|.*\.log|.*\.aux|.*\.out|.*\.toc|.*\.fls|.*\.fdb_latexmk|.*\.synctex\.gz|main\.pdf)$'; then
  fail "arXiv source tarball contains internal planning/build/output files."
fi

echo "OK: Paper 1 arXiv readiness checks passed."
echo "Source bundle: /tmp/paper1_arxiv_v1_src.tar.gz"
