#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIBLING_ACPX="${SCRIPT_DIR}/../../AutoResearchClaw/experiments/arc_bench/scripts/acpx-node22"

MODEL="${CAIRNLAB_CODEX_MODEL:-gpt-5.5[medium]}"
if [[ -n "${CAIRNLAB_ACPX_COMMAND:-}" ]]; then
  ACPX="${CAIRNLAB_ACPX_COMMAND}"
elif [[ -x "${SIBLING_ACPX}" ]]; then
  ACPX="${SIBLING_ACPX}"
else
  ACPX="acpx-node22"
fi

exec "${ACPX}" --model "${MODEL}" "$@"
