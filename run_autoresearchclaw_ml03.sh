#!/usr/bin/env bash
set -euo pipefail

REPO="/opt/huawei/explorer-env/dataset/ag_data/code/AutoResearchClaw"
RUN_DIR="${REPO}/experiments/arc_bench/results/e2e/ML03/e2e-ML03-20260604-093524"
CONFIG="${RUN_DIR}/config.yaml"
ACPX="${REPO}/experiments/arc_bench/scripts/acpx-node22"
SESSION_NAME="arc-ml03-codex"
CODEX_MODEL="gpt-5.4[high]"

cd "${REPO}"
source .venv/bin/activate

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  read -rsp "GITHUB_TOKEN: " GITHUB_TOKEN
  echo
  export GITHUB_TOKEN
fi

export ARC_SKIP_JUDGE=1
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:8888}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTP_PROXY}"
export WS_PROXY="${WS_PROXY:-$HTTP_PROXY}"
export WSS_PROXY="${WSS_PROXY:-$HTTP_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export ALL_PROXY="${ALL_PROXY:-socks5h://127.0.0.1:1080}"

echo "[1/4] proxy check"
curl -fsS -x "${HTTP_PROXY}" https://api.ipify.org
echo

echo "[2/4] github token/code-search check"
python - <<'PY'
from researchclaw.agents.code_searcher.github_client import GitHubClient

r = GitHubClient().search_code(
    "scipy optimize powell nonconvex example",
    max_results=1,
)
if not r:
    raise SystemExit("GitHub code search failed or returned no results")
print("github_code_search_ok=True")
print("first=", f"{r[0].repo_full_name}:{r[0].file_path}")
PY

echo "[3/4] acpx/codex session check"
"${ACPX}" --ttl 0 \
  --cwd "${REPO}" \
  --model "${CODEX_MODEL}" \
  codex sessions ensure --name "${SESSION_NAME}"

"${ACPX}" --approve-all --max-turns 1 --ttl 0 \
  --cwd "${REPO}" \
  codex -s "${SESSION_NAME}" 'Reply exactly: READY'

echo "[4/4] run ML03 from Stage 10"
python -u -m researchclaw run \
  --config "${CONFIG}" \
  --output "${RUN_DIR}" \
  --profile ml_general \
  --auto-approve \
  --skip-preflight \
  --from-stage CODE_GENERATION
