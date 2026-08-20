#!/bin/bash
# =============================================================================
# HMATS v6.8.0 — Hetzner Deploy / Update
# =============================================================================
# Run from LOCAL machine:
#   bash scripts/hetzner_deploy.sh <server-alias>
#
# Prerequisites:
#   - SSH config has the server alias (e.g., "hmats")
#   - Server bootstrapped with hetzner_bootstrap.sh
#   - .env placed on server at ~/hmats/app/.env
#   - Models uploaded to ~/hmats/models/
# =============================================================================
set -euo pipefail

SERVER="${1:-hmats}"
REMOTE_USER="hmats"
APP_DIR="/home/${REMOTE_USER}/hmats/app"
COMPOSE_FILE="docker-compose.hetzner.yml"

echo "=== HMATS Deploy to ${SERVER} ==="

# --- Step 0: Pre-deploy gates ---
# [P253b] TWO gates, replacing the earlier designs:
#
#   0a. CI-GREEN ON THE DEPLOYED SHA. Step 1 below runs `git pull origin
#       main` on the server, so what gets deployed is origin/main — and the
#       authority on whether origin/main type-checks is CI, not this
#       machine: the mypy baseline is a fingerprint of CI's environment
#       (P227: identical code measures 1076 findings in CI and 1083+ on the
#       operator's Windows venv at the SAME mypy 2.3.0). The first P253
#       design (--require-all-gates locally) therefore blocked EVERY deploy
#       from this machine on phantom findings; requiring "mypy installed
#       locally" (P192's note) had the same flaw one step later. Verifying
#       the deployed sha's CI conclusions closes the P187 hole in the only
#       environment where the type gate is meaningful — and it
#       operationalizes the P233/P252 standing rule (a push is not done
#       until both workflow conclusions are READ via the API).
#
#   0b. Local env-independent scanners (--skip-mypy). stdlib+git only, so
#       they mean the same thing on every machine; kept because they catch
#       a scanner regression before any server churn (P111).
#
#       [P328] They run against a WORKTREE AT THE DEPLOYED SHA, not against
#       the working tree. Step 1 deploys whatever origin/main holds, so
#       scanning the checkout answers a different question — and in a shared
#       checkout it answers it about someone else's uncommitted edits. That
#       happened: a clean commit was refused because a parallel session had
#       four unrelated findings in files it was still editing. The working
#       tree cannot tell you what you committed (P311b). If the worktree
#       cannot be created the scan falls back to the checkout and SAYS SO —
#       a silently different subject is what this fixes.
#
# Override for a genuine emergency (API outage while the box is on fire):
#   HMATS_DEPLOY_SKIP_CI_CHECK=1 bash scripts/hetzner_deploy.sh hmats
echo "[0/5] Pre-deploy gates..."

DEPLOY_SHA="$(git ls-remote origin refs/heads/main | cut -f1)"
LOCAL_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -z "${DEPLOY_SHA}" ]; then
    echo "ERROR: could not resolve origin/main (git ls-remote failed)."
    echo "Refusing to deploy a sha we cannot identify."
    exit 1
fi
if [ "${DEPLOY_SHA}" != "${LOCAL_SHA}" ]; then
    echo "WARNING: local HEAD (${LOCAL_SHA:0:9}) != origin/main (${DEPLOY_SHA:0:9})."
    echo "  The server deploys ORIGIN/MAIN. Unpushed local commits will NOT deploy."
fi

# [P287] Resolve the python interpreter ONCE, loudly. `command -v python`
# alone silently skipped the whole local scanner gate on python3-only
# machines (the P159 shape: a check that cannot run reading as a check
# that passed), and the CI-verdict parser below needs an interpreter too.
if command -v python &>/dev/null; then
    PY_BIN=python
elif command -v python3 &>/dev/null; then
    PY_BIN=python3
else
    echo "ERROR: neither 'python' nor 'python3' is on PATH — the CI-verdict"
    echo "  parser and the local scanner gate cannot run. Refusing to deploy"
    echo "  with unverifiable gates (P159/P287)."
    exit 1
fi

if [ "${HMATS_DEPLOY_SKIP_CI_CHECK:-0}" = "1" ]; then
    echo "  !! CI-green check SKIPPED by HMATS_DEPLOY_SKIP_CI_CHECK=1 —"
    echo "  !! deploying ${DEPLOY_SHA:0:9} with UNVERIFIED CI status."
else
    ORIGIN_URL="$(git remote get-url origin)"
    REPO_SLUG="$(echo "${ORIGIN_URL}" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
    echo "  Checking CI conclusions for ${REPO_SLUG}@${DEPLOY_SHA:0:9}..."
    CI_JSON="$(curl -sf --max-time 30 \
        "https://api.github.com/repos/${REPO_SLUG}/actions/runs?head_sha=${DEPLOY_SHA}" \
        || echo "")"
    CI_VERDICT="$(printf '%s' "${CI_JSON}" | "${PY_BIN}" -X utf8 -c "
import json, sys
try:
    runs = json.load(sys.stdin).get('workflow_runs', [])
except Exception:
    print('API_UNREADABLE'); raise SystemExit
need = {'codebase-invariants', 'test-suite'}
got = {}
for r in runs:
    n = r.get('name')
    # [P287] The API returns runs NEWEST-FIRST; keep only the FIRST match
    # per workflow. The old unconditional overwrite meant the OLDEST run
    # won — a sha whose first run was green and whose re-run went red read
    # as GREEN and deployed (backwards fail direction on a safety gate).
    if n in need and n not in got:
        got[n] = (r.get('status'), r.get('conclusion'))
missing = need - set(got)
if missing:
    print('MISSING:' + ','.join(sorted(missing)))
elif any(s != 'completed' for s, c in got.values()):
    print('PENDING:' + ' '.join(f'{k}={s}' for k, (s, c) in sorted(got.items())))
elif any(c != 'success' for s, c in got.values()):
    print('RED:' + ' '.join(f'{k}={c}' for k, (s, c) in sorted(got.items())))
else:
    print('GREEN')
" 2>/dev/null || echo "CHECK_FAILED")"
    if [ "${CI_VERDICT}" != "GREEN" ]; then
        echo "ERROR: CI is not verified green for origin/main (${CI_VERDICT})."
        echo "  A red/pending/unreachable CI must never deploy silently (P233)."
        echo "  Wait for CI, fix the red, or in a genuine emergency:"
        echo "    HMATS_DEPLOY_SKIP_CI_CHECK=1 bash scripts/hetzner_deploy.sh ${SERVER}"
        exit 1
    fi
    echo "  CI: GREEN (codebase-invariants + test-suite) for ${DEPLOY_SHA:0:9}"
fi

# [P328] Scan the COMMIT that will deploy, not the checkout it is launched
# from. A worktree gives the scanners a real .git, which they require: the
# authority audit refuses to run without a git-grep engine rather than emit
# false findings (P158), so a bare file copy is not a substitute.
SCAN_TREE=""
SCAN_TMP=""
if git fetch -q origin main 2>/dev/null && [ -n "${DEPLOY_SHA}" ]; then
    SCAN_TMP="$(mktemp -d 2>/dev/null || true)"
    if [ -n "${SCAN_TMP}" ] && git worktree add --detach -q "${SCAN_TMP}" "${DEPLOY_SHA}" 2>/dev/null; then
        SCAN_TREE="${SCAN_TMP}"
    fi
fi

if [ -n "${SCAN_TREE}" ]; then
    echo "  Running local env-independent scanners (--skip-mypy) on ${DEPLOY_SHA:0:9}..."
    ( cd "${SCAN_TREE}" && "${PY_BIN}" -X utf8 tools/ci_check_invariants.py --skip-mypy )
    SCAN_RC=$?
    git worktree remove --force "${SCAN_TREE}" >/dev/null 2>&1 || true
else
    echo "  WARNING: could not build a worktree at the deployed sha."
    echo "  Falling back to the WORKING TREE, which in a shared checkout"
    echo "  measures uncommitted edits that will NOT deploy (P311b/P328)."
    "${PY_BIN}" -X utf8 tools/ci_check_invariants.py --skip-mypy
    SCAN_RC=$?
fi

if [ ${SCAN_RC} -ne 0 ]; then
    echo "ERROR: Local scanner gate FAILED (types are adjudicated by the"
    echo "  CI check above; this failure is a stdlib scanner regression)."
    echo "  Fix the findings, or rebaseline if intentional."
    echo "Refusing to deploy."
    exit 1
fi
echo "  Local scanners: PASS"

# --- Step 1: Pull latest code ---
echo "[1/5] Pulling latest code..."
ssh "${SERVER}" "su - ${REMOTE_USER} -c 'cd ${APP_DIR} && git pull origin main'" 2>&1

# --- Step 2: Verify .env exists ---
echo "[2/5] Checking .env..."
ssh "${SERVER}" "test -f ${APP_DIR}/.env || { echo 'ERROR: .env not found. Copy env/.env.template to .env and fill in keys.'; exit 1; }"

# --- Step 3: Build images ---
echo "[3/5] Building Docker images..."
ssh "${SERVER}" "su - ${REMOTE_USER} -c 'cd ${APP_DIR} && docker compose -f ${COMPOSE_FILE} build'" 2>&1

# --- Step 4: Copy models into engine volume ---
echo "[4/5] Syncing models to Docker volume..."
ssh "${SERVER}" "su - ${REMOTE_USER} -c '
    # Create temp container to copy models into volume
    docker volume create hmats-models 2>/dev/null || true
    docker run --rm -v hmats-models:/models -v /home/${REMOTE_USER}/hmats/models:/src:ro alpine sh -c \"cp -r /src/* /models/ 2>/dev/null || true\"
    echo \"Models synced\"
'" 2>&1

# --- Step 5: Bring up services ---
echo "[5/5] Starting services..."
ssh "${SERVER}" "su - ${REMOTE_USER} -c 'cd ${APP_DIR} && docker compose -f ${COMPOSE_FILE} up -d'" 2>&1

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Verify:"
echo "  ssh ${SERVER} 'docker ps'"
echo "  ssh ${SERVER} 'docker logs hmats-engine --tail 20'"
echo "  ssh ${SERVER} 'curl -s localhost:8080/health | python3 -m json.tool'"
echo ""
echo "Validate:"
echo "  bash scripts/hetzner_validate.sh ${SERVER}"
