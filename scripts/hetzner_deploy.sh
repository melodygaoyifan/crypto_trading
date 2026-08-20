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
    # [P344] ONE implementation of this check now lives in
    # tools/ci_status.py. It was CORRECT here and unreachable from anywhere
    # else, so every ad-hoc "did CI pass?" got hand-rolled again and worse --
    # including one that hardcoded a repo slug that does not exist and then
    # printed "no runs yet" twenty times at it, burning the API budget on a
    # question GitHub had already refused (the P159/P199 conflation inside a
    # retry loop). The tool derives the slug from the remote, keeps the
    # NEWEST run per workflow (P287), and gives UNREADABLE its own exit code.
    CI_TOOL="$(dirname "$0")/../tools/ci_status.py"
    if [ ! -f "${CI_TOOL}" ]; then
        echo "ERROR: tools/ci_status.py is missing — the CI gate cannot run."
        echo "  A gate that cannot run must REFUSE, never skip (P159/P187)."
        exit 1
    fi
    echo "  Checking CI conclusions for ${DEPLOY_SHA:0:9}..."
    set +e
    CI_OUT="$("${PY_BIN}" -X utf8 "${CI_TOOL}" --sha "${DEPLOY_SHA}" 2>&1)"
    CI_RC=$?
    set -e
    echo "  ${CI_OUT}"
    if [ "${CI_RC}" -ne 0 ]; then
        echo "ERROR: CI is not verified green for origin/main (rc=${CI_RC})."
        echo "  A red/pending/unreachable CI must never deploy silently (P233)."
        echo "  rc: 1=red 2=unreadable 3=pending 4=no-run-yet"
        echo "  Wait for CI, fix the red, or in a genuine emergency:"
        echo "    HMATS_DEPLOY_SKIP_CI_CHECK=1 bash scripts/hetzner_deploy.sh ${SERVER}"
        exit 1
    fi
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
        # [P328] Remove it even if this script is INTERRUPTED. Observed: a
        # deploy whose stdout was piped through `head` took SIGPIPE partway
        # through and left a full checkout behind, which then shows up in
        # `git worktree list` forever. The explicit remove below is the happy
        # path; this is the backstop.
        trap 'git worktree remove --force "${SCAN_TREE}" >/dev/null 2>&1 || true; git worktree prune >/dev/null 2>&1 || true' EXIT INT TERM HUP PIPE
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
