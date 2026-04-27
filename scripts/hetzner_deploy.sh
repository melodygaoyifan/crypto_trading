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

# --- Step 0: Local pre-deploy CI gate ---
# [P111 Tier1#2 2026-04-27] Run scanner baselines BEFORE pushing the
# deploy. CI gate (codebase-invariants.yml) catches regressions on
# push, but local pre-deploy catches them BEFORE 30s of container
# churn + the operator round-trip of "push → deploy → see CI fail →
# revert → push → deploy". Same scanners, different timing.
if command -v python &>/dev/null; then
    echo "[0/5] Running local CI gate (scanner baselines)..."
    if ! python -X utf8 tools/ci_check_invariants.py; then
        echo "ERROR: Local CI gate FAILED. Either:"
        echo "  - rebaseline if intentional: python -X utf8 tools/ci_check_invariants.py --update"
        echo "  - or fix the new findings before deploying."
        echo "Refusing to deploy. Re-run after rebaseline OR fix."
        exit 1
    fi
    echo "  CI gate: PASS"
fi

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
