#!/bin/bash
# HMATS v5.1 — Post-Deploy Verification
# =======================================
#
# Runs after `bash scripts/hetzner_deploy.sh hmats` to verify v5.1 advisory
# tooling is wired and writing JSONL ledgers. Designed to be safe and
# read-only — never mutates engine state.
#
# Usage: bash scripts/v51_post_deploy_verify.sh [SERVER]
# Default SERVER=hmats
#
# Exit codes:
#   0 = all v5.1 advisory layers verified writing
#   1 = one or more layers not yet writing (may need a tick or two)
#   2 = engine container not running

set -uo pipefail

SERVER="${1:-hmats}"
TODAY="$(date -u +%Y%m%d)"

echo "=== HMATS v5.1 Post-Deploy Verification ==="
echo "  server: ${SERVER}"
echo "  date:   ${TODAY} UTC"
echo ""

# Step 1: container running?
if ! ssh "${SERVER}" "docker ps --format '{{.Names}}' | grep -q '^hmats-engine$'" 2>/dev/null; then
    echo "✗ hmats-engine container not running"
    exit 2
fi
echo "✓ hmats-engine container is running"

# Step 2: startup log shows all 3 v5.1 advisory harnesses ACTIVE?
echo ""
echo "--- v5.1 startup signals (last 10 min of logs) ---"
ssh "${SERVER}" "docker logs hmats-engine --since 10m 2>&1 | grep -E 'v5.1 PHASE[478]'" || \
    echo "  (no v5.1 PHASE startup markers found yet — may need restart)"

# Step 3: count startup banners — expect at least 3 (PHASE4 + PHASE7 + PHASE8)
echo ""
n_active=$(ssh "${SERVER}" "docker logs hmats-engine --since 10m 2>&1 | grep -cE 'v5\.1 PHASE[478].*ACTIVE'" 2>/dev/null || echo 0)
n_active=${n_active//[^0-9]/}
n_active=${n_active:-0}
if [ "${n_active}" -ge 3 ]; then
    echo "✓ ${n_active} v5.1 advisory layers reported ACTIVE in startup banner"
else
    echo "⚠ only ${n_active}/3 v5.1 advisory layers ACTIVE in recent logs (may need a tick)"
fi

# Step 4: ledger files present?
echo ""
echo "--- v5.1 advisory ledger files ---"
fail=0
for prefix in microstructure cascade sleeve_allocations; do
    path="/opt/hmats/data/strategy_shadow/${prefix}_${TODAY}.jsonl"
    n_lines=$(ssh "${SERVER}" "docker exec hmats-engine wc -l < ${path} 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
    n_lines=${n_lines:-0}
    if [ "${n_lines}" -gt 0 ]; then
        echo "  ✓ ${prefix}_${TODAY}.jsonl  (${n_lines} records)"
    else
        echo "  ✗ ${prefix}_${TODAY}.jsonl  (missing or empty — may need 4H tick to elapse)"
        fail=1
    fi
done

# Step 5: most-recent record sample (per prefix)
echo ""
echo "--- last record per ledger ---"
for prefix in microstructure cascade sleeve_allocations; do
    path="/opt/hmats/data/strategy_shadow/${prefix}_${TODAY}.jsonl"
    last=$(ssh "${SERVER}" "docker exec hmats-engine tail -1 ${path} 2>/dev/null | head -c 200" 2>/dev/null)
    if [ -n "${last}" ]; then
        echo "  ${prefix}: ${last}..."
    fi
done

# Step 6: drl_promotion_state.json sanity check (Iron Law 5 review-time)
echo ""
echo "--- DRL Authority (Iron Law 5) ---"
drl_level=$(ssh "${SERVER}" "docker exec hmats-engine cat /opt/hmats/data/drl_promotion_state.json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"authority_level\",\"unknown\"))'" 2>/dev/null)
drl_level=${drl_level:-unknown}
if [ "${drl_level}" = "ACTIVE" ]; then
    echo "  ✓ drl_promotion_state.authority_level = ACTIVE"
else
    echo "  ✗ drl_promotion_state.authority_level = ${drl_level}  (Iron Law 5 floor expects ACTIVE)"
    fail=1
fi

echo ""
if [ "${fail}" -eq 0 ]; then
    echo "=== v5.1 post-deploy verification: PASS ==="
    exit 0
else
    echo "=== v5.1 post-deploy verification: PARTIAL (some ledgers missing or DRL not ACTIVE) ==="
    echo "    if just-deployed, wait one 4H tick and re-run."
    exit 1
fi
