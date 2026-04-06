#!/bin/bash
# =============================================================================
# HMATS v6.8.0 — Hetzner Post-Deploy Validation
# =============================================================================
# Run from LOCAL machine after deploy:
#   bash scripts/hetzner_validate.sh <server-alias>
# =============================================================================
set -uo pipefail

SERVER="${1:-hmats}"
PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" = "PASS" ]; then
        echo "  [PASS] ${label}"
        PASS=$((PASS + 1))
    elif [ "$result" = "WARN" ]; then
        echo "  [WARN] ${label}"
        WARN=$((WARN + 1))
    else
        echo "  [FAIL] ${label}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== HMATS Post-Deploy Validation ==="
echo ""

# --- 1. Containers running ---
echo "Containers:"
ENGINE_UP=$(ssh "${SERVER}" "docker ps --filter name=hmats-engine --format '{{.Status}}'" 2>/dev/null)
API_UP=$(ssh "${SERVER}" "docker ps --filter name=hmats-api --format '{{.Status}}'" 2>/dev/null)

if echo "$ENGINE_UP" | grep -q "Up"; then
    check "hmats-engine running" "PASS"
else
    check "hmats-engine running (${ENGINE_UP:-NOT FOUND})" "FAIL"
fi

if echo "$API_UP" | grep -q "Up"; then
    check "hmats-api running" "PASS"
else
    check "hmats-api running (${API_UP:-NOT FOUND})" "FAIL"
fi

# --- 2. API health ---
echo ""
echo "API Health:"
HEALTH=$(ssh "${SERVER}" "curl -s localhost:8080/health 2>/dev/null" || echo "{}")
HEALTH_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")

if [ "$HEALTH_STATUS" = "healthy" ]; then
    check "/health returns healthy" "PASS"
elif [ "$HEALTH_STATUS" = "degraded" ]; then
    check "/health returns degraded (engine may still be warming up)" "WARN"
else
    check "/health endpoint (got: ${HEALTH_STATUS:-no response})" "FAIL"
fi

# --- 3. Engine logs ---
echo ""
echo "Engine Logs:"
RECENT_LOGS=$(ssh "${SERVER}" "docker logs hmats-engine --tail 10 2>&1" || echo "")

if echo "$RECENT_LOGS" | grep -q "KRAKEN_FUTURES\|REGIME\|PROOF\|SENTIMENT"; then
    check "Engine producing tick logs" "PASS"
else
    check "Engine tick logs (no tick output found yet)" "WARN"
fi

if echo "$RECENT_LOGS" | grep -qi "error\|traceback\|exception"; then
    check "No errors in recent logs" "FAIL"
else
    check "No errors in recent logs" "PASS"
fi

# --- 4. Data files ---
echo ""
echo "Data Files:"
DASHBOARD_EXISTS=$(ssh "${SERVER}" "docker exec hmats-engine test -f /opt/hmats/data/dashboard_state.json && echo yes || echo no" 2>/dev/null)
POSITIONS_EXISTS=$(ssh "${SERVER}" "docker exec hmats-engine test -f /opt/hmats/data/paper_positions.json && echo yes || echo no" 2>/dev/null)

if [ "$DASHBOARD_EXISTS" = "yes" ]; then
    check "dashboard_state.json exists" "PASS"
else
    check "dashboard_state.json exists (may need first tick)" "WARN"
fi

if [ "$POSITIONS_EXISTS" = "yes" ]; then
    check "paper_positions.json exists" "PASS"
else
    check "paper_positions.json exists (may need first trade)" "WARN"
fi

# --- 5. Resource usage ---
echo ""
echo "Resources:"
MEM_USAGE=$(ssh "${SERVER}" "docker stats hmats-engine --no-stream --format '{{.MemPerc}}'" 2>/dev/null | tr -d '%')
if [ -n "$MEM_USAGE" ]; then
    MEM_INT=${MEM_USAGE%.*}
    if [ "$MEM_INT" -lt 70 ]; then
        check "Engine memory usage: ${MEM_USAGE}%" "PASS"
    elif [ "$MEM_INT" -lt 85 ]; then
        check "Engine memory usage: ${MEM_USAGE}% (elevated)" "WARN"
    else
        check "Engine memory usage: ${MEM_USAGE}% (critical)" "FAIL"
    fi
fi

DISK=$(ssh "${SERVER}" "df / --output=pcent | tail -1 | tr -d '% '" 2>/dev/null)
if [ -n "$DISK" ] && [ "$DISK" -lt 80 ]; then
    check "Disk usage: ${DISK}%" "PASS"
else
    check "Disk usage: ${DISK:-unknown}%" "WARN"
fi

SWAP=$(ssh "${SERVER}" "swapon --show --bytes 2>/dev/null | tail -1 | awk '{print \$3}'" 2>/dev/null)
if [ -n "$SWAP" ] && [ "$SWAP" -gt 0 ]; then
    check "Swap configured" "PASS"
else
    check "Swap not configured (risk of OOM on CPX21)" "WARN"
fi

# --- Summary ---
echo ""
echo "================================"
echo "Results: ${PASS} PASS, ${WARN} WARN, ${FAIL} FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "STATUS: VALIDATION FAILED"
    exit 1
elif [ "$WARN" -gt 2 ]; then
    echo "STATUS: NEEDS ATTENTION"
    exit 0
else
    echo "STATUS: VALIDATION PASSED"
    exit 0
fi
