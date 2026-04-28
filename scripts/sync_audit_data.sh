#!/usr/bin/env bash
# sync_audit_data.sh — pull HMATS production audit data to local
# ================================================================
# v3 Track A item 1.3 (P1-7). Unblocks P1-5 (IC cron) + P1-6
# (correlation matrix) + future audits.
#
# Pulls the 5 critical audit data sources from the production
# Hetzner host's Docker volumes to a date-stamped local directory:
#   data/audit_sync/YYYY-MM-DD/
#
# Files pulled:
#   - equity_history.jsonl       (drawdown + Sharpe analysis)
#   - kq_firing_stats.json       (current snapshot, 12-strat fire counts)
#   - kq_firing_stats.jsonl      (P128 append-only audit log)
#   - ic_signals/                (IC framework signal records)
#   - outcomes_*.jsonl           (per-trade attribution outcomes)
#   - proof_log_*.log            (decision proof chain — for veto audit)
#
# Validation: file size > 0 + last JSON line parseable.
#
# Usage (manual):
#   bash scripts/sync_audit_data.sh
#
# Usage (cron, local audit machine — NOT production):
#   0 2 * * * /path/to/scripts/sync_audit_data.sh >> /tmp/hmats_audit_sync.log 2>&1
#
# Kill criteria (per v3 prompt):
#   - local file mtime > 7 days → sync broken, fix
#   - > 10% lines unparseable JSON → cloud writer corrupting

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-hmats}"
REMOTE_DATA="/var/lib/docker/volumes/hmats-data/_data"
REMOTE_LOGS="/var/lib/docker/volumes/hmats-logs/_data"

SYNC_DATE="$(date +%Y-%m-%d)"
LOCAL_DIR="data/audit_sync/${SYNC_DATE}"

mkdir -p "${LOCAL_DIR}/ic_signals"
mkdir -p "${LOCAL_DIR}/attribution"
mkdir -p "${LOCAL_DIR}/proof_log"

echo "=== HMATS audit data sync — ${SYNC_DATE} ==="
echo "Remote: ${REMOTE_HOST}"
echo "Local:  ${LOCAL_DIR}"
echo

# 1. equity_history.jsonl — drawdown + Sharpe
echo "[1/5] equity_history.jsonl..."
scp -q "${REMOTE_HOST}:${REMOTE_DATA}/equity_history.jsonl" "${LOCAL_DIR}/" \
    || { echo "FAIL: equity_history.jsonl scp failed"; exit 1; }

# 2. kq_firing_stats.{json,jsonl} — kraken_quant 12-strategy stats
echo "[2/5] kq_firing_stats..."
scp -q "${REMOTE_HOST}:${REMOTE_DATA}/kq_firing_stats.json" "${LOCAL_DIR}/" \
    || { echo "FAIL: kq_firing_stats.json scp failed"; exit 1; }
# .jsonl may not exist yet (P128 just deployed today) — non-fatal
scp -q "${REMOTE_HOST}:${REMOTE_DATA}/kq_firing_stats.jsonl" "${LOCAL_DIR}/" \
    || echo "WARN: kq_firing_stats.jsonl missing (expected if pre-P128 capture window)"

# 3. ic_signals/ — IC framework signal records (recursive)
echo "[3/5] ic_signals/..."
scp -q -r "${REMOTE_HOST}:${REMOTE_LOGS}/ic_signals/" "${LOCAL_DIR}/" \
    || { echo "FAIL: ic_signals scp failed"; exit 1; }

# 4. outcomes_*.jsonl — per-trade outcomes
echo "[4/5] attribution/outcomes_*.jsonl..."
# scp doesn't support globbing on the remote side without wrapping in sh -c
ssh "${REMOTE_HOST}" "ls ${REMOTE_LOGS}/attribution/outcomes_*.jsonl 2>/dev/null" | while read -r f; do
    bn=$(basename "$f")
    scp -q "${REMOTE_HOST}:$f" "${LOCAL_DIR}/attribution/${bn}" \
        || echo "WARN: ${bn} scp failed"
done

# 5. proof_log_*.log — decision proof chain (last 10 only — these are huge)
echo "[5/5] proof_log_*.log (last 10)..."
ssh "${REMOTE_HOST}" "ls -t ${REMOTE_LOGS}/proof_log_*.log 2>/dev/null | head -10" | while read -r f; do
    bn=$(basename "$f")
    scp -q "${REMOTE_HOST}:$f" "${LOCAL_DIR}/proof_log/${bn}" \
        || echo "WARN: ${bn} scp failed"
done

# ============================================================
# Validation
# ============================================================
echo
echo "=== Validation ==="

REQUIRED_FILES=(
    "${LOCAL_DIR}/equity_history.jsonl"
    "${LOCAL_DIR}/kq_firing_stats.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        echo "FAIL: $f is empty or missing"
        exit 1
    fi
    echo "OK: $f ($(wc -l < "$f") lines, $(stat -c %s "$f" 2>/dev/null || stat -f %z "$f") bytes)"
done

# JSON parseability check (last line of each .jsonl)
for f in "${LOCAL_DIR}/equity_history.jsonl" "${LOCAL_DIR}/kq_firing_stats.jsonl"; do
    if [ -s "$f" ]; then
        if ! tail -1 "$f" | python -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
            echo "FAIL: $f tail line not parseable as JSON"
            exit 1
        fi
        echo "OK: $f tail JSON parseable"
    fi
done

# Snapshot file is a single JSON object, not JSONL
if ! python -c "import json; json.load(open('${LOCAL_DIR}/kq_firing_stats.json'))" 2>/dev/null; then
    echo "FAIL: kq_firing_stats.json not parseable"
    exit 1
fi
echo "OK: kq_firing_stats.json parseable"

# IC signals coverage
IC_COUNT=$(ls "${LOCAL_DIR}/ic_signals/"*.jsonl 2>/dev/null | wc -l)
echo "OK: ${IC_COUNT} IC signal files synced"

# Outcomes coverage
OUT_COUNT=$(ls "${LOCAL_DIR}/attribution/"*.jsonl 2>/dev/null | wc -l)
echo "OK: ${OUT_COUNT} outcome files synced"

# Proof log coverage
PL_COUNT=$(ls "${LOCAL_DIR}/proof_log/"*.log 2>/dev/null | wc -l)
echo "OK: ${PL_COUNT} proof log files synced"

echo
echo "=== Sync complete: ${LOCAL_DIR} ==="
