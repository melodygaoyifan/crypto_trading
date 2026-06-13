#!/bin/bash
# ============================================================================
# v5.1 Phase 10 — automated shadow -> promotion review (weekly host cron).
# Runs the shadow-IC + promotion-gate pipeline INSIDE the engine container on
# the live shadow ledgers, saves the dated plan, and logs a one-line summary.
# While shadow data is < ~30d the gate returns EXTEND_SHADOW for everything
# (correct: never promote/kill on insufficient samples). Once data matures it
# emits real PROMOTE / KILL / HOLD decisions. Read-only analysis — it does NOT
# change live trading; promotion is applied separately + deliberately via
# analytics/promotion_gate/apply_promotion_plan.py after operator review.
# ============================================================================
set -uo pipefail
DAY=$(date -u +%Y%m%d)
DATA=/opt/hmats/data/promotion_gate
IC=$DATA/shadow_ic_${DAY}.json
PLAN=$DATA/plan_${DAY}.json

docker exec hmats-engine sh -c "mkdir -p $DATA" 2>/dev/null

docker exec hmats-engine python3 -X utf8 analytics/shadow_ic/compute_shadow_ic.py \
  --ledger-dir /opt/hmats/data/strategy_shadow \
  --prefixes "microstructure,cascade,funding,ml_factor" \
  --window-days 30 --output "$IC" >/dev/null 2>&1

docker exec hmats-engine python3 -X utf8 analytics/promotion_gate/promotion_plan.py \
  --shadow-ic-report "$IC" --output "$PLAN" >/dev/null 2>&1

# one-line summary into the engine log stream (picked up by Discord ERROR/4H tail)
SUMMARY=$(docker exec hmats-engine python3 -X utf8 -c "
import json
try:
    d = json.load(open('$PLAN'))
    sm = d.get('summary', d)
    promo = sm.get('n_strategy_promote', 0); kill = sm.get('n_strategy_kill', 0)
    ext = sm.get('n_strategy_extend', 0)
    print(f'[PROMOTION_REVIEW $DAY] promote={promo} kill={kill} extend={ext} plan=$PLAN')
except Exception as e:
    print(f'[PROMOTION_REVIEW $DAY] no plan ({e})')
" 2>/dev/null)
echo "$SUMMARY"
# escalate if anything is actually promotable/killable so the operator looks
echo "$SUMMARY" | grep -qE "promote=[1-9]|kill=[1-9]" && \
  docker exec hmats-engine python3 -c "import logging;logging.getLogger('HMATS.v510').error('$SUMMARY — operator review: apply_promotion_plan.py')" 2>/dev/null
exit 0
