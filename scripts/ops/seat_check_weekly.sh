#!/bin/sh
# [P295d] Weekly seat-controller read. Runs AFTER agent_ic_review (06:10) so it
# scores that morning's report. Resolves the newest report at run time — the
# filename is timestamped. A missing report exits 2 (refusal): "no report" must
# never read as "flat wins" (P199), which is exactly the failure P295d fixed
# inside the controller itself.
#
# [P423] Step 6/6 added: the regimebook's OWN pooled ledger read, in-container.
# The live quant series was flat all window (P421e) so the agent_ic report
# could not score the incumbent; compute_shadow_ic scores the book's claim
# series instead — priced on Kraken public 4H OHLC (--allow-kraken-ohlcv,
# ~120d) because the parquets are not in the image (P213). Both verdicts are
# printed; NEITHER is applied. Tracked copy: scripts/ops/seat_check_weekly.sh
# (install by file, P235: scp to /home/hmats/seat_check_weekly.sh).
D=/opt/hmats/data/evidence_reports
R=$(docker exec hmats-engine sh -c "ls -1t $D/agent_ic_*.json 2>/dev/null | head -1")
if [ -z "$R" ]; then
  echo "[seat_check] REFUSING: no agent_ic report in $D — not a flat verdict."
  exit 2
fi
echo "[seat_check] scoring $R"
docker exec hmats-engine python -X utf8 scripts/seat_check.py --ic-report "$R"
echo "[seat_check] exit=$? (0=hold 3=SWITCH-recommended 2=refusal; NEVER auto-applied)"

# [P423] step 6/6 — the book's own ledger, pooled, 1/4-bar horizons
S=$D/shadow_ic_regimebook_$(date -u +%Y%m%d).json
docker exec hmats-engine python -X utf8 analytics/shadow_ic/compute_shadow_ic.py \
  --ledger-dir /opt/hmats/data/strategy_shadow --window-days 90 \
  --horizons 1,4 --prefixes regimebook --pool-assets --allow-kraken-ohlcv \
  --output "$S"
SC=$?
if [ "$SC" -ne 0 ]; then
  echo "[seat_check] ledger read REFUSED/empty (exit=$SC) — the shadow-based seat verdict is NOT available this week; this is not 'flat wins'."
  exit 2
fi
docker exec hmats-engine python -X utf8 scripts/seat_check.py --shadow-ic-report "$S" \
  --ledger-dir /opt/hmats/data/strategy_shadow
echo "[seat_check] ledger-read exit=$? (0=hold 3=SWITCH-recommended 2=refusal; NEVER auto-applied — decide via analytics/decisions/pending_decisions.json)"
