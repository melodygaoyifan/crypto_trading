#!/bin/bash
# ============================================================================
# Operator-run (via ! prefix): set which assets the Coinbase DERIVATIVES sleeve
# trades, then restart the engine. Enables autonomous Coinbase perp trading for
# the given assets — bounded by the sleeve guard (per-asset contract cap, 15%
# sleeve-drawdown halt, venue-resting protective stops) with exit-management
# active (flattens on hold).
#
#   bash scripts/coinbase_set_assets.sh BTC,ETH,SOL   # the home trio
#   bash scripts/coinbase_set_assets.sh SOL           # SOL only
#   bash scripts/coinbase_set_assets.sh ""            # OFF -> inert (revert)
#
# [P291] BREADTH assets (XRP,ADA,LTC,DOGE,BNB — the P262-certified transfer
# set) are ACCEPTED by this script but require TWO other things before the
# sleeve can act on them: a SYMBOL_MAP perp entry and per-asset caps/fractions
# in the live config. Without those the routing state is written and the asset
# stays inert. Their forward read is ~2026-09-15; widening before it is a
# decision that needs its own P-entry.
#
# [P291 correction] The old header said "The Kraken spot sleeve trades all 3
# regardless" — falsified by P152 (2026-06-14): Kraken skips every NEW entry
# for a Coinbase-routed asset that is flat, and all three were flattened
# 2026-06-12, so Kraken directional trading has been structurally zero since.
# The Coinbase sleeve is the SOLE directional driver for routed assets.
#
# Reversible at any time.
# ============================================================================
set -uo pipefail
ASSETS_CSV="${1:-}"

# ---- validation -------------------------------------------------------------
# [P291] Before this, ANY string was accepted and written into the routing
# state verbatim: `coinbase_set_assets.sh BTS` (a typo for BTC) wrote
# coinbase_assets:["BTS"], which routes nothing, silently leaves the intended
# asset on the dead Kraken path, and looks like a successful run. An unknown
# asset is now a refusal, not a no-op.
HOME_ASSETS="BTC ETH SOL"
BREADTH_ASSETS="XRP ADA LTC DOGE BNB"

VALIDATED=$(python3 -c "
import sys
csv = sys.argv[1].strip()
home = set(sys.argv[2].split())
breadth = set(sys.argv[3].split())
a = [x.strip().upper() for x in csv.split(',') if x.strip()]
unknown = [x for x in a if x not in home | breadth]
if unknown:
    print('ERR:' + ','.join(unknown)); sys.exit(0)
dupes = [x for i, x in enumerate(a) if x in a[:i]]
if dupes:
    print('ERR_DUP:' + ','.join(sorted(set(dupes)))); sys.exit(0)
print('OK:' + ','.join(a) + '|' + ','.join(x for x in a if x in breadth))
" "$ASSETS_CSV" "$HOME_ASSETS" "$BREADTH_ASSETS")

case "$VALIDATED" in
  ERR:*)
    echo "REFUSED: unknown asset(s): ${VALIDATED#ERR:}" >&2
    echo "  known home assets:    $HOME_ASSETS" >&2
    echo "  known breadth assets: $BREADTH_ASSETS  (see the P291 note above)" >&2
    echo "  nothing was written; the routing state is unchanged." >&2
    exit 2 ;;
  ERR_DUP:*)
    echo "REFUSED: duplicate asset(s): ${VALIDATED#ERR_DUP:}" >&2
    exit 2 ;;
esac

REQUESTED="${VALIDATED#OK:}"
REQUESTED_ASSETS="${REQUESTED%%|*}"
REQUESTED_BREADTH="${REQUESTED##*|}"

# ---- breadth confirmation ---------------------------------------------------
if [ -n "$REQUESTED_BREADTH" ]; then
  cat >&2 <<BANNER

  ============================================================================
  BREADTH WIDENING REQUESTED: $REQUESTED_BREADTH
  ============================================================================
  These are NOT the home trio. Before confirming, know:

   * THIN BOOKS. Live probe 2026-08-17 (24h contract volume): XRP 9,361 |
     ADA 5,997 | DOGE 1,651 | LTC 843 | BNB 322 — against BTC 193,404 and
     ETH 132,772. XRP is comparable to SOL (12,658); BNB and LTC are two
     orders of magnitude thinner than the majors and a few contracts is a
     visible share of their daily flow.
   * EVIDENCE DATE. The breadth forward books have been accruing since
     P271; their P166 read is ~2026-09-15. Widening before that read is
     trading on transfer evidence (P262: trend/hold beat flat 5/5 on these
     never-fitted assets) rather than on this venue's forward evidence.
   * ONE ASSET FIRST (P197). Widen to a single breadth asset, watch a full
     cycle of [COINBASE-MANAGE] / [COINBASE-STOP] / reconcile, and only then
     consider a second.
   * PREREQUISITES. The sleeve can only act on an asset that ALSO has a
     SYMBOL_MAP perp entry and per-asset caps/fractions in the live config.
     Without them this writes routing state and the asset stays inert —
     which is safe, but is not the widening you intended.

BANNER
  printf '  Type the breadth asset list back to confirm (%s): ' "$REQUESTED_BREADTH" >&2
  read -r CONFIRM
  if [ "$CONFIRM" != "$REQUESTED_BREADTH" ]; then
    echo "  ABORTED: confirmation did not match; routing state unchanged." >&2
    exit 3
  fi
fi

# ---- dry run stop ------------------------------------------------------------
# [P291] HMATS_SET_ASSETS_DRY_RUN=1 stops here: validation + the breadth
# banner run, nothing is written and the engine is NOT restarted. Two uses:
# an operator previewing a widening, and the test suite — which MUST set it.
# (Written after a P291 test invoked this script for the home trio and
# restarted the live engine: the network half has no business executing
# from a test, and "the routing state was identical anyway" is luck, not a
# safety property. P186 class: a test that reaches production.)


if [ "${HMATS_SET_ASSETS_DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN: validated '$REQUESTED_ASSETS' (breadth: '${REQUESTED_BREADTH:-none}')."
  echo "DRY RUN: nothing written, engine NOT restarted."
  exit 0
fi

# ---- write routing state ----------------------------------------------------
python3 -c "
import json, sys
csv = sys.argv[1].strip()
home = sys.argv[2].split()
a = [x.strip().upper() for x in csv.split(',') if x.strip()]
print(json.dumps({
    'phase': 'DUAL_VENUE' if a else 'PRE_PHASE_2',
    'coinbase_assets': a,
    # kraken_assets stays the HOME trio only: breadth assets were never
    # Kraken-routed, so listing them here would assert a venue fallback
    # that does not exist.
    'kraken_assets': [k for k in home if k not in a],
}))
" "$REQUESTED_ASSETS" "$HOME_ASSETS" > /tmp/cb_routing.json
echo "ROUTING -> $(cat /tmp/cb_routing.json)"

scp -o BatchMode=yes /tmp/cb_routing.json \
  hmats:/var/lib/docker/volumes/hmats-data/_data/coinbase_routing_state.json
rm -f /tmp/cb_routing.json

CHECK_ASSETS="$HOME_ASSETS $REQUESTED_BREADTH"
ssh -o BatchMode=yes hmats "cd /home/hmats/hmats/app && \
  docker compose -f docker-compose.hetzner.yml up -d --force-recreate hmats-engine >/dev/null 2>&1; \
  sleep 16; \
  echo \"STATUS: \$(docker ps --filter name=hmats-engine --format '{{.Status}}')\"; \
  docker exec -i hmats-engine python3 -c \"
import core.execution_service as es
from types import SimpleNamespace
ctx = SimpleNamespace(config=SimpleNamespace(coinbase_routing_enabled=True))
es._coinbase_get_routing()
print('ROUTED:', {a: es._coinbase_routed(ctx, a) for a in '$CHECK_ASSETS'.split()})
\" 2>/dev/null | grep ROUTED"
echo "done. Watch [COINBASE-MANAGE] in the heartbeat; revert with: bash scripts/coinbase_set_assets.sh \"\""
