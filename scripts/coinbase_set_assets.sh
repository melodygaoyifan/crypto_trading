#!/bin/bash
# ============================================================================
# Operator-run (via ! prefix): set which assets the Coinbase DERIVATIVES sleeve
# trades, then restart the engine. Enables autonomous Coinbase perp trading for
# the given assets — bounded by the sleeve guard (1-contract cap/asset, 15%
# sleeve-drawdown halt) with exit-management active (flattens on hold).
#
#   bash scripts/coinbase_set_assets.sh BTC,ETH,SOL   # all 3 (full coverage)
#   bash scripts/coinbase_set_assets.sh SOL           # SOL only
#   bash scripts/coinbase_set_assets.sh ""            # OFF -> inert (revert)
#
# Reversible at any time. The Kraken spot sleeve trades all 3 regardless.
# ============================================================================
set -uo pipefail
ASSETS_CSV="${1:-}"

python3 -c "
import json, sys
csv = sys.argv[1].strip()
a = [x.strip().upper() for x in csv.split(',') if x.strip()]
print(json.dumps({
    'phase': 'DUAL_VENUE' if a else 'PRE_PHASE_2',
    'coinbase_assets': a,
    'kraken_assets': [k for k in ('BTC','ETH','SOL') if k not in a],
}))
" "$ASSETS_CSV" > /tmp/cb_routing.json
echo "ROUTING -> $(cat /tmp/cb_routing.json)"

scp -o BatchMode=yes /tmp/cb_routing.json \
  hmats:/var/lib/docker/volumes/hmats-data/_data/coinbase_routing_state.json
rm -f /tmp/cb_routing.json

ssh -o BatchMode=yes hmats 'cd /home/hmats/hmats/app && \
  docker compose -f docker-compose.hetzner.yml up -d --force-recreate hmats-engine >/dev/null 2>&1; \
  sleep 16; \
  echo "STATUS: $(docker ps --filter name=hmats-engine --format "{{.Status}}")"; \
  docker exec -i hmats-engine python3 -c "import core.execution_service as es; from types import SimpleNamespace; ctx=SimpleNamespace(config=SimpleNamespace(coinbase_routing_enabled=True)); rp=es._coinbase_get_routing(); print(\"ROUTED:\", {a: es._coinbase_routed(ctx,a) for a in (\"BTC\",\"ETH\",\"SOL\")})" 2>/dev/null | grep ROUTED'
echo "done. Watch [COINBASE-MANAGE] in the heartbeat; revert with: bash scripts/coinbase_set_assets.sh \"\""
