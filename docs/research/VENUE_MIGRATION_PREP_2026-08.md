# Venue Migration Prep — percentage-fee perp venue (2026-08)

**Status:** DECISION-READY, nothing built, nothing live. Follows the proven
Coinbase-migration playbook (`docs/COINBASE_MIGRATION_PREP.md`, `exchange/cutover.py`):
adapter -> read-only probe -> shadow parity -> phased cutover, every LIVE step
operator-executed (P141). Companion to `GROWTH_PROGRAM_2026-08.md` (Gate 0 = YES).

**NOT LEGAL ADVICE.** The operator asserted a percentage-fee perp venue is legally
accessible to them. Per-venue eligibility — especially the US-access reality below —
is the operator's determination. This document does not advise circumventing any
geoblock/geofence; it compares venues operationally and flags the access column as
the pivotal per-venue question.

**Verification honesty:** the fee/spec figures below are web-research-sourced
(2026), NOT independently probed. Several items were unverifiable from the
research host (live spreads/depth on ALL venues; Binance/Bybit contract filters,
geoblocked from the host; dYdX exact maker rate). Treat as inputs to verify from
the operator's own access before committing — the P289 discipline (probe the venue
read-only for real spreads/filters) is Stage B.

---

## 1. The access reality (the pivotal column)

| | US-IP reachable? | custody |
|---|---|---|
| Binance / Bybit / OKX / Gate / KuCoin | **NO — geoblock US** (KuCoin under a 2026 CFTC US bar; OKX-US/Binance.US are spot-only) | custodial CEX |
| **Hyperliquid, dYdX v4** | permissionless PROTOCOL reachable; **front-ends geofence US** | self-custody DEX |

If the operator is US-based, the custodial CEXes are out regardless of fee
attractiveness, and the realistic set is the two DEX protocols — whose front-end
geofencing the operator must independently confirm is legal for them to access at
the protocol level. **This is the decision that gates everything else, and it is
not one I make or engineer around.**

## 2. Decision matrix (research-sourced; verify before committing)

| Axis | Hyperliquid | Binance USDs-M | Bybit | dYdX v4 |
|---|---|---|---|---|
| fee maker/taker % | **0.015 / 0.045** | 0.020 / 0.050 | 0.020 / 0.055 | **0.010** / 0.050 |
| flat/min per-order component | none (pure %) | none | none | none |
| min notional / granularity | ~$10, all pairs | ~$5-20; BTC step ~$96 | **5 USDT (finest)** | ~$5-18 (SOL coarse) |
| BTC/ETH/SOL perps | yes | yes | yes | yes |
| breadth (BNB/XRP/ADA/DOGE/LTC) | all 8 | all | all 8 | majors; rest unconf. |
| post-only | yes (ALO) | yes (GTX) | yes | yes |
| reduce-only (fixes CDE orphan-stop) | yes | yes | yes | **FOK/IOC only, not resting** |
| venue-resting stops | yes (oracle) | yes | yes | on-chain conditional |
| Python SDK | official | official | pybit (official) | signed-tx client |
| rate limit vs 4H cadence | ample | ample | ample | **2 stateful orders/block** — throttles P197/P270 |
| custody | **DEX self-custody** | custodial | custodial | DEX self-custody |
| withdraw-disabled trade key | **structural (agent wallet)** | config | config | order key != withdraw |
| liquidity majors | deep (rivals CEX) | **#1** | #2 | thinner; SOL thin |
| **US-IP reachable** | **protocol yes** | no | no | **protocol yes** |

## 3. What migration buys (net RT cost on a ~$640 order vs certified edge 24/88/222 bps/RT)

| venue | taker RT (fee+spread, modeled) | maker-ladder RT | BTC net (edge 24) |
|---|---|---|---|
| **Coinbase CDE (now)** | ~19-28 bps | **no benefit** (per-contract fee) | **~0 / negative -> BTC OUT** |
| Hyperliquid | ~11-13 bps | **~3 bps** | **+11 to +21 bps** |
| Binance | ~12-14 | ~4 | +10 to +20 |
| Bybit | ~13-15 | ~4 | +9 to +19 |
| dYdX v4 | ~12-14 | **~2** | +10 to +20 |

- **BTC flips from untradeable to net-positive** — the core justification.
- **ETH/SOL clear everywhere by 6-20x** — never the constraint.
- **The P270 maker ladder finally pays** (~3-4 bps RT) because the fee is now
  per-notional, not per-contract; P278's "measure the fill rate" conditional is
  moot since even full-taker clears BTC. (We already MEASURED f=0.82 maker on CDE,
  P375 — the ladder works; a % venue makes its savings real.)

## 4. Integration cost (HMATS side — bounded, architecture built for this)

A new venue = one adapter implementing the `exchange.adapter.ExchangeAdapter` ABC
(~12 methods: place/cancel/get_order, fetch_positions/balance/open_orders/
funding_rate/orderbook, to_venue_symbol, _contract_size). The sleeve engine
(`coinbase_sleeve.py`: maker ladder, resting stops, drawdown halt, reconcile)
reuses against the interface.
- **ccxt-supported venue (Binance/Bybit/OKX)** -> adapter ~140 lines (Kraken template).
- **Hyperliquid/dYdX (native SDK)** -> adapter ~300-500 lines (Coinbase template, 522).
- Routing is 2-venue hardcoded (kraken/coinbase); since Kraken is already flat and
  this is a MIGRATION, the clean path is a fresh sleeve+adapter that REPLACES the
  Coinbase sleeve as the directional driver — not generalizing routing to N venues.

## 5. Phased migration plan (Coinbase playbook; LIVE steps operator-executed)

- **Stage A — decide + credential [OPERATOR].** Confirm per-venue legal
  eligibility (section 1); pick custody model; create a **trade-only /
  withdraw-disabled** key (Hyperliquid's agent wallet is structurally this —
  better than the P272 CDE key concern). Fund a small test balance.
- **Stage B — adapter + read-only probe [ME builds, OPERATOR runs].** Implement the
  adapter; run a P289-style read-only probe from the operator's access: real L2
  spread/depth per asset, contract filters (min-notional/tick/step), funding
  cadence. This replaces every modeled number in sections 2/3 with measured ones.
- **Stage C — shadow parity [ME + OPERATOR].** New-venue sleeve runs read-only
  beside the live book; compare intended vs achievable fills, spread, funding for
  1-2 weeks. Validate the maker ladder's fill rate on the new venue (the P375
  ledger, new venue).
- **Stage D — phased cutover [OPERATOR].** Route ONE asset first (P197 rule:
  one-asset-first), watch, then widen. Coinbase sleeve on standby for rollback.
- **Stage E — capital [OPERATOR].** Scale only once the new-venue book shows
  positive net forward PnL — at a % venue, capital scales linearly at any size
  (no $427k tier needed).

## 6. Honest risks + what migration does NOT fix

- **Self-custody (Hyperliquid/dYdX):** bridge + validator + key-management risk;
  a leaked master key is unrecoverable. Mitigant: agent/order key that cannot
  withdraw. **Custodial (CEX):** solvency/freeze/compromise (Bybit's Feb-2025
  ~$1.5B hack) + the US geoblock.
- **Hourly funding (Hyperliquid) vs 8h (Coinbase)** changes the carry model — the
  P245 carry math and any funding-leg logic must be re-derived for the new cadence.
- **Migration removes the COST barrier; it does not create edge.** The book's
  forward edge is still being measured (P320c: recent-era conditional edge not yet
  distinguishable from zero). What the % venue uniquely enables: BTC finally
  trades, so we get BTC forward evidence we can NEVER get on CDE — and ETH/SOL's
  already-clearing margins are captured more cheaply. Profit is not guaranteed;
  measurability and the ETH/SOL cost win are.

## 7. The decision needed before I build

1. **Which venue is legally accessible to you, and custodial or self-custody?**
   (Everything from Stage B on is venue-specific.) The operational recommendation
   is Hyperliquid for a US-context self-custody fit, Binance for non-US custodial —
   but the legal call is yours per section 1.
2. Confirm you want a trade-only / withdraw-disabled key from the start.

Once 7.1 is answered I build the adapter (Stage B) and the read-only probe, which
replaces every modeled number here with a measured one before any capital moves.
