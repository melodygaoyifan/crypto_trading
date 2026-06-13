# Coinbase Engine Integration Plan — v5.1 Phase 2 (DUAL_VENUE)

**Date:** 2026-06-13
**Status:** DESIGN — for operator review before any `main.py`/`execution_manager` edit.
**Precondition met:** `CoinbaseAdapter` is live-validated + order-ready (see `COINBASE_MIGRATION_PREP.md`). Account flat. Trade key in `.coinbase_key.json`.

---

## Principles (non-negotiable)

1. **SHADOW-first.** No Coinbase order is placed by the engine until a read-only in-loop shadow phase has run and is reviewed.
2. **Default-off flag.** Everything gates behind `coinbase_routing_enabled` (config, default `false`). With the flag off, the engine is byte-for-byte today's behavior.
3. **Rollback always available.** `RoutingPolicy.advance_phase(ROLLBACK)` forces all orders back to Kraken instantly; flag-off is the kill switch.
4. **Iron Laws preserved.** obs_dim=126 untouched; constitution untouched; DRL ACTIVE throughout (Law 8, enforced by `cutover.assert_safe_to_advance`); maker-first; `SINGLE_EXCHANGE` still forbids DEX and forbids Coinbase unless the flag + routing both allow it.
5. **No silent state divergence** (the P139/P140 lesson) — Coinbase position state is reconciled against `list_futures_positions` (venue is authoritative), never inferred.

---

## Current call path (what we're hooking into)

```
_process_4h_tick_inner (main.py)
  -> execute_intent_v2 (core/execution_service.py)
       sizing: exposure_to_quantity -> base_quantity -> notional   (~ln 607-643)
       -> ctx.execution_manager.execute_order(                      (ln 1760 sliced / 1788 single)
              symbol=_canonical_spot_symbol(asset),  # kraken symbol
              side, size=base_quantity, price, order_type,
              leverage=..., venue defaults "kraken")
            -> SINGLE_EXCHANGE HARD GATE: venue!="kraken" -> REJECT  (execution_manager.py ln 1032)
            -> ccxt kraken order path (limit/market, P87/P138 clamps)
ExecutionManager built at main.py:2754 with self._ccxt_exchange = ccxt.kraken(...) (main.py:4312)
```

Key facts:
- `execute_order` already has a `venue` param (default `"kraken"`) — the routing seam exists.
- Both call sites pass a **Kraken** symbol and **no venue** → today everything is Kraken.
- `_paper_positions[asset]`, tranche state, fills, anti-churn, existence-fuse are all **single-venue / Kraken-shaped**.

---

## Hook points (file-by-file)

### H1. RoutingPolicy on the runner (new state)
- Add `self._routing_policy: RoutingPolicy` in `HMATSProductionRunner.__init__`, phase loaded from a small state file (`data/coinbase_routing_state.json`), default `PRE_PHASE_2`.
- Add config flag `coinbase_routing_enabled: bool = False` (ProductionConfig + JSON, same pattern as B1's `block_short_entry_on_spot`).
- Construct one `CoinbaseAdapter` (shared) when the flag is on; else leave `None`.

### H2. Venue decision in `execute_intent_v2` (core/execution_service.py ~1760/1788)
- Before the `execute_order` call: `venue = "kraken"` unless `coinbase_routing_enabled` AND `routing.venue_for(asset) == "coinbase"`.
- Choose symbol per venue: Kraken → `_canonical_spot_symbol(asset)`; Coinbase → `to_venue_symbol(asset, "coinbase", "perp")`.
- Pass `venue=venue` to `execute_order`.

### H3. SINGLE_EXCHANGE gate exception (execution_manager.py:1032)
- Change the gate from `venue != "kraken" -> REJECT` to:
  - `venue == "kraken"` → allow (unchanged).
  - `venue == "coinbase"` AND `self._coinbase_enabled` → allow, route to Coinbase branch.
  - else (incl. all DEX) → REJECT exactly as today (DEX CRITICAL path untouched).
- `self._coinbase_enabled` set only when the runner passes the live adapter in.

### H4. Coinbase execution branch in `execute_order`
- New branch: translate `(symbol, side, size, price, order_type, leverage)` → `OrderRequest` → `CoinbaseAdapter.place_order` → map `OrderResult` back to the engine's `OrderResult` shape.
- **Sync/async:** `execute_order` is sync; the adapter is async over a sync SDK. Use a thin **sync** path (call the SDK directly in the branch, reusing the adapter's `_contract_size`/`_round_to_tick`/parse helpers) to avoid `asyncio.run` inside the sync tick. (Refactor the adapter's body into sync helpers + async wrappers.)
- Adapter already handles base→contract sizing, tick rounding, no-reduce_only, futures positions (all live-validated).

### H5. Contract-granularity in the sizer
- For a Coinbase-routed asset, exposure must round to **whole contracts** (BTC 0.01 / ETH 0.1 / SOL 5.0 base units; min 1 contract = ~$166–$635). 
- The adapter rejects sub-1-contract, but the **sizer/tranche** must round target exposure to a contract multiple so the engine's intended size == placed size (avoid the "thought I placed X, placed Y" class of P139-style drift).
- Add a `venue_min_increment(asset, venue)` consulted in `execute_intent_v2` sizing.

### H6. Per-venue position + fill tracking (THE HARD PART)
- `_paper_positions` is Kraken-shaped. Coinbase positions must be tracked separately and **reconciled from `list_futures_positions`** each tick (authoritative), not inferred from fills (this is the explicit anti-P139 measure).
- Equity/PnL aggregation spans two venues (Kraken USD + Coinbase USDC buying power). The heartbeat/existence-fuse/equity-history need a combined view.
- Minimum viable: namespace Coinbase positions under `_coinbase_positions[asset]`, reconciled from the venue; keep Kraken path fully separate; aggregate equity for reporting only.

---

## Phased rollout (each gated by `cutover.assert_safe_to_advance`)

| Phase | RoutingPolicy | Coinbase orders? | What runs | Exit criteria |
|---|---|---|---|---|
| **A. SHADOW** | SHADOW | **none** | per-tick read-only Coinbase parity (funding/spread/mark) logged next to Kraken; routing returns kraken for all | ≥ N days clean reads, no adapter errors, parity sane |
| **B. DUAL (1 asset, capped)** | DUAL_VENUE, `coinbase_assets=[ONE]` | yes, **1-contract cap** | one asset routes to Coinbase, hard size cap = 1 contract, watched | ≥ M clean round-trips, positions reconcile, PnL correct |
| **C. DUAL (scale)** | DUAL_VENUE | yes | lift size cap, add assets | stable Sharpe/exec quality vs Kraken |
| **D. COINBASE_PRIMARY** | COINBASE_PRIMARY | yes | all 3 assets on Coinbase; Kraken standby | 60-day review |

Rollback at any phase: `advance_phase(ROLLBACK)` → all Kraken; or flag-off.

---

## Risks & mitigations

1. **State-model divergence (highest).** Two-venue position tracking on a codebase built for one. → Reconcile Coinbase from the venue every tick; never infer. Keep venues namespaced. Start B with a 1-contract cap so any divergence is ≤ $166.
2. **Sync/async + latency.** SDK is sync; tick is sync. → Sync helper path; the SDK call is ~100–300ms, fine for a 4H tick + 30s fast-risk.
3. **Contract granularity vs sizing.** → H5 rounds target exposure to contract multiples up front.
4. **USDC vs USD margin / equity aggregation.** → Report-only combined equity first; don't let Coinbase buying power feed Kraken sizing.
5. **DRL/fusion semantics.** They emit base-asset direction/exposure — venue-agnostic, so no model change (Iron Law 1/8 safe). Only the execution leg changes.
6. **Spread cost (4–9 bps on CDE).** → maker-first/post-only default already in `OrderRequest`; monitor fill quality in Phase B.

---

## Test plan

- Unit: routing decision (flag off → always kraken; flag on + SHADOW → kraken; DUAL → per asset), venue-branch arg translation (mock adapter), sizing→contract rounding, gate allows coinbase only with flag. (extends `tests/test_exchange_adapter_v5_1.py`.)
- Static: `python -X utf8 -m py_compile main.py core/execution_service.py execution/execution_manager.py`.
- Live: the 1-contract path is already validated (`coinbase_test_order.py`). Phase B is the first engine-driven live order, 1-contract capped, watched.

---

## Concrete change list (on approval)

1. `configs/live_high_risk.json` + `ProductionConfig`: `coinbase_routing_enabled` (default false).
2. `main.py`: build `RoutingPolicy` + shared `CoinbaseAdapter` (flag-gated); pass adapter into `ExecutionManager` (`self._coinbase_enabled`).
3. `core/execution_service.py`: venue decision + per-venue symbol + `venue=` arg at the two `execute_order` sites; contract-rounding in sizing.
4. `execution/execution_manager.py`: gate exception (H3) + Coinbase execution branch (H4).
5. `exchange/coinbase_adapter.py`: factor sync helpers out of the async methods (so the sync engine path reuses them).
6. New: `data/coinbase_routing_state.json` + per-tick SHADOW parity log; `_coinbase_positions` reconcile.
7. Tests as above.

**Estimated:** Phase A (SHADOW wiring) ~1 focused session; Phase B readiness another. The state-tracking (H6) is the bulk of the real work and risk.
