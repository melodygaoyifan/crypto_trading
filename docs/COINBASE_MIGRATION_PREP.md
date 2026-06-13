# Coinbase Migration Prep — v5.1 Phase 2 (status + blockers)

**Date:** 2026-06-13
**Scope:** prep/scaffolding only. **No live cutover. Nothing wired into the live path.**
**Verified against:** ccxt 4.5.34, local repo, web (Coinbase docs).

---

## TL;DR

The Phase-2 abstraction is **substantially built and tested** (adapter ABC, Kraken/Coinbase adapters, symbol map, routing state machine, cutover invariants, funding scaffold — 60 tests green). It is **inert** (no live wiring). Advancing past prep is **blocked on a venue/product access question**, not on code:

> **ccxt perps for Coinbase exist only via `coinbaseinternational` (US-restricted). The US-accessible `coinbaseadvanced` has `swap=False` (no perps). Neither exposes funding via ccxt's unified method. "US Perpetual-Style Futures" are a different product (CFTC, 5yr-dated) with their own API.**

This is the same shape as v5.1's **V13 (Deribit US-restricted → deferred)**. The operator must confirm which Coinbase perp product their account can trade *via API* before any real adapter wiring or cutover.

---

## What already existed (prior session, well-built)

| File | Role | State |
|---|---|---|
| `exchange/adapter.py` | `ExchangeAdapter` ABC + `OrderRequest`/`OrderResult`/`FundingRateData` | complete |
| `exchange/kraken_adapter.py` | wraps ccxt Kraken; `place_order` returns `DELEGATED` (live still flows through `execution_manager` during dual-venue) | complete |
| `exchange/coinbase_adapter.py` | Coinbase skeleton; fail-closed `NOT_CONFIGURED` until client wired | skeleton |
| `exchange/symbol_mapping.py` | `SYMBOL_MAP` kraken/coinbase × perp/spot | complete (perp symbols assume International-style `BTC-PERP`) |
| `exchange/routing.py` | `RoutingPolicy` + `CutoverPhase` state machine, Iron-Law-8 phase gating | complete |
| `tests/test_exchange_adapter_v5_1.py` | adapter + routing + symbol tests | complete |

## What this session added

| File | Role |
|---|---|
| `exchange/cutover.py` | **NEW** — was referenced by `adapter.py`/`routing.py` but missing. Pure-logic Iron-Law invariants (1: obs_dim=126, 5/8: DRL ACTIVE continuous, 9: maker-first), `cutover_invariants()`, `assert_safe_to_advance()` (ROLLBACK always permitted; never mutates policy). |
| `data_mgmt/feeds/coinbase_funding_feed.py` | **NEW** — Phase 2.3 funding scaffold. Fail-closed (returns `None` → caller falls back to Kraken funding), normalizes to 8h-equivalent. Disabled by default (zero side effects on import). |
| `tests/test_cutover_v5_1.py` | **NEW** — 14 tests for cutover invariants + funding scaffold. |

**Test result:** `60 passed` (new + existing).

---

## BLOCKERS — operator must resolve before real wiring/cutover

1. **🚩 Product/access verification (highest priority).** Which Coinbase perp can the account trade *via API*?
   - `coinbaseinternational` (ccxt `swap=True`) — **US-restricted**. If the operator is US-based, likely unavailable.
   - `coinbaseadvanced` (ccxt `swap=False`) — US-accessible but **no perps via ccxt**.
   - **US Perpetual-Style Futures** (Coinbase Financial Markets/Derivatives, CFTC, 5yr-dated, funding accrues hourly/settles twice daily) — different symbols + API, **no confirmed ccxt unified support**. If this is the operator's path, the `BTC-PERP` symbol map and the funding feed both need reworking for that product.
2. **PARAMETER 3 — cutover mode:** hot-swap / dual-venue / phased. `RoutingPolicy` defaults to the dual-venue 4-week schedule; confirm or override.
3. **Credentials + auth scheme:** `COINBASE_API_KEY` / `COINBASE_API_SECRET` (+ CDP key vs legacy HMAC vs passphrase — depends on product).
4. **Funding via raw endpoint:** ccxt unified `fetchFundingRate` is unavailable on the perp class; the feed's `_raw_funding()` must call the confirmed product's raw endpoint.
5. **SOL-PERP listing:** the map lists `SOL-PERP`, but the v5.1 V14 note said "SOL pending list verify". Confirm SOL perp exists on the chosen product.
6. **Fee confirmation:** V14 GREEN said 0bps maker / 3bps taker promotional — confirm still current and which product it applies to.

## What is intentionally NOT done (until unblocked)

- No real Coinbase HTTP/WS client (adapter stays skeleton). Wiring an unverified API would likely be wrong and can't be tested without access + creds.
- No `coinbase_funding_feed` live calls (scaffold disabled by default).
- No changes to `execution_manager` / `main.py` / live config. The `SINGLE_EXCHANGE_GATE` (kraken-only) in `execution_manager.py` remains in force.

## Iron-law compliance

- Iron Law 1 (obs_dim=126): untouched; `cutover.validate_obs_dim` enforces it as a cutover gate.
- Iron Law 2 (constitution): untouched.
- Iron Law 3 (training/): untouched.
- Iron Law 5/8 (DRL ACTIVE): `cutover` + `routing.advance_phase` refuse to advance if DRL ≠ ACTIVE; ROLLBACK always permitted.
- Iron Law 9 (maker-first): `OrderRequest.post_only=True` default; `cutover.validate_maker_first` gates it.
- All additions are new files; the live path is unchanged.

## Suggested next steps once the product question is answered

1. Operator confirms product + provides API creds + access region → set the `coinbaseinternational` (or US-derivatives) client.
2. Implement `CoinbaseAdapter` place/cancel/fetch against the confirmed API; implement `coinbase_funding_feed._raw_funding`.
3. Run SHADOW phase (read-only) for ≥2 weeks: compare Coinbase vs Kraken funding/spread/depth parity. No orders.
4. Only then consider DUAL_VENUE per PARAMETER 3, with `cutover.assert_safe_to_advance` gating every transition.
