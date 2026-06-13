# Coinbase Migration Prep — v5.1 Phase 2 (status + blockers)

**Date:** 2026-06-13
**Scope:** prep/scaffolding only. **No live cutover. Nothing wired into the live path.**
**Verified against:** ccxt 4.5.34, local repo, web (Coinbase docs).

---

## TL;DR

The Phase-2 abstraction is **substantially built and tested** (adapter ABC, Kraken/Coinbase adapters, symbol map, routing state machine, cutover invariants, funding scaffold — 60 tests green). It is **inert** (no live wiring).

**Goal (operator, 2026-06-13):** Coinbase is for **futures/derivatives** — real perpetuals so the short-biased strategy can express shorts cleanly (the absence of which caused P140) and unlock funding-rate strategies (Phase 3). Spot + Kraken 2x margin (B2) is a stopgap.

**Updated viability (web-verified 2026-06-13): Coinbase IS a viable derivatives venue via the Advanced Trade REST API.** The earlier ccxt-only pessimism was misleading:

> The **Coinbase Advanced Trade REST API** supports perpetual futures programmatically — `product_type="future"`, `contract_expiry_type="perpetual"`, **up to 10x leverage**, Market + Limit, **0.00% maker / 0.03% taker** (confirms V14), **USDC** margin in a perpetuals portfolio, **10 USDC min notional**. ccxt's `coinbaseadvanced.has.swap=False` is an incomplete unified-flag — use the **raw Advanced Trade / CDP-authenticated endpoints**, not ccxt unified swap.

The remaining gate is **account eligibility + the exact product set**, not "does the API exist" (so it is *less* blocked than V13/Deribit). See blockers below.

---

## What already existed (prior session, well-built)

| File | Role | State |
|---|---|---|
| `exchange/adapter.py` | `ExchangeAdapter` ABC + `OrderRequest`/`OrderResult`/`FundingRateData` | complete |
| `exchange/kraken_adapter.py` | wraps ccxt Kraken; `place_order` returns `DELEGATED` (live still flows through `execution_manager` during dual-venue) | complete |
| `exchange/coinbase_adapter.py` | Coinbase US perps adapter — **real, SDK-backed** (`coinbase-advanced-py` RESTClient, CDP auth). place/cancel/balance/positions/orderbook/funding implemented; fail-closed without creds. Pending: confirmed `product_id`s + creds + SHADOW validation of leverage/reduce_only/funding-field placement | implemented |
| `exchange/symbol_mapping.py` | `SYMBOL_MAP` kraken/coinbase × perp/spot | complete (coinbase perp = real US CDE ids `BIP/ETP/SLP-20DEC30-CDE`, confirmed via probe) |
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

## Target product (CONFIRMED 2026-06-13)

**Coinbase US Perpetual-Style Futures** (CFTC-regulated, Coinbase Derivatives) — operator's account is perp-enabled for this product. Long-dated (5yr) contracts with a funding mechanism, 24/7, accessed via the Advanced Trade API (`product_type=FUTURE`, `contract_expiry_type=PERPETUAL`), CDP auth.

**Asset coverage — all 3 HMATS assets available:** US Perpetual-Style Futures now list **BTC, ETH, XRP, SOL** (SOL added post-launch). So BTC/ETH/SOL can all migrate to Coinbase perps — no forced split with Kraken. (Leverage: BTC/ETH up to 10x nano, others up to ~5x — confirm per product.)

## RESOLVED gates

- ✅ **Account perp-eligibility** — confirmed (US Perpetual-Style Futures enabled).
- ✅ **SOL availability** — confirmed listed (BTC/ETH/XRP/SOL).
- ✅ **PARAMETER 3 — cutover mode = DUAL-VENUE** (decided 2026-06-13). Rationale: account just recovered from the P140 incident (−25%), the Coinbase integration is new with live field names unverified, and Iron Law 8 requires DRL ACTIVE throughout — so phase in (SHADOW read-only → 50/50 → 100%, rollback always permitted) rather than hot-swap the whole account onto an unproven path. `RoutingPolicy` already defaults to this progression; `cutover.assert_safe_to_advance` gates every step.
- ✅ **Product IDs — resolved via live read-only probe 2026-06-13.** The account trades **Coinbase Derivatives Exchange (-CDE, US FCM)**, NOT International (`-PERP-INTX` is visible but **US-restricted** — do not use). The perpetual-style (5yr-dated) products, now in `exchange/symbol_mapping.py`:
  - BTC → `BIP-20DEC30-CDE` (disp "BTC PERP")
  - ETH → `ETP-20DEC30-CDE` (disp "ETH PERP")
  - SOL → `SLP-20DEC30-CDE` (disp "SOL PERP")
  (The `20DEC30` tag is the current perpetual-style contract; re-probe if Coinbase rolls it.)
- ✅ **CDP credentials** — read-only key provided + working (probe authenticated). Recommend rotating it (it was pasted in chat). A trade-enabled key is needed later for live orders.

## REMAINING gates (operational — code side is done)

1. **USDC funding — diagnosed, ops step remains.** Probe shows `futures_buying_power = 4000 USD` on the **FCM (CDE) side** but `cfm_usd_balance = 0` / `available_margin = 0`, and only a **Default** portfolio exists. The 4000 is recognized as US-futures buying power but is not yet settled as usable margin in the CFM futures wallet; the operator was likely trying to fund **"Perpetuals" (INTX, US-restricted)** which a US account cannot. **Action:** fund/confirm via the **"Futures" (CDE)** surface, not "Perpetuals"; reconcile the buying_power(4000) vs available_margin(0) gap before live orders (Coinbase Futures UI or a tiny SHADOW->DUAL test order).
2. **Trade-enabled CDP key.** The current key is read-only (correct for the probe); live orders need a trade-scoped key. Also rotate the read-only key (it was pasted in chat).
3. **SHADOW validation (read-only):** parity-compare Coinbase CDE vs Kraken funding/spread/depth and confirm live response shapes (funding fields on `get_product`, positions, leverage/reduce_only on orders) before DUAL_VENUE.

**Net:** product_ids done, read-only creds working, cutover mode decided, all 3 assets present, adapter implemented + tested. Remaining is operational: settle/confirm the USDC margin on the CDE side, mint a trade-scoped key, then run SHADOW.

## What is intentionally NOT done (until unblocked)

- **Not wired into the live path.** `execution_manager` / `main.py` / live config unchanged; the `SINGLE_EXCHANGE_GATE` (kraken-only) remains in force. The adapter is built and unit-tested but no order path calls it yet.
- **No live ORDER/trade call made.** Read-only probe calls (list products, accounts, balances) ran 2026-06-13. The adapter's order-path behavior (exact funding field names, leverage/reduce_only placement on the perp order body, positions response shape) is still unverified against live responses — confirmed in the SHADOW phase.
- `coinbase_funding_feed` left as the fail-closed fallback scaffold; the adapter's `fetch_funding_rate` is the primary path once creds land.

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
