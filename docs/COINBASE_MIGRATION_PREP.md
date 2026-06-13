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

## SHADOW run results (read-only, 2026-06-13)

`scripts/coinbase_shadow_compare.py` — live read-only parity, Coinbase CDE perp vs Kraken spot. No orders.

| Asset | CB product | basis vs KR | CB spread | KR spread | CB funding/8h |
|---|---|---|---|---|---|
| BTC | BIP-20DEC30-CDE | +0.5 bps | 3.9 bps | 0.02 bps | 0.000016 |
| ETH | ETP-20DEC30-CDE | -0.4 bps | 9.0 bps | 0.06 bps | 0.000032 |
| SOL | SLP-20DEC30-CDE | -2.2 bps | 9.0 bps | 1.5 bps | 0.000008 |

**Live-verified findings (these are why SHADOW exists):**
- ✅ Price tracks Kraken within ±2.2 bps — the CDE perp follows spot tightly.
- ✅ `fetch_orderbook` + `fetch_funding_rate` adapter methods work against live responses.
- 🔧 **Funding field is `future_product_details.funding_interval` ("3600s" duration string), not `funding_interval_hours`** — adapter fixed + unit-tested.
- ⚠️ **CDE spreads are 4–9 bps vs Kraken spot 0.02–1.5 bps** (nano-perp venue is younger/thinner). Maker-first is essential; 0 bps maker fee offsets, but taker crossing is materially more expensive than Kraken spot.
- ⚠️ **Nano contracts: `contract_size` = 0.01 BTC / (per-asset).** Orders are sized in **contracts**, not base units — the execution layer must convert HMATS base-asset exposure → contract count when wiring live. Not yet handled.
- ✅ `region_enabled.US = true`, `intraday_margin_rate ≈ 0.10` → ~10x available, funding hourly, 24/7.

## REMAINING gates

- ✅ **USDC margin — trade-ready (confirmed 2026-06-13).** Account is enabled to trade derivatives; collateral is **USDC**. The earlier puzzle is explained: USD-denominated fields (`cfm_usd_balance`, `available_margin`) read 0 because the balance is **USDC**, while `futures_buying_power = 4000` reflects that USDC. Fund/trade via the **Futures (CDE)** surface, not INTX Perpetuals.
- ✅ **Nano-contract sizing — implemented + tested.** Orders trade in **whole contracts** (base_increment=1): BTC `contract_size`=0.01 (~$635/contract), ETH=0.1 (~$166), SOL=5.0 (~$333). `CoinbaseAdapter.place_order` now converts HMATS base-asset exposure → integer contracts (cached `contract_size`, fallback table), rejecting sub-1-contract orders (`BELOW_MIN_CONTRACT`). **Implication:** min position is coarse ($166–$635) — the position sizer must respect this on a ~$11K account.

Still open before live orders:
1. **Trade-enabled CDP key.** Current key is read-only (correct for probe + SHADOW). Live orders need a trade-scoped key. Rotate the read-only key (pasted in chat).
2. **Wire DUAL_VENUE into the execution path.** The adapter is order-ready but **not yet called by `execution_manager`/`main.py`** (SINGLE_EXCHANGE kraken-only gate still in force). This is the remaining integration: route per `RoutingPolicy`, gated by `cutover.assert_safe_to_advance`, starting at SHADOW.

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
