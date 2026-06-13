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

## Target product (CONFIRMED 2026-06-13)

**Coinbase US Perpetual-Style Futures** (CFTC-regulated, Coinbase Derivatives) — operator's account is perp-enabled for this product. Long-dated (5yr) contracts with a funding mechanism, 24/7, accessed via the Advanced Trade API (`product_type=FUTURE`, `contract_expiry_type=PERPETUAL`), CDP auth.

**Asset coverage — all 3 HMATS assets available:** US Perpetual-Style Futures now list **BTC, ETH, XRP, SOL** (SOL added post-launch). So BTC/ETH/SOL can all migrate to Coinbase perps — no forced split with Kraken. (Leverage: BTC/ETH up to 10x nano, others up to ~5x — confirm per product.)

## RESOLVED gates

- ✅ **Account perp-eligibility** — confirmed (US Perpetual-Style Futures enabled).
- ✅ **SOL availability** — confirmed listed (BTC/ETH/XRP/SOL).
- ✅ **PARAMETER 3 — cutover mode = DUAL-VENUE** (decided 2026-06-13). Rationale: account just recovered from the P140 incident (−25%), the Coinbase integration is new with live field names unverified, and Iron Law 8 requires DRL ACTIVE throughout — so phase in (SHADOW read-only → 50/50 → 100%, rollback always permitted) rather than hot-swap the whole account onto an unproven path. `RoutingPolicy` already defaults to this progression; `cutover.assert_safe_to_advance` gates every step.

## REMAINING gates (mechanical — read-only probe resolves most)

1. **Exact `product_id`s.** The symbol map's `BTC-PERP` is a placeholder. Run `scripts/coinbase_probe.py` (read-only, no orders) with a CDP key → lists the live perp `product_id`s + confirms the API surface (Advanced Trade vs Derivatives-FCM). Paste results → finalize `exchange/symbol_mapping.py` `coinbase/perp`.
2. **🚩 USDC margin funding (operator is blocked here, 2026-06-13).** Operator holds **4000 USDC** but cannot load it into the derivatives account. Margin USDC must live in the **derivatives/perps portfolio**, not the default/spot portfolio. Funding paths: perp market → *Manage funds → Transfer funds for perpetuals* (Default → Perpetuals), or *Deposit → Receive crypto → USDC → destination = Perpetuals*, or the `move_portfolio_funds` API. **Likely cause of the block:** a product/portfolio mismatch — the "Perpetuals portfolio" is **INTX (International, US-restricted)**, whereas **US Perpetual-Style Futures** fund a **Coinbase Derivatives (FCM) futures wallet** via a different flow (and may require completing the futures-account onboarding + have settlement-window cutoffs). The probe now dumps `get_portfolios` + `get_futures_balance_summary` + `get_perps_portfolio_summary` to show exactly which surface exists and where the USDC is.
3. **CDP credentials.** Advanced Trade uses CDP API keys (ES256 JWT, not legacy HMAC). Read-only key for the probe; a trade-enabled key later for the adapter. → `.coinbase_key.json` (downloaded CDP key file, gitignored) or env `COINBASE_API_KEY`/`COINBASE_API_SECRET`.
4. **Funding endpoint:** `coinbase_funding_feed._raw_funding()` (or the adapter's `fetch_funding_rate` via `get_product`) — confirm the live funding field names during SHADOW once product_ids known.

**Net:** viable and largely de-risked. Gates 1+3 are resolved in one read-only probe run; gate 2 is an ops step (USD→USDC); gates 4+5 are wiring once product_ids are known.

## What is intentionally NOT done (until unblocked)

- **Not wired into the live path.** `execution_manager` / `main.py` / live config unchanged; the `SINGLE_EXCHANGE_GATE` (kraken-only) remains in force. The adapter is built and unit-tested but no order path calls it yet.
- **No live API call made** — no credentials exist, so the adapter's real behavior (exact funding field names, leverage/reduce_only placement on the perp order body, positions response shape) is unverified against live responses. These are confirmed in the SHADOW phase.
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
