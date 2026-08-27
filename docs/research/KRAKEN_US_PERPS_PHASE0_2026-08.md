# Kraken US Perps — Phase-0 Viability Probe (measured 2026-08-27, read-only)

## TL;DR
**Economics say GO; access says NOT YET.** A percentage-fee perp venue with
tight spreads makes the IC≈0.04 directional signal — measured DEAD on the flat
CDE fee (P385/P403) — *viable* (clears break-even at maker on BTC/ETH/SOL). But
the `KRAKEN_DERIVS` keys do **not** authenticate against standard Kraken
Futures, which is US-restricted anyway; the US-legal path (Bitnomial) is a
different API the code does not target.

## What was measured (public + signed reads, NO orders, MODE stayed off)
- **Live spreads (/tickers):** BTC **0.13 bps**, ETH **0.40 bps**, SOL **2.91 bps** — deep/liquid.
- **Fee model:** percentage, published base tier ~2 bps maker / ~5 bps taker.
- **Break-even RT fee for our signal (P407m):** BTC 8.3 / ETH 12.6 / SOL 12.5 bps.
- **Total RT (maker) = 2×2 + spread:** BTC 4.1 / ETH 4.4 / SOL 6.9 → **ALL CLEAR**.
  Taker (2×5 + spread): ETH 10.4 clears; BTC 10.1 / SOL 12.9 just over.
- **Universe:** 276 perps incl. PF_XBTUSD/ETHUSD/SOLUSD + breadth (ADA/XRP/UNI…),
  contractSize=1 — finer granularity than CDE's nano (0.01 BTC).

## Why it matters
Every trained-model verdict (TQC 0/21, supervised zoo dead, ridge lockbox FAIL)
was rendered at the FLAT CDE fee (~28 bps RT), which is ~2× the signal. The
signal itself is REAL (IC 0.04, t≈3). On a percentage venue at these
spreads/fees the fee wall drops and the signal clears — the first **measured**
support for the P385/P407m thesis, and the RULE stack (SMA200/skew) benefits
first, TQC only if a formulation ever earns it.

## The blocker (access) — this is what stops it being real today
- `KRAKEN_DERIVS_API_KEY/SECRET` present; the probe's signing is byte-identical
  to `KrakenDerivativesClient._sign` (verified), yet `/accounts` returns
  `authenticationError` at `futures.kraken.com` (live) and non-JSON at
  `demo-futures.kraken.com`. So the keys are stale/revoked OR for a different
  endpoint/product.
- `futures.kraken.com` is standard **international** Kraken Futures =
  **US-restricted** — the reason the whole stack is `MODE=off`, "disabled until
  Kraken Futures wired" (`live_phase1.json`), and the system pivoted to Coinbase CDE.
- The US-legal product is Kraken's **Bitnomial-backed, CFTC-regulated US perps**
  (P399) — a *separate* API the existing `KrakenDerivativesClient` does NOT target.

## Phased plan (gated, P141 — no speculative build)
- **Phase 0 — NEEDS OPERATOR (the gate):** confirm what the `KRAKEN_DERIVS` keys
  are for — current? which entity? is there a *separate* US-perps (Bitnomial)
  credential + endpoint? Without authenticated US-eligible access, nothing
  downstream is real.
- **Phase 1:** if US perps = Bitnomial, build a Bitnomial adapter (existing client
  points at the wrong venue); read-only account + real fee-tier + order-book RT.
- **Phase 2:** route a small book there, SHADOW-first, maker-ladder execution.
- **Phase 3:** re-run edge_probe / retrain at the **measured** new RT (never
  assumed) → Rung-0 → forward shadow → P166 gate → deploy. No tune-to-pass.

## Explicitly NOT justified now
Building the Bitnomial adapter before Phase 0 confirms access — a new exchange
integration for a venue we cannot even authenticate to is the P141
premature-activation trap. The measurement is the deliverable; the venue
decision is the operator's.
