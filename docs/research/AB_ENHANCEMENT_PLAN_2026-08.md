# A/B Return-Enhancement Plan — status + measured results (2026-08-27)

Covers every "not-yet-done" lever the operator asked about (A = skew/options
gates; B = bigger improvements). The honest headline: **everything cheap and
runnable was run, and A1/A2/A3/A4/B2/B4 are measured DEAD/INERT** — which is the
expected outcome (the direction-seat search is exhausted; research 2026-08-26).
The one live lever remains **WS2 (trend+skew conviction sizing, BTC/ETH)**,
already built + forward-shadowed.

## A — skew / options gates (all judged as GATES on the WS2 book, not new alpha)

| probe | hypothesis | result | verdict |
|---|---|---|---|
| **A1** skew term-structure (front vs back tenor) as a de-risk gate | cap conviction when near-term skew spikes | 0/3 eras BTC+ETH, only reduces net, DD not improved | **DEAD** |
| **A2** VRP (Deribit DVOL − realized vol) as a de-risk gate | compressed VRP (stress) → cap conviction to 1x | BTC 1/3 eras (design +0.222 only), ETH 0/3; **negative in the honest recent read** (BTC −0.298, ETH −0.231), no DD payoff | **DEAD** — the WS2 skew de-risk leg already cuts euphoria; VRP is redundant, not orthogonal |
| **A3** dealer gamma (GEX) sign as a size gate | cap size-up when dealer gamma negative | changes ±0.00–0.03, no effect | **INERT** |
| **A4** skew-MOMENTUM vs skew-LEVEL | momentum of skew sizes better | far worse (Δ −0.7 to −3.3) | **DEAD** — confirms the LEVEL-contrarian skew (P407) is the right form |

`training/gate_probes_lab.py` records A1/A2/A3/A4 (+ B2/B4) so they are not
re-run. A2 uses Deribit DVOL (keyless, ~5.4y from 2021-03, operator-local at
`training/training_data/dvol/`) — real implied vol, not the 3-month Laevitas API
cap; the probe refuses gracefully (SKIPPED) when the pull is absent, so CI stays
clean.

## B — bigger improvements

| item | status |
|---|---|
| **B1** objective reframe (reward captured return per era, not just Sharpe) | **PRE-COMMITTED below** — the rule WS1/WS2 forward reads are judged by |
| **B2** WS2 for SOL via trend+REGIME | **DEAD** — SOL's bull regimes (STEADY_UPTREND/QUIET_ACCUMULATION/MOMENTUM_RALLY) OVERLAP the trend, so regime is NOT orthogonal → sizing up is just leverage → deeper drawdowns, lower net (0/3 eras). This is exactly WHY WS2 works: skew is orthogonal (contrarian) to trend; a correlated signal doesn't help. SOL stays on the base trend book. |
| **B3** WS2 + ETF flow | **DEFERRED** — ETF history is thin (~1.7y); forward-only, add once its shadow matures |
| **B4** wide crash-stop as a WS2 drawdown enabler | **INERT-to-harmful** — BTC near-zero, ETH negative (a 15% stop triggers on normal vol). Not worth it. |
| **B5** scale / capital | **OPERATOR / non-code** — the flat fee amortizes with notional; the largest structural lever, but a capital decision |

## B1 — the pre-committed verdict rule (Move #0)

For the WS1 and WS2 forward reads (and any future overlay), the acceptance
objective is a JOINT, per-era judgment that rewards captured return, so the
pipeline stops rejecting return improvements for shaving Sharpe:

> **PROMOTE a variant to a live P141 cutover iff, on the FORWARD ledger:
> (1) raw net-return increment > 0 in the majority of read windows;
> (2) Sharpe does NOT fall vs the base; (3) the maxDD increase is justified by
> the return increment (return/DD ratio not worse); and (4) it beats its control
> (random-tier for WS2). Reported per era, raw-return-weighted.**

This is a decision-rule pre-commitment (P332 discipline), recorded BEFORE the
forward reads so the verdict cannot be selected after seeing the numbers.

## Net conclusion
The cheap enhancement space is now **fully measured-exhausted** — A2 (VRP) was
the last untested cheap gate and it is dead. **WS2 is the lever**, already
live-shadowed. What remains is not code: **B3 (ETF)** is forward-only (thin
history, add when its shadow matures) and **B5 (scale)** is a capital decision
(the flat CDE fee amortizes with notional; the sizing is already equity-scaled
per P274, so arriving capital deploys itself — see B5 note below). No further
gate-hunting is justified on this evidence.

## B5 — scale / capital (verified 2026-08-27, non-code)

The single largest structural return lever is **notional**, because the CDE fee
is FLAT per nano contract (~$0.60), so its cost in bps falls linearly as the
book grows — the "crack in the wall" (P385/P407m: break-even RT is BTC 8.3 /
ETH 12.6 / SOL 12.5 bps, vs ~28bps flat-fee at $11k; the signal IC ~0.04 clears
at a percentage venue OR at enough scale on the flat fee). **The system is
already scale-ready**: `coinbase_target_fraction_by_asset` sizes each asset as a
FRACTION of live sleeve equity (P274), floored to `max(1, ...)` contracts, so
arriving capital deploys itself with no config change — the identity is pinned
(at the activation equity the fractions reproduce the exact contract book).
Nothing to build; the decision is the operator's, and it is now informed: at
~$11k the flat fee is the binding constraint on every rules edge (regimebook,
skew seat, WS2), and it only relaxes with scale (or a percentage venue, which is
US-blocked — Kraken US-restricted, Bitnomial can't-open, spot-only API).
