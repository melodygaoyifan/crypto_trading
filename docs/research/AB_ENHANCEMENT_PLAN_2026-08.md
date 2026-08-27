# A/B Return-Enhancement Plan — status + measured results (2026-08-27)

Covers every "not-yet-done" lever the operator asked about (A = skew/options
gates; B = bigger improvements). The honest headline: **everything cheap and
runnable was run, and A1/A3/A4/B2/B4 are measured DEAD/INERT** — which is the
expected outcome (the direction-seat search is exhausted; research 2026-08-26).
The one live lever remains **WS2 (trend+skew conviction sizing, BTC/ETH)**,
already built + forward-shadowed.

## A — skew / options gates (all judged as GATES on the WS2 book, not new alpha)

| probe | hypothesis | result | verdict |
|---|---|---|---|
| **A1** skew term-structure (front vs back tenor) as a de-risk gate | cap conviction when near-term skew spikes | 0/3 eras BTC+ETH, only reduces net, DD not improved | **DEAD** |
| **A2** VRP (IV−RV) as a regime gate | high VRP → size up, stress → down | **NOT RUN — no ATM-IV locally** | **DEFERRED** (needs a Laevitas ATM-IV pull; the only untested item) |
| **A3** dealer gamma (GEX) sign as a size gate | cap size-up when dealer gamma negative | changes ±0.00–0.03, no effect | **INERT** |
| **A4** skew-MOMENTUM vs skew-LEVEL | momentum of skew sizes better | far worse (Δ −0.7 to −3.3) | **DEAD** — confirms the LEVEL-contrarian skew (P407) is the right form |

`training/gate_probes_lab.py` records A1/A3/A4 (+ B2/B4) so they are not re-run.

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
The cheap enhancement space is now measured-exhausted: **WS2 is the lever**, and
it is already live-shadowed. The only genuinely-untested item is **A2 (VRP)**,
which needs an ATM-IV pull; **B3 (ETF)** and **B5 (scale)** are the deferred/
operator items. No further gate-hunting is justified on this evidence.
