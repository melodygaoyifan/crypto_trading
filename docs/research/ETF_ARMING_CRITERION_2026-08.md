# ETF-flow de-risk overlay — pre-committed arming criterion (2026-08-24, P404)

Pre-committed BEFORE the forward evidence lands, so the eventual decision is a
gate, not post-hoc selection (the P332 discipline). Nothing here arms anything;
arming is an operator flip (P141) once every clause below is satisfied.

## What would be armed
A **de-risk overlay** on the live sleeve, NOT a new model:
- **BTC:** SMA200 long/flat (already the live decider) gated by ETF flow —
  flatten to 0 when ETF signals a strong OUTFLOW (z < -1.0). P404 backtest:
  Sharpe +1.49 / maxDD -9.7% vs SMA-alone +0.37 / -22.1%; corr(SMA,ETF) -0.12
  (complementary). 2y history.
- **ETH:** ETF **standalone** (long/short z-deadband) — SMA200 is the weak link
  on ETH (Sh -1.41), so the combination hurts; ETF alone is +0.63.
- **SOL:** nothing (no SOL spot ETF exists).

## The arming bar — ALL must hold
1. **Leak-free.** `etfflow_timing_check.py` reports ZERO in-progress-day leaks
   over >= 10 completed flow-days (exit 0, not 3).
2. **Lag matches the backtest.** Effective-lag median <= 2 days (~ lag-1,
   weekend-adjusted). If routinely > 3 (feed publishes late), re-run the P400
   backtest at that lag and require it still clears before arming — a later feed
   is conservative-but-weaker, not automatically disqualifying.
3. **Live-consistency.** The combination shadow's live `combo_direction` matches
   what the 2y backtest rule produces for the same completed flow-days over the
   accrual window (no wiring divergence).
4. **Operator risk acceptance.** It is a MODEST, regime-concentrated (recent
   ETF era, could fade) de-risk overlay on a fee-floor-bound ~$11k book;
   revert is one config flip.

## What arming REQUIRES building (deliberately NOT built now, P141)
The combination shadow uses a PROXY SMA200 (flow-history daily price) to measure
the concept live. Actual arming needs a small, flag-gated ETF-outflow de-risk
veto on the sleeve driver (the P206 translator pattern), off by default, built
at arming time on the LIVE sleeve position — not a live-order gate built ahead of
the evidence. Until clause 1-4 are met, that build is not justified.

## Timeline
Clause 1 (leak/timing): ~2 weeks (10 completed flow-days; cron'd). Clauses 2-3:
~2-4 weeks. A ~30-trade/yr signal CANNOT forward-CERTIFY fast (~1-2 years,
P293g); the evidence base is the 2y backtest + complementarity + leak-free +
live-consistency, and the decision is then a risk call, not a stat-cert wait.
