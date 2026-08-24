# Data-Feed Research — the right feed to lift IC above the fee floor (2026-08-24)

**Constraint (operator):** stay on **Coinbase + Kraken**. No percentage-fee venue.
So the flat per-contract CDE fee stays, the required IC stays **0.07–0.11** at $11k,
and **new data is the only code-side lever** — it must roughly DOUBLE the current
IC-0.04 signal to clear the floor (or capital must scale, which lowers the floor in
bps but is non-code).

This is a research pass on which data feed could plausibly do that, graded by
**evidence × fetchability-for-BTC/ETH/SOL × cost**, with a concrete probe plan. It
combines a current-literature scan (2025–26) with fresh screens on data we already
hold.

---

## 0. What the ceiling is, measured

Six bases now sit at OOS IC ≈ 0.04 (t≈3), ~2× below the floor: single-asset TA
(P385), higher-freq flow (P375b), L2 depth (P379), tick microstructure (P385b),
cross-asset breadth (P385c), and — new here — the **horizon sweep**: predicting
16h/24h/48h/72h forward returns on the existing features tops at BTC 24h IC **0.043**
(t 3.26), ETH 48h **0.029**, SOL negative — none clears. The ceiling is robust to
model, feature set, data basis, AND horizon. So a new feed must carry information the
price/flow series does not.

---

## 1. Literature scan (2025–26), net-of-cost lens

- **Derivatives positioning (funding, open interest, liquidations, long/short
  ratio)** is the most-cited crypto-specific predictor set — framed as
  **reversal / regime early-warning**: crowded leverage (high funding, high OI,
  long-skewed accounts) precedes unwinds. Practitioner consensus + academic
  (random-forest-on-TA studies find daily-horizon outperformance; carry/positioning
  factors in BIS "Crypto carry"). **This is the strongest fetchable lead.**
- **On-chain (exchange netflows, stablecoin supply, MVRV)** is widely used but the
  rigorous OOS evidence is thin/anecdotal, and quality feeds (Glassnode/CryptoQuant)
  are **paid**. High narrative, low proven edge.
- **Options (DVOL term structure, 25-delta skew, put/call)** — real mechanism (vol
  risk premium, dealer positioning). Deribit API is **free** but **BTC/ETH only**
  (no SOL). Partial coverage.
- **Cross-sectional carry / basis** — a real factor, but P374 already tested it and
  it FAILED net (short leg fights the trending winner). Not new.
- **Intraday / quarter-hour effects** — real but the wrong scale for a 4H book.

---

## 2. Fresh screen on data we already hold — the OI-level lead

Raw single-feature IC of CoinGlass derivatives features vs forward 24h return, on the
~186 days of 4H history we have (screen only — in-sample, one window):

| feature | BTC | ETH | SOL | in basis already? |
|---|---|---|---|---|
| funding (level) | −0.079 | −0.108 | −0.022 | **yes** (`funding_rate_zscore`) |
| **OI level (z)** | −0.034 | **−0.131** | **−0.151** | **NO** — basis has only `oi_change_5d` |
| OI change | +0.002 | +0.008 | +0.032 | ~yes |
| liq_imbalance | +0.028 | −0.033 | +0.018 | **yes** |
| liq total (z) | −0.022 | −0.041 | −0.028 | no |

**The lead: OI-LEVEL z-score (`oi_z`)** — a genuinely NEW feature (the basis has OI
*change*, not *level*) — screens at **−0.12 to −0.15 on ETH/SOL**, the first thing
this session to sit materially above the 0.04 ceiling. Contrarian sign: high OI
(crowded leverage) → lower forward return. Funding screens −0.08/−0.11 too, but it is
ALREADY in the basis (its 6y walk-forward contribution is counted in the 0.04
ceiling), so its 186d strength is a recent-regime artifact, not new edge.

**Three caveats that keep this a HYPOTHESIS, not a finding:**
1. **In-sample, 186 days, one window, no cost, no walk-forward** (P348: cannot
   conclude). The five dead bases also had raw ICs of 0.04–0.07; oi_z's 0.12–0.15 is
   notably larger but on one thin window.
2. **OI/liquidation history is capped at ~186 days** by CoinGlass's rolling window
   (P266) — so oi_z **cannot be validated OOS today**. This is the binding constraint.
3. **A level-z of a secularly-growing series over a short window carries a spurious
   trend/level bias**, and a contrarian sign is regime-fragile (funding's own
   contrarian signal averages to ~0.04 over 6y; P244/P374 show funding-reversal
   signals invert across regimes). oi_z could be the same.

---

## 3. Graded feed shortlist

| tier | feed | mechanism | coverage | cost | evidence | verdict |
|---|---|---|---|---|---|---|
| **1** | **OI-level + long/short account ratio** (CoinGlass) | crowded-leverage reversal | BTC/ETH/SOL | have key | raw IC 0.12–0.15 (oi_z, 186d screen) — **best lead** | ACCUMULATE forward, then probe |
| 2 | options: DVOL term, 25Δ skew, put/call (Deribit) | vol risk premium / dealer positioning | BTC/ETH only | free | real mechanism, untested here | probe BTC/ETH; SOL gap |
| 3 | on-chain: exchange netflows, stablecoin supply (Glassnode/CryptoQuant) | supply/demand pressure | BTC/ETH (SOL weaker) | **paid** | high narrative, thin OOS | only if Tier 1/2 fail + operator pays |
| — | cross-sectional carry/basis | factor premium | all | have | **FAILED net (P374)** | dead |
| — | higher-freq flow / L2 depth / tick micro / breadth | microstructure | all | have | **no pulse (P375b/P379/P385b/c)** | dead |
| — | horizon extension (24h/48h) | — | all | free | tested §0 — **no pulse** | dead |

---

## 4. The plan (gated, honest)

**Enabling action (do first, cheap, operator/cron):** start **accumulating
full-resolution OI + liquidation + long/short-ratio history forward** via the
CoinGlass key (a daily append cron, the pattern P282 started for CDE funding). This
is the load-bearing step: oi_z's only blocker is that OI history is a 186-day rolling
window, so OOS validation is impossible until we bank forward data. In ~6–12 months
there is enough to walk-forward-probe it honestly.

**Then, gated (the P200 ladder, unchanged):**
1. Build `oi_z` + `long_short_ratio_z` (+ Deribit skew/DVOL-term for BTC/ETH) as
   causal, detrended features.
2. **Rung-0 hold-aware probe** (`signal_hold_backtest.py`, P386 — fee on flips only,
   long/short + funding, walk-forward deadband) on the accumulated data. Bar: held
   OOS net > buy-and-hold on ≥2/3 assets with a pre-committed deadband.
3. Only if it clears: add the features → refit GMM split-aware (paired, P215) →
   retrain DRL/supervised → 30d forward shadow → P166 gate → DECIDE seat.
   (Per `PROFITABILITY_ROOT_CAUSE_PLAN_2026-08.md` §4.)

**Do NOT:** retrain on the 186-day in-sample oi_z now (the P164/P200 thin-window leak
trap); pay for on-chain before Tiers 1–2; or read the 186d screen as a validated
edge.

---

## 5. Honest bottom line

The research found ONE feed direction worth pursuing — **derivatives positioning
(OI level + long/short ratio, contrarian)** — the only thing that screens above the
0.04 ceiling. But it is a **hypothesis blocked on data depth**, not a ready edge: OI
history is a 186-day rolling window, so it can only be validated after months of
forward accumulation, and its contrarian sign is regime-fragile. The single most
useful action is therefore **operational, not modeling**: start banking
full-resolution OI/liq/long-short data now, so the probe becomes possible. Everything
else (retrain, DRL) stays gated behind that probe. And the surest lever remains the
one the venue constraint rules out or defers — scale/percentage-fee — which would
make the IC-0.04 signal we ALREADY have tradeable without any new data.
