# The September Decision Tree (P271, pre-committed 2026-08-16)

**Why this document exists.** The system's profitability question resolves in
early September through several independent evidence reads. Deciding the
criteria NOW — before any result is visible — is the entire point: a
criterion chosen after seeing the number is selection, not evidence
(P198/P243's lesson). Every read below is executed by ONE command:

```bash
python -X utf8 scripts/september_check.py --window-days 30
```

Weekly trajectory reads (`--window-days 10`, informational only, cannot
promote — the scorer blocks windows ≤ 14d) run each Monday until then.

**The standing bar for every candidate** is the P166 cost-aware gate, at
every horizon with enough samples: IC positive, |IC| > 0.05,
overlap-corrected |t| ≥ 2, expected edge ≥ 2× round-trip cost. No candidate
promotes on backtest, trajectory reads, or partial windows — that is the
mistake this project has paid for twice (P147, P198).

---

## ~Aug 28 — the mlp early-seat criterion (P285, operator-adopted 2026-08-16)

**The exception to the "nothing before September" rule, and why it is not a
loosening:** the P283b-certified BTC `mlp_small` may take the BTC direction
seat EARLY — at **unchanged** size, caps, stops, and gates — because the
incumbent trend signal has measured-NEGATIVE live forward IC (P198) and is
itself tripwire-bound above. The swap chooses which signal drives an
already-trading, already-bounded ±1-contract book; it adds no risk
parameter. Pre-committed BEFORE any shadow evidence existed.

**Instrument:** `scripts/mlp_seat_check.py` (operator-local, after a
`september_check.py` ledger pull). **Actuator:** `mlpshadow_mode:
"enforce"` in the live profile (P284/P285 seat — absent by design until the
criterion fires; adding it IS the firing action).

| condition (ALL required) | value |
|---|---|
| date | ≥ 2026-08-28 |
| mlpshadow_BTC ledger | ≥ 14 days span, ≥ 40 directional records |
| 16h shadow IC | ≥ 0.0 — a **kill-screen**, deliberately weaker than P166 |
| trend incumbent | still GATE-CLOSED on the latest weekly slope report |

A negative 2-week trajectory kills the early swap (the model waits for its
full ~09-15 P166 read like everyone else); missing data never counts as
passing (P199). **The ~09-15 P166 read still governs anything beyond the
current book** — the early seat changes the direction source only, never
the alpha bar, the sizing, or the promotion standard.

## Sep 1 — the trend tripwire (P237)

**Instrument:** `tripwire_check.py` on the weekly slope-calibrator reports
(crons 08-10 ✓ armed, 08-17, 08-24, 08-31 → exactly 4 weekly-spaced reports
by the date gate).

| outcome | action |
|---|---|
| 4/4 GATE-CLOSED for an asset AND no promotable basis from the reads below | **remove that asset from `trend_assets`** in the live profile + redeploy (the trend injection comes off; the sleeve goes flat on that asset unless another seat is promoted). One config line; the P237 addendum built the lever. |
| any report shows the gate honestly OPEN for an asset | that asset's trend injection stays; re-arm the 4-report window |

This is the recorded operator decision with a date; nothing before Sep 1
should fire it. (Status at pre-commit: 1/4 reports, all three assets armed.)

## ~Sep 7 — ma_filter (ledger since 08-08)

The single most likely promotion: model_alpha has measured positive in three
independent windows (P230 +0.289; the P236 counterfactual −78.9bps/tick when
quant disagrees with it; the 08-10 report 4h IC +0.127 t=2.07).

| outcome | action |
|---|---|
| PASS (P166 bar on the `ma_filtered` ledger) | set `coinbase_ma_filter_enforce: true` (+ its own P-entry). Entry filter only — it never force-exits (P236 semantics). |
| FAIL | flag stays false; model_alpha's IC keeps accruing via the weekly cron; re-examine at 60d only if the trajectory is monotone. |

## ~Sep 9 — regimebook raw + adjusted (ledgers since 08-10)

| outcome | action |
|---|---|
| raw book PASSES | `regimebook_mode: "enforce"` (P256 seat) — the book target takes the direction seat, replacing the trend layer where they disagree (+ its own P-entry). |
| adjusted book PASSES and beats raw | seat consumes the ADJ target instead. |
| BTC passes but its funding legs' cells are the failing part | enforce trend/hold legs only; the funding legs stay shadow (P262 already marks them the UNCERTIFIED slice). |
| FAIL | books stay observation-only; the trend/hold mechanism remains certified (P262) but unexpressed; combined with a fired tripwire the honest system is FLAT until evidence changes. |

## ~Sep 9 — derivflow (ledger since 08-08)

Twin ledgers, exact negations — at most one can pass.

| outcome | action |
|---|---|
| squeeze OR exhaustion PASSES | new P-entry: design a bounded expression (tilt/filter, never a standalone book) through the lab ladder first — a passing IC is a basis, not a strategy. |
| both fail | the liq_imbalance basis joins the dead list at THIS horizon; the strict-LOO attribution (P256) still stands for slower designs. |

## ~Sep 15 — volskip + etfflow (ledgers since 08-16)

| outcome | action |
|---|---|
| volskip PASSES | wire the vol gate into the ETH seat path IF the regimebook seat was enforced; otherwise it remains an overlay property of the shadow book. |
| etfflow PASSES | new P-entry: candidate for a BTC/ETH entry tilt (daily cadence). Its design-era backtest stays labeled hypothesis (reporting-lag caveat) regardless — only this forward read counts. |
| breadth books (XRP/ADA/LTC/DOGE/BNB) PASS collectively (majority positive, none catastrophic) | operator widening decision: extend sleeve assets + routing + per-asset caps + stops (P141 activation, one asset first per the P197 rollout pattern). Volumes are thin — check depth before sizing even ±1ct. |

## What does NOT happen in September

- No TQC/supervised retrain on the price-feature basis (settled: 0/39 folds
  across three campaigns, P258/P263).
- No promotion on a trajectory read, a partial window, or an in-sample
  number (P147/P198).
- No weakening of any P166 criterion to admit a near-miss — a near-miss
  waits for its next 30 days.

## Failure-of-everything branch (pre-committed, so it is not improvised)

If the tripwire fires on all three assets AND every candidate fails: the
system converges to flat/defensive — trend/hold books in shadow, capital
preserved, ~zero fees. That is the honest state, not a malfunction. The
next moves from there are (a) new information bases through the same ladder
(the ETF/liquidation families have successors: per-ETF dispersion, OI-basis
interactions), (b) breadth via the P262-certified assets if their forward
books hold up on a longer window, (c) an operator capital/venue decision.
What does NOT happen from there: re-litigating the dead model families or
loosening gates to make something trade.
