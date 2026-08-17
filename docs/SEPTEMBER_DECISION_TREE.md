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

> **SUSPENDED same day (P285b, 2026-08-16):** the 10-seed robustness probe
> returned **FRAGILE** — median decisive Sharpe −0.09 across seeds 1..10,
> the certified seed 7 the second-best draw. The certified pooled pass is
> substantially seed luck, which removes the criterion's premise (that the
> candidate's history certification outranks the incumbent's). The
> suspension is MECHANICAL: `mlp_seat_check.py` refuses (exit 2) while the
> FRAGILE report stands un-answered. Re-arming requires the **P285c
> 10-seed ensemble certification to PASS** (CI excludes zero on the same
> decisive cell), followed by a recorded export-swap P-entry that restarts
> the shadow clocks for the ensemble. If the ensemble FAILS, mlp_small is
> dead as a candidate (pre-committed in `mlp_ensemble_cert.py`), the
> early seat is withdrawn outright, and the shadow ledger keeps accruing
> as free evidence only.
>
> **WITHDRAWN same day (P285c):** the ensemble certification FAILED —
> mlp_ens10 pooled Sharpe +0.49, CI[−0.72, +1.65] includes zero, DSR 0.29.
> The pre-committed disposition executed: **mlp_small is dead as a
> candidate; this early-seat section is void.** The seed-7 shadow ledger
> keeps accruing as free evidence (its ~09-15 P166 read would now be a
> surprise, not a confirmation); the seat actuator stays wired but the
> checker refuses permanently under FRAGILE + ensemble-FAIL. The trained-
> model verdict on the current data basis is restored to the P281 arc's
> conclusion: dead at every rung under fully honest conditions.

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

## ~Sep 16 — the trend-rule challengers (P288 lab, P289 ledgers, P291 criteria)

**What is new here.** P288 ran the one design axis SMA200 had never faced —
its own rule family — and produced the campaign's first dethroning:
DONCHIAN-100 and EMA-ENSEMBLE beat the incumbent at ~1/4 its turnover,
era-stable on BOTH lab windows, where no fitted model ever was. P289 wired
their forward ledgers (first live 2026-08-17 → 30d read ~2026-09-16).

**Instrument:** `scripts/challenger_seat_check.py` (operator-local, after a
`september_check.py` pull). Pre-committed 2026-08-17, before any forward
evidence existed.

| condition (ALL required, per `(strategy, asset)` CELL) | value |
|---|---|
| date | ≥ 2026-09-16 |
| **lab precondition** | the cell must be one P288 actually dethroned |
| ledger | ≥ 30 days span, ≥ 20 directional (non-flat) records |
| forward bar | the scorer's own verdict is `PROMOTE` — consumed, never re-derived |
| beats the incumbent | the cell's **minimum-horizon** IC > `regimebook_{ASSET}`'s minimum-horizon IC on the same window |

**The lab precondition is the sharpest edge, and "ETH+SOL" is not precise
enough to state it.** Per cell, the dethroned set is exactly
`donchian/ETH`, `donchian/SOL`, `emaens/SOL` — **`emaens/ETH` is NOT eligible**
(design −0.231 against the incumbent's +0.088), and **no BTC cell is**
(SMA200 stood: donchian +0.582, emaens +0.467 vs +0.594). A non-dethroned cell
may PASS its forward bar and still have no seat claim; promoting it would be
selection on one forward window against design-era evidence that already
rejected it.

**Why "beats the incumbent" is a separate condition from the P166 bar:** the
P166 gate certifies a signal against *costs*. A seat swap is *comparative* —
it must also beat the signal already in the seat, on the same bars. For ETH
and SOL the `regimebook` ledger IS the SMA200 trend/hold book (P250), so the
comparison is like-for-like; BTC's carries funding legs and is not, which
costs nothing because no BTC cell is eligible anyway.

| outcome | action |
|---|---|
| a cell clears all five | **exit 3 = ELIGIBLE, not "fire".** A seat swap is an operator **risk-preference** decision with its own P-entry: the challenger replaces the SMA200 labeler for that asset in the regimebook seat path, at unchanged size, caps, stops and gates. |
| no cell clears | ledgers keep accruing; re-read at 60d only if a trajectory is monotone. |

> **The pre-committed caveat that makes this a preference and not a
> threshold (P288 virgin-era probe = PARTIAL for both challengers):** they
> pass the breadth leg 5/5 and beat SMA200 on every out-of-selection TOTAL,
> but they FAIL the property the incumbent's certification is built on and
> the live book is deployed FOR — the BTC-2018 crash-dodge (DONCHIAN −0.60,
> EMA-ENSEMBLE −0.51 vs the incumbent's −0.28, bar −0.35). Their hysteresis
> exits trends later: **more upside captured, ~2× deeper bear-year
> drawdown.** No instrument can settle a preference, so the checker names the
> trade-off and stops.

## The sizing ladder (P291, pre-committed 2026-08-17)

> **Governing sentence: size follows certification, never precedes it.**

What happens to `coinbase_target_fraction_by_asset` on each September
outcome, fixed in advance so it is not improvised on the day. Constraints:
live fractions **0.15 / 0.15 / 0.15**; `post_leverage_caps` BTC 0.25, ETH
0.25, SOL 0.20; the P208 sleeve net cap **0.50** (gate is strict `>`, so 0.50
is *not* blocked but leaves zero headroom); P274 clamps any fraction to
≤ 0.25 in the ctor. Nominal max net = the sum of fractions when all live
assets point the same way.

**Today: 0.15 × 3 = 0.45 nominal max net** — that already consumes the whole
prudent budget under the 0.50 cap.

| September outcome | pre-committed fractions | arithmetic |
|---|---|---|
| nothing certifies | **unchanged** 0.15/0.15/0.15 | 0.45 ≤ 0.50 ✓ |
| one asset certifies, **all three still live** | **unchanged** | 0.20+0.15+0.15 = **0.50 — exactly at the cap, zero headroom**. Funding a step-up by trimming the two uncertified assets is a *second* decision about assets whose own evidence did not change; it is not pre-approved here. |
| tripwire removes **one** asset (it goes flat) | the two survivors may step 0.15 → **0.20** | 0.20+0.20 = 0.40 ≤ 0.45 ✓; SOL's own 0.20 cap is exactly binding, BTC/ETH sit under 0.25 |
| tripwire removes **two** assets | the survivor may step 0.15 → its per-asset cap (BTC/ETH **0.25**, SOL **0.20**) | max net = that fraction ✓, and ≤ the P274 ctor clamp |
| tripwire fires on all three / everything fails | **unchanged** | a flat book's size is moot; the seat question dominates |

Every step-up is its own recorded P-entry (P141) and may never exceed the
asset's `post_leverage_caps` value or the 0.25 ctor clamp. **A fraction is
never raised to make something trade** — only after that asset's signal has
certified, which is the same rule that governs every other lever in this
document.

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
