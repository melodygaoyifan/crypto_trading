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

**WHICH READ GOVERNS — decided 2026-08-20, BEFORE any read (P332).** This was
unspecified until now, and `september_check.py` ran the per-asset read only.
That would have decided the whole roster on a clock that cannot fire: P293g
measured that a 30-day window can certify only IC ≥ 0.302 at the 16h horizon,
against an economic bar of ~0.13. P299 built `--pool-assets` for exactly this
and nothing had wired it in.

* For a family in `POOLABLE_FAMILIES` (declared same-rule, standardized per
  asset before pooling), the **POOLED** read governs promote/kill. Measured on
  the 2026-08-20 ledgers, pooling roughly triples n — ma_filtered 116 → 348,
  regimebook 101 → 573, regimebook_adj → 300.
* For any other family the **per-asset** read governs; merging genuinely
  per-asset claims is the P294 defect.
* The per-asset table is printed either way and is **diagnosis**: it shows
  WHERE a family works, which is what P307c needed to reconcile two labs that
  appeared to disagree. It cannot by itself promote a poolable family, and a
  pooled PASS with a per-asset breakdown that is positive on one asset and
  negative on the others is an era/asset-fragility finding to record, not a
  reason to promote (P243/P244).

Fixing this after seeing a verdict would be a goalpost move; that is why it is
decided here, 18–27 days before the earliest read.

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

## Sep 1 — the trend tripwire (P237) — **PRESCRIPTION RETIRED (P299), detection kept**

**[P382 brought to the P299 state.]** The original table told the operator
to **remove an asset from `trend_assets`** on 4/4 GATE-CLOSED. **Do NOT do
that.** P299 retired the prescription (`tripwire_check.py` no longer exits 3;
it prints "SUPERSEDED" and "do NOT edit trend_assets") for two reasons:
(a) its actuator no longer targets the decider — trend has not held the
DECIDE seat since 2026-08-17 (whale P293j, then the regimebook P298), so
removing an asset from `trend_assets` only removes the FALLBACK that covers
whale's silent ticks and hands them to Best-of-N, whose weights are modulated
by the never-validated `[SENT-SWITCH]` firing on ~47% of days (P293i) —
removing a measured-weak signal into an unmeasured one is not de-risking;
(b) the **P295 seat controller** (`scripts/seat_check.py`, Monday cron)
reaches the same verdict BY COMPARISON across candidates, and it is the one
instrument allowed to recommend a seat change (it never edits config; P141).

**Instrument (detection only):** `tripwire_check.py` on the weekly
slope-calibrator reports (crons 08-10, 08-17, 08-24, 08-31 → exactly 4
weekly-spaced reports by the date gate). A GATE-CLOSED streak is still real
evidence and is **fed into the seat decision**; the no-reports REFUSAL
(exit 2) is untouched — "cannot be evaluated" must never read as "not fired"
(P199).

| outcome | action |
|---|---|
| 4/4 GATE-CLOSED for an asset | **no config edit from this tool.** The streak is recorded evidence for the seat controller's comparison; the seat controller decides whether trend (now the fallback) keeps its role, by comparing candidates on the same window. |
| any report shows the gate honestly OPEN for an asset | noted; the 4-report window re-arms. |
| no reports readable | exit 2 refusal — not a verdict. |

(Status at pre-commit: 1/4 reports, all three assets armed. P361/P362
recorded the cron healthy through 08-17 with 08-24/08-31 still to come.)

## ~Sep 7 — ma_filter (ledger since 08-08)

**[P361, CORRECTED] This was written as "the single most likely promotion".
Four entries have overtaken that, and the flag has since been DISARMED —
so a PASS here is not a green light, it is the start of an argument.**

| entry | what it measured |
|---|---|
| P324 | NOT EARNED at the pre-committed bar (model_alpha disagreement −7.8/−31.2bps, t=−1.41/−1.38; the contrast fails too) |
| P337 | measured against the decider it *actually* filters (the regimebook, not the retired trend seat): its disagreements marked entries that did **BETTER** — contrast −10.0, t −0.72 |
| P348 | **no obtainable sample makes it significant** — this very read moves model_alpha's t from 0.73 to 0.79; |t|≥2 needs ~503 disagreements ≈ 2.8 years |
| P356 | **DISARMED** (`coinbase_ma_filter_enforce: false`) by operator instruction on that arithmetic |

The read still happens, because the ledger accrues whether or not enforcement
is on (P340) and evidence is free. What changed is the disposition on each
outcome.

| outcome | action |
|---|---|
| PASS (P166 bar on the `ma_filtered` ledger) | **do NOT simply set the flag.** A PASS must be reconciled with P337 (opposite sign against the right decider) and P348 (the sample cannot support it). If it survives that, re-arming is its own P-entry AND an operator decision (P141) — it is a live-money reversal of P356. |
| FAIL | confirms P324/P337/P348; the flag stays false, which is already the deployed state. No action. |

## ~Sep 17 — whale_filter (ledger since 08-18)

**[P361] This section did not exist.** `whale_filter_*.jsonl` had been
accruing every tick since 2026-08-18, `whale_filtered` is a registered
POOLABLE scorer family, and neither this tree nor the countdown mentioned it
— so its ledger would have grown unexamined. An evidence stream with no read
date is the P199/P230 gap this document exists to close, and it was open on
one of the two entry filters while its sibling had a full section.

Same standing as ma_filter, with one less entry against it (P337 measured
whale's disagreements as marking *no* difference rather than a better one):

| outcome | action |
|---|---|
| PASS (P166 bar on the `whale_filtered` ledger, pooled — it is in `POOLABLE_FAMILIES`) | reconcile with P324 (NOT EARNED) and P348 (unsettleable) first; re-arming `coinbase_whale_filter_enforce` is its own P-entry and an operator decision (P141), reversing P356. |
| FAIL | confirms P324/P348; flag stays false, the deployed state. No action. |

Note P352 changed what this ledger *records* (the claim became a SIGN rather
than a contract count, and the `whale_count` evidence floor arrived), so rows
before 2026-08-20 are not comparable with rows after — read the post-P352
window, never one spanning both.

## ~Sep 9 — regimebook raw + adjusted (ledgers since 08-10)

**[P382 brought to the P298/P299/P379 state.]** The table below was written
when `regimebook_mode` was `"off"`. Three things have moved since:
`regimebook_mode` has been **`"enforce"` since 2026-08-18 (P298, on the P297
six-year evidence)** — the book ALREADY holds the DECIDE slot and whale
defers to it; **P299 relabelled SOL `v1_trend_only`** (ETH's certified
trend-only book — the P250 deletion removed a leg SOL never certified, so
SOL's rows are scoreable, not "unavailable"); and **P379 ran the adjusted
book's out-of-sample validation read: OVERFIT** (adj loses net on ALL three
assets — BTC +0.628 vs raw +0.688, ETH +0.468 vs +0.500, SOL **−0.415** vs
+0.093 — despite winning in-design; the churn cut generalises, the timing
does not). So this read **confirms or disarms**; it no longer arms anything.

| outcome | action |
|---|---|
| raw book PASSES (pooled — `regimebook` is in `POOLABLE_FAMILIES`) | **confirms** the standing `regimebook_mode: "enforce"`. No config change. Record the pass. |
| raw book FAILS | a **DISARM decision**, not a non-event: the revert is `regimebook_mode: "shadow"` (harness keeps recording, zero live effect) + its own P-entry + operator decision (P141). **[P420 corrected]** What covers the ticks on disarm is NOT the whale seat — it is OFF (`whale_seat_mode: "off"`, P417; whale is a 0.10-weight ADVISE member now). The regimebook decides **SOL, XRP and BNB only**; BTC and ETH direction already comes from the **skew seat** (P407, calibrated P407e/g) with the **ETF seat** de-risking on outflow (P405/P414), both of which stay armed through a regimebook disarm. So a FAIL flattens the regimebook-decided assets to the trend fallback (measured GATE-CLOSED, so effectively FLAT) and leaves BTC/ETH on skew/ETF; the honest system on those three is mostly FLAT. |
| adjusted book PASSES and beats raw on the ledger | **stays OFF regardless.** P379 measured `regimebook_adj` OVERFIT out-of-sample on all three assets; a 30-day forward read cannot outrank a six-year out-of-sample FAIL (P243/P244 era-fragility). A pass here is recorded as a forward-vs-history disagreement worth a trace, never as an arming. |
| BTC passes but its funding legs' cells are the failing part | funding legs remain the UNCERTIFIED slice (P262; P297 certified them over six years on the INCREMENT with a CI that includes zero — the criterion that also rejects buy-and-hold). Record per-leg; a leg-level change to the enforced book is its own P-entry. |

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
| etfflow PASSES / FAILS | **[P420 corrected — the old row said "candidate for an entry tilt"; the ETF seat is already ARMED.]** `etf_seat_mode: "enforce"` with `etf_derisk_assets: [BTC, ETH]` and `etf_decide_assets: []` (P405, re-horizoned to de-risk-only by P414): a fresh strong ETF OUTFLOW flattens a held long, it never enters. So PASS = **confirm** the standing de-risk seat (no change; record it); FAIL = a **DISARM decision** (`etf_seat_mode: "off"`, its own P-entry + operator, P141). A directional ETF seat is NOT re-opened by a PASS here — P414 removed it as under-evidenced and the research says ETF flow is coincident context, not next-bar alpha; its directional value lives only in the agree-gated combiner read below (`skewetf`). The `etfflow_timing_check` cron (root, Mondays) must read leak-free before either outcome is acted on. |
| breadth books PASS / FAIL | **[P420 corrected — the old row prescribed "extend sleeve assets + routing, one asset first"; that already happened.]** XRP and BNB have been ROUTED and TRADING since P412b/c (1ct each, measured preview fees, calibrated seat alpha); ADA and LTC were **REJECTED by measurement** (negative / sub-friction median seat alpha, P412c) and DOGE was **HELD BACK by operator decision** (validation era −93.8 at 2x cost, P412c). So this read is per-asset, and it never re-opens the rejected three: XRP/BNB PASS = confirm (no change); XRP/BNB FAIL = a per-asset **de-routing decision** (`scripts/coinbase_set_assets.sh BTC,ETH,SOL` minus the failing one; P141) with its own P-entry; ADA/LTC/DOGE any result = recorded only — routing them is the P412c decision re-opened, which needs the measured seat alpha to change first, not a forward IC. See the `regimebook_breadth` section below for the ledger this reads. |

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

## The sizing ladder (P291, pre-committed 2026-08-17; arithmetic corrected P420)

> **Governing sentence: size follows certification, never precedes it.**

**[P420] The numbers below replace the 0.15 × 3 = 0.45 that stood here — that
book has not existed since P370/P412c.** Live `coinbase_target_fraction_by_asset`
is `{BTC .20, ETH .15, SOL .095, XRP .01, BNB .005}` (vol-parity for the trio,
P370; tiny breadth fractions, P412/P412c), `post_leverage_caps` BTC/ETH 0.25,
SOL 0.20, XRP/BNB 0.10, the P208 sleeve net cap **0.50** (strict `>`), P274 ctor
clamp ≤ 0.25.

**The nominal fraction sum is 0.46 — and it is NOT what the book holds.**
`_sized_contracts` = `max(1, int(fraction × equity / one-contract notional))`:
it FLOORS at one contract, and one XRP contract (500 XRP ≈ $700) or one BNB
contract (1 BNB ≈ $700) is **~6.4% of a ~$10.9k sleeve each**, not 1% / 0.5%.
The P412 "0.455 < 0.46 headroom" claim reasoned in fractions the floor ignores
and is off by ~12pp on the breadth pair alone. Real arithmetic, all live assets
the same way (nominal → floored contracts → realized share of equity, at the
late-August marks — the shares move with price and equity):

| asset | fraction | per-contract notional | contracts (int-floor, min 1) | realized share |
|---|---|---|---|---|
| BTC | 0.20 | ~$765 (0.01 BTC) | 2 | ~14% |
| ETH | 0.15 | ~$241 (0.1 ETH) | 6 | ~13% |
| SOL | 0.095 | ~$467 (5 SOL) | 2 | ~9% |
| XRP | 0.01 | ~$705 (500 XRP) | **1 (floor)** | **~6.5%** |
| BNB | 0.005 | ~$700 (1 BNB) | **1 (floor)** | **~6.4%** |
| **all long** | 0.46 nominal | | | **~49–50%, at the 0.50 net cap** |

So the fully-sized all-long book sits **at** the net cap with zero headroom; a
small price move in the rounding direction takes it over, and the P208 gate then
BLOCKS whichever asset's entry arrives last (never flattens — de-risking is
always free). That is a scheduled sizing fault, not a signal fault, and it is
what a fraction step-up would compound. Because the trio's realized shares are
themselves int-floored (2/6/2ct) the sum depends on price: state it from
`sleeve_exposure()` at decision time, never from the fraction sum.

| September outcome | pre-committed fractions | arithmetic |
|---|---|---|
| nothing certifies | **unchanged** | the all-long book already touches the 0.50 cap (table above); no step-up is available |
| an asset certifies, **all five still live** | **unchanged** | any step-up on a ~49–50% book breaches the net cap. Funding it by trimming an uncertified asset — or by de-routing a breadth asset whose 1ct floor is the headroom eater — is a *second* decision about assets whose evidence did not change; not pre-approved here. |
| a breadth asset is de-routed (its book FAILS) | the freed ~6.4% may fund **one** step of 0.05 on a certified trio asset, within its `post_leverage_caps` | e.g. BTC 0.20 → 0.25 (its cap binds), or SOL 0.095 → 0.145 (under its 0.20 cap); re-check `sleeve_exposure()` ≤ 0.45 after the step |
| the seat controller / tripwire removes **one** trio asset | the freed share may step **one** certified survivor 0.05 (P291: survivors 0.20 at the old 0.45 book; now bounded by `sleeve_exposure()`) | check `sleeve_exposure()` ≤ 0.45 after the step, never the fraction sum |
| the seat controller / tripwire removes **two** trio assets | the survivor may step to its own `post_leverage_caps` value (P291) | e.g. BTC 0.20 → 0.25; the breadth pair keeps its 1ct floor |
| all three trio assets certify | **unchanged** | with all five live the book is at the cap (row 2); certification does not create headroom |
| everything fails | **unchanged** | a flat book's size is moot; the seat question dominates |

Every step-up is its own recorded P-entry (P141) and may never exceed the
asset's `post_leverage_caps` value or the 0.25 ctor clamp. **A fraction is
never raised to make something trade** — only after that asset's signal has
certified, which is the same rule that governs every other lever in this
document. And the breadth floor is the constraint to state first: a "tiny"
fraction on a ~$700 contract is not tiny.

## ~Sep 24 — skewetf: the live skew decider's own A/B (ledger since 2026-08-25, P407j)

**[P420] Unscheduled until now.** `defense/skew_etf_combo_shadow.py` records
three claims per tick for BTC/ETH: `skewetf_skew` (skew solo = the LIVE
override), `skewetf_etf` (ETF solo), `skewetf_agree` (skew iff skew == etf ≠ 0
— the combiner P414 pinned EQUAL-WEIGHT, never learned). P407i measured the
agree-gate better than skew-override on every 1.7y cut, thin (15–30 trades),
and deliberately did NOT flip the live precedence on that window. This forward
read is the certification it was waiting for.

| outcome | action |
|---|---|
| `skewetf_agree` PASSES the P166 bar AND beats `skewetf_skew` on the same window (minimum-horizon IC, the P291 comparative rule) | new P-entry: flip the BTC/ETH seat precedence from skew-override to agree-gated (the skew seat still decides, the ETF seat's disagreement now HOLDS instead of being overridden). Size/caps/stops/gate unchanged — a direction-source choice, not a risk addition. Operator decision (P141). |
| `skewetf_skew` passes and `agree` does not | confirms the standing override; record it. |
| neither passes | the live skew seat's forward IC is failing its own exam — that is a **DISARM candidate** for `skew_seat_mode` (P141), not a reason to try `agree`. |

## ~Sep 26 — convsize: the WS2 conviction-sizing shadow (ledger since 2026-08-27)

**[P420] Unscheduled until now, and NOT a shadow-IC candidate.** A sizing
overlay is judged on PnL increment over the 1x base, not rank-IC, so its reader
is `scripts/conviction_sizing_review.py` (forward net/maxDD of the
conviction-sized book vs the 1x trend base over the ledger's own closes,
`data/conviction_shadow/convsize_{BTC,ETH}.jsonl`; exit 2 below 30 rows). The
channel was turned OFF by P417's pre-committed 6.6y verdict (pure holding beats
every conviction variant); this ledger is the forward side of that same
question and can only RE-ARM it.

| outcome | action |
|---|---|
| increment > 0 on BOTH assets at ≥ 30 rows AND the P417 lab's verdict does not contradict it on re-run | new P-entry: `fusion_conviction_to_sleeve: true` is a recorded operator arming (P141), with the P416 resize persistence still in force |
| increment ≤ 0 on either | confirms P417; the channel stays off. No action. |
| < 30 rows | no verdict (P199/P348) — wait, do not read a thin ledger |

## regimebook_breadth — the breadth books' own family (ledger since the P420 deploy)

**[P420] Unscheduled until now.** Before P420 the breadth books (XRP/ADA/LTC/
DOGE/BNB, P271) wrote rows under the shared `regimebook` name and their price
series was a Kraken window `september_check.py` MERGED INTO the calibration
input (which is how the live XRP/BNB seat-alpha constants drifted a day after
they shipped). From the P420 deploy the rows carry `strategy:
"regimebook_breadth"` and the scorer reads `{ASSET}_4H_ohlcv_kraken.parquet` as
an EXTENSION of the Binance primary (union, primary wins). The 30d clock starts
at that deploy; `september_check.py --countdown-only` prints it as
`(clock unstarted)` until the date is filled in. Actions are the per-asset rows
in the ~Sep 15 breadth table above.

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
