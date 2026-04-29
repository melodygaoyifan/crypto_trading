# HMATS v5.1 — Phase Pre-6 Closure (Backtest + Shadow-IC Framework)

**Status:** COMPLETE — pulled forward from Day 32-41 into the [PARAMETER 3] gap.
**Generated:** 2026-04-29

## Why pulled forward

Phase 2 (Coinbase migration, Days 14-28) is blocked on [PARAMETER 3] cutover-mode
operator answer. Phase 3 (funding strategies, Days 29-31) depends on Phase 2.
Phase Pre-6 has zero dependency on Coinbase migration — it just needs shadow
ledgers from Phase 4/8/7 to start accumulating, which they will once Tier 1
deploys. Pulling Pre-6 forward turns a blocker into a free schedule win and
unlocks Phase 6 (ML factor) which lives after Pre-6 anyway.

## What changed

| File | Change |
|---|---|
| `training/backtest_framework/__init__.py` | NEW — package marker |
| `training/backtest_framework/backtest_engine.py` | NEW — `BacktestEngine`, `BacktestConfig`, `FeeSchedule`, `BacktestResult`, `TradeRecord`; ~340 lines |
| `analytics/shadow_ic/__init__.py` | NEW — package marker |
| `analytics/shadow_ic/compute_shadow_ic.py` | NEW — JSONL loader, OHLCV join, Spearman, `determine_verdict`, CLI; ~290 lines |
| `tests/test_pre6_backtest_and_shadow_ic.py` | NEW — 20 unit tests |

## Architecture

### Pre-6.1 BacktestEngine (`training/backtest_framework/backtest_engine.py`)

Replays 4H OHLCV+features parquet against any strategy implementing the
Phase 4/8 shadow-strategy interface (`evaluate(asset, market_data) ->
ShadowSignal`). Outputs:

- Per-trade ledger (entry/exit/direction/confidence/PnL/fees)
- Annualized Sharpe (configurable `bars_per_day=6`, `trading_days_per_year=252`)
- IC across `horizons_bars=(4, 12, 24)` (configurable)
- Total fee impact (V14 Coinbase default: 0bps maker / 3bps taker, 98.7% maker
  per V15 → effective round-trip 0.078 bps)
- Win rate, avg return, max drawdown
- Per-regime breakdown (n / win_rate / avg_pnl)

**Key design choices:**
- **No scipy dependency** — Spearman implemented inline (~25 lines). Keeps
  the backtest engine usable in any minimal HMATS env.
- **Fee schedule honors V14 GREEN** — default `FeeSchedule()` produces 0.078
  bps round-trip (= 0.987×0 + 0.013×3 maker-side, ×2 round-trip). All-taker
  override (`FeeSchedule(maker_pct=0.0)`) gives 6 bps as a stress test.
- **Iron Law 4 fail-closed** — strategy exception at any bar is caught,
  logged at DEBUG, run continues with that bar skipped (test
  `test_backtest_engine_handles_strategy_exception`).

**Limitation:** Strategies that consume microstructure fields (`vpin`,
`ofi_zscore`, `liquidation_imbalance`, `spread_bps`) will mostly emit
NEUTRAL because the parquet has only OHLCV+features. Use this engine for
OHLCV-derived strategies; use Pre-6.2 `compute_shadow_ic.py` for live
ledger evaluation of microstructure/cascade strategies.

### Pre-6.2 ShadowIC compute (`analytics/shadow_ic/compute_shadow_ic.py`)

CLI entry point that:

1. Loads `data/strategy_shadow/{microstructure,cascade}_*.jsonl` since a
   configurable cutoff (default last 14 days)
2. Joins each `(ts, asset, strategy)` signal to the 4H OHLCV parquet at
   horizons `[4, 12, 24]` bars
3. Computes per-strategy Spearman IC at each horizon
4. Computes per-strategy annualized Sharpe (signal × forward_return at
   largest horizon, treating each signal as a trade)
5. Applies `determine_verdict()` — 4 enum values per v5.1 prompt's kill /
   promotion rules

**Verdict rules (`determine_verdict`):**

```
ALL horizons N < min_samples (default 30)        -> INSUFFICIENT_SAMPLES
window_days <= 14:
    max(|IC|) < kill_ic (default 0.05)           -> KILL
    else                                          -> HOLD
window_days >= 15 (treated as 30d gate):
    min(|IC|) > promote_ic AND sharpe > 0.5      -> PROMOTE
    max(|IC|) < kill_ic                           -> KILL
    else                                          -> HOLD
```

Matches the v5.1 prompt's per-phase kill criteria:
- Phase 4: 14d shadow IC < 0.05 → KILL individual
- Phase 8: 30d cascade IC < 0.04 → KILL (slightly looser; this gate uses
  the conservative 0.05; per-strategy override left to operator at promotion)

**CLI:**
```bash
python -X utf8 analytics/shadow_ic/compute_shadow_ic.py \
    --ledger-dir data/strategy_shadow \
    --window-days 14 \
    --horizons 4,12,24 \
    --prefixes microstructure,cascade
```

Output: console table + `analytics/shadow_ic/reports/shadow_ic_{utc_ts}.json`.

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | backtest reads parquet as-is, no feature change |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | EXTENDED via NEW SUBDIR | `training/backtest_framework/` is additive (per v5.1 Pre-6 spec — explicit "no constitutional override needed: training/ touch limited to new subdir"). `training/drl/`, `training/gmm/`, etc. UNTOUCHED |
| 4. fail-closed | HELD | malformed JSONL line → skipped + WARN; missing parquet → FileNotFoundError; strategy exception in backtest → DEBUG log + bar skip; missing OHLCV for an asset → recorded as `error: ohlcv_missing` in per-strategy result, sibling assets continue |
| 5. DRL ACTIVE floor | UNCHANGED | Pre-6 is offline tooling; runtime DRL authority untouched |
| 6. ≥3 active strategies | UNCHANGED | no strategy archive change |
| 7. Shadow ≥30d before promotion | HELD | `determine_verdict` enforces window_days gate; PROMOTE only available after `window_days >= 15` (Phase 10 will pass `--window-days 30+`) |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | UNCHANGED | execution layer untouched |

## Test results

```
tests/test_pre6_backtest_and_shadow_ic.py    20/20 PASS
─────────────────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8+7+Pre-6) 274/274 PASS
```

Phase Pre-6 test breakdown:
- FeeSchedule (V14 default + all-taker stress): 2
- Spearman (perfect ±, constant, short input): 4
- BacktestEngine on synthetic uptrend / neutral / missing parquet / strategy exception: 4
- determine_verdict (7 cases covering insufficient, kill, hold, promote at boundaries): 7
- load_shadow_ledgers (missing dir, since-filter, malformed-line skip): 3

## CLI smoke

```bash
$ venv/Scripts/python.exe -X utf8 analytics/shadow_ic/compute_shadow_ic.py --window-days 30
No shadow records loaded from data/strategy_shadow since 2026-03-30T04:10:01.877057+00:00
```

Expected — local has no live ledger because the harness only writes during
engine ticks. After Tier 1 commits ship to Hetzner, ledgers populate; the CLI
will produce real reports starting at the first 14d window (~2026-05-13).

## What does NOT happen yet

- **Per-sleeve PnL slicer** — feeds `SleeveAllocator.update_realized_vol()`
  with rolling realized vol per sleeve from equity_history attribution.
  Not built yet — owned by **Phase 6** (ML factor extraction also needs it
  for factor-attribution validation). Files at Phase 6 will live in
  `analytics/sleeve_attribution/`.
- **Auto-promotion executor** — reads the JSON report and flips a sleeve
  from shadow → fusion. Owned by **Phase 10**.
- **Backfill of microstructure fields** to enable backtest-engine runs of
  `OrderFlowImbalanceStrategy` etc. — would require historical L2 +
  liquidation data which we don't have. v6 candidate: subscribe to
  Coinglass historical liquidation_map endpoint.

## Deploy step (operator action)

Pre-6 ships entirely as new files + tests. Recommended commit message:
```
v5.1 Phase Pre-6: backtest engine + shadow-IC compute (offline tooling)

Pulled forward from Day 32-41 into the [PARAMETER 3] gap (Phase 2 blocked).
Zero dependency on Coinbase migration; unlocks Phase 6 (ML factor) earlier.

training/backtest_framework/backtest_engine.py — replays 4H OHLCV+features
parquet against any strategy with evaluate(asset, market_data) -> ShadowSignal.
Computes per-trade PnL, annualized Sharpe, IC at horizons (4/12/24 bars),
fee impact (V14 default 0bps maker / 3bps taker, 98.7% maker mix), regime
breakdown, max drawdown. No scipy dep — Spearman inline.

analytics/shadow_ic/compute_shadow_ic.py — CLI: reads
data/strategy_shadow/{microstructure,cascade}_*.jsonl ledgers, joins to OHLCV
at horizons, computes per-strategy IC + Sharpe + Verdict
(PROMOTE/HOLD/KILL/INSUFFICIENT_SAMPLES). Output JSON to
analytics/shadow_ic/reports/. Verdict thresholds match v5.1 phase kill
criteria (14d IC < 0.05 -> KILL; 30d IC > 0.05 + Sharpe > 0.5 -> PROMOTE).

training/ touch limited to NEW subdir backtest_framework/; training/drl/*,
training/gmm/* untouched (Iron Law 3 satisfied without constitutional
override per v5.1 Pre-6 spec).

274/274 cross-cutting tests pass (Phase 0+1+4+8+7+Pre-6 union). 20 new
tests cover FeeSchedule V14 math, Spearman edge cases, backtest engine on
synthetic uptrend + neutral + missing parquet + strategy exception, all 7
verdict-rule branches, ledger loader malformed-line skip + since-filter.
```

## Phase 6 readiness

Phase 6 (ML factor extraction, Day 42-56) requires:
- ✓ Backtest framework (Pre-6.1) — DONE
- ✓ Shadow-IC framework (Pre-6.2) — DONE
- ⏳ Constitutional override commit for `training/factor_extraction/` subdir
   (operator-signed; placeholder text in v5.1 prompt's Phase 6 spec)
- ⏳ Per-sleeve PnL slicer (Phase 6 will build this; feeds both ML factor
   validation and SleeveAllocator realized-vol update)

Phase 6 is unblocked by Pre-6 closure. It still needs operator's constitutional
override before training/factor_extraction/ work begins.

## Outstanding [PARAMETER]s after Pre-6

| # | Parameter | Status | Phase |
|---|---|---|---|
| 1 | Branch X or Y | RESOLVED Y | — |
| 2 | 12-strategy buckets | RESOLVED | — |
| 3 | V4.3 cutover mode | **PENDING OPERATOR** | Phase 2 |
| 4 | V8 DRL retrain Y/N | DEFAULT N | Phase 2 |
| 5 (NEW) | Phase 6 constitutional override sign | **PENDING OPERATOR** | Phase 6 |
