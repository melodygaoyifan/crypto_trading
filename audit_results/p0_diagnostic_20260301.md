# P0 DIAGNOSTIC REPORT — 2026-03-01

> **Mode:** READ-ONLY diagnostic (no code modified)
> **Scope:** 6 P0 areas across main.py (13,727 lines), constitution.py, integration_v36.py, exit_alpha.py, leverage_guard.py, unified_position_sizer.py, thesis_budget_governor.py

---

## VERDICT SUMMARY

| P0 | Area | Verdict | Priority |
|----|------|---------|----------|
| P0-1 | Exit Alpha Attribution | **EXIT_ALPHA_TRACKED** | Monitor |
| P0-2 | Regime Power Calibration | **CALIBRATED** | OK |
| P0-3 | OPPORTUNITY Triggers | **PARTIAL_ACTIVE** | This week |
| P0-4 | Veto Chain Multiplier Floor | **FLOOR_EXISTS (0.15)** | OK |
| P0-5 | $10K Equity Chain | **EQUITY_CORRECT (95%)** | Minor fix |
| P0-6 | Leverage x Power Interaction | **EXPOSURE_SAFE** | OK |

---

## P0-1: EXIT ALPHA ATTRIBUTION

### Verdict: EXIT_ALPHA_TRACKED

### Evidence

**6 active exit paths + 1 DRL path (structurally defined, not connected):**

| Exit Path | Tag | Location | Status |
|-----------|-----|----------|--------|
| MAX_HOLD timeout | T7_MAX_HOLD | main.py:4261-4318 | EXISTS |
| Soft stop | T10_SOFT_STOP | main.py:6645-6665 | EXISTS |
| Gambler exit | T9_GAMBLER | main.py:6668-6699 | EXISTS |
| Adaptive trailing stop | T1_TRAILING_STOP | main.py:6703-6738 | EXISTS |
| Exit Alpha (5 sub-triggers) | T3/T4/T5/T12 | main.py:6746-6832 | EXISTS |
| Runner release | T11_RUNNER_RELEASE | main.py:6835-6877 | EXISTS |
| Direction flip | T13_FLIP | main.py:11280-11358 | EXISTS |
| DRL action | T6_DRL_ACTION | exit_alpha.py:306 | **PARTIAL** |

**ExitAlphaTracker** (analytics/exit_alpha_tracker.py):
- `record_entry()` — price, direction, regime at open
- `update_peak()` — MFE tracking (peak_unrealized_bps, peak_price)
- `record_exit()` — trigger classification, retention_pct, peak_to_exit_bps
- `update_counterfactual()` — 12-tick post-exit tracking (left money? avoided loss?)
- Classification: `by_trigger`, `by_asset`, `by_regime`
- Persistence: `data/exit_alpha_tracking.jsonl`, `data/exit_counterfactual.jsonl`
- Singleton: `get_exit_alpha_tracker()` (line 458)
- Wired in main.py at EA-4 locations (L6800, L6859, L11348-11358)

**5 Scale-Out Triggers** (execution/exit_alpha.py):
1. `PHASE_TRANSITION` (L258-271) — Regime to SATURATION/EXHAUSTION
2. `CRACK_DECAY` (L273-288) — CRACK weight below threshold
3. `MOMENTUM_STALL` (L290-300) — Momentum flatlines for N bars
4. `DRL_ACTION` (L303-312) — **Defined but `drl_output=None` always passed**
5. `DRAWDOWN_FROM_PEAK` (L314-326) — Gave back >30% of peak profit

**Runner Management** (exit_alpha.py L365-498):
- `create_runner()` — 75% of position becomes runner with 3% initial trail
- `manage_runner()` — HOLD/TIGHTEN/RELEASE based on phase + trail stop
- Trail: initial=3%, tight=1.5% (configured main.py:3209-3220)

### Issues Found

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| DRL exit signals not connected | LOW (DRL DISABLED) | main.py:6793, 6855 | Wire `agent_signals.get('drl_exit_action')` when DRL activates |
| Shadow ledger lacks trigger metadata | LOW | shadow_ledger/*.jsonl | `_exit_trigger_tag` is in-memory only, not persisted to JSONL |

### Recommendation
- **No immediate action** — DRL is DISABLED, so `drl_output=None` is correct behavior
- When DRL activates: wire `drl_output` parameter to enable T6_DRL_ACTION trigger
- Consider persisting `_exit_trigger_tag[asset]` in shadow ledger FILL records for post-hoc analysis

---

## P0-2: REGIME POWER CALIBRATION

### Verdict: CALIBRATED

### Single Source of Truth: main.py:1541-1560

```
REGIME_POWER_MULTIPLIERS:
  STRONG_TREND:        1.5
  TREND:               1.2
  BULL_TREND:           1.2
  BEAR_TREND:           1.0
  UNCERTAIN:            0.8    # [UNLEASH] was 0.6
  MEAN_REVERT:          0.5    # [UNLEASH] was 0.3
  TRANSITION:           0.3
  SIDEWAYS:             0.5
  UNKNOWN:              0.75   # [VC-6] was 0.5
  ---- GMM 6-regime ----
  MOMENTUM_RALLY:       1.3    # was 1.5
  QUIET_ACCUMULATION:   0.8    # was 0.5
  PANIC_SELLOFF:        1.2    # was 0.2
  VOLATILE_CHOP:        1.5    # was 0.3, 71% WR +503%
  EXTREME_VOLATILITY:   0.5    # was 0.1
  WEAK_CONSOLIDATION:   0.6    # was 0.5
  STEADY_UPTREND:       0.75   # TODO calibrate
  NEUTRAL_DRIFT:        0.75   # TODO calibrate
```

### VC-4 Merge (NOT multiplicative)
```
Line 7860: _vc4_merged = max(regime_multiplier, _ra_size_mult) if _ra_size_mult != 1.0 else regime_multiplier
```
- MOMENTUM_RALLY: max(1.3, 0.70) = **1.3** (aggression ignored)
- PANIC_SELLOFF: max(1.2, 1.15) = **1.2** (power wins)
- VOLATILE_CHOP: aggression=1.0 (neutral) -> **1.5** (power applies)

### SIG-4 Direction Scaling (main.py:7890-7925)

Applied AFTER VC-4 merge, SHORT only:

| Regime | Power Mult | Leverage Cap | Logic |
|--------|-----------|--------------|-------|
| PANIC_SELLOFF | x1.2 | None | SHORT is pro-trend, boost |
| MOMENTUM_RALLY | x0.5 | 1.0x | SHORT is counter-trend, reduce+cap |
| STEADY_UPTREND | x0.5 | 1.0x | SHORT is counter-trend |
| VOLATILE_CHOP | x1.0 | None | Neutral |
| All others | x1.0 | None | No change |

### Scale-In Gate (main.py:1563-1582)

Separate boolean per regime (gates T1->T2+ escalation):
- **Allowed**: STRONG_TREND, TREND, BULL_TREND, MEAN_REVERT, MOMENTUM_RALLY, PANIC_SELLOFF, VOLATILE_CHOP
- **Blocked**: UNCERTAIN, BEAR_TREND, TRANSITION, SIDEWAYS, UNKNOWN, QUIET_ACCUMULATION, EXTREME_VOLATILITY, WEAK_CONSOLIDATION, STEADY_UPTREND, NEUTRAL_DRIFT

### Conflicts Found
- **sota_config.json** has stale values (UNCERTAIN=0.6 vs 0.8, MEAN_REVERT=0.3 vs 0.5, UNKNOWN=0.5 vs 0.75)
- **Impact: NONE** -- file is marked `_DEPRECATED` and never loaded by runtime code

### Recommendation
- **No action needed** -- single source of truth, no conflicts, backtest-driven values
- STEADY_UPTREND and NEUTRAL_DRIFT marked `TODO: calibrate` -- calibrate after 2 weeks of data

---

## P0-3: OPPORTUNITY TRIGGER VERIFICATION

### Verdict: PARTIAL_ACTIVE

### Trigger Group Status

| Group | Trigger | Data Source | Status | Threshold |
|-------|---------|------------|--------|-----------|
| A | Lead-Lag Edge | Binance WS + Kraken | **WIRED_NEVER_FIRES** | 25bps + 0.65 conf + CVD 0.6 + VPIN<0.80 |
| B | CRACK Window | Pipeline structure_break_pct | **WIRED_NEVER_FIRES** | 2% break + 2.5x vol + 0.4 imbalance |
| C | DVOL Expansion | Deribit (NOT CONNECTED) | **DEAD** | 2.5 <= dvol_zscore <= 4.0 |
| D | Sentiment Shock | F&G/LLM (NOT WIRED) | **DATA_MISSING** | abs(shock_sigma) >= 2.0 |
| E | SOL Flow Surge | OnChain + Whale detection | **WIRED_NEVER_FIRES** | 3x DEX vol OR $50M inflow OR whale active |

### Detailed Analysis

**Group A (Lead-Lag):** Constitution.py L588-710. All 4 conditions must be true simultaneously (edge + confidence + CVD + VPIN). Lead-lag engine at `market/lead_lag_engine.py` requires Binance WS (offline) + Deribit WS (offline). Running in SHADOW authority mode.

**Group B (CRACK):** Constitution.py L596-748. Requires 2% structure break + 2.5x volume spike + aligned orderbook imbalance. `structure_break_pct` IS populated from pipeline. Threshold combination is rare but valid.

**Group C (DVOL):** Constitution.py L603-767. `dvol_zscore` defaults to 0.0 everywhere because Deribit is not connected. This trigger can NEVER fire. The 2.5-4.0 band is intentionally narrow (>5.0 = NO_TRADE).

**Group D (Sentiment):** Constitution.py L608-787. main.py passes `sentiment_data={}` (empty dict) to `compute_opportunity_triggers()`. The `_check_sentiment_shock()` method is never called because dict is falsy.

**Group E (SOL Flow):** Constitution.py L612-835. Recently enhanced with whale detection (SIG-1 fix). Now has 3 OR conditions: DEX ratio >= 3.0, |inflow| >= $50M, or whale_dir in ("buy","sell"). OnChain feed exists but in COMPOSITE mode.

### Density Fallback
- integration_v36.py L1475-1478: `phase_result.opportunity_density >= 0.88` -> OPPORTUNITY
- This is the ONLY viable path to OPPORTUNITY mode currently
- Has NEVER fired in paper run (phase density never reached threshold)

### Phase-Based IGNITION
- Alternative path to OPPORTUNITY via phase detector IGNITION phase
- integration_v36.py L1467-1472

### Shadow Ledger Evidence
- All INTENT records show `mode="PAPER"` -- never `"OPPORTUNITY"`
- Zero OPPORTUNITY events in paper trading history

### Issues Found

| Issue | Severity | Fix |
|-------|----------|-----|
| Group C DVOL permanently dead | MEDIUM | Wire Kraken Futures DVOL OR remove trigger |
| Group D sentiment data empty | LOW | Wire F&G zscore to `sentiment_data` dict |
| OPPORTUNITY never fires | HIGH (profit blocker) | Threshold review + data source activation |

### Recommendation
- **This week**: Wire `dvol_zscore` from Kraken Futures feed (already have premium/mark data). Remove dependency on Deribit.
- **This week**: Wire `sentiment_zscore` to Group D's `shock_sigma` parameter
- **Threshold review**: Group E's $50M inflow threshold may be too high. Consider lowering to $10M.
- **Root cause**: This is a DATA INFRASTRUCTURE problem, not a threshold-tuning problem

---

## P0-4: VETO CHAIN MULTIPLIER FLOOR

### Verdict: FLOOR_EXISTS (0.15)

### Floor Implementation: main.py:8584-8608

```python
_VC5_CUMULATIVE_MULT_FLOOR = 0.15   # Line 8584
_VC5_MIN_NOTIONAL_USD = 50.0        # Line 8585

if (intent.is_actionable and not intent.veto_active
        and _vc0_original_exposure > 0 and abs(intent.target_exposure) > 0):
    _vc5_current = abs(intent.target_exposure)
    _vc5_floor = _vc0_original_exposure * _VC5_CUMULATIVE_MULT_FLOOR
    if _vc5_current < _vc5_floor:
        intent.target_exposure = _vc5_floor * _vc5_sign   # Line 8597
```

### Floor Mechanics
- **Baseline captured at**: main.py:6431 as `_vc0_original_exposure` = raw engine.decide() output
- **Floor calculation**: 15% x engine_exposure
- **Placement**: AFTER all 16 soft multipliers, BEFORE hard caps
- **Secondary gate**: $50 USD minimum notional (L8600) -- if below, trade is VETOED
- **Sign-preserving**: `_vc5_sign` preserves long/short direction (L8591)

### All 16 Soft Multipliers (in application order)

| # | Name | Location | Min | Max | Variable |
|---|------|----------|-----|-----|----------|
| 1 | P0 Exposure | L7279 | 0.0 | 1.0 | `p0_exposure_multiplier` |
| 2 | Correlation Adj | L7289 | 0.0 | 1.0 | `correlation_exposure_adj` |
| 3 | Corr RT Control | L7300 | 0.0 | 1.0 | `_crt_combined` |
| 4 | SOL Toxicity | L7667 | 0.0 | 1.0 | `_tox_mult` |
| 5 | Thesis Budget Reentry | L7772 | ~0.5 | 1.0 | `budget_result.reentry_size_multiplier` |
| 6 | SIG-4 Direction | L7914 | 0.5 | 1.2 | `_sig4_pmult` |
| 7 | Regime Power (VC-4) | L7861 | 0.0 | 1.5 | `_vc4_merged` |
| 8 | TQC Discount | L7996 | 0.5 | 1.0 | `(1.0 - _tqc_discount)` |
| 9 | Profit Max | L8129 | 0.3 | 1.0 | `profit_max_result.size_multiplier` |
| 10 | WIRE-3 Confidence | L8172 | 1.0 | 4.0 | `_w3_mult` (AMPLIFIER) |
| 11 | WIRE-3b Carry | L8211 | 1.0 | 1.3 | `_w3b_boost` (AMPLIFIER) |
| 12 | WIRE-G6b Weight | L8302 | 0.3 | 1.5 | `_g6b_w` |
| 13 | G6 Portfolio | L8327 | 0.5 | 2.0 | `_g6_mult` |
| 14 | WIRE-G2 Carry | L8367 | 0.90 | 1.15 | `(1.0 + _g2_boost)` |
| 15 | WIRE-VA Suppress | L8409 | 0.5 | 1.0 | `_va_suppress` |
| 16 | WIRE-VA Boost | L8420 | 1.0 | 1.3 | `_va_boost` (AMPLIFIER) |
| **FLOOR** | **VC-5** | **L8584** | **0.15x** | -- | **Post-mult guarantee** |

### Hard Vetoes (~30 total)
Separate from soft multipliers. Hard vetoes set `intent.veto_active = True` which SKIPS the floor check entirely (floor only protects against cascading soft reduction, not hard vetoes).

### Worst-Case Scenario
```
Engine decides: 0.10
All min multipliers: 0.0 × 0.0 × ... → effectively 0.0
WITH VC-5 FLOOR: max(~0, 0.15 × 0.10) = 0.015 (15% of original)
$50 notional check: 0.015 × $10K = $150 > $50 → PASSES
```

### Recommendation
- **No action needed** -- floor exists, correctly placed, sign-preserving, with secondary $50 notional gate

---

## P0-5: $10K EQUITY CHAIN

### Verdict: EQUITY_CORRECT (95%)

### Authoritative $10K Sources (all correct)

| File | Line | Value |
|------|------|-------|
| .env | 146 | `HMATS_INITIAL_CAPITAL=10000` |
| configs/canonical_config.py | 39 | `INITIAL_CAPITAL = 10_000.0` |
| configs/cloud_production.json | 38 | `"initial_capital_default": 10000.0` |
| core/cloud_config.py | 85 | `initial_capital: float = 10000.0` |
| core/account_sync.py | 142 | `self._dry_run_equity = 10000.0` |
| main.py | 978 | `INITIAL_CAPITAL = 10_000.0` (fallback) |

### Resolution Chain (correct priority)
```
1. ENV: HMATS_INITIAL_CAPITAL=10000  (highest priority)
   ↓ (if not set)
2. JSON: cloud_production.json "initial_capital_default"
   ↓ (if missing)
3. CODE: canonical_config.INITIAL_CAPITAL = 10_000.0
```

### Position Sizing (CORRECT)
- Uses LIVE NAV via `account_sync.get_equity()` (main.py:9600-9611)
- Formula: `risk_amount = account_balance * risk_per_trade / stop_loss_pct` (percentage-based)
- Compound cap: `account_balance = min(equity, initial_capital * MAX_SIZING_EQUITY_MULTIPLIER)`

### Drawdown (CORRECT)
- Tracks running peak: `self._peak_equity = max(self._peak_equity, current_equity)` (main.py:12598-12607)
- Formula: `drawdown_pct = (peak - current) / peak`

### Stop Loss (CORRECT)
- Percentage-based: 2% default (not fixed USD)
- Scales with equity automatically

### $100K Residuals Found

| File | Line | Context | Impact |
|------|------|---------|--------|
| risk/thesis_budget_governor.py | 136, 500, 519, 543 | Default `nav=100_000.0` | **MINOR BUG** -- overridden at runtime (main.py:2653 passes `self.config.initial_capital`) |
| risk/leverage_guard.py | 75 | `max_total_notional_usd=100000.0` | **INTENTIONAL** -- leverage limit, not account size |
| risk/leverage_guard.py | 74 | `max_position_notional_usd=50000.0` | **WARNING** -- allows 5x notional on $10K account |
| core/risk_governor.py | 165-166 | `_portfolio_value=100000.0` | TEST ONLY |
| core/runtime_spine.py | 364-390 | State init defaults | TEST ONLY |
| training/train_drl_full.py | 378 | `initial_balance=100000` | Training env (irrelevant) |

### Issues Found

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| thesis_budget_governor defaults $100K | LOW | L136, L500, L519, L543 | Change default to 10_000.0 (cosmetic, runtime overrides) |
| leverage_guard absolute limits | MEDIUM | L74-75 | $50K single + $100K total may be too permissive for $10K account; consider percentage-based |

### Recommendation
- **Minor fix**: Change thesis_budget_governor.py default `nav` from `100_000.0` to `10_000.0`
- **Monitor**: leverage_guard limits are intentional (Kraken leverage limits), but document that they represent 5x/10x on $10K account

---

## P0-6: LEVERAGE x POWER INTERACTION

### Verdict: EXPOSURE_SAFE

### Leverage Configuration

| Regime | Leverage | Source |
|--------|----------|--------|
| VOLATILE_CHOP | 3.0x | main.py:1587 + sota_flags.py:254 |
| MOMENTUM_RALLY | 2.0x | main.py:1589 + sota_flags.py:255 |
| PANIC_SELLOFF | 2.0x | main.py:1590 + sota_flags.py:256 |
| All others | 1.0x | main.py:1592 + sota_flags.py:257 |
| **MAX_LEVERAGE** | **3.0x** | main.py:988 + sota_flags.py:253 |

### Exposure Cap Chain (execution order)

```
1. SIZE_CAP (pre-leverage)          L8673    BTC/ETH=0.40, SOL=0.50
2. Regime Power (x target_exposure) L7861    0.0 - 1.5x
3. SIG-4 Direction Scaling          L7914    SHORT: 0.5x - 1.2x power
4. SIG-4 Leverage Cap               L7939    SHORT in bull: force 1.0x
5. Drawdown-Adaptive Reduction      L7947    DD>12% linear, DD>22% force 1x
6. LEVERAGE MULTIPLICATION          L9633    exposure_fraction *= regime_leverage
7. Dynamic Gross Cap (P1-1D)        L9640    Checks sum across 3 assets
8. Global Exposure Cap (V10)        L9672    Correlation-aware cross-asset
9. POST-LEVERAGE CAP (hard limit)   L9702    BTC/ETH=0.25, SOL=0.20
10. Absolute ceiling                L9702    0.60 (any single asset)
```

### Worst-Case Scenario

```
All 3 assets, same direction, VOLATILE_CHOP regime:
  Pre-leverage:  BTC=0.40 + ETH=0.40 + SOL=0.50 = 1.30
  x 3.0 leverage = 3.90 (theoretical)

  Post-leverage caps: BTC=0.25 + ETH=0.25 + SOL=0.20 = 0.70

  Result: 70% gross exposure <= 150% global cap
  Account risk: 70% x $10K = $7,000 notional (SAFE)
```

### SIG-4 Counter-Trend Protection

SHORT in MOMENTUM_RALLY:
```
Power:     x0.5 (halved)
Leverage:  capped to 1.0x (no leverage)
Combined:  0.10 base x 0.5 power x 1.0 lev = 0.05 (5% exposure)
vs LONG:   0.10 base x 1.3 power x 2.0 lev = 0.25 -> capped 0.25
```

### Drawdown-Adaptive Leverage

| Drawdown | Factor | 3x Regime -> Effective |
|----------|--------|----------------------|
| 0-12% | 1.00 | 3.0x |
| 12% | 1.00 | 3.0x (threshold) |
| 15% | 0.70 | 2.4x |
| 17% | 0.50 | 2.0x |
| 20% | 0.20 | 1.4x |
| 22%+ | 0.00 | 1.0x (forced) |

### Issues Found
- **None** -- multiple safety layers enforce hard limits at every stage

### Recommendation
- **No action needed** -- system is EXPOSURE_SAFE with 5 independent cap layers

---

## ACTION ITEMS (Priority Order)

### Immediate (Safety Gap) -- None Found
All P0 safety mechanisms are correctly implemented.

### This Week (Profit Blocker)

1. **P0-3: Wire DVOL from Kraken Futures**
   - File: data_mgmt/market_data_pipeline.py + integration_v36.py
   - Goal: Enable Group C OPPORTUNITY trigger
   - Impact: Adds volatility expansion detection to OPPORTUNITY mode

2. **P0-3: Wire sentiment_zscore to Group D**
   - File: integration_v36.py (where compute_opportunity_triggers is called)
   - Goal: Enable Group D OPPORTUNITY trigger
   - Impact: Enables sentiment shock detection

3. **P0-3: Review Group E $50M inflow threshold**
   - Current: $50M minimum inflow for SOL flow trigger
   - Consider: Lower to $10M based on actual SOL flow volumes

### Minor (Housekeeping)

4. **P0-5: thesis_budget_governor default nav**
   - File: risk/thesis_budget_governor.py L136, L500, L519, L543
   - Change: `nav: float = 100_000.0` -> `nav: float = 10_000.0`
   - Impact: Cosmetic (runtime always overrides with correct value)

5. **P0-1: Shadow ledger trigger metadata**
   - File: main.py (shadow_ledger write calls)
   - Goal: Persist `_exit_trigger_tag[asset]` in FILL records
   - Impact: Enables post-hoc exit trigger analysis without log parsing

---

## VERIFICATION CHECKLIST

To re-verify after fixes:

```bash
# P0-3: Verify DVOL is non-zero
grep "dvol_zscore" logs/proof_log_*.log | grep -v "=0.0" | head -5

# P0-3: Verify OPPORTUNITY mode fires
grep "OPPORTUNITY" data/shadow_ledger/ledger_*.jsonl | head -5

# P0-5: Verify thesis_budget_governor uses $10K
grep "nav=" risk/thesis_budget_governor.py | grep -v "100_000"

# P0-6: Verify post-leverage caps enforced
grep "POST_LEV_CAP" logs/proof_log_*.log | tail -5
```

---

*Generated by P0 diagnostic audit, 2026-03-01*
*Audited files: main.py, constitution.py, integration_v36.py, exit_alpha.py, exit_alpha_tracker.py, leverage_guard.py, unified_position_sizer.py, thesis_budget_governor.py, sota_flags.py, canonical_config.py*