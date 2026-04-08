# HMATS vs SOTA Gap Analysis Report

> **Disclaimer**: This is NOT legal, financial, or investment advice. All analysis is for engineering evaluation purposes only. Past performance and backtested results do not guarantee future returns. Crypto trading involves significant risk of capital loss.

## Executive Summary

**16 items compared. 2 P0 blockers, 4 P1 improvements, 5 P2 optimizations, 5 items where HMATS meets or exceeds SOTA.**

### !!CRITICAL Findings

1. **PA Executor friction=50bps blocking ALL trades** despite alpha gate passing (32bps > 11bps threshold). The PA edge_multiplier requires 3x friction = 150bps minimum edge, which no signal can meet. This is the #1 reason for zero executed live trades.

2. **'Insufficient funds' execution failure** on all 4 order slices for BTC. Order sizing does not check available balance before submission.

---

## Diff Table

| # | Area | HMATS State | SOTA Requirement | Gap Type | Priority | Capital Risk | Fix |
|---|------|-------------|------------------|----------|----------|-------------|-----|
| 1 | **PA friction** | friction=50bps, edge_mult=3x, blocks all trades | Derivatives 2-5bps friction | !!CRITICAL blocker | **P0** | 4/5 | Debug PA friction source; FREE tier should be ~15bps not 50bps |
| 2 | **Execution funds check** | 'Insufficient funds' on all slices | Balance check before order | !!CRITICAL blocker | **P0** | 3/5 | Add available_balance check pre-order, clamp size |
| 3 | **Spot vs Derivatives fees** | Spot-only (26bps taker default) | Derivatives 2-5bps | Implementation gap | **P1** | 4/5 | Add Kraken Futures perpetual trading |
| 4 | **Leverage cap** | Lmax=3, vol_target=32% | Lmax=5-8, sigma=60-120% | Conservative vs SOTA | **P1** | 3/5 | Raise Lmax=5, vol_target=60-80% |
| 5 | **Purged CV** | Gap=42 bars, no formal PurgedKFold | Purged CV + embargo + combinatorial | Partial implementation | **P1** | 2/5 | Implement PurgedKFold in training |
| 6 | **Vol target** | ~32% annual target | 60-120% for high risk | Conservative | **P1** | 2/5 | Raise to 60-80% |
| 7 | **VWAP execution** | Not implemented (TWAP enum only) | VWAP + TWAP + depth-aware | Missing | **P2** | 1/5 | Simple TWAP sufficient at $10K |
| 8 | **ADV/capacity** | No capacity check | Q <= alpha*ADV | Missing | **P2** | 1/5 | Not binding at $10K |
| 9 | **Event replay** | Archived, not live | Durable message bus | Gap | **P2** | 1/5 | File-backed event log |
| 10 | **Rebalance ID** | No cross-asset rebalance ID | Idempotent rebalance_id per tick | Gap | **P2** | 2/5 | Add hash(tick_ts) |
| 11 | **Cost stress test** | No 2x/3x cost test in backtest | Mandatory cost stress | Gap | **P2** | 2/5 | Add to training pipeline |
| 12 | Funding/carry | Full implementation | Carry as alpha source | **Meets SOTA** | - | - | - |
| 13 | Order flow/VPIN | VPIN + whale + toxicity | ML order-flow prediction | **Exceeds SOTA** | - | - | - |
| 14 | On-chain data | BTC/ETH/SOL on-chain feeds | On-chain regime filter | **Meets SOTA** | - | - | - |
| 15 | Dead-man switch | Server-side + dedicated client | Exchange circuit breaker | **Exceeds SOTA** | - | - | - |
| 16 | DRL safety | Auto-demotion + OOD + bounded | Per-agent drawdown limit | **Exceeds SOTA** | - | - | - |

---

## P0 Deep Analysis

### P0-1: PA Friction Blocking All Trades

**Evidence:**
```
[PA_PROOF] BTC: friction=50.0bps, edge insufficient (32.3 bps < 3.0x 50.0 bps)
```

**Root cause chain:**
1. Alpha gate passes: 32bps > 11bps threshold
2. Veto chain passes: all gates clear
3. PA executor evaluates: `edge (32.3bps) < edge_multiplier (3.0) * friction (50.0bps) = 150bps`
4. PA rejects as insufficient edge
5. No trade executes

**Why friction=50bps when fee tier=FREE (0bps)?**
- `[FIX-FEE-TIER] Fee tier updated: taker=40.0bps, maker=25.0bps` — the fee tier query returned Kraken's DEFAULT tier fees, not the free-tier override
- FREE tier logic exists in `defense/constitution.py` FrictionComponents but the PA executor uses a DIFFERENT friction path that queries the exchange directly
- Exchange returned standard fees (40/25bps) which the PA executor uses as its friction basis

**Fix:**
```python
# In execution/passive_aggressive.py, before edge check:
# Use the same fee source as constitution.py FrictionComponents
if fee_blender and fee_blender.is_free_tier():
    friction_bps = slippage_bps + latency_bps  # ~15bps, no fee component
```

### P0-2: Insufficient Funds on Order Execution

**Evidence:**
```
[KrakenREST] Market order BTC/USD: EOrder:Insufficient funds
```

**Root cause:**
- Account equity = $9,400 but this includes non-USD assets (BABY, BTC tokens)
- Available USD balance may be significantly less than total equity
- Order sizing uses total equity as basis but Kraken requires sufficient USD margin

**Fix:**
```python
# In execution/execution_manager.py, before create_market_order:
available = self.exchange.fetch_balance()['USD']['free']
max_notional = available * 0.95  # 5% safety buffer
if order_notional > max_notional:
    order_notional = max_notional
    size = order_notional / price
```

---

## Priority Action List

### P0 — Immediate (blocks all live trading)

| # | Action | Owner | Milestone | Acceptance |
|---|--------|-------|-----------|------------|
| 1 | Fix PA friction source — use fee_blender free-tier check | Engineering | Friction <20bps in proof log | PA_PROOF shows edge > friction |
| 2 | Add balance check pre-order | Engineering | No 'Insufficient funds' errors | Order placed successfully OR gracefully sized down |

### P1 — Significant improvement (next 2 weeks)

| # | Action | Owner | Milestone | Acceptance |
|---|--------|-------|-----------|------------|
| 3 | Evaluate Kraken Derivatives access for account | Engineering/Ops | Derivatives API access confirmed | Lower fee tier available |
| 4 | Raise vol target 32%->60% | Engineering | Config change | Higher position sizes in low-vol regimes |
| 5 | Raise Lmax 3->5 with vol-target clamping | Engineering | Config + code | Dynamic leverage follows vol target |
| 6 | Add PurgedKFold to training pipeline | Research | Training scripts updated | Next model retrain uses purged CV |

### P2 — Optimization (4-8 weeks)

| # | Action | Owner | Milestone | Acceptance |
|---|--------|-------|-----------|------------|
| 7 | Add cost stress test (2x/3x) to backtest | Research | Backtest script updated | Stress results logged |
| 8 | Simple TWAP execution (3-5 slices, 10min) | Engineering | Execution path updated | Sliced orders in proof log |
| 9 | ADV capacity check | Engineering | Pre-order check | Logged when ADV would bind |
| 10 | Rebalance ID per tick cycle | Engineering | Cross-asset coordination | rebalance_id in event log |
| 11 | File-backed event log | Engineering | Append-only log | Replay capability |

---

## Test Matrix

| Test Type | P0-1 (PA friction) | P0-2 (Funds check) | P1-4 (Vol target) | P1-5 (Leverage) |
|-----------|--------------------|--------------------|--------------------|-----------------|
| Unit test | PA friction returns <20bps for free tier | Order clamped to available balance | Vol target = 60% loaded from config | Lmax=5 respected |
| Integration | Alpha gate + PA both pass for 32bps signal | Full order flow: signal -> size -> balance check -> order | Position size increases in low-vol | Leverage dynamically follows vol |
| Backtest | Simulated trades not blocked by friction | No insufficient funds in paper | Higher utilization, more trades | Higher returns, check DD |
| Paper trading | First trade executes within 24h | No execution errors | Utilization >25% within 1 week | Leverage >1x in trending regimes |

---

## Roadmap

### Short-term (2 weeks)
- **P0-1**: Fix PA friction (0.5 day)
- **P0-2**: Fix funds check (1 day)
- **P1-4**: Raise vol target (0.5 day)
- **P1-5**: Raise Lmax (0.5 day)
- **Total: 2.5 days engineering**
- **Key dependency**: None, all config/code changes

### Medium-term (2-8 weeks)
- **P1-3**: Evaluate + enable Kraken Derivatives (3-5 days)
- **P1-6**: PurgedKFold in training (2-3 days)
- **P2-7**: Cost stress test (1 day)
- **P2-8**: TWAP execution (2 days)
- **Total: 8-11 days engineering**
- **Key dependency**: Kraken Derivatives account eligibility

### Long-term (>8 weeks)
- **P2-9 to P2-11**: ADV check, rebalance ID, event log (5-7 days)
- **Full Derivatives integration**: spot+perp dual venue (5-7 days)
- **Total: 10-14 days engineering**
- **Key dependency**: Derivatives account, sufficient trading history for purged CV

---

## Scoring Rubric

| Score | Security | Capital | Performance | Maintainability | Compliance | Cost |
|-------|----------|---------|-------------|-----------------|------------|------|
| 0 | No risk | No risk | No impact | Clean | Compliant | No cost impact |
| 1 | Minor | <1% equity risk | <1% CAGR | Minor debt | Minor | <5bps/trade |
| 2 | Low | 1-5% risk | 1-3% CAGR | Moderate debt | Potential issue | 5-15bps |
| 3 | Medium | 5-10% risk | 3-8% CAGR | Significant debt | Needs review | 15-30bps |
| 4 | High | 10-20% risk | 8-15% CAGR | Architecture issue | Likely violation | 30-50bps |
| 5 | Critical (exploit) | >20% loss possible | >15% CAGR | Unmaintainable | Definite violation | >50bps |
