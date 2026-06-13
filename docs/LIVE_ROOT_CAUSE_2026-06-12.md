# Live Root-Cause — Spot/Margin Mismatch + State Divergence

**Date:** 2026-06-12 (investigation before money reconciliation)
**Trigger:** "Fix trading first" — investigate before touching real funds.
**Scope:** read-only cloud forensics + local code/config trace. No money moved, no live state mutated.

---

## TL;DR

The bot is a **short-biased margin strategy**, but in every regime it has actually seen for weeks it runs at **`regime_leverage = 1.0` → orders route as SPOT, not margin**. A spot account cannot hold a short. So every "short" decision becomes a spot **SELL** of held coin, every "cover" becomes a spot **BUY**. Six weeks of this churn (with a buy bias from the pre-P139 phantom loop) accumulated **~$7,160 of real spot LONGS** while the tracker recorded **phantom margin SHORTS**. P139 stopped the *inflation* but the *direction/instrument mismatch* remains, and the "short" positions are now stuck (exits blocked by anti-churn). Equity drifted **~$9,600 → ~$7,180 (-25%)**.

**Two independent fixes are required to restore real trading:** (A) reconcile the money, and (B) stop running a short strategy at leverage=1 (the actual root cause).

---

## Evidence

### 1. Reality vs. tracker (live, 2026-06-12)
| Asset | Kraken actually holds (spot) | Tracker believes |
|---|---|---|
| BTC | LONG 0.0489 (≈$3,105) | SHORT ~$801 |
| ETH | LONG 2.091 (≈$3,481) | SHORT ~$264 |
| SOL | LONG 8.608 (≈$575) | not tracked |
| Cash | USDT $0.12 + USD $19.29 | — |
| Open orders | 0 | — |

Account ≈ **$7,180, ~100% spot longs, ~$19 free**. Matches heartbeat equity.

### 2. Orders route as SPOT, not margin
- `[P0_EXECUTE] ETH SELL … leverage=0.01x (regime_leverage=1.0x)` — effective leverage ≈ 0, regime_leverage = 1.0.
- **0** `[MARGIN] Kraken isolated margin` log lines in 24h.
- Code path: `core/execution_service.py:1766` `leverage = int(round(regime_leverage)) if regime_leverage > 1.0 else None`. `regime_leverage=1.0` → `None` → spot (P138 confirms `leverage=1` is treated as spot).

### 3. Config forces leverage=1 in all observed regimes
`configs/live_high_risk.json → regime_leverage`:
```
VOLATILE_CHOP: 3.0   MOMENTUM_RALLY: 2.0   PANIC_SELLOFF: 2.0
WEAK_CONSOLIDATION: 1.0   QUIET_ACCUMULATION: 1.0   EXTREME_VOLATILITY: 1.0
```
Observed regimes for weeks: **WEAK_CONSOLIDATION / QUIET_ACCUMULATION / NEUTRAL** — all → 1.0 → spot. The margin-capable regimes (CHOP/RALLY/SELLOFF) essentially never fired.

### 4. The "short" positions are now stuck
From one ETH tick (2026-06-13 00:03):
- Signals conflicting: `sentiment=-1.00`, `whale=-1.00/0.97` (bearish) vs `drl=+0.83/0.37` (bullish punch-through).
- Position underwater: `[EXIT_ALPHA] ETH: SCALE-OUT 25% … profit=-277bps`.
- Exit blocked: `[ANTI_CHURN_NET] ETH: blocked PARTIAL_EXIT … net_bps=-318.9 (min_bps=5.0)` → `status=ANTI_CHURN_NET_BLOCKED`.
- Result: spot SELLs occasionally dump real spot; exits get blocked; P139 skips cached fills. Net = slow bleed + churn.

---

## Root-cause chain
1. Short-biased margin strategy deployed with `regime_leverage=1.0` in the only regimes the market has shown → **all orders spot**.
2. Spot can't short → "short" = sell held coin, "cover" = buy coin → **directionless spot churn**, net LONG accumulation (~$7,160).
3. Tracker records **intent direction** (short), not spot effect (long) → divergence. Pre-P139 phantom loop inflated recorded sizes (245 vs 8.6 SOL).
4. P139 stopped inflation; mismatch + stuck underwater "shorts" remain; anti-churn blocks exits; equity −25%.

---

## What "fix trading first" actually requires — TWO fixes

**(A) Money reconciliation** — the ~$7,160 spot longs vs tracker shorts. Options:
- A1. Flatten spot → cash (~$7,150 USD), reset tracker flat. Restores clean collateral.
- A2. Pause bot, freeze holdings as manual longs.

**(B) Root-cause config/architecture fix** — stop expressing shorts on spot. Options:
- B1. **Gate trading off when `regime_leverage ≤ 1`** (don't pretend to short on spot; hold instead). Safest, smallest change, fully reversible. The system simply does nothing in low-leverage regimes rather than churning spot.
- B2. Raise low-conviction-regime leverage to ≥2 so shorts are real margin. *Increases risk on a system that just lost 25%; needs free collateral (currently $0.12).* Not recommended now.
- B3. Convert to spot-long-only strategy (flip the bias). Fundamentally different system.
- B4. Migrate to a venue with real perps (Coinbase Perp = v5.1 Phase 2). Correct long-term fix; months out.

**Recommendation:** A1 (flatten to cash) **+** B1 (gate off leverage≤1 trading) as the minimal, reversible combination that both cleans up the money and stops the bleed, without adding leverage/risk. Then revisit B2/B4 inside the v5.1/Coinbase plan with clean state and a real IC window.

---

## Notes
- No code changed and no orders placed during this investigation.
- B1 is a config/guard change in the live decision path — would need a small patch + redeploy, with operator sign-off, per the trade-frequency discipline.
