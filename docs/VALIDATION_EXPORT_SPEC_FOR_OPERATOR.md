# Export spec — what the operator must produce so the validation harness can run

> ## ✅ DRL HALF (Part A) — RESOLVED LOCALLY 2026-06-14, no operator export needed
> The premise "what's on the box can't be validated" was **wrong for the DRL**.
> All artifacts existed (val parquets, 9/9 fold checkpoints + scalers, the
> `TQCTeacherOracle` rollout). `scripts/validate_drl_oos.py` now rolls each fold
> over the val window, **exports the per-bar return series** to
> `data/validation_export/drl_{ASSET}_fold_{k}_val_returns.jsonl`, and runs PSR +
> CSCV-PBO. **Result (all 3 assets): val-window Sharpe +11..+19, PSR 100%, PBO
> 1–7% ("ROBUST") — yet LIVE Sharpe is −2.62. Verdict: `BACKTEST_CONTRADICTS_LIVE`.**
> The backtest (even with CPCV/PBO) is NON-PREDICTIVE of live because PBO only
> tests fold-to-fold generalization within the same historical window — it is
> blind to the regime shift between that window (Oct'25–Mar'26) and the live
> future (Apr–Jun'26). **Conclusion: forward-live PSR is the ONLY valid arbiter;
> do NOT promote/trust any DRL version on backtest metrics.** This vindicates the
> current bounded-ADVISE posture (don't demote, don't expand). Report:
> `data/validation_export/drl_validation_report.json`.
>
> **Part B (v5.1 strategies) still needs work** — those strategies are live-only
> (no backtest harness, hardcoded assumed Sharpes) and are already demoted OFF
> live (P147, zero measured signal). A proper backtest needs a ~2–3 day refactor
> to make them accept historical OHLCV; lower priority since they're already off.

---

## A. DRL (TQC) — per-fold validation return series

**What exists now (insufficient):** `models/retrained/{ASSET}/results.json` has only
`mean_reward` + `std_reward=0` per fold. No return path → can't compute OOS Sharpe,
can't run CPCV, can't even test if the best-fold pick is within noise. And `reward`
is the RL training reward, NOT realized OOS Sharpe — the wrong metric (P143).

**Export needed** — for EACH asset × EACH fold, the **validation-window realized
return series** (the per-bar PnL the policy produced on that fold's held-out val data):

```
data/validation_export/drl_{ASSET}_fold_{k}_val_returns.jsonl
  one record per 4H val bar:
  {"bar_ts": <ms>, "ret": <float realized return that bar, net of the friction
                           already in TradingEnvFull.step()>}
```

How to produce: during `train_drl_full.py` evaluation, you already roll the policy
over each fold's val window — log the per-step NAV change (the env already deducts
costs) to the file above instead of (or alongside) collapsing to `mean_reward`.

**Then validate:**
```python
from analytics.validation.sharpe_validation import annualized_sharpe, probabilistic_sharpe_ratio
from analytics.validation.cpcv import cscv_pbo
# load the 3 folds' return series per asset -> matrix [fold][bar]
res = cscv_pbo(fold_return_matrix)         # PBO of the best-fold selection
# + per-fold annualized_sharpe + PSR; promote ONLY if PBO < 0.5 and OOS PSR > 0.95
```
Expectation given the live evidence (Sharpe −2.62, PSR 21%): the DRL will likely
**fail** — at which point "don't demote" still holds (runtime guards bound it), but
no *new* DRL version is trusted until it passes.

---

## B. v5.1 strategies — actual backtest return series (currently ASSUMED)

**What exists now (a finding, not a validation):** `risk/sleeve_allocator_v5_1.py`
registers the 4 strategies with **hardcoded assumed Sharpes** —
`directional_short sharpe=0.8`, `microstructure 1.0`, `cascade 0.9`, `funding 1.5`,
`ml_factor 1.0`. These are literals, not measured. The strategies were promoted to
live ADVISE on made-up numbers. There is no return data to validate.

**Export needed** — for EACH of the 4 strategies, a **historical backtest return
series** on BTC/ETH/SOL 4H (same cost model as the trend backtest: 15bps/turn,
no lookahead):
```
data/validation_export/v5_1_{strategy}_returns.jsonl
  {"bar_ts": <ms>, "ret": <float net return that bar from the strategy's signal>}
```

**Then validate (same harness):** run each through `annualized_sharpe` + `probabilistic_sharpe_ratio`,
and run the 4 together through `cscv_pbo` to see if "the best of the 4" generalizes.
Replace the hardcoded `sharpe=` literals in `register_sleeve(...)` with the measured
OOS values. Demote any whose PSR < 0.95 from live ADVISE.

---

## C. Acceptance gate (applies to ALL future model promotions)
A model/strategy may go live (or up an authority level) ONLY if, on the exported
OOS return series:
1. **PSR(true SR > 0) ≥ 0.95** (Probabilistic Sharpe), AND
2. **CSCV PBO < 0.5** (the selection generalizes), AND
3. realized OOS annualized Sharpe is **positive after costs**.
This replaces `best_fold = max(folds, key=mean_reward)` (`train_drl_full.py:1488`) —
the exact selection-bias mechanism behind backtest 9–10 → live −2.62.
