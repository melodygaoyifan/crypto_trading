# HMATS Phase 1R — report

Generated 2026-10-18. Supersedes Phase 1 (`reports/HMATS_PHASE1_SHADOW.md` keeps its own shadow criterion). All commits are local; nothing pushed or deployed.

Commits (each `[P1R-x]`, on top of `6fa4d40`): `cfaa316` R-1 · `6bf3ac6` R-2 · `4b6e42b` R-3 · `86e315b` R-4 · `13e13c8` R-5 · `f22fad0` R-6 · `ffa833d` R-7 · `87b331f` R-8 · `3640f34` R-1/R-5 follow-up (VENUE_FEE_STD frozen, dynamic `main` import).

Checks: `git diff 6fa4d40 -- defense/constitution.py training/` = 0 lines (also 0 vs `4e2078f`); no `obs_dim` line touched; full suite 7831 passed / 0 failed after the follow-up (run before it: 1 failure, the registry stamp — fixed); invariant gate (`--skip-mypy`) clean; `tools/mypy_attribution.py 6fa4d40 HEAD`: 803 → 803, 0 added.

---

## 1. R-5 — does the current configuration have edge at current cost? (`reports/phase1r_backtest.json`)

**Pre-registered rule** (fixed in `analytics/backtest/current_config_replay.py` before the first run):
NO_EDGE_AT_CURRENT_COST iff all three assets have median gross alpha ≤ median all-in cost; an asset is EDGE_CONFIRMED_BACKTEST iff median gross > median all-in cost AND the bootstrap 95% CI lower bound of its median net > 0; otherwise UNCERTAIN. Counterfactuals can only be HYPOTHESIS_SUPPORTED / NOT_SUPPORTED. Live ≈ replay × 0.3–0.5.

**Result: `UNCERTAIN`** (the NO_EDGE trigger does not fire, because BTC's median gross is above its cost).

| asset | decider (replay) | n RT | median gross α | median all-in | median net | CI95 median net | win rate (net) | median hold | fills/RT | Σ net $ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | skew_contra (100% of bars) | 12 | **+133.9** | 18.4 | +93.5 | **[−567, +1236]** | 50% | 186 bars | 2.7 | +1012 | **UNCERTAIN** |
| ETH | regimebook Donchian (357/2195 bars ETF de-risked to flat) | 11 | −19.9 | 37.0 | −44.6 | [−503, +311] | 36% | 54 | 4.4 | +128 | **NO_EDGE_ASSET** |
| SOL | regimebook SMA200 hold-bull | 7 | −60.7 | 21.8 | −86.7 | [−111, +30] | 29% | 5 | 2.1 | +200 | **NO_EDGE_ASSET** (n=7 < 10, INSUFFICIENT) |

How long until BTC could be decided from live: at the replay's frequency of 0.23 RT/week, closing the CI would take about 1,100 RTs, or roughly 4,800 weeks. **BTC's question cannot be answered by waiting.**

**What the medians hide, stated because the rule is fixed and must not be re-read after the fact:** ETH and SOL lose on the median RT and *make* money in sum (+$128 / +$200). A trend/hold book is many small losers plus a few large winners, and a median-based rule is structurally hostile to that shape. This does not change the verdict; it is the reason the verdict is not "the books lose money". BTC's +$1,012 comes from three of its 12 trades: two long-held shorts through the 2025-Q4 and 2026-Q1 declines (+$415, +$382) and a long into the 2026 summer rally (+$402). Together they exceed the total, so the other nine net negative.

**Counterfactuals** (net $ change vs base over the 12 months; n RT unchanged):

| run | BTC | ETH | SOL | total | verdict |
|---|---|---|---|---|---|
| A: FLIP off (a flip only closes) | −173.8 | −10.7 | +9.1 | −175.5 | NOT_SUPPORTED |
| B: MIN_HOLD 2 | −36.1 | −6.3 | −36.0 | −78.4 | NOT_SUPPORTED |
| B: MIN_HOLD 3 | −36.1 | −6.3 | −42.8 | −85.1 | NOT_SUPPORTED |
| B: MIN_HOLD 6 | −5.2 | −15.0 | −54.3 | −74.5 | NOT_SUPPORTED |

In the current config, BTC's flips are the skew seat's sign changes, and those are the trades that make the money. Removing them costs the most.

**Hold buckets.** Every BTC and ETH RT falls in the 9+ bucket. SOL: bucket 1 n=3 (−89 bps), 5–8 n=1, 9+ n=3. All cells except BTC/9+ and ETH/9+ are INSUFFICIENT.
**Regime.** The GMM regime is available only from the live tick diagnostics (2026-04-08 onward), so 16 of 30 RTs are UNKNOWN. Every GMM cell has n ≤ 4 and is INSUFFICIENT; the tables are in the JSON.

**Sensitivity.** Maker share 34.6% / 46.9% (base: current config) / 69.7% (ladder fills only) leaves every verdict unchanged (BTC median net 93–94, ETH −43 to −45, SOL −86 to −87).

**Parity check.** The replay's end state is BTC −2 and ETH/SOL flat, which matches the live book.

**Modelled, from the live chain (imported, not restated):**
- regimebook labels, with Donchian-100 on ETH
- ETF de-risk via `etf_seat_decision` on the live z-deadband
- skew seat via the live z-blend and deadband on the runtime bank
- alpha gate: calibrated edge × 0.75 vs the lowest observed live threshold
- P274 sizing
- flip and resize persistence (2 ticks) and the 2-tick re-entry cooldown, from the live profile

Costs are applied per fill:
- **fee:** the R-1 tiered schedule, with the tier set by the replay's own rolling 30-day volume
- **spread:** taker fills pay the Phase 0 median taker spread cost (BTC 0.64, ETH 2.07, SOL ~0 bps); maker fills pay 0
- **funding:** the CDE rate from 2026-08-16; the Binance daily rate before that, as a proxy (P218); 159 of 3,340 held bars had no rate and carry 0

**NOT modelled:** the 30-second FastRisk watchdog, the 10% venue stop, P0/data-quality vetoes, the correlation entry block. The watchdog matters. In the current config it caused 16 of the 26 taker fills since 08-27: a REDUCE_50 on depth-drop or vol-spike, followed by a 4H re-add. That loop is a real cost the replay omits.

**Why not `BacktestEngine`.** It opens a trade on every signal bar, closes it after a fixed 24 bars (trades overlap), sizes at 10% notional, and prices fees at 0/3 bps with 98.7% maker. It cannot represent a hold-until-the-seat-changes, contract-quantized book, and a per-fill cost callback would not change that. The replay is position-based instead and lives under `analytics/backtest/`; `training/` is untouched.

**Correction to Phase 0.** Phase 0 took gross alpha from fill prices and then added spread as a separate cost, so its net counts the spread twice (about 1–3 bps per RT). The replay measures gross mid-to-mid.

---

## 2. R-1 — fee model (`reports/phase1r_fee_replay.txt`)

The tier table is derived from the fills; nothing is typed in:

| rolling 30d volume | taker | maker | source |
|---|---|---|---|
| 0 | 10.0 | 9.5 | venue `get_transaction_summary` (Intro, usd_to 10,000) |
| ≥ 10,000 | 8.0 | 7.5 | 80 taker + 22 maker fills at 10.5k–49.9k |
| ≥ 50,365 | 7.0 | 6.5 | 30 taker + 48 maker fills at 50.4k–74.7k |

Fixed charge per contract: venue $0.10 + clearing $0.01 + regulatory $0.00 = $0.11. This is the median over the last 14 days of fills, pooled across assets (the three components are asset-independent). Unknown volume prices at the most expensive tier.

| replay (207 fills BTC/ETH/SOL) | error = Σ\|model−actual\| / Σactual | gate |
|---|---|---|
| tier (pct) part alone | 0.56% (only the June fills at the old 9 bps Intro rate) | — |
| D2: rolling tier + fixed charge at its monthly epoch | **1.08%** | < 3% ✅ |
| D: rolling tier + today's fixed charge applied to all history | 3.80% | informational |
| per tier (D): 7 / 8 / 10 bps | 3.84% / 4.15% / 2.20% | < 5% ✅ |
| A: current-tier fills only | 2.88% | < 5% ✅ |

D misses 3% only because the venue cut its clearing and regulatory charges over the window ($43.08 charged against $37.62 modelled at today's rate). The schedule prices today's charges, which is the right choice for live. Fills with error above $0.02 are listed in the replay file: 25 under D2 and 100 under D.

**Wired to:**
- the alpha gate via `venue_fee_bps`, now tier-aware from the fill ledger
- the P1-2 realized-friction floor
- the replay
- `execution_service._COINBASE_*_BPS` (fee context and telemetry base)

**BLOCKED:** `core.paper_fee_service.VENUE_FEE_STD["coinbase"]` stays at 0/3 bps. `training/train_drl_full.py` imports it as the DRL trainer's fee source and keeps a parity-tested mirror of it, so changing it would change training (iron rule 1). Live, it is read only on the Kraken execution path, which every routed asset skips (P152).

**SKIP (unreachable):** PnL attributor and ExitAlphaTracker are on the Kraken path. `BacktestEngine` is in `training/`. Sleeve PnL uses the venue's own reported fees.

## 3. R-2 — instrumentation (IMPLEMENTED)

Every sleeve fill now records:
- `intent_type`, classified from the venue position before the fill and the real target, which `execute_target` stamps
- `order_type` and `post_only_requested`
- `liquidity_indicator`: the venue's own maker/taker verdict, from `get_fills` on the filled order
- `rolling_30d_volume_usd` and `fee_tier_taker_bps`
- `cause`

On cause: the watchdog's REDUCE_50 branch now stamps `risk_exit:fast_risk_reduce`. The live urgent BTC fills on 2026-10-14 and 10-15 recorded cause=None, because P423 only stamped EXIT_ONLY. The other cause paths were already covered by P423.

Decision bid/ask/mid were already recorded (P290).

A new hourly `data/funding_accrual.jsonl` records per held position: rate × notional, labelled as an ESTIMATE because the venue has no payment ledger.

Log retention: 5 → 100 files of 10 MB, about 90 days at the measured ~10 MB/day (33 GB free on the server).

Not backfilled. Verified by unit tests rather than a paper tick, because paper mode does not drive the sleeve recorder. The first fill after deploy is the live check.

## 4. R-3 — where the 34.6% maker share comes from (`reports/phase1r_maker_attribution.md`)

- **Before maker-first (until 2026-08-16):** 93 fills, 1 maker. Maker-first was off, so every order was a 0.2%-through LIMIT, which is a taker by construction.
- **After maker-first:** 118 fills, **61.0% maker**.
- **Post-only reaches the venue:** all 69 maker-ladder fills came back MAKER, 0 TAKER. The flag is passed `OrderRequest(post_only=True)` → `limit_order_gtc(post_only=…)`, and the venue rejects a crossing post-only order (P270 probe).
- **Ladder timeouts:** 19 of 88 ladder fills (21.6%) timed out into a taker cross.
- **Where the taker fills come from after maker-first:**

| source | fills | breakdown |
|---|---|---|
| urgent MARKET watchdog | 16 | all PARTIAL_EXIT |
| ladder timeout → taker cross | 19 | 8 ESCALATE, 4 FULL_EXIT, 4 NEW_ENTRY, 3 PARTIAL_EXIT |
| direct (maker-first off) | 3 | — |
| pre-ledger | 8 | — |

- **Current config (since 08-27):** 49 fills, 46.9% maker. Of the 26 takers, 16 are the watchdog. Of the 23 maker fills, 18 are the ESCALATE re-adds that followed it.
- **What V15's 98.7% was:** an order-type share. `maker_fee_ratio = n_limit / n_classified` counts LIMIT orders (91.9% of these fills are LIMIT), not the venue's liquidity verdict. `BacktestEngine.FeeSchedule.maker_pct = 0.987` cites it.
- **Disposition:** no bug to fix. Watchdog taker fills are risk exits and correct, and are never suppressed. Ladder timeouts are recorded only, not changed.

## 5. R-4 — FLIP diagnosis (`reports/phase1r_flip_diagnosis.md`)

58 FLIP orders (66 fills), **all between 2026-06-24 and 08-05, in the trend-seat ±1 era.** Fees on them were $59.36.

| | BTC | ETH | SOL | total |
|---|---|---|---|---|
| FLIP orders | 29 | 13 | 16 | 58 |
| gross of the closed RTs $ | −109.7 | −8.4 | −57.5 | −175.5 |
| gross of the opened RTs $ | −119.4 | −8.2 | −49.1 | −176.6 |
| flipped while losing | 22 | 10 | 14 | 46 |
| whipsawed back within ≤2 bars | 12 | 5 | 6 | 23 |

Median hold was 3 bars, both for the flipped position and for the one it opened.

**Current reachability:**
- The Kraken `[FLIP_GATE]` (core/execution_service.py:~2284) sits past the P152 return at :684, so it is unreachable.
- The sleeve flip path is live behind P198's 2-tick flip persistence (2026-08-07). There have been 0 FLIP orders since.
- The current config's 2 closed RTs + 1 open contain no FLIP.
- The path stays reachable through BTC's skew sign change. R-5 counterfactual A shows those flips are where BTC's money comes from.

## 6. R-6 — gate friction → realized RT cost `[OPERATOR DECISION]` (`reports/phase1r_gate_consequence.json`)

**Current values:**
- friction: round-trip = 2 legs × (venue-schedule taker fee + spread + latency) + hold
- N = `NORMAL_MULTIPLIER` 1.10
- `MAX_ALPHA_BPS` = 200, used only on the non-seat path; the seats assert calibrated edges

**The data reverse the spec's premise.** Today's gate friction is already *above* the measured round-trip all-in, so option A **loosens** the gate:

| asset | friction now (RT) | threshold now | option A friction | option A threshold | ticks passed (60d, now → A) |
|---|---|---|---|---|---|
| BTC | 29.4 | 32.3 | 23.9 (current-tier projection) / 21.8 (Jun–Oct realized) | 31.6 / 28.9 | 159 → 160 / 165 |
| ETH | 45.5 | 50.1 | 31.7 / 32.1 | 40.3 / 40.8 | 255 → 255 |
| SOL | 42.7 | 47.0 | 25.6 / 26.6 | 27.9 / 29.1 | 323 → 323 |

ETH and SOL do not change, because their seated edges already clear the threshold by 3–6×. BTC gains at most 6 passing ticks out of 155 vetoed ones. **Default B stands.** Given R-5's result (UNCERTAIN, not NO_EDGE), neither A nor B changes the answer. The P1-2 realized-friction gate stays shadow and tighten-only.

## 7. R-7 — netting (regression tests only)

The 18 same-4H opposite-direction pairs from Phase 0 were traced one by one:
- **Restart ticks on deploy days** (2026-08-16/18/19/22/25/26): `run_live` runs a tick immediately when it starts (P154), so a deploy plus the scheduled tick produced two decisions in one bucket.
- **One watchdog urgent exit** landing in the same bucket as a 4H add (08-28).
- **2026-06-13:** the cutover day.

None of them was two intents of one tick netted wrongly. Within a tick, each asset gets one `manage_to_signal`, which produces one target and at most one `execute_target`, and a flip is a single net order. The pairs stopped with the deploy freeze and the P416/P420 persisted resize/flip streaks.

Tests pin that property: `tests/test_p1r_7_netting.py`, 10 tests, all pass with no runtime change.

## 8. R-8 — side bugs

1. **Contract expiry:** the API reports `contract_expiry` 2089-12-30, while the product IDs carry the 20DEC30 tag. Comment corrected in `core/cde_fees.py`; no code reads either date.
2. **`fill_quality.target_contracts`:** it held the signed direction. Fixed in R-2: the real contract count is now stamped in `execute_target`, and the direction is recorded as `target_direction`. Nothing in production read the field.

## 9. Status

| item | status | evidence |
|---|---|---|
| R-0 | done | **CURRENT_CONFIG_START = 2026-10-13, `cf290c5`** (BTC skew seat re-armed, P424). ETH/SOL stack since 2026-08-27 (`9058f49` P416 resize persistence, `e12b12d` P417 whale seat + conviction off, `7429797` P419 ETH Donchian). The ±1 seats ended with `305f6f0` P298 (regimebook seat, 08-18) and `e12b12d` (whale off, 08-27); last FLIP fill 08-05. BacktestEngine has no per-fill hook (flat round-trip bps `FeeSchedule`). MARKET orders only from `CoinbaseSleeve.URGENT_ORDER_TYPE` via `urgent=True`: the FastRisk watchdog EXIT_ONLY and REDUCE_50 (main.py:4097, :4130) and FORCE_FLAT (main.py:20251). |
| R-1 | IMPLEMENTED; `VENUE_FEE_STD` coinbase BLOCKED | §2 |
| R-2 | IMPLEMENTED | §3, `tests/test_p1r_2_instrumentation.py` |
| R-3 | DIAGNOSED (no bug) | §4 |
| R-4 | DIAGNOSED | §5 |
| R-5 | DONE — UNCERTAIN | §1 |
| R-6 | consequence table done, `[OPERATOR DECISION]`, default B | §6 |
| R-7 | SKIP (no runtime fix needed) + regression tests | §7 |
| R-8 | IMPLEMENTED | §8 |

**Mapping to the superseded Phase 1:**
- P1-1 → R-1 (extended)
- P1-2 → R-6 option A, shadow and tighten-only (unchanged)
- P1-3, P1-5 → deleted by the Phase 1R spec
- P1-4 contract floor → still shadow
- P1-6, P1-7 → kept

Found but out of scope: tests (`tests/test_p290_fill_quality.py`, `test_p291_maker_ladder.py`, `test_p420_sleeve_maker_fill_truth.py`) write `oid-*` rows into the local `data/fill_quality.jsonl`. That is local only and never live evidence (P255).
