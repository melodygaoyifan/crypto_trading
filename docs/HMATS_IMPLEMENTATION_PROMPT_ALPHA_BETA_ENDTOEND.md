# HMATS Implementation Prompt — Alpha/Beta End-to-End Fix — DRAFT

**Status**: DRAFT for operator review. Do NOT execute as-is. Each layer is SHADOW-gated on post-fix live data.
**Triggered by**: Alpha/beta forensic (CLAUDE.md P143, memory `live-performance-apr-jun-2026`) + deep-research pass (memory `crypto-quant-research-2026-06`).
**Date**: 2026-06-13
**Author**: Claude (Opus 4.8)
**Depends on**: Layer-2 churn control (commit `228a984`, LIVE) + alpha-feedback (commit `e79af37`, LIVE).
**Companion**: `docs/HMATS_IMPLEMENTATION_PROMPT_LAYER1_4_IC_GATE.md` (the IC-gate is the substrate for Layers 1–2 here).

---

## The diagnosis in one paragraph (what this fixes)
Kraken-authoritative −25% = −$2,314 trading + −$125 fees. Decomposition: **beta** +0.54 net-long into a −23% market (~half the loss) + **alpha** −62%/yr (~half), with system vol 1.8× the market from churn. Per-agent live IC (1456 records): the DECIDE `quant` agent is **noise aggregate (−0.018) but a regime SIGN-FLIP** (+0.17 weak-consol / −0.41 uptrend / +0.40 neutral); `model_alpha`/`llm_sentiment` inverted; `drl` break-even (+9 backtest Sharpe ≠ live); **10 directional agents emit zero for 2 months** (condition-specific strategies dormant in the quiet/chop regime that dominated). Verdict: **~half recoverable-engineering, ~half design-evolution — not a write-off.**

## Research basis (verified 3-0 unless noted; full cites in the memory)
- **R1** Regime-conditional application beats sign-blind (HMM/GMM + Jump Models, CV-tuned jump penalty). Wang/Lin/Mikhelson 2020; Jump models arXiv:2402.05272.
- **R2** Combine weak alphas by realized IC × alpha-decay, NOT equal-weight; linear-time optimal weighting. Garleanu-Pedersen J.Finance 2013 (NBER w15205); Kakushadze-Yu arXiv:1603.05937.
- **R3** Overfitting gate: Deflated Sharpe Ratio + Combinatorial Purged CV + RL hypothesis-test. Bailey/López de Prado SSRN 2460551; Arian/Norouzi/Seco 2024 (SSRN 4686376); Gort et al FinRL arXiv:2209.05559. (RL test empirical 2-1.)
- **R4** Crypto INDEX beta-hedging is structurally ineffective (R²<0.20 vs 0.58 equities); use instrument-level perp hedges + a beta budget. Sila/Mark/Kristoufek/Weber, Financial Innovation 2025.
- **R5** Turnover dies via partial-adjustment toward an aim portfolio + no-trade bands + turnover-adjusted IR. Garleanu-Pedersen; Constantinides 1986; Gray et al Smart Rebalancing FAJ 2024.
- **REFUTED (do NOT build):** regime-switching "guaranteed Sharpe"; 3-state MS-GARCH; 27-expert MoE; max-Sharpe Euler-Mascheroni formula; one paper's specific IC numbers.
- **CAVEAT:** sources are US-equity/synthetic, UNVALIDATED on crypto 4H → every layer ships SHADOW-first and is promoted only on live evidence. **Problem 6 (quiet-regime alpha) is OPEN — no off-the-shelf answer.**

---

## The six layers (problem → method → sized → code/design → status)

### Layer 1 — Regime-aware signal application *(biggest recoverable lever)*
- **Problem:** signals with strong per-regime IC applied sign-blind → cancel to ~0.
- **Method (R1):** GMM regime (already present) → per-`(signal, regime)` sign/weight from rolling IC. **Selective** — only signals that flip AND hold their per-regime sign out-of-sample.
- **Sized — TWO tests, and they DISAGREE (this is the headline caution):**
  - 50/50 split (learn-1st-half/test-2nd, `C:/tmp/regime_sign_fusion.py`): quant blind −7.1 → regime-sign **+2.9bps**; llm_sentiment −4.0 → +5.1. Looked like recoverable alpha.
  - **WALK-FORWARD OOS** (learn-from-past-only, the honest test, `C:/tmp/validate_regime_ic.py` via `signals/regime_ic_fusion.py`): blind quant **+11.8bps/55%** vs regime-IC fusion **+8.6bps/59%**. **The fusion does NOT beat blind quant per-decision** — blending the weaker agents dilutes quant's edge. The 50/50 "+10bps" did NOT survive walk-forward.
- **Correction:** blind `quant`'s *sign* has real edge (+11.8bps/55% walk-forward); the −0.018 Spearman IC was a rank-correlation artifact that understated the sign edge. quant is NOT noise — it is the best single signal here.
- **STATUS (2026-06-13): module BUILT** (`signals/regime_ic_fusion.py`, shadow-only, 9 tests) but **OOS does NOT justify enforcing.** Stays SHADOW; promote only if it beats blind-quant on *forward* (post-2026-06-13) data. Do NOT tune it to beat this period (in-sample overfit). **The validation discipline caught a non-win before deploy — working as intended.**
- **Code+design.** SHADOW only for now.

### Layer 2 — IC/decay-weighted combination
- **Problem:** equal-ish authority fusion lets dead/inverted agents dilute the blend.
- **Method (R2):** weight each signal by `rolling_IC × persistence`; zero-IC ⇒ zero weight automatically (the principled "demote the dead/inverted" without hardcoding). This is the `LiveICBucketGate` from the companion prompt, generalized from gate→weight.
- **Code+design.** Subsumes the IC-gate prompt; shares its rolling-IC infra with Layer 1.

### Layer 3 — Model-validation harness (stop trusting backtest Sharpe) **(BUILT 2026-06-14)**
- **Problem (the root DRL flaw):** `training/train_drl_full.py:1488` selects `best_fold = max(folds, key=mean_reward)` — picking the best of 3 folds is selection bias (no deflation), validated on RL *reward* not realized OOS Sharpe, `purge_window` default 0. That IS the mechanism: backtest Sharpe 9-10 → live −2.62. The purged K-fold *structure* is fine; the SELECTION + METRIC are wrong.
- **Built:** `analytics/validation/sharpe_validation.py` (PSR + Deflated Sharpe + backtest-vs-live) + `analytics/validation/cpcv.py` (Combinatorial Purged splits + **CSCV PBO** — Bailey/López de Prado Probability of Backtest Overfitting: fraction of combinatorial IS/OOS partitions where the IS-best config ranks below the OOS median). Tests: `tests/test_sharpe_validation.py` (9), `tests/test_cpcv.py` (5). This is the **pre-promotion gate** that replaces max-reward selection — it would have CAUGHT the DRL overfit at selection time.
- **Trend-following re-validated through it (honest result):** PBO over an 8-point param grid = **0.42 (MARGINAL)**; param Sharpes range 0.12–0.75 (mostly ~0.4); OOS-Sharpe of the IS-best param ≈ +0.47. So the edge is **real and OOS-positive (~0.4 Sharpe) but param-selection-sensitive** — use textbook defaults, do NOT chase the lucky 0.75 (PBO says that's partly luck). Tempers the single-split 0.53 down to a robust ~0.4.
- **Apply it to:** DRL (re-select via DSR on realized OOS Sharpe, not reward — likely fails → demote) and the 4 v5.1 strategies (promoted with ZERO validation). Required gate before any model promotion.

### Layer 4 — Beta budget + instrument-level perp hedge
- **Problem:** +0.54 net-long, no exposure control.
- **Method (R4):** a net-signed-exposure **beta budget** (cap |Σ signed notional|) + **per-instrument perp hedges** (BTC-perp hedges BTC) — NOT a basket/index hedge (proven weak in crypto). **Independently validates the operator's Coinbase perp sleeve as the correct instrument.**
- **Design (new control).** Coordinate with the operator's active Coinbase two-sleeve work — do NOT duplicate.

### Layer 5 — Aim-portfolio execution (formalize Layer-2)
- **Problem:** churn = 84% of gross alpha (partly fixed: min-hold 12h + flip-persistence are LIVE).
- **Method (R5):** generalize to Garleanu-Pedersen **partial adjustment toward an aim portfolio** (trade a fraction toward target, fraction ↓ as cost ↑) + **no-trade bands** (Constantinides) + **turnover-adjusted IR** for go/no-go.
- **Code.** Mostly shipped; this is the principled generalization.

### Layer 6 — ROOT-CAUSE STRATEGY PIVOT **(VALIDATED 2026-06-14)**
- **Root cause (the real one):** the system's strategy CLASS — directional-ensemble single-asset 4H timing — has no edge (live Sharpe −2.62, PSR 21%). Patching its signals/fusion can't fix a strategy class that doesn't work. The fix is to REPLACE the decision layer.
- **The replacement, BUILT + VALIDATED:** `strategies/trend_following.py` — vol-targeted time-series trend-following (textbook params, NOT tuned). Backtest on 5.3y BTC/ETH/SOL 4H, 15bps cost, no lookahead: per-asset Sharpe 0.26/0.33/0.73; **equal-risk 3-asset portfolio Sharpe 0.53, PSR 89%, maxDD −13.6%, +5.4%/yr.** Module reproduces it (0.52/88%). Tests: `tests/test_trend_following.py`. **vs the current engine's Sharpe −2.62 / PSR 21%** — a real, modest, positive, regime-robust edge (it goes SHORT in downtrends — the regime that crushed the net-long engine).
- **Honest bounds:** Sharpe ~0.5 is modest (consistent with the research's ~1.0 realistic ceiling), PSR 89% is positive-leaning not iron-clad (<95%), crypto trend edge concentrates in big-trend years and is flat/negative in chop. This is a *first-pass* validation (textbook params avoid in-sample tuning, but no purged-CV yet). It is FAR better than the current negative-edge engine, not a guaranteed winner.
- **Rollout (gated, NOT shipped live):** (1) wire trend-following as a SHADOW strategy — log its target_position vs the live engine each tick; (2) confirm on FORWARD data + purged-CV via the Layer-3 harness; (3) promote it to the decision layer, composed with the net cap (P144) + carry sleeve (regime-gated, currently off). Coordinate with the operator (main.py + Coinbase actively edited). Do NOT flip the live decision layer without forward confirmation — that would repeat the backtest-vs-live trap.
- **Companion leg — regime-gated funding carry (VALIDATED 2026-06-14):** delta-neutral cash-and-carry (long spot / short perp) backtested on 5.5y real funding (`C:/tmp/{funding_long.json,backtest_carry.py}`). Gross yield: **2021 froth +29–38%/yr, 2024 bull +12–14%/yr; full-5y BTC +11% / ETH +11.9%** (real structural edge, uncorrelated to trend) but **SOL only +0.8%** (volatile, negative-funding spells → EXCLUDE SOL). **Heavily regime-gated** — thin in 2022/2023/2025/2026, and **~+0.0%/yr right now** (dormant). DECISION: do NOT build the sleeve code now (would be dead, ~0-EV). **Activation rule:** turn a BTC/ETH delta-neutral carry sleeve ON only when rolling annualized funding clears ~+12%/yr (covers fees + the BIS-documented liquidation/basis risk on the short leg); OFF in the current regime. Build when it next pays.

---

## Operator decisions required
- **D1** Layer-1 OOS-stability gate: min samples per `(signal,regime)` bucket + sign-stability test before enforcing (provisional: ≥30 post-fix samples, sign consistent across 2 sub-windows).
- **D2** Layer-2 weighting: cap per-signal weight? floor at 0 or allow negative (auto-invert) weights? (Recommend: clip ≥0; inversion is risky — prefer Layer-1 regime-sign for that.)
- **D3** Layer-3: does a failed DSR/CPCV **block** promotion (hard gate) or just **warn**? (Recommend hard gate for DECIDE-authority; warn for ADVISE.)
- **D4** Layer-4 beta budget: target net-beta band (e.g. [−0.2, +0.2]) and whether to auto-hedge via perp or just cap entries.
- **D5** Sequencing vs the operator's live v5.1 + Coinbase migration — Layers 3 & 4 overlap their domain; align before building.

## Iron Laws (sustained)
1. SHADOW-first, post-fix live data only. No layer enforces on pre-2026-06-13 / in-sample evidence.
2. Every layer reversible by config/env flag; fail-open (a validation/IC bug never blocks trading silently — it WARNs).
3. Reduces/exits/safety paths never gated by Layers 1–4.
4. Confidence is NOT a feature anywhere (measured anti-predictive).
5. Academic methods are crypto-4H-UNVALIDATED until the live-IC/DSR evidence says otherwise.

## Failure modes
- **Small-sample regime buckets** (Layer 1/2) — sparsity → gate never fires or overfits noise. Mitigate: min-sample floor, coarsen regimes, log per-bucket n.
- **Reflexivity** — a down-weighted signal stops generating outcomes; keep recording counterfactual IC so it can recover.
- **Beta-hedge basis risk** (Layer 4) — perp funding/basis can dominate; monitor hedge cost vs variance reduction (R4 says only worth it per-instrument).
- **Double-gating** — Layers 1/2/3 + the existing alpha gate + B1 + Layer-2 churn can compound into zero trades. Track the funnel; a negative-alpha system SHOULD trade less, but watch for full starvation.
- **Operator collision** — Layers 3/4 touch DRL promotion + Coinbase hedging (operator-active). Coordinate.

## Output checklist (when ready)
- [ ] `analytics/validation/` DSR + CPCV + RL-test; retro-run on TQC + v5.1 (Layer 3).
- [ ] `risk/live_ic_bucket_gate.py` extended gate→weight (Layer 2) + regime-sign table (Layer 1), SHADOW mode.
- [ ] Beta-budget control + perp-hedge hook coordinated with Coinbase sleeve (Layer 4).
- [ ] Aim-portfolio partial-adjust + no-trade band formalization (Layer 5).
- [ ] Layer-6 experiment harness (quiet-regime candidates).
- [ ] CLAUDE.md P-entry + memory; CI gate + verify green; SHADOW deploy; ≥21d evidence before any enforce.
