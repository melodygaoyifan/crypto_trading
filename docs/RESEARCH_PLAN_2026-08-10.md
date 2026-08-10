# Research Plan 2026-08-10 — exploit the measured edge; stop hunting a bigger one

**Provenance:** research-mode pass 2026-08-10 (operator-instructed): internal
evidence synthesis (clean-parquet edge probe, p250 lab artifacts, live ledger
inventory verified ON THE SERVER) + external literature/practice scan
(2024-2026, net-of-cost sources). Supersedes nothing — this SEQUENCES the
standing Plan V3 machinery against the current calendar.

---

## 1. The situation, in five measured facts

1. **The honest signal ceiling is thin, linear, and 16h:** clean-parquet
   strict probe — BTC 16h ALL/ridge IC 0.089 (t=6.8, +33.7bps q75); ETH
   0.053 (+35.8bps); SOL external 0.074 (+45.5bps). Nonlinear buys nothing
   (HGB−ridge gap ≈ 0, thrice measured). Break-even IC for naive per-bar
   trading ≈ 0.13 — **no measured signal clears naive trading; several clear
   cost-aware exploitation.**
2. **The `external` (flow/positioning) group is the only one clearing the
   cost bar on ALL THREE assets at 16h**, with ridge AND HGB agreeing —
   the strongest cross-sectional fact in the whole campaign.
3. **The live direction seat is held by an unvalidated constant** (trend
   `base_edge_bps=40`, calibrator: GATE-CLOSED ×3) with a tripwire dated
   2026-09-01; the validated-candidate roster (regime books) is shadow-only
   with its 30d P166 clock started 2026-08-10 (deploy) → readable ~09-09;
   **no seat wiring exists for a passing candidate.**
4. **External research corroborates every internal verdict** with numbers:
   carry Sharpe 6.45→negative in 2025 (Borri et al.; BIS WP1087); flow
   features predict but fail taker costs STANDALONE in every 2025-26
   net-of-cost study (tilts survive); RL's cost-sensitivity is survey-level
   consensus (our 0/9×2 is the expected result); surviving trend is much
   slower than 4H.
5. **The canonical mechanisms for exploiting IC 0.03-0.09 exist and we run
   crude versions of two:** Gârleanu-Pedersen partial adjustment ("trade a
   fraction toward an aim that overweights slow signals") ≈ our di=4 +
   hold-band; jump-model regime switching with penalty CV'd on strategy
   performance WITH costs+delay (Shu/Mulvey 2024) ≈ the named upgrade for
   our measured regime-turn losses (ETH fold_1 −32% at the SMA200 lag).

**The strategic conclusion both evidence streams force:** the bottleneck is
not model capacity and not pipeline machinery — it is information content vs
cost structure. Raising measurement quality (done) cannot raise the ceiling;
only NEW INFORMATION (flow/positioning features), LOWER EFFECTIVE COSTS
(turnover mechanics), or DIFFERENT PROBLEM STRUCTURE (regime-conditioning)
can. Therefore: **exploit-mechanics over signal-hunting.**

---

## 2. The plan

### Week 0 (now → 2026-08-17): stage the succession, verify the foundation
- **W0.1 Seat wiring (the gap):** default-off `regimebook` → sleeve-driver
  injection path (analogous to the trend injection; P206 translation rules
  apply). Built and tested NOW so the ~09-09 gate verdict has an actuator.
  Flag: `regimebook_mode: off|shadow|enforce` (P141: enforce = operator flip).
- **W0.2 External-group cleanliness check:** leave-one-out probe over the 6
  `_EXTERNAL` features (esp. WITHOUT `funding_rate_zscore` — the group's
  clear must survive removing the ex-leak column entirely — and WITHOUT
  `liq_imbalance`, whose 180d CoinGlass depth is a coverage confound).
  ~10 min CPU. If the group's clear collapses without one column, that
  column IS the finding and gets its own scrutiny.
- **W0.3 Pre-commit the September decision tree** (see §3) so neither date
  arrives as a surprise.

### Weeks 1–4 (design-era work only; ZERO validation reads; ZERO GPU on RL/transformers)
- **W1 Partial-adjustment execution (the biggest lever per both evidence
  streams):** formalize trade-toward-aim for the sleeve — aim = book/tilt
  target, cost-scaled asymmetric no-trade band (cheap to hold, expensive to
  flip), acting cadence matched to the 16h signal horizon. The lab already
  prices this (turnover cost columns); candidate enters the ladder as a
  MECHANISM wrapping existing books, not a new forecaster.
- **W2 Jump-model regime switch** (Shu/Mulvey protocol: jump penalty tuned
  by CV on STRATEGY performance with costs + 1-bar delay) as the SMA200
  replacement candidate for the same books. This attacks the one loss mode
  the books demonstrably have (regime-turn lag).
- **W3 fv2-as-tilt on SOL/BTC** through the lab ladder (16h horizon only;
  SOL 4h after-cost is 1bp — dead). Never standalone (external evidence:
  flow fails taker costs standalone everywhere).
- **W4 SOL vol-bucket filter probe** (P249's loss concentration: −59.8% of
  PnL in LOW vol) — one script, ladder rule, design era. NOTE: P249's
  "earned feature set" was measured on leaked X and is NOT the candidate;
  the vol-bucket filter is.
- **Explicitly de-funded:** TQC/RL retraining (0/9 twice + literature
  consensus; the SOL churn study stays parked at its resumable sqlite),
  transformers, new data vendors, any funding-carry harvesting (inversion
  is market-wide and unconditioned).

### September decision window (pre-committed)
| date | event | action |
|---|---|---|
| 09-01 | Tripwire report #4 | If GATE-CLOSED persists for an asset and no promotable basis exists: that asset's trend injection comes OFF (`trend_assets` edit) → the asset goes FLAT. **Flat is a position; stated in advance so it is a plan, not a surprise.** |
| ~09-07 | ma_filter ledger reaches 30d | `compute_shadow_ic --window-days 30`. Passes P166 → flip `coinbase_ma_filter_enforce` (gate already built). Fails → the +24.9/−78.9 split was window-picked; record and move on. |
| ~09-09 | regimebook ledger reaches 30d | P166 read per asset. BTC book passes → seat via W0.1 wiring (operator flip). ETH: trend-only IS the book — continuation. SOL: hold-bull/flat stands unless the W3/W4 candidates earn a shadow slot. |
| ~09-09 | derivflow 30d | P166: promote one of squeeze/exhaustion, or kill both (both-noise is an expected outcome). |

### The standing rule this plan operates under
Every candidate above is design-era work → ONE ledgered validation read →
forward shadow → P166. Nothing promotes from backtest. The scarce resource
is unread forward windows, not GPU.

---

## 3. Why retraining keeps "failing" (the operator's question, answered)

Retraining has NOT been rejected — **eight full campaigns ran in the seven
days to 2026-08-09** (official_p221b TQC ×3 assets, p242_run1, p243/p243b,
p244, p245, p246 six-cell, p247 leakfix rerun, p249 label lab, p250 full
ladder: 6 model families × 18 cells). What keeps happening is that the
RESULTS fail the promotion gates. The EDA/feature-engineering/model-selection
changes were real — but almost every one of them was the REMOVAL of a way
the old pipeline had been fooling itself (0-fee eval, reward-hacked
selection, wavelet leak, GMM leak, funding leak, future-peeking oracle,
strawman baselines). **Improving the pipeline therefore made the numbers
WORSE, because the old numbers were fiction** — the wavelet leak alone
produces IC +0.41 on a pure random walk. The new machinery converges the
measurement; it cannot add information to the data. What the data holds is
now measured three independent ways at IC 0.03-0.09 (linear, 16h), against
a ~0.13 naive break-even — a gap that is a property of DATA + COSTS, and
that no amount of pipeline sophistication moves. What moves it is §2: new
information (flow/positioning), lower effective costs (partial adjustment),
and conditioning (regime switch). And note what DID pass the honest
pipeline: the BTC assembly (+33.6% validation, zero fragility flags), ETH
trend-only, the external-group probe — the gate is not "reject everything";
it is "forward evidence before money", and the first forward window only
started counting on 2026-08-10.
