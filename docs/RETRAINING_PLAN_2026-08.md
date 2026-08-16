# The Model Retraining Plan (2026-08-16, P283)

**Supersedes** TRAINING_PIPELINE_PLAN_V3 as the retraining roadmap (V3's
methodology stages remain valid; its spot cells, ±1 sizing and 3-asset
universe are retired — P279 §3). Grounded in the measured record through
P282; every claim below has a P-ledger citation.

## The organizing principle

**Retraining is EVENT-DRIVEN, not calendar-driven.** Every prior campaign
that ran on schedule or enthusiasm re-measured a settled verdict (P258).
Every stage below has a TRIGGER (an evidence event), an ACTION, a BAR, and
a KILL. Between triggers, the correct training activity is zero — the
scarce resource is unread forward windows, not GPU (RESEARCH_PLAN §2, re-
confirmed P281: the decisive reads cost minutes; the waiting is the data).

## Where the evidence stands (the plan's inputs)

- **Certified**: the trend/hold rule books (virgin-era + never-fitted-asset
  transfer, P262). The only certified class. Live as the regimebook seat
  candidate.
- **Measured dead from history, under fully honest conditions (P281)**:
  every trained-model class on the current feature basis — RL 0/39
  (fee-independent causes), supervised fold-level 0/9, fresh 196-trial
  searches (CV winner → lockbox −0.03), and the ridge_a30 incumbent's
  re-certification (+0.27/+0.11 vs the leaked-era +1.28/+1.73).
- **Real but uncertifiable residue**: models beat BOTH baselines in 3/9
  honest folds (was 0/18) — blocked only by the fold-length CI criterion
  that nothing, including buy-and-hold, can pass at ~1,900 bars (P242).
- **"Enough models?" — closed by measurement (P282)**: ~16 families tried;
  the nonlinearity residual is NEGATIVE (HGB−ridge −0.012); untried
  families lack a mechanism. Capacity does not bind; information does.
- **The three channels that can change the answer**: (1) measured cost
  (maker fill rate), (2) new venue-true data (accruing), (3) forward
  certification (twelve P166 exams).

## Stage R0 — NOW, continuous: evidence accrual (no training)

Running already; the plan's job is to not interrupt it.
- Maker fill rate: `scripts/maker_fill_review.py` weekly (first reading
  1 maker/3 taker, n=4 — UNMEASURED; verdict at n≥20).
- CDE funding history: `data/cde_funding_history.jsonl` accrues every tick
  (started P282 — the venue-true series no API can backfill).
- Twelve forward ledgers + weekly `scripts/september_check.py`.
- Data hygiene on rails: `make refresh-data` (~Sep 1-2 for the August
  archive; the futures fetch is now in the chain, P281).

## Stage R1 — TRIGGER: the operator disposes the certification criterion
*(the P242 standing question, re-demanded by the P282 review — currently
the only stage blocked on a human)*

- **IF pooled/lockbox-length certification is adopted** (recommended:
  P243 built the pooled machinery; multiplicity control via deflated
  Sharpe per P262): re-examine the three beat-both-baselines families
  (BTC mlp_small, ETH composite_bull_ridge, SOL hgb_small) on POOLED
  cross-fold windows — CPU, minutes, no new selection (the candidates are
  fixed by the p281 run). BAR: pooled after-cost CI excluding zero at the
  honest cost, DSR-adjusted for the zoo's trial count. KILL: fails →
  the supervised class is closed until R3/R4; the zoo is not re-run again
  under any criterion.
- **IF the per-fold criterion is kept**: the zoo is never re-run (running
  a search under an unpassable bar converts compute to zero bits).

## Stage R2 — TRIGGER: `maker_fill_review` returns a verdict (n≥20)

- **UNLOCKED (effective RT ≤ ~3bps)**: re-price the P166 forward-exam
  edge bar to the MEASURED cost (edge ≥ 2× measured RT) — this changes
  what the September candidates must clear, in their favor, honestly. It
  does NOT resurrect the recert incumbent (its Sharpe gap is ~1.0+, not a
  cost-sized gap — recorded so nobody re-litigates).
- **NOT unlocked**: the taker bar stands; nothing changes (overcharging
  is the safe direction, P167/P278).

## Stage R3 — TRIGGER: any September P166 PASS (~Sep 7/9/15)

Per the pre-committed decision tree (docs/SEPTEMBER_DECISION_TREE.md).
Training-relevant consequences by candidate type:
- **ma_filter / regimebook / volskip pass** → config enforcement, NO
  training (they are complete strategies).
- **derivflow / etfflow / stablecoinflow / oidiv / calbasis / xsmom
  pass** → the passing signal is a BASIS, not a strategy: design a bounded
  tilt/filter through the lab ladder (design era → pre-design →
  validation-read-before-ledger per P259b) — CPU-days at most.
- **A certified basis + a policy-layer question** → this, and only this,
  reopens learned models (the P258 rule), supervised-first, judged at the
  R2-measured cost. RL reopens last, and only after BOTH prerequisites:
  (a) the derivatives-native environment (the P279 ten-gap spec: discrete
  sized contracts under the shared net cap, funding carry per bar,
  gate/stop/flatten events, maker/taker fill model, multi-asset) and
  (b) the decision-interval runtime hold mechanism (currently a HARD
  Rung-3 block — zero runtime code exists, re-verified P282).
- **Everything fails** → the pre-committed branch: books-or-flat, no
  gate loosening, no dead-family re-litigation; R4 becomes the only path.

## Stage R4 — CALENDAR: the venue-true feature era (months out)

The first genuinely derivatives-native retrain becomes possible when the
accruing series reach useful depth:
- CDE funding: accruing from 2026-08-16 (this file's D-day).
- CoinGlass liq/OI 4h: 187d today, +1/day (P266 merge cadence ≤150d is
  load-bearing).
- calbasis: forward-only, persisted since P281.
- Breadth/xsmom training series: **backfillable today** (6y of closes
  banked, P281) — the one R4 item that can be built early if a
  cross-sectional design earns a lab slot.
Admission of ANY new column to the feature manifest is an obs-contract
change: its own P-entry, atomic {GMM, parquets, checkpoints} (P215), and
the P164 future-perturbation causality test at construction.

## Hard invariants (mechanical, not memorial)

Enforced in code — no plan step may bypass them: the clean-GMM gate at
every trainer/lab entry (P280, no override); ROUND-TRIP cost convention
(P281, cross-lab parity pinned); fresh tags; the window-usage ledger on
every validation/lockbox read; P259b ordering (spend unread history
BEFORE wiring a forward ledger); no ETF-history backfill (reporting-lag
leak); breadth assets unfitted (P262 virgin-evidence); kill-at-decision-
point; provenance triplet; EDGE_CANDIDATE is a necessary gate carrying
near-zero positive information (P282 — it preceded every dead campaign).

## What this plan explicitly does NOT contain

A scheduled retrain date. The record is unambiguous: every information-
bearing training event of this round cost minutes once its trigger was
real, and every scheduled campaign before it burned days re-measuring
settled verdicts. The next training dollar is spent by R1's operator
decision, R2's measurement, or R3's September verdicts — whichever fires
first.
