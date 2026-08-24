# Profitability Root-Cause Plan — 2026-08-24

Written after the P381–P386 measurement campaign. This is a plan for **root-cause
fixes and the end-to-end changes each one implies**, honest about which are code
and which are not. It supersedes ad-hoc "retrain and hope" as the roadmap.

---

## 1. The root cause (measured this session, not asserted)

The system does not generate alpha, and the DRL contributes nothing. **Neither is a
code bug.** Two constraints, each measured:

- **(A) Signal quality.** The extractable predictive signal in 4H crypto direction
  is **IC ≈ 0.04–0.05, t ≈ 3** — real but weak — and this ceiling is identical
  across **five independent data bases**: single-asset features (P385), higher-freq
  order flow (P375b), L2 order-book depth (P379), tick microstructure / VPIN
  (P385b), and cross-asset breadth (P385c). It is the information content of the
  problem at this cadence, not a model or feature gap.
- **(B) Scale / fee.** The flat per-contract CDE fee on ~$11k is ~2× that signal
  (required IC 0.07–0.11 vs achievable 0.04). Holding a position instead of
  round-tripping every bar cuts the fee ~20× (P386) — the correct execution — but
  the IC-0.04 signal is too **noisy** to hold profitably beyond what the SMA200
  rule already extracts (P386: its best held config is a lookahead-picked deadband;
  neighbours fail).

**The DRL's failure (0/23 folds, P381/P381b) is a symptom of (A), not a cause.** It
churns (483 trades even at decision-interval 8) because it is not a smooth,
committable rule; its held-equivalent is the overfit ML signal from P386.

**What already works** is the live **SMA200 regimebook rule**: a smooth
hold-while-trend-persists position that trades ~6–13×/yr, certified net-positive
after honest cost on ETH/SOL (P321) — a drawdown-reduction product, not alpha over
holding. The system already implements "hold when we predict up."

---

## 2. What does NOT work — do not re-attempt without a new premise

Each is measured, not assumed. Re-running any of these is burning compute to
reconfirm a settled verdict (the Trade-Frequency anti-pattern):

- Retraining DRL/supervised models on the **current** data (0/23 clean folds).
- Any of the **five local new-data bases** above (all fail the cheap Rung-0 gate).
- **Per-bar** trading (fee-ruinous). The fix — holding — is already live via SMA200.
- **Mass agent/flag/model removal** (P377: risk without benefit — 50+ test files,
  recorded decisions, live-path code, zero profit gain).

---

## 3. The real levers (root-cause fixes), sequenced by cost then evidence

### Lever 1 — SCALE / VENUE  (highest leverage; NON-CODE; operator decision)
The fee floor is the binding constraint (§1B). Drop it and the signal we ALREADY
have becomes tradeable: at ~10bps round-trip (a percentage-fee venue) ETH/SOL clear;
at ~3bps (deep scale amortizing the flat fee, or maker-heavy) BTC clears (P385).
- **Action:** operator sources a percentage-fee venue OR adds capital (the flat
  per-contract fee in bps falls as notional rises; the code already sizes to equity,
  P274, so arriving capital deploys itself).
- **Code changes required: none to unlock; see §4 for the sizing/fee re-pricing that
  follows.**

### Lever 2 — NEW DATA that clears the gate  (CODE + acquisition; gated)
The five local bases are exhausted. The untested ones are acquisitions, not files:
- **options-native**: funding term structure, open-interest curve, gamma / dealer
  positioning (Deribit / CoinGlass paid tiers).
- **on-chain flows**: exchange net-flows, stablecoin supply, whale wallets (paid).
- **Gate (mandatory, cheap, BEFORE any GPU):** the **hold-aware** edge probe (§5) —
  a basis must show a held-position OOS net > buy-and-hold with a **pre-committed,
  walk-forward-selected** deadband (never a lookahead sweep), on >=2/3 assets. Only
  then does the end-to-end retrain in §4 begin.

### Lever 3 — SIGNAL SMOOTHNESS  (CODE; research; low probability, honest)
P386's real finding: the ML signal fails because it is NOISY, not because it is
executed wrong. So the research direction is a signal as **smooth/committable as
SMA200 but carrying more edge** — not a higher per-bar IC.
- Candidates: multi-horizon trend ensembles, regime-conditional trend rules,
  denoised/persistence-weighted targets, jump-model regime switching.
- **Honest prior:** this space is largely explored — donchian/emaens were era-fragile
  (P288), the overlay reframe borderline (P377), and SMA200 is the certified
  survivor (P262). Pursue only after Levers 1–2; expect SMA200 to remain the bar.

---

## 4. End-to-end change spec — what changes in EACH layer when a lever is pursued

This is the "change other agents/strategies/modules correspondingly" part. It is a
SPEC, executed only when Lever 1 or 2 fires — not now.

| layer | change when a new BASIS clears (Lever 2) | change when the VENUE/SCALE changes (Lever 1) | invariant to hold |
|---|---|---|---|
| **data / features** | add the new feature columns to the parquet build (`rebuild_pipeline.py` + `build_flow_features.py` STEP 5b) | none (fee is a runtime constant) | causal construction test (P164); no lookahead |
| **GMM** | REFIT split-aware on the new feature set, as part of the rebuild; deploy paired | none | {GMM, parquets, checkpoints} move as ONE versioned set (P215); split-aware (P280 gate) |
| **DRL (TQC)** | retrain on the new basis, di>=4, honest fees, P182 gate; judge fold_1 first (P258) | retrain at the new fee (it changes the reward's cost term) | live only after Rung-3 forward shadow + P166 gate; never on backtest (P141) |
| **supervised zoo** | re-run `train_supervised_full` on the new basis (cheaper than DRL; often the least-dead class) | re-price COST_BPS to the venue | ridge lockbox recert required (P281) |
| **regimebook (live decider)** | unchanged unless the new basis yields a smoother rule that beats SMA200 out-of-sample | unchanged (rule is fee-agnostic; the alpha GATE re-prices) | 6y OOS certification bar (P262) |
| **agents / fusion** | a new signal enters as ADVISE + a shadow ledger FIRST; promotion only via forward IC through P166; it reaches an order only as a DECIDE seat after certification | none | P293d (only DECIDE sets direction); 3-file contract (P8); no bulk arming (P228) |
| **alpha gate** | none | re-price friction to the new venue (`resolve_venue_fee_bps`); at a percentage venue BTC's 24bps edge clears | round-trip cost, 2 legs (P167); the gate stays the honest arbiter |
| **sleeve / execution** | none | sizing already equity-scaled (P274); maker-first already live (P270); confirm caps/halts at new scale | venue-authoritative reconcile (P139); protective stop (P197) |
| **risk** | none | re-confirm the sleeve drawdown halt + net cap at the new notional | halts fed live equity (P351); fuse (P209) |

**The sequence for Lever 2 is fixed:** Rung-0 hold-aware probe → (if clears) features
→ GMM refit (paired) → DRL/supervised retrain → 30d forward shadow → P166 gate →
DECIDE seat. Each arrow is a gate; a failure at any arrow stops and is recorded.

---

## 5. The methodological root-cause fix — DONE now (the one genuine code fix)

P386 exposed a real defect in HOW we judged signals: the Rung-0 edge probe (P385)
charged cost **per bar**, as if every 4h prediction is a round trip. That overstated
the fee barrier ~20× and would falsely reject a signal that is fine when HELD. That
is a methodology bug, and it is fixable:

- **Adopt hold-aware evaluation as the standard signal gate.**
  `training/scripts/signal_hold_backtest.py` (P386) is the correct evaluator:
  position held across bars, fee on flips only, long/short + funding, deadband
  selected walk-forward (not swept with lookahead). Any future "is there edge after
  cost?" question is answered with this, not the per-bar probe.
- **`edge_probe.py` carries a caveat** pointing to the hold-aware evaluator, so its
  per-bar "NO_EDGE" is read as the worst-case bound, not the verdict.

This does not create edge (P386 shows the current signal still fails hold-aware), but
it ensures the NEXT candidate basis (Lever 2) is judged correctly rather than
falsely rejected — which is the actual root-cause fix available in code today.

---

## 6. Deliberately NOT changed (and why)

- **No live architectural refactor / agent removal.** The working strategy (SMA200
  hold) is already live and certified; the evidence-only agent layer is by-design
  (P228/P293d), and ripping it out is risk-without-benefit (P377).
- **No DRL retrain now.** Every probed basis fails Rung-0; a retrain reproduces the
  negative (P381/P385c). Retrain only when a basis clears §5's gate.
- **No fee/gate weakening to manufacture a pass.** That is the P164/P179 artifact
  that produced the old fake +9 Sharpes; a number made positive by removing honest
  cost loses real money.

---

## 7. One-line summary

The system already does the right thing (hold a certified trend rule); it does not
make alpha because the signal is IC-0.04 and the fee floor is ~2× that. The only
root-cause fixes are **more scale / a percentage venue** (non-code, unlocks the
current signal) and **genuinely new data** (code + acquisition, gated on the
now-corrected hold-aware probe). Everything else is readiness, and the readiness
spec is §4.
