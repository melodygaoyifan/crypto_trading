# Task-Separation Plan — each model on the job it is best at (2026-08-24)

**Instruction:** separate the trading problem into sub-tasks, assign each to the
model class that excels at it, make a plan, act on it.

**Answer up front:** the system is *already* task-separated, and correctly — the
plan below mostly *formalizes* an assignment that is implicit today, and confirms
each live component is the right model for its task. Separation is sound
engineering; it does **not** create edge, because the one alpha-bearing task
(direction) has a signal (~IC 0.04) below the flat-fee floor (~0.07–0.11), and
no re-assignment of tasks changes that number.

---

## The decomposition — task -> best-fit model -> status -> verdict

| # | sub-task | question | best-fit model class | in HMATS today | verdict |
|---|---|---|---|---|---|
| 1 | **Regime detection** | what state is the market in? | **unsupervised** (GMM) | split-aware GMM, paired, ready | ✓ correct tool; but regime labels carry ~0.01 direct IC (a gate/feature, not a forecaster — P242) |
| 2 | **Direction forecast** | up or down next bar? | **regularized linear** / **smooth trend rule** | ridge zoo + **SMA200 regimebook (live, 6y-certified)** | SMA200 is the certified winner; ML direction ~0.04, below fee floor |
| 3 | **Regime x direction gating** | *when* is the direction signal reliable? | GMM gates the direction model | **`regime_ridge`** (per-GMM-regime ridge), `composite_bull`, `trend_regime_gate` | **tried, failed OOS** — P250 pooled −71%/−16%; P243 composite overfit pre-design |
| 4 | **Position sizing** | how much? | **vol-parity / risk rules** | P273/P370 vol-parity fractions | ✓ done |
| 5 | **Execution** | how to enter/exit cheaply? | **RL / optimization** (tick / L2) | maker-first (P270) | blocked: CDE charges a FLAT per-contract fee (P315/P374), so maker can't reduce it, and RL execution needs tick/L2 data we don't have at profit |
| 6 | **Risk / exit** | when to bail? | **rules + watchdog** | FastRisk, halts, venue stops | ✓ done |

**Reading the table:** five of six tasks already use their best-fit model. The
sixth (execution RL) is blocked by the venue's fee structure, not by a modelling
choice. The alpha-bearing task is #2, and #3 is the *only* place separation could
still add alpha — and #3 has been built and measured dead OOS.

---

## The evidence that separation is correct but doesn't unlock profit

- **Direction task, fresh on corrected parquets (P392 zoo, this run):** BTC winner
  `mlp|volscaled|top24` CV Sharpe **+1.41 -> LOCKBOX -0.01** (B&H +2%). Best of 196
  trials, GRU included, collapses OOS. The direction task has no tradeable model —
  the P281/P285 overfit pattern, reconfirmed on 2026-08-23 data.
- **Regime-gated direction (task #3), already run:** `regime_ridge` (the per-GMM-
  regime specialist — literally "let the unsupervised regime model gate the
  supervised direction model") pooled **−71% / −16%** (P250); the coarse
  `composite_bull` beat baselines in-sample and **inverted pre-design** (P243).
  The measured regime-conditional IC sign-flips (P242) are real in-sample and do
  not survive OOS — the finer the regime split, the worse the overfit.
- **Sequence specialists (LSTM/GRU/TCN/DT):** 0/6 at honest cost (P286); the
  LSTM-FiLM DRL is 0/23. No temporal structure to exploit beyond a stacked-window
  ridge (P242 nonlinearity residual negative).

The through-line: every task-separated component is the right tool, and the
combination is near break-even because the alpha source (#2) is capped by the
data, and the conditioning that could rescue it (#3) overfits OOS.

---

## The plan (what to ACT on)

**A. Formalize the assignment (done — this document).** The role of each model is
now explicit, which prevents the recurring "throw a bigger model at direction"
churn (deep/RL on #2 is measured dead; that decision is now written down).

**B. Reframe the deliverable to what the separated system actually produces.** The
task-separated stack is not an alpha engine at this scale — it is a **drawdown-
reduction overlay**: the SMA200 trend/hold rule (#2 live) keeps ~70% of buy-hold
return at ~half the max drawdown, higher Sharpe (P377). That is the honest product
of correct task separation here: *own crypto with less pain*, not *beat the market*.

**C. Gate any further #3 work on the direction task first.** A regime-gated
composite is only worth re-testing if the direction signal it conditions has
edge to condition — and the zoo (B) says it does not on current data. So no new
regime-gating experiment is justified until the direction task changes, which
requires a new data basis (#2 input), not a new model.

**D. The levers that move #2, none of them a model choice:**
- new data basis (options-chain / on-chain — acquisition, paid; free slices dead: P385/P388/P391);
- scale (lowers fee in bps — but CDE floors at ~14bps RT spread+impact, insufficient; a percentage-fee venue clears it, operator-ruled-out);
- execution venue with tick/L2 + percentage fees (unblocks task #5 for RL).

**What NOT to do:** re-run deep/RL on task #2 (dead), re-fit `regime_ridge` on the
same data (dead OOS), or add model capacity anywhere — the constraint is signal
and cost, which live upstream of every model in the table.
