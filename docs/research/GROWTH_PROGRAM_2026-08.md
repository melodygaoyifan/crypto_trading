# HMATS Growth Program — 2026-08 (predictor + venue + capital)

**Origin.** Operator, 2026-08-22: "the system has systematic issues and even
can't make trade [narrower truth: it trades 2-5 nano contracts and doesn't beat
hold] ... we can make big changes ... our goal is a system that can predict
trade, make the order with correct amount." Chosen paths: **new-data predictor,
venue change, capital scale** (declined: "simplify + run as-is").

**The one-line thesis.** The three chosen paths are COUPLED, and the coupling is
the plan: a **percentage-fee venue** makes a **higher-frequency / order-flow
predictor** affordable, the predictor is the core build, and **capital** scales
it *after* it proves out — never before. Run in that order, behind gates, so no
large spend rests on faith.

**What is already settled (do not re-litigate).** Every trained-model class is
dead at honest cost on the CURRENT data basis: TQC 0/21 folds (P200/P258),
supervised zoo 0/18 (P281), GRU/TCN 0/6 (P286), meta-labeling dead (P249), and
both remaining offline signal rules — xsmom (P373), cross-sectional funding
carry (P374) — lose to buy-and-hold. **More models on the current data
re-measure ~0.** The predictor bet is therefore a NEW-DATA bet, not a new-model
bet. This is the P269/P283 discipline: the (N+1)th rule on the same window is the
mistake.

---

## The venue lever, in numbers (the decision input)

Certified per-round-trip edge vs cost, CDE flat fee vs a ~10bps percentage venue:

| asset | edge (bps/RT) | CDE net | %-venue net | delta |
|---|---|---|---|---|
| BTC | 24.1 | **-3.6** (untradeable) | **+14.1** | +17.7 |
| ETH | 88.1 | +44.1 | +78.1 | +34.0 |
| SOL | 221.7 | +180.7 | +211.7 | +31.0 |

A percentage venue turns BTC tradeable and widens ETH/SOL ~30bps/RT. The
second-order effect is larger: cheap fees make HIGHER-FREQUENCY trading
affordable, which is the strategy class where crypto edge more plausibly lives
(order-flow > price features; ~H6 optimum — Nguyen 2026, arXiv:2602.11708). That
higher-frequency class is what the predictor targets and what the current 4H
rule book cannot reach.

**THE BLOCKER, stated plainly and neutrally:** the venues that offer
percentage fees + fine granularity (Hyperliquid, Binance, Bybit perps) are
**US-restricted**; HMATS runs on Coinbase CDE precisely because offshore perps
are not US-legal (`core/exchange_guard.py`, CLAUDE.md). Whether a
percentage-fee venue is legally accessible to you is a **jurisdiction/legal
decision that is yours** — I will not implement or advise a way around a
regulatory restriction. Everything downstream (the predictor's target strategy
class, the capital math) forks on this answer, so it is Gate 0.

## The capital lever, in numbers

The CDE fee is flat per CONTRACT, so more nano contracts = same bps. Fee relief
comes only from (a) the BTI tier (1 BTC/contract), which needs **~$427k** for
usable granularity (P315), or (b) BTC price rising (nano notional up → fee bps
down: net +6 at $69k → +10 at $100k). **Below ~$427k on CDE, capital scales the
book without fixing its economics.** So capital is a LATE-stage lever — it
multiplies a proven-positive book, and multiplies a losing one just as fast.

---

## Sequenced program (gates prevent spending on faith)

### GATE 0 — Venue feasibility [OPERATOR, no build]
Decide whether a percentage-fee perp venue is legally accessible to you.
- **YES** → the predictor targets higher-frequency/order-flow (Path A below);
  BTC becomes tradeable on the current book immediately as a side benefit.
- **NO** → the predictor must clear the CDE flat-fee floor, which raises the
  edge bar sharply (a signal must beat ~28-44bps RT, not ~10). The honest read
  is that few higher-frequency edges survive a flat per-contract fee, so on CDE
  the realistic path is the current 4H book + capital to the BTI tier, and the
  predictor bet is weaker. Say which, because it changes everything downstream.

### STAGE 0 — Predictor Rung-0 edge probe [ME, no procurement, ~1 day]
Before anyone buys data or changes venue, establish whether a
higher-frequency/new-basis predictor has a PULSE, using data already on disk
(6y 60m closes + futures taker-flow columns for all 8 assets).
- Build a leakage-safe (P164 causal-construction test) edge probe at 1h/4h/12h
  horizons on 60m data + flow features, scored at BOTH CDE-flat and
  percentage-venue cost.
- **GATE:** does any group clear its required-IC bar at percentage-venue cost,
  at a horizon faster than 4H? Report per asset per horizon per cost.
- **Kill:** if even the higher-frequency probe on existing data is dead at
  percentage cost, the predictor needs genuinely exotic data (L2/tick), which
  raises the procurement bar — decide then whether it is worth it.

### STAGE 1 — Data basis + harness [OPERATOR procures, ME builds; only if Stage 0 has a pulse]
- **Operator:** procure the chosen data basis. Options by promise/cost:
  - **L2 order-book / tick microstructure** — biggest upside (execution/market-
    making class), needs a real-time feed + tick storage infra. Only worth it if
    Gate 0 = YES (needs cheap fees).
  - **High-fidelity on-chain** (CryptoQuant/Glassnode ~$30-100/mo) — modest;
    free versions were marginal (P293), paid untested.
  - **Options/vol surface** (Deribit) — partly used; extendable cheaply.
- **Me:** leakage-safe ingestion + backtest harness on the NEW features, with
  the P164 causal test and a held-out lockbox baked in from the start (the whole
  point is to not repeat P200's leaked pipeline).

### STAGE 2 — Build + train + forward-shadow the predictor [ME]
- Walk-forward, honest cost, three-baseline gate (P182: beat B&H + SMA + the
  incumbent rule book), DSR/multiplicity control (P285b), then a 30-day live
  SHADOW through the cost-aware forward gate (P166). Nothing trades real money
  from backtest.
- **GATE:** forward shadow IC clears the cost bar at the target venue's cost.
  **Kill:** no clean forward edge → the predictor bet has failed honestly;
  fall back to the certified rule book. (This is the outcome the evidence so
  far predicts; the program is built so failing here is cheap and clear.)

### STAGE 3 — Sizing + capital [ME builds sizing, OPERATOR funds]
- Correct-amount sizing (the operator's stated goal) = size by edge-to-cost
  ratio and vol, capped by the risk stack — buildable once a predictor with a
  measured edge exists (sizing a ~0-edge signal is meaningless).
- Scale capital only against a forward-proven book. On CDE that means the BTI
  tier at ~$427k; on a percentage venue, capital scales linearly at any size.

---

## STAGE 0 RESULT (run 2026-08-22): NO PULSE — the predictor gate FAILED on existing data

`training/scripts/edge_probe_hf.py`, report `training/reports/edge_probe_hf_p375.json`.
Higher-frequency order-flow probe on 6y of on-disk 60m data (real taker-buy /
trade-count / volume — the order-flow basis the literature favours), walk-forward
ridge, 1h/4h/12h horizons, scored at both cost models:

- **Every group, every horizon, every asset is net-NEGATIVE — even at the cheap
  10bps percentage-venue cost.** Spearman IC is 0.01-0.04 (noise); best gross is
  ~5-9bps at 12h, below the required-IC bar and below cost. Flow features add
  nothing over price. Verdict: **NO PULSE**.

**What this means for the three chosen paths (evidence, not opinion):**
1. **Predictor** — the CHEAP gate failed. A higher-frequency/order-flow predictor
   on the data we have (and can get free) has no edge. It is not dead as an idea,
   but it now requires genuinely EXOTIC data — L2 order-book / tick microstructure
   / perp-native liquidation flow — which needs a real-time feed + tick storage
   and is a large, uncertain procurement. **The predictor is a moonshot, not a
   quick win, and its cheapest gate just came back empty.** Do NOT fund data/GPU
   for it on the strength of hope; fund it only if you are committing to the
   exotic-data build with eyes open.
2. **Venue** — now the CLEAREST near-term win, and it needs no predictor. A
   percentage-fee venue makes the ALREADY-CERTIFIED book tradeable on BTC (+14bps)
   and widens ETH/SOL ~30bps/RT (table above). That value is real today and
   independent of the failed predictor gate. Gate 0 (legal accessibility) is the
   binding question.
3. **Capital** — only against a proven book. The existing book is Sharpe-positive
   / return-negative-vs-hold, so capital scales that profile; it becomes
   compelling only AFTER a venue change (cheaper cost → positive net) or a
   predictor (none yet). Not first.

**Revised recommendation:** the evidenced order is **venue first** (it pays off on
the existing book), **capital second** (scale the post-venue positive book), and
**predictor last and only as a deliberate exotic-data project** — its free/cheap
gate is now known empty, so it is the highest-cost, highest-uncertainty path, not
the starting one.

## What needs YOUR decision
1. **Gate 0 (venue):** is a percentage-fee perp venue legally accessible to you?
   (Everything forks here.)
2. **Stage 1 (data):** which data basis to procure, once Stage 0 shows a pulse.
3. **Stage 3 (capital):** how much, and only against a forward-proven book.

## What is explicitly NOT on this path
More models on the current data (dead, P281); wiring dropped feed fields as
evidence-only shadows that can't reach an order (P293d); tuning the current
gate/sizing on the existing ~0-edge signal. None of these is a profit lever.
