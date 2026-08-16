# Retraining Research: The Derivatives-First Question (2026-08-16, P279)

**Operator question:** comprehensive research on retraining, given that derivatives
(the Coinbase CDE perp sleeve) are now the key trade items. Three deep code/doc
audits (training-env fidelity, data basis, doctrine) + direct verification of
every load-bearing claim.

**Executive verdict:** the training system was built for a world that no longer
exists — a single-asset, spot-priced, taker-fee, ungated, continuous-position,
carry-free world — while 100% of live risk now sits on a sized, maker-first,
funding-charged, gate-controlled, multi-asset derivatives book. **Retraining any
existing model class today would re-measure the spot-basis verdicts already
rendered dead (P258/P263) using an environment that misprices the venue in ten
distinct ways.** The correct program is not "retrain the DRL"; it is a staged
derivatives-native rebuild whose stages are gated by evidence that is already
accruing: the maker fill rate (weeks), the September P166 reads (weeks), and the
new data families' depth (months-to-years). Meanwhile there is a short list of
cheap, immediate fixes this research surfaced — including three silently-zero
feature columns fed to every model ever trained, a 2× cost-convention split
between the two labs, and a runtime gap that would break any decision-interval
checkpoint at serve time.

---

## 1. Env-vs-venue fidelity: ten gaps (agent-audited, spot-verified)

| # | Dimension | Training env | Live venue | Materiality |
|---|---|---|---|---|
| 1 | Action space | continuous Box(−1,1) on a $100k book | bang-bang sized contracts under caps | **HIGH** — ~85% of learned actions unexpressible |
| 2 | Fees | flat taker, no maker model | maker-first 0/3bps mix | LOW-MED (overcharge = safe direction) |
| 3 | Slippage/impact | Kraken-spot constants, impact pinned at cap | sub-$1k nano orders, 0-slip maker fills | MED (overstated friction) |
| 4 | **Funding carry** | **ABSENT from the DRL env** (verified: zero matches) | charged/credited every held position; even the live gate prices it | **HIGH — biases WHAT is learned**; P245 measured carry flipping selections |
| 5 | Churn controls | none (deadband default 0) | flip-persist 2, cooldown 2, hold band 0.65 | MED — flips vetoed live that trained free |
| 6 | Stops/halts | none; positions never taken away | venue-resting 10% stop, 15% halt, watchdog | MED-HIGH — trained tail-holding realizes as stop-outs |
| 7 | **Alpha gate** | env trades unconditionally | actions reach the book only on gate-open ticks; gate-fail force-flattens | **HIGH — off-policy at deployment by construction** |
| 8 | **Decision cadence** | di=4 enforced inside the env | **NO runtime counterpart exists** (grep: zero matches in drl/, core/, main.py) | **HIGH — any di=4 checkpoint's core contract silently breaks at serve time** |
| 9 | Price series | Binance spot | CDE 2030-dated contracts (basis + carry) | LOW-MED signal / MED P&L |
| 10 | Multi-asset | single-asset env | shared 0.50 net budget + shared equity | MED — intent censored exactly when signals correlate |

**The three transfer-invalidators:** (a) carry never reached the DRL env —
the policy optimizes a different objective than the book settles in, with the
error concentrated in the book's historical loss mode (crowded longs); (b) the
gate/stop/flatten regime means deployment is off-policy on a gate-filtered
state distribution with exit events of training-probability zero; (c) cadence +
granularity strip the policy's two structural properties at the venue boundary
— what executes is a thresholded sign extractor, closer to the rule books than
to the trained policy.

## 2. Data basis: the parquets are a spot dataset with a derivatives garnish

Census of the 122+13 columns: **~118 are pure functions of Binance spot
price/volume.** The genuine derivatives content: `funding_rate_zscore` (full
depth — but the WRONG VENUE's funding: Binance perp, whose sign differed from
CDE on BTC/SOL when probed, P218), `liq_imbalance` + `oi_change_5d` (alive on
**8%** of bars — CoinGlass depth), and **three all-zero constants**
(`taker_ratio_zscore`, `tradecount_zscore`, `taker_vol_momentum` — the loader
silently backfills 0.0 because `training_data/futures/` never existed; every
model ever trained consumed three dead inputs without an error).

**Three price bases coexist:** training features = Binance spot; LIVE features
= Kraken spot through the same FeatureEngineer; monetized returns = CDE dated
contracts (spot + term-basis drift + funding). At 4h the signal-side skew is
minor; the **P&L side is unmodeled everywhere** — no backtest in the repo has
ever priced term-basis drift, and the carry that P245 wired in is the
wrong-venue proxy.

**History depth vs the ~9,200-bar walk-forward requirement:**

| Family | Depth today | Walk-forward viable? |
|---|---|---|
| Binance perp funding | ~6y | yes (already in) |
| CoinGlass liq/OI (4h) | 187d | no (~5y of accretion needed; P266 merge is load-bearing) |
| ETF flow | ~1.8y | no — capped at ETF launch AND backfilling imports a reporting-lag leak (P164 class). Forward ledger only. |
| CDE basis (calbasis) | **0d** | never from history; forward-only (and its state is RAM-only — persist it) |
| **Breadth/xsmom closes** | **~6y** | **YES — the one backfillable family nobody has built** |

**Split every future evaluation into signal basis vs P&L basis:** Binance
spot/funding are defensible *inputs* (Binance funding is arguably the better
crowding signal than a thin nano-perp's own); they are the wrong *economics*
for a CDE book. Forward ledgers are the only venue-true measurements.

## 3. Doctrine (distilled; the full 24-rule list with citations lives in the
P-ledger — P164/P179-184/P198/P200/P215/P221/P241-263/P266-278)

Pre-flight: retrain = measurement; measure-before-GPU; edge-probe first; fresh-
mind review; kill-point defined before launch. Data: 6y fetch; causal-by-
construction with future-perturbation proof; split-aware GMM verified by
`fit_policy`; live-computable single-source features from birth. Economics:
venue-true fees; di=4; fresh tags; carry in every arithmetic; select on
after-cost realized PnL. Judging: three baselines incl. ridge_16h + bootstrap
CI; design-era-only selection; ONE ledgered validation read, spent BEFORE
wiring a forward ledger (P259b); era battery + virgin-era + transfer gates;
overlap-corrected t + deflated Sharpe; provenance triplet. Deployment: nothing
promotes from backtest — 30d forward shadow through P166; {GMM, parquets,
checkpoints} atomic; ordered operator flips. **The scarce resource is unread
forward windows, not GPU.**

Re-entry conditions on record: RL only with a new formulation premise (P258 —
fee changes do NOT revive it; its deaths are fee-independent); supervised
ridge revival IF measured effective RT ≤ ~3bps after 2-4 weeks of maker ledger
(P278); new-information bases enter as lab-laddered tilts only after a P166
PASS; breadth assets stay deliberately unfitted (P262 virgin-evidence).

## 4. New findings from THIS research, each with a disposition

1. **Three zero-constant manifest columns** (taker_ratio_zscore etc.) — every
   trained model consumed dead inputs. FIX CHEAP: delete from the manifest (or
   build from the fv2 spot-taker data, which exists) at the next manifest
   change; harmless to verdicts (constants carry no information either way)
   but the loader must REFUSE, not backfill zeros (P199 shape).
2. **`decision_interval` has no runtime counterpart** — recorded here as a
   HARD Rung-3 prerequisite: no di-trained checkpoint may deploy to shadow
   until a runtime hold mechanism exists, or its shadow IC measures a
   different policy than was trained (P214 skew class).
3. **Lab cost-convention split (verified at both sites):** `train_supervised_
   full` charges COST_BPS per SIDE (12bps RT for BTC) while `mechanism_lab`
   charges it as ROUND-TRIP (6bps — matching the P166 gate). Each lab is
   internally consistent; cross-lab numbers are 2× apart in cost, and the
   0/18 supervised kills were rendered at DOUBLE the gate's cost — biased
   AGAINST models (they trade more than hold baselines). DISPOSITION: unify
   the convention (name constants `_RT` or `_PER_SIDE`), and re-read the
   supervised zoo at honest cost — cheap CPU, and it composes with the P278
   maker re-read (both lower the effective cost bar).
4. **calbasis `_basis_hist` is RAM-only** — the only CDE-native signal's
   warmup restarts on every deploy (P154 class). DISPOSITION: persist.
5. **Live features come from KRAKEN spot while training features come from
   BINANCE spot** — a third basis in the train/serve chain, previously
   unrecorded. Minor at 4h; recorded so no one discovers it as a surprise.
6. **Docs still encode the spot world** (Plan V3's spot cells and ±1
   assumption, the Guide's taker-only table and DRL-shaped Rung-4, 3-asset
   universe language). DISPOSITION: update at the next doc pass; the
   September tree is the current source of truth for promotion mechanics.

## 5. The program: what "retraining for derivatives" actually means, staged

**Stage D0 — now, free (no GPU, no windows spent):**
fix the zero columns + loader refusal; unify the lab cost convention and
re-read the supervised zoo at honest RT; persist calbasis state; **build the
breadth/xsmom training series** (6y of closes, the one backfillable new
family) and run it through the existing lab ladder; keep accruing the maker
fill-rate ledger and the twelve P166 exams.

**Stage D1 — after the maker measurement (~2-4 weeks):** if effective RT ≤
~3bps, execute the P278 supervised revival (ridge-class on clean parquets at
MEASURED cost, Rung-3 shadow). This is the only model-training on any current
evidence path.

**Stage D2 — after September, ONLY on a certified basis:** if the reads
certify signals (books, liq, ETF, ma_filter…), the policy-layer question
reopens per P258 — and THIS is where a derivatives-native environment is
justified: discrete sized contracts under the shared net cap, funding carry
in every bar's PnL, gate/stop/flatten events in the simulator, maker/taker
fill model, multi-asset, and NO decision-interval without its runtime
counterpart. Section 1's ten gaps are that env's requirements spec. Building
it before a certified basis exists would be infrastructure for a tenant that
three campaigns say isn't coming on the current signal set.

**Stage D3 — the long game (months-years):** the venue-true families reach
walk-forward depth only by forward accretion (liq 4h ~5y out; calbasis
forward-only; ETF capped). The P266 merge cadence (≤150d) is the load-bearing
mechanism; a missed re-fetch permanently loses the middle.

**What is explicitly NOT in the program:** a TQC/RL retrain on the current
basis (0/39 folds, fee-independent causes); fitting the breadth assets
(burns P262); backfilling ETF history into training (imports the lag leak);
loosening the P166 gate before the maker fill rate is measured.
