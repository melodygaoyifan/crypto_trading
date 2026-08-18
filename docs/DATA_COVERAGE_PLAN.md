# Data Coverage: Audit, Repairs, and Improvement Plan
**[P293] · 2026-08-17 · macro / big-money flow / market sentiment**

Answers three questions the operator asked, in order:
1. Do we have sufficient data for macro, whale-flow and sentiment? *(audit)*
2. Can it be repaired? *(done — this document records what and how)*
3. **Do we need new sources?** *(mostly no — see §3, the headline finding)*

---

## 0. The one-paragraph version

Of the three signal families, **only sentiment's cheapest layer was actually
running**. Macro was dead through *three independent paths*, and the
whale/flow family had exactly one live component whose output fusion is
configured to ignore. The cause was the same in every case and it was not a
missing data source: **a consumer wired to a cache that no producer ever
fills.** FRED, LunarCrush and GlobalContextInformer all had valid keys,
non-mock initialisation, and no caller of `fetch()` / `update_async()`
anywhere in the tree. Every reader called a pure cache accessor, got the
neutral default, and the default was indistinguishable from a measurement.

So the answer to "do we need new sources" is **no for macro and social, no
for exchange flow, and no for options** — three of those were one function
call away, and the fourth was on an endpoint we already pay for. Two free
keyless APIs (Deribit, alternative.me history) close the rest. The only
genuinely unbuyable gaps are SOL options and SOL exchange flow (§3.3).

---

## 1. What was actually wrong (measured, not inferred)

Evidence: live engine log + per-tick diagnostics + read-only API probes,
2026-08-17.

| Family | Component | State found | Mechanism |
|---|---|---|---|
| **Macro** | FRED feed | Key valid, `mock=False`, **cache never filled** | No caller of `fetch()`/`start()` in the entire non-test tree |
| | GlobalContextInformer | `"GlobalContext updated"` logged **0 times ever**; needs yfinance, which is **not in the image** | `update_async()` has no caller either |
| | `macro_feed.py` | `source=MOCK`; VIX/DXY/rates are `random.uniform()` | Constructed with no args → MOCK default |
| | Net effect | `macro_leverage_cap = 1.0`, `risk_appetite = 0.5`, regime NEUTRAL — **macro constrained nothing** | |
| **Flow / whale** | Exchange netflow | **Structurally 0.0 forever** | `net_exchange_flow` sums `raw["exchange_flows"]`, a list no source populates |
| | `flow_direction` | Pinned at 0 (needs \|flow\| > $1M) | Consequence of the above; agent carries ADVISE weight **0.20** |
| | Tape whale detector | **Working** (BTC +1.00/0.64) | — but `whale` is one of the 12 deliberately zero-weighted agents (P228) |
| | CryptoCompare on-chain | Quota **exhausted 90/90** since ~Aug 10 | Our own P220 monthly budget |
| | SOL on-chain | Dead | `HELIUS_API_KEY` absent |
| **Sentiment** | L1 Fear & Greed | **Working, real** (F&G=31), fusion-consumed at weight 0.10 | — |
| | `sentiment_zscore` | **Not a z-score** — `(fg-50)/50*3` | Feed asks for `limit=1`; no distribution exists |
| | L3 LLM (Haiku) | Dead: `headlines=0`, `status=fallback_global` | CryptoPanic 429s + CC News quota-capped |
| | LunarCrush social | Cache never filled | Same no-producer bug as FRED |
| | Options PCR | Pinned at **1.0 neutral** | CoinGlass v3 option paths 404 (P218) |

**The sharpest single finding.** F&G = 31 scores **-1.14** under the linear
form → the sentiment agent emits a confident bearish signal (conf 0.38) on
every tick, on all three assets. Against its own trailing 365 days — which
averaged ~27 — the same reading is **z = +0.25, 71st percentile**, i.e.
*above* average. The one signal of these three families that reaches a live
trading decision has been miscalibrated by a hardcoded constant.

---

## 2. What was repaired

All feeds now run and **ledger** every tick. Every hop into the *decision*
layer is behind a default-OFF flag, because feeding a previously-zero agent
into fusion is an activation (P141), not a bugfix.

| # | Repair | Live effect now | Gate |
|---|---|---|---|
| 1 | FRED + LunarCrush `fetch_if_stale()` wired into the tick driver | Caches fill; macro_risk / attention / crowding become real | none — pure repair |
| 2 | `source_status` derived from feed provenance, not dict shape | Can now report `defaults_only` | none |
| 3 | **Deribit feed** (new, free, keyless): real PCR + DVOL | Logged every tick | `options_use_deribit`, `dvol_to_market_data` |
| 4 | **Exchange netflow feed** (new, existing CoinGlass key) | Logged every tick | `exchange_netflow_to_flow_agent` |
| 5 | **F&G history** (3116 days) + real z-score, persisted | Both forms logged side by side | `sentiment_zscore_mode` |
| 6 | GCI indicators from FRED instead of absent yfinance; ETF half **refuses** instead of mocking | Real VIX/SPX/US10Y/DXY available | `macro_gci_live` |

Config flags are declared **and** parsed (P201 trio), default OFF, and
absent from the live profile — a test fails if any appears there.

### Three traps handled explicitly, because each would have made things worse

- **The sign trap.** The exchange-netflow *producer* convention
  (`onchain_feed`) is positive = inflow; the *consumer* in main.py reads
  positive = outflow (`"outflow = bullish" → +1`). They disagree by a sign
  and it never surfaced only because the producer always returned exactly
  0.0. The new feed publishes its own unambiguous keys and the bridge
  **negates explicitly**, pinned by test. Arming a sign-inverted flow signal
  would be strictly worse than the zero it replaces.
- **The ETF fabrication trap.** Driving GCI while its yfinance-backed ETF
  tracker returns *mock flows* would inject invented fund flows into
  `macro_regime`. That path now refuses (returns `None`). Real ETF data
  already exists in the P270 `EtfFlowShadow` (CoinGlass v4).
- **The two-directional cap.** GCI's leverage-cap map both tightens
  (RISK_ON 0.7) **and loosens** (RISK_OFF 1.2, CRISIS 1.5). Driving it is
  not a one-way safety fix, so it is gated like everything else.

### Absences kept as absences (never defaulted to neutral)

| Asset | What is missing | Why it stays missing |
|---|---|---|
| SOL | Options PCR, DVOL | Deribit lists **zero** SOL options |
| SOL | Exchange netflow | CoinGlass returns **zero** SOL exchanges |
| all | GOLD macro indicator | FRED's daily gold series is discontinued |
| all | ETF flows in GCI | No FRED substitute for fund flows |

A fabricated value for any of these would be indistinguishable from a
measurement — the exact defect this batch removes.

---

## 3. Do we need new sources?

### 3.1 No — three of four gaps were already paid for or free

| Need | Verdict | Source | Cost |
|---|---|---|---|
| Macro (VIX, SPX, rates, DXY, NFCI) | **No new source** | FRED — key already held, unlimited | $0 |
| Social attention / crowding | **No new source** | LunarCrush — key already held | $0 |
| Exchange netflow | **No new source** | CoinGlass `/api/exchange/balance/list` — **existing key** | $0 |
| Options PCR + DVOL | New, but **free and keyless** | Deribit public API | $0 |
| Sentiment distribution | New, but **free and keyless** | alternative.me `?limit=0` | $0 |

**Total added cost: zero.** The measured coverage from the probes:
Deribit BTC 782 instruments / ETH 654, DVOL live on both; CoinGlass balances
BTC 21 exchanges (2.51M BTC), ETH 20 exchanges (12.07M ETH); F&G 3116 daily
rows back to 2018-02-01.

### 3.2 The one place a purchase might be justified — and I would not make it yet

**SOL coverage.** SOL has no options and no exchange-balance data from any
free or currently-held source. CryptoQuant / Glassnode sell SOL exchange
flows (~$30–100/mo entry tiers). **Recommendation: do not buy yet.** BTC and
ETH netflow are now live and unproven; buy SOL coverage only if the BTC/ETH
netflow signal clears its P166 forward gate (§4). Buying data for a signal
that has not yet demonstrated edge on the assets we *can* measure is the
P269 mistake — spending on assumed economics.

### 3.3 Sources deliberately NOT added

- **yfinance** — would "fix" GCI, but it is a fragile scraper and FRED
  serves the same series on an official API we already authenticate to.
  Adding a dependency to reach worse data is a bad trade.
- **A replacement news feed** — the LLM sentiment layer is dead, but P228
  already records that `llm_sentiment` sign-flips between measurement
  windows (i.e. noise). Buying news to feed a signal with no demonstrated
  edge inverts the priority order. Fix the billing (§5) before buying.
- **Trading Economics** — key absent, feed already commented out as dead
  code, and `macro_shock` is hardcoded 0.0. FRED covers the need.

---

## 4. Activation plan — pre-committed before any evidence exists

Criteria fixed *now*, so a later decision cannot be selection dressed up as
evidence. Each flag flips only on its own criterion, never in bulk.

| Order | Flag | Criterion to enable | Risk if wrong |
|---|---|---|---|
| **1** | `options_use_deribit` | None — enable at next deploy | Near zero: `options` is a zero-weighted ADVISE agent, so it cannot move an order. Real data replacing a known-broken constant. |
| **2** | `macro_gci_live` | ≥ 2 weeks of `[P293]` macro logs showing FRED indicators stable and regime transitions sane | Two-directional cap change. Watch for the cap exceeding 1.0. |
| **3** | `sentiment_zscore_mode: historical` | ≥ 30 days of both forms logged, then compare forward IC of each through the P166 cost-aware gate | Changes the sign and confidence of a fusion-consumed agent. **The current evidence says the linear form is miscalibrated, but "miscalibrated" ≠ "the other one is profitable."** |
| **4** | `exchange_netflow_to_flow_agent` | ≥ 30 days of ledger + forward IC clearing P166 (IC > 0 every horizon, \|t\| ≥ 2, edge ≥ 2× round-trip cost) | Largest change in the batch: `flow` carries ADVISE weight 0.20 and has emitted 0.0 forever. |
| **5** | `dvol_to_market_data` | Only after a shadow read shows the DVOL z-score would not have force-flattened the book | **Arms `EMERGENCY_FLAT` at z ≥ 5, a path that has never executed** (P265d). Highest-consequence flag here. |

Revert for every one: remove the key, redeploy.

---

## 5. News quota — what I got wrong, and what was actually fixable

My first pass called both news feeds "operator-only". **Probing them properly
found a real code defect behind each**, and in one case my diagnosis was
simply wrong. Recorded in full because the wrong version was stated
confidently.

### 5.1 CryptoPanic — I said "not volume". The API says otherwise.

I claimed ~6 req/day against a 3000/mo plan meant the 429 could not be a
quota issue. Probing from the server returned, verbatim:

```
HTTP 429  {"status":"api_error",
           "info":"API monthly quota exceeded - Upgrade your API plan"}
```

So it **is** a plan matter. But two real bugs sat on top of it:

- **A monthly quota was being retried like a transient rate limit.**
  CryptoPanic sends **no `Retry-After`**, so the handler took its 900s
  default and re-asked **every 15 minutes for the rest of the month** —
  ~2,900 pointless requests, each logging a warning that reads like a
  transient fault (the P202 alert-fatigue shape). **Fixed:** the 429 body is
  now parsed; a monthly-quota message backs off to the 1st of next month UTC
  and says once, plainly, that this is a budget state. An unrecognised 429
  still takes the short backoff — the fail direction is "retry sooner than
  necessary", never "go dark by mistake".
- **A Cloudflare client-signature hazard.** From the server, the default
  stdlib User-Agent returns `HTTP 403 error code: 1010` — Cloudflare's
  banned-client code — a *different* failure from a rate limit that would
  keep a feed dark even with quota available. **Fixed:** every session built
  by `create_session()` now identifies itself.

**[P293c] And then it stopped needing you.** Asked for a cheaper CryptoPanic
option, the honest answer is that no cheaper *vendor* is worth buying —
because a free one already publishes the only thing this system consumes.

`SentimentLLMAgent` sends **headline text** to Haiku and derives sentiment
itself; it does not use CryptoPanic's vote tallies for direction. So the
expensive part (curated sentiment metadata) is not what the tradeable signal
rests on. **Public RSS is free, keyless and unquota'd**, and a live probe
returned **94 headlines per poll**:

| source | items |
|---|---|
| cointelegraph.com/rss | 30 |
| decrypt.co/feed | 34 |
| theblock.co/rss.xml | 20 |
| bitcoinmagazine.com/feed | 10 |

(CoinDesk returns 308 to its feed URL and is deliberately omitted rather than
followed blindly — an unverified redirect target is not a source.)

Wired as a **third blend source**, deliberately last and additive: it can
only add headlines the two curated sources did not supply, never displace
them, and it applies the *same* 4h window and dedup, so stale copy cannot
satisfy the `_c3_live` starvation gate. Verified end-to-end **with no API
keys at all**: BTC 5 fresh headlines, ETH 1, where both were 0.

It does **not** replace `panic_score` / `news_velocity` /
`narrative_intensity` — RSS has no vote data and this feed does not
fabricate any (test-pinned).

**Cost comparison:** CryptoPanic upgrade ≈ $XX/mo for metadata we don't use
for direction → **RSS $0/mo for the headlines we do**. Upgrading is now
optional rather than blocking.

### 5.2 CryptoCompare — the budget wasn't the problem, the restarts were

The 90/90 pool was exhausted by **Aug 10 — nine days into the month** —
against a designed demand of ~3 calls/day (news 12h TTL, on-chain 48h). The
cause was not the budget size:

**Both feeds kept their throttle clock in RAM.** `cc_onchain`'s check is
`now - self._last_fetch_time < MIN_FETCH_INTERVAL and self._data`, and
*both* halves reset on restart; `cc_news` cached per-asset in memory only.
P253d had persisted the *backoff* one line below — leaving the throttle in
memory, directly under its own comment that "a limiter that re-arms on
restart is not a limiter". Measured: **14 process starts** across the
retained logs, each re-spending calls.

**Fixed:** both feeds now persist the throttle clock *and* the payload, so a
restart can neither re-spend quota nor lose the data. Mock rows are never
persisted (a restored mock would survive as if it were a reading).

**No budget change was needed** — raising `HMATS_CC_MONTHLY_BUDGET` would
have masked a restart bug with a bigger allowance, against an account cap of
100/month that cannot be upgraded.

### 5.3 Genuinely operator-only

| Item | Action | Note |
|---|---|---|
| **CryptoPanic plan** | **Optional now** | RSS covers the headline need at $0. Upgrade only if the vote-derived panic metrics are wanted. |
| ~~`HELIUS_API_KEY`~~ | **NOTHING TO DO — I was wrong** | See §5c. No key is required; SOL on-chain is already live. |
| **`whale` promotion** | See §5d | Its own instrument says HOLD, and the promotion mechanism is a measured no-op. |

### 5c. Helius — correcting an error of mine

I listed `HELIUS_API_KEY` as an operator action because it was absent from
`.env`. **That was inference, not verification, and it was wrong.**

- `SOLANA_RPC_URL` **is already set** on the server to the public endpoint.
- The SOL on-chain feed is **live and healthy** — TPS ~4,072, slot times,
  Jito tip percentiles, refreshing every ~2 minutes.
- `HELIUS_API_KEY` is referenced only by an *unselected* source enum in
  `onchain_feed.py`. Nothing on the live path requires it.

`solana_onchain.py` defaults to `https://api.mainnet-beta.solana.com`, which
is keyless and answered fine on probe. **There is nothing to set, and no
purchase to make.** That `onchain_sol` emits a flat direction is a separate
question about what a congestion/TPS feed can say about *direction* — not a
missing credential.

### 5d. Whale promotion — the honest blocker is not the weight

Two independent findings, both measured:

**1. Its own instrument says HOLD.** `agent_ic_review --window-days 60`:

| horizon | n | IC | t | edge | required IC |
|---|---|---|---|---|---|
| 4h | 460 | +0.053 | 1.15 | 3.8bps | 0.168 |
| 16h | 458 | +0.024 | 0.26 | 3.2bps | 0.092 |

Positive, but statistically indistinguishable from zero (|t| ≥ 2 required)
and economically below cost (~12bps needed). Verdict: **HOLD**, not PROMOTE.

**2. Adding an ADVISE weight is a no-op on live orders — verified in code.**
The weight enters as `contribution = w × confidence × alignment`, which
scales `base_exposure` (a *magnitude*, never a direction). That value is then
**overwritten unconditionally** at `integration_v36.py:1678`
(`intent.target_exposure = tranche_decision.target_exposure`, commented
"Bug #44: Always apply tranche's target_exposure"). And even if it survived,
the sleeve sizes by **sign** (`target_for(asset, dir)`), discarding magnitude
entirely. Two independent mechanisms kill it.

So the literal request — add `whale` to `ADVISE_WEIGHTS_BY_REGIME` — would
change nothing while *looking* like a promotion. That is the P177 shape
(a control that reads as active and cannot act), and I have not shipped it.

**What would actually give whale influence**, in increasing order of cost:

| option | what it does | fit for current evidence |
|---|---|---|
| **A. Entry filter** (P236 `ma_filter` pattern) | whale disagreement blocks entry-from-flat; never forces exits | Best fit — a weak-but-positive signal is more defensible as a veto than as a driver |
| **B. Direction seat** (P285 `mlp` pattern) | whale takes the quant slot, driving the book | **Not justified at t=0.26** |
| **C. Fix Bug #44** | makes ADVISE weights real for *all* agents | Large refactor; re-arms the whole dead pre-1678 exposure stack |

Recommendation: **A, default-OFF, with a shadow ledger**, so the P166
evidence accrues on the filter rather than on the raw agent.

**[P293d] All three are now built, all default-OFF and absent from the live
profile.** A and B work; C is the honest reconnection with a stated limit.

---

## 6. [P293d] End-to-end promotion sweep — is anything ELSE ready?

**No. At 90 days, not one agent clears the P166 bar.**

| agent | 4h IC (t) | 16h IC (t) | verdict |
|---|---|---|---|
| llm_sentiment | +0.040 (1.12) | +0.048 (0.67) | HOLD |
| sentiment | +0.001 (0.04) | +0.049 (0.87) | HOLD |
| whale | +0.040 (0.98) | +0.011 (0.14) | HOLD |
| model_alpha | +0.005 (0.11) | +0.030 (0.35) | HOLD |
| micro | +0.007 (0.04) | +0.202 (0.53) | HOLD (n=35) |
| **quant (the DECIDER)** | +0.007 (0.27) | **−0.046 (−0.89)** | HOLD |
| drl | −0.005 (−0.22) | −0.059 (−1.19) | HOLD |
| funding | −0.014 (−0.18) | −0.138 (−0.88) | HOLD |

Required IC is 0.159 (4h) / 0.085 (16h). Everything is an order of magnitude
short, and every |t| is below 1.2. **Nothing is being unfairly held back.**

Worth naming without over-reading: the agent currently *driving* the book
has the second-worst 16h IC in the table, while several zero-weighted agents
are mildly positive. At |t| < 1.2 that is noise-versus-noise, not an
argument for a swap — but it is the reason option B exists as a config flip.

## 7. [P293d] Do the signals actually play downstream? — mostly NO

Traced in code, not inferred. `coinbase_use_gated_intent = True`, so fusion's
intent does drive the sleeve. But of what fusion produces, only **direction**
survives:

| fusion layer | what it modifies | reaches an order? |
|---|---|---|
| LAYER 3 DECIDE (quant, kraken_quant) | `result.direction` | **YES** |
| LAYER 2 / 7 VETO | zeroes direction / `veto_active` | **YES** (flatten) |
| LAYER 4 CONFIRM (regime, two_stage, structure) | `base_exposure` only | no |
| LAYER 4.25 HTF, 4.5 partial consensus | `base_exposure` | no |
| LAYER 4.75 ADVISE (18 agents) | `base_exposure` | no |
| LAYER 6 CAP (macro) | `base_exposure` | no |

Everything below LAYER 3 modulates exposure, and exposure is severed twice:

1. `intent.target_exposure` is **overwritten** at `integration_v36.py:1678`
   by the tranche value (Bug #44 — itself a real fix; it stopped an
   unchecked 80% fusion exposure reaching the risk governor);
2. the sleeve sizes by **sign** (`target_for(asset, dir)`), discarding
   magnitude — its own docstring names the gap: *"partial sizes would need a
   conviction-magnitude input the ±1-era driver never had."*

**So 22 of 26 agents cannot influence a live order through fusion at all** —
not because of their weights, but because their only channel is discarded.
The live chain is: `trend_decision_layer` → quant slot → LAYER 3 → direction
→ gated intent → sleeve sign. Everything else is instrumentation.

That is why option A acts on the **sleeve driver** rather than through a
weight, and why option C had to supply the conviction input the sleeve
docstring asked for.

## 8. [P293d] Sentiment audit — running on real data, interpreted by three stacked assertions

The pipeline is healthy (F&G is live and real). The *interpretation* is
where the questions are, and there is a genuine contradiction:

**Two contradictory Fear & Greed readings coexist in the same system.**

| | mapping | at F&G=31 (fear) | reaches fusion? |
|---|---|---|---|
| `main.py:7338` (live) | **momentum**: `(fg-50)/50*3` | z=−1.14 → **bearish**, conf 0.38 | **YES** |
| `signals/deterministic_sentiment.py` | **contrarian**: greed −0.6, extreme fear 0.0 | ~neutral | no (logged only) |

The module the docs call "Sentiment L1 (F&G) ACTIVE" implements the
**contrarian** reading — the classic use of this index — while the value
that actually reaches fusion implements the **momentum** reading. They
disagree in sign whenever the market is greedy.

Two further properties, both measured:

- **The contrarian engine can never emit a bullish value.** Every branch is
  ≤ 0 (`>75 → −0.6`, `>55 → −0.3`, `<45 → −0.2`, else 0.0). It is
  structurally short-biased, so "sentiment turned bullish" is not an output
  it can produce.
- **The live signal is three assertions stacked**: a hardcoded linear
  rescale (not a z-score — §2), read as momentum (not contrarian), crossing
  a fixed `|z| > 1.0` band (≈ F&G outside 33–67). None of the three is
  measured, and the 90-day IC (+0.001 / +0.049, both insignificant) cannot
  distinguish right from wrong.

**[P293e] Built.** `defense/sentiment_variant_shadow.py` records all three
claims per tick to `sentvariant_{ASSET}.jsonl`, judged by the same P166 gate:
`sent_momentum_linear` (the live rule), `sent_momentum_hist` (same rule on a
real historical z), `sent_contrarian` (the deterministic engine's own
asymmetric mapping). Verified they genuinely disagree — at F&G=80 momentum
says **+1 (bullish)** and contrarian says **−1 (bearish)**.

The contrarian variant is deliberately **not** `-momentum`: a pure negation
has an IC exactly the negative of the momentum form and would add no
information. It is the engine's asymmetric mapping — greed penalised hard,
extreme fear mapped to **neutral, not bullish** — which is a different signal
shape, and it is what the codebase already calls its sentiment engine.
Observation-only; touches no signal.

---

## 9. [P293g] "If everything is off and never promoted, why build it?"

The sharpest question asked so far. Measured rather than defended.

### 9.1 The premise is half wrong — 11 things HAVE been promoted

Of 41 feature gates in `ProductionConfig`, 15 are explicitly decided in the
live profile, and **11 of those are ON**:

| promoted and live | category |
|---|---|
| `coinbase_maker_first`, `coinbase_maker_reprice` | execution cost |
| `coinbase_venue_aware_fees` / `_funding` / `_spreads` / `_true_hold` | cost accuracy |
| `coinbase_routing_enabled`, `coinbase_use_gated_intent` | plumbing / risk binding |
| `fast_risk_sleeve_enabled`, `flip_persistence_enabled` | risk control |
| `trend_following_mode: enforce` | the one live alpha source |

**Every promoted item is execution, cost, or risk — never alpha.** That is
not timidity: their evidence is *arithmetic*. A maker fill costs 0bps against
3bps taker; you do not need 30 days to know that. Alpha claims need
statistics, and none has passed.

So the real question is narrower: **why do the ~14 alpha shadows never
promote?**

### 9.2 The answer: the 30-day clock cannot fire at the horizon that matters

The gate requires `|t| = IC·sqrt(n_eff − 1) ≥ 2` with `n_eff = n/h`
(overlap-corrected). At 6 bars/day on 4H data:

| window | IC needed for \|t\|≥2 @4h | @16h |
|---|---|---|
| 14d | 0.220 | 0.447 |
| **30d** | 0.149 | **0.302** |
| 90d | 0.086 | 0.173 |
| 365d | 0.043 | 0.086 |

The **economic** bar (P166, cost-aware) asks ~0.26 at 4h and ~0.13 at 16h.
So:

- **At 4h the economic bar binds** and ~27 days is enough. Coherent.
- **At 16h the statistical bar binds and is ~2.3× stricter.** A candidate
  whose edge is economically adequate (IC 0.13) needs **~330 days**. The best
  IC ever observed live (0.048) would need **~3 years**.

**Consequence: a 30-day clock on a 16h claim can only certify IC ≥ 0.30** —
an implausible edge. The ~14 candidates sitting on 30-day clocks against 16h
horizons are on clocks that structurally cannot fire.

That is a **check that cannot pass** — the exact inverse of P174's *check
that cannot fail*, and a defect by the same reasoning. It was invisible
because the gate reported its requirement in **samples** (`n_required~=1980`)
and nobody converted it to days.

### 9.3 What was done about it

`assess_promotion` now states the requirement in days beside the samples:

```
h=4: IC +0.0900 is 0.60 SE from zero (need |t| >= 2.0; n=180,
     n_eff=45 overlap-corrected, n_required~=1980
     = ~330d of 4H bars vs ~30d held)
```

Cheap, mechanical, and it stops the next clock being set blind (the P230
rule: build the instrument, don't rely on remembering).

### 9.4 The honest verdict

The shadow apparatus is **not** failing. It is correctly reporting that no
alpha candidate has earned promotion — and 11 non-alpha improvements *were*
promoted, quickly, because their evidence was arithmetic.

But two things are genuinely wrong and worth fixing:

1. **The clocks are mis-set.** A 16h claim needs ~6–12 months, not 30 days.
   Either lengthen the clocks, judge those claims at 4h where 30 days
   suffices, or pool across assets to raise `n`. Anything else is theatre.
2. **The carrying cost is real.** Each shadow adds code, tests, baselines and
   review surface. Fourteen of them against a bar none can reach is a poor
   trade. **Recommendation: stop adding alpha shadows until the clock
   question is settled** — the marginal one costs maintenance and cannot
   produce a verdict any sooner than the thirteen already running.

---

## 9b. [P293i] The sentiment question, ANSWERED OFFLINE — no 30-day wait needed

Asked whether the tripwire / 14 clocks / 30-day contrarian exam are actually
needed, the contrarian one turned out to be answerable **immediately**: the
F&G history is 3,116 days, and price history is on disk. A forward exam was
never the right instrument for a question that 6 years of data already
settles.

**Backtest, 3 assets, ~2,175 days each, daily F&G vs forward returns:**

| asset | horizon | momentum IC | contrarian IC | sent_switch IC |
|---|---|---|---|---|
| BTC | 7d | **+0.056** | −0.056 | +0.024 |
| ETH | 7d | **+0.088** | −0.088 | +0.028 |
| SOL | 7d | **+0.126** | −0.126 | +0.008 |

(1d horizons agree in sign, smaller magnitude.)

**Verdict: the MOMENTUM reading is right on all three assets at both
horizons, and the contrarian reading is decisively wrong** — it is the exact
negation, so it loses by construction in every cell. The live
`main.py:7338` interpretation is correct; `deterministic_sentiment.py`'s
contrarian mapping is the one that should never be promoted.

**Honest caveat:** alternative.me builds F&G partly FROM price (volatility
25% + market momentum/volume 25%), so "momentum on F&G" is partly momentum
on price. This settles *which interpretation to use*; it does not establish
F&G as independent alpha.

**Consequence for the machinery:** the 30-day contrarian exam is
**superseded before it collected a single record**. It stays wired (zero
cost, and it confirms the offline result forward) but nothing waits on it.

### The coupling nobody had noticed

`[SENT-SWITCH]` (`market_data_pipeline.py:1189`) is a THIRD F&G
interpretation — fear<25 boosts mean_revert, greed>75 boosts momentum — and
it feeds the Best-of-N **strategy selector**, i.e. `quant_direction`, the
DECIDE agent. It fires on **30.8% of days** historically (**171 days of
extreme fear in the last 365, and zero of extreme greed**).

It is dormant today **only because `trend_following_mode: enforce`
overwrites `quant_direction`** after the pipeline computes it.

**So firing the tripwire does not take the book flat — it hands the wheel
back to Best-of-N, whose strategy weights are modulated by this
never-validated rule, in a regime where it fires ~47% of days.** Anyone
removing the trend injection must decide what takes over first.

## 10. [P293f] API fetch efficiency — audited across every feed

### 10.1 Research: what is actually available

Probed the response headers of every dependency (2026-08-17):

| source | ETag | Last-Modified | Cache-Control | rate-limit headers |
|---|---|---|---|---|
| alternative.me (F&G) | no | no | no | no |
| CoinGlass v4 | no | no | no | no |
| Deribit | no | no | `no-store` | no |
| **RSS (cointelegraph)** | no | **yes** | `s-maxage=300` | no |

So the textbook first move — **conditional requests** (`If-None-Match` /
`If-Modified-Since`, answered with a bodiless `304`) — is available on **RSS
only**. The JSON APIs expose no validators, so for those the only levers are
client-side TTL matched to the data's *update cadence* (not the poll
cadence), request coalescing, and batching. None of them exposes remaining
quota in headers either, which is why client-side accounting (the P220 CC
budget) is the only way to know.

### 10.2 The waste found, and fixed

| feed | defect | fix |
|---|---|---|
| **CoinGlass** (paid) | `fetch()` had **no throttle** and the tick calls it once per **asset**, while `_fetch_real` already loops all three symbols internally across 3 endpoint families → **~3× the requests needed** | `fetch_if_stale()` on its 300s interval |
| **Fear & Greed** | value updates **daily**; the outer staleness check uses the *data* timestamp so it reads stale on every asset → refetched per asset | `fetch_if_stale()` on fetch-time |
| **RSS** | full body fetched every poll | conditional GET — verified **4× `304`** on the second fetch, corpus intact |
| FRED, LunarCrush, Deribit, netflow, F&G history | (P293/P293b) no producer / no throttle | `fetch_if_stale()` |

The shared `cache_age_seconds()` helper now has one definition instead of a
sixth hand-rolled copy, and handles both datetime and float-epoch stamps
(the feeds use both) with the naive/aware defence P40/P97 keeps requiring.

**Also already correct, worth recording so it is not "fixed" again:** CC News
uses one request to serve all three assets (P219), CryptoCompare has a
persisted account-wide budget (P220), and every throttle now survives restart
(P293b). The remaining per-symbol loops (CoinGlass, Deribit, netflow) are
*inside* one throttled fetch and are inherent to those APIs — they have no
multi-symbol endpoint.

---

## 5b. Two more fixes at source (rather than worked around)

- **The flow sign disagreement is fixed, not bridged.** The first pass
  negated inside the new bridge. The honest fix is at the consumer: the
  producer convention is positive = inflow, and `flow_direction` mapped
  positive to **+1 (bullish)** — the inverse of its own comment. Corrected
  in main.py, so producer and consumer now share one convention and the
  bridge passes values straight through. Behaviour-neutral today (the key is
  always 0.0, which is sign-invariant) and correct the moment it carries a
  number — which is exactly the right time to fix a sign.
- **GCI now gets real ETF flows** instead of refusing. It reuses P270's
  `EtfFlowShadow` reader (CoinGlass v4, including its in-progress-day
  guard) rather than adding a second implementation of the same endpoint.
  One aggregate record **per asset**, deliberately: the caller sums the
  list, so one per *ticker* would multiply the real flow by the number of
  tickers. A flow older than 4 days reads as absent, not as current.
  First live read: **BTC −$56.2M, ETH flat**.

---

## 6. What this does NOT claim

- No new edge has been demonstrated. This batch makes signals *exist* and
  *be honest*; whether they are *profitable* is what the forward ledgers and
  the P166 gate decide. Several of these agents may turn out to be noise.
- `data_health` may move in either direction after deploy, because MACRO can
  now honestly report `defaults_only`. It dampens `quant_confidence`
  (caution ×0.85, reduced ×0.6). Movement toward tightening on genuinely
  absent data is correct, but it *is* a behaviour change.
- The sentiment finding (§0) says the live signal is miscalibrated. It does
  **not** say the historical z-score will make money. Criterion 3 above
  exists precisely so that question is settled by forward data.
