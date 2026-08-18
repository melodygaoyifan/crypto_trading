# Tripwire Enhancement Research
**[P293L] · 2026-08-17 · goal: a system that actually trades**

Research pass on whether the P237 tripwire is the right instrument, given the
operator's stated goal. Short answer: **it is the wrong shape for the goal,
and one level down the alpha gate is the actual blocker.**

---

## 1. What the tripwire is today

Counts 4 consecutive weekly `GATE-CLOSED` calibrator reports per asset, then
instructs removal of that asset's trend injection. Currently 2/4, all three
assets closed.

It is a **kill switch**: one variable (calibrator slope), one threshold, one
direction (remove).

## 2. Five design flaws, four of them research-backed

### 2.1 One variable, one threshold — explicitly called non-viable

The practitioner literature is blunt: *"A kill switch with one threshold based
on one variable will not be viable. Risk limits should be tailored to your own
system… but define them before the strategy goes live."*
([Stratzy](https://stratzy.in/blog/algo-kill-switch-engineering-how-smart-traders-protect-capital-in-volatile-markets/))

Ours keys on a single OLS slope. Today that slope has |t| < 0.8 in every cell —
it is not distinguishable from zero, so the trigger is driven by noise plus a
floor-at-zero rule.

### 2.2 It can only ever make the system trade LESS

The tripwire has no "promote", no "resize", no "restore". Applied to a system
whose problem is *too few trades*, its entire action space is in the wrong
direction. **A kill-only instrument cannot serve a goal of trading more.**

### 2.3 Binary, where the theory says partial

Gârleanu & Pedersen's closed-form optimum for trading with predictable returns
and transaction costs is: *aim in front of the target, and* **trade partially
towards the aim** — a continuous partial adjustment, never an on/off switch
([NBER w15205](https://www.nber.org/system/files/working_papers/w15205/revisions/w15205.rev0.pdf),
[J. Finance 2013](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080)).

Our live path is binary at **two** places:
- the alpha gate: `effective_threshold = max(min_alpha_bps, friction × 1.10)`,
  then pass → full size, fail → **nothing**;
- the sleeve: sizes by sign, minimum 1 contract.

So a signal with a small-but-real edge produces **zero** trades rather than a
small position. That is the architectural reason the book sits flat.

### 2.4 No comparison to an alternative

It asks "is trend's slope above the bar?" — never "compared to what?". Removing
trend hands the decider slot to Best-of-N (modulated by the unvalidated
`[SENT-SWITCH]` rule, P293i). **Removing a measured-weak signal into an
unmeasured one is not an improvement**, and a single-signal gate cannot see
that.

### 2.5 Threshold staleness it cannot detect (found and fixed today)

The bar was stamped 2026-08-08 and overtaken twice by *code* (P289 spreads,
P291b venue-true hold), leaving it 1.5–1.9× too high. The staleness guard is
**time**-based (30d) while the thresholds move on **code** changes — so it
reported "verified 10d ago" while already wrong. Fixed in P293k; immaterial to
today's verdict, but it would distort the first period where slopes rise.

---

## 3. What the data actually says about "no trades"

Measured, not assumed:

| candidate | state right now |
|---|---|
| trend (live decider) | \|dir\|≈0.33 → **10bps** asserted alpha vs thresholds 19–29bps → blocked |
| regimebook (certified) | BTC in-market **27%** of ticks, **0 flips/7.8d**; ETH **0%**, SOL **0%** (degraded) |
| whale | directional **54% / 43% / 12%**, binary ±1.0 → **30bps** → clears all three |

Two things follow:

1. **The market is quiet and most candidates agree.** The certified book is
   flat on ETH/SOL because the regime is not bull. That part is not a bug —
   a trend/hold book *should* be flat here.
2. **Only whale currently produces a tradeable signal**, and only because its
   binary encoding asserts full conviction. That is why it was armed (P293j) —
   with the honest caveat that binary ≠ better-evidenced.

---

## 4. Proposed enhancement: replace the kill switch with a SEAT CONTROLLER

The right instrument is not a tripwire but a **weekly allocator over seat
candidates**, which subsumes the tripwire's job and can also *increase*
trading.

```
inputs   per candidate {trend, whale, regimebook, flat}:
           - forward IC (4h and 16h, overlap-corrected)
           - realized PnL of its shadow/live ledger
           - calibrator slope, on BOTH alignment bases (P293k)
output   which candidate holds the DECIDE seat
rules    1. pick the best-evidenced candidate
         2. require a MARGIN over the incumbent to switch (hysteresis —
            otherwise it thrashes on noise, the same reason flip-persistence
            exists)
         3. "flat" wins only if every candidate is negative
         4. never switch on |t| < 1 alone; require the PnL ledger to agree
```

Why this is strictly better than the tripwire:

- it is **multi-variable** (the literature's requirement),
- it is **bidirectional** — it can seat a signal, not just unseat one,
- it makes the choice **relative**, which is the only well-posed version of
  the question,
- and the tripwire becomes a special case: "flat wins" is just one outcome.

**Cost:** small. The three inputs already exist (`agent_ic_review`,
`slope_calibrator`, the shadow ledgers) and `september_check.py` already
aggregates them. This is a scoring function plus a config write, not new
infrastructure.

---

## 5. The higher-leverage change: hold longer, not signal more

Gârleanu-Pedersen's core result is that **alpha decay determines how long you
enjoy the return, and therefore the whole return/cost trade-off**. Cost is
paid per round trip; edge accrues per unit of holding. So the ratio that
decides whether a weak signal is tradeable is *cost per unit of holding time*.

Current live turnover controls:

| control | value |
|---|---|
| `coinbase_flip_persist_ticks` | 2 (≈8h minimum before a flip) |
| `coinbase_reentry_cooldown_ticks` | 2 |
| `alpha_gate_hold_ratio` | 0.65 |
| `regimebook_mode` | **off** |

The certified book measured **0 flips in 7.8 days** on BTC, versus the live
trend layer's 29 flips in 54 days (P198). At the same per-trade cost, the
low-turnover book pays roughly an order of magnitude less friction per unit of
exposure — which is exactly the lever that moves a sub-threshold signal above
the bar.

The repo already has the mechanism measured: **P256's `adjust_step`
(k_exit / k_flip / min_hold) EARNED on all three assets in the lab** and runs
in shadow as `regimebook_adj`. It is the partial-adjustment policy the theory
prescribes, already implemented, already validated in-design, and switched off.

---

## 6. Recommended order of work

1. **Do not fire the tripwire as designed.** It cannot help the stated goal and
   its action hands the seat to an unmeasured path (§2.4).
2. **Build the seat controller** (§4). It absorbs the tripwire, is
   multi-variable per the literature, and can *add* trading.
3. **Read the low-turnover candidates' forward ledgers** — `regimebook` and
   `regimebook_adj` — as seat candidates, not as promotion paperwork. The
   turnover argument (§5) is the strongest reason to expect them to clear a
   cost bar that the 4H trend signal cannot.
4. **Accept the structural limit honestly:** at 1-contract granularity there is
   no way to express "trade small". Until equity supports fractional sizing,
   the system's only levers are *which* signal holds the seat and *how long* it
   holds a position — not *how much*.

---

## 7. What this research does NOT claim

- No evidence that any candidate has positive expected value after costs. Every
  measured IC in this system is inside noise (all |t| < 1.5).
- The seat controller improves *decision quality and trade frequency*, not
  edge. If every candidate is genuinely zero, the honest outcome is still flat.
- Gârleanu-Pedersen assumes a known return-predicting signal. We do not have
  one; their result is used here only for the structural point (partial
  adjustment beats binary, and alpha decay drives the cost trade-off).

**Sources:**
[Gârleanu & Pedersen, *Dynamic Trading with Predictable Returns and Transaction Costs*, NBER w15205](https://www.nber.org/system/files/working_papers/w15205/revisions/w15205.rev0.pdf) ·
[Journal of Finance version](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080) ·
[Stratzy, *Algo Kill-Switch Engineering*](https://stratzy.in/blog/algo-kill-switch-engineering-how-smart-traders-protect-capital-in-volatile-markets/) ·
[NYIF, *Trading System Kill Switch: Panacea or Pandora's Box?*](https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box) ·
[LuxAlgo, *Risk Management Strategies for Algo Trading*](https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/)
