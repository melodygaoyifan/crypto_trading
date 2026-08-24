# Venue/Scale/Data research — the venue constraint has a 2026 US-legal resolution (2026-08-24)

**Operator:** "any way to resolve the scale and venue, any other data option, do the research."

## Headline: US crypto derivatives moved ONSHORE in 2026 — the venue block lifted

Every prior entry called the percentage-fee / low-fee perp venue "regulatorily
blocked (US-restricted INTX)". That was true in the short-bias era and is **no
longer true**: in 2026 the CFTC approved Kalshi's BTC perpetual and cleared a
path for Coinbase, and **Kraken launched CFTC-regulated perpetual futures for US
traders (announced 2026-05-29, live ~2026-06-15) via Bitnomial** (a CFTC-
regulated exchange its parent Payward acquired April 2026). It is on **Kraken —
one of the two venues the operator already uses** — and covers **BTC/ETH/SOL**
(+ XRP/ADA/LINK/DOGE/LTC/AVAX), USD collateral, ~$25 intraday margin, managed
alongside spot from one account.

## The fee math — it clears the floor on fee alone

Kraken Derivatives US (Bitnomial) charges a **flat $0.15/contract/side, all-in**
(commission 0.03 + exchange/clearing 0.10 + NFA 0.02). Contract sizes: **BTC 0.01,
ETH 0.5, SOL 5.0**. At live prices (BTC ~77.9k / ETH ~2.48k / SOL ~94.9):

| asset | contract notional | fee/side | **fee-only RT** | CDE RT (flat ~$0.60) |
|---|---|---|---|---|
| BTC | ~$779 | 1.93 bps | **3.9 bps** | ~15 bps |
| ETH | ~$1,238 | 1.21 bps | **2.4 bps** | ~49 bps |
| SOL | ~$474 | 3.16 bps | **6.3 bps** | ~19 bps |

Break-even RT for the existing IC-0.04 signal is **~8 bps** (P396: required IC
≈ 0.00487 × RT_bps at BTC 16h; 0.04 clears at RT ≤ ~8). **Bitnomial's fee alone
(2.4–6.3 bps RT) is below that** — so the signal we ALREADY have (0.04, real,
statistically significant) is plausibly tradeable here with NO new data.

## What this resolves

- **VENUE: resolved (pending live spread verification).** A US-legal, CFTC-
  regulated perp venue on an exchange already in use, at a fee 4–20× below CDE and
  below the signal's break-even.
- **SCALE: eased, no longer the binding path.** The flat fee is what scale was
  needed to amortize; Bitnomial's low fee + larger contracts (ETH 0.5, SOL 5)
  amortize it without the ~$427k BTI tier. The fund can stay fixed.
- **DATA: the hunt becomes moot in the best way.** Every data lead was a proxy for
  "beat the fee floor"; the venue does that directly. On-chain is free-and-dead
  (P397), options-paid too expensive / CoinAPI too thin (P398) — but if the fee
  floor drops below the 0.04 signal, none of that is needed. **Pause the data hunt.**

## Caveats — do not migrate before verifying these

1. **Live spread/impact/liquidity is UNVERIFIED.** Bitnomial is ~2.5 months old;
   total RT = fee + spread + impact, and only the FEE is confirmed low. Thin new-
   venue liquidity could add spread that offsets the fee win. **This is the
   decisive unknown** — check the live BTC/ETH/SOL order-book depth/spread first.
2. **The 0.04 signal is still weak.** Even tradeable, it is a thin edge — so the
   low-turnover SMA200-hold framing (fee on flips only) is essential; a per-bar
   churn strategy would still lose to fees.
3. **Migration is real work + a live-money decision (P141):** new API, contract
   specs, and routing (the sleeve currently targets Coinbase CDE). Not a config flip.
4. **Eligibility** ("eligible US clients") and **funding cadence** (sources
   conflict: 8-hour vs daily) must be confirmed.

## Next step

Verify (a) account eligibility for Kraken Derivatives US, and (b) live BTC/ETH/SOL
order-book spread + depth on Bitnomial. If total RT (fee + spread + impact) stays
under ~8 bps, port the certified SMA200 trend/hold signal to the new venue — the
first genuine profitability path this investigation has found. It does not require
new data, new capital, or a new model — only the venue that 2026 regulation opened.

## Sources
- Kraken US perps launch: cryptotimes.io, thetradenews.com, coindesk.com, blog.kraken.com (2026-06)
- Fees: support.kraken.com/articles/us-futures-fees ($0.15/contract/side flat)
- Contract sizes: support.kraken.com/articles/contract-specifications (BTC 0.01 / ETH 0.5 / SOL 5.0)
