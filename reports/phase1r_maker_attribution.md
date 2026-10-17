# HMATS Phase 1R — R-3 maker-share attribution

211 venue fills 2026-06-13 → 2026-10-15; MAKER 73 (34.6%), TAKER 138.

## Liquidity × month

| month | MAKER | TAKER | All |
|---|---|---|---|
| 2026-06 | 0 | 27 | 27 |
| 2026-07 | 0 | 43 | 43 |
| 2026-08 | 67 | 54 | 121 |
| 2026-09 | 3 | 11 | 14 |
| 2026-10 | 3 | 3 | 6 |
| All | 73 | 138 | 211 |

## Liquidity × intent class (Phase 0 inference)

| cls | MAKER | TAKER | All |
|---|---|---|---|
| ESCALATE | 27 | 10 | 37 |
| FLIP | 0 | 66 | 66 |
| FULL_EXIT | 14 | 23 | 37 |
| NEW_ENTRY | 17 | 19 | 36 |
| PARTIAL_EXIT | 12 | 20 | 32 |
| REBALANCE | 3 | 0 | 3 |
| All | 73 | 138 | 211 |

## Liquidity × placement path (fill_quality label) × order type

|  | MAKER | TAKER | All |
|---|---|---|---|
| direct / LIMIT | 2 | 3 | 5 |
| maker / LIMIT | 69 | 0 | 69 |
| market_urgent / MARKET | 0 | 16 | 16 |
| pre_ledger / LIMIT | 2 | 99 | 101 |
| pre_ledger / STOP_LIMIT | 0 | 1 | 1 |
| taker_cross / LIMIT | 0 | 19 | 19 |
| All /  | 73 | 138 | 211 |

## Before vs after maker-first (2026-08-16)

- before: 93 fills, MAKER 1 (maker-first was OFF: every order was a 0.2%-through LIMIT, i.e. a taker by construction)

- after: 118 fills, MAKER 72 (61.0%)


## Q2 — does post-only reach the venue?

- fills placed by the maker ladder (post-only LIMIT): 69; venue verdict MAKER on 69, TAKER on 0.

- code path: CoinbaseSleeve._maker_first_attempt / _maker_reprice_step build OrderRequest(post_only=True) -> CoinbaseAdapter.place_order -> client.limit_order_gtc(post_only=bool(request.post_only)); the P270 probe showed the venue REJECTS a crossing post-only order.


## Q3 — maker-ladder timeouts

- ladder outcomes with a fill (after 2026-08-16): maker 69, timed out into a taker cross 19 -> fallback rate 21.6% of ladder fills.


## Q1 — where the TAKER fills come from (after maker-first)

| path | ESCALATE | FULL_EXIT | NEW_ENTRY | PARTIAL_EXIT | All |
|---|---|---|---|---|---|
| direct | 0 | 3 | 0 | 0 | 3 |
| market_urgent | 0 | 0 | 0 | 16 | 16 |
| pre_ledger | 2 | 4 | 1 | 1 | 8 |
| taker_cross | 8 | 4 | 4 | 3 | 19 |
| All | 10 | 11 | 5 | 20 | 46 |


## Q4 — V15's 98.7%

- LIMIT order-type share of these fills: 194/211 = 91.9%; LIMIT share of ORDERS: 90.9%.

- V15 / analytics/sixty_day_review/review_aggregator.maker_fee_ratio = n_limit / n_classified (ORDER TYPE), and training/backtest_framework FeeSchedule.maker_pct=0.987 cites it. A 0.2%-through LIMIT is a TAKER fill; the ratio counted intent, not the venue's verdict (the P166 finding). The venue liquidity share is the 34.6% above.


## Current configuration — ETH/SOL stack (P416/P417/P419), since 2026-08-27

- 49 fills, MAKER 23 (46.9%)

|  | MAKER | TAKER |
|---|---|---|
| maker / ESCALATE | 18 | 0 |
| maker / FULL_EXIT | 2 | 0 |
| maker / NEW_ENTRY | 1 | 0 |
| maker / PARTIAL_EXIT | 2 | 0 |
| market_urgent / PARTIAL_EXIT | 0 | 16 |
| pre_ledger / FULL_EXIT | 0 | 1 |
| taker_cross / ESCALATE | 0 | 4 |
| taker_cross / FULL_EXIT | 0 | 3 |
| taker_cross / NEW_ENTRY | 0 | 2 |


## Current configuration — BTC skew seat (P424), since 2026-10-13

- 5 fills, MAKER 2 (40.0%) — INSUFFICIENT (n<10)

|  | MAKER | TAKER |
|---|---|---|
| maker / ESCALATE | 1 | 0 |
| maker / NEW_ENTRY | 1 | 0 |
| market_urgent / PARTIAL_EXIT | 0 | 2 |
| taker_cross / ESCALATE | 0 | 1 |
