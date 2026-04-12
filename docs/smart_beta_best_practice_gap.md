# Smart Beta V1.1 — Best Practice Gap Analysis

## Summary
V1 core implementation is solid and meets ~90% of best practices.
Remaining gaps are measurement depth and documentation, not architecture.

## Already Meets Best Practice
- ✅ Runtime-only, bounded, feature-flagged
- ✅ 3 professional contexts (Trend/Vol/Liquidity)
- ✅ Reuse-first (20 signals, all REUSE_AS_IS)
- ✅ No DECIDE/VETO authority
- ✅ No obs_dim change
- ✅ observe_only + bounded_influence modes
- ✅ enabled=false = strict no-op
- ✅ Multiplicative injection via existing paths
- ✅ Downside beta computed
- ✅ Regime-bucket reporting
- ✅ 35 tests passing

## Gaps Addressed in V1.1
- Added crowding-event bucket to regime_bucket_report
- Added 2 missing docs (this file + patch_plan)
- Bar-level measurement documented as deferred (requires continuous equity curve, not yet available for live)

## Gaps Deferred to V2
- Bar-level continuous equity curve for rolling beta (requires position mark-to-market every 4H)
- Equal-weight market basket as continuous benchmark series
- Formal coverage % reporting with missing-window diagnostics
- Online beta observer running per-tick (requires persistence layer)

## Double-Count Assessment: ACCEPTABLE
Funding/OI/Liq are consumed by BOTH SmartBeta and Sentiment L1, but through
different semantic channels (SmartBeta → size/gate modulation vs Sentiment → zscore → fusion).
This parallel consumption is intentional and bounded — not a double-counting error.
