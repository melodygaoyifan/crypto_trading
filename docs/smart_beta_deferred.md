# Smart Beta — Deferred Items (V2+)

Items requiring retraining, new data sources, or architectural changes.
Do NOT implement in V1.

## Requires DRL Retrain (obs_dim change)
- Add beta_to_btc as 123rd feature
- Add cross_asset_correlation as 124th feature
- Add vol_regime as 125th feature (already in obs via vol features, but not explicitly)
- Any change to 126-dim contract

## Requires New Data Source
- Broad crypto market index (e.g., TOTAL2 from TradingView)
- DeFi TVL time series
- Stablecoin flow aggregator
- Options IV surface data

## Requires New Model
- Neural regime classifier to replace/augment GMM
- Multi-asset attention model for cross-asset beta estimation
- Online beta-aware DRL fine-tuning

## Requires New Module
- Explicit hedge sleeve (long alpha + short beta)
- Market-neutral overlay
- Cross-asset beta-weighted portfolio optimizer
- Dynamic benchmark rotation

## Future Integration
- Smart Beta → DRL reward shaping (beta-penalized reward)
- Smart Beta → Authority escalation (CAP authority in extreme regime)
- Smart Beta → Automated A/B framework with statistical gates
