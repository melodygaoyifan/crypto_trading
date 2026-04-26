# Smart Beta V1 — Reuse Audit

## Data Reuse Map

| Data/Signal | Producer | Reuse Status | New Consumer |
|------------|---------|-------------|-------------|
| GMM regime_state | market_data_pipeline._predict_gmm_regime | REUSE_AS_IS | TrendMomentumContext |
| Hurst exponent | market_data_pipeline._compute_advanced_metrics | REUSE_AS_IS | TrendMomentumContext |
| MTF momentum | market_data_pipeline._compute_advanced_metrics | REUSE_AS_IS | TrendMomentumContext |
| ADX | market_data_pipeline (ta library) | REUSE_AS_IS | TrendMomentumContext |
| vol_regime | market_data_pipeline._compute_advanced_metrics | REUSE_AS_IS | VolatilityRegimeContext |
| vol_z_score | market_data_pipeline._compute_advanced_metrics | REUSE_AS_IS | VolatilityRegimeContext |
| VPIN | market.vpin_calculator | REUSE_AS_IS | VolatilityRegimeContext |
| kurtosis | market_data_pipeline._compute_advanced_metrics | REUSE_AS_IS | VolatilityRegimeContext |
| BSS | market_data_pipeline (Black Swan Sentinel) | REUSE_AS_IS | VolatilityRegimeContext |
| funding_rate | Kraken Futures feed | REUSE_AS_IS | LiquidityFundingContext |
| oi_change_24h_pct | Coinglass feed | REUSE_AS_IS | LiquidityFundingContext |
| liquidation_imbalance | Coinglass feed | REUSE_AS_IS | LiquidityFundingContext |
| Fear & Greed | alternative.me | REUSE_AS_IS | LiquidityFundingContext |
| SSC crowding_score | SimpleSentimentCalculator | REUSE_AS_IS | LiquidityFundingContext |
| _regime_alpha_gate_mult | main.py regime aggression | REUSE_AS_IS | Injection path (multiplicative) |
| _regime_position_size_mult | main.py regime aggression | REUSE_AS_IS | Injection path (multiplicative) |
| allow_scale_in | main.py regime_power config | REUSE_AS_IS | Scale-in modulation |
| beta_to_btc | market_data_pipeline (cov/var) | REUSE_AS_IS | BetaObserver |
| benchmark_summary.json | scripts/benchmark_suite.py | REUSE_AS_IS | BetaObserver reference |
| OHLCV bars (BTC/ETH/SOL) | Kraken REST via CCXT | REUSE_AS_IS | Benchmark construction |

## Zero New External Data Sources
V1 uses ONLY data already flowing through the system. No new API subscriptions, vendors, or schemas.
