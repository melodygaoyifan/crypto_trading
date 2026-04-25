"""IC (Information Coefficient) monitoring framework.

Quantifies each agent's actual alpha contribution by computing the Spearman
rank correlation between every agent's per-bar signal and forward returns.

IRON LAW: this is observability only — never modifies fusion decisions or
trading state. All hot-path interactions go through `log_bar_safe()` which
swallows every exception.

Files:
- ic_logger.py        — async per-bar JSONL writer
- compute_ic.py       — offline IC computation from logs + OHLCV
- backfill_ic.py      — bootstrap IC for price-driven agents from history
"""
