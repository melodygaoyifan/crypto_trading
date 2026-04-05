"""
Kraken Quant Agent - Re-exports from agents.kraken_quant_agent

STATUS: EXPERIMENTAL - backward-compatibility shim only.
Not used in the production runtime (main.py).
"""
import warnings as _warnings
_warnings.warn(
    "exchange.kraken.kraken_quant_agent is EXPERIMENTAL and not used in production. "
    "Use agents.quant_agent.QuantAgent instead.",
    DeprecationWarning,
    stacklevel=2,
)
from agents.kraken_quant_agent import *
