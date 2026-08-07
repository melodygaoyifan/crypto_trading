"""[P219] Derivatives-flow shadow strategies, sourced from data we already pay for.

WHY THIS EXISTS. The `flow` agent emits 0.00 on every tick because its whale
proxy is gated on CryptoCompare's `large_transaction_count`, and that account is
capped at 100 API calls/month (measured: 283 used, "please upgrade your
account"). Upgrading is blocked. But CoinGlass — already paid for, already
fetched every tick — carries a directional flow signal that is arguably better
suited to this book anyway:

    liquidation_imbalance   BTC -0.53   ETH -0.50   SOL +0.06
    liquidations 24h        BTC $83M    ETH $64M    SOL $7.4M

CryptoCompare's metric counts large ON-CHAIN TRANSFERS. This book trades
PERPETUAL FUTURES. Liquidation imbalance is derivatives positioning being
forcibly unwound — a more direct read on the flow that moves perp prices than
counting wallet-to-wallet transfers.

SIGN CONVENTION, verified against the live API rather than assumed
(`coinglass_feed.py:650`): `liquidation_imbalance = (long - short) / total`, so
POSITIVE means more LONGS were liquidated. Confirmed live at the feed's own 24h
range: BTC long $8.5M vs short $31.7M -> -0.576, matching the -0.526 in
market_data. (A 1h probe showed the opposite sign, which is exactly why this was
checked against the same window the feed uses.)

WHY TWO STRATEGIES. Forced liquidation is genuinely ambiguous in direction and
the honest thing is to let the data choose:

  * SQUEEZE (momentum): shorts being liquidated = forced BUYING = upward
    pressure that tends to continue. direction = -imbalance.
  * EXHAUSTION (reversion): a large liquidation cascade is capitulation, and
    the move fades once the forced flow is done. direction = +imbalance.

They are exact opposites by construction, so at most one can be right, and
both being noise is a perfectly possible — and informative — outcome. Emitting
both and letting `analytics/shadow_ic/compute_shadow_ic.py` score them under the
P166 cost-aware gate is the point: it costs nothing extra and replaces a guess
with a measurement. This is the P147 lesson (do not promote an unvalidated
signal into live fusion) and the P198 one (an in-sample split is a hypothesis to
shadow, never a gate to enforce).

OBSERVATION-ONLY. Iron Law 7: nothing here touches agent_signals or fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# A liquidation print smaller than this carries no information — it is the
# background hum of a normal session, not a flow event. Emitting a direction off
# $50k of liquidations on a $99B open-interest market would be noise dressed as
# signal, and would pollute the IC measurement this exists to produce.
MIN_LIQUIDATION_USD = 1_000_000.0

# |imbalance| below this is "balanced" — both sides being liquidated equally
# tells us nothing directional.
MIN_ABS_IMBALANCE = 0.15


@dataclass
class FlowSignal:
    strategy_name: str
    direction: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _inputs(market_data: Dict[str, Any]) -> Dict[str, float]:
    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = market_data.get(key)
            return default if v is None else float(v)
        except (TypeError, ValueError):
            return default

    return {
        "imbalance": _f("liquidation_imbalance"),
        "total_liq": _f("total_liquidations_24h"),
        "long_liq": _f("long_liquidations_24h"),
        "short_liq": _f("short_liquidations_24h"),
        "oi": _f("open_interest"),
    }


def _confidence(total_liq: float, oi: float, abs_imb: float) -> float:
    """Scale with how big the liquidation event is RELATIVE to open interest.

    Absolute USD is not comparable across assets ($83M is enormous for SOL and
    ordinary for BTC), so it is normalised by OI. Capped at 1.0.
    """
    if oi <= 0:
        # Without OI there is no scale reference. Fall back to the imbalance
        # magnitude alone and say so in the diagnostics — do NOT invent a
        # denominator, which would make a small print look like a large one.
        return round(min(1.0, abs_imb), 4)
    ratio = total_liq / oi
    # 1% of OI liquidated in 24h is already a violent session.
    return round(min(1.0, (ratio / 0.01) * abs_imb), 4)


class _BaseLiquidationFlowStrategy:
    """Shared gating so the two readings differ ONLY in sign."""

    _NAME = "base"
    _SIGN = 0.0

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> FlowSignal:
        v = _inputs(market_data)
        diag = {
            "liquidation_imbalance": round(v["imbalance"], 4),
            "total_liquidations_24h": round(v["total_liq"], 2),
            "long_liquidations_24h": round(v["long_liq"], 2),
            "short_liquidations_24h": round(v["short_liq"], 2),
            "open_interest": round(v["oi"], 2),
            "observation_only": True,
        }

        if v["total_liq"] <= 0:
            return FlowSignal(self._NAME, 0.0, 0.0,
                              "no_liquidation_data", diag)
        if v["total_liq"] < MIN_LIQUIDATION_USD:
            return FlowSignal(
                self._NAME, 0.0, 0.0,
                f"quiet(total_liq=${v['total_liq']:,.0f} < "
                f"${MIN_LIQUIDATION_USD:,.0f})", diag)

        abs_imb = abs(v["imbalance"])
        if abs_imb < MIN_ABS_IMBALANCE:
            return FlowSignal(
                self._NAME, 0.0, 0.0,
                f"balanced(imbalance={v['imbalance']:+.3f})", diag)

        direction = max(-1.0, min(1.0, self._SIGN * v["imbalance"]))
        conf = _confidence(v["total_liq"], v["oi"], abs_imb)
        side = "longs" if v["imbalance"] > 0 else "shorts"
        return FlowSignal(
            self._NAME, round(direction, 4), conf,
            f"{side}_liquidated(imb={v['imbalance']:+.3f}, "
            f"liq=${v['total_liq']:,.0f})", diag)


class LiquidationSqueezeStrategy(_BaseLiquidationFlowStrategy):
    """MOMENTUM reading: shorts liquidated -> forced buying -> keeps going up."""
    _NAME = "liquidation_squeeze"
    _SIGN = -1.0


class LiquidationExhaustionStrategy(_BaseLiquidationFlowStrategy):
    """REVERSION reading: the cascade IS the capitulation; the move then fades."""
    _NAME = "liquidation_exhaustion"
    _SIGN = +1.0


def build_derivatives_flow_strategies():
    """The two opposite readings. At most one can be right."""
    return [LiquidationSqueezeStrategy(), LiquidationExhaustionStrategy()]
