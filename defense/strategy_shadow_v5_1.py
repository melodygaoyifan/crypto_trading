"""
HMATS v5.1 - StrategyShadow harness (lightweight, observation-only)
====================================================================

Phase 4 ships this as a write-only sink: candidate strategies emit signals,
the harness logs them to JSONL with a forward-look stub. Phase Pre-6 (Day
32) extends this with IC computation + promotion-gate logic.

Iron Laws honored:
  4. Fail-closed: any per-strategy exception is caught, logged, and the
     strategy is bypassed for the rest of the tick (does NOT corrupt other
     observations or main fusion).
  7. Shadow >=30d: signals are NEVER injected into agent_signals/market_data
     by this harness. Only Phase 10 promotion can wire them into fusion.

Schema of `data/strategy_shadow/microstructure_YYYYMMDD.jsonl`:
  {
    "ts": "ISO8601 UTC",
    "asset": "BTC|ETH|SOL",
    "strategy": "ofi|vpin_spike|kyle_lambda",
    "direction": -1.0 / 0.0 / +1.0,
    "confidence": float in [0,1],
    "reason": str,
    "diagnostics": { ... }
  }

Forward-return join is deferred to Phase Pre-6 (will join against equity_history
or OHLCV at 4b/12b/24b horizons offline).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_DIR_REL = "data/strategy_shadow"


class MicrostructureShadowHarness:
    """Observation-only sink for the 3 Phase-4 microstructure strategies.

    Usage from main.py per-tick (after market_data is fully assembled):

        if self._micro_shadow is not None:
            self._micro_shadow.observe(asset, market_data)

    The harness is constructed once at engine startup. It buffers nothing
    in memory beyond strategy state — every observation flushes to JSONL
    immediately (Iron Law 4: durable persistence + crash-safe).

    Per Iron Law 7, signals NEVER touch agent_signals or fusion at this layer.
    """

    def __init__(
        self,
        strategies: Iterable[Any],
        log_dir: Optional[Path] = None,
        log_prefix: str = "microstructure",
    ) -> None:
        self._strategies = list(strategies)
        self._log_dir = log_dir or self._resolve_log_dir()
        self._log_prefix = log_prefix
        self._observation_count = 0
        self._error_count = 0
        # Warn-once per strategy on first error so repeated faults don't spam
        self._warned: set = set()

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"[SHADOW_MICRO] log dir create failed ({self._log_dir}): "
                f"{type(e).__name__}: {e} — observations will be dropped"
            )

        names = [getattr(s, "__class__", type(s)).__name__ for s in self._strategies]
        logger.info(
            f"[SHADOW_MICRO] harness init: {len(self._strategies)} strategies "
            f"({', '.join(names)}); log_dir={self._log_dir}"
        )

    @staticmethod
    def _resolve_log_dir() -> Path:
        # Repo root = parents[1] from defense/strategy_shadow_v5_1.py
        repo = Path(__file__).resolve().parents[1]
        # Honor HMATS_DATA_DIR if set (production container uses /opt/hmats/data)
        env_data = os.environ.get("HMATS_DATA_DIR")
        if env_data:
            return Path(env_data) / "strategy_shadow"
        return repo / _DEFAULT_DIR_REL

    def _log_path(self, ts: datetime) -> Path:
        return self._log_dir / f"{self._log_prefix}_{ts:%Y%m%d}.jsonl"

    def observe(self, asset: str, market_data: Dict[str, Any]) -> None:
        """Run all shadow strategies against market_data; persist each
        result to JSONL. Per-strategy errors are isolated (one bad strategy
        does not kill the rest). Iron Law 4 fail-closed."""
        if not self._strategies:
            return

        ts = datetime.now(timezone.utc)
        path = self._log_path(ts)
        ts_iso = ts.isoformat()

        # Build payload list first to avoid partial writes on error
        payloads = []
        for strat in self._strategies:
            strat_name = type(strat).__name__
            try:
                sig = strat.evaluate(asset, market_data)
                payloads.append(
                    {
                        "ts": ts_iso,
                        "asset": asset,
                        "strategy": getattr(sig, "strategy_name", strat_name),
                        "direction": float(getattr(sig, "direction", 0.0) or 0.0),
                        "confidence": float(getattr(sig, "confidence", 0.0) or 0.0),
                        "reason": str(getattr(sig, "reason", "")),
                        "diagnostics": dict(getattr(sig, "diagnostics", {}) or {}),
                    }
                )
            except Exception as e:
                self._error_count += 1
                if strat_name not in self._warned:
                    logger.warning(
                        f"[SHADOW_MICRO] {strat_name}.evaluate raised "
                        f"{type(e).__name__}: {e} — bypassing this strategy "
                        f"(warn-once)"
                    )
                    self._warned.add(strat_name)

        if not payloads:
            return

        # Single open() per observe call -> all strategies in one fsync window
        try:
            with path.open("a", encoding="utf-8") as f:
                for p in payloads:
                    f.write(json.dumps(p, default=str) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:  # noqa: silent-swallow
                    # fsync best-effort; not all platforms / fs support it for
                    # the f.fileno() (Windows + network mounts). f.flush() already
                    # gave OS-level durability; fsync is the crash-safety upgrade.
                    # CLAUDE.md P72 lint annotation.
                    pass
            self._observation_count += len(payloads)
        except Exception as e:
            self._error_count += len(payloads)
            logger.warning(
                f"[SHADOW_MICRO] write failed ({path}): "
                f"{type(e).__name__}: {e} — {len(payloads)} observations dropped"
            )

    def stats(self) -> Tuple[int, int]:
        """Return (observation_count, error_count). Used by /step15 status export."""
        return self._observation_count, self._error_count


def build_microstructure_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """Convenience builder for main.py startup.

    Returns a harness pre-loaded with the 3 Phase-4 microstructure strategies.
    Lazy import to avoid eager strategy initialization in tooling that imports
    just the harness shell (e.g. tests of the harness itself).
    """
    from strategies.microstructure_v5_1 import build_phase4_microstructure_strategies
    return MicrostructureShadowHarness(
        strategies=build_phase4_microstructure_strategies(),
        log_dir=log_dir,
        log_prefix="microstructure",
    )


def build_cascade_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[v5.1 Phase 8] Cascade-prefix shadow harness.

    Same harness class as Phase 4 (the harness is phase-agnostic — it just
    runs evaluate(asset, market_data) on each strategy and writes to JSONL).
    Distinct prefix='cascade' segregates Phase 8 observations from Phase 4
    so Pre-6 IC compute can iterate them independently.

    Strategies: CascadeAnticipationStrategy, StopHuntDefenseStrategy.
    Output ledger: data/strategy_shadow/cascade_YYYYMMDD.jsonl
    """
    from strategies.liquidation_cascade_v5_1 import build_phase8_cascade_strategies
    return MicrostructureShadowHarness(
        strategies=build_phase8_cascade_strategies(),
        log_dir=log_dir,
        log_prefix="cascade",
    )


def build_funding_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[v5.1 Phase 3] Funding-prefix shadow harness.

    3 strategies: FundingRateExtreme, FundingRateMeanReversion,
    FundingRatePostETFRegime.
    Output ledger: data/strategy_shadow/funding_YYYYMMDD.jsonl
    """
    from strategies.funding_rate_v5_1 import build_phase3_funding_strategies
    return MicrostructureShadowHarness(
        strategies=build_phase3_funding_strategies(),
        log_dir=log_dir,
        log_prefix="funding",
    )


def build_ml_factor_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[v5.1 Phase 6] ML factor fusion agent shadow harness.

    Single MLFactorDispatcher conforms to the evaluate(asset, market_data)
    interface; per-asset routing happens inside the dispatcher.

    Output ledger: data/strategy_shadow/ml_factor_YYYYMMDD.jsonl
    """
    from agents.ml_factor_fusion_agent import MLFactorDispatcher
    return MicrostructureShadowHarness(
        strategies=[MLFactorDispatcher()],
        log_dir=log_dir,
        log_prefix="ml_factor",
    )


def build_derivflow_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[P219] Derivatives-flow shadow harness.

    Sources a directional flow signal from CoinGlass liquidation data — which
    is already paid for and already fetched every tick — because the `flow`
    agent's CryptoCompare whale proxy is capped at 100 calls/month and the
    upgrade is blocked.

    Two strategies that are exact opposites by construction (squeeze /
    exhaustion), so the IC gate decides which reading, if either, has edge
    rather than the author guessing. See strategies/derivatives_flow_v1.py.

    Output ledger: data/strategy_shadow/derivflow_YYYYMMDD.jsonl
    """
    from strategies.derivatives_flow_v1 import build_derivatives_flow_strategies
    return MicrostructureShadowHarness(
        strategies=build_derivatives_flow_strategies(),
        log_dir=log_dir,
        log_prefix="derivflow",
    )


class MAFilterEchoStrategy:
    """[P236] Echo strategy for the model_alpha disagreement filter.

    Unlike the other shadow strategies, this one computes NOTHING: the
    decision is made by ``main.sleeve_ma_filter_decision`` inside the sleeve
    driver (where the gated direction and the venue position actually live),
    and the driver passes the result in via the observe() dict. This class
    only reshapes it into the ledger record schema, so the filter rides the
    same durable-write + prefix machinery as every other shadow family
    instead of growing a second JSONL writer (P172: no estimator twins).

    Ledger claim: ``direction`` is the FILTERED signal — the sleeve target
    with any model_alpha disagreement zeroed. That is the strategy the P166
    cost-aware IC gate scores.

    [P294] ``strategy_name`` is a CONSTRUCTOR ARGUMENT, not a constant. A
    second filter (whale, P293d option A) reuses this echo, and
    ``compute_per_strategy_ic`` groups records by the record's ``strategy``
    field — NOT by filename or ledger prefix. With a hardcoded name the two
    ledgers pooled into one series: the whale claim became unmeasurable, and
    the ma_filter exam (which has a live config decision attached) was
    contaminated by rows from a different filter, with the inflated n
    loosening the gate's |t| requirement on a meaningless merge.

    The default is unchanged ("ma_filtered"), so every existing P236 row and
    pin keeps its exact meaning.
    """

    def __init__(self, strategy_name: str = "ma_filtered"):
        self._strategy_name = str(strategy_name)

    class _Sig:
        __slots__ = ("strategy_name", "direction", "confidence", "reason",
                     "diagnostics")

        def __init__(self, direction, confidence, reason, diagnostics,
                     strategy_name="ma_filtered"):
            self.strategy_name = strategy_name
            self.direction = direction
            self.confidence = confidence
            self.reason = reason
            self.diagnostics = diagnostics

    def evaluate(self, asset: str, market_data: Dict[str, Any]):
        ma = float(market_data.get("_maf_ma_dir", 0.0) or 0.0)
        direction = float(market_data.get("_maf_ledger_dir", 0.0) or 0.0)
        # [P236-followup] compute_shadow_ic scores x = direction x confidence,
        # so a directional claim MUST carry full confidence — anything less
        # silently down-weights that tick out of the IC, and the scored
        # series stops being the filtered signal (the first cut gave
        # ma_silent pass-throughs conf=0, which would have zeroed ~60% of
        # ticks out of the measurement). Flat claims carry 0 — a confidence
        # must never saturate on a zero direction (P224). The opinion's own
        # strength lives in diagnostics.ma_dir.
        reason = str(market_data.get("_maf_reason", "") or "")
        conf = min(1.0, abs(direction))
        diagnostics = {
            "raw_target": market_data.get("_maf_raw_target"),
            "ma_dir": ma,
            "sleeve_dir": market_data.get("_maf_sleeve_dir"),
            "pos_contracts": market_data.get("_maf_pos"),
            "action": market_data.get("_maf_action"),
            "enforce": market_data.get("_maf_enforce"),
        }
        # [P349] This dict is a WHITELIST, so a caller that adds an observation
        # key gets it silently dropped unless it is named here — the P2
        # reader/writer shape inside the harness itself. The whale sample size
        # is added CONDITIONALLY because this class is shared with the
        # ma_filter (P294), which has no such quantity: an unconditional key
        # would write a null into every ma_filter row and read as "measured
        # zero whales" rather than "not applicable" (P2 again, one level down).
        for _src_key, _dst_key in (("_maf_whale_count", "whale_count"),
                                   ("_maf_whale_pressure", "whale_pressure")):
            if _src_key in market_data:
                diagnostics[_dst_key] = market_data.get(_src_key)
        return self._Sig(
            direction=direction,
            confidence=conf,
            reason=reason,
            diagnostics=diagnostics,
            strategy_name=self._strategy_name,
        )


def build_ma_filter_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[P236] model_alpha disagreement filter shadow harness.

    Output ledger: data/strategy_shadow/ma_filter_YYYYMMDD.jsonl
    Scored by analytics/shadow_ic/compute_shadow_ic.py (prefix registered
    there). Promotion = P166 forward evidence + its own P-entry, never
    automatic.
    """
    return MicrostructureShadowHarness(
        strategies=[MAFilterEchoStrategy()],
        log_dir=log_dir,
        log_prefix="ma_filter",
    )


def build_whale_filter_shadow_harness(
    log_dir: Optional[Path] = None,
) -> MicrostructureShadowHarness:
    """[P293d option A] whale disagreement filter shadow harness.

    Output ledger: data/strategy_shadow/whale_filter_YYYYMMDD.jsonl
    Scored by analytics/shadow_ic/compute_shadow_ic.py (prefix registered
    there). Reuses MAFilterEchoStrategy — the echo reads the same
    `_maf_*` observation keys and the sleeve driver populates them
    identically for both filters, so the two ledgers are directly
    comparable and there is ONE echo implementation (P172).

    [P294] It carries its OWN strategy name. The scorer groups by the
    record's `strategy` field, not by prefix or filename, so sharing the
    echo's default name pooled this ledger into the ma_filter exam — see
    MAFilterEchoStrategy's docstring.

    Promotion = P166 forward evidence + its own P-entry, never automatic.
    Whale's raw-agent IC does NOT clear that bar today (60d 16h t=0.26);
    this ledger measures the FILTER, which is a different claim from the
    raw agent and has to earn its own verdict.
    """
    return MicrostructureShadowHarness(
        strategies=[MAFilterEchoStrategy(strategy_name="whale_filtered")],
        log_dir=log_dir,
        log_prefix="whale_filter",
    )


# [P310] SINGLE SOURCE — see the note in defense/regime_book_shadow.py.
# The echo harnesses write these two; the v5.1 strategy objects carry
# their own `strategy_name` and are listed so the conformance test can
# see the ARCHIVED (P199-KILL) families too.
SHADOW_STRATEGY_NAMES = frozenset({
    "ma_filtered", "whale_filtered",
    "ofi", "vpin_spike", "kyle_lambda", "stop_hunt_defense",
    "cascade_anticipation", "ml_factor",
    "funding_extreme", "funding_mean_reversion",
    "funding_post_etf_regime",
})
