"""
HMATS v5.1 Phase 7 - Portfolio Risk-Parity Sleeve Sizing (ADVISORY MODE)
==========================================================================

Risk-parity allocator across sleeves + Quarter-Kelly internal cap per sleeve.

This Phase ships the allocator + per-tick advisory record. The output is
written to data/strategy_shadow/sleeve_allocations_*.jsonl as a parallel
log to the strategy shadows. It does NOT yet drive UnifiedPositionSizer —
Phase 10 (Day 57+) will wire the recommendations into live sizing after
30d of advisory observation validates against realized portfolio vol.

Iron Laws honored:
  1. obs_dim=126: untouched (no feature changes).
  2. constitution: untouched.
  3. training/: untouched.
  4. fail-closed: missing/invalid vol -> sleeve dropped from allocation;
                  empty allocator -> {} (caller defaults to existing sizer).
  5. DRL ACTIVE: allocator outputs are advisory; DRL authority unaffected.
  6. >=3 active strategies: allocator does NOT archive sleeves; only weights.
  7. Shadow >=30d: advisory output, no live sizer integration until Phase 10.

Design notes:
  - Inverse-vol risk parity: sleeve weight ~ 1 / sleeve_vol.
  - Hard sleeve cap (default 0.50) prevents low-vol sleeves from concentrating.
  - Total leverage cap (default 2.0) bounds gross portfolio exposure.
  - Quarter-Kelly per-sleeve = (sharpe / vol) / 4.0, then clamped to [0, 1].
  - Sleeve realized vol updated by `update_realized_vol(name, vol)` from
    equity-history-derived per-sleeve PnL. For Phase 7, shadow sleeves use
    their `estimated_vol_pct` until they have real PnL (Phase 10+).
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Iron Law caps (overridable per instance)
DEFAULT_MAX_PORTFOLIO_LEVERAGE = 2.0
DEFAULT_MAX_SLEEVE_WEIGHT = 0.50
DEFAULT_TARGET_PORTFOLIO_VOL = 0.25     # 25% per CLAUDE.md memory
DEFAULT_VOL_FLOOR = 0.05                # never divide-by-zero on a sleeve
DEFAULT_VOL_CEILING = 1.50              # absurd-vol guard
DEFAULT_KELLY_CLAMP = (0.0, 1.0)        # Quarter-Kelly bound


@dataclass
class SleeveProfile:
    """One row in the allocator's sleeve table."""
    name: str
    estimated_vol_pct: float            # initial vol estimate (pre-realized)
    sharpe_estimate: float              # initial Sharpe estimate
    is_live: bool = False               # True = real PnL drives realized vol
    realized_vol_history: deque = field(default_factory=lambda: deque(maxlen=30))

    def current_vol(self) -> float:
        """Return realized vol if available; else estimate."""
        if self.realized_vol_history:
            # Rolling-window average; tolerant to sparse data
            vals = [v for v in self.realized_vol_history if v is not None]
            if vals:
                return max(DEFAULT_VOL_FLOOR, min(DEFAULT_VOL_CEILING, sum(vals) / len(vals)))
        return max(DEFAULT_VOL_FLOOR, min(DEFAULT_VOL_CEILING, self.estimated_vol_pct))


class SleeveAllocator:
    """Risk-parity sleeve sizer with Quarter-Kelly internal bound.

    Usage (Phase 7 advisory, Phase 10+ live):
        alloc = SleeveAllocator(target_portfolio_vol=0.25)
        alloc.register_sleeve("directional_short", est_vol=0.45, sharpe=0.8, is_live=True)
        alloc.register_sleeve("microstructure",    est_vol=0.20, sharpe=1.0, is_live=False)
        alloc.register_sleeve("cascade",           est_vol=0.30, sharpe=0.9, is_live=False)
        weights = alloc.compute_allocations()
        record = alloc.advisory_record_for(asset="BTC")  # writes diagnostics
    """

    def __init__(
        self,
        target_portfolio_vol: float = DEFAULT_TARGET_PORTFOLIO_VOL,
        max_portfolio_leverage: float = DEFAULT_MAX_PORTFOLIO_LEVERAGE,
        max_sleeve_weight: float = DEFAULT_MAX_SLEEVE_WEIGHT,
    ) -> None:
        self.target_portfolio_vol = float(target_portfolio_vol)
        self.max_portfolio_leverage = float(max_portfolio_leverage)
        self.max_sleeve_weight = float(max_sleeve_weight)
        self._sleeves: Dict[str, SleeveProfile] = {}
        self._allocation_count = 0

    # -------------------- registration --------------------

    def register_sleeve(
        self,
        name: str,
        estimated_vol_pct: float,
        sharpe_estimate: float,
        is_live: bool = False,
    ) -> None:
        """Add or replace a sleeve. Idempotent on `name`."""
        if not name:
            logger.warning("[SLEEVE_ALLOC] register_sleeve: empty name ignored")
            return
        try:
            est_vol = float(estimated_vol_pct)
            sharpe = float(sharpe_estimate)
        except (TypeError, ValueError):
            logger.warning(f"[SLEEVE_ALLOC] register_sleeve({name}): non-numeric inputs, skipped")
            return
        if est_vol <= 0 or est_vol != est_vol:  # NaN guard
            logger.warning(f"[SLEEVE_ALLOC] register_sleeve({name}): vol={est_vol} invalid, skipped")
            return

        # Preserve realized history if re-registering (re-init on first add only)
        prev = self._sleeves.get(name)
        if prev is not None:
            prev.estimated_vol_pct = est_vol
            prev.sharpe_estimate = sharpe
            prev.is_live = is_live
        else:
            self._sleeves[name] = SleeveProfile(
                name=name,
                estimated_vol_pct=est_vol,
                sharpe_estimate=sharpe,
                is_live=is_live,
            )
        logger.info(
            f"[SLEEVE_ALLOC] {'updated' if prev else 'registered'} sleeve: "
            f"name={name} est_vol={est_vol:.2%} sharpe={sharpe:.2f} live={is_live}"
        )

    def update_realized_vol(self, name: str, vol: float) -> None:
        """Append a realized-vol observation to a sleeve's rolling history.

        Caller computes vol from per-sleeve PnL slice (e.g. equity-history
        attribution per sleeve). For Phase 7 only directional_short has live
        attribution; shadow sleeves rely on their estimated_vol_pct until
        Phase 10 promotion.
        """
        sleeve = self._sleeves.get(name)
        if sleeve is None:
            logger.debug(f"[SLEEVE_ALLOC] update_realized_vol({name}): unknown sleeve")
            return
        try:
            f = float(vol)
            if f != f or f <= 0:
                return
        except (TypeError, ValueError):  # noqa: silent-swallow
            # Per-call type coercion; bad input is a no-op observation.
            return
        sleeve.realized_vol_history.append(f)

    # -------------------- math --------------------

    def compute_allocations(self) -> Dict[str, float]:
        """Return {sleeve_name: weight}. Empty dict if no sleeves registered.

        Algorithm:
          1. inverse-vol weighting -> raw_w[i] = (1 / vol[i]) / sum(1/vol)
          2. cap each weight at max_sleeve_weight; redistribute residual
             proportionally to remaining sleeves (one pass — sufficient for
             N <= 8 typical sleeve counts).
          3. compute portfolio_vol_unit = sum(w[i] * vol[i])
          4. leverage_scalar = clamp(target_vol / portfolio_vol_unit, 0,
             max_portfolio_leverage)
          5. final_w[i] = raw_w[i] * leverage_scalar
        """
        if not self._sleeves:
            return {}

        vols = {n: s.current_vol() for n, s in self._sleeves.items()}
        inv_vol = {n: 1.0 / v for n, v in vols.items() if v > 0}
        if not inv_vol:
            return {}

        total_inv = sum(inv_vol.values())
        if total_inv <= 0:
            return {}

        raw = {n: iv / total_inv for n, iv in inv_vol.items()}

        # Apply per-sleeve cap with residual redistribution
        capped = self._apply_sleeve_cap(raw)

        # Portfolio-level leverage scalar
        portfolio_vol_unit = sum(capped[n] * vols[n] for n in capped)
        if portfolio_vol_unit <= 0:
            return {n: 0.0 for n in capped}

        leverage_scalar = self.target_portfolio_vol / portfolio_vol_unit
        leverage_scalar = max(0.0, min(self.max_portfolio_leverage, leverage_scalar))

        final = {n: capped[n] * leverage_scalar for n in capped}
        self._allocation_count += 1
        return final

    def _apply_sleeve_cap(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Cap each weight at max_sleeve_weight; redistribute excess proportionally
        to non-capped sleeves. Single-pass — converges for typical sleeve counts."""
        cap = self.max_sleeve_weight
        if not raw:
            return {}
        if all(w <= cap for w in raw.values()):
            return dict(raw)

        capped: Dict[str, float] = {}
        excess = 0.0
        uncapped_total = 0.0
        for n, w in raw.items():
            if w > cap:
                capped[n] = cap
                excess += (w - cap)
            else:
                capped[n] = w
                uncapped_total += w

        if excess > 0 and uncapped_total > 0:
            scale = (uncapped_total + excess) / uncapped_total
            for n in capped:
                if capped[n] < cap:
                    capped[n] = min(cap, capped[n] * scale)
        return capped

    @staticmethod
    def quarter_kelly_per_sleeve(sharpe: float, vol: float) -> float:
        """(sharpe / vol) / 4, clamped to [0, 1]."""
        try:
            v = float(vol)
            s = float(sharpe)
            if v <= 0 or v != v or s != s:
                return 0.0
        except (TypeError, ValueError):  # noqa: silent-swallow
            # Quarter-Kelly defaults to 0 (no position) on bad input; safe.
            return 0.0
        raw = (s / v) / 4.0
        lo, hi = DEFAULT_KELLY_CLAMP
        return max(lo, min(hi, raw))

    # -------------------- advisory output --------------------

    def advisory_record_for(self, asset: str) -> Dict[str, Any]:
        """Build a JSON-serializable advisory record for the JSONL sink.

        Iron Law 7: this record is what the allocator WOULD recommend at
        Phase 10 promotion. It is NOT consumed by UnifiedPositionSizer in
        Phase 7. The record includes per-sleeve weight, current_vol,
        Quarter-Kelly bound, and the leverage scalar that would apply.
        """
        weights = self.compute_allocations()
        sleeve_records = []
        for name, sleeve in self._sleeves.items():
            v = sleeve.current_vol()
            sleeve_records.append({
                "name": name,
                "weight": float(weights.get(name, 0.0)),
                "current_vol": float(v),
                "estimated_vol": float(sleeve.estimated_vol_pct),
                "sharpe_estimate": float(sleeve.sharpe_estimate),
                "is_live": bool(sleeve.is_live),
                "quarter_kelly": self.quarter_kelly_per_sleeve(sleeve.sharpe_estimate, v),
                "realized_vol_n": len(sleeve.realized_vol_history),
            })
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "target_portfolio_vol": self.target_portfolio_vol,
            "max_portfolio_leverage": self.max_portfolio_leverage,
            "max_sleeve_weight": self.max_sleeve_weight,
            "sleeves": sleeve_records,
            "total_weight": float(sum(weights.values())),
            "advisory_only": True,
        }


# ---------------------------------------------------------------------------
# Persistence: lightweight JSONL sink, fail-closed
# ---------------------------------------------------------------------------

class SleeveAdvisorySink:
    """JSONL writer for sleeve allocator advisory records.

    Mirrors the strategy_shadow sink shape so Pre-6 IC compute can iterate
    cleanly. Path: data/strategy_shadow/sleeve_allocations_YYYYMMDD.jsonl.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self._log_dir = log_dir or self._resolve_log_dir()
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"[SLEEVE_SINK] log dir create failed ({self._log_dir}): "
                f"{type(e).__name__}: {e}"
            )
        self._observation_count = 0
        self._error_count = 0

    @staticmethod
    def _resolve_log_dir() -> Path:
        repo = Path(__file__).resolve().parents[1]
        env_data = os.environ.get("HMATS_DATA_DIR")
        if env_data:
            return Path(env_data) / "strategy_shadow"
        return repo / "data" / "strategy_shadow"

    def _log_path(self, ts: datetime) -> Path:
        return self._log_dir / f"sleeve_allocations_{ts:%Y%m%d}.jsonl"

    def write(self, record: Dict[str, Any]) -> None:
        try:
            ts_str = record.get("ts", datetime.now(timezone.utc).isoformat())
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        path = self._log_path(ts)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:  # noqa: silent-swallow
                    # fsync best-effort: not all platforms / filesystems support
                    # fsync on this fd type (e.g. Windows + some network mounts).
                    # f.flush() already gave us OS-level durability; fsync is the
                    # crash-safety upgrade. CLAUDE.md P72 lint annotation.
                    pass
            self._observation_count += 1
        except Exception as e:
            self._error_count += 1
            logger.warning(
                f"[SLEEVE_SINK] write failed ({path}): {type(e).__name__}: {e}"
            )

    def stats(self) -> Dict[str, int]:
        return {
            "observations": self._observation_count,
            "errors": self._error_count,
        }


# ---------------------------------------------------------------------------
# Builder for main.py — Phase 7 sleeve set
# ---------------------------------------------------------------------------

def build_phase7_sleeve_allocator() -> SleeveAllocator:
    """Construct the Phase 7 allocator with the 3 currently active sleeves.

    Sleeves NOT yet registered (deferred to their phases):
      - funding         (Phase 3, post-Coinbase migration)
      - ml_factor       (Phase 6, after Pre-6 backtest framework lands)
      - cash_and_carry  (deferred v6 per V5 BIS analysis at $10K AUM)
      - options         (deferred v6 per V13 Deribit access RED)

    Vol/Sharpe estimates per v5.1 prompt's V9 sleeve table.
    """
    a = SleeveAllocator(
        target_portfolio_vol=DEFAULT_TARGET_PORTFOLIO_VOL,
        max_portfolio_leverage=DEFAULT_MAX_PORTFOLIO_LEVERAGE,
        max_sleeve_weight=DEFAULT_MAX_SLEEVE_WEIGHT,
    )
    a.register_sleeve("directional_short", estimated_vol_pct=0.45, sharpe_estimate=0.8, is_live=True)
    a.register_sleeve("microstructure",    estimated_vol_pct=0.20, sharpe_estimate=1.0, is_live=False)
    a.register_sleeve("cascade",           estimated_vol_pct=0.30, sharpe_estimate=0.9, is_live=False)
    return a
