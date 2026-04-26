"""
agents/exit_drl_agent.py — Exit DRL runtime inference wrapper (Stage 3 SHADOW).

Loads the discrete-SAC checkpoint trained by `training/exit_drl/train_exit_sac.py`
and exposes a per-tick prediction call that the runtime can consult for exit-
timing actions.

Stage-3 v1 SHADOW design (per HMATS_EXIT_DRL_TRAINING.md):
- The agent runs every tick and writes (state-summary, predicted action, probs)
  to `data/exit_drl_shadow.jsonl`. It does **not** influence trading decisions.
- The existing `execution/exit_alpha.py` triggers (phase / CRACK / momentum / DRL
  PARTIAL_EXIT / drawdown) keep operating exactly as today.
- Promotion to EXIT_ONLY happens later, after 30 days of shadow + ≥30 real exit
  events, gated by an Exit-DRL-specific authority gate (NOT the TQC promotion
  gate — System 1 outage shouldn't demote System 3, see CLAUDE.md P28).

Three DRL systems in this codebase — see CLAUDE.md P28:
  1. `drl/ensemble.py` (TQC, ACTIVE)              — direction + confidence.
  2. `agents/drl_agent.py` (DISABLED scaffolding) — Phase-2 stub.
  3. `agents/exit_drl_agent.py` (THIS FILE, SHADOW) — exit-timing-only SAC.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Shared constants — must match training/exit_drl/*.py
# ────────────────────────────────────────────────────────────────
STATE_DIM = 40
ACTION_DIM = 4

MARKET_COLS = [
    'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 'ret_168h',
    'vol_24h', 'vol_ratio_s', 'atr_14_denoised',
    'rsi_14', 'macd_12_26', 'bb_pos_20', 'bb_width_20',
    'vol_zscore', 'vol_spike',
]
FLOW_COLS = [
    'funding_rate_zscore', 'oi_change_5d', 'liq_imbalance',
    'taker_ratio_zscore', 'tradecount_zscore', 'taker_vol_momentum',
    'panic', 'capitulation',
]


class ExitAction(Enum):
    HOLD = 0
    PARTIAL_EXIT = 1
    RELEASE_RUNNER = 2
    EXIT_ALL = 3


class ExitDRLMode(Enum):
    DISABLED = "DISABLED"   # don't even run inference
    SHADOW = "SHADOW"       # run + log, never influence decisions
    EXIT_ONLY = "EXIT_ONLY" # consult for exit triggers (post-promotion)


@dataclass
class ExitDRLPrediction:
    action: ExitAction
    probabilities: List[float]      # [HOLD, PARTIAL, RELEASE, EXIT]
    confidence: float                # = max prob
    second_choice_action: ExitAction
    state_summary: Dict[str, float]  # bars_held, unrealized_pnl, drawdown_from_peak
    inference_ms: float


# ────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────
class ExitDRLAgent:
    """
    Runtime inference wrapper for the per-asset Exit DRL.

    Lazy-loads `models/exit_drl/{ASSET}/exit_sac_best.pt` for each asset on first
    use. Missing checkpoint → that asset stays DISABLED for the lifetime of the
    process (no per-tick load attempts).
    """

    def __init__(
        self,
        models_dir: str = "models/exit_drl",
        shadow_log_path: str = "data/exit_drl_shadow.jsonl",
        mode: ExitDRLMode = ExitDRLMode.SHADOW,
        assets: Optional[List[str]] = None,
        device: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        per_asset_modes: Optional[Dict[str, ExitDRLMode]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.shadow_log_path = Path(shadow_log_path)
        self.mode = mode  # default for assets not in per_asset_modes
        self.assets = assets or ["BTC", "ETH", "SOL"]
        # Per-asset mode override — promotion is per-asset so we can isolate
        # bad behavior on one asset from corrupting the others (CLAUDE.md P28
        # accelerated promotion path #3).
        self._asset_modes: Dict[str, ExitDRLMode] = {
            a: (per_asset_modes or {}).get(a, mode) for a in self.assets
        }
        # Forwarded to safe_torch_load.validate_model_path so tests (or
        # alternate deploy layouts) can authorize a non-default model root
        # without touching the global allowlist.
        self._extra_allowed_roots = list(extra_allowed_roots or [])

        # Lazy-init torch — keeps the agent importable on a runtime without torch
        try:
            import torch
            self._torch = torch
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
        except ImportError:
            self._torch = None
            self._device = "cpu"
            logger.warning("[EXIT_DRL] torch unavailable — agent will be DISABLED")
            self.mode = ExitDRLMode.DISABLED

        # Per-asset state
        self._actors: Dict[str, Any] = {}              # asset → torch.nn.Module
        self._asset_status: Dict[str, str] = {}         # asset → "READY" | "MISSING" | "ERROR:..."
        self._predictions_logged: int = 0
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)

        # P81 2026-04-26: self-contained peak tracker for the
        # `max_unrealized_pnl_pct` state input. The runtime caller never
        # set position['max_unrealized_pnl_pct'] (P15-shape half-wired
        # bug — reader at build_state:220 had no writer). Result:
        # build_state's fallback resolved max_unrealized==unrealized_pnl
        # ALWAYS → drawdown_from_peak==0 ALWAYS → state inputs [3] and
        # [4] carried zero information about peak position. Production
        # action distribution collapsed to HOLD/EXIT_ALL only — zero
        # PARTIAL_EXIT predictions despite 26-31% expected from offline
        # validation.
        #
        # Fix: track peak per asset inside the agent itself. Reset when
        # the (direction, entry_price) tuple changes (a new position
        # opened). The agent now guarantees its own state-input
        # correctness regardless of caller discipline.
        self._peak_unrealized_per_asset: Dict[str, float] = {}
        self._position_id_per_asset: Dict[str, Tuple[float, float]] = {}

        if self.mode != ExitDRLMode.DISABLED:
            self._load_all()

        logger.info(
            f"[EXIT_DRL] init mode={self.mode.value} device={self._device} "
            f"ready={[a for a, s in self._asset_status.items() if s == 'READY']}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        for asset in self.assets:
            self._asset_status[asset] = self._load_asset(asset)

    def _load_asset(self, asset: str) -> str:
        ckpt_path = self.models_dir / asset / "exit_sac_best.pt"
        if not ckpt_path.exists():
            logger.info(f"[EXIT_DRL] {asset}: no checkpoint at {ckpt_path} — DISABLED for this asset")
            return "MISSING"

        try:
            from training.exit_drl.train_exit_sac import DiscreteActor
            from infra.safe_torch_load import safe_torch_load
            ckpt = safe_torch_load(
                ckpt_path, map_location=self._device, weights_only=False,
                torch_module=self._torch,
                extra_allowed_roots=self._extra_allowed_roots or None,
            )
            ckpt_state_dim = ckpt.get("state_dim", STATE_DIM)
            ckpt_action_dim = ckpt.get("action_dim", ACTION_DIM)
            if ckpt_state_dim != STATE_DIM or ckpt_action_dim != ACTION_DIM:
                logger.warning(
                    f"[EXIT_DRL] {asset}: checkpoint dim mismatch "
                    f"(state {ckpt_state_dim} vs {STATE_DIM}, action {ckpt_action_dim} vs {ACTION_DIM}) — DISABLED"
                )
                return f"DIM_MISMATCH"

            actor = DiscreteActor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(self._device)
            actor.load_state_dict(ckpt["actor_state_dict"])
            actor.eval()
            self._actors[asset] = actor

            align = ckpt.get("val_alignment", float("nan"))
            epoch = ckpt.get("epoch", -1)
            logger.info(
                f"[EXIT_DRL] {asset}: loaded {ckpt_path.name} "
                f"(epoch={epoch}, val_alignment={align:.3f})"
            )
            return "READY"
        except Exception as e:
            logger.warning(f"[EXIT_DRL] {asset}: load failed: {e}")
            return f"ERROR:{type(e).__name__}"

    # ------------------------------------------------------------------
    # State construction (mirrors training/exit_drl/generate_expert_trajectories.py)
    # ------------------------------------------------------------------
    @staticmethod
    def build_state(
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        bars_held: int,
        max_bars_held: int = 48,
    ) -> np.ndarray:
        """
        Build the 40-d state vector at runtime.

        Args:
            position: {direction, entry_price, exposure, max_unrealized_pnl_pct, ...}
            market_data: tick's market_data dict (has the parquet columns +
                         current_price + regime info).
            bars_held: ticks since entry (0 if no position).
            max_bars_held: must equal training MAX_BARS_HELD (48).
        """
        direction = float(position.get("direction", 0.0) or 0.0)
        if direction == 0:
            # Caller should have skipped — defensive zero
            return np.zeros(STATE_DIM, dtype=np.float32)

        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        current_price = float(market_data.get("current_price", 0.0) or 0.0)
        if entry_price <= 0 or current_price <= 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        unrealized_pnl = direction * (current_price - entry_price) / entry_price
        max_unrealized = float(position.get("max_unrealized_pnl_pct", unrealized_pnl) or unrealized_pnl)
        max_unrealized = max(max_unrealized, unrealized_pnl)
        drawdown_from_peak = max(0.0, max_unrealized - unrealized_pnl)

        position_arr = np.array([
            direction,
            min(bars_held / max_bars_held, 1.0),
            unrealized_pnl,
            max_unrealized,
            drawdown_from_peak,
            min(bars_held / 12.0, 1.0),
            current_price / entry_price - 1.0,
            1.0 if bars_held > 12 else 0.0,
        ], dtype=np.float32)

        regime_scalar = float(market_data.get("regime", 0) or 0) / 7.0
        regime_arr = np.array([
            float(market_data.get(f"regime_proba_{i}", 0.0) or 0.0) for i in range(8)
        ] + [
            float(market_data.get("regime_confidence", 0.0) or 0.0),
            regime_scalar,
        ], dtype=np.float32)

        market_arr = np.array(
            [float(market_data.get(c, 0.0) or 0.0) for c in MARKET_COLS],
            dtype=np.float32,
        )
        flow_arr = np.array(
            [float(market_data.get(c, 0.0) or 0.0) for c in FLOW_COLS],
            dtype=np.float32,
        )

        state = np.concatenate([position_arr, regime_arr, market_arr, flow_arr])
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        asset: str,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        bars_held: int,
    ) -> Optional[ExitDRLPrediction]:
        """
        Run one inference. Returns None if the agent is unavailable for this
        asset. In SHADOW mode the prediction is logged to JSONL but NOT used.
        In EXIT_ONLY mode the prediction is logged AND the runtime is allowed
        to act on it (caller must check `should_act_on(asset)`).
        """
        if self.mode_for_asset(asset) == ExitDRLMode.DISABLED:
            return None
        if self._torch is None or self._asset_status.get(asset) != "READY":
            return None
        if not position or float(position.get("direction", 0.0) or 0.0) == 0.0:
            # No position — clear the peak tracker so a future entry starts
            # fresh. Per CLAUDE.md P81 invariant.
            self._peak_unrealized_per_asset.pop(asset, None)
            self._position_id_per_asset.pop(asset, None)
            return None

        # P81 2026-04-26: maintain own peak tracker (caller was never
        # writing position['max_unrealized_pnl_pct']). Reset when the
        # position identity changes — different (direction, entry_price)
        # = different trade.
        _direction = float(position.get("direction", 0.0) or 0.0)
        _entry_price = float(position.get("entry_price", 0.0) or 0.0)
        _current_price = float(market_data.get("current_price", 0.0) or 0.0)
        if _entry_price > 0 and _current_price > 0:
            _curr_pnl = _direction * (_current_price - _entry_price) / _entry_price
            _pos_id = (_direction, _entry_price)
            _prev_id = self._position_id_per_asset.get(asset)
            if _prev_id != _pos_id:
                # New position — start peak from current pnl (not from
                # zero — entry might have already moved by the time we
                # see it).
                self._peak_unrealized_per_asset[asset] = _curr_pnl
                self._position_id_per_asset[asset] = _pos_id
            else:
                # Same position — ratchet peak upward.
                _prev_peak = self._peak_unrealized_per_asset.get(asset, _curr_pnl)
                self._peak_unrealized_per_asset[asset] = max(_prev_peak, _curr_pnl)

            # Inject the tracked peak into the position dict so build_state
            # gets the real peak instead of falling back to current pnl.
            # Honor a caller-set value if it's HIGHER (caller might be
            # tracking via a different mechanism); otherwise use ours.
            _caller_peak = position.get("max_unrealized_pnl_pct")
            _our_peak = self._peak_unrealized_per_asset[asset]
            if _caller_peak is None:
                position["max_unrealized_pnl_pct"] = _our_peak
            else:
                try:
                    position["max_unrealized_pnl_pct"] = max(
                        float(_caller_peak), _our_peak
                    )
                except (TypeError, ValueError):  # noqa: silent-swallow
                    # Malformed caller value — defensive fallback to ours.
                    position["max_unrealized_pnl_pct"] = _our_peak

        state_np = self.build_state(position, market_data, bars_held)
        t0 = time.monotonic()
        with self._torch.no_grad():
            state_t = self._torch.tensor(state_np, device=self._device).unsqueeze(0)
            probs, _ = self._actors[asset](state_t)
            probs_np = probs.squeeze(0).cpu().numpy()
        inference_ms = (time.monotonic() - t0) * 1000.0

        action_idx = int(np.argmax(probs_np))
        sorted_idx = np.argsort(-probs_np)
        action = ExitAction(action_idx)
        second = ExitAction(int(sorted_idx[1]))
        confidence = float(probs_np[action_idx])

        prediction = ExitDRLPrediction(
            action=action,
            probabilities=[float(p) for p in probs_np.tolist()],
            confidence=confidence,
            second_choice_action=second,
            state_summary={
                "direction": float(position.get("direction", 0.0) or 0.0),
                "bars_held": float(bars_held),
                "unrealized_pnl": float(state_np[2]),
                "max_unrealized": float(state_np[3]),
                "drawdown_from_peak": float(state_np[4]),
            },
            inference_ms=inference_ms,
        )

        # Always log in SHADOW + EXIT_ONLY (auditable trail of every prediction
        # that influenced — or was ignored by — the runtime).
        if self.mode_for_asset(asset) in (ExitDRLMode.SHADOW, ExitDRLMode.EXIT_ONLY):
            self._log_shadow(asset, prediction)

        return prediction

    # ------------------------------------------------------------------
    # Per-asset mode helpers
    # ------------------------------------------------------------------
    def mode_for_asset(self, asset: str) -> ExitDRLMode:
        return self._asset_modes.get(asset, self.mode)

    def set_mode_for_asset(self, asset: str, mode: ExitDRLMode) -> None:
        prev = self._asset_modes.get(asset, self.mode)
        self._asset_modes[asset] = mode
        if prev != mode:
            logger.warning(
                f"[EXIT_DRL] {asset}: mode {prev.value} -> {mode.value}"
            )

    def should_act_on(self, asset: str) -> bool:
        """True iff the runtime is authorized to act on this asset's prediction."""
        return self.mode_for_asset(asset) == ExitDRLMode.EXIT_ONLY

    # ------------------------------------------------------------------
    # Shadow logging
    # ------------------------------------------------------------------
    def _log_shadow(self, asset: str, p: ExitDRLPrediction) -> None:
        """Append one JSONL line. Best-effort — failure does not block trading."""
        try:
            record = {
                "ts": time.time(),
                "asset": asset,
                "mode": self.mode.value,
                "action": p.action.name,
                "action_idx": p.action.value,
                "confidence": round(p.confidence, 4),
                "probs": [round(x, 4) for x in p.probabilities],
                "second_choice": p.second_choice_action.name,
                "state": {k: round(v, 6) for k, v in p.state_summary.items()},
                "inference_ms": round(p.inference_ms, 2),
            }
            with self.shadow_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._predictions_logged += 1
        except Exception as e:
            logger.debug(f"[EXIT_DRL] shadow log write failed: {e}")

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "asset_modes": {a: m.value for a, m in self._asset_modes.items()},
            "device": self._device,
            "assets": dict(self._asset_status),
            "predictions_logged": self._predictions_logged,
            "shadow_log": str(self.shadow_log_path),
        }


# ────────────────────────────────────────────────────────────────
# Module-level singleton accessor (mirror the pattern used elsewhere)
# ────────────────────────────────────────────────────────────────
_singleton: Optional[ExitDRLAgent] = None


def get_exit_drl_agent(
    models_dir: str = "models/exit_drl",
    shadow_log_path: str = "data/exit_drl_shadow.jsonl",
    mode: ExitDRLMode = ExitDRLMode.SHADOW,
    assets: Optional[List[str]] = None,
    per_asset_modes: Optional[Dict[str, ExitDRLMode]] = None,
    extra_allowed_roots: Optional[List[str]] = None,
) -> ExitDRLAgent:
    global _singleton
    if _singleton is None:
        _singleton = ExitDRLAgent(
            models_dir=models_dir,
            shadow_log_path=shadow_log_path,
            mode=mode,
            assets=assets,
            per_asset_modes=per_asset_modes,
            extra_allowed_roots=extra_allowed_roots,
        )
    return _singleton


def reset_singleton() -> None:
    """For tests."""
    global _singleton
    _singleton = None
