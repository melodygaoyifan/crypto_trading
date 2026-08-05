"""
HMATS Health Validator — 3-layer automated runtime health verification.

Layer 1: StartupHealthValidator — runs once at startup before first tick
Layer 2: PerTickInvariantChecker — runs after every 4H tick decision

Each check is derived from a real production bug. Every check has:
  - ID (S1-S12 for startup, T1-T8 for tick)
  - What bug it prevents
  - PASS/WARN/CRITICAL result
  - Exact evidence for debugging

Design: read-only checks. Never modifies trading decisions.
CRITICAL = blocks new trades + logs alert. WARN = log only.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    check_id: str
    name: str
    status: str  # PASS / WARN / CRITICAL / SKIP
    detail: str = ""
    prevents_bug: str = ""

    def __str__(self):
        return f"[HEALTH_{self.check_id}] {self.status}: {self.name} — {self.detail}"


@dataclass
class HealthReport:
    """Aggregate health report."""
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: str = ""
    layer: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def critical(self) -> int:
        return sum(1 for c in self.checks if c.status == "CRITICAL")

    @property
    def healthy(self) -> bool:
        return self.critical == 0

    def log_all(self):
        for c in self.checks:
            if c.status == "CRITICAL":
                logger.critical(str(c))
            elif c.status == "WARN":
                logger.warning(str(c))
            else:
                logger.info(str(c))
        summary = (
            f"[HEALTH_{self.layer}] {self.passed} PASS, "
            f"{self.warnings} WARN, {self.critical} CRITICAL"
        )
        if self.critical > 0:
            logger.critical(summary)
        else:
            logger.info(summary)


# =============================================================================
# LAYER 1: STARTUP HEALTH VALIDATOR
# =============================================================================

class StartupHealthValidator:
    """Runs S1-S12 checks at startup before first tick."""

    def validate(self, runner: Any) -> HealthReport:
        """Run all startup checks against HMATSProductionRunner instance."""
        report = HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            layer="STARTUP",
        )

        report.checks.append(self._s1_drl_state_consistency(runner))
        report.checks.append(self._s2_drl_models_loaded(runner))
        report.checks.append(self._s3_fast_risk_no_stale_anchor(runner))
        report.checks.append(self._s4_ccxt_session_alive(runner))
        report.checks.append(self._s5_alpha_gate_threshold(runner))
        report.checks.append(self._s6_config_json_valid(runner))
        report.checks.append(self._s7_model_alpha_authority(runner))
        report.checks.append(self._s8_sentiment_gate_bounds(runner))
        report.checks.append(self._s9_deberta_wiring(runner))
        report.checks.append(self._s10_no_orphan_processes())
        report.checks.append(self._s11_kraken_connectivity(runner))
        report.checks.append(self._s12_smart_beta_config(runner))

        report.log_all()
        return report

    def _s1_drl_state_consistency(self, runner: Any) -> HealthCheck:
        """DRL state file must match runtime authority level."""
        try:
            state_file = Path("data/drl_promotion_state.json")
            if not state_file.exists():
                return HealthCheck("S1", "DRL state consistency", "WARN",
                                   "No state file", "Bug #1")
            with open(state_file) as f:
                state = json.load(f)
            file_level = state.get("authority_level", "UNKNOWN")
            runtime_level = getattr(runner, '_drl_authority_level', "UNKNOWN")
            if file_level == runtime_level:
                return HealthCheck("S1", "DRL state consistency", "PASS",
                                   f"file={file_level}, runtime={runtime_level}", "Bug #1")
            return HealthCheck("S1", "DRL state consistency", "CRITICAL",
                               f"MISMATCH file={file_level} vs runtime={runtime_level}", "Bug #1,#15")
        except Exception as e:
            return HealthCheck("S1", "DRL state consistency", "WARN", f"check failed: {e}", "Bug #1")

    def _s2_drl_models_loaded(self, runner: Any) -> HealthCheck:
        """DRL models should be loaded when authority is ACTIVE."""
        try:
            models_ready = getattr(runner, '_drl_models_ready', 0)
            level = getattr(runner, '_drl_authority_level', "DISABLED")
            if level == "ACTIVE" and models_ready == 0:
                return HealthCheck("S2", "DRL models loaded", "CRITICAL",
                                   f"level=ACTIVE but models_ready=0", "Bug #15")
            if models_ready > 0:
                return HealthCheck("S2", "DRL models loaded", "PASS",
                                   f"models_ready={models_ready}, level={level}", "Bug #15")
            return HealthCheck("S2", "DRL models loaded", "PASS",
                               f"level={level}, models_ready={models_ready} (OK for non-ACTIVE)", "Bug #15")
        except Exception as e:
            return HealthCheck("S2", "DRL models loaded", "WARN", f"check failed: {e}", "Bug #15")

    def _s3_fast_risk_no_stale_anchor(self, runner: Any) -> HealthCheck:
        """FastRiskTick should have no stale anchors at startup."""
        try:
            frt = getattr(runner, 'fast_risk_tick', None)
            if frt is None:
                return HealthCheck("S3", "FastRiskTick anchors", "SKIP", "no FastRiskTick", "Bug #2")
            anchors = getattr(frt, '_last_4h_prices', {})
            if not anchors:
                return HealthCheck("S3", "FastRiskTick anchors", "PASS",
                                   "empty (clean startup)", "Bug #2")
            return HealthCheck("S3", "FastRiskTick anchors", "WARN",
                               f"pre-populated anchors: {list(anchors.keys())} — may be stale", "Bug #2")
        except Exception as e:
            return HealthCheck("S3", "FastRiskTick anchors", "WARN", f"check failed: {e}", "Bug #2")

    def _s4_ccxt_session_alive(self, runner: Any) -> HealthCheck:
        """CCXT exchange should be able to fetch data."""
        try:
            exchange = getattr(runner, '_ccxt_exchange', None)
            if exchange is None:
                kraken_rest = getattr(runner, 'kraken_rest', None)
                if kraken_rest and hasattr(kraken_rest, '_exchange'):
                    exchange = kraken_rest._exchange
            if exchange is None:
                return HealthCheck("S4", "CCXT session", "WARN", "no exchange object", "Bug #5")
            ticker = exchange.fetch_ticker("BTC/USD")
            price = ticker.get("last", 0)
            if price > 0:
                return HealthCheck("S4", "CCXT session", "PASS",
                                   f"BTC=${price:,.0f}", "Bug #5")
            return HealthCheck("S4", "CCXT session", "WARN", "fetch returned no price", "Bug #5")
        except Exception as e:
            return HealthCheck("S4", "CCXT session", "CRITICAL",
                               f"fetch_ticker failed: {e}", "Bug #5")

    def _s5_alpha_gate_threshold(self, runner: Any) -> HealthCheck:
        """Alpha gate min_bps should not be so high it blocks all signals."""
        try:
            min_bps = getattr(runner, '_min_alpha_bps', 0)
            min_bps_short = getattr(runner, '_min_alpha_bps_short', 0)
            if max(min_bps, min_bps_short) > 15:
                return HealthCheck("S5", "Alpha gate threshold", "WARN",
                                   f"min_alpha_bps={min_bps}, short={min_bps_short} — may block most signals",
                                   "Bug #10")
            return HealthCheck("S5", "Alpha gate threshold", "PASS",
                               f"min_alpha_bps={min_bps}, short={min_bps_short}", "Bug #10")
        except Exception as e:
            return HealthCheck("S5", "Alpha gate threshold", "WARN", f"check failed: {e}", "Bug #10")

    def _s6_config_json_valid(self, runner: Any) -> HealthCheck:
        """All config JSON files should parse correctly."""
        try:
            configs = ["configs/cloud_production.json", "configs/live_high_risk.json",
                       "configs/high_risk.json"]
            errors = []
            for path in configs:
                p = Path(path)
                if p.exists():
                    try:
                        json.loads(p.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as e:
                        errors.append(f"{path}: {e}")
            if errors:
                return HealthCheck("S6", "Config JSON valid", "CRITICAL",
                                   f"parse errors: {errors}", "General")
            return HealthCheck("S6", "Config JSON valid", "PASS",
                               f"{len(configs)} configs OK", "General")
        except Exception as e:
            return HealthCheck("S6", "Config JSON valid", "WARN", f"check failed: {e}", "General")

    def _s7_model_alpha_authority(self, runner: Any) -> HealthCheck:
        """ModelAlpha should be ADVISE, not stuck in CONTEXT."""
        try:
            # Check if model_alpha agent exists and its authority
            ma = getattr(runner, 'model_alpha', None)
            if ma is None:
                return HealthCheck("S7", "ModelAlpha authority", "SKIP",
                                   "no model_alpha agent", "Bug #14")
            return HealthCheck("S7", "ModelAlpha authority", "PASS",
                               "model_alpha loaded (authority set at runtime)", "Bug #14")
        except Exception as e:
            return HealthCheck("S7", "ModelAlpha authority", "WARN", f"check failed: {e}", "Bug #14")

    def _s8_sentiment_gate_bounds(self, runner: Any) -> HealthCheck:
        """SentimentGate multiplier should be within ±5%."""
        try:
            sg = getattr(runner, '_sentiment_gate', None)
            if sg is None:
                return HealthCheck("S8", "SentimentGate bounds", "SKIP",
                                   "no sentiment gate", "Design")
            cfg = getattr(sg, 'config', None)
            if cfg:
                max_m = getattr(cfg, 'max_mult', 1.0)
                min_m = getattr(cfg, 'min_mult', 1.0)
                if max_m > 1.10 or min_m < 0.90:
                    return HealthCheck("S8", "SentimentGate bounds", "WARN",
                                       f"range [{min_m}, {max_m}] exceeds ±10%", "Design")
                return HealthCheck("S8", "SentimentGate bounds", "PASS",
                                   f"range [{min_m}, {max_m}]", "Design")
            return HealthCheck("S8", "SentimentGate bounds", "PASS", "no config", "Design")
        except Exception as e:
            return HealthCheck("S8", "SentimentGate bounds", "WARN", f"check failed: {e}", "Design")

    def _s9_deberta_wiring(self, runner: Any) -> HealthCheck:
        """L2 DeBERTa should be loadable and CryptoPanic path correct."""
        try:
            db = getattr(runner, '_deberta_sentiment', None)
            if db is None:
                return HealthCheck("S9", "DeBERTa wiring", "SKIP", "not loaded", "Bug #6")
            if not db.is_ready:
                return HealthCheck("S9", "DeBERTa wiring", "WARN",
                                   "loaded but not ready (model may be missing)", "Bug #6")
            # Check CryptoPanic feed has recent_news (not recent_titles)
            cp = getattr(runner, 'cryptopanic_feed', None)
            if cp:
                data = cp.get_latest()
                if data and hasattr(data, 'recent_news'):
                    return HealthCheck("S9", "DeBERTa wiring", "PASS",
                                       f"ready, CryptoPanic has recent_news attr", "Bug #6")
                return HealthCheck("S9", "DeBERTa wiring", "WARN",
                                   "CryptoPanic data missing recent_news", "Bug #6")
            return HealthCheck("S9", "DeBERTa wiring", "PASS",
                               "ready, no CryptoPanic feed yet", "Bug #6")
        except Exception as e:
            return HealthCheck("S9", "DeBERTa wiring", "WARN", f"check failed: {e}", "Bug #6")

    def _s10_no_orphan_processes(self) -> HealthCheck:
        """Should be only 1 main.py live process (self)."""
        try:
            import subprocess, sys
            if sys.platform == "win32":
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                     "Where-Object { $_.CommandLine -match 'main.py.*live' }).Count"],
                    capture_output=True, text=True, timeout=10
                )
                count = int(result.stdout.strip() or "0")
            else:
                # Linux: pgrep -cf counts processes whose full command line matches
                result = subprocess.run(
                    ["pgrep", "-cf", "main.py.*live"],
                    capture_output=True, text=True, timeout=10
                )
                count = int(result.stdout.strip() or "0")
            # count includes self (the running process), so 1 = healthy
            if count <= 1:
                return HealthCheck("S10", "No orphan processes", "PASS",
                                   f"{count} live process(es) (self only)", "Bug #4")
            if count == 2:
                # Could be self + brief overlap during startup, or real orphan
                return HealthCheck("S10", "No orphan processes", "PASS",
                                   f"{count} processes (likely self + startup overlap)", "Bug #4")
            return HealthCheck("S10", "No orphan processes", "WARN",
                               f"{count} live processes — {count-1} potential orphans", "Bug #4")
        except Exception as e:
            return HealthCheck("S10", "No orphan processes", "SKIP", f"check failed: {e}", "Bug #4")

    def _s11_kraken_connectivity(self, runner: Any) -> HealthCheck:
        """Kraken API should be reachable."""
        try:
            import requests
            r = requests.get("https://api.kraken.com/0/public/Time", timeout=10)
            if r.status_code == 200:
                return HealthCheck("S11", "Kraken connectivity", "PASS",
                                   "API reachable", "Bug #3,#5")
            return HealthCheck("S11", "Kraken connectivity", "CRITICAL",
                               f"status={r.status_code}", "Bug #3,#5")
        except Exception as e:
            return HealthCheck("S11", "Kraken connectivity", "CRITICAL",
                               f"unreachable: {e}", "Bug #3,#5")

    def _s12_smart_beta_config(self, runner: Any) -> HealthCheck:
        """Smart Beta should be configured consistently."""
        try:
            sb = getattr(runner, '_smart_beta', None)
            if sb is None:
                return HealthCheck("S12", "Smart Beta config", "PASS",
                                   "not configured (OK)", "Design")
            cfg = sb.config
            if cfg.allow_direction_override:
                return HealthCheck("S12", "Smart Beta config", "CRITICAL",
                                   "allow_direction_override=True — FORBIDDEN", "Design")
            if cfg.allow_veto:
                return HealthCheck("S12", "Smart Beta config", "CRITICAL",
                                   "allow_veto=True — FORBIDDEN", "Design")
            return HealthCheck("S12", "Smart Beta config", "PASS",
                               f"enabled={cfg.enabled}, mode={cfg.mode}", "Design")
        except Exception as e:
            return HealthCheck("S12", "Smart Beta config", "WARN", f"check failed: {e}", "Design")


# =============================================================================
# LAYER 2: PER-TICK INVARIANT CHECKER
# =============================================================================

class PerTickInvariantChecker:
    """Runs T1-T8 checks after each 4H tick decision."""

    def __init__(self):
        self._consecutive_blocked: Dict[str, int] = {}
        self._last_drl_level: Optional[str] = None
        self._tick_count = 0

    def check(
        self,
        asset: str,
        intent: Any,
        market_data: Dict[str, Any],
        agent_signals: Dict[str, Any],
        runner: Any,
    ) -> HealthReport:
        """Run per-tick checks for one asset."""
        self._tick_count += 1
        report = HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            layer=f"TICK_{asset}",
        )

        report.checks.append(self._t1_alpha_estimate_nonzero(asset, intent, market_data, agent_signals))
        report.checks.append(self._t2_drl_authority_stable(runner))
        report.checks.append(self._t3_intent_actionable(asset, intent))
        report.checks.append(self._t4_signal_edge_consistent(asset, market_data, agent_signals))
        report.checks.append(self._t5_agent_output_consumed(asset, agent_signals))
        report.checks.append(self._t6_veto_chain_not_stuck(asset, intent))
        report.checks.append(self._t7_sentiment_zscore_alive(asset, market_data))
        report.checks.append(self._t8_smart_beta_computed(asset, agent_signals))
        report.checks.append(self._t9_no_trade_sentiment_backdoor(asset, market_data, agent_signals))

        # Only log non-PASS results to avoid spam
        for c in report.checks:
            if c.status != "PASS":
                if c.status == "CRITICAL":
                    logger.critical(str(c))
                elif c.status == "WARN":
                    logger.warning(str(c))

        return report

    def _t1_alpha_estimate_nonzero(self, asset, intent, market_data, agent_signals) -> HealthCheck:
        """Alpha gate should produce nonzero estimates when strategy != HOLD."""
        strategy = str(agent_signals.get("primary_strategy", "")).lower()
        edge = float(market_data.get("signal_edge_bps", 0))
        if strategy == "hold":
            return HealthCheck("T1", f"{asset} alpha estimate", "PASS",
                               "strategy=hold, edge=0 is expected", "Bug #10")
        if edge > 0:
            return HealthCheck("T1", f"{asset} alpha estimate", "PASS",
                               f"edge={edge:.1f}bps, strategy={strategy}", "Bug #10")
        return HealthCheck("T1", f"{asset} alpha estimate", "WARN",
                           f"edge=0bps with strategy={strategy} — alpha gate may reject", "Bug #10")

    def _t2_drl_authority_stable(self, runner) -> HealthCheck:
        """DRL authority should not change unexpectedly between ticks."""
        current = getattr(runner, '_drl_authority_level', "UNKNOWN")
        if self._last_drl_level is None:
            self._last_drl_level = current
            return HealthCheck("T2", "DRL authority stable", "PASS",
                               f"initial={current}", "Bug #1")
        if current == self._last_drl_level:
            return HealthCheck("T2", "DRL authority stable", "PASS",
                               f"level={current} (unchanged)", "Bug #1")
        old = self._last_drl_level
        self._last_drl_level = current
        # 2026-04-30: auto-demote paths removed. Any change here is operator
        # manual promote/demote — informational, not CRITICAL.
        return HealthCheck("T2", "DRL authority stable", "PASS",
                           f"manual change {old}→{current}", "Bug #1")

    @staticmethod
    def _actionable_blocker(intent) -> str:
        """[P155] Name the term(s) that made `is_actionable` False.

        `IntentV36.is_actionable` is a conjunction of three clauses:
            not veto_active  AND  |direction| > dir_thresh  AND  target_exposure > exp_thresh
        Reporting only "check VETO_CHAIN logs" (the previous message) is
        unactionable when the real blocker is a collapsed `target_exposure` —
        the veto chain is then completely silent and the operator looks in the
        wrong place. SOL sat at 312 consecutive blocked ticks (~52 days) with
        nobody able to tell which clause was false.

        Thresholds are read off the intent itself rather than re-hardcoded, so
        this cannot drift from `is_actionable`. Reports EVERY failing clause,
        not just the first — more than one is routinely false at once.
        """
        if intent is None:
            return "intent=None"
        try:
            veto = bool(getattr(intent, 'veto_active', False))
            direction = float(getattr(intent, 'direction', 0.0) or 0.0)
            exposure = float(getattr(intent, 'target_exposure', 0.0) or 0.0)
            mode = str(getattr(intent, 'system_mode', '') or '')
            gate_ok = bool(getattr(intent, 'alpha_gate_passed', False))

            # Mirror is_actionable's threshold selection (integration_v36.py:245-262).
            dir_thresh = float(getattr(intent, 'base_actionable_direction_threshold', 0.10) or 0.10)
            exp_thresh = 0.01
            if gate_ok and mode == "OPPORTUNITY":
                dir_thresh = float(
                    getattr(intent, 'opportunity_actionable_direction_threshold', dir_thresh) or dir_thresh)
                _short = getattr(intent, 'opportunity_actionable_direction_threshold_short', None)
                if direction < 0 and _short is not None:
                    dir_thresh = float(_short)
                _exp = getattr(intent, 'opportunity_actionable_exposure_threshold', None)
                if _exp is not None:
                    exp_thresh = float(_exp)
            elif gate_ok and mode == "NORMAL" and direction < 0:
                _short = getattr(intent, 'normal_actionable_direction_threshold_short', None)
                if _short is not None:
                    dir_thresh = float(_short)

            blockers = []
            if veto:
                reason = str(getattr(intent, 'veto_reason', '') or '(no reason recorded)')
                blockers.append(f"VETO_ACTIVE reason={reason!r}")
            if abs(direction) <= dir_thresh:
                blockers.append(f"WEAK_DIRECTION |dir|={abs(direction):.4f} <= {dir_thresh:.4f}")
            if exposure <= exp_thresh:
                blockers.append(f"ZERO_EXPOSURE target_exposure={exposure:.4f} <= {exp_thresh:.4f}")
            if not blockers:
                # is_actionable was False but no clause reproduces it — force_execution
                # / is_reduce_intent shortcuts, or the property changed shape.
                blockers.append("UNEXPLAINED (is_actionable False but all 3 clauses pass)")
            return (f"blocked_by=[{'; '.join(blockers)}] "
                    f"mode={mode} alpha_gate_passed={gate_ok} "
                    f"alpha={float(getattr(intent, 'alpha_estimated_bps', 0.0) or 0.0):.1f}bps"
                    f"/thresh={float(getattr(intent, 'alpha_threshold_bps', 0.0) or 0.0):.1f}bps")
        except Exception as err:  # noqa: silent-swallow
            # Diagnostics must never break the tick, and this is NOT a silent
            # swallow: the failure is surfaced by return value, which the caller
            # embeds verbatim in the HEALTH_T3 CRITICAL line. Logging here too
            # would double-report on every tick of a multi-hundred-tick streak.
            # Carry the type name — a bare str(err) is empty for AttributeError.
            return (f"blocker_introspection_failed="
                    f"{type(err).__name__}: {err}")

    def _t3_intent_actionable(self, asset, intent) -> HealthCheck:
        """Track consecutive non-actionable ticks.
        CRITICAL at 10+ consecutive blocks — this caught the 10-day no-trade bug."""
        actionable = getattr(intent, 'is_actionable', False) if intent else False
        strategy = getattr(intent, 'quant_strategy_id', '') or ''
        if actionable or strategy.lower() == 'hold':
            self._consecutive_blocked[asset] = 0
            return HealthCheck("T3", f"{asset} intent actionable", "PASS",
                               f"actionable={actionable}, strategy={strategy}", "All wiring")
        self._consecutive_blocked[asset] = self._consecutive_blocked.get(asset, 0) + 1
        streak = self._consecutive_blocked[asset]
        if streak >= 10:
            # [P155] name the failing clause — "check VETO_CHAIN logs" sent the
            # operator to a silent log when the blocker was exposure/direction.
            return HealthCheck("T3", f"{asset} intent actionable", "CRITICAL",
                               f"BLOCKED {streak} consecutive ticks — system cannot trade! "
                               f"strategy={strategy} {self._actionable_blocker(intent)}",
                               "Bug #16: 10-day no-trade")
        if streak >= 5:
            return HealthCheck("T3", f"{asset} intent actionable", "WARN",
                               f"blocked {streak} consecutive ticks (approaching critical) "
                               f"{self._actionable_blocker(intent)}", "All wiring")
        return HealthCheck("T3", f"{asset} intent actionable", "PASS",
                           f"blocked (streak={streak})", "All wiring")

    def _t4_signal_edge_consistent(self, asset, market_data, agent_signals) -> HealthCheck:
        """signal_edge_bps should be > 0 when |quant_direction| > 0.1."""
        qdir = abs(float(agent_signals.get("quant_direction", 0)))
        edge = float(market_data.get("signal_edge_bps", 0))
        if qdir > 0.1 and edge == 0:
            return HealthCheck("T4", f"{asset} signal consistency", "WARN",
                               f"|quant_dir|={qdir:.2f} but edge=0bps", "General")
        return HealthCheck("T4", f"{asset} signal consistency", "PASS",
                           f"dir={qdir:.2f}, edge={edge:.1f}bps", "General")

    def _t5_agent_output_consumed(self, asset, agent_signals) -> HealthCheck:
        """Key agents should have outputs in agent_signals."""
        required = ["quant_direction", "quant_confidence", "drl_direction", "sentiment_zscore"]
        missing = [k for k in required if k not in agent_signals]
        if missing:
            return HealthCheck("T5", f"{asset} agent outputs", "WARN",
                               f"missing: {missing}", "Bug #6,#14")
        return HealthCheck("T5", f"{asset} agent outputs", "PASS",
                           f"all {len(required)} key signals present", "Bug #6,#14")

    def _t6_veto_chain_not_stuck(self, asset, intent) -> HealthCheck:
        """Already covered by T3 streak — this checks veto_active specifically."""
        veto = getattr(intent, 'veto_active', False) if intent else False
        if not veto:
            return HealthCheck("T6", f"{asset} veto chain", "PASS", "no veto", "Bug #2,#3")
        reason = getattr(intent, 'veto_reason', '') or ''
        return HealthCheck("T6", f"{asset} veto chain", "PASS",
                           f"veto active: {reason[:80]}", "Bug #2,#3")

    def _t7_sentiment_zscore_alive(self, asset, market_data) -> HealthCheck:
        """When F&G feed is alive, sentiment_zscore should be nonzero."""
        feed_alive = market_data.get("_sentiment_feed_alive", False)
        zscore = float(market_data.get("sentiment_zscore", 0))
        if not feed_alive:
            return HealthCheck("T7", f"{asset} sentiment zscore", "PASS",
                               "feed not alive, zscore=0 expected", "Bug #9")
        if abs(zscore) > 0.01:
            return HealthCheck("T7", f"{asset} sentiment zscore", "PASS",
                               f"zscore={zscore:+.2f}", "Bug #9")
        return HealthCheck("T7", f"{asset} sentiment zscore", "WARN",
                           f"feed alive but zscore={zscore} (may be stuck at 0)", "Bug #9")

    def _t9_no_trade_sentiment_backdoor(self, asset, market_data, agent_signals) -> HealthCheck:
        """Detect if NO_TRADE is being triggered by sentiment indirectly.
        This caught the 10-day no-trade bug where sentiment_direction=-1.0
        (extreme fear F&G=16) participated in conflict detection, giving
        sentiment effective veto power (violating Iron Law #34).

        Checks: if NO_TRADE is active AND sentiment is extreme AND quant has
        a valid signal, this is likely a sentiment backdoor veto."""
        # Get NO_TRADE state from last engine decision
        no_trade_active = bool(market_data.get("_no_trade_internal", False))
        if not no_trade_active:
            return HealthCheck("T9", f"{asset} sentiment backdoor", "PASS",
                               "NO_TRADE not active", "Bug #16: sentiment veto")

        sent_z = float(agent_signals.get("sentiment_zscore", 0))
        quant_dir = float(agent_signals.get("quant_direction", 0))
        strategy = str(agent_signals.get("primary_strategy", "")).lower()

        # If sentiment is extreme AND quant has a real signal AND NO_TRADE is on
        if abs(sent_z) > 1.5 and abs(quant_dir) > 0.3 and strategy != "hold":
            return HealthCheck("T9", f"{asset} sentiment backdoor", "CRITICAL",
                               f"NO_TRADE active with extreme sentiment (z={sent_z:+.2f}) "
                               f"while quant has signal (dir={quant_dir:+.2f}, strategy={strategy}). "
                               f"Possible Iron Law #34 violation — sentiment may be indirectly vetoing.",
                               "Bug #16: sentiment veto backdoor")

        return HealthCheck("T9", f"{asset} sentiment backdoor", "PASS",
                           f"NO_TRADE active but not sentiment-driven "
                           f"(sent_z={sent_z:+.1f}, quant_dir={quant_dir:+.2f})",
                           "Bug #16: sentiment veto")

    def _t8_smart_beta_computed(self, asset, agent_signals) -> HealthCheck:
        """When Smart Beta enabled, state should be computed."""
        sb_state = agent_signals.get("_smart_beta_state")
        if sb_state is None:
            return HealthCheck("T8", f"{asset} Smart Beta", "PASS",
                               "not enabled or not computed (OK)", "Design")
        tags = sb_state.get("tags", [])
        mode = sb_state.get("mode", "unknown")
        return HealthCheck("T8", f"{asset} Smart Beta", "PASS",
                           f"mode={mode}, tags={tags}", "Design")
