#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - LLM Trade Explainer
================================================================================
Purpose: Generate human-readable explanations for trading decisions using LLM

P3 SOTA Requirement:
- "LLM 变成'交易解释器 + 复盘助手'"
- 输入: proof log / event log + 当前 regime + risk gate 决策
- 输出:
  1. "本次为什么做/不做" 一句话解释
  2. 触发了哪些 veto/cap
  3. 需要人确认的异常

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      LLM Trade Explainer                         │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
    │  │ Proof    │  │ Event    │  │ Regime   │  │ Risk     │        │
    │  │ Log      │  │ Log      │  │ State    │  │ Vetoes   │        │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
    │       └──────────────┴──────────────┴──────────────┘            │
    │                              │                                   │
    │                              ▼                                   │
    │                    ┌─────────────────┐                          │
    │                    │  LLM Inference  │                          │
    │                    │  (Local/API)    │                          │
    │                    └────────┬────────┘                          │
    │                             │                                    │
    │         ┌───────────────────┼───────────────────┐               │
    │         ▼                   ▼                   ▼               │
    │  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
    │  │ Decision   │     │ Veto       │     │ Anomaly    │          │
    │  │ Summary    │     │ Analysis   │     │ Alerts     │          │
    │  └────────────┘     └────────────┘     └────────────┘          │
    └─────────────────────────────────────────────────────────────────┘

Author: HMATS v5.1 P3 Fix
Version: 5.1.0
================================================================================
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# EXPLANATION TYPES
# =============================================================================

class ExplanationType(Enum):
    """Types of explanations generated."""
    DECISION_SUMMARY = "decision"    # Why trade/no-trade
    VETO_ANALYSIS = "veto"           # What triggered vetoes
    ANOMALY_ALERT = "anomaly"        # Unusual conditions
    REVIEW_REQUEST = "review"        # Needs human attention
    PERFORMANCE_NOTE = "performance" # Trade outcome explanation


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeContext:
    """Context for trade decision explanation."""
    # Tick info
    tick_id: str
    timestamp: datetime
    symbol: str
    
    # System state
    system_mode: str  # NORMAL, OPPORTUNITY, NO_TRADE
    regime: str       # trending, ranging, volatile, etc.
    regime_confidence: float
    
    # Signal info
    fused_direction: float  # -1 to +1
    fused_confidence: float
    contributing_agents: List[str]
    authority_source: str
    
    # Risk state
    risk_vetoes: List[Dict[str, Any]] = field(default_factory=list)
    gate_checks: Dict[str, bool] = field(default_factory=dict)
    exposure_state: Dict[str, float] = field(default_factory=dict)
    
    # Execution (if trade made)
    trade_made: bool = False
    trade_details: Optional[Dict] = None
    
    # Additional context
    data_quality: Dict[str, float] = field(default_factory=dict)
    anomalies_detected: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Generated explanation."""
    type: ExplanationType
    summary: str           # One-line summary
    details: str           # Full explanation
    severity: AlertSeverity = AlertSeverity.INFO
    requires_review: bool = False
    key_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "summary": self.summary,
            "details": self.details,
            "severity": self.severity.value,
            "requires_review": self.requires_review,
            "key_factors": self.key_factors,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ExplainerResult:
    """Complete explainer output for a tick."""
    tick_id: str
    timestamp: datetime
    decision_summary: Explanation
    veto_analysis: Optional[Explanation] = None
    anomaly_alerts: List[Explanation] = field(default_factory=list)
    requires_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "tick_id": self.tick_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_summary": self.decision_summary.to_dict(),
            "veto_analysis": self.veto_analysis.to_dict() if self.veto_analysis else None,
            "anomaly_alerts": [a.to_dict() for a in self.anomaly_alerts],
            "requires_human_review": self.requires_human_review,
            "review_reasons": self.review_reasons
        }


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptTemplates:
    """Templates for LLM prompts."""
    
    DECISION_SUMMARY = """You are a trading system explainer. Given the following context, provide a ONE SENTENCE explanation of why the system made (or didn't make) a trade.

Context:
- Symbol: {symbol}
- System Mode: {system_mode}
- Market Regime: {regime} (confidence: {regime_confidence:.1%})
- Signal Direction: {direction_word} ({fused_direction:+.2f}, confidence: {fused_confidence:.1%})
- Contributing Agents: {agents}
- Authority Source: {authority_source}
- Trade Made: {trade_made}
- Risk Vetoes: {vetoes}

Provide a clear, professional ONE SENTENCE explanation starting with "The system" that a human trader would understand."""

    VETO_ANALYSIS = """Analyze the following risk vetoes that blocked or modified trading:

Vetoes:
{vetoes_formatted}

Gate Checks:
{gate_checks_formatted}

Provide:
1. Which veto was most impactful
2. The specific thresholds that were breached
3. What market conditions caused these vetoes
4. Whether these vetoes were appropriate

Keep response under 100 words."""

    ANOMALY_ALERT = """Analyze these detected anomalies in trading system behavior:

Anomalies:
{anomalies}

Data Quality Issues:
{data_quality}

Determine:
1. Severity (info/warning/critical)
2. Whether human review is needed
3. Recommended action

Response format:
SEVERITY: [info/warning/critical]
REVIEW_NEEDED: [yes/no]
SUMMARY: [one sentence]
ACTION: [recommendation]"""


# =============================================================================
# RULE-BASED EXPLAINER (No LLM)
# =============================================================================

class RuleBasedExplainer:
    """
    Rule-based explanation generator (no LLM required).
    
    Provides deterministic explanations based on system state.
    Used as fallback when LLM unavailable or for fast explanations.
    """
    
    def __init__(self):
        self._direction_words = {
            1: "bullish",
            0: "neutral", 
            -1: "bearish"
        }
        
    def explain_decision(self, context: TradeContext) -> Explanation:
        """Generate decision explanation using rules."""
        
        # Determine direction word
        if context.fused_direction > 0.3:
            direction = "bullish"
        elif context.fused_direction < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Build explanation based on state
        if context.system_mode == "NO_TRADE":
            summary = f"The system entered NO_TRADE mode due to adverse market conditions in {context.regime} regime."
            key_factors = ["NO_TRADE mode active", f"Regime: {context.regime}"]
            
        elif context.risk_vetoes:
            veto_reasons = [v.get("reason", "unknown") for v in context.risk_vetoes]
            summary = f"The system blocked trading due to risk veto: {veto_reasons[0]}."
            key_factors = veto_reasons[:3]
            
        elif not context.trade_made:
            if context.fused_confidence < 0.5:
                summary = f"The system did not trade due to low signal confidence ({context.fused_confidence:.0%}) despite {direction} signals."
                key_factors = ["Low confidence", f"Direction: {direction}"]
            else:
                summary = f"The system observed {direction} signals but did not trade - waiting for better entry conditions."
                key_factors = [f"Direction: {direction}", f"Mode: {context.system_mode}"]
                
        else:
            summary = f"The system executed a {direction} trade in {context.regime} regime with {context.fused_confidence:.0%} confidence."
            key_factors = [f"Direction: {direction}", f"Confidence: {context.fused_confidence:.0%}"]
            
        return Explanation(
            type=ExplanationType.DECISION_SUMMARY,
            summary=summary,
            details=self._build_details(context),
            key_factors=key_factors
        )
        
    def _build_details(self, context: TradeContext) -> str:
        """Build detailed explanation."""
        lines = [
            f"Tick: {context.tick_id}",
            f"Mode: {context.system_mode}",
            f"Regime: {context.regime} ({context.regime_confidence:.0%})",
            f"Signal: {context.fused_direction:+.2f} (conf: {context.fused_confidence:.0%})",
            f"Authority: {context.authority_source}",
            f"Agents: {', '.join(context.contributing_agents)}",
        ]
        
        if context.risk_vetoes:
            lines.append("Vetoes:")
            for v in context.risk_vetoes[:3]:
                lines.append(f"  - {v.get('reason', 'unknown')}: {v.get('actual_value', 'N/A')} vs {v.get('threshold', 'N/A')}")
                
        return "\n".join(lines)
        
    def analyze_vetoes(self, context: TradeContext) -> Optional[Explanation]:
        """Analyze risk vetoes."""
        if not context.risk_vetoes:
            return None
            
        # Find most impactful veto
        hard_vetoes = [v for v in context.risk_vetoes if v.get("veto_type") == "HARD"]
        soft_vetoes = [v for v in context.risk_vetoes if v.get("veto_type") == "SOFT"]
        
        if hard_vetoes:
            primary = hard_vetoes[0]
            severity = AlertSeverity.WARNING
        elif soft_vetoes:
            primary = soft_vetoes[0]
            severity = AlertSeverity.INFO
        else:
            primary = context.risk_vetoes[0]
            severity = AlertSeverity.INFO
            
        summary = f"Primary veto: {primary.get('reason', 'unknown')} ({primary.get('veto_type', 'unknown')} veto)"
        
        details_lines = [
            f"Total vetoes: {len(context.risk_vetoes)}",
            f"Hard vetoes: {len(hard_vetoes)}",
            f"Soft vetoes: {len(soft_vetoes)}",
            "",
            "Veto details:"
        ]
        for v in context.risk_vetoes:
            details_lines.append(
                f"  [{v.get('veto_type', '?')}] {v.get('reason', 'unknown')}: "
                f"actual={v.get('actual_value', 'N/A')} vs threshold={v.get('threshold', 'N/A')}"
            )
            
        return Explanation(
            type=ExplanationType.VETO_ANALYSIS,
            summary=summary,
            details="\n".join(details_lines),
            severity=severity,
            key_factors=[v.get("reason", "unknown") for v in context.risk_vetoes[:3]]
        )
        
    def detect_anomalies(self, context: TradeContext) -> List[Explanation]:
        """Detect and explain anomalies."""
        anomalies = []
        
        # Check for data quality issues
        for source, quality in context.data_quality.items():
            if quality < 0.3:
                anomalies.append(Explanation(
                    type=ExplanationType.ANOMALY_ALERT,
                    summary=f"Low data quality from {source}: {quality:.0%}",
                    details=f"Data source '{source}' has quality score of {quality:.0%}, which may affect signal reliability.",
                    severity=AlertSeverity.WARNING,
                    requires_review=quality < 0.1,
                    key_factors=[source, f"quality={quality:.0%}"]
                ))
                
        # Check for explicit anomalies
        for anomaly in context.anomalies_detected:
            anomalies.append(Explanation(
                type=ExplanationType.ANOMALY_ALERT,
                summary=f"Anomaly detected: {anomaly}",
                details=f"System detected anomaly: {anomaly}. Review recommended.",
                severity=AlertSeverity.WARNING,
                requires_review=True,
                key_factors=[anomaly]
            ))
            
        # Check for regime/signal mismatch
        if context.regime == "trending" and abs(context.fused_direction) < 0.2:
            anomalies.append(Explanation(
                type=ExplanationType.ANOMALY_ALERT,
                summary="Weak signals in trending regime",
                details="Market is classified as trending but signals are weak/neutral. Possible regime misclassification.",
                severity=AlertSeverity.INFO,
                requires_review=False,
                key_factors=["regime_signal_mismatch"]
            ))
            
        return anomalies


# =============================================================================
# LLM-POWERED EXPLAINER
# =============================================================================

class LLMExplainer:
    """
    LLM-powered explanation generator.
    
    Uses local LLM (via transformers) or API for natural language explanations.
    Falls back to RuleBasedExplainer if LLM unavailable.
    """
    
    def __init__(
        self,
        use_local_llm: bool = True,
        model_name: str = "gpt2",  # Small model for demo
        api_endpoint: Optional[str] = None
    ):
        self.use_local_llm = use_local_llm
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        
        self._llm_available = False
        self._pipeline = None
        
        # Always have rule-based as fallback
        self.rule_based = RuleBasedExplainer()
        
        # Try to load local LLM
        if use_local_llm:
            self._init_local_llm()
            
    def _init_local_llm(self):
        """Initialize local LLM pipeline."""
        try:
            from transformers import pipeline
            self._pipeline = pipeline("text-generation", model=self.model_name)
            self._llm_available = True
            logger.info(f"[LLMExplainer] Loaded local model: {self.model_name}")
        except Exception as e:
            logger.warning(f"[LLMExplainer] Could not load LLM: {e}. Using rule-based fallback.")
            self._llm_available = False
            
    def explain(self, context: TradeContext) -> ExplainerResult:
        """
        Generate full explanation for a trade decision.
        
        Args:
            context: TradeContext with all decision info
            
        Returns:
            ExplainerResult with all explanations
        """
        # Decision summary (always generate)
        if self._llm_available:
            decision = self._llm_decision_summary(context)
        else:
            decision = self.rule_based.explain_decision(context)
            
        # Veto analysis (if vetoes present)
        veto_analysis = self.rule_based.analyze_vetoes(context)
        
        # Anomaly detection
        anomalies = self.rule_based.detect_anomalies(context)
        
        # Determine if human review needed
        requires_review = False
        review_reasons = []
        
        if any(a.requires_review for a in anomalies):
            requires_review = True
            review_reasons.extend([a.summary for a in anomalies if a.requires_review])
            
        if veto_analysis and veto_analysis.severity == AlertSeverity.CRITICAL:
            requires_review = True
            review_reasons.append("Critical veto triggered")
            
        return ExplainerResult(
            tick_id=context.tick_id,
            timestamp=context.timestamp,
            decision_summary=decision,
            veto_analysis=veto_analysis,
            anomaly_alerts=anomalies,
            requires_human_review=requires_review,
            review_reasons=review_reasons
        )
        
    def _llm_decision_summary(self, context: TradeContext) -> Explanation:
        """Generate decision summary using LLM."""
        # Build prompt
        direction_word = "bullish" if context.fused_direction > 0.3 else "bearish" if context.fused_direction < -0.3 else "neutral"
        vetoes = ", ".join([v.get("reason", "unknown") for v in context.risk_vetoes]) or "none"
        
        prompt = PromptTemplates.DECISION_SUMMARY.format(
            symbol=context.symbol,
            system_mode=context.system_mode,
            regime=context.regime,
            regime_confidence=context.regime_confidence,
            direction_word=direction_word,
            fused_direction=context.fused_direction,
            fused_confidence=context.fused_confidence,
            agents=", ".join(context.contributing_agents),
            authority_source=context.authority_source,
            trade_made="Yes" if context.trade_made else "No",
            vetoes=vetoes
        )
        
        try:
            # Generate with LLM
            result = self._pipeline(prompt, max_length=200, num_return_sequences=1)
            generated = result[0]["generated_text"]
            
            # Extract just the response (after prompt)
            response = generated[len(prompt):].strip()
            
            # Find first sentence
            if "." in response:
                summary = response[:response.index(".") + 1]
            else:
                summary = response[:100]
                
            return Explanation(
                type=ExplanationType.DECISION_SUMMARY,
                summary=summary,
                details=self.rule_based._build_details(context),
                key_factors=[direction_word, context.system_mode, context.regime]
            )
            
        except Exception as e:
            logger.warning(f"[LLMExplainer] LLM generation failed: {e}. Using rule-based.")
            return self.rule_based.explain_decision(context)


# =============================================================================
# EXPLANATION LOGGER
# =============================================================================

class ExplanationLogger:
    """
    Log and persist explanations for review.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/explanations")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._daily_file = None
        self._current_date = None
        
    def log(self, result: ExplainerResult):
        """Log explanation result."""
        # Update daily file if needed
        today = datetime.now(timezone.utc).date()
        if self._current_date != today:
            if self._daily_file:
                self._daily_file.close()
            log_path = self.log_dir / f"explanations_{today.isoformat()}.jsonl"
            self._daily_file = open(log_path, 'a', encoding="utf-8")
            self._current_date = today
            
        # Write as JSON line
        self._daily_file.write(json.dumps(result.to_dict()) + "\n")
        self._daily_file.flush()
        
        # Also log to standard logger
        if result.requires_human_review:
            logger.warning(f"[REVIEW NEEDED] {result.tick_id}: {', '.join(result.review_reasons)}")
        else:
            logger.info(f"[Explanation] {result.tick_id}: {result.decision_summary.summary}")

    def close(self):
        """Close active daily log file handle."""
        if self._daily_file:
            try:
                self._daily_file.close()
            finally:
                self._daily_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_explainer: Optional[LLMExplainer] = None
_explanation_logger: Optional[ExplanationLogger] = None


def get_trade_explainer(use_llm: bool = False) -> LLMExplainer:
    """Get singleton LLMExplainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = LLMExplainer(use_local_llm=use_llm)
    return _explainer


def get_explanation_logger(log_dir: Optional[Path] = None) -> ExplanationLogger:
    """Get singleton ExplanationLogger instance."""
    global _explanation_logger
    if _explanation_logger is None:
        _explanation_logger = ExplanationLogger(log_dir)
    return _explanation_logger


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LLMExplainer',
    'RuleBasedExplainer',
    'TradeContext',
    'Explanation',
    'ExplainerResult',
    'ExplanationType',
    'AlertSeverity',
    'ExplanationLogger',
    'get_trade_explainer',
    'get_explanation_logger',
]
