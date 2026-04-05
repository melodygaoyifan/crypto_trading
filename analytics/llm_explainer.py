"""
================================================================================
HMATS v5.1.0 - LLM Decision Explainer
================================================================================
Purpose: Natural language explanations for trading decisions

SOTA Requirement:
- Add an LLM-based decision explainer:
  * Input: current proof log + trade action
  * Output: natural language summary of why trade was made or blocked
- Optional: integrate an interactive LLM chat to answer user questions

Features:
1. Trade decision explanations
2. Rejection/veto reasoning
3. Interactive Q&A about system state
4. Historical decision queries
5. Multi-level detail (summary, detailed, technical)

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import logging
import json
import os
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum
import re

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ExplanationLevel(Enum):
    """Detail level for explanations."""
    BRIEF = "brief"           # One sentence
    SUMMARY = "summary"       # 2-3 sentences
    DETAILED = "detailed"     # Full paragraph
    TECHNICAL = "technical"   # With all data


@dataclass
class ExplainerConfig:
    """Configuration for decision explainer."""
    # LLM settings
    model_name: str = "local"  # "local", "openai", "anthropic"
    api_key_env: str = "LLM_API_KEY"
    max_tokens: int = 500
    temperature: float = 0.3
    
    # History
    max_history: int = 100
    
    # Templates
    use_templates: bool = True  # Use template-based explanations if LLM unavailable


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ProofLog:
    """Structured proof log from trading decision."""
    timestamp: datetime
    version: str
    
    # Data validity
    data_valid: bool
    
    # Mode
    mode: str  # "NO_TRADE", "OPPORTUNITY", "NORMAL"
    
    # Agents
    quant_signal: str
    drl_status: str
    
    # Risk
    alpha_gate: Dict
    veto: Dict
    
    # Execution
    intent: Dict
    execution_mode: str
    
    # SOL specific
    sol_dominance: Dict
    
    # Raw data
    raw: Dict = field(default_factory=dict)
    
    @classmethod
    def from_log_line(cls, log_line: str) -> Optional['ProofLog']:
        """Parse a proof log line."""
        try:
            # Pattern matching for proof log format
            # [PROOF] 2026-01-26T02:40:27.986455 | v3.6.1 | DataValid=True | ...
            
            parts = log_line.split("|")
            if len(parts) < 5:
                return None
            
            # Extract timestamp
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}T[\d:.]+)', parts[0])
            timestamp = datetime.fromisoformat(ts_match.group(1)) if ts_match else datetime.now(timezone.utc)
            
            # Extract version
            version = parts[1].strip() if len(parts) > 1 else "unknown"
            
            # Parse key-value pairs
            data = {}
            for part in parts[2:]:
                if "=" in part:
                    key, value = part.strip().split("=", 1)
                    data[key.strip()] = value.strip()
            
            return cls(
                timestamp=timestamp,
                version=version,
                data_valid=data.get("DataValid", "False") == "True",
                mode=data.get("Mode", "UNKNOWN"),
                quant_signal=data.get("Quant", ""),
                drl_status=data.get("DRL", ""),
                alpha_gate=cls._parse_dict(data.get("AlphaGate", "{}")),
                veto=cls._parse_dict(data.get("Veto", "{}")),
                intent=cls._parse_dict(data.get("Intent", "{}")),
                execution_mode=data.get("Exec", ""),
                sol_dominance=cls._parse_dict(data.get("SOL_dom", "{}")),
                raw=data
            )
        except Exception as e:
            logger.debug(f"Failed to parse proof log: {e}")
            return None
    
    @staticmethod
    def _parse_dict(s: str) -> Dict:
        """Parse a dict-like string."""
        try:
            # Handle {key=value,key2=value2} format
            s = s.strip("{}")
            if not s:
                return {}
            result = {}
            for pair in s.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    result[k.strip()] = v.strip()
            return result
        except Exception:
            return {}


@dataclass
class TradeAction:
    """A trade action to explain."""
    action_type: str  # "entry", "exit", "blocked", "veto"
    asset: str
    direction: str  # "long", "short", "flat"
    size: float
    price: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class Explanation:
    """Generated explanation."""
    timestamp: datetime
    level: ExplanationLevel
    summary: str
    details: str
    factors: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "summary": self.summary,
            "details": self.details,
            "factors": self.factors,
            "confidence": self.confidence
        }


# =============================================================================
# TEMPLATE-BASED EXPLAINER
# =============================================================================

class TemplateExplainer:
    """
    Generates explanations using templates when LLM is unavailable.
    
    Templates cover common scenarios with placeholder substitution.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates."""
        return {
            # Trade entry templates
            "entry_opportunity": (
                "Entered {direction} position on {asset} at ${price:.2f}. "
                "Market regime is in OPPORTUNITY mode with {quant_signal}. "
                "Position size: {size:.4f} ({exposure:.1%} exposure)."
            ),
            
            "entry_normal": (
                "Opened {direction} position on {asset} at ${price:.2f}. "
                "Signal confidence: {confidence:.0%}. "
                "Risk-adjusted size: {size:.4f}."
            ),
            
            # Trade exit templates
            "exit_stop": (
                "Exited {asset} position due to stop-loss trigger. "
                "Price moved against position by {loss_pct:.1%}."
            ),
            
            "exit_target": (
                "Closed {asset} position at profit target. "
                "Realized gain: ${pnl:.2f} ({pnl_pct:.1%})."
            ),
            
            "exit_signal": (
                "Closed {asset} position on signal reversal. "
                "Original direction: {original_dir}, new signal: {new_signal}."
            ),
            
            # Blocked/Veto templates
            "blocked_no_trade": (
                "Trade blocked: System in NO_TRADE mode. "
                "Reason: {reason}. "
                "Will resume when conditions improve."
            ),
            
            "blocked_veto": (
                "Trade vetoed by risk management. "
                "Veto type: {veto_type}. "
                "Current drawdown: {drawdown:.1%}."
            ),
            
            "blocked_alpha": (
                "Trade blocked by alpha gate. "
                "Estimated alpha ({alpha_est:.2f}) below threshold ({alpha_thresh:.2f})."
            ),
            
            "blocked_correlation": (
                "Position size reduced due to correlation risk. "
                "{asset} is {corr:.0%} correlated with existing {corr_asset} position."
            ),
            
            # Risk templates
            "risk_drawdown": (
                "Risk alert: Drawdown has reached {drawdown:.1%}. "
                "Position sizing reduced by {reduction:.0%}. "
                "Mode: {risk_mode}."
            ),
            
            "risk_correlation": (
                "Correlation warning: Portfolio assets showing {corr:.0%} correlation. "
                "Exposure limits tightened."
            ),
            
            # Regime templates
            "regime_change": (
                "Market regime changed from {old_regime} to {new_regime}. "
                "Strategy parameters adjusted. "
                "Agent weights rebalanced."
            )
        }
    
    def generate(
        self,
        template_key: str,
        variables: Dict,
        level: ExplanationLevel = ExplanationLevel.SUMMARY
    ) -> Explanation:
        """Generate explanation from template."""
        template = self.templates.get(template_key, "Action processed: {action_type}")
        
        try:
            summary = template.format(**variables)
        except KeyError as e:
            summary = f"Trade action: {variables.get('action_type', 'unknown')} (missing: {e})"
        
        # Build factors list
        factors = []
        if "mode" in variables:
            factors.append(f"Mode: {variables['mode']}")
        if "quant_signal" in variables:
            factors.append(f"Quant: {variables['quant_signal']}")
        if "drl_status" in variables:
            factors.append(f"DRL: {variables['drl_status']}")
        if "veto_type" in variables:
            factors.append(f"Veto: {variables['veto_type']}")
        
        return Explanation(
            timestamp=datetime.now(timezone.utc),
            level=level,
            summary=summary,
            details=summary,  # Same for template-based
            factors=factors,
            confidence=0.8  # Templates have fixed confidence
        )


# =============================================================================
# LLM DECISION EXPLAINER
# =============================================================================

class LLMDecisionExplainer:
    """
    Generates natural language explanations for trading decisions.
    
    Can use:
    1. Local LLM (vLLM wrapper)
    2. OpenAI API
    3. Anthropic API
    4. Template fallback
    """
    
    def __init__(self, config: ExplainerConfig = None):
        self.config = config or ExplainerConfig()
        self._resolve_provider_from_env()
        
        # Template fallback
        self.template_explainer = TemplateExplainer()
        
        # LLM client (initialized on first use)
        self._llm_client = None
        self._http_client: Optional[httpx.Client] = None
        self._llm_available = False
        self._provider_hard_disable_until: Optional[datetime] = None
        self._provider_hard_disable_reason: str = ""
        
        # History
        self.explanation_history: deque = deque(maxlen=self.config.max_history)
        self.proof_log_history: deque = deque(maxlen=self.config.max_history)
        
        # Check LLM availability
        self._check_llm_availability()
        
        logger.info(f"[LLMExplainer] Initialized (LLM available: {self._llm_available})")

    def _resolve_provider_from_env(self):
        """Allow runtime provider override without code changes."""
        env_provider = (
            os.environ.get("LLM_EXPLAINER_PROVIDER")
            or os.environ.get("LLM_EXPLAINER_MODEL")
            or os.environ.get("LLM_PROVIDER")
        )
        if not env_provider:
            return
        provider = str(env_provider).strip().lower()
        if provider in ("openai", "anthropic", "local"):
            self.config.model_name = provider
            logger.info(f"[LLMExplainer] Provider overridden by env: {provider}")
        else:
            logger.warning(
                f"[LLMExplainer] Ignoring invalid provider override: {env_provider}"
            )

    def _provider_api_key(self, provider: str) -> str:
        """Resolve API key with provider-specific fallback env vars."""
        provider = (provider or "").lower()
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
                self.config.api_key_env, ""
            )
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get(
                self.config.api_key_env, ""
            )
        return os.environ.get(self.config.api_key_env, "")

    def _is_provider_disabled(self) -> bool:
        """Whether external LLM provider is temporarily disabled."""
        if self._provider_hard_disable_until is None:
            return False
        now = datetime.now(timezone.utc)
        if now >= self._provider_hard_disable_until:
            logger.info("[LLMExplainer] Provider hard-disable expired")
            self._provider_hard_disable_until = None
            self._provider_hard_disable_reason = ""
            return False
        return True

    def _hard_disable_provider(self, reason: str, hours: int = 6):
        """Disable provider after non-retryable API failures."""
        self._provider_hard_disable_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self._provider_hard_disable_reason = reason
        logger.warning(
            f"[LLMExplainer] Provider hard-disabled for {hours}h "
            f"(reason={reason}, until={self._provider_hard_disable_until.isoformat()})"
        )
    
    def _check_llm_availability(self):
        """Check if LLM is available."""
        if self.config.model_name == "local":
            # Check for local vLLM
            try:
                from engine.compute.vllm_inference_wrapper import VLLMInferenceWrapper
                self._llm_available = True
            except ImportError:
                self._llm_available = False
        elif self.config.model_name in ["openai", "anthropic"]:
            api_key = self._provider_api_key(self.config.model_name)
            self._llm_available = bool(api_key)

    def _get_http_client(self) -> httpx.Client:
        """Get/reuse shared HTTP client for external LLM calls."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def close(self):
        """Close shared HTTP resources."""
        if self._http_client is not None:
            try:
                self._http_client.close()
            finally:
                self._http_client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    # =========================================================================
    # PROOF LOG HANDLING
    # =========================================================================
    
    def add_proof_log(self, log_line: str):
        """Add a proof log for context."""
        proof_log = ProofLog.from_log_line(log_line)
        if proof_log:
            self.proof_log_history.append(proof_log)
    
    def get_recent_context(self, count: int = 5) -> List[ProofLog]:
        """Get recent proof logs for context."""
        return list(self.proof_log_history)[-count:]
    
    # =========================================================================
    # EXPLANATION GENERATION
    # =========================================================================
    
    def explain_trade(
        self,
        action: TradeAction,
        proof_log: ProofLog = None,
        level: ExplanationLevel = ExplanationLevel.SUMMARY
    ) -> Explanation:
        """
        Generate explanation for a trade action.
        
        Args:
            action: The trade action to explain
            proof_log: Associated proof log (optional)
            level: Detail level
            
        Returns:
            Explanation object
        """
        # Build context
        context = self._build_context(action, proof_log)
        
        # Try LLM first
        if self._llm_available and not self.config.use_templates:
            explanation = self._explain_with_llm(context, level)
            if explanation:
                self.explanation_history.append(explanation)
                return explanation
        
        # Fall back to templates
        explanation = self._explain_with_template(action, proof_log, level)
        self.explanation_history.append(explanation)
        return explanation
    
    def _build_context(self, action: TradeAction, proof_log: ProofLog = None) -> Dict:
        """Build context dict for explanation."""
        context = {
            "action_type": action.action_type,
            "asset": action.asset,
            "direction": action.direction,
            "size": action.size,
            "price": action.price,
            "timestamp": action.timestamp.isoformat()
        }
        
        if proof_log:
            context.update({
                "mode": proof_log.mode,
                "data_valid": proof_log.data_valid,
                "quant_signal": proof_log.quant_signal,
                "drl_status": proof_log.drl_status,
                "alpha_gate": proof_log.alpha_gate,
                "veto": proof_log.veto,
                "intent": proof_log.intent,
                "execution_mode": proof_log.execution_mode
            })
        
        return context
    
    def _explain_with_template(
        self,
        action: TradeAction,
        proof_log: ProofLog,
        level: ExplanationLevel
    ) -> Explanation:
        """Generate template-based explanation."""
        variables = {
            "asset": action.asset,
            "direction": action.direction,
            "size": action.size,
            "price": action.price,
            "action_type": action.action_type,
            "exposure": action.size * action.price / 100000,  # Approximate
            "confidence": 0.7
        }
        
        if proof_log:
            variables.update({
                "mode": proof_log.mode,
                "quant_signal": proof_log.quant_signal,
                "drl_status": proof_log.drl_status,
                "veto_type": proof_log.veto.get("type", "none"),
                "drawdown": float(proof_log.raw.get("Drawdown", "0").rstrip("%")) / 100,
                "alpha_est": float(proof_log.alpha_gate.get("est", "0")),
                "alpha_thresh": float(proof_log.alpha_gate.get("thresh", "0"))
            })
        
        # Select template based on action
        if action.action_type == "entry":
            if proof_log and proof_log.mode == "OPPORTUNITY":
                template_key = "entry_opportunity"
            else:
                template_key = "entry_normal"
        elif action.action_type == "exit":
            template_key = "exit_signal"
        elif action.action_type == "blocked":
            if proof_log and proof_log.mode == "NO_TRADE":
                template_key = "blocked_no_trade"
                variables["reason"] = proof_log.raw.get("NO_TRADE_reason", "Market conditions")
            elif proof_log and proof_log.alpha_gate.get("pass") == "False":
                template_key = "blocked_alpha"
            else:
                template_key = "blocked_veto"
        else:
            template_key = "entry_normal"
        
        return self.template_explainer.generate(template_key, variables, level)
    
    def _explain_with_llm(self, context: Dict, level: ExplanationLevel) -> Optional[Explanation]:
        """Generate LLM-based explanation."""
        # Build prompt
        prompt = self._build_prompt(context, level)
        
        try:
            # Call LLM (implementation depends on provider)
            response = self._call_llm(prompt)
            
            if response:
                return Explanation(
                    timestamp=datetime.now(timezone.utc),
                    level=level,
                    summary=response.get("summary", ""),
                    details=response.get("details", ""),
                    factors=response.get("factors", []),
                    confidence=0.9
                )
        except Exception as e:
            logger.error(f"[LLMExplainer] LLM call failed: {e}")
        
        return None
    
    def _build_prompt(self, context: Dict, level: ExplanationLevel) -> str:
        """Build prompt for LLM."""
        detail_instructions = {
            ExplanationLevel.BRIEF: "Respond in one sentence.",
            ExplanationLevel.SUMMARY: "Respond in 2-3 sentences.",
            ExplanationLevel.DETAILED: "Provide a detailed paragraph explanation.",
            ExplanationLevel.TECHNICAL: "Include all technical details and data points."
        }
        
        return f"""You are explaining a trading system decision to the user.

Context:
{json.dumps(context, indent=2)}

{detail_instructions[level]}

Explain why this trade action was taken or blocked, based on the proof log data.
Focus on the key factors that led to this decision.

Response format:
- summary: Main explanation
- factors: List of key decision factors
"""
    
    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """
        Call the LLM based on configured provider.
        
        Supports:
        - local: Returns None (use templates)
        - openai: Uses OpenAI API (requires OPENAI_API_KEY)
        - anthropic: Uses Anthropic API (requires ANTHROPIC_API_KEY)
        """
        if self.config.model_name == "local":
            return None

        provider_disabled = self._is_provider_disabled()
        if provider_disabled:
            # Hard-disable means current provider is unhealthy; still allow
            # cross-provider fallback below when keys are available.
            logger.info("[LLMExplainer] Primary provider disabled; trying fallback")

        primary = (self.config.model_name or "local").lower()
        fallback = "openai" if primary == "anthropic" else None
        provider_chain = [primary] + ([fallback] if fallback else [])

        for provider in provider_chain:
            if provider not in ("openai", "anthropic"):
                logger.warning(f"Unknown LLM provider: {provider}")
                continue
            if provider_disabled and provider == primary:
                continue

            api_key = self._provider_api_key(provider)
            if not api_key:
                logger.debug(f"No API key for {provider}, skipping")
                continue

            try:
                if provider == "openai":
                    result = self._call_openai(prompt, api_key)
                else:
                    result = self._call_anthropic(prompt, api_key)

                if result:
                    if provider != primary:
                        logger.warning(
                            f"[LLMExplainer] Fallback provider in use: {primary} -> {provider}"
                        )
                    return result
            except Exception as e:
                logger.error(f"LLM API call failed ({provider}): {e}")

        logger.debug(f"No successful LLM provider for mode={self.config.model_name}, using templates")
        return None
    
    def _call_openai(self, prompt: str, api_key: str) -> Optional[Dict]:
        """Call OpenAI API."""
        try:
            client = self._get_http_client()
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a trading system explainer. Provide clear, concise explanations for trading decisions."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Try to parse as JSON, otherwise return as summary
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"summary": content, "factors": []}
            else:
                logger.warning(f"OpenAI API returned {response.status_code}: {response.text}")
                    
        except httpx.TimeoutException:
            logger.warning("OpenAI API timeout")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
        
        return None
    
    def _call_anthropic(self, prompt: str, api_key: str) -> Optional[Dict]:
        """Call Anthropic API."""
        try:
            client = self._get_http_client()
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": self.config.max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "system": "You are a trading system explainer. Provide clear, concise explanations for trading decisions."
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"]
                
                # Try to parse as JSON, otherwise return as summary
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"summary": content, "factors": []}
            else:
                if response.status_code in (401, 403, 404, 422):
                    self._hard_disable_provider(f"anthropic_http_{response.status_code}")
                logger.warning(f"Anthropic API returned {response.status_code}: {response.text}")
                    
        except httpx.TimeoutException:
            logger.warning("Anthropic API timeout")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
        
        return None
    
    # =========================================================================
    # INTERACTIVE Q&A
    # =========================================================================
    
    def answer_question(self, question: str) -> str:
        """
        Answer a user question about trading decisions.
        
        Args:
            question: User's question (e.g., "why did we exit ETH?")
            
        Returns:
            Answer string
        """
        # Parse question for context
        asset_match = re.search(r'\b(BTC|ETH|SOL)\b', question.upper())
        asset = asset_match.group(1) if asset_match else None
        
        # Look for relevant history
        relevant_logs = []
        for log in self.proof_log_history:
            if asset:
                intent_asset = log.intent.get("asset", "")
                if asset.lower() in intent_asset.lower():
                    relevant_logs.append(log)
            else:
                relevant_logs.append(log)
        
        if not relevant_logs:
            return f"I don't have recent trading history to answer that question."
        
        # Get most recent relevant log
        latest = relevant_logs[-1]
        
        # Generate contextual answer
        if "why" in question.lower():
            if "exit" in question.lower() or "sell" in question.lower():
                return self._explain_exit(asset, latest)
            elif "entry" in question.lower() or "buy" in question.lower():
                return self._explain_entry(asset, latest)
            elif "blocked" in question.lower() or "didn't" in question.lower():
                return self._explain_blocked(asset, latest)
        
        if "which agent" in question.lower():
            return self._explain_agent_contribution(latest)
        
        # Default: summarize current state
        return self._summarize_state(latest)
    
    def _explain_exit(self, asset: str, log: ProofLog) -> str:
        """Explain an exit decision."""
        asset = asset or "the position"
        
        reasons = []
        if log.mode == "NO_TRADE":
            reasons.append("The system entered NO_TRADE mode")
        if "exit" in log.sol_dominance.get("urgency", "").lower():
            reasons.append("SOL dominance triggered a forced exit")
        if log.veto.get("type") != "NONE":
            reasons.append(f"Risk veto was triggered ({log.veto.get('type')})")
        
        if reasons:
            return f"{asset} was exited because: {'; '.join(reasons)}."
        return f"The exit for {asset} was triggered by signal reversal in {log.mode} mode."
    
    def _explain_entry(self, asset: str, log: ProofLog) -> str:
        """Explain an entry decision."""
        asset = asset or "the position"
        
        factors = []
        if log.mode == "OPPORTUNITY":
            factors.append("strong opportunity signals detected")
        if log.quant_signal:
            factors.append(f"quant signal: {log.quant_signal}")
        if log.alpha_gate.get("pass") == "True":
            factors.append(f"alpha gate passed (est={log.alpha_gate.get('est')})")
        
        return f"Entered {asset} because: {'; '.join(factors) if factors else 'signals aligned'}."
    
    def _explain_blocked(self, asset: str, log: ProofLog) -> str:
        """Explain why a trade was blocked."""
        asset = asset or "The trade"
        
        if log.mode == "NO_TRADE":
            return f"{asset} was blocked because the system is in NO_TRADE mode due to unfavorable market conditions."
        if log.alpha_gate.get("pass") == "False":
            return f"{asset} was blocked by the alpha gate. Estimated alpha ({log.alpha_gate.get('est')}) was below the threshold ({log.alpha_gate.get('thresh')})."
        if log.veto.get("type") != "NONE":
            return f"{asset} was vetoed by risk management. Veto type: {log.veto.get('type')}."
        
        return f"{asset} was not executed due to insufficient signal strength."
    
    def _explain_agent_contribution(self, log: ProofLog) -> str:
        """Explain which agent drove the decision."""
        agents = []
        
        if log.quant_signal and "REAL" in log.quant_signal:
            agents.append(f"Quant Agent ({log.quant_signal})")
        if log.drl_status and "DISABLED" not in log.drl_status:
            agents.append(f"DRL Agent ({log.drl_status})")
        
        if not agents:
            return "The decision was primarily driven by the risk management system and market conditions."
        
        return f"The decision was driven by: {', '.join(agents)}."
    
    def _summarize_state(self, log: ProofLog) -> str:
        """Summarize current trading state."""
        return (
            f"Current state: Mode={log.mode}, "
            f"Quant={log.quant_signal}, "
            f"DRL={log.drl_status}, "
            f"Execution={log.execution_mode}."
        )
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_recent_explanations(self, count: int = 10) -> List[Dict]:
        """Get recent explanations."""
        return [e.to_dict() for e in list(self.explanation_history)[-count:]]


# =============================================================================
# SINGLETON
# =============================================================================

_explainer: Optional[LLMDecisionExplainer] = None


def get_decision_explainer(config: ExplainerConfig = None) -> LLMDecisionExplainer:
    """Get or create the global decision explainer."""
    global _explainer
    if _explainer is None:
        _explainer = LLMDecisionExplainer(config)
    return _explainer


def reset_decision_explainer():
    """Reset the global explainer."""
    global _explainer
    _explainer = None
