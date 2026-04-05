"""
HMATS Agents Module - 多智能体协作系统
======================================

SOTA对照: "多智能体协作机制，各代理负责不同层面决策"

核心代理:
- QuantAgent: 量化信号生成 (机构级策略 v3.3)
- RiskAgent: 风险管理 (opportunity-aware)
- DRLAgent: 深度强化学习代理 (仓位管理)
- MasterPipeline: 主决策管道
"""

# Import what exists - use try/except for robustness
try:
    from .drl_agent import DRLAgent, DRLAction, DRLState, DRLOutput
except ImportError as e:
    print(f"[WARN]️ drl_agent import warning: {e}")
    DRLAgent = None

try:
    from .risk_agent import RiskAgentV34, RiskConfig, RiskAssessment
    RiskAgent = RiskAgentV34  # Alias for compatibility
except ImportError as e:
    print(f"[WARN]️ risk_agent import warning: {e}")
    RiskAgent = None
    RiskAgentV34 = None

# [UPG7] DEPRECATED: quant_agent.py is no longer used by main.py.
# Signals now come from main.py _build_agent_signals() + kraken_quant_agent.py.
# Import retained for: core/runtime_spine.py (line 494) + tests.
try:
    from .quant_agent import QuantAgent
except ImportError as e:
    print(f"[WARN]️ quant_agent import warning: {e}")
    QuantAgent = None

# SOTA v5.2.0 Agents
try:
    from .sentiment_llm_agent import (
        SentimentLLMAgent, 
        SentimentAgentConfig,
        SentimentLevel,
        SentimentSignal,
        get_sentiment_agent,
        reset_sentiment_agent
    )
except ImportError as e:
    print(f"[WARN]️ sentiment_llm_agent import warning: {e}")
    SentimentLLMAgent = None
    get_sentiment_agent = None

try:
    from .microstructure_agent import (
        MicrostructureArbitrageAgent,
        MicrostructureConfig,
        MicroSignalType,
        MicrostructureSignal,
        get_microstructure_agent,
        reset_microstructure_agent
    )
except ImportError as e:
    print(f"[WARN]️ microstructure_agent import warning: {e}")
    MicrostructureArbitrageAgent = None
    get_microstructure_agent = None

try:
    from .onchain_solana_agent import (
        SolanaOnChainAgent,
        OnChainSolanaAgent,  # backward compat alias
        OnChainSignalType,
        OnChainMetrics,
        OnChainAlphaSignal,
        get_onchain_agent,
        reset_onchain_agent,
    )
except ImportError as e:
    print(f"[WARNING] onchain_solana_agent import warning: {e}")
    SolanaOnChainAgent = None
    OnChainSolanaAgent = None
    get_onchain_agent = None
    reset_onchain_agent = None

# V6.2.3: Short-biased Agent (做空专用智能体)
try:
    from .short_bias_agent import (
        ShortBiasAgent,
        ShortBiasConfig,
        ShortBiasSignal,
        ShortSignalType,
        get_short_bias_agent,
        reset_short_bias_agent
    )
except ImportError as e:
    print(f"[WARN]️ short_bias_agent import warning: {e}")
    ShortBiasAgent = None
    get_short_bias_agent = None

__all__ = [
    # Core agents
    "QuantAgent",
    "RiskAgent",
    "RiskAgentV34",
    "DRLAgent",
    "DRLAction",
    "DRLState",
    
    # SOTA v5.2.0 agents
    "SentimentLLMAgent",
    "SentimentAgentConfig", 
    "SentimentLevel",
    "SentimentSignal",
    "get_sentiment_agent",
    "reset_sentiment_agent",
    
    "MicrostructureArbitrageAgent",
    "MicrostructureConfig",
    "MicroSignalType",
    "MicrostructureSignal",
    "get_microstructure_agent",
    "reset_microstructure_agent",
    
    "SolanaOnChainAgent",
    "OnChainSolanaAgent",
    "OnChainSignalType",
    "OnChainMetrics",
    "OnChainAlphaSignal",
    "get_onchain_agent",
    "reset_onchain_agent",
    
    # V6.2.3: Short-biased Agent
    "ShortBiasAgent",
    "ShortBiasConfig",
    "ShortBiasSignal",
    "ShortSignalType",
    "get_short_bias_agent",
    "reset_short_bias_agent",
]

# V6.2.3 Backfill: Onchain Graph Alpha Agent
try:
    from .onchain_graph_alpha import (
        OnChainGraphAlphaEngine,
        WalletProfile,
        WhaleAction,
        OnChainSignalType as GraphSignalType,
        AddressType,
    )
    __all__.extend([
        "OnChainGraphAlphaEngine",
        "WalletProfile",
        "WhaleAction",
        "GraphSignalType",
        "AddressType",
    ])
except ImportError as e:
    print(f"[WARN]️ onchain_graph_alpha import warning: {e}")
    OnChainGraphAlphaEngine = None

# Attribution infrastructure (P0)
try:
    from .signal_envelope import AgentSignalEnvelope, wrap_agent_signal
    from .attribution_tracker import (
        AttributionTracker,
        AgentScorecard,
        get_attribution_tracker,
        reset_attribution_tracker,
    )
    __all__.extend([
        "AgentSignalEnvelope",
        "wrap_agent_signal",
        "AttributionTracker",
        "AgentScorecard",
        "get_attribution_tracker",
        "reset_attribution_tracker",
    ])
except ImportError as e:
    print(f"[WARN]️ attribution import warning: {e}")
    AgentSignalEnvelope = None
    wrap_agent_signal = None
    get_attribution_tracker = None
