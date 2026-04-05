"""
vLLM Inference Wrapper - Flash Sentiment Auditor
=================================================
Production-ready wrapper for Llama-3-8B sentiment analysis using vLLM.

P1 FIX: Added explicit sentiment configuration with mock mode detection.
- When in mock mode, sentiment-based OPPORTUNITY triggers are disabled
- Explicit ENABLE_SENTIMENT flag for production control

Features:
- High-throughput vLLM serving on RTX 5090
- Flash Attention 3 optimization
- Batch sentiment inference
- Consensus verification (divergence > 3σ = "Artificial Noise")
- Real-time SharedStateBuffer integration

Supported Models:
- meta-llama/Meta-Llama-3-8B-Instruct (primary)
- mistralai/Mistral-7B-Instruct-v0.2 (fallback)
- TheBloke/Llama-3-8B-Instruct-GPTQ (quantized)

Hardware Target: RTX 5090 (32GB VRAM)
Author: Senior Performance Engineer
Version: 3.2
"""

import os
import sys
import time
import asyncio
import logging
import json
import re
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Optional vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Optional transformers for fallback
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


# =============================================================================
# P1: SENTIMENT CONFIGURATION
# =============================================================================

class SentimentMode(Enum):
    """
    Unified sentiment mode enum.
    
    P3 FIX: Single source of truth for sentiment mode tracking.
    """
    REAL = "REAL"           # Full sentiment model available
    MOCK = "MOCK"           # Mock sentiment (triggers disabled)
    DISABLED = "DISABLED"   # Sentiment completely disabled


@dataclass
class SentimentConfig:
    """
    Sentiment configuration with explicit mock mode control.
    
    P1 FIX: Mock sentiment cannot trigger OPPORTUNITY mode.
    """
    # Master enable flag
    ENABLE_SENTIMENT: bool = True
    
    # Auto-detect mock mode
    IS_MOCK: bool = not (VLLM_AVAILABLE or TRANSFORMERS_AVAILABLE)
    
    # When mock, disable these features
    DISABLE_OPPORTUNITY_TRIGGER_IN_MOCK: bool = True  # CRITICAL: mock cannot trigger OPPORTUNITY
    
    def can_trigger_opportunity(self) -> bool:
        """Check if sentiment can trigger OPPORTUNITY mode."""
        if not self.ENABLE_SENTIMENT:
            return False
        if self.IS_MOCK and self.DISABLE_OPPORTUNITY_TRIGGER_IN_MOCK:
            return False
        return True
    
    def get_status_string(self) -> str:
        """Get human-readable status."""
        if not self.ENABLE_SENTIMENT:
            return "DISABLED"
        if self.IS_MOCK:
            return "MOCK (triggers disabled)"
        return "REAL"


_sentiment_config: Optional[SentimentConfig] = None


def get_sentiment_config() -> SentimentConfig:
    """Get or create sentiment config with explicit startup logging."""
    global _sentiment_config
    if _sentiment_config is None:
        _sentiment_config = SentimentConfig()
        
        # P3 FIX: Clear startup banner for sentiment mode
        logger = logging.getLogger('SentimentConfig')
        
        # Determine mode
        if not _sentiment_config.ENABLE_SENTIMENT:
            mode = SentimentMode.DISABLED
        elif _sentiment_config.IS_MOCK:
            mode = SentimentMode.MOCK
        else:
            mode = SentimentMode.REAL
        
        # Store mode for later access
        _sentiment_config._mode = mode
        
        # Print startup banner
        banner = f"""
================================================================================
[STARTUP] SENTIMENT CONFIGURATION
================================================================================
    Mode:                   {mode.value}
    vLLM Available:         {VLLM_AVAILABLE}
    Transformers Available: {TRANSFORMERS_AVAILABLE}
    Is Mock:                {_sentiment_config.IS_MOCK}
    Can Trigger Opportunity: {_sentiment_config.can_trigger_opportunity()}
    
    {"[WARN]️  MOCK MODE: Sentiment-based OPPORTUNITY triggers are DISABLED" if mode == SentimentMode.MOCK else ""}
    {"OFF DISABLED: Sentiment is completely disabled" if mode == SentimentMode.DISABLED else ""}
================================================================================
"""
        # Log at WARNING level to ensure visibility
        logger.warning(banner)
        
        # Also print to stdout for CLI visibility
        print(banner)
        
        # Standard log entry for parsing
        logger.warning(
            f"[STARTUP] SENTIMENT_MODE={mode.value} "
            f"CAN_TRIGGER_OPPORTUNITY={_sentiment_config.can_trigger_opportunity()} "
            f"IS_MOCK={_sentiment_config.IS_MOCK}"
        )
        
    return _sentiment_config


def get_sentiment_mode() -> SentimentMode:
    """Get current sentiment mode."""
    config = get_sentiment_config()
    if hasattr(config, '_mode'):
        return config._mode
    if not config.ENABLE_SENTIMENT:
        return SentimentMode.DISABLED
    if config.IS_MOCK:
        return SentimentMode.MOCK
    return SentimentMode.REAL


def set_sentiment_config(config: SentimentConfig):
    """Set sentiment config."""
    global _sentiment_config
    _sentiment_config = config
    logger = logging.getLogger('SentimentConfig')
    logger.info(
        f"[Sentiment] Config updated: status={config.get_status_string()}, "
        f"can_trigger_opportunity={config.can_trigger_opportunity()}"
    )


def sentiment_can_trigger_opportunity() -> bool:
    """Check if sentiment can trigger OPPORTUNITY mode."""
    return get_sentiment_config().can_trigger_opportunity()


def is_sentiment_mock() -> bool:
    """Check if sentiment is in mock mode."""
    return get_sentiment_config().IS_MOCK


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('vLLMInferenceWrapper')


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Default models
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QUANTIZED_MODEL = "TheBloke/Llama-3-8B-Instruct-GPTQ"

# Sentiment analysis prompts
SENTIMENT_SYSTEM_PROMPT = """You are a crypto market sentiment analyzer. Your task is to analyze news/social media text and output a sentiment score.

Rules:
1. Output ONLY a JSON object with no other text
2. Score range: -1.0 (extremely bearish) to +1.0 (extremely bullish)
3. Confidence range: 0.0 to 1.0
4. Be objective and fact-based

Output format:
{"score": <float>, "confidence": <float>, "reason": "<brief explanation>"}"""

SENTIMENT_USER_TEMPLATE = """Analyze the sentiment of this crypto market text:

"{text}"

Asset focus: {asset}

Output the JSON sentiment analysis:"""


class SentimentCategory(Enum):
    """Sentiment classification"""
    EXTREME_BEARISH = auto()  # score < -0.7
    BEARISH = auto()          # -0.7 <= score < -0.3
    NEUTRAL = auto()          # -0.3 <= score < 0.3
    BULLISH = auto()          # 0.3 <= score < 0.7
    EXTREME_BULLISH = auto()  # score >= 0.7


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    asset: str
    score: float                  # -1.0 to +1.0
    confidence: float             # 0.0 to 1.0
    category: SentimentCategory
    reason: str
    divergence_sigma: float = 0.0  # Divergence from consensus
    is_artificial_noise: bool = False
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text[:100],
            'asset': self.asset,
            'score': self.score,
            'confidence': self.confidence,
            'category': self.category.name,
            'reason': self.reason,
            'divergence_sigma': self.divergence_sigma,
            'is_artificial_noise': self.is_artificial_noise,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp
        }


@dataclass
class ConsensusState:
    """Running consensus for divergence detection"""
    asset: str
    scores: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def mean(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(list(self.scores)))
    
    @property
    def std(self) -> float:
        if len(self.scores) < 2:
            return 1.0  # Default to avoid division issues
        return float(np.std(list(self.scores)))
    
    def add_score(self, score: float):
        self.scores.append(score)
    
    def get_divergence(self, score: float) -> float:
        """Get divergence in sigma units"""
        std = self.std
        if std < 0.01:
            return 0.0
        return abs(score - self.mean) / std


# =============================================================================
# VLLM ENGINE
# =============================================================================

class vLLMEngine:
    """
    vLLM-based inference engine for high-throughput serving.
    Optimized for RTX 5090 with Flash Attention 3.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "auto"
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        
        self._llm: Optional[Any] = None
        self._sampling_params: Optional[Any] = None
        self._initialized = False
        self._lock = threading.Lock()
        
        logger.info(f"vLLMEngine created for {model_name}")
    
    def initialize(self):
        """Initialize the vLLM model"""
        if self._initialized:
            return
        
        if not VLLM_AVAILABLE:
            logger.warning("vLLM not available, will use mock inference")
            self._initialized = True
            return
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=True
            )
            
            self._sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent outputs
                top_p=0.9,
                max_tokens=256,
                stop=["\n\n", "```"]
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Model loaded in {elapsed:.1f}s")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialized = True  # Mark as initialized to prevent retries
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Any = None
    ) -> List[str]:
        """
        Generate completions for prompts.
        
        Args:
            prompts: List of prompt strings
            sampling_params: Optional custom sampling params
            
        Returns:
            List of generated texts
        """
        if not self._initialized:
            self.initialize()
        
        if self._llm is None:
            # Mock inference
            return [self._mock_generate(p) for p in prompts]
        
        params = sampling_params or self._sampling_params
        
        with self._lock:
            outputs = self._llm.generate(prompts, params)
        
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")
        
        return results
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing"""
        # Random sentiment
        score = np.random.uniform(-0.5, 0.5)
        confidence = np.random.uniform(0.5, 0.9)
        return json.dumps({
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "reason": "Mock analysis - model not loaded"
        })


# =============================================================================
# TRANSFORMERS FALLBACK ENGINE
# =============================================================================

class TransformersEngine:
    """
    Fallback engine using HuggingFace Transformers.
    Used when vLLM is not available.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        load_in_4bit: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._lock = threading.Lock()
        
        logger.info(f"TransformersEngine created for {model_name}")
    
    def initialize(self):
        """Initialize model and tokenizer"""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, will use mock inference")
            self._initialized = True
            return
        
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.load_in_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Model loaded in {elapsed:.1f}s")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialized = True
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.1
    ) -> List[str]:
        """Generate completions"""
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            return [self._mock_generate(p) for p in prompts]
        
        results = []
        
        with self._lock:
            for prompt in prompts:
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id
                    )
                
                generated = self._tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                results.append(generated)
        
        return results
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation"""
        score = np.random.uniform(-0.5, 0.5)
        confidence = np.random.uniform(0.5, 0.9)
        return json.dumps({
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "reason": "Mock analysis - model not loaded"
        })


# =============================================================================
# FLASH SENTIMENT AUDITOR
# =============================================================================

class FlashSentimentAuditor:
    """
    High-speed sentiment auditor using local SLM.
    
    Features:
    - Batch processing for efficiency
    - Consensus tracking and divergence detection
    - "Artificial Noise" flagging (> 3σ divergence)
    - Asset-specific analysis
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_vllm: bool = True,
        divergence_threshold: float = 3.0,  # 3σ
        device: str = "cuda:0"
    ):
        self.model_name = model_name
        self.divergence_threshold = divergence_threshold
        
        # Select engine
        if use_vllm and VLLM_AVAILABLE:
            self._engine = vLLMEngine(model_name=model_name)
        else:
            self._engine = TransformersEngine(model_name=model_name, device=device)
        
        # Consensus tracking per asset
        self._consensus: Dict[str, ConsensusState] = {}
        for asset in ['BTC', 'ETH', 'SOL', 'CRYPTO']:
            self._consensus[asset] = ConsensusState(asset=asset)
        
        # Results history
        self._results_history: deque = deque(maxlen=1000)
        
        # Metrics
        self._total_analyzed = 0
        self._noise_detected = 0
        
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='SentimentAuditor')
        
        logger.info(f"FlashSentimentAuditor initialized with {type(self._engine).__name__}")
    
    def initialize(self):
        """Initialize the underlying engine"""
        self._engine.initialize()
    
    def _build_prompt(self, text: str, asset: str = 'CRYPTO') -> str:
        """Build the sentiment analysis prompt"""
        # Llama-3 chat format
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": SENTIMENT_USER_TEMPLATE.format(text=text, asset=asset)}
        ]
        
        # Format as chat template
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _parse_response(self, response: str) -> Tuple[float, float, str]:
        """
        Parse model response to extract sentiment.
        
        Returns:
            (score, confidence, reason)
        """
        # Try to extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get('score', 0.0))
                confidence = float(data.get('confidence', 0.5))
                reason = str(data.get('reason', ''))
                
                # Clamp values
                score = max(-1.0, min(1.0, score))
                confidence = max(0.0, min(1.0, confidence))
                
                return score, confidence, reason
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try to extract score from text
        score_match = re.search(r'(-?\d+\.?\d*)', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                if -10 <= score <= 10:  # Assume 0-10 scale
                    score = score / 10 if score > 1 else score
                score = max(-1.0, min(1.0, score))
                return score, 0.5, "Extracted from text"
            except ValueError:
                pass
        
        return 0.0, 0.3, "Failed to parse"
    
    def _classify_sentiment(self, score: float) -> SentimentCategory:
        """Classify sentiment score into category"""
        if score >= 0.7:
            return SentimentCategory.EXTREME_BULLISH
        elif score >= 0.3:
            return SentimentCategory.BULLISH
        elif score >= -0.3:
            return SentimentCategory.NEUTRAL
        elif score >= -0.7:
            return SentimentCategory.BEARISH
        return SentimentCategory.EXTREME_BEARISH
    
    def analyze(
        self,
        text: str,
        asset: str = 'CRYPTO'
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            asset: Target asset (BTC, ETH, SOL, CRYPTO)
            
        Returns:
            SentimentResult
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(text, asset)
        
        # Generate
        responses = self._engine.generate([prompt])
        response = responses[0] if responses else ""
        
        # Parse response
        score, confidence, reason = self._parse_response(response)
        
        # Get consensus state
        consensus = self._consensus.get(asset, self._consensus['CRYPTO'])
        divergence = consensus.get_divergence(score)
        is_noise = divergence > self.divergence_threshold
        
        # Update consensus
        if not is_noise:  # Don't include noise in consensus
            consensus.add_score(score)
        
        # Build result
        result = SentimentResult(
            text=text,
            asset=asset,
            score=score,
            confidence=confidence if not is_noise else confidence * 0.5,  # Reduce confidence for noise
            category=self._classify_sentiment(score),
            reason=reason,
            divergence_sigma=divergence,
            is_artificial_noise=is_noise,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Update metrics
        self._total_analyzed += 1
        if is_noise:
            self._noise_detected += 1
            logger.warning(f"Artificial noise detected: {asset} score={score:.2f} divergence={divergence:.1f}σ")
        
        self._results_history.append(result)
        
        return result
    
    def analyze_batch(
        self,
        texts: List[str],
        assets: List[str] = None
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            assets: List of assets (default: CRYPTO for all)
            
        Returns:
            List of SentimentResults
        """
        if assets is None:
            assets = ['CRYPTO'] * len(texts)
        
        start_time = time.time()
        
        # Build prompts
        prompts = [
            self._build_prompt(text, asset)
            for text, asset in zip(texts, assets)
        ]
        
        # Generate all at once (vLLM is optimized for batching)
        responses = self._engine.generate(prompts)
        
        # Process results
        results = []
        for text, asset, response in zip(texts, assets, responses):
            score, confidence, reason = self._parse_response(response)
            
            consensus = self._consensus.get(asset, self._consensus['CRYPTO'])
            divergence = consensus.get_divergence(score)
            is_noise = divergence > self.divergence_threshold
            
            if not is_noise:
                consensus.add_score(score)
            
            result = SentimentResult(
                text=text,
                asset=asset,
                score=score,
                confidence=confidence if not is_noise else confidence * 0.5,
                category=self._classify_sentiment(score),
                reason=reason,
                divergence_sigma=divergence,
                is_artificial_noise=is_noise,
                processing_time_ms=0  # Set below
            )
            results.append(result)
            
            self._total_analyzed += 1
            if is_noise:
                self._noise_detected += 1
        
        # Set processing time for all
        total_time = (time.time() - start_time) * 1000
        per_item_time = total_time / len(texts) if texts else 0
        for result in results:
            result.processing_time_ms = per_item_time
        
        self._results_history.extend(results)
        
        return results
    
    def get_aggregate_sentiment(
        self,
        asset: str = 'CRYPTO',
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get aggregate sentiment for recent period.
        
        Args:
            asset: Target asset
            window_minutes: Time window
            
        Returns:
            Dict with aggregate metrics
        """
        cutoff = time.time() - window_minutes * 60
        
        recent = [
            r for r in self._results_history
            if r.timestamp > cutoff and (r.asset == asset or asset == 'CRYPTO')
            and not r.is_artificial_noise
        ]
        
        if not recent:
            return {
                'asset': asset,
                'score': 0.0,
                'confidence': 0.0,
                'count': 0,
                'bullish_ratio': 0.5,
                'noise_ratio': 0.0
            }
        
        scores = [r.score for r in recent]
        confidences = [r.confidence for r in recent]
        
        # Confidence-weighted average
        weights = np.array(confidences)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights)
        weighted_score = float(np.sum(np.array(scores) * weights))
        
        bullish = sum(1 for r in recent if r.score > 0.3)
        
        return {
            'asset': asset,
            'score': weighted_score,
            'confidence': float(np.mean(confidences)),
            'count': len(recent),
            'bullish_ratio': bullish / len(recent),
            'noise_ratio': self._noise_detected / self._total_analyzed if self._total_analyzed > 0 else 0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get auditor metrics"""
        return {
            'total_analyzed': self._total_analyzed,
            'noise_detected': self._noise_detected,
            'noise_ratio': self._noise_detected / self._total_analyzed if self._total_analyzed > 0 else 0,
            'consensus': {
                asset: {
                    'mean': state.mean,
                    'std': state.std,
                    'count': len(state.scores)
                }
                for asset, state in self._consensus.items()
            }
        }


# =============================================================================
# VLLM INFERENCE WRAPPER (Main Interface)
# =============================================================================

class vLLMInferenceWrapper:
    """
    Main wrapper class combining all components.
    
    Provides:
    - SentimentAuditor for news/social analysis
    - Consensus verification
    - SharedStateBuffer integration
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_vllm: bool = True,
        auto_initialize: bool = False
    ):
        self.model_name = model_name
        
        # Create sentiment auditor
        self.sentiment_auditor = FlashSentimentAuditor(
            model_name=model_name,
            use_vllm=use_vllm
        )
        
        # State integration
        self._last_update: Optional[Dict] = None
        self._callback: Optional[Callable] = None
        
        if auto_initialize:
            self.initialize()
        
        logger.info("vLLMInferenceWrapper created")
    
    def initialize(self):
        """Initialize model"""
        self.sentiment_auditor.initialize()
    
    def set_state_callback(self, callback: Callable[[Dict], None]):
        """Set callback for state updates"""
        self._callback = callback
    
    def analyze_sentiment(
        self,
        text: str,
        asset: str = 'CRYPTO'
    ) -> SentimentResult:
        """Analyze sentiment and update state"""
        result = self.sentiment_auditor.analyze(text, asset)
        
        # Update state
        self._last_update = result.to_dict()
        
        # Trigger callback
        if self._callback:
            self._callback(self._last_update)
        
        return result
    
    def analyze_batch(
        self,
        texts: List[str],
        assets: List[str] = None
    ) -> List[SentimentResult]:
        """Analyze batch and update state"""
        results = self.sentiment_auditor.analyze_batch(texts, assets)
        
        # Update state with aggregate
        if results:
            self._last_update = self.sentiment_auditor.get_aggregate_sentiment()
            if self._callback:
                self._callback(self._last_update)
        
        return results
    
    def get_aggregate(
        self,
        asset: str = 'CRYPTO',
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get aggregate sentiment"""
        return self.sentiment_auditor.get_aggregate_sentiment(asset, window_minutes)
    
    def get_drl_features(self) -> np.ndarray:
        """
        Get sentiment features for DRL state vector.
        
        Returns:
            Array of shape (9,) with sentiment features:
            - btc_sentiment, btc_confidence, btc_noise_flag
            - eth_sentiment, eth_confidence, eth_noise_flag
            - sol_sentiment, sol_confidence, sol_noise_flag
        """
        features = []
        
        for asset in ['BTC', 'ETH', 'SOL']:
            agg = self.sentiment_auditor.get_aggregate_sentiment(asset, 60)
            features.extend([
                agg['score'],
                agg['confidence'],
                agg['noise_ratio']
            ])
        
        return np.array(features, dtype=np.float32)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.sentiment_auditor.get_metrics()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstration of vLLMInferenceWrapper"""
    print("="*70)
    print("vLLM Inference Wrapper Demo (Flash Sentiment Auditor)")
    print("="*70)
    
    print(f"\n▶ Library Status:")
    print(f"  vLLM: {VLLM_AVAILABLE}")
    print(f"  Transformers: {TRANSFORMERS_AVAILABLE}")
    
    # Create wrapper (will use mock if no model available)
    wrapper = vLLMInferenceWrapper(
        model_name=DEFAULT_MODEL,
        use_vllm=VLLM_AVAILABLE,
        auto_initialize=True
    )
    
    # Test texts
    test_texts = [
        "Bitcoin breaks $100k as institutional buying accelerates",
        "ETH faces selling pressure amid regulatory concerns",
        "SOL network processes 50k TPS, showing massive adoption",
        "Market crash incoming? Whales moving to exchanges",
        "Neutral market activity expected as traders await Fed decision",
    ]
    
    test_assets = ['BTC', 'ETH', 'SOL', 'CRYPTO', 'CRYPTO']
    
    # Analyze one by one
    print("\n▶ Single Analysis:")
    result = wrapper.analyze_sentiment(test_texts[0], 'BTC')
    print(f"  Text: {test_texts[0][:50]}...")
    print(f"  Score: {result.score:.2f} ({result.category.name})")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Noise: {result.is_artificial_noise}")
    print(f"  Time: {result.processing_time_ms:.1f}ms")
    
    # Batch analysis
    print("\n▶ Batch Analysis:")
    results = wrapper.analyze_batch(test_texts, test_assets)
    for i, result in enumerate(results):
        status = "🚨 NOISE" if result.is_artificial_noise else "✓"
        print(f"  {status} {result.asset}: {result.score:+.2f} ({result.category.name})")
    
    # Aggregate sentiment
    print("\n▶ Aggregate Sentiment (Last 60 min):")
    for asset in ['BTC', 'ETH', 'SOL', 'CRYPTO']:
        agg = wrapper.get_aggregate(asset, 60)
        print(f"  {asset}: score={agg['score']:+.2f}, "
              f"confidence={agg['confidence']:.2f}, "
              f"count={agg['count']}")
    
    # DRL features
    print("\n▶ DRL Features:")
    features = wrapper.get_drl_features()
    print(f"  Shape: {features.shape}")
    print(f"  Values: {features}")
    
    # Metrics
    print("\n▶ Auditor Metrics:")
    metrics = wrapper.get_metrics()
    print(f"  Total analyzed: {metrics['total_analyzed']}")
    print(f"  Noise detected: {metrics['noise_detected']}")
    print(f"  Noise ratio: {metrics['noise_ratio']:.1%}")
    
    print("\n✓ Demo complete")


if __name__ == '__main__':
    demo()
