"""
L2 Sentiment: DeBERTa v2.2 local inference for crypto sentiment classification.

Sits between L3 (Haiku LLM, most accurate but requires API) and L1 (deterministic F&G).
Zero API cost, ~50ms inference per text on GPU, ~200ms on CPU.

Output: {direction: float [-1, 1], confidence: float [0, 1], label: str}
Labels: 0=negative, 1=neutral, 2=positive

Usage:
    engine = DeBERTaSentimentEngine()
    result = engine.predict("Bitcoin ETF approved by SEC")
    # {"direction": 0.8, "confidence": 0.92, "label": "positive"}
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_engine_instance: Optional["DeBERTaSentimentEngine"] = None


class DeBERTaSentimentEngine:
    """DeBERTa v2.2 sentiment inference engine."""

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    DIRECTION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}  # negative=-1, neutral=0, positive=+1
    MODEL_PATH = Path("models/sentiment_v22_best.pt")

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self._loaded = False

        _path = Path(model_path) if model_path else self.MODEL_PATH
        if not _path.exists():
            logger.warning(f"[L2_SENTIMENT] Model not found at {_path}")
            return

        try:
            import torch
            from transformers import AutoTokenizer

            # Determine device
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            # Register __main__ classes for unpickling
            import __main__
            from training.sentiment.train_sentiment_agent_v22 import (
                SentimentConfigV22, SentimentModelV22,
            )
            __main__.SentimentConfigV22 = SentimentConfigV22
            __main__.SentimentModelV22 = SentimentModelV22

            # Load checkpoint
            ckpt = torch.load(str(_path), map_location="cpu", weights_only=False)
            self.config = ckpt["config"]

            # Build model
            model = SentimentModelV22(self.config)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model.to(self.device)
            model.eval()
            self.model = model

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
            )

            self._loaded = True
            logger.info(
                f"[L2_SENTIMENT] DeBERTa v2.2 loaded: {self.config.model_name}, "
                f"device={self.device}, val_acc={ckpt.get('val_acc', 0):.3f}"
            )
        except Exception as e:
            logger.warning(f"[L2_SENTIMENT] Failed to load: {e}")
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single text. Returns direction, confidence, label."""
        if not self.is_ready:
            return {"direction": 0.0, "confidence": 0.0, "label": "neutral", "source": "l2_unavailable"}

        try:
            import torch
            import torch.nn.functional as F

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]  # (1, 3)
                probs = F.softmax(logits, dim=-1)[0]  # (3,)

            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            direction = self.DIRECTION_MAP[pred_idx]
            label = self.LABEL_MAP[pred_idx]

            # Refine direction with probability spread
            # If positive prob is 0.6 and negative is 0.1, direction = 0.6 - 0.1 = 0.5
            direction_refined = float(probs[2] - probs[0])  # positive - negative

            return {
                "direction": direction_refined,
                "confidence": confidence,
                "label": label,
                "probs": {self.LABEL_MAP[i]: float(probs[i]) for i in range(3)},
                "source": "l2_deberta",
            }
        except Exception as e:
            logger.debug(f"[L2_SENTIMENT] Predict failed: {e}")
            return {"direction": 0.0, "confidence": 0.0, "label": "neutral", "source": "l2_error"}

    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Classify multiple texts and return aggregated sentiment."""
        if not self.is_ready or not texts:
            return {"direction": 0.0, "confidence": 0.0, "label": "neutral", "source": "l2_unavailable", "count": 0}

        results = [self.predict(t) for t in texts[:20]]  # cap at 20 texts
        valid = [r for r in results if r["source"] == "l2_deberta"]

        if not valid:
            return {"direction": 0.0, "confidence": 0.0, "label": "neutral", "source": "l2_empty", "count": 0}

        avg_direction = sum(r["direction"] for r in valid) / len(valid)
        avg_confidence = sum(r["confidence"] for r in valid) / len(valid)

        if avg_direction > 0.15:
            label = "positive"
        elif avg_direction < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return {
            "direction": avg_direction,
            "confidence": avg_confidence,
            "label": label,
            "source": "l2_deberta",
            "count": len(valid),
        }


def get_deberta_sentiment_engine() -> DeBERTaSentimentEngine:
    """Singleton getter."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DeBERTaSentimentEngine()
    return _engine_instance
