"""
================================================================================
HMATS - Sentiment Prompt Calibration Pipeline
================================================================================
Run ONCE before deploying L3 LLM sentiment.

Steps:
  1. Sample 500 diverse tweets from Bitcoin_tweets.csv (stratified by date)
  2. User manually labels 100 gold examples (~30 min) -> calibration_gold_100.csv
  3. Haiku batch-labels 400 tweets with the BASIC prompt (~$0.02)
  4. Score Haiku on gold labels -> find systematic errors
  5. Re-score with calibrated prompt (from agents/sentiment_llm_agent.py)
  6. Output calibration_report.json

Usage:
  # Step 1: Sample tweets
  python -X utf8 tools/calibrate_sentiment_prompt.py sample \
    --tweets hmats_training_v5/training_data/sentiment/Bitcoin_tweets.csv \
    --output data/calibration/sample_500.csv

  # Step 2: Manually label 100 (outputs template CSV with empty label columns)
  python -X utf8 tools/calibrate_sentiment_prompt.py label-template \
    --sample data/calibration/sample_500.csv \
    --output data/calibration/gold_100_template.csv

  # Step 3+4+5+6: Run calibration (after manual labeling)
  python -X utf8 tools/calibrate_sentiment_prompt.py calibrate \
    --sample data/calibration/sample_500.csv \
    --gold data/calibration/calibration_gold_100.csv \
    --output data/calibration/calibration_report.json

Cost: ~$0.04 one-time.  Time: ~30 min manual labeling + ~10 min script.
================================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BASIC PROMPT (no few-shot - used as baseline for calibration comparison)
# ---------------------------------------------------------------------------

_BASIC_SYSTEM_PROMPT = (
    "You are a crypto market sentiment analyst. Given a tweet about Bitcoin, "
    "output a JSON object:\n"
    '{"sentiment": <float -1.0 to 1.0>, "confidence": <float 0.0 to 1.0>}\n'
    "sentiment: -1=very bearish, 0=neutral, +1=very bullish.\n"
    "confidence: 0=no signal, 1=very clear signal.\n"
    "Output ONLY JSON. No markdown."
)


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse LLM JSON with fence stripping."""
    try:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# STEP 1: Sample diverse tweets
# ---------------------------------------------------------------------------

def sample_diverse(tweets_path: str, n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Sample n tweets from Bitcoin_tweets.csv, stratified across dates.

    Filters: English-likely (ASCII), >10 words, no pure links/retweets.
    """
    logger.info(f"Loading tweets from {tweets_path} (first 200K rows for speed)...")
    df = pd.read_csv(tweets_path, nrows=200_000, encoding="utf-8", on_bad_lines="skip")

    # Identify text column (varies by dataset)
    text_col = None
    for candidate in ["text", "tweet", "content", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Fallback: use first string column
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    logger.info(f"Using text column: '{text_col}', {len(df)} rows loaded")

    # Basic filters
    df = df.dropna(subset=[text_col])
    df["_text"] = df[text_col].astype(str)
    df = df[df["_text"].str.len() > 20]  # Skip very short
    df = df[df["_text"].str.split().str.len() > 5]  # >5 words
    df = df[~df["_text"].str.startswith("RT ")]  # No retweets
    df = df[~df["_text"].str.match(r"^https?://")]  # No pure links

    # Try to parse dates for stratification
    date_col = None
    for candidate in ["date", "timestamp", "created_at", "datetime"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        try:
            df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["_date"])
            # Bin into 10 equal date bins for stratification
            df["_date_bin"] = pd.qcut(df["_date"].astype(int), q=min(10, len(df) // 50), labels=False, duplicates="drop")
            sample = df.groupby("_date_bin", group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), n // 10 + 1), random_state=seed),
                include_groups=False,
            ).head(n)
            logger.info(f"Stratified sample: {len(sample)} tweets across {sample['_date_bin'].nunique()} date bins")
        except Exception as e:
            logger.warning(f"Date stratification failed ({e}), using random sample")
            sample = df.sample(n=min(n, len(df)), random_state=seed)
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed)

    # Clean output
    result = sample[["_text"]].copy()
    result.columns = ["text"]
    if date_col and "_date" in df.columns:
        result["date"] = sample["_date"].values
    result = result.reset_index(drop=True)
    logger.info(f"Final sample: {len(result)} tweets")
    return result


def create_label_template(sample_df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create CSV template for manual labeling (first 100 from sample)."""
    template = sample_df.head(n).copy()
    template["sentiment"] = ""  # User fills: -1.0 to 1.0
    template["confidence"] = ""  # User fills: 0.0 to 1.0
    template["notes"] = ""  # Optional notes
    return template


# ---------------------------------------------------------------------------
# STEP 3: Batch label with Haiku
# ---------------------------------------------------------------------------

async def batch_label_haiku(
    texts: List[str],
    system_prompt: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 3,
    delay_ms: int = 200,
) -> List[Optional[Dict]]:
    """Label texts with Haiku. Returns list of parsed dicts (None on failure)."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed. Run: pip install anthropic")
        return [None] * len(texts)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[Dict]] = [None] * len(texts)

    async def label_one(idx: int, text: str):
        async with sem:
            try:
                await asyncio.sleep(delay_ms / 1000.0)
                resp = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=150,
                        temperature=0,
                        system=system_prompt,
                        messages=[{"role": "user", "content": f"Tweet: {text}"}],
                    ),
                    timeout=15.0,
                )
                parsed = _parse_json_response(resp.content[0].text)
                results[idx] = parsed
            except Exception as e:
                logger.debug(f"Label failed for idx={idx}: {e}")

    tasks = [label_one(i, t) for i, t in enumerate(texts)]
    await asyncio.gather(*tasks)

    success = sum(1 for r in results if r is not None)
    logger.info(f"Batch labeling: {success}/{len(texts)} succeeded")
    return results


# ---------------------------------------------------------------------------
# STEP 4: Score predictions against gold labels
# ---------------------------------------------------------------------------

def score_predictions(
    gold_sentiments: List[float],
    pred_sentiments: List[float],
    gold_confidences: List[float],
    pred_confidences: List[float],
) -> Dict:
    """Score Haiku predictions against gold labels."""
    gold_s = np.array(gold_sentiments, dtype=float)
    pred_s = np.array(pred_sentiments, dtype=float)

    # Filter out missing predictions
    valid = ~(np.isnan(pred_s))
    gold_s = gold_s[valid]
    pred_s = pred_s[valid]

    if len(gold_s) == 0:
        return {"error": "no valid predictions"}

    # MAE on sentiment
    mae = float(np.mean(np.abs(gold_s - pred_s)))

    # 3-class accuracy: bearish (<-0.2), neutral, bullish (>0.2)
    def to_class(arr):
        return np.where(arr < -0.2, -1, np.where(arr > 0.2, 1, 0))

    gold_cls = to_class(gold_s)
    pred_cls = to_class(pred_s)
    accuracy = float(np.mean(gold_cls == pred_cls))

    # Systematic bias
    bias = float(np.mean(pred_s - gold_s))

    # Per-class bias
    class_bias = {}
    for cls_name, cls_val in [("bearish", -1), ("neutral", 0), ("bullish", 1)]:
        mask = gold_cls == cls_val
        if mask.sum() > 0:
            class_bias[cls_name] = {
                "count": int(mask.sum()),
                "mean_gold": float(gold_s[mask].mean()),
                "mean_pred": float(pred_s[mask].mean()),
                "bias": float((pred_s[mask] - gold_s[mask]).mean()),
            }

    return {
        "n_valid": int(valid.sum()),
        "n_total": len(gold_sentiments),
        "mae": round(mae, 4),
        "accuracy_3class": round(accuracy, 4),
        "systematic_bias": round(bias, 4),
        "per_class": class_bias,
    }


def find_disagreements(
    texts: List[str],
    gold_sentiments: List[float],
    pred_sentiments: List[float],
    top_n: int = 10,
) -> List[Dict]:
    """Find cases with largest |haiku - gold| for few-shot example candidates."""
    diffs = []
    for i, (text, gold, pred) in enumerate(zip(texts, gold_sentiments, pred_sentiments)):
        if pred is not None and not np.isnan(pred):
            diffs.append({
                "index": i,
                "text": text[:200],
                "gold_sentiment": gold,
                "pred_sentiment": pred,
                "abs_error": abs(gold - pred),
            })
    diffs.sort(key=lambda x: x["abs_error"], reverse=True)
    return diffs[:top_n]


# ---------------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------------

async def run_calibrate(args):
    """Run full calibration pipeline."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return

    # Load sample and gold (try utf-8 first, fall back to latin-1 for mixed-encoding CSVs)
    for enc in ("utf-8", "latin-1"):
        try:
            sample_df = pd.read_csv(args.sample, encoding=enc, on_bad_lines="skip")
            break
        except (UnicodeDecodeError, Exception):
            continue
    for enc in ("utf-8", "latin-1"):
        try:
            gold_df = pd.read_csv(args.gold, encoding=enc, on_bad_lines="skip")
            break
        except (UnicodeDecodeError, Exception):
            continue

    gold_df["sentiment"] = pd.to_numeric(gold_df["sentiment"], errors="coerce")
    gold_df["confidence"] = pd.to_numeric(gold_df["confidence"], errors="coerce")
    gold_df = gold_df.dropna(subset=["sentiment"])

    logger.info(f"Gold labels: {len(gold_df)} tweets")

    gold_texts = gold_df["text"].tolist()
    gold_sentiments = gold_df["sentiment"].tolist()
    gold_confidences = gold_df["confidence"].fillna(0.5).tolist()

    # Step 3: Label gold set with BASIC prompt (baseline)
    logger.info("=== Baseline: labeling gold set with BASIC prompt ===")
    basic_results = await batch_label_haiku(
        gold_texts, _BASIC_SYSTEM_PROMPT, api_key,
    )
    basic_sentiments = [
        r.get("sentiment", float("nan")) if r else float("nan")
        for r in basic_results
    ]
    basic_confidences = [
        r.get("confidence", float("nan")) if r else float("nan")
        for r in basic_results
    ]

    # Step 4: Score baseline
    baseline_score = score_predictions(
        gold_sentiments, basic_sentiments, gold_confidences, basic_confidences,
    )
    logger.info(f"Baseline MAE={baseline_score['mae']:.4f}, "
                f"Accuracy={baseline_score['accuracy_3class']:.4f}")

    # Find top disagreements
    disagreements = find_disagreements(gold_texts, gold_sentiments, basic_sentiments)
    logger.info(f"Top {len(disagreements)} disagreements (for few-shot candidates):")
    for d in disagreements[:5]:
        logger.info(f"  [{d['index']}] gold={d['gold_sentiment']:+.2f} "
                     f"pred={d['pred_sentiment']:+.2f} err={d['abs_error']:.2f}: "
                     f"{d['text'][:80]}...")

    # Step 5: Label gold set with CALIBRATED prompt (from agents/sentiment_llm_agent.py)
    logger.info("=== Calibrated: labeling gold set with SHORT-BIAS prompt ===")
    # Add project root to path so agents/ is importable
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from agents.sentiment_llm_agent import _SYSTEM_PROMPT as calibrated_prompt

    # The calibrated prompt uses factual/subjective split - extract combined sentiment
    calibrated_results = await batch_label_haiku(
        gold_texts, calibrated_prompt, api_key,
    )
    calibrated_sentiments = []
    for r in calibrated_results:
        if r and "factual_direction" in r and "subjective_direction" in r:
            # Simple average for comparison scoring
            s = 0.5 * r.get("factual_direction", 0) + 0.5 * r.get("subjective_direction", 0)
            calibrated_sentiments.append(s)
        elif r and "sentiment" in r:
            calibrated_sentiments.append(r["sentiment"])
        else:
            calibrated_sentiments.append(float("nan"))

    # Step 6: Score calibrated
    calibrated_score = score_predictions(
        gold_sentiments, calibrated_sentiments, gold_confidences,
        [0.5] * len(calibrated_sentiments),
    )
    logger.info(f"Calibrated MAE={calibrated_score['mae']:.4f}, "
                f"Accuracy={calibrated_score['accuracy_3class']:.4f}")

    # Build report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gold_count": len(gold_df),
        "baseline": baseline_score,
        "calibrated": calibrated_score,
        "improvement": {
            "mae_delta": round(baseline_score["mae"] - calibrated_score["mae"], 4),
            "accuracy_delta": round(
                calibrated_score["accuracy_3class"] - baseline_score["accuracy_3class"], 4
            ),
        },
        "top_disagreements": disagreements[:10],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Calibration report saved to {output_path}")
    logger.info(f"MAE improvement: {report['improvement']['mae_delta']:+.4f}")
    logger.info(f"Accuracy improvement: {report['improvement']['accuracy_delta']:+.4f}")

    if calibrated_score["mae"] < 0.3:
        logger.info("PASS: MAE < 0.3 - prompt is calibrated for deployment")
    else:
        logger.warning("NEEDS WORK: MAE >= 0.3 - refine few-shot examples and re-run")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="HMATS Sentiment Prompt Calibration")
    sub = parser.add_subparsers(dest="command")

    # sample
    p_sample = sub.add_parser("sample", help="Sample diverse tweets")
    p_sample.add_argument("--tweets", required=True, help="Path to Bitcoin_tweets.csv")
    p_sample.add_argument("--output", required=True, help="Output CSV path")
    p_sample.add_argument("--n", type=int, default=500, help="Number of tweets to sample")

    # label-template
    p_label = sub.add_parser("label-template", help="Create manual labeling template")
    p_label.add_argument("--sample", required=True, help="Path to sample CSV")
    p_label.add_argument("--output", required=True, help="Output template CSV path")
    p_label.add_argument("--n", type=int, default=100, help="Number for manual labeling")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Run full calibration")
    p_cal.add_argument("--sample", required=True, help="Path to sample CSV")
    p_cal.add_argument("--gold", required=True, help="Path to gold labels CSV")
    p_cal.add_argument("--output", required=True, help="Output report JSON path")

    args = parser.parse_args()

    if args.command == "sample":
        df = sample_diverse(args.tweets, n=args.n)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved {len(df)} tweets to {args.output}")

    elif args.command == "label-template":
        sample_df = pd.read_csv(args.sample)
        template = create_label_template(sample_df, n=args.n)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(args.output, index=False)
        logger.info(f"Saved {len(template)}-row template to {args.output}")
        logger.info("Fill 'sentiment' (-1 to 1) and 'confidence' (0 to 1) columns, then run calibrate")

    elif args.command == "calibrate":
        asyncio.run(run_calibrate(args))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()