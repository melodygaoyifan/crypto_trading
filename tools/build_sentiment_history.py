"""
================================================================================
HMATS - Build Historical Sentiment Timeseries (Stage 21+ optional)
================================================================================
Creates backtest-ready daily sentiment from Bitcoin_tweets.csv using the
calibrated Haiku prompt from agents/sentiment_llm_agent.py.

Pipeline:
  1. Group Bitcoin_tweets.csv by date
  2. Sample 50 tweets per day (matching production: 50 titles/call)
  3. One Haiku call per day with calibrated prompt
  4. Output: daily_sentiment.parquet
     columns: date, sentiment, confidence, crowding_risk, n_tweets, summary

Cost estimate:
  ~2000 days of data x 750 tokens/call = 1.5M tokens
  1.5M / 1M x $0.25 = ~$0.38 total

Usage:
  python -X utf8 tools/build_sentiment_history.py \
    --tweets hmats_training_v5/training_data/sentiment/Bitcoin_tweets.csv \
    --output data/sentiment_history/daily_sentiment.parquet \
    --max-days 2000

Output usage (Stage 21+ if sentiment proves valuable):
  - merge_asof to 4H bars -> potential obs feature
  - Requires retraining if added to obs (changes obs_dim)
  - Only pursue if L3 A/B shows >5% PnL improvement
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


async def process_day(
    client,
    model: str,
    system_prompt: str,
    date_str: str,
    tweets: List[str],
    sem: asyncio.Semaphore,
) -> Optional[Dict]:
    """Process one day's tweets through Haiku."""
    # Sample 50 tweets max (matches production)
    if len(tweets) > 50:
        indices = np.random.choice(len(tweets), 50, replace=False)
        tweets = [tweets[i] for i in sorted(indices)]

    joined = "\n".join(f"- {t}" for t in tweets)
    user_msg = f"Asset: BTC\n\nRecent headlines:\n{joined}\n\nAnalyze sentiment."

    async with sem:
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=200,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                ),
                timeout=15.0,
            )
            data = _parse_json_response(resp.content[0].text)
            if data is None:
                return None

            # Compute combined sentiment from factual/subjective
            fact_dir = float(data.get("factual_direction", 0))
            subj_dir = float(data.get("subjective_direction", 0))
            fact_conf = float(data.get("factual_confidence", 0))
            subj_conf = float(data.get("subjective_confidence", 0))
            total_w = fact_conf + subj_conf
            if total_w > 0.01:
                sentiment = (fact_dir * fact_conf + subj_dir * subj_conf) / total_w
            else:
                sentiment = 0.0
            confidence = max(fact_conf, subj_conf)

            return {
                "date": date_str,
                "sentiment": round(max(-1.0, min(1.0, sentiment)), 4),
                "confidence": round(min(1.0, confidence), 4),
                "factual_direction": round(max(-1.0, min(1.0, fact_dir)), 4),
                "subjective_direction": round(max(-1.0, min(1.0, subj_dir)), 4),
                "crowding_risk": bool(data.get("crowding_risk", False)),
                "n_tweets": len(tweets),
                "summary": str(data.get("summary", ""))[:200],
            }
        except Exception as e:
            logger.debug(f"Day {date_str} failed: {e}")
            return None


async def build_history(args):
    """Build daily sentiment history."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed. Run: pip install anthropic")
        return

    # Load calibrated prompt
    from agents.sentiment_llm_agent import _SYSTEM_PROMPT

    logger.info(f"Loading tweets from {args.tweets}...")
    df = pd.read_csv(args.tweets, encoding="utf-8", on_bad_lines="skip")

    # Find text and date columns
    text_col = None
    for c in ["text", "tweet", "content", "body"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break

    date_col = None
    for c in ["date", "timestamp", "created_at", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if text_col is None or date_col is None:
        logger.error(f"Cannot find text/date columns. Columns: {list(df.columns)}")
        return

    logger.info(f"Using text='{text_col}', date='{date_col}', {len(df)} rows")

    df["_text"] = df[text_col].astype(str)
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_text", "_date"])
    df["_date_str"] = df["_date"].dt.strftime("%Y-%m-%d")

    # Group by day
    daily = df.groupby("_date_str")["_text"].apply(list).to_dict()
    dates = sorted(daily.keys())

    if args.max_days and len(dates) > args.max_days:
        # Sample evenly across the range
        indices = np.linspace(0, len(dates) - 1, args.max_days, dtype=int)
        dates = [dates[i] for i in indices]

    logger.info(f"Processing {len(dates)} days...")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    model = "claude-haiku-4-5-20251001"

    # Resume support: check for existing partial output
    output_path = Path(args.output)
    existing_dates = set()
    results = []
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            existing_dates = set(existing_df["date"].astype(str))
            results = existing_df.to_dict("records")
            logger.info(f"Resuming: {len(existing_dates)} days already processed")
        except Exception:
            pass

    remaining = [d for d in dates if d not in existing_dates]
    logger.info(f"Remaining: {len(remaining)} days to process")

    # Process in batches
    batch_size = 20
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        tasks = [
            process_day(client, model, _SYSTEM_PROMPT, d, daily[d], sem)
            for d in batch
        ]
        batch_results = await asyncio.gather(*tasks)

        for r in batch_results:
            if r is not None:
                results.append(r)

        # Save checkpoint
        if results:
            out_df = pd.DataFrame(results)
            out_df["date"] = pd.to_datetime(out_df["date"])
            out_df = out_df.sort_values("date").reset_index(drop=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(output_path, index=False)

        processed = min(batch_start + batch_size, len(remaining))
        logger.info(f"Progress: {processed}/{len(remaining)} days, {len(results)} total results")

    logger.info(f"Done! {len(results)} daily sentiment records saved to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="HMATS Historical Sentiment Builder")
    parser.add_argument("--tweets", required=True, help="Path to Bitcoin_tweets.csv")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max-days", type=int, default=2000, help="Max days to process")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent API calls")

    args = parser.parse_args()
    asyncio.run(build_history(args))


if __name__ == "__main__":
    main()