#!/usr/bin/env python3
"""
Clean and prepare sentiment training data for DeBERTa v2.2.

Steps:
1. Load sentiment_data_full.parquet (104k, source of truth)
2. Remove near-duplicates (first 50 char prefix dedup)
3. Remove bot/spam patterns
4. Remove very short texts (<30 chars) and URL-only
5. Remove non-English (heuristic)
6. Balance classes via undersampling (not oversampling!)
7. Save clean dataset
"""

import pandas as pd
import re
import sys
from pathlib import Path


def is_spam(text: str) -> bool:
    """Detect bot/spam content."""
    patterns = [
        r'Buy Crypto With Your Credit Card',
        r'Buy BTC With Your Credit Card',
        r'Fast Crypto Exchang',
        r'Follow me on @betfury',
        r'I joined a network',
        r'I found #bitcoin in a .* vault',
        r'More than \d+% Profits and',
        r'Redfin and Zillow are a scam',
        r'Hey I joined a network called Survey',
        r'voice vault',
        r'Join .* airdrop',
        r'Free .* giveaway',
        r'DM me for .* profit',
        r'100% guaranteed',
        r'SELLING PRESSURE ALERT',
        r'BUYING PRESSURE ALERT',
        r'weakly trending (up|down) currently',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def is_low_quality(text: str) -> bool:
    """Detect low quality text."""
    # Too short
    if len(text.strip()) < 30:
        return True
    # URL only
    if re.match(r'^https?://', text.strip()):
        return True
    # Mostly hashtags/mentions
    words = text.split()
    tag_count = sum(1 for w in words if w.startswith('#') or w.startswith('@'))
    if len(words) > 0 and tag_count / len(words) > 0.7:
        return True
    # Mostly emojis/symbols (less than 30% alphanumeric)
    alnum = sum(1 for c in text if c.isalnum() or c == ' ')
    if len(text) > 0 and alnum / len(text) < 0.3:
        return True
    return False


def dedup_by_prefix(df: pd.DataFrame, prefix_len: int = 60) -> pd.DataFrame:
    """Remove near-duplicates by keeping first occurrence of each prefix."""
    df = df.copy()
    df['_prefix'] = df['text'].str[:prefix_len]
    df = df.drop_duplicates(subset='_prefix', keep='first')
    df = df.drop(columns=['_prefix'])
    return df


def main():
    data_dir = Path('training/training_data/sentiment')

    # Load primary source
    print("Loading sentiment_data_full.parquet...")
    df = pd.read_parquet(data_dir / 'sentiment_data_full.parquet')
    print(f"  Raw: {len(df)} rows, labels: {df['label'].value_counts().to_dict()}")

    # Label mapping: 0=negative/bearish, 1=neutral, 2=positive/bullish

    # Step 1: Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first')
    print(f"  After exact dedup: {len(df)} (-{before - len(df)})")

    # Step 2: Remove spam/bot
    before = len(df)
    df = df[~df['text'].apply(is_spam)]
    print(f"  After spam removal: {len(df)} (-{before - len(df)})")

    # Step 3: Remove low quality
    before = len(df)
    df = df[~df['text'].apply(is_low_quality)]
    print(f"  After quality filter: {len(df)} (-{before - len(df)})")

    # Step 4: Near-dedup by prefix
    before = len(df)
    df = dedup_by_prefix(df, prefix_len=60)
    print(f"  After prefix dedup (60 chars): {len(df)} (-{before - len(df)})")

    # Step 5: Check distribution
    print(f"\n  Clean distribution:")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        print(f"    Label {label}: {count}")

    # Step 6: Balance by undersampling to min class size
    min_count = df['label'].value_counts().min()
    print(f"\n  Balancing to {min_count} per class (undersample)...")
    balanced = df.groupby('label').apply(
        lambda x: x.sample(n=min_count, random_state=42),
        include_groups=False
    ).reset_index(level=0)  # keep label from groupby

    # Fix: reset_index may produce different column layout
    # Just re-sample properly
    balanced_parts = []
    for label in sorted(df['label'].unique()):
        subset = df[df['label'] == label].sample(n=min_count, random_state=42)
        balanced_parts.append(subset)
    balanced = pd.concat(balanced_parts, ignore_index=True)

    # Shuffle
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Final: {len(balanced)} rows, labels: {balanced['label'].value_counts().to_dict()}")

    # Step 7: Verify quality
    print(f"\n  Quality check:")
    print(f"    Unique texts: {balanced['text'].nunique()} / {len(balanced)}")
    prefix_dupes = balanced['text'].str[:50].value_counts()
    remaining_dupes = (prefix_dupes > 1).sum()
    print(f"    Remaining near-dupe prefixes: {remaining_dupes}")
    print(f"    Avg text length: {balanced['text'].str.len().mean():.0f} chars")
    print(f"    Min text length: {balanced['text'].str.len().min()} chars")

    # Save
    out_path = data_dir / 'sentiment_clean.csv'
    balanced.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Show samples
    print("\n  Samples per label:")
    for label in [0, 1, 2]:
        name = ['bearish/negative', 'neutral', 'bullish/positive'][label]
        print(f"\n  --- Label {label} ({name}) ---")
        samples = balanced[balanced['label'] == label].head(3)
        for _, row in samples.iterrows():
            print(f"    {row['text'][:120]}")


if __name__ == '__main__':
    main()
