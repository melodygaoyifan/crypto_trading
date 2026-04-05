"""
HMATS - Tweet Cleaning Filters for Calibration Pipeline
========================================================
Filters for cleaning Bitcoin_tweets.csv before calibration sampling.
Production uses CryptoPanic (pre-filtered English headlines) - these
filters are for the calibration pipeline only.

Expected filter rates on Bitcoin_tweets.csv (~22.4M rows):
  ~18M after English filter (garbled/non-English removed)
  ~14M after crypto relevance filter
  ~8M after spam/ad filter (removes ~40% of crypto tweets - ads are pervasive)
  ~6M after quality filter (RT, too short, walls of text)
"""

import math
import re
from typing import Dict, Optional


def is_english_enough(text: str, threshold: float = 0.7) -> bool:
    """Reject non-English tweets by ASCII ratio. Zero external dependencies."""
    if not text or len(text) < 10:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > threshold


def is_crypto_relevant(text: str, asset: Optional[str] = None) -> bool:
    """Filter tweets to crypto-related content only."""
    text_lower = text.lower()

    ASSET_KEYWORDS = {
        "BTC": ["bitcoin", "btc", "#btc", "$btc", "satoshi", "sats", "\u20bf"],
        "ETH": ["ethereum", "eth", "#eth", "$eth", "vitalik", "erc20", "erc-20"],
        "SOL": ["solana", "sol", "#sol", "$sol", "jito"],
    }
    GENERAL_KEYWORDS = [
        "crypto", "blockchain", "defi", "nft", "exchange", "binance",
        "coinbase", "kraken", "altcoin", "stablecoin", "usdt", "usdc",
        "bull", "bear", "hodl", "moon", "pump", "dump", "whale",
        "liquidat", "margin", "leverage", "funding rate", "short", "long",
    ]

    if asset and asset in ASSET_KEYWORDS:
        if any(kw in text_lower for kw in ASSET_KEYWORDS[asset]):
            return True

    all_keywords = GENERAL_KEYWORDS.copy()
    for kws in ASSET_KEYWORDS.values():
        all_keywords.extend(kws)

    return any(kw in text_lower for kw in all_keywords)


def is_quality_tweet(text: str) -> bool:
    """Filter out low-quality tweets."""
    if not text:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    if len(text) > 500:
        return False
    if text.startswith("RT "):
        return False
    if text.count("http") > 2:
        return False
    if text.count("#") > 5:
        return False
    if len(set(words)) / len(words) < 0.3:
        return False
    return True


def clean_tweet_text(text: str) -> str:
    """Normalize tweet text for LLM consumption."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_spam_or_ad(text: str) -> bool:
    """Detect promotional/spam tweets. Returns True if spam."""
    text_lower = text.lower()

    PROMO_KEYWORDS = [
        'new project', 'launching soon', 'presale', 'pre-sale',
        'whitelist', 'white list', 'wl spot', 'guaranteed profit',
        'join now', 'join our', 'sign up now', 'register now',
        'limited spots', "don't miss", 'last chance',
        '100x', '1000x', '10000x', 'next 100x gem',
    ]
    AIRDROP_KEYWORDS = [
        'airdrop', 'air drop', 'giveaway', 'give away', 'free tokens',
        'free mint', 'free nft', 'claim now', 'claim your',
        'drop wallet', 'send wallet', 'wallet address',
    ]
    ENGAGEMENT_BAIT = [
        'follow and retweet', 'follow + rt', 'like and retweet',
        'tag 3 friends', 'tag three', 'must follow',
        'follow me for', 'follow us for',
    ]
    SCAM_PATTERNS = [
        'send me', 'dm me for', 'dm for', 'guaranteed return',
        'double your', 'investment opportunity',
        'passive income', 'make money fast', 'financial freedom',
        'telegram group', 't.me/', 'discord.gg/',
    ]

    all_spam = PROMO_KEYWORDS + AIRDROP_KEYWORDS + ENGAGEMENT_BAIT + SCAM_PATTERNS
    if any(kw in text_lower for kw in all_spam):
        return True

    if re.search(r'send\s+\d+\s*(eth|btc|sol|bnb)', text_lower):
        return True
    if re.search(r'\b\d+x\s+(return|profit|gain)', text_lower):
        return True

    # Excessive caps + exclamation (typical promo style)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count('!')
    if caps_ratio > 0.5 and excl_count > 3:
        return True

    # Too many emojis (promo tweets are emoji-heavy)
    emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
    if emoji_count > 5:
        return True

    return False


def full_filter_pipeline(text: str, asset: Optional[str] = None) -> Optional[str]:
    """Apply all filters. Returns cleaned text or None if rejected."""
    if not is_english_enough(text):
        return None
    if not is_crypto_relevant(text, asset):
        return None
    if is_spam_or_ad(text):
        return None
    if not is_quality_tweet(text):
        return None
    return clean_tweet_text(text)


# ═══════════════════════════════════════════════════════════════
# Engagement-weighted sampling
# ═══════════════════════════════════════════════════════════════

def compute_tweet_weight(row: Dict, method: str = "log_engagement") -> float:
    """
    Compute sampling weight for a tweet based on engagement metrics.

    Higher weight = more likely to be selected in calibration sample.
    This is NOT a sentiment weight - it's a data quality/importance signal.
    """
    followers = float(row.get('user_followers', row.get('followers_count', 0)) or 0)
    likes = float(row.get('favorites', row.get('likes', row.get('favorite_count', 0))) or 0)
    retweets = float(row.get('retweets', row.get('retweet_count', 0)) or 0)
    verified = bool(row.get('user_verified', row.get('verified', False)))

    if method == "log_engagement":
        engagement = likes + retweets * 2
        engagement_score = math.log1p(engagement)
        follower_score = math.log1p(followers)

        rate_bonus = 0.0
        if followers > 100:
            engagement_rate = engagement / followers
            rate_bonus = min(engagement_rate * 10, 3.0)

        verified_bonus = 1.5 if verified else 0.0
        weight = 1.0 + engagement_score * 0.3 + follower_score * 0.1 + rate_bonus + verified_bonus
        return max(weight, 1.0)

    elif method == "follower_tier":
        if followers >= 100_000:    return 5.0
        elif followers >= 10_000:   return 3.0
        elif followers >= 1_000:    return 2.0
        elif followers >= 100:      return 1.5
        else:                       return 1.0

    return 1.0


# ═══════════════════════════════════════════════════════════════
# Known high-signal accounts (optional whitelist for calibration)
# ═══════════════════════════════════════════════════════════════
# Tweets from these get 2x weight boost in calibration sampling.
# For CALIBRATION SAMPLING ONLY, not for production filtering.

HIGH_SIGNAL_ACCOUNTS = {
    'glassnode', 'whale_alert', 'santimentfeed', 'ki_young_ju',
    'cryptoquant_com', 'coinglass_com', 'messaricrypto',
    'zerohedge', 'delaboracrypto', 'pentoshi', 'cryptocobain',
    'coinbase', 'kaboraken', 'binance', 'cz_binance',
    'vaboritalikbuterin', 'aabornatoly',
}