"""
[FRINGE-PANIC] Keyword Matrix for Fringe Panic Detector
========================================================
8 keyword groups, each independently tracked for z-score spikes.
Multi-group simultaneous activation is itself a signal.

To add keywords: just edit the lists below. No code changes needed.
"""

PANIC_KEYWORD_GROUPS = {
    # GROUP 1: Financial system collapse
    # Historical: SVB (2023.03) - "bank run" 10x spike 48H before collapse
    "financial_collapse": {
        "keywords": [
            "bank run", "bank collapse", "banking crisis",
            "financial crisis", "systemic risk", "bank failure",
            "FDIC insolvency", "deposit freeze", "bail-in",
            "credit freeze", "liquidity crisis",
        ],
        "x_query": (
            '("bank run" OR "bank collapse" OR "banking crisis" '
            'OR "financial crisis" OR "systemic risk" OR "bail-in" '
            'OR "credit freeze") -is:retweet lang:en'
        ),
        "weight": 0.15,
        "baseline_daily_tweets": 500,
    },

    # GROUP 2: Geopolitical crisis
    # Historical: Russia-Ukraine - "world war 3" 5x spike 48H before
    "geopolitical_crisis": {
        "keywords": [
            "world war 3", "nuclear war", "martial law",
            "military strike", "iran war", "taiwan invasion",
            "missile launch", "declaration of war", "NATO article 5",
            "strait of hormuz", "oil embargo",
        ],
        "x_query": (
            '("world war 3" OR "nuclear war" OR "martial law" '
            'OR "military strike" OR "taiwan invasion" '
            'OR "strait of hormuz") -is:retweet lang:en'
        ),
        "weight": 0.15,
        "baseline_daily_tweets": 800,
    },

    # GROUP 3: China/CCP crisis narrative
    # Historical: Evergrande 72H before, "China collapse" 3x
    "china_crisis": {
        "keywords": [
            "China collapse", "China bank run", "China economic crisis",
            "China debt crisis", "Evergrande default", "China property crash",
            "shadow banking China", "China deflation",
            "CCP collapse", "Xi Jinping coup",
            "PBOC emergency", "yuan devaluation", "yuan crash",
            "capital flight China", "China bank freeze",
            "Taiwan strait crisis", "US China war",
        ],
        "x_query": (
            '("China collapse" OR "China bank run" OR "CCP collapse" '
            'OR "yuan crash" OR "PBOC emergency" OR "China debt crisis" '
            'OR "China property crash" OR "capital flight China" '
            'OR "Taiwan strait" OR "yuan devaluation") -is:retweet'
        ),
        "weight": 0.15,
        "baseline_daily_tweets": 300,
    },

    # GROUP 4: Anti-establishment / revolution narrative
    # Slow background indicator - when it spikes, social tension at tipping point
    "anti_establishment": {
        "keywords": [
            "eat the rich", "class war", "revolution now",
            "overthrow capitalism", "general strike",
            "wealth inequality crisis", "billionaire tax",
            "late stage capitalism", "abolish fed",
            "market rigged", "stock market manipulation",
        ],
        "x_query": (
            '("eat the rich" OR "class war" OR "revolution now" '
            'OR "general strike" OR "abolish fed" '
            'OR "market rigged" OR "late stage capitalism") '
            '-is:retweet lang:en'
        ),
        "weight": 0.08,
        "baseline_daily_tweets": 1200,
    },

    # GROUP 5: Crypto-specific doom
    # Historical: FTX - "exchange insolvency" 20x in 24H
    "crypto_doom": {
        "keywords": [
            "exchange hack", "crypto crash", "bitcoin crash",
            "tether depeg", "USDT depeg", "stablecoin collapse",
            "exchange insolvency", "crypto rug pull",
            "SEC crypto crackdown", "crypto ban",
            "Binance collapse", "exchange frozen withdrawals",
        ],
        "x_query": (
            '("exchange hack" OR "tether depeg" OR "USDT depeg" '
            'OR "exchange insolvency" OR "crypto ban" '
            'OR "frozen withdrawals") -is:retweet lang:en'
        ),
        "weight": 0.15,
        "baseline_daily_tweets": 400,
    },

    # GROUP 6: Dollar/Treasury crisis
    # Historical: 2023 debt ceiling - "US default" 5x
    "dollar_crisis": {
        "keywords": [
            "dollar collapse", "US default", "debt ceiling crisis",
            "treasury crisis", "hyperinflation",
            "de-dollarization", "BRICS currency",
            "petrodollar dead", "fed audit",
            "US credit downgrade", "bond market crash",
        ],
        "x_query": (
            '("dollar collapse" OR "US default" OR "debt ceiling" '
            'OR "hyperinflation" OR "de-dollarization" '
            'OR "treasury crisis") -is:retweet lang:en'
        ),
        "weight": 0.12,
        "baseline_daily_tweets": 600,
    },

    # GROUP 7: Conspiracy canary
    # When conspiracy communities shift from vague to specific = info leak
    # Highest noise - needs cross-validation with other groups
    "conspiracy_canary": {
        "keywords": [
            "government coverup", "insider trading exposed",
            "banking cartel", "market manipulation proof",
            "whistleblower", "classified leak",
            "deep state", "false flag",
            "controlled demolition economy",
            "planned collapse", "great reset",
        ],
        "x_query": (
            '("government coverup" OR "insider trading exposed" '
            'OR "whistleblower" OR "classified leak" '
            'OR "planned collapse" OR "great reset") '
            '-is:retweet lang:en'
        ),
        "weight": 0.08,
        "baseline_daily_tweets": 2000,
    },

    # GROUP 8: Black swan / doomsday narrative
    "black_swan": {
        "keywords": [
            "black swan event", "economic collapse imminent",
            "market meltdown", "flash crash",
            "circuit breaker triggered", "trading halted",
            "panic selling", "capitulation",
            "great depression 2", "everything bubble",
        ],
        "x_query": (
            '("black swan" OR "economic collapse" OR "market meltdown" '
            'OR "circuit breaker" OR "trading halted" '
            'OR "panic selling" OR "capitulation") '
            '-is:retweet lang:en'
        ),
        "weight": 0.12,
        "baseline_daily_tweets": 350,
    },
}

# Chinese keywords - tracked separately, merged into china_crisis group
CHINA_KEYWORDS_ZH = {
    "x_query": (
        '("银行挤兑" OR "人民币暴跌" OR "经济崩溃" '
        'OR "房地产崩盘" OR "资本外逃" OR "央行紧急" '
        'OR "习近平下台" OR "共产党危机") -is:retweet'
    ),
    "google_trends_keywords": [
        "银行挤兑", "人民币暴跌", "经济崩溃",
        "房地产崩盘", "资本外逃",
    ],
    "weight": 0.10,
}

# Google Trends representative keywords (pytrends max 5 per batch)
TRENDS_KEYWORDS_BATCH1 = [
    "bank run", "world war 3", "China collapse",
    "crypto crash", "dollar collapse",
]
TRENDS_KEYWORDS_BATCH2 = [
    "market crash", "hyperinflation",
    "economic collapse", "trading halted",
]

# Reddit panic subreddits
PANIC_SUBREDDITS = [
    "conspiracy",
    "collapse",
    "preppers",
    "economicCollapse",
    "wallstreetbets",
    "geopolitics",
]