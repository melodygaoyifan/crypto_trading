"""
Standalone Kraken credentials + permissions diagnostic with adaptive
nonce-baseline recovery.

Run BEFORE redeploying after a key rotation OR after a stuck-nonce
incident. Probes four permissions independently:
  - Query Funds                (account_sync, balance)
  - Query Open Orders & Trades (positions sync)
  - Cancel/Close Orders        (dead-man switch, emergency cancel)

If a permission probe fails with `EAPI:Invalid nonce`, the script
ESCALATES the nonce by exponential jumps until it clears Kraken's
tracker, then PERSISTS the discovered baseline to
`data/kraken_nonce_state.json`. The next time the engine starts, it
loads that state and avoids the same failure.

This is the cure for "fresh process still gets Invalid nonce" — Kraken
holds a per-key nonce baseline that survives our process restarts. The
in-memory ratchet alone can't recover from it.

Usage:
    python -X utf8 scripts/kraken_credentials_check.py
    python -X utf8 scripts/kraken_credentials_check.py --reset-nonce-state
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

try:
    import ccxt  # type: ignore
except ImportError:
    print("FAIL: ccxt not installed. pip install ccxt")
    sys.exit(1)


NONCE_STATE_PATH = REPO_ROOT / "data" / "kraken_nonce_state.json"

# Escalation ladder. Each step is total seconds AHEAD of clock-time the
# nonce gets bumped on failure. Scales fast so we cover plausible drift
# (clock skew → minutes → hours → multi-day) before giving up.
NONCE_BUMP_LADDER_SEC = [60, 300, 1800, 7200, 21600, 86400, 259200]  # 1m, 5m, 30m, 2h, 6h, 1d, 3d


def _key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _load_state() -> dict:
    if not NONCE_STATE_PATH.exists():
        return {}
    try:
        with open(NONCE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"  WARN  could not parse {NONCE_STATE_PATH}: {e}")
        return {}


def _save_state(state: dict) -> None:
    NONCE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = NONCE_STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, separators=(",", ":"))
    os.replace(tmp, NONCE_STATE_PATH)


def _make_exchange_with_nonce_floor(api_key: str, api_secret: str, floor_ms: int):
    """Build a ccxt Kraken exchange whose nonce() is monotonically >= floor_ms.

    Each call returns max(local_ms, last_issued + 1). The shared cell lets
    us read the highest nonce we've issued from the outside.
    """
    cell = {"v": int(floor_ms)}
    exchange = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })

    def _nonce() -> int:
        now_ms = int(time.time() * 1000)
        cell["v"] = max(cell["v"] + 1, now_ms)
        return cell["v"]

    exchange.nonce = _nonce  # type: ignore[assignment]
    return exchange, cell


def _try_with_escalation(label: str, fn, exchange, cell, fingerprint: str) -> tuple[bool, int]:
    """Try fn(exchange). On Invalid nonce, escalate the floor and retry.

    Returns (success, highest_nonce_used). On success, persist the highest
    nonce to disk so the engine boots above it.
    """
    last_nonce = cell["v"]
    try:
        fn()
        last_nonce = cell["v"]
        print(f"  OK    {label}  (nonce={last_nonce})")
        return True, last_nonce
    except Exception as e:
        if "Invalid nonce" not in str(e):
            print(f"  ERR   {label}: {type(e).__name__}: {e}")
            return False, cell["v"]

    print(f"  FAIL  {label}: Invalid nonce at clock-time. Escalating...")
    base_ms = int(time.time() * 1000)
    for step_idx, jump_sec in enumerate(NONCE_BUMP_LADDER_SEC, start=1):
        cell["v"] = base_ms + (jump_sec * 1000)
        try:
            fn()
            last_nonce = cell["v"]
            print(
                f"        step {step_idx}: bumped +{jump_sec}s  "
                f"(nonce={last_nonce}) -> CLEARED"
            )
            return True, last_nonce
        except Exception as e:
            if "Invalid nonce" not in str(e):
                print(
                    f"        step {step_idx}: bumped +{jump_sec}s  "
                    f"non-nonce error: {type(e).__name__}: {e}"
                )
                return False, cell["v"]
            print(f"        step {step_idx}: bumped +{jump_sec}s  still Invalid nonce")

    print(f"  FAIL  {label}: nonce stuck even after +{NONCE_BUMP_LADDER_SEC[-1]}s. "
          f"Manual key rotation required.")
    return False, cell["v"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-nonce-state",
        action="store_true",
        help="Delete persisted nonce state for the current key (use after Kraken-side key rotation).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
    api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()

    print(f"KRAKEN_API_KEY: {'set, len=' + str(len(api_key)) if api_key else 'NOT SET'}")
    print(f"KRAKEN_API_SECRET: {'set, len=' + str(len(api_secret)) if api_secret else 'NOT SET'}")
    if not api_key or not api_secret:
        print("\nFAIL: missing credentials in environment. Edit .env and re-source.")
        return 2

    fingerprint = _key_fingerprint(api_key)
    print(f"Key fingerprint: {fingerprint}")

    state = _load_state()
    if args.reset_nonce_state:
        if fingerprint in state:
            del state[fingerprint]
            _save_state(state)
            print(f"  Reset nonce state for fp={fingerprint}.")
        else:
            print(f"  No persisted state for fp={fingerprint} — nothing to reset.")
        return 0

    persisted = int(state.get(fingerprint, {}).get("highest_nonce_ms", 0))
    floor_ms = max(persisted, int(time.time() * 1000))
    print(f"Loaded persisted nonce floor: {persisted} (using floor={floor_ms})")

    exchange, cell = _make_exchange_with_nonce_floor(api_key, api_secret, floor_ms)

    try:
        exchange.load_time_difference()
        print(f"  OK    timeDifference: {exchange.options.get('timeDifference', 0)}ms")
    except Exception as e:
        print(f"  WARN  load_time_difference failed: {e}")

    print("\nPermission probes (with adaptive nonce escalation):")
    results: dict[str, bool] = {}
    highest_nonce = floor_ms

    for label, perm, fn in [
        ("Query Funds", "fetch_balance", lambda: exchange.fetch_balance()),
        ("Query Open Orders", "fetch_open_orders", lambda: exchange.fetch_open_orders()),
        ("Cancel/Close Orders", "cancel_all_orders_after(0)", lambda: exchange.cancel_all_orders_after(0)),
    ]:
        ok, used = _try_with_escalation(f"{perm}  ({label})", fn, exchange, cell, fingerprint)
        results[label] = ok
        highest_nonce = max(highest_nonce, used)

    # Persist whatever the highest-issued nonce ended up being. Even on
    # partial failure, we want to keep ground we won't lose on next run.
    if highest_nonce > persisted:
        state[fingerprint] = {
            "highest_nonce_ms": highest_nonce,
            "updated_at": int(time.time()),
        }
        _save_state(state)
        print(f"\n  Persisted nonce floor for fp={fingerprint}: {highest_nonce}")
    else:
        print(f"\n  Persisted nonce floor unchanged ({persisted})")

    print()
    if all(results.values()):
        print("PASS — key has all required permissions for LIVE mode.")
        return 0
    missing = [k for k, v in results.items() if not v]
    print(f"FAIL — missing permissions or unrecoverable nonce: {missing}")
    print(
        "If 'Cancel/Close Orders' failed: open Kraken > Settings > API, edit the key, "
        "enable that permission, save, and re-run.\n"
        "If a probe failed with 'nonce stuck even after +3 days': the key has a "
        "Kraken-side nonce baseline beyond reasonable bounds. Rotate the key on "
        "Kraken (creates a fresh nonce window), then re-run with --reset-nonce-state."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
