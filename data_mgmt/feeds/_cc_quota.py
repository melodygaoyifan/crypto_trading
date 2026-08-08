"""[P220] One CryptoCompare account, one quota — shared accounting for it.

MEASURED, from the account's own `/stats/rate/limit`:

    calls_made  hour 3 · day 43 · month 283 · total 39,223
    max_calls   hour 100 · day 100 · month 100

The binding limit is **100 calls per MONTH** — roughly 3 per day. Two separate
feeds spend it:

    cryptocompare_news_feed   1 call/fetch  (3 before P219)   -> ~180/month
    cryptocompare_onchain     2 calls/fetch (BTC + ETH)       -> ~360/month
                                                       total  ~540/month

So the system was asking for ~5.4x its allowance, and the structural fault is
that **neither feed knew the other existed**. Each had its own backoff, both
keyed off the same `CRYPTOCOMPARE_API_KEY`, so one could exhaust the month and
the other would keep hammering — then both would sit in independent 15-minute
backoffs while the real constraint (the month) had already been blown.

A per-feed rate limiter cannot express a per-ACCOUNT budget. This module is that
budget: one persisted counter, shared by every CryptoCompare caller, reserved
BEFORE the request rather than discovered afterwards via a 429.

DESIGN NOTES

* Persisted (`data/cc_quota_state.json`), because an in-RAM budget re-arms on
  every restart — P154's lesson, and restart-heavy failure modes are exactly
  when the budget matters most.
* Reserve-then-call. `try_consume()` is called BEFORE the HTTP request, so a
  refused call costs nothing. Counting after the fact would let a burst spend
  the month before anything noticed.
* The budget defaults BELOW the real cap (90 of 100). Headroom matters because
  the provider's own counter includes calls we cannot see — ad-hoc probes,
  another process on the same key — and because `/stats/rate/limit` itself
  consumes a call.
* Exhaustion is LOUD once per period, and callers fall back to cached data.
  Silent exhaustion would be indistinguishable from "the market has no news",
  which is the conflation this codebase keeps producing (P199/P216/P218).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("CCQuota")

# The provider's hard monthly cap is 100. Default below it: the provider counts
# calls this process cannot see, and a budget that exactly equals the cap will
# always discover the difference as a 429.
DEFAULT_MONTHLY_BUDGET = 90

_STATE_VERSION = "cc_quota_v1"


def _data_dir() -> str:
    return os.environ.get("HMATS_DATA_DIR", "data")


def _state_path() -> str:
    return os.path.join(_data_dir(), "cc_quota_state.json")


def _period_key(ts: Optional[float] = None) -> str:
    """Calendar month in UTC — the unit the provider bills in."""
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


class CryptoCompareQuota:
    """Process-wide monthly call budget for the shared CryptoCompare key."""

    def __init__(self, monthly_budget: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        env = os.environ.get("HMATS_CC_MONTHLY_BUDGET", "").strip()
        if monthly_budget is None and env:
            try:
                monthly_budget = int(env)
            except ValueError:
                logger.warning(f"[CC_QUOTA] bad HMATS_CC_MONTHLY_BUDGET={env!r}")
        self._budget = int(monthly_budget if monthly_budget is not None
                           else DEFAULT_MONTHLY_BUDGET)
        self._period = _period_key()
        self._used = 0
        self._by_caller: Dict[str, int] = {}
        self._exhausted_logged = False
        self._restore()

    # ----- persistence ----------------------------------------------------
    def _restore(self) -> None:
        try:
            p = _state_path()
            if not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as fh:
                st = json.load(fh)
            if st.get("state_version") != _STATE_VERSION:
                return
            if st.get("period") != self._period:
                # New month — the provider's counter reset, so ours does too.
                logger.info(
                    f"[CC_QUOTA] new period {self._period} "
                    f"(was {st.get('period')}) — budget reset")
                return
            self._used = int(st.get("used") or 0)
            self._by_caller = dict(st.get("by_caller") or {})
            logger.info(
                f"[CC_QUOTA] restored {self._used}/{self._budget} used "
                f"for {self._period}: {self._by_caller}")
        except Exception as e:  # noqa: silent-swallow — a bad state file must not break startup
            logger.warning(f"[CC_QUOTA] restore failed: {type(e).__name__}: {e}")

    def _persist(self) -> None:
        try:
            p = _state_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "state_version": _STATE_VERSION,
                    "period": self._period,
                    "used": self._used,
                    "budget": self._budget,
                    "by_caller": self._by_caller,
                    "saved_ts": time.time(),
                }, fh)
            os.replace(tmp, p)
        except Exception as e:  # noqa: silent-swallow — accounting must not break a tick
            logger.debug(f"[CC_QUOTA] persist failed: {type(e).__name__}: {e}")

    # ----- API ------------------------------------------------------------
    def _roll_period(self) -> None:
        now = _period_key()
        if now != self._period:
            logger.info(
                f"[CC_QUOTA] period rolled {self._period} -> {now}: "
                f"{self._used}/{self._budget} used last month")
            self._period = now
            self._used = 0
            self._by_caller = {}
            self._exhausted_logged = False
            self._persist()

    def try_consume(self, n: int = 1, caller: str = "unknown") -> bool:
        """Reserve `n` calls. False = do NOT make the request.

        Reserved BEFORE the HTTP call so a refusal is free. Counting afterwards
        would let a burst spend the month before anything noticed.
        """
        if n <= 0:
            return True
        with self._lock:
            self._roll_period()
            if self._used + n > self._budget:
                if not self._exhausted_logged:
                    self._exhausted_logged = True
                    logger.warning(
                        f"[CC_QUOTA] EXHAUSTED for {self._period}: "
                        f"{self._used}/{self._budget} calls used, {caller} "
                        f"wanted {n}. Further CryptoCompare requests are "
                        f"REFUSED until the month rolls — callers fall back to "
                        f"cached data. This is a BUDGET decision, not a market "
                        f"condition. Raise HMATS_CC_MONTHLY_BUDGET only if the "
                        f"account's real cap has changed. Spend so far: "
                        f"{self._by_caller}")
                return False
            self._used += n
            self._by_caller[caller] = self._by_caller.get(caller, 0) + n
            self._persist()
            return True

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._roll_period()
            return {
                "period": self._period,
                "used": self._used,
                "budget": self._budget,
                "remaining": max(0, self._budget - self._used),
                "by_caller": dict(self._by_caller),
            }


_QUOTA: Optional[CryptoCompareQuota] = None
_QUOTA_LOCK = threading.Lock()


def get_cc_quota() -> CryptoCompareQuota:
    """Process-wide singleton — the point is that every caller shares it."""
    global _QUOTA
    if _QUOTA is None:
        with _QUOTA_LOCK:
            if _QUOTA is None:
                _QUOTA = CryptoCompareQuota()
    return _QUOTA
