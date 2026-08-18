"""
[P304] Vendor-SDK log hygiene — stop a third-party error page reaching Discord.

THE INCIDENT (2026-08-18 10:54:29). Coinbase returned a single transient 502
and the `coinbase.RESTClient` logger emitted it at **ERROR** with the venue's
entire HTML error page attached — 27 lines of CSS. `DiscordLogHandler` is
installed on the ROOT logger at `min_level=ERROR`, so that blob went straight
to the operator's Discord.

Three things were wrong with that, none of them about the 502 itself:

  1. **It was already handled, correctly.** The very next line was ours, at
     WARNING, and it did the right thing — refused the wrong-denomination FCM
     subset and served last-known equity (P153/P265a):

         [COINBASE_SLEEVE] portfolio equity UNAVAILABLE; futures summary shows
         FCM-subset $445.51 (WRONG denomination, P153/P265 — NOT substituted).
         Serving last-known true equity $10,838.50

     An ERROR alert for a condition the system absorbed by design is the
     P202/P240 shape: its only resolutions are theatre or ignoring it.

  2. **It was unreadable.** A multi-KB HTML page is not an alert. Whatever
     signal the 502 carried was buried in `-webkit-text-size-adjust`.

  3. **It was indiscriminate.** A 5xx is the venue's problem and we have
     fallbacks; a 4xx is OURS (bad key, missing permission, malformed
     request) and must stay loud. One severity for both loses that.

WHAT THIS DOES — and, as importantly, what it does NOT do:

  * TRUNCATES the message and strips any HTML body. Always.
  * DEMOTES an isolated 5xx from ERROR to WARNING, because the fallbacks are
    real and tested.
  * KEEPS every 4xx at ERROR. Those are our bugs.
  * RE-ESCALATES a SUSTAINED 5xx back to ERROR (>= SUSTAINED_5XX_COUNT inside
    SUSTAINED_WINDOW_SEC). A venue that is genuinely down IS actionable — we
    cannot trade — so quieting the blip must not also quiet the outage.
  * NEVER drops a record. `filter()` returns True on every path; the only
    thing that ever changes is the level and the text.

The window is measured from `record.created`, the LogRecord's own timestamp,
so this needs no clock of its own and reports the time the event actually
happened.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Deque, Optional

# Cap for any single vendor log line. Long enough for a real message with a
# status code and a reason; far too short for an error page.
MAX_MESSAGE_CHARS = 300

# A 5xx burst this dense stops being a blip and becomes an outage.
SUSTAINED_5XX_COUNT = 5
SUSTAINED_WINDOW_SEC = 300.0

_STATUS_RE = re.compile(r"\b([45]\d{2})\b")
_HTML_START_RE = re.compile(r"<\s*(?:!doctype|html|head|body|style|meta)\b",
                            re.IGNORECASE)


def strip_html_body(msg: str) -> str:
    """Cut a vendor error page out of a log message.

    Keeps everything before the markup — that is where the status code and
    reason live — then collapses whitespace and caps the length. A message
    with no markup is only length-capped.
    """
    if not msg:
        return msg
    m = _HTML_START_RE.search(msg)
    if m:
        msg = msg[:m.start()].rstrip()
        if not msg:
            msg = "(vendor returned an HTML error page)"
        msg += " [HTML body stripped]"
    # Collapse newlines so a multi-line body can never become multi-line output.
    msg = " ".join(msg.split())
    if len(msg) > MAX_MESSAGE_CHARS:
        msg = msg[:MAX_MESSAGE_CHARS - 3] + "..."
    return msg


def http_status_in(msg: str) -> Optional[int]:
    """The 4xx/5xx status a vendor message is reporting, if any."""
    if not msg:
        return None
    m = _STATUS_RE.search(msg)
    return int(m.group(1)) if m else None


class VendorHTTPLogFilter(logging.Filter):
    """Truncate vendor HTTP noise; keep the severity honest.

    Attach to a third-party logger (we cannot edit the SDK, but we own its
    logger). Stateful only in the 5xx timestamps it needs to tell a blip from
    an outage.
    """

    def __init__(self, name: str = "",
                 max_chars: int = MAX_MESSAGE_CHARS,
                 sustained_count: int = SUSTAINED_5XX_COUNT,
                 sustained_window_sec: float = SUSTAINED_WINDOW_SEC):
        super().__init__(name)
        self._max_chars = int(max_chars)
        self._sustained_count = int(sustained_count)
        self._window = float(sustained_window_sec)
        self._recent_5xx: Deque[float] = deque()

    # -- helpers -----------------------------------------------------------
    def _note_5xx(self, when: float) -> int:
        """Record a 5xx at `when` and return how many are inside the window."""
        self._recent_5xx.append(when)
        cutoff = when - self._window
        while self._recent_5xx and self._recent_5xx[0] < cutoff:
            self._recent_5xx.popleft()
        return len(self._recent_5xx)

    # -- logging.Filter ----------------------------------------------------
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: silent-swallow — a record we cannot render must still be emitted, unchanged
            return True

        status = http_status_in(msg)
        cleaned = strip_html_body(msg)
        if len(cleaned) > self._max_chars:
            cleaned = cleaned[:self._max_chars - 3] + "..."

        if status is not None and 500 <= status <= 599 and \
                record.levelno >= logging.ERROR:
            n = self._note_5xx(float(getattr(record, "created", 0.0) or 0.0))
            if n >= self._sustained_count:
                cleaned += (f" — SUSTAINED ({n} 5xx in "
                            f"{int(self._window)}s), the venue is not merely "
                            f"blipping")
            else:
                # Isolated server-side failure: our callers have tested
                # fallbacks (P265a refuses the wrong-denomination substitute
                # and serves last-known equity), so this is not an ERROR.
                record.levelno = logging.WARNING
                record.levelname = "WARNING"
                cleaned += " — transient venue-side failure; caller fallbacks apply"

        # Rewrite the payload so nothing downstream (Discord included) can
        # re-expand the original args into the blob we just removed.
        record.msg = cleaned
        record.args = ()
        return True


# Loggers known to attach vendor HTML/large bodies to their messages.
VENDOR_LOGGERS = ("coinbase", "coinbase.RESTClient")


def install_vendor_log_filters(logger_names=VENDOR_LOGGERS) -> int:
    """Attach the filter to each vendor logger. Idempotent.

    Returns how many loggers were newly filtered. Safe to call more than once
    (a second call adds nothing), and never raises: log hygiene must not be
    able to stop the engine starting.
    """
    installed = 0
    for name in logger_names:
        try:
            lg = logging.getLogger(name)
            if any(isinstance(f, VendorHTTPLogFilter) for f in lg.filters):
                continue
            lg.addFilter(VendorHTTPLogFilter())
            installed += 1
        except Exception as e:  # noqa: silent-swallow — logged below; a filter that fails to install must not stop startup
            logging.getLogger(__name__).warning(
                "[P304] could not filter vendor logger %s: %s: %s",
                name, type(e).__name__, e)
    return installed


__all__ = [
    "VendorHTTPLogFilter", "install_vendor_log_filters", "strip_html_body",
    "http_status_in", "VENDOR_LOGGERS", "MAX_MESSAGE_CHARS",
    "SUSTAINED_5XX_COUNT", "SUSTAINED_WINDOW_SEC",
]
