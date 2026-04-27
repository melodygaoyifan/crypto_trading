"""
chaos/harness.py — minimal chaos test framework for HMATS
===========================================================

[P113 (6/6) 2026-04-27] Deliberately injects production failure modes
that have actually hurt HMATS this session, then asserts the system
DEGRADES GRACEFULLY instead of crashing or silently corrupting state.

DESIGN GOALS:
1. **No real exchange calls** — all chaos via monkey-patch + fake data.
2. **Asserts on observable outcomes** — does the engine emit the
   right WARN log? does the right gate trigger? does state stay
   consistent?
3. **Each scenario is a single test function** — failures isolate
   to one scenario; debugging stays scoped.
4. **Pytest-native** — runs via `pytest tests/chaos/` like any other
   test, no separate runner.

WHAT WE'VE PROVEN HURTS PRODUCTION (recreated as scenarios):
  - NaN volatility silently passing leverage pullback (P94)
  - Empty Kraken orderbook → spread/depth math NaN (audit)
  - Kraken 429 with no Retry-After → silent retry storm (P38)
  - SIGKILL mid-state-write → corrupt JSON on restart (P85)
  - Userref scheme drift across deploys → order leak (P95)
  - cancel_order on already-gone order → ERROR spam (P98b)
  - self.config undefined → silent mock fallback (P101)
  - Stop size below Kraken min → unprotected position (P91)
  - Sub-50%-balance clamp → POSITION-DESYNC cascade (P93)

USAGE:
    pytest tests/chaos/                 # all scenarios
    pytest tests/chaos/ -v -s           # see WARN/CRITICAL output
    pytest tests/chaos/ -k nan          # only NaN-related scenarios
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest import mock


REPO = Path(__file__).resolve().parents[2]


@contextmanager
def captured_logs(level: int = logging.WARNING) -> Iterator[List[str]]:
    """Capture all log records at >= level into a list. Restores
    handlers on exit. Use:
        with captured_logs() as logs:
            run_thing()
        assert any("EXPECTED_PATTERN" in r for r in logs)
    """
    captured: List[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(self.format(record))

    handler = CaptureHandler(level=level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(level)
    root.addHandler(handler)
    try:
        yield captured
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


@contextmanager
def temp_state_dir() -> Iterator[Path]:
    """Provide a temp directory + cd into it. State writes land here
    so chaos scenarios don't touch real production state files."""
    with tempfile.TemporaryDirectory(prefix="hmats_chaos_") as tmp:
        tmp_path = Path(tmp)
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            yield tmp_path
        finally:
            os.chdir(cwd)


def assert_warn_in_logs(logs: List[str], pattern: str, context: str = ""):
    """Assert at least one captured log line matches pattern at WARN+."""
    matches = [r for r in logs if pattern in r and ("WARNING" in r or "ERROR" in r or "CRITICAL" in r)]
    assert matches, (
        f"Expected WARN/ERROR/CRITICAL log matching {pattern!r}, "
        f"none found.{(' ' + context) if context else ''}\n"
        f"Captured logs ({len(logs)}):\n  " + "\n  ".join(logs[:30])
    )


def assert_no_crash(callable_or_value):
    """Run a callable and assert it returns rather than raising."""
    if callable(callable_or_value):
        try:
            return callable_or_value()
        except Exception as e:
            raise AssertionError(
                f"Operation crashed instead of degrading gracefully: "
                f"{type(e).__name__}: {e}"
            )
    return callable_or_value


def write_corrupt_json(path: Path, content: Optional[str] = None) -> None:
    """Write a known-bad JSON file (default: truncated) for state-load
    chaos. Caller controls content to simulate specific corruption shapes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        # Truncated mid-write — simulates SIGKILL during atomic-rename gap
        content = '{"key": "value", "incomplete'
    path.write_text(content, encoding="utf-8")


@contextmanager
def patch_external_api(
    target: str,
    response: Any = None,
    side_effect: Any = None,
) -> Iterator[mock.MagicMock]:
    """Replace an external API call site with a mock. Use:
        with patch_external_api(
            'data_mgmt.feeds.kraken_futures_feed.fetch',
            response={'fundingRate': float('nan')}
        ):
            run_chaos_scenario()
    """
    with mock.patch(target, side_effect=side_effect, return_value=response) as m:
        yield m
