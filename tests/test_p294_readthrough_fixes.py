"""
================================================================================
HMATS [P294] - read-through fixes
================================================================================

Covers the defects found reading the (then-uncommitted) P293 batch end to end:

  1. the whale-filter ledger pooled into the ma_filter exam
     -> tests/test_p293d_whale_options.py (distinctness + scorer premise)
  2. the whale stash had no reset despite its comment
     -> tests/test_p293d_whale_options.py (reset ordering)
  3. a transfer during downtime was invisible to the flow detector
     -> tests/test_p293e_sentiment_and_efficiency.py (persistence round-trip)
  4. the ENCODING trap                          -> here
  5. advisory feeds counted toward a CRITICAL   -> here
  6. netflow per-symbol carry + mock provenance -> here
================================================================================
"""

import ast
import io
import os
from pathlib import Path

import pytest

# [P311] Guard pins go through assert_guard_live: a plain substring
# assertion survives `if False and <condition>`, which is how P234,
# P251 and P307 each shipped a neutered guard that still read as pinned.
from tests._guard_pins import assert_guard_live  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"venv", "archive", "__pycache__", ".git", "node_modules",
             "models", "training_data", "dashboard"}


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# =============================================================================
# 4 - the encoding trap
# =============================================================================

def _mode_is_binary(call: ast.Call) -> bool:
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
            and isinstance(call.args[1].value, str):
        return "b" in call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return "b" in kw.value.value
    return False


def _has_encoding(call: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in call.keywords)


def scan_source(src: str):
    """Return [(lineno, what)] for text reads/writes with no explicit encoding.

    AST-based on purpose: a regex matches `open(` inside string literals and
    inside `zipfile.open` / `tarfile.open`, which are BINARY and reject the
    kwarg. The mechanical pass that fixed this class made exactly those two
    mistakes - one of them on the live data path - so the guard that keeps it
    fixed must not repeat them.
    """
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # builtin open(...) only - NOT zf.open / archive.open / gzip.open
        if isinstance(f, ast.Name) and f.id == "open":
            if not _mode_is_binary(node) and not _has_encoding(node) \
                    and (node.args or node.keywords):
                out.append((node.lineno, "open()"))
        elif isinstance(f, ast.Attribute) and f.attr in ("read_text",
                                                         "write_text"):
            if not _has_encoding(node):
                out.append((node.lineno, "." + f.attr + "()"))
    return out


TEXTY_SUBPROCESS = {"run", "check_output", "check_call", "call", "Popen"}


def scan_subprocess(src: str):
    """Return [(lineno, name)] for text-mode subprocess calls with no encoding.

    The OTHER half of the same trap: `subprocess.run(..., text=True)` decodes
    the child's output with `locale.getpreferredencoding()`, so a child that
    prints an em-dash gives the parent `stderr=None` on a GBK box. Fixing the
    read sites alone left 24 failures — every one a test asserting on the
    output of a script it had spawned.
    """
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.attr if isinstance(f, ast.Attribute)
                else (f.id if isinstance(f, ast.Name) else None))
        if name not in TEXTY_SUBPROCESS:
            continue
        kws = {k.arg for k in node.keywords}
        texty = any(
            k.arg in ("text", "universal_newlines")
            and isinstance(k.value, ast.Constant) and k.value.value is True
            for k in node.keywords)
        if texty and "encoding" not in kws:
            out.append((node.lineno, name))
    return out


def _repo_py_files():
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


class TestEncodingIsAlwaysExplicit:
    """The whole local suite was red on the operator's own machine.

    MEASURED 2026-08-17: `pytest tests/` gave 45 failed + 14 errors, every
    one `UnicodeDecodeError: 'gbk' codec` - files read with the Windows
    locale codec because no `encoding=` was passed. Under `-X utf8` the same
    tree gave 4,895 passed / 0 failed.

    That is P194's lesson inverted: CI (Linux, UTF-8) was green while the
    operator's box was red, so the local suite carried no signal and a real
    failure would have been indistinguishable from the noise. It reached the
    guards for P164 (the wavelet leak), P232, P285, P287, P291, P230, P213.

    There is no pytest.ini or pyproject.toml in this repo to pin an
    interpreter flag, and a flag would only fix the machines that remember
    it. The durable fix is the one P171 applied to the scanners: state the
    encoding at the read site.
    """

    def test_no_encoding_less_text_reads_anywhere(self):
        offenders = []
        for p in _repo_py_files():
            try:
                src = io.open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue    # non-UTF-8 file; not readable either way
            for lineno, what in scan_source(src):
                offenders.append(str(p.relative_to(REPO)) + ":" +
                                 str(lineno) + " " + what)
        assert not offenders, (
            "these read/write text with the AMBIENT locale codec, so they "
            "fail on any non-UTF-8 machine:\n  " + "\n  ".join(offenders[:40])
        )

    def test_no_text_mode_subprocess_without_encoding(self):
        """The child emits UTF-8; the parent must not decode it as GBK."""
        offenders = []
        for p in _repo_py_files():
            try:
                src = io.open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, name in scan_subprocess(src):
                offenders.append(str(p.relative_to(REPO)) + ":" +
                                 str(lineno) + " " + name + "(text=True)")
        assert not offenders, (
            "these decode a child process's output with the AMBIENT locale "
            "codec, so `.stdout` comes back None on a non-UTF-8 machine:\n  "
            + "\n  ".join(offenders[:40])
        )

    def test_the_subprocess_scanner_bites(self):
        """[P174] anti-vacuity for the second half."""
        bad = "subprocess.run(cmd, text=True)\n"
        assert [n for _, n in scan_subprocess(bad)] == ["run"]
        ok = ('subprocess.run(cmd, text=True, encoding="utf-8")\n'
              "subprocess.run(cmd)\n"
              "subprocess.run(cmd, capture_output=True)\n")
        assert scan_subprocess(ok) == []

    def test_the_gate_itself_is_covered(self):
        """`tools/ci_check_invariants.py` shells out to every scanner with
        text=True. At HEAD it CRASHED on this machine with the same decode
        error before running a single check — so the repo's own gate was
        unrunnable here, which is how the type baseline's environment
        offset stayed unexamined."""
        src = _src(REPO / "tools" / "ci_check_invariants.py")
        assert 'encoding="utf-8"' in src

    def test_the_scanner_actually_bites(self):
        """[P174] anti-vacuity: a guard that finds nothing on bad input is
        indistinguishable from a clean tree."""
        bad = "open('x')\nPath('y').read_text()\nopen('z', 'w')\n"
        found = {w for _, w in scan_source(bad)}
        assert found == {"open()", ".read_text()"}, found

    def test_the_scanner_does_not_flag_binary_or_member_opens(self):
        """The three false positives the mechanical pass actually produced:
        zipfile/tarfile members reject `encoding` outright, and one of them
        (`archive.open` in market_data_pipeline) is on the live data path."""
        ok = (
            "open('x', 'rb')\n"
            "open('x', mode='rb')\n"
            "zf.open(name)\n"
            "archive.open(members[0])\n"
            "gzip.open(p)\n"
            "open('x', encoding='utf-8')\n"
        )
        assert scan_source(ok) == []

    def test_no_binary_open_carries_an_encoding(self):
        """The inverse error, and the mechanical pass really made it.

        `path.open("rb", encoding="utf-8")` raises `ValueError: binary mode
        doesn't take an encoding argument` at CALL time, not import time, so
        it survives a compile-everything check and only shows up when that
        branch runs. The regex pass planted one in
        `execution/learned_execution_policy.py` — model loading — and it was
        the single remaining suite failure after everything else was green.

        Note this scans ANY receiver (`path.open`, not just the builtin),
        which is exactly the opposite of `scan_source` above: there, a
        non-builtin receiver means "leave it alone"; here it means "check
        it", because Path.open takes the mode as its FIRST argument.
        """
        offenders = []
        for p in _repo_py_files():
            try:
                src = io.open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                nm = (f.attr if isinstance(f, ast.Attribute)
                      else (f.id if isinstance(f, ast.Name) else None))
                if nm != "open" or not _has_encoding(n):
                    continue
                mode = None
                if isinstance(f, ast.Attribute) and n.args and \
                        isinstance(n.args[0], ast.Constant) and \
                        isinstance(n.args[0].value, str):
                    mode = n.args[0].value          # Path.open(mode, ...)
                if len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) \
                        and isinstance(n.args[1].value, str):
                    mode = n.args[1].value          # open(path, mode, ...)
                for k in n.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        mode = k.value.value
                if mode and "b" in mode:
                    offenders.append(str(p.relative_to(REPO)) + ":" +
                                     str(n.lineno))
        assert not offenders, (
            "binary mode does not take an encoding argument; these raise "
            "ValueError when the branch runs:\n  " + "\n  ".join(offenders)
        )

    def test_zip_and_tar_member_opens_stayed_binary(self):
        """Regression pin for the three that were broken and reverted."""
        for rel, needle in [
            ("data_mgmt/market_data_pipeline.py", "archive.open(members[0])"),
            ("training/download_more_data.py", "zf.open(csv_name)"),
            ("training/get_data.py", "zf.open(csv_name)"),
        ]:
            src = _src(REPO / rel)
            assert needle in src, rel + ": member open lost its binary form"
            assert needle[:-1] + ", encoding=" not in src, (
                rel + ": encoding= on a zip/tar member raises at runtime"
            )


# =============================================================================
# 5 - advisory feeds must not escalate to CRITICAL
# =============================================================================

class TestFeedDegradationCountsOnlyLiveFeeds:
    """P293 added six feeds to a gather whose >=2/>=3 thresholds were
    calibrated when every member fed a live signal. Five of the six have
    consumers behind default-OFF flags, so counting them produced a Discord
    CRITICAL for something the operator can neither act on nor be harmed by
    - the P202/P240 shape."""

    def test_advisory_roster_is_declared_and_gci_follows_its_flag(self):
        src = _src(REPO / "main.py")
        assert '_advisory_feeds = {"DERIBIT", "EXCH_NETFLOW", "FNG_HISTORY"}' \
            in src
        assert 'if not getattr(self.config, "macro_gci_live", False):' in src, (
            "GCI is advisory only while its flag is off - driving it is "
            "exactly what makes it live"
        )

    def test_escalation_reads_the_live_subset(self):
        src = _src(REPO / "main.py")
        assert_guard_live(src, "if len(_live_failures) >= 2:")
        # [P311] the real guard is a CONJUNCTION — assert_guard_live made
        # that explicit, where the old substring pin could not tell a
        # whole condition from one clause of one.
        assert_guard_live(
            src, "if self.alert_manager and len(_live_failures) >= 3:")
        assert "if len(_feed_failures) >= 2:" not in src, (
            "the raw count must not drive the escalation any more"
        )

    def test_advisory_failures_are_still_reported(self):
        """Quieter, not silent: an unreported failure is how a feed goes
        dark unnoticed (P155)."""
        src = _src(REPO / "main.py")
        assert "advisory feeds failed" in src


# =============================================================================
# 6 - netflow carry + mock provenance
# =============================================================================

class TestExchangeNetflowCarryAndProvenance:
    def _feed(self):
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowFeed
        return ExchangeNetflowFeed(api_key="k", mock_mode=False)

    def test_a_failed_symbol_carries_forward_instead_of_vanishing(self):
        """Replacing the whole snapshot dropped a still-good sibling, and the
        fetch stamp then suppressed any retry for a full poll interval - one
        transient 500 erased ETH for an hour. The per-family-carry corner
        P265f/P287 closed for CoinGlass, reopened in a new feed."""
        import asyncio
        from datetime import datetime, timezone
        from data_mgmt.feeds.exchange_netflow_feed import (
            ExchangeNetflowMetrics, ExchangeNetflowSnapshot)

        f = self._feed()
        t0 = datetime.now(timezone.utc)
        prev = ExchangeNetflowSnapshot(timestamp=t0)
        for sym in ("BTC", "ETH"):
            prev.metrics[sym] = ExchangeNetflowMetrics(
                symbol=sym, total_balance_coins=100.0, netflow_coins_1d=1.0,
                exchange_count=3, timestamp=t0)
        f._last_data = prev

        async def _only_btc(session, headers, symbol, now):
            if symbol == "ETH":
                return None
            return ExchangeNetflowMetrics(
                symbol="BTC", total_balance_coins=200.0,
                netflow_coins_1d=2.0, exchange_count=4, timestamp=now)

        f._fetch_symbol = _only_btc
        snap = asyncio.run(f._fetch_real())

        assert snap.metrics["BTC"].netflow_coins_1d == 2.0, "fresh BTC"
        assert "ETH" in snap.metrics, "ETH was dropped instead of carried"
        assert snap.metrics["ETH"].timestamp == t0, (
            "a carried record must keep its ORIGINAL timestamp - carrying is "
            "not re-measuring, and its age must stay honest"
        )
        assert any("carried_forward" in e for e in snap.errors)

    def test_mock_readings_are_labelled(self):
        """Mock emits netflow 0.0 with exchange_count=1, so is_usable() is
        True and the value is byte-identical to a measured 'flat flows'."""
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowFeed
        f = ExchangeNetflowFeed(api_key="", mock_mode=True)
        snap = f._build_mock()
        m = snap.get("BTC")
        assert m.is_usable() and m.netflow_coins_1d == 0.0
        assert m.mock is True, "a fabricated reading must say so"
        assert m.to_dict()["mock"] is True

    def test_real_readings_are_not_labelled_mock(self):
        from data_mgmt.feeds.exchange_netflow_feed import ExchangeNetflowMetrics
        assert ExchangeNetflowMetrics(symbol="BTC").mock is False


# =============================================================================
# 7 - the deposit disarmed the sleeve loss cap
# =============================================================================

class TestDrawdownIsNetOfExternalFlows:
    """MEASURED LIVE 2026-08-18. The 15% sleeve drawdown halt compares equity
    to the INCEPTION anchor, so the 2026-08-16 deposit of $7,074.27 (anchor
    $3,997.75, equity $10,847.88) made `dd` read -171.3% and pushed the halt
    trigger down to $3,398 - a 68.7% loss of the capital actually at risk.

    P274 called the deposit direction "conservative", which is true of the P0
    fuse and FALSE of this control; P287's note covers only withdrawals.
    Netting the recorded flow out of the basis fixes both ends.
    """

    def _sleeve(self, start, flows, eq, halt=0.15):
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = CoinbaseSleeve.__new__(CoinbaseSleeve)
        s._sleeve_start_equity = start
        s._external_flow_usd = flows
        s._last_dd_pct = 0.0
        s._halted = False
        s._halt_reason = ""
        s._max_sleeve_drawdown_pct = halt
        s.sleeve_equity_usd = lambda: eq
        s._equity_is_stale = lambda: False
        s.sleeve_equity_age_sec = lambda: 1.0
        s._persist_state = lambda: None
        return s

    def test_the_live_numbers_now_read_honestly(self):
        s = self._sleeve(3997.7520864755475, 7074.27, 10847.88)
        r = s.update_risk()
        assert r["drawdown_pct"] == pytest.approx(0.0202, abs=5e-4), (
            "the 2.02% trading loss on invested capital, not -171%"
        )
        assert not r["halted"]

    def test_the_halt_binds_on_capital_actually_at_risk(self):
        """15% of the $11,072 invested base is ~$9,411 - not the $3,398 the
        inception anchor implied."""
        s = self._sleeve(3997.7520864755475, 7074.27, 9400.0)
        r = s.update_risk()
        assert r["halted"], "a 15% loss of invested capital must halt"
        s2 = self._sleeve(3997.7520864755475, 7074.27, 9500.0)
        assert not s2.update_risk()["halted"], "just under 15% must not halt"

    def test_the_old_anchor_would_not_have_halted_there(self):
        """Pins WHY this changed: at equity $9,400 the pre-fix arithmetic
        reported a NEGATIVE drawdown, i.e. a profit."""
        start, eq = 3997.7520864755475, 9400.0
        assert (start - eq) / start < 0

    def test_no_recorded_flows_is_byte_identical(self):
        """Every sleeve that has never seen a transfer must be unaffected."""
        for eq in (3600.0, 3997.75, 3398.0, 3000.0):
            s = self._sleeve(3997.7520864755475, 0.0, eq)
            expected = (3997.7520864755475 - eq) / 3997.7520864755475
            assert s.update_risk()["drawdown_pct"] == pytest.approx(expected)

    def test_a_withdrawal_no_longer_reads_as_drawdown(self):
        """P287 recorded that a withdrawal trips the sticky halt with one bank
        transfer and prescribed a manual re-anchor. Once the P293h detector
        has recorded the outflow, the basis shrinks with it and no operator
        step is needed."""
        s = self._sleeve(10000.0, -5000.0, 5000.0)
        r = s.update_risk()
        assert r["drawdown_pct"] == pytest.approx(0.0), (
            "withdrawing half the capital is not a 50% loss"
        )
        assert not r["halted"]

    def test_a_total_withdrawal_holds_rather_than_dividing_by_zero(self):
        s = self._sleeve(4000.0, -4000.0, 0.01)
        s._last_dd_pct = 0.03
        r = s.update_risk()
        assert r["drawdown_pct"] == pytest.approx(0.03), "held, not fabricated"
        assert not r["halted"], "an unusable basis must never trip the halt"
