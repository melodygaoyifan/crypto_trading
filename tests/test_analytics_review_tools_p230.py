"""[P230] The two review tools + the attribution reasoning-key closure.

- agent_ic_review: per-AGENT forward IC with the P166 cost-aware bar — the
  instrument the P228 promotion path assumes but never had.
- sleeve_beta_review: realized sleeve beta vs BTC from the P150 PnL series —
  the first live measurement of the book's documented (P143/P201) beta risk.
  Both REFUSE (exit 2) on missing data or unreachable prices — 'no data'
  must never read as 'no signal' (P199/P213/P227b family).
- attribution: 4 of the 5 reasoning-key gaps closed (quant / micro /
  kraken_quant / model_alpha); sentiment_source stays open (no producer
  exists anywhere — needs a writer first, recorded in the code comment).
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


def _run_tool(rel, args, cwd):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(REPO / rel), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


class TestRefusals:
    def test_agent_ic_refuses_on_missing_logs(self, tmp_path):
        r = _run_tool("analytics/ic/agent_ic_review.py",
                      ["--log-dir", str(tmp_path / "nope")], tmp_path)
        assert r.returncode == 2, (r.returncode, r.stderr[-200:])
        assert "REFUSING TO REPORT" in r.stderr
        assert "no data source" in r.stderr or "not 'no agent signal'" in r.stderr

    def test_sleeve_beta_refuses_on_missing_pnl(self, tmp_path):
        r = _run_tool("analytics/sleeve_attribution/sleeve_beta_review.py",
                      ["--pnl-file", str(tmp_path / "nope.jsonl")], tmp_path)
        assert r.returncode == 2, (r.returncode, r.stderr[-200:])
        assert "REFUSING TO REPORT" in r.stderr

    def test_sleeve_beta_refuses_on_too_few_points(self, tmp_path):
        f = tmp_path / "thin.jsonl"
        import time
        now = time.time()
        f.write_text("\n".join(
            json.dumps({"ts": now - i * 14400, "equity_usd": 3800.0})
            for i in range(3)))
        r = _run_tool("analytics/sleeve_attribution/sleeve_beta_review.py",
                      ["--pnl-file", str(f)], tmp_path)
        assert r.returncode == 2
        assert "usable points" in r.stderr


class TestIcMath:
    @pytest.fixture(scope="class")
    def mod(self):
        import analytics.ic.agent_ic_review as m
        return m

    def test_pearson_known_values(self, mod):
        assert mod._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
        assert mod._pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
        assert mod._pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance
        assert mod._pearson([1, 2], [1, 2]) is None        # n too small

    def test_required_ic_matches_p166_edge_model(self, mod):
        """Round-trip: the IC the inverter returns must reproduce exactly the
        edge bar when pushed back through the forward edge model."""
        vol = 107.0  # the P166 doc's measured 16h vol
        req = mod.required_ic(vol)
        assert req is not None and 0 < req < 1
        edge = 0.7979 * 2.0 * math.sin(math.pi * req / 6.0) * vol
        assert edge == pytest.approx(
            mod.SAFETY_MARGIN * mod.TAKER_RT_BPS, rel=1e-9)

    def test_required_ic_refuses_unreachable_vol(self, mod):
        assert mod.required_ic(0.0) is None
        assert mod.required_ic(1.0) is None  # bar unreachable at 1bp vol


class TestAttributionReasoningKeys:
    """[P230] The extractors read reasoning keys _attr_collected never passed
    — reasoning was '' for the life of the tracker (P227 audit item 7)."""

    @pytest.fixture(scope="class")
    def envelope_src(self):
        return (REPO / "agents" / "signal_envelope.py").read_text(
            encoding="utf-8-sig", errors="replace")

    @pytest.mark.parametrize("key", [
        "quant_strategy", "micro_primary_signal",
        "kq_primary_strategy", "model_alpha_reasons",
    ])
    def test_extractor_key_is_now_collected(self, key, envelope_src):
        assert key in envelope_src, f"extractor no longer reads {key}"
        assert f'"{key}"' in MAIN, (
            f"P230 regression: _attr_collected no longer passes {key} — that "
            f"agent's attribution reasoning is back to the empty string."
        )

    def test_micro_primary_signal_is_bridged_into_agent_signals(self):
        assert "agent_signals['micro_primary_signal']" in MAIN, (
            "the bridge is gone — the collected key would read an absent "
            "agent_signals entry and pass '' (P2 shape)."
        )

    def test_model_alpha_reasons_collected_as_list_not_float(self):
        """The extractor does ', '.join(reasons); a comprehension default of
        0.0 would raise TypeError inside attribution."""
        idx = MAIN.find('"model_alpha_reasons": list(')
        assert idx > 0

    def test_sentiment_source_still_honestly_open(self, envelope_src):
        """No producer exists; if one appears, close the gap and drop this."""
        assert "sentiment_source" in envelope_src
        assert '"sentiment_source"' not in MAIN, (
            "someone added sentiment_source to _attr_collected — good, but "
            "verify a PRODUCER writes it (none existed as of P230), then "
            "retire this guard."
        )
