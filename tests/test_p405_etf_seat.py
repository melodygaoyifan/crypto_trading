"""[P405] Live ETF-flow seat: reduce-only de-risk (BTC) + directional decide
(ETH), fail-safe on stale/absent. Pins the invariants that make it safe to
arm on a live account."""
import time
import pytest

from main import etf_seat_decision, ProductionConfig
from defense.etf_flow_shadow import EtfFlowShadow

D = ["BTC"]; C = ["ETH"]


class TestEtfSeatDecision:
    def test_derisk_is_reduce_only(self):
        # fires ONLY on outflow against a LONG book -> flatten
        assert etf_seat_decision("BTC", -1.0, True, 1.0, D, C) == ("etf_derisk", 0.0)
        # flat book, short book, or inflow -> untouched (never adds/flips)
        assert etf_seat_decision("BTC", -1.0, True, 0.0, D, C) is None
        assert etf_seat_decision("BTC", -1.0, True, -1.0, D, C) is None
        assert etf_seat_decision("BTC", 1.0, True, 1.0, D, C) is None

    def test_decide_is_directional(self):
        assert etf_seat_decision("ETH", 1.0, True, 0.0, D, C) == ("etf_flow", 1.0)
        assert etf_seat_decision("ETH", -1.0, True, 1.0, D, C) == ("etf_flow", -1.0)
        # a flat ETF signal takes no seat
        assert etf_seat_decision("ETH", 0.0, True, 0.0, D, C) is None

    def test_fail_safe_when_not_fresh(self):
        # stale/absent/warmup -> NO seat, incumbent stands (P2)
        assert etf_seat_decision("BTC", -1.0, False, 1.0, D, C) is None
        assert etf_seat_decision("ETH", 1.0, False, 0.0, D, C) is None

    def test_unconfigured_asset_takes_no_seat(self):
        assert etf_seat_decision("SOL", -1.0, True, 1.0, D, C) is None

    def test_derisk_never_returns_a_directional_position(self):
        # exhaustive: a derisk-asset decision is only ever flat or None
        for bd in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for ed in (-1.0, 0.0, 1.0):
                r = etf_seat_decision("BTC", ed, True, bd, D, [])
                assert r is None or r == ("etf_derisk", 0.0)


class TestSeatDirection:
    def _shadow(self, tmp_path):
        s = object.__new__(EtfFlowShadow)
        s._seat_state = {}
        return s

    def test_fresh_reading(self, tmp_path):
        s = self._shadow(tmp_path)
        s._seat_state["BTC"] = (-1.0, True, time.time())
        assert s.seat_direction("BTC") == (-1.0, True)

    def test_not_fresh_reading(self, tmp_path):
        s = self._shadow(tmp_path)
        s._seat_state["BTC"] = (0.0, False, time.time())
        assert s.seat_direction("BTC") == (0.0, False)

    def test_aged_reading_is_not_fresh(self, tmp_path):
        s = self._shadow(tmp_path)
        s._seat_state["BTC"] = (1.0, True, time.time() - 13 * 3600)  # > 12h
        d, fresh = s.seat_direction("BTC")
        assert fresh is False

    def test_missing_asset(self, tmp_path):
        assert self._shadow(tmp_path).seat_direction("ETH") is None


class TestConfigParse:
    def test_defaults_off(self):
        cfg = ProductionConfig()
        assert cfg.etf_seat_mode == "off"
        assert cfg.etf_derisk_assets is None and cfg.etf_decide_assets is None

    def test_parses_enforce_and_lists(self, tmp_path):
        import json
        p = tmp_path / "c.json"
        p.write_text(json.dumps({
            "etf_seat_mode": "enforce",
            "etf_derisk_assets": ["BTC"],
            "etf_decide_assets": ["ETH"]}), encoding="utf-8")
        cfg = ProductionConfig.from_file(p)
        assert cfg.etf_seat_mode == "enforce"
        assert cfg.etf_derisk_assets == ["BTC"]
        assert cfg.etf_decide_assets == ["ETH"]

    def test_unknown_mode_falls_back_to_off(self, tmp_path):
        import json
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"etf_seat_mode": "garbage"}), encoding="utf-8")
        assert ProductionConfig.from_file(p).etf_seat_mode == "off"


def test_live_config_etf_seat_is_de_risk_only_re_horizoned():
    """[research 2026-08-26] The ETF seat was re-horizoned from next-bar
    directional-decide to a slow de-risk/context gate: ETF flow is coincident/
    lagged and decayed 2024->2026 (matches P400), so it must NOT take a
    directional bet. This pins the LIVE decided state: both majors are de-risk
    (reduce-only), and NO asset is a directional decide-asset. A silent
    re-arming of a directional ETF seat fails here loudly (P237/P141) — the
    ETF signal's directional value belongs only in the agree-gated combiner
    (P407j), not a standalone next-bar seat.
    """
    import json, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c = json.load(open(os.path.join(root, "configs", "live_high_risk.json"),
                       encoding="utf-8"))
    assert c.get("etf_seat_mode") == "enforce"
    assert c.get("etf_derisk_assets") == ["BTC", "ETH"], "both majors de-risk-only"
    assert c.get("etf_decide_assets") == [], (
        "the ETF seat must take NO directional decision (re-horizoned to context)")
