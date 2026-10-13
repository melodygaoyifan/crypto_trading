"""[P423] The four evidence/decision gaps closed by measurement + mechanism.

G1 decider/cause stamp on every sleeve fill (realized P&L per decider)
G2 multi-factor / conditional / ex-ante beta
G3 the standing-decision registry with deadlines + weekly alarm + heartbeat
G4 compute_shadow_ic --allow-kraken-ohlcv (opt-in; the P213 refusal stands)
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.decisions import registry as reg  # noqa: E402
from analytics.sleeve_attribution import decider_attribution as da  # noqa: E402
from analytics.sleeve_attribution import sleeve_beta_review as sbr  # noqa: E402


# ---------------------------------------------------------------- G3 registry
def _reg(tmp_path, entries):
    p = tmp_path / "pending_decisions.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


class TestDecisionRegistry:
    def test_the_committed_registry_loads_and_every_entry_has_a_default(self):
        entries = reg.load_registry()
        assert entries, "the registry must not be empty while decisions are open"
        for d in entries:
            assert d.get("default"), d["id"]
            assert d.get("options"), d["id"]
            assert d.get("evidence"), d["id"]
            assert "auto-de-risk" in d["default"] or "keep the current" in d["default"]

    def test_overdue_is_strictly_after_the_deadline(self, tmp_path):
        p = _reg(tmp_path, [{"id": "a", "deadline": "2026-10-20"},
                            {"id": "b", "deadline": "2026-10-21", "status": "decided"}])
        r = reg.load_registry(p)
        assert [d["id"] for d in reg.overdue(r, date(2026, 10, 20))] == []
        assert [d["id"] for d in reg.overdue(r, date(2026, 10, 21))] == ["a"]
        assert reg.days_overdue(r[0], date(2026, 10, 25)) == 5

    def test_decided_entries_never_count(self, tmp_path):
        p = _reg(tmp_path, [{"id": "a", "deadline": "2020-01-01", "status": "decided"}])
        assert reg.heartbeat_line(p, date(2026, 1, 1)) == "none pending"

    def test_heartbeat_line_names_the_overdue_items_and_the_default(self, tmp_path):
        p = _reg(tmp_path, [{"id": "late", "deadline": "2026-10-01"},
                            {"id": "soon", "deadline": "2026-12-01"}])
        line = reg.heartbeat_line(p, date(2026, 10, 13))
        assert "OVERDUE" in line and "late (+12d)" in line and "keep" in line
        line2 = reg.heartbeat_line(p, date(2026, 9, 1))
        assert "next due 2026-10-01: late" in line2 and "OVERDUE" not in line2

    def test_an_unreadable_registry_is_an_error_not_none_pending(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(reg.RegistryUnreadable):
            reg.heartbeat_line(p)
        with pytest.raises(reg.RegistryUnreadable):
            reg.load_registry(tmp_path / "missing.json")

    def test_decision_check_exit_codes(self, tmp_path):
        p = _reg(tmp_path, [{"id": "late", "deadline": "2026-10-01", "title": "t",
                             "options": ["o"], "default": "keep"}])
        base = [sys.executable, "-X", "utf8", str(REPO / "scripts" / "decision_check.py"),
                "--registry", str(p)]
        r = subprocess.run(base + ["--today", "2026-10-13"], capture_output=True,
                           text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 3 and "OVERDUE" in r.stdout, r.stdout
        r = subprocess.run(base + ["--today", "2026-09-01"], capture_output=True,
                           text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 0 and "due in" in r.stdout
        r = subprocess.run(base[:-1] + [str(tmp_path / "nope.json"), "--today", "2026-09-01"],
                           capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 2 and "REFUSING" in r.stderr

    def test_decision_check_never_edits_anything(self):
        src = (REPO / "scripts" / "decision_check.py").read_text(encoding="utf-8")
        assert "write_text" not in src and "json.dump" not in src

    def test_decision_check_ships_in_the_image_both_halves(self):
        df = (REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        di = (REPO / ".dockerignore").read_text(encoding="utf-8")
        assert "scripts/decision_check.py" in df
        assert "!scripts/decision_check.py" in di

    def test_heartbeat_helper_never_raises(self, monkeypatch):
        import main as m
        assert "pending" in m.decisions_heartbeat_text()
        monkeypatch.setattr(reg, "heartbeat_line", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
        txt = m.decisions_heartbeat_text()
        assert "unreadable" in txt and "decision_check" in txt

    def test_heartbeat_field_is_wired(self):
        import inspect
        import main as m
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        i = src.index('"Passive books (advisory)": _hb_adv_txt,')
        assert '"Decisions pending": decisions_heartbeat_text(),' in src[i:i + 600]


# ---------------------------------------------------------------- G1 fill stamp
class _FakeSleeve:
    """Enough of CoinbaseSleeve for _record_fill_quality + the context API."""
    def __init__(self, tmp_path):
        self._data_dir = tmp_path
        self._last_positions = {"ETH": {"signed_contracts": 6}}
        self._reconcile_ok = True


def _bind(cls, tmp_path):
    from exchange.coinbase_sleeve import CoinbaseSleeve
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._last_positions = {"ETH": {"signed_contracts": 6}}
    s._reconcile_ok = True
    return s


class TestFillStamp:
    def test_context_round_trip_and_clear(self, tmp_path):
        s = _bind(None, tmp_path)
        s.set_order_context("ETH", decider="regimebook", src="pre_gate", cause="signal",
                            intent_direction=0.5, alpha_bps=88.1, target_contracts=6)
        assert s._order_context["ETH"]["decider"] == "regimebook"
        assert s._order_context["ETH"]["alpha_bps"] == 88.1
        s.clear_order_context("ETH")
        assert "ETH" not in s._order_context
        s.clear_order_context("NEVER_SET")  # no raise

    def test_record_carries_the_stamp_and_the_pre_leg_position(self, tmp_path, monkeypatch):
        from exchange import coinbase_sleeve as cs
        s = _bind(None, tmp_path)
        written = []
        monkeypatch.setattr(cs.CoinbaseSleeve, "_append_fill_quality",
                            lambda self, rec: written.append(rec), raising=False)
        # locate the ledger write: the method appends via a helper or open();
        # patch the file path through the data dir instead
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        s.set_order_context("ETH", decider="regimebook", src="pre_gate", cause="signal",
                            intent_direction=0.5, alpha_bps=88.1, target_contracts=6)
        s._record_fill_quality("ETH", "oid", "SELL", 6, "taker_cross", False,
                               decision_mid=2689.75, decision_bid=2689.0,
                               decision_ask=2690.5,
                               order_payload={"status": "FILLED",
                                              "average_filled_price": "2689.0",
                                              "filled_size": "6"})
        rec = _last_record(tmp_path, written)
        assert rec["decider"] == "regimebook" and rec["cause"] == "signal"
        assert rec["position_before"] == 6 and rec["alpha_bps"] == 88.1

    def test_no_context_records_none_not_missing(self, tmp_path, monkeypatch):
        s = _bind(None, tmp_path)
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        s._record_fill_quality("ETH", "oid", "SELL", 6, "taker_cross", False)
        rec = _last_record(tmp_path, [])
        assert "decider" in rec and rec["decider"] is None
        assert da.tag_of(rec) == da.NO_CTX
        assert da.tag_of({"asset": "ETH"}) == da.UNTAGGED

    def test_decider_and_cause_helpers(self):
        import main as m
        it = types.SimpleNamespace(quant_strategy_id="regimebook")
        assert m.sleeve_order_decider(it, "pre_gate") == "regimebook"
        assert m.sleeve_order_decider(it, "pre_gate+ma_veto") == "ma_filter"
        assert m.sleeve_order_decider(None, "pre_gate") == "pre_gate_signal"
        assert m.sleeve_order_decider(None, "veto_flat:x") == "veto_flat"
        assert m.sleeve_order_cause("pre_gate", 6) == "signal"
        assert m.sleeve_order_cause("pre_gate", 0) == "signal_flat"
        assert m.sleeve_order_cause("veto_flat:[P0]", 0) == "veto_flat"
        assert m.sleeve_order_cause("entry_blocked_flat:x", 0) == "entry_veto"
        assert m.sleeve_order_cause("pre_gate+whale_veto", 0) == "filter"

    def test_driver_sets_before_and_clears_after_manage(self):
        import inspect
        import main as m
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        i = src.index("_sl.set_order_context(")
        j = src.index("_m_res = await _sl.manage_to_signal(", i)
        k = src.index("_sl.clear_order_context(_m_a)", j)
        assert i < j < k
        assert "finally:" in src[j:k]

    def test_emergency_paths_stamp_their_own_cause(self):
        import inspect
        import main as m
        src = inspect.getsource(m.sleeve_fast_risk_action)
        assert 'cause="risk_exit:fast_risk"' in src
        src2 = inspect.getsource(m.HMATSProductionRunner._check_and_execute_force_flat)
        assert 'cause="risk_exit:force_flat"' in src2

    def test_contract_size_tables_match_the_adapter(self):
        from exchange.coinbase_adapter import CoinbaseAdapter
        pid = {"BTC": "BIP", "ETH": "ETP", "SOL": "SLP", "XRP": "XPP", "ADA": "ADP",
               "LTC": "LCP", "DOGE": "DOP", "BNB": "BNB"}
        for a, cs in da.CONTRACT_SIZE.items():
            assert CoinbaseAdapter._CONTRACT_SIZE_FALLBACK[f"{pid[a]}-20DEC30-CDE"] == cs, a
        for a, cs in sbr.CONTRACT_SIZE.items():
            assert da.CONTRACT_SIZE[a] == cs


def _last_record(tmp_path, written):
    if written:
        return written[-1]
    for p in list(tmp_path.rglob("fill_quality.jsonl")) + list((REPO / "data").glob("fill_quality.jsonl")):
        lines = p.read_text(encoding="utf-8").splitlines()
        if lines:
            return json.loads(lines[-1])
    raise AssertionError("no fill record written")


# ---------------------------------------------------------------- G1 reader
def _leg(ts, asset, side, n, px, decider="regimebook", cause="signal", pb=None, fee=0.5):
    r = {"ts": ts, "asset": asset, "side": side, "contracts": n, "status": "filled",
         "fill_avg_price": px, "fees_usd": fee, "decider": decider, "cause": cause,
         "src": "pre_gate", "liquidity": "maker", "urgent": False}
    if pb is not None:
        r["position_before"] = pb
    return r


class TestDeciderAttribution:
    def test_round_trip_attributed_to_entry_decider_with_exit_cause(self):
        rows = [_leg(1, "ETH", "BUY", 6, 2000.0, pb=0),
                _leg(2, "ETH", "SELL", 6, 2010.0, decider="fast_risk_watchdog",
                     cause="risk_exit:fast_risk", pb=6)]
        res = da.attribute(rows)
        t = res["round_trips"]
        assert len(t) == 1
        assert t[0]["entry_decider"] == "regimebook"
        assert t[0]["exit_cause"] == "risk_exit:fast_risk"
        assert t[0]["gross_usd"] == pytest.approx(10.0 * 6 * 0.1)
        assert res["positions"]["ETH"] == 0 and not res["open_lots"]

    def test_flip_closes_then_opens(self):
        rows = [_leg(1, "SOL", "BUY", 2, 100.0, pb=0),
                _leg(2, "SOL", "SELL", 4, 110.0, decider="skew_contra", pb=2),
                _leg(3, "SOL", "BUY", 2, 105.0, cause="signal_flat", pb=-2)]
        res = da.attribute(rows)
        t = res["round_trips"]
        assert [x["entry_decider"] for x in t] == ["regimebook", "skew_contra"]
        assert t[0]["gross_usd"] == pytest.approx(10.0 * 2 * 5.0)
        assert t[1]["gross_usd"] == pytest.approx(5.0 * 2 * 5.0)  # short 110 -> 105

    def test_unresolved_leg_makes_the_trip_unpriced_never_fabricated(self):
        rows = [_leg(1, "BTC", "BUY", 1, None, pb=0),
                _leg(2, "BTC", "SELL", 1, 65000.0, pb=1)]
        rows[0]["status"] = "unresolved"
        res = da.attribute(rows)
        assert res["round_trips"][0]["gross_usd"] is None and res["unpriced"] == 1

    def test_snapshot_reanchors_a_drifted_walk(self):
        rows = [_leg(1, "ETH", "BUY", 6, 2000.0, pb=0),
                # the ledger missed a flatten; the venue says flat before this leg
                _leg(2, "ETH", "BUY", 3, 2100.0, pb=0),
                _leg(3, "ETH", "SELL", 3, 2200.0, pb=3)]
        res = da.attribute(rows)
        assert res["drift"] == 1
        assert res["round_trips"][-1]["gross_usd"] == pytest.approx(100.0 * 3 * 0.1)
        assert res["positions"]["ETH"] == 0

    def test_tag_states_are_distinct(self):
        res = da.attribute([{"ts": 1, "asset": "ETH", "side": "BUY", "contracts": 1,
                             "status": "filled", "fill_avg_price": 1.0},
                            {"ts": 2, "asset": "ETH", "side": "SELL", "contracts": 1,
                             "status": "filled", "fill_avg_price": 1.0, "decider": None}])
        assert res["untagged"] == 1 and res["no_ctx"] == 1
        s = da.summarize(res)
        assert list(s.values())[0]["entry_decider"] == da.UNTAGGED

    def test_cli_refuses_on_missing_file(self, tmp_path):
        r = subprocess.run([sys.executable, "-X", "utf8",
                            str(REPO / "analytics/sleeve_attribution/decider_attribution.py"),
                            "--fill-file", str(tmp_path / "nope.jsonl")],
                           capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 2 and "REFUSING" in r.stderr


# ---------------------------------------------------------------- G2 beta
class TestBetaReview:
    def test_ols_hc1_recovers_known_coefficients(self):
        import random
        rnd = random.Random(7)
        X, y = [], []
        for _ in range(400):
            a, b, c = rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1)
            X.append([a, b, c])
            y.append(0.5 + 1.0 * a - 2.0 * b + 0.3 * c + rnd.gauss(0, 0.1))
        r = sbr.ols_hc1(X, y)
        assert r["coef"][0] == pytest.approx(0.5, abs=0.03)
        assert r["coef"][1] == pytest.approx(1.0, abs=0.03)
        assert r["coef"][2] == pytest.approx(-2.0, abs=0.03)
        assert r["coef"][3] == pytest.approx(0.3, abs=0.03)
        assert all(0 < s < 0.02 for s in r["se"][1:])
        assert r["r2"] > 0.99

    def test_ols_singular_returns_none(self):
        assert sbr.ols_hc1([[1.0, 1.0]] * 10, [1.0] * 10) is None
        assert sbr.ols_hc1([[1.0]], [1.0]) is None

    def test_pairs_skip_external_flows_and_use_pnl_delta(self):
        ts = [float(i * 14400) for i in range(10)]
        series = {a: (ts, [100.0 + i for i in range(10)]) for a in sbr.FACTOR_PAIRS}
        rows = [{"ts": ts[0], "equity_usd": 1000.0, "pnl_usd": 0.0, "external_flow_usd": 0.0,
                 "positions": {"BTC": 1.0}},
                {"ts": ts[1], "equity_usd": 1010.0, "pnl_usd": 10.0, "external_flow_usd": 0.0,
                 "positions": {}},
                {"ts": ts[2], "equity_usd": 8010.0, "pnl_usd": 10.0, "external_flow_usd": 7000.0,
                 "positions": {}},
                {"ts": ts[3], "equity_usd": 8020.0, "pnl_usd": 20.0, "external_flow_usd": 7000.0,
                 "positions": {}}]
        pairs = sbr.build_pairs(rows, series)
        assert len(pairs) == 2  # the deposit pair is dropped
        assert pairs[0]["sleeve"] == pytest.approx(0.01)
        assert pairs[0]["in_market"] and pairs[0]["expo"]["BTC"] == pytest.approx(1.0 * 0.01 * 100.0 / 1000.0)
        assert pairs[1]["sleeve"] == pytest.approx(10.0 / 8010.0)

    def test_factor_pairs_match_agent_ic_review(self):
        from analytics.ic.agent_ic_review import KRAKEN_PAIRS
        for a, p in sbr.FACTOR_PAIRS.items():
            assert KRAKEN_PAIRS[a] == p


# ---------------------------------------------------------------- G4 kraken fallback
class TestKrakenOhlcvFallback:
    def test_default_off_and_flag_parses(self):
        from analytics.shadow_ic import compute_shadow_ic as c
        assert c.ALLOW_KRAKEN_OHLCV is False
        src = (REPO / "analytics/shadow_ic/compute_shadow_ic.py").read_text(encoding="utf-8")
        assert '"--allow-kraken-ohlcv"' in src

    def test_without_the_flag_a_missing_parquet_still_raises(self, monkeypatch, tmp_path):
        from analytics.shadow_ic import compute_shadow_ic as c
        monkeypatch.setattr(c, "OHLCV_DIR", tmp_path)
        called = []
        monkeypatch.setattr(c, "fetch_kraken_ohlcv", lambda a: called.append(a))
        with pytest.raises(FileNotFoundError):
            c.load_ohlcv("BTC")
        assert called == []

    def test_with_the_flag_the_fallback_is_used_and_unknown_pairs_still_fail(self, monkeypatch, tmp_path):
        from analytics.shadow_ic import compute_shadow_ic as c
        monkeypatch.setattr(c, "OHLCV_DIR", tmp_path)
        monkeypatch.setattr(c, "fetch_kraken_ohlcv",
                            lambda a: {"BTC": "df"}[a] if a == "BTC" else (_ for _ in ()).throw(KeyError(a)))
        assert c.load_ohlcv("BTC", allow_kraken=True) == "df"
        with pytest.raises(KeyError):
            c.load_ohlcv("ZZZ", allow_kraken=True)

    def test_weekly_wrapper_uses_the_flag_and_never_applies(self):
        src = (REPO / "scripts/ops/seat_check_weekly.sh").read_text(encoding="utf-8")
        assert "--allow-kraken-ohlcv" in src and "--shadow-ic-report" in src
        assert "NEVER auto-applied" in src
        assert "--pool-assets" in src and "--horizons 1,4" in src
