"""[P1-6] Round-trip reconstruction (the Phase 0 algorithm) + weekly KPI script."""
from pathlib import Path

import pytest

from analytics.unit_economics.roundtrips import (build_round_trips, kpi_table,
                                                 normalize_fills, render_markdown)

REPO = Path(__file__).resolve().parents[1]
CS = {"BTC": 0.01}
PIDS = {"BIP-20DEC30-CDE": "BTC"}


def _fill(t, side, size, price, comm=1.0, oid=None):
    return {"trade_time": t, "product_id": "BIP-20DEC30-CDE", "side": side,
            "size": str(size), "price": str(price), "commission": str(comm),
            "commission_detail_total": {"client_commission": str(comm - 0.11 * size),
                                        "venue_commission": str(0.1 * size),
                                        "clearing_commission": str(0.01 * size)},
            "liquidity_indicator": "TAKER", "order_id": oid or t}


def _rts(fills, mids=None, funding=None):
    F = normalize_fills(fills, CS, PIDS, mids)
    return build_round_trips(F, CS, funding)


class TestFifo:
    def test_simple_long_round_trip(self):
        rts, op = _rts([_fill("2026-10-01T00:00:00Z", "BUY", 2, 80000),
                        _fill("2026-10-01T08:00:00Z", "SELL", 2, 80400)])
        assert len(rts) == 1 and op == {}
        r = rts[0]
        assert r["gross_pnl_usd"] == pytest.approx(2 * 0.01 * 400)
        assert r["gross_alpha_bps"] == pytest.approx(50.0)
        assert r["fee_usd"] == pytest.approx(2.0)
        assert r["hold_bars"] == pytest.approx(2.0)

    def test_flip_splits_into_two_round_trips(self):
        rts, op = _rts([_fill("2026-10-01T00:00:00Z", "BUY", 1, 80000),
                        _fill("2026-10-01T04:00:00Z", "SELL", 2, 80800, comm=2.0),
                        _fill("2026-10-01T08:00:00Z", "BUY", 1, 80000)])
        assert len(rts) == 2 and op == {}
        assert rts[0]["gross_pnl_usd"] == pytest.approx(8.0)      # long +800 x 0.01
        assert rts[1]["gross_pnl_usd"] == pytest.approx(8.0)      # short +800 x 0.01
        assert rts[0]["fee_usd"] == pytest.approx(1.0 + 1.0)      # flip fee split by quantity

    def test_open_position_reported_not_closed(self):
        rts, op = _rts([_fill("2026-10-01T00:00:00Z", "SELL", 2, 80000)])
        assert rts == [] and op["BTC"]["contracts"] == -2

    def test_missing_mid_is_none_not_zero(self):
        rts, _ = _rts([_fill("2026-10-01T00:00:00Z", "BUY", 1, 80000),
                       _fill("2026-10-01T04:00:00Z", "SELL", 1, 80000)])
        assert rts[0]["spread_cost_bps"] is None

    def test_spread_signed_cost(self):
        f = [_fill("2026-10-01T00:00:00Z", "BUY", 1, 80008, oid="a"),
             _fill("2026-10-01T04:00:00Z", "SELL", 1, 79992, oid="b")]
        rts, _ = _rts(f, mids={"a": 80000.0, "b": 80000.0})
        assert rts[0]["spread_cost_bps"] == pytest.approx(2.0, abs=1e-3)

    def test_funding_estimate_and_absent_rate(self):
        f = [_fill("2026-10-01T00:00:00Z", "BUY", 1, 80000),
             _fill("2026-10-01T04:00:00Z", "SELL", 1, 80000)]
        import datetime as dt
        t0 = dt.datetime(2026, 9, 30, 23, tzinfo=dt.timezone.utc).timestamp()
        rts, _ = _rts(f, funding={"BTC": [(t0, 1e-5)]})
        assert rts[0]["funding_bps"] == pytest.approx(4 * 1e-5 * 1e4, rel=0.3)
        rts2, _ = _rts(f, funding={})
        assert rts2[0]["funding_bps"] is None


class TestKpi:
    def test_insufficient_below_ten(self):
        rts, _ = _rts([_fill("2026-10-01T00:00:00Z", "BUY", 1, 80000),
                       _fill("2026-10-01T04:00:00Z", "SELL", 1, 80100)])
        t = kpi_table(rts)
        assert t["BTC"]["status"] == "INSUFFICIENT" and t["BTC"]["n_rt"] == 1

    def test_markdown_renders_regime_rows_without_breaking_tables(self):
        rts, op = _rts([_fill("2026-10-01T00:00:00Z", "BUY", 1, 80000),
                        _fill("2026-10-01T04:00:00Z", "SELL", 1, 80100)])
        md = render_markdown(rts, op, 0.0, "t")
        assert "BTC / UNKNOWN" in md and "BTC|UNKNOWN" not in md


class TestPhase0Parity:
    def test_reproduces_phase0_round_trips(self):
        """The Phase 0 audit's 90 RTs / per-asset counts / gross sums, from
        the same venue fills (operator-local file; skipped in CI)."""
        pq = REPO / "data" / "phase0" / "fills_raw.parquet"
        if not pq.exists():
            pytest.skip("data/phase0/fills_raw.parquet absent")
        pd = pytest.importorskip("pandas")
        F = pd.read_parquet(pq)
        raw = [{"trade_time": r.iso, "product_id": r.product_id, "side": r.side,
                "size": r.size_contracts, "price": r.price, "commission": r.commission,
                "commission_detail_total": {"client_commission": r.client_commission,
                                            "venue_commission": r.venue_commission,
                                            "clearing_commission": r.clearing_commission},
                "liquidity_indicator": r.liquidity_indicator, "order_id": r.order_id}
               for r in F.itertuples()]
        pids = {"BIP-20DEC30-CDE": "BTC", "ETP-20DEC30-CDE": "ETH",
                "SLP-20DEC30-CDE": "SOL", "XPP-20DEC30-CDE": "XRP"}
        cs = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0, "XRP": 500.0}
        rts, op = build_round_trips(normalize_fills(raw, cs, pids), cs)
        t = kpi_table(rts)
        assert {a: t[a]["n_rt"] for a in ("BTC", "ETH", "SOL", "XRP")} == \
            {"BTC": 41, "ETH": 23, "SOL": 24, "XRP": 2}
        assert sum(r["gross_pnl_usd"] for r in rts) == pytest.approx(-5.85, abs=0.01)
        assert t["BTC"]["alpha_per_rt_bps"] == pytest.approx(-23.86, abs=0.01)
        assert op["BTC"]["contracts"] == -2


class TestScript:
    def test_refuses_on_zero_fills(self, tmp_path):
        import importlib.util
        import json
        spec = importlib.util.spec_from_file_location("wue", REPO / "scripts" / "weekly_unit_economics.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        p = tmp_path / "f.json"; p.write_text(json.dumps([]), encoding="utf-8")
        assert m.main(["--fills-json", str(p), "--data-dir", str(tmp_path),
                       "--out", str(tmp_path / "o.md")]) == 2
        assert not (tmp_path / "o.md").exists()

    def test_ships_in_image_both_halves(self):
        assert "scripts/weekly_unit_economics.py" in (REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        assert "!scripts/weekly_unit_economics.py" in (REPO / ".dockerignore").read_text(encoding="utf-8")
