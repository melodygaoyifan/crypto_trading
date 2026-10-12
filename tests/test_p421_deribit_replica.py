"""[P421] The Deribit-direct tenor-30 skew replica — recipe fidelity pinned on
synthetic chains, and the bank-append semantics pinned so the replica can never
overwrite an engine/MCP-written row.

LIVE VALIDATION RECORD (2026-10-12, replica vs the vendor's own 04:23:00Z
snapshot, tenor 30): BTC skew_25d 1.8656020 vs 1.8655993 (|d| 2.7e-6),
skew_10d |d| 4.7e-5, atm/call/put IVs |d| < 1e-4; ETH all fields |d| <= 0.007
IV points (one quote refresh inside the minute). The recipe reproduces the
vendor's computation; these tests keep it reproducing.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.deribit_skew_replica import (  # noqa: E402
    append_row, call_delta, parse_instrument, tenor30_row)

NOW = dt.datetime(2026, 10, 12, 4, 0, tzinfo=dt.timezone.utc)


def _chain(expiry_days, iv, fwd=100000.0, n_strikes=81):
    """A flat-IV chain: every strike carries the same mark_iv, so every
    selected series must interpolate to that expiry's flat IV and skew == 0."""
    rows = []
    for days in expiry_days:
        expiry = NOW + dt.timedelta(days=days)
        name_date = expiry.strftime("%d%b%y").upper().lstrip("0")
        for i in range(n_strikes):
            strike = fwd * (0.5 + i * (1.5 / n_strikes))
            for typ in ("C", "P"):
                rows.append({
                    "instrument_name": f"BTC-{name_date}-{int(strike)}-{typ}",
                    "mark_iv": iv[days] if isinstance(iv, dict) else iv,
                    "underlying_price": fwd,
                })
    return rows


class TestParsing:
    def test_instrument_round_trip(self):
        expiry, strike, is_call = parse_instrument("BTC-12OCT26-84500-C")
        assert expiry == dt.datetime(2026, 10, 12, 8, tzinfo=dt.timezone.utc)
        assert strike == 84500.0 and is_call

    def test_put_and_malformed(self):
        assert parse_instrument("ETH-3JAN27-2500-P")[2] is False
        assert parse_instrument("BTC-PERPETUAL") is None
        assert parse_instrument("BTC-12XXX26-1-C") is None

    def test_atm_call_delta_is_near_half(self):
        d = call_delta(100.0, 100.0, 40.0, 30 / 365)
        assert 0.5 < d < 0.55  # N(0.5*sigma*sqrt(T)) just above 0.5
        assert call_delta(100.0, 50.0, 40.0, 30 / 365) > 0.95   # deep ITM
        assert call_delta(100.0, 200.0, 40.0, 30 / 365) < 0.05  # deep OTM


class TestTenor30:
    def test_flat_chain_yields_zero_skew_and_interpolated_atm(self):
        # two expiries bracketing 30d with different flat IVs: the tenor-30
        # value must be the total-variance interpolation, and skew exactly 0
        iv = {20: 40.0, 40: 50.0}
        chain = _chain([20, 40], iv)
        row = tenor30_row(chain, NOW)
        assert abs(row["skew_25d"]) < 1e-9 and abs(row["skew_10d"]) < 1e-9
        # derive T the way the PARSER does (expiry re-parses to 08:00 UTC —
        # hand-assuming days/365 is the P408 wrong-oracle trap, hit here first)
        t = sorted({(parse_instrument(r["instrument_name"])[0] - NOW)
                    .total_seconds() / (365.0 * 86400.0) for r in chain})
        t0, t1, t30 = t[0], t[1], 30 / 365
        w = (40.0 ** 2 * t0) + ((50.0 ** 2 * t1) - (40.0 ** 2 * t0)) * (t30 - t0) / (t1 - t0)
        expect = math.sqrt(w / t30)
        assert abs(row["atm_iv"] - expect) < 1e-6
        assert 40.0 < row["atm_iv"] < 50.0

    def test_flat_extrapolation_outside_the_expiry_range(self):
        # single 10d expiry: tenor-30 stays flat at that IV (the recipe's
        # outside-range rule), never extrapolated in variance
        row = tenor30_row(_chain([10], 37.5), NOW)
        assert abs(row["atm_iv"] - 37.5) < 1e-9

    def test_expired_and_zero_iv_quotes_are_ineligible(self):
        rows = _chain([20], 40.0)
        rows += _chain([-1], 99.0)  # expired: must not poison the surface
        for r in _chain([20], 0.0, n_strikes=5):
            rows.append(r)          # zero IV: ineligible
        row = tenor30_row(rows, NOW)
        assert abs(row["atm_iv"] - 40.0) < 1e-9

    def test_empty_chain_refuses(self):
        try:
            tenor30_row([], NOW)
            assert False, "must raise — a fabricated row is worse than a gap"
        except ValueError:
            pass


class TestBankAppend:
    def test_existing_date_wins_and_is_never_overwritten(self, tmp_path):
        bank = tmp_path / "bank.jsonl"
        sentinel = {"date": "2026-10-12", "tenor": 30, "skew_25d": -9.9,
                    "skew_10d": 0.0, "call_25d_iv": 1.0, "put_25d_iv": 1.0,
                    "atm_iv": 1.0, "source": "engine"}
        bank.write_text(json.dumps(sentinel) + "\n", encoding="utf-8")
        verdict = append_row(str(bank), {**sentinel, "skew_25d": 7.7,
                                         "source": "deribit_replica_v1"})
        assert verdict == "kept_existing"
        kept = json.loads(bank.read_text(encoding="utf-8").strip())
        assert kept["skew_25d"] == -9.9 and kept["source"] == "engine"

    def test_new_date_appends_sorted(self, tmp_path):
        bank = tmp_path / "bank.jsonl"
        bank.write_text(json.dumps({"date": "2026-10-11", "x": 1}) + "\n",
                        encoding="utf-8")
        assert append_row(str(bank), {"date": "2026-10-12", "tenor": 30,
                                      "skew_25d": 1.0, "skew_10d": 2.0,
                                      "call_25d_iv": 3.0, "put_25d_iv": 4.0,
                                      "atm_iv": 5.0,
                                      "source": "deribit_replica_v1"}) == "appended"
        lines = bank.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(l)["date"] for l in lines] == ["2026-10-11", "2026-10-12"]
