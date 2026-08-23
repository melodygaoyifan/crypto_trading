"""[P382] The promotion gate's cost bar must be REAL in pooled mode and priced
at the venue's MEASURED fee — two defects, both verified at the call site.

(1) `compute_shadow_ic.compute_per_strategy_ic` in POOLED mode (`--pool-assets`,
    POOLABLE_FAMILIES) standardizes forward returns per asset BEFORE pooling
    (correct for the rank IC, P299) — and then computed `fwd_vol_bps` from that
    z-scored series. sigma of a z-score is 1.0 == 10,000 bps, so `required_ic`
    collapsed to ~0.001 and the P166 cost bar was vacuous on exactly the read
    P332 pre-committed as GOVERNING (a pooled PROMOTE where every member HOLDs).
    Fix: pooled vol = n-weighted RMS of each member asset's RAW sigma,
    sqrt(sum n_a sigma_a^2 / sum n_a), computed before standardizing.

(2) Both tools priced the REFUTED 3bps/side percentage model (6.0 bps round
    trip). P315/P334 measured CDE fees at 9.4-14.5 bps PER LEG; P374 all-in
    round trips BTC 27.7 / ETH 44.0 / SOL 41.0. Fix: derive the round trip from
    `core.cde_fees.CDE_FEE_BPS` (the registered calibration), per asset; pooled
    and per-agent rows take the MAX over their members; 6.0 stays ONLY as a
    floor; an unreadable calibration falls back to 6.0 and SAYS so.

These tests DRIVE the real functions end-to-end (monkeypatched prices, a real
ledger on disk) and read the numbers — the P299 suite pinned "standardiz" as a
substring, which is the P234 shape (a pin that proves the code was written,
not what it computes). Every pin here was falsification-probed red by
reverting the corresponding hunk.
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import analytics.shadow_ic.compute_shadow_ic as sic  # noqa: E402
import analytics.ic.agent_ic_review as air  # noqa: E402
from core.cde_fees import CDE_FEE_BPS  # noqa: E402  (P310: the PRODUCER)

ASSETS = ("BTC", "ETH", "SOL")
H = 4
N_BARS = 400          # price bars per asset
N_RECS = 200          # ledger records per asset (>= 30 at every horizon)


# ---------------------------------------------------------------------------
# synthetic world: three assets with DIFFERENT dispersion, a rule that follows
# the forward return (high IC) on each
# ---------------------------------------------------------------------------
def _prices(sigma_per_bar: float, seed: int, start: datetime):
    """A geometric random walk; returns (timestamps, closes)."""
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(1, N_BARS):
        closes.append(closes[-1] * (1.0 + rng.gauss(0.0, sigma_per_bar)))
    ts = [start + timedelta(hours=4 * i) for i in range(N_BARS)]
    return ts, closes


def _world(start: datetime | None = None, sigmas=None):
    """{asset: (timestamps, closes)} with per-asset sigma (per 4H bar)."""
    start = start or datetime(2026, 7, 1, tzinfo=timezone.utc)
    sigmas = sigmas or {"BTC": 0.0010, "ETH": 0.0020, "SOL": 0.0045}
    return {a: _prices(sigmas[a], seed=i + 1, start=start)
            for i, a in enumerate(ASSETS)}


def _ohlcv_patch(monkeypatch, world):
    import pandas as pd

    def _load(asset):
        ts, closes = world[asset]
        return pd.DataFrame({"timestamp": pd.DatetimeIndex(ts), "close": closes})
    monkeypatch.setattr(sic, "load_ohlcv", _load)


def _records(world, strategy="regimebook", flip_noise=0.2, seed=3):
    """Ledger rows whose direction FOLLOWS the h-bar forward return (with a
    little noise), so every asset carries a real, high IC. The parsed-ts
    field is what compute_per_strategy_ic reads."""
    rng = random.Random(seed)
    out = []
    for a in ASSETS:
        ts, closes = world[a]
        for i in range(N_RECS):
            fr = closes[i + H] / closes[i] - 1.0
            d = 1.0 if fr > 0 else -1.0
            if rng.random() < flip_noise:
                d = -d
            out.append({"strategy": strategy, "asset": a, "direction": d,
                        "confidence": 1.0, "_parsed_ts": ts[i]})
    return out


def _rows(monkeypatch, world=None, recs=None):
    world = world or _world()
    _ohlcv_patch(monkeypatch, world)
    recs = recs or _records(world)
    per_asset = sic.compute_per_strategy_ic(recs, horizons_bars=(H,),
                                            pool_assets=False)
    pooled = sic.compute_per_strategy_ic(recs, horizons_bars=(H,),
                                         pool_assets=True)
    return per_asset, pooled[("regimebook", sic.POOLED_KEY)]


# ===========================================================================
# 1. The pooled forward vol is REAL
# ===========================================================================
class TestPooledVolIsRealNotStandardized:

    def test_pooled_vol_and_required_ic_lie_inside_the_per_asset_range(
            self, monkeypatch):
        """The headline pin: a pooled row's vol/required_ic must sit between
        its members' — never ~10,000 bps / ~0.001."""
        per_asset, pooled = _rows(monkeypatch)
        member_vols = [v["fwd_vol_bps_per_horizon"][H] for v in per_asset.values()]
        pv = pooled["fwd_vol_bps_per_horizon"][H]
        assert min(member_vols) <= pv <= max(member_vols), (member_vols, pv)
        assert pv < 10_000.0 / 10.0, f"pooled vol {pv} is the z-score artifact"

        req = {a: sic.assess_record(v, 30).per_horizon[H]["required_ic"]
               for (s, a), v in per_asset.items()}
        req_pool = sic.assess_record(pooled, 30).per_horizon[H]["required_ic"]
        assert min(req.values()) <= req_pool <= max(req.values()), (req, req_pool)
        assert req_pool > 0.01, f"required_ic {req_pool} is the vacuous bar"

    def test_pooled_vol_is_the_n_weighted_rms_of_member_sigmas(self, monkeypatch):
        """Pin the FORMULA against the per-asset rows (same pairs, same
        sample variance): sqrt(sum n_a sigma_a^2 / sum n_a)."""
        per_asset, pooled = _rows(monkeypatch)
        num = sum(v["n_per_horizon"][H] * v["fwd_vol_bps_per_horizon"][H] ** 2
                  for v in per_asset.values())
        den = sum(v["n_per_horizon"][H] for v in per_asset.values())
        assert pooled["fwd_vol_bps_per_horizon"][H] == pytest.approx(
            math.sqrt(num / den), rel=1e-9)
        assert pooled["n_per_horizon"][H] == den, "pooling must keep every pair"

    def test_the_old_arithmetic_would_have_read_ten_thousand(self, monkeypatch):
        """Falsification of the defect, stated as a number: the sigma of the
        z-scored pooled series is ~1.0 -> ~10,000 bps, and the new value must
        be less than a tenth of that. (Computed from the same per-asset rows:
        each member standardized to unit sigma, then pooled.)"""
        per_asset, pooled = _rows(monkeypatch)
        # each member contributes n_a values of unit variance about a zero
        # mean -> pooled sample sd == sqrt(sum (n_a - 1) / (N - 1)) ~ 1.0
        n_each = [v["n_per_horizon"][H] for v in per_asset.values()]
        n_tot = sum(n_each)
        old_vol_bps = math.sqrt(sum(n - 1 for n in n_each) / (n_tot - 1)) * 1e4
        assert old_vol_bps > 9_000.0           # the artifact, reproduced
        new_vol_bps = pooled["fwd_vol_bps_per_horizon"][H]
        assert new_vol_bps < old_vol_bps / 10.0, (old_vol_bps, new_vol_bps)

    def test_pooling_cannot_promote_what_every_member_holds_on_edge(
            self, monkeypatch):
        """The live consequence. Low-dispersion assets + a real high IC: every
        member clears significance and FAILS the edge bar; the pooled row must
        fail it too. Under the old arithmetic the pooled row's edge was
        ~0.8 * r * 10,000 bps -> PROMOTE, from the same data."""
        world = _world(sigmas={"BTC": 0.0004, "ETH": 0.0005, "SOL": 0.0006})
        per_asset, pooled = _rows(monkeypatch, world=world,
                                  recs=_records(world, flip_noise=0.05))
        for (s, a), v in per_asset.items():
            asm = sic.assess_record(v, 30)
            assert asm.verdict is not sic.Verdict.PROMOTE, (a, asm.blockers)
            assert asm.per_horizon[H]["t_stat"] >= 2.0, "fixture control: t clears"
            assert any("edge" in b and "required" in b for b in asm.blockers), (
                a, asm.blockers)
        pa = sic.assess_record(pooled, 30)
        assert pa.per_horizon[H]["t_stat"] >= 2.0
        assert pa.verdict is not sic.Verdict.PROMOTE, pa.blockers
        assert any("edge" in b and "required" in b for b in pa.blockers), pa.blockers
        # and the old arithmetic WOULD have promoted: same IC, vol 10,000
        old = sic.assess_promotion(
            pooled["ic_per_horizon"], pooled["n_per_horizon"],
            pooled["annualized_sharpe"], 30,
            fwd_vol_bps_per_h={H: 10_000.0},
            round_trip_cost_bps=pooled["round_trip_cost_bps"])
        assert old.verdict is sic.Verdict.PROMOTE, (
            "fixture control: with the z-score vol the same row PROMOTES — "
            "that is the defect", old.blockers)

    def test_the_pooled_ic_is_still_computed_on_the_standardized_series(
            self, monkeypatch):
        """P299's part is kept: the rank IC uses the z-scored series. Pin it
        by recomputing the pooled Spearman by hand from per-asset z-scores."""
        world = _world()
        _ohlcv_patch(monkeypatch, world)
        recs = _records(world)
        pooled = sic.compute_per_strategy_ic(recs, horizons_bars=(H,),
                                             pool_assets=True)
        row = pooled[("regimebook", sic.POOLED_KEY)]
        xs_all, zs_all = [], []
        for a in ASSETS:
            ts, closes = world[a]
            xs, ys = [], []
            for r in recs:
                if r["asset"] != a:
                    continue
                i = ts.index(r["_parsed_ts"])
                xs.append(r["direction"] * r["confidence"])
                ys.append(closes[i + H] / closes[i] - 1.0)
            m = sum(ys) / len(ys)
            sd = math.sqrt(sum((y - m) ** 2 for y in ys) / (len(ys) - 1))
            xs_all.extend(xs)
            zs_all.extend((y - m) / sd for y in ys)
        assert row["ic_per_horizon"][H] == pytest.approx(
            sic._spearman(xs_all, zs_all), abs=1e-12)


# ===========================================================================
# 2. End-to-end through the LEDGER and main() — the vacuity test P299 lacked
# ===========================================================================
def _write_ledger(ledger_dir: Path, world, recs):
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for a in ASSETS:
        with (ledger_dir / f"regimebook_{a}.jsonl").open("w", encoding="utf-8") as fh:
            for r in recs:
                if r["asset"] != a:
                    continue
                fh.write(json.dumps({
                    "strategy": r["strategy"], "asset": a,
                    "direction": r["direction"], "confidence": r["confidence"],
                    "ts": r["_parsed_ts"].timestamp()}) + "\n")   # epoch float (P264)


class TestEndToEndThroughTheLedgerAndMain:

    def test_main_pooled_row_carries_real_vol_and_the_cost_source(
            self, monkeypatch, tmp_path, capsys):
        start = datetime.now(timezone.utc) - timedelta(days=28)
        world = _world(start=start)
        _ohlcv_patch(monkeypatch, world)
        recs = _records(world)
        _write_ledger(tmp_path / "ledger", world, recs)
        out_path = tmp_path / "rep.json"
        rc = sic.main(["--ledger-dir", str(tmp_path / "ledger"),
                       "--window-days", "30", "--horizons", str(H),
                       "--pool-assets", "--output", str(out_path)])
        assert rc == 0
        printed = capsys.readouterr().out
        rep = json.loads(out_path.read_text(encoding="utf-8"))
        rows = {(r["strategy"], r["asset"]): r for r in rep["per_strategy"]}
        pooled = rows[("regimebook", sic.POOLED_KEY)]
        vol = pooled["fwd_vol_bps_per_horizon"][str(H)]
        assert 0.0 < vol < 1_000.0, f"pooled vol {vol} — the z-score artifact"
        assert pooled["cost_source"] == sic.COST_SOURCE_CDE
        assert pooled["round_trip_cost_bps"] == pytest.approx(
            max(2.0 * CDE_FEE_BPS[a]["taker"] for a in ASSETS))
        assert sorted(pooled["pooled_assets"]) == sorted(ASSETS)
        # the assessment object in the report carries the same provenance
        assert pooled["cost_source"] == sic.COST_SOURCE_CDE
        assert pooled["per_horizon"][str(H)]["required_ic"] > 0.01
        # and the console prints it per row
        assert f"cost_source={sic.COST_SOURCE_CDE}" in printed
        assert "pooled_assets=" in printed
        assert "core.cde_fees" in printed          # the footer names the model


# ===========================================================================
# 3. The cost bar is the venue's measured fee, floored at the refuted model
# ===========================================================================
class TestCostIsDerivedFromCdeFees:

    @pytest.mark.parametrize("asset", ASSETS)
    def test_shadow_ic_rt_cost_is_at_least_two_cde_taker_legs(self, asset):
        """Drift guard against the PRODUCER: if core.cde_fees is raised, the
        gate's bar rises with it (P310 — import the producer, never restate)."""
        rt, src, by = sic.round_trip_cost_bps_for([asset])
        assert src == sic.COST_SOURCE_CDE
        assert rt >= 2.0 * CDE_FEE_BPS[asset]["taker"]
        assert rt >= sic.REFUTED_MODEL_RT_BPS
        assert by[asset] == rt

    @pytest.mark.parametrize("asset", ASSETS)
    def test_agent_ic_rt_cost_is_at_least_two_cde_taker_legs(self, asset):
        rt, src = air.rt_cost_bps_for([asset])
        assert src == air.COST_SOURCE_CDE
        assert rt >= 2.0 * CDE_FEE_BPS[asset]["taker"]
        assert rt >= air.REFUTED_MODEL_RT_BPS

    def test_the_measured_bar_is_materially_above_the_refuted_model(self):
        """Not a tautology: the CDE bar must actually MOVE the gate. BTC is the
        cheapest asset and even it is ~3x the old 6.0."""
        for a in ASSETS:
            assert sic.round_trip_cost_bps_for([a])[0] > 2.5 * 6.0

    def test_pooled_and_mixed_rows_take_the_dearest_member(self):
        rt_all, _, by = sic.round_trip_cost_bps_for(list(ASSETS))
        assert rt_all == max(by.values())
        assert rt_all == max(sic.round_trip_cost_bps_for([a])[0] for a in ASSETS)
        assert air.rt_cost_bps_for(list(ASSETS))[0] == pytest.approx(rt_all)
        # no attribution -> the same dearest-member bar
        assert sic.round_trip_cost_bps_for(None)[0] == rt_all
        assert air.TAKER_RT_BPS == pytest.approx(rt_all)

    def test_an_unknown_asset_prices_at_the_worst_not_the_floor(self):
        worst = max(2.0 * v["taker"] for v in CDE_FEE_BPS.values())
        assert sic.round_trip_cost_bps_for(["XRP"])[0] == pytest.approx(worst)
        assert air.rt_cost_bps_for(["XRP"])[0] == pytest.approx(worst)

    def test_the_six_bps_floor_binds_when_the_table_is_cheaper(self, monkeypatch):
        """The floor is a FLOOR: a calibration lowered below the refuted model
        cannot drag the gate under it (P167: only ever raise)."""
        monkeypatch.setattr(sic, "_cde_taker_leg_bps", lambda: {"BTC": 1.0})
        rt, src, _ = sic.round_trip_cost_bps_for(["BTC"])
        assert rt == sic.REFUTED_MODEL_RT_BPS and src == sic.COST_SOURCE_CDE
        monkeypatch.setattr(air, "_cde_taker_leg_bps", lambda: {"BTC": 1.0})
        assert air.rt_cost_bps_for(["BTC"])[0] == air.REFUTED_MODEL_RT_BPS

    def test_unreadable_calibration_falls_back_to_the_floor_AND_SAYS_SO(
            self, monkeypatch, caplog, capsys):
        """Never silently: the fallback is logged and named in cost_source."""
        monkeypatch.setitem(sys.modules, "core.cde_fees", None)   # import -> ImportError
        monkeypatch.setattr(sic, "_cde_fallback_warned", False)
        with caplog.at_level("WARNING", logger=sic.logger.name):
            rt, src, by = sic.round_trip_cost_bps_for(["BTC"])
        assert (rt, src, by) == (sic.REFUTED_MODEL_RT_BPS, sic.COST_SOURCE_FALLBACK, {})
        assert any("REFUTED" in r.getMessage() for r in caplog.records)
        rt2, src2 = air.rt_cost_bps_for(["BTC"])
        assert (rt2, src2) == (air.REFUTED_MODEL_RT_BPS, air.COST_SOURCE_FALLBACK)
        assert "REFUTED" in capsys.readouterr().err

    def test_twelve_bps_of_edge_no_longer_promotes_at_a_twenty_bps_fee(
            self, monkeypatch):
        """The bar that mattered: at the refuted 6.0 x 2 = 12 bps a 12.5 bps
        edge PROMOTED. With a 20 bps/leg CDE fee (40 round trip, 80 required)
        it must not — through assess_promotion, and through a record produced
        by compute_per_strategy_ic on that fee table."""
        ic = {4: 0.10, 12: 0.10, 24: 0.10}
        n = {4: 4000, 12: 12000, 24: 24000}     # n_eff = n/h -> t ~ 5.8 each
        vol = {h: 12.5 / sic.expected_edge_bps(0.10, 1.0) for h in ic}   # edge == 12.5
        old = sic.assess_promotion(ic, n, 0.9, 30, fwd_vol_bps_per_h=vol,
                                   round_trip_cost_bps=6.0)
        assert old.verdict is sic.Verdict.PROMOTE, old.blockers
        new = sic.assess_promotion(ic, n, 0.9, 30, fwd_vol_bps_per_h=vol,
                                   round_trip_cost_bps=40.0)
        assert new.verdict is not sic.Verdict.PROMOTE
        assert any("edge" in b and "required" in b for b in new.blockers)
        # and through the producer: a record priced on a 20bps/leg table
        monkeypatch.setattr(sic, "_cde_taker_leg_bps",
                            lambda: {a: 20.0 for a in ASSETS})
        world = _world(sigmas={"BTC": 0.0007, "ETH": 0.0007, "SOL": 0.0007})
        per_asset, pooled = _rows(monkeypatch, world=world,
                                  recs=_records(world, flip_noise=0.02))
        for (s, a), v in per_asset.items():
            assert v["round_trip_cost_bps"] == pytest.approx(40.0)
            asm = sic.assess_record(v, 30)
            assert asm.round_trip_cost_bps == pytest.approx(40.0)
            assert asm.per_horizon[H]["required_bps"] == pytest.approx(80.0)
            assert asm.verdict is not sic.Verdict.PROMOTE, (a, asm.blockers)
        assert sic.assess_record(pooled, 30).verdict is not sic.Verdict.PROMOTE

    def test_agent_ic_twelve_bps_edge_no_longer_promotes(self):
        """Same claim on the agent tool: a clearly varying, significant signal
        whose edge is ~14 bps PROMOTED at 12 bps required; at the CDE bar it
        is a HOLD — and the row says which bar it was priced on."""
        rng = random.Random(1)
        dirs = [rng.choice([-1.0, 1.0]) for _ in range(300)]
        rets = [d * 0.0020 + rng.gauss(0.0, 0.0004) for d in dirs]
        vol = {1: 20.0, 4: 20.0}      # bps: edge ~ 0.8 * 0.95 * 20 ~ 15 bps
        rows, verdict = air.decide_agent_verdict({1: dirs, 4: dirs},
                                                 {1: rets, 4: rets}, vol)
        assert 12.0 < rows[1]["edge_bps"] < 20.0, rows[1]
        assert abs(rows[1]["t"]) >= 2.0 and rows[1]["ic"] > 0
        assert rows[1]["cost_source"] == air.COST_SOURCE_CDE
        assert rows[1]["rt_cost_bps"] == pytest.approx(air.TAKER_RT_BPS, abs=0.01)
        assert rows[1]["required_bps"] >= 2.0 * 2.0 * min(
            v["taker"] for v in CDE_FEE_BPS.values())
        assert verdict != "PROMOTE-CANDIDATE"
        assert rows[1]["clears_p166"] is False
        # control: the same rows on the refuted 6.0 would have cleared
        assert rows[1]["edge_bps"] >= air.SAFETY_MARGIN * air.REFUTED_MODEL_RT_BPS
        assert air.required_ic(20.0, 6.0) < rows[1]["ic"]

    def test_agent_rows_are_priced_on_the_assets_they_signalled_on(self):
        rng = random.Random(2)
        dirs = [rng.choice([-1.0, 1.0]) for _ in range(120)]
        rets = [d * 0.01 + rng.gauss(0.0, 0.003) for d in dirs]
        vol = {1: 100.0, 4: 200.0}
        btc_only = {1: ["BTC"] * 120, 4: ["BTC"] * 120}
        mixed = {1: ["BTC", "ETH"] * 60, 4: ["BTC", "ETH"] * 60}
        r_btc, _ = air.decide_agent_verdict({1: dirs, 4: dirs}, {1: rets, 4: rets},
                                            vol, btc_only)
        r_mix, _ = air.decide_agent_verdict({1: dirs, 4: dirs}, {1: rets, 4: rets},
                                            vol, mixed)
        assert r_btc[1]["rt_cost_bps"] == pytest.approx(2.0 * CDE_FEE_BPS["BTC"]["taker"], abs=0.01)
        assert r_mix[1]["rt_cost_bps"] == pytest.approx(
            2.0 * max(CDE_FEE_BPS["BTC"]["taker"], CDE_FEE_BPS["ETH"]["taker"]), abs=0.01)
        assert r_mix[1]["rt_cost_bps"] > r_btc[1]["rt_cost_bps"]
        # required_ic follows the row's own bar, not the module default
        assert r_btc[1]["required_ic"] == pytest.approx(
            round(air.required_ic(100.0, r_btc[1]["rt_cost_bps"]), 4), abs=1e-4)

    def test_required_ic_default_is_the_module_bar(self):
        """The P230 identity test compares against TAKER_RT_BPS; keep the
        one-argument form bound to it, and the two-argument form honest."""
        assert air.required_ic(150.0) == pytest.approx(
            air.required_ic(150.0, air.TAKER_RT_BPS))
        assert air.required_ic(150.0, 6.0) < air.required_ic(150.0)


# ===========================================================================
# 4. Provenance travels with the number (P169)
# ===========================================================================
class TestProvenanceIsReported:

    def test_per_asset_record_carries_its_own_cost_and_source(self, monkeypatch):
        per_asset, _ = _rows(monkeypatch)
        for (s, a), v in per_asset.items():
            assert v["cost_source"] == sic.COST_SOURCE_CDE
            assert v["cost_assets"] == [a]
            assert v["round_trip_cost_bps"] == pytest.approx(
                max(sic.REFUTED_MODEL_RT_BPS, 2.0 * CDE_FEE_BPS[a]["taker"]))
            asm = sic.assess_record(v, 30)
            assert asm.cost_source == sic.COST_SOURCE_CDE
            assert asm.to_dict()["cost_source"] == sic.COST_SOURCE_CDE
            assert asm.round_trip_cost_bps == pytest.approx(v["round_trip_cost_bps"])

    def test_a_pre_p382_record_without_a_cost_is_judged_on_the_floor_and_says_so(self):
        rec = {"ic_per_horizon": {4: 0.12}, "n_per_horizon": {4: 4000},
               "annualized_sharpe": 0.9, "fwd_vol_bps_per_horizon": {4: 400.0}}
        asm = sic.assess_record(rec, 30)
        assert asm.round_trip_cost_bps == sic.DEFAULT_ROUND_TRIP_COST_BPS
        assert "record_missing_cost" in asm.cost_source
        assert sic.COST_SOURCE_FALLBACK in asm.cost_source

    def test_render_summary_prints_the_cost_and_its_source_per_row(self, monkeypatch):
        per_asset, pooled = _rows(monkeypatch)
        table = dict(per_asset)
        table[("regimebook", sic.POOLED_KEY)] = pooled
        text = sic.render_summary(table, 30, (H,))
        assert text.count(f"cost_source={sic.COST_SOURCE_CDE}") == len(table)
        assert "pooled_assets=BTC,ETH,SOL" in text
        assert "core.cde_fees" in text and "floor 6.0" in text
        assert "cost model: 6.0bps round trip" not in text, (
            "the footer must not advertise the refuted model when the "
            "calibration was read")

    def test_render_summary_footer_names_the_fallback_when_unreadable(
            self, monkeypatch):
        per_asset, _ = _rows(monkeypatch)
        monkeypatch.setattr(sic, "_cde_taker_leg_bps", lambda: None)
        text = sic.render_summary(per_asset, 30, (H,))
        assert sic.COST_SOURCE_FALLBACK in text and "REFUTED" in text

    def test_agent_ic_report_carries_the_cost_model(self):
        rep = air.build_report(30, {1: 100.0, 4: 200.0}, "now")
        cm = rep["cost_model"]
        assert cm["default_rt_bps"] == pytest.approx(air.TAKER_RT_BPS)
        assert cm["cost_source"] == air.COST_SOURCE
        assert cm["floor_rt_bps"] == 6.0
        assert cm["safety_margin"] == air.SAFETY_MARGIN
        # the P312 shape contract is untouched
        assert air.AGENTS_CONTAINER_KEY in rep and rep[air.AGENTS_CONTAINER_KEY] == {}

    def test_the_refuted_literal_is_a_floor_not_the_bar(self):
        """Source pins on the two constants, so a revert to a bare 6.0 bar is
        caught even if a future test suite stops exercising the arithmetic."""
        s1 = (REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py").read_text(
            encoding="utf-8-sig")
        s2 = (REPO / "analytics" / "ic" / "agent_ic_review.py").read_text(
            encoding="utf-8-sig")
        assert "REFUTED_MODEL_RT_BPS = DEFAULT_ROUND_TRIP_COST_BPS" in s1
        assert "from core.cde_fees import CDE_FEE_BPS" in s1
        assert "TAKER_RT_BPS, COST_SOURCE = rt_cost_bps_for(None)" in s2
        assert "TAKER_RT_BPS = 6.0" not in s2
        assert "from core.cde_fees import CDE_FEE_BPS" in s2
