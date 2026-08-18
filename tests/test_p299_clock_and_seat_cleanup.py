"""
[P299] The four standing decisions, executed.

  1. The P237 tripwire is SUPERSEDED — it reports, it no longer prescribes.
  2. Shadow clocks: pooled scoring (the change that makes a 16h clock able to
     fire at all) + the measured-dead legacy families archived.
  3. regimebook/SOL relabelled trend-only, and an UNAVAILABLE book can no
     longer reach the live seat.
  4. CryptoPanic's monthly-quota backoff stops guessing the vendor's reset
     date.

Every test here pins a property that a plausible "cleanup" would break.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# =============================================================================
# 1. The tripwire reports, it does not prescribe
# =============================================================================

class TestTripwireIsEvidenceNotAnInstruction:

    def _write(self, d: Path, day: str, tradeable: bool):
        v = "TRADEABLE" if tradeable else "GATE-CLOSED"
        (d / f"slope_{day.replace('-', '')}_000000.json").write_text(
            json.dumps({
                "generated": f"{day}T06:20:00+00:00",
                "assets": {a: {"4": {"vs_threshold": v}}
                           for a in ("BTC", "ETH", "SOL")},
            }), encoding="utf-8")

    def _run(self, d: Path, today: str):
        return subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "analytics" / "calibration" / "tripwire_check.py"),
             "--reports-dir", str(d), "--today", today],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))

    def test_a_full_streak_is_detected_but_never_exits_3(self, tmp_path):
        """The DETECTION must survive the retirement — a GATE-CLOSED streak is
        real evidence and feeds the seat decision. Only the prescription goes.
        """
        for day in ("2026-08-11", "2026-08-18", "2026-08-25", "2026-09-01"):
            self._write(tmp_path, day, tradeable=False)
        r = self._run(tmp_path, "2026-09-08")
        assert r.returncode == 0, (
            "exit 3 meant 'act on one variable'; the seat controller owns "
            f"that now\nstdout={r.stdout}\nstderr={r.stderr}")
        assert "FIRED" in r.stdout, "the streak must still be counted"
        assert "SUPERSEDED" in r.stdout

    def test_it_tells_the_reader_not_to_act_on_the_old_instruction(self, tmp_path):
        """A retired prescription that still reads like an instruction is
        worse than one that was deleted."""
        for day in ("2026-08-11", "2026-08-18", "2026-08-25", "2026-09-01"):
            self._write(tmp_path, day, tradeable=False)
        out = self._run(tmp_path, "2026-09-08").stdout
        assert "do NOT edit trend_assets" in out
        assert "seat_check" in out

    def test_no_reports_still_REFUSES(self, tmp_path):
        """P199 — 'cannot be evaluated' must never read as 'not fired', and
        retiring the firing rule must not quietly retire the refusal."""
        assert self._run(tmp_path, "2026-09-08").returncode == 2

    def test_the_actuator_reason_is_recorded_at_source(self):
        """The WHY must live in the code, not only in a changelog: the
        actuator targets trend, and trend is no longer the decider."""
        src = (REPO / "analytics" / "calibration"
               / "tripwire_check.py").read_text(encoding="utf-8-sig")
        assert "no longer targets the decider" in src.lower() or \
               "NO LONGER TARGETS THE DECIDER" in src
        assert "SENT-SWITCH" in src, (
            "the reason removing trend is NOT de-risking (it hands ticks to "
            "an unvalidated path) must be stated where someone would undo it")


# =============================================================================
# 2. Shadow clocks — pooling, and the archived families
# =============================================================================

class TestPooledScoringMakesTheClockAbleToFire:

    def _recs(self, strategy, assets, n=40, sign=1.0):
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)
        out = []
        for a in assets:
            for i in range(n):
                out.append({
                    "strategy": strategy, "asset": a,
                    "direction": sign * (1.0 if i % 2 else -1.0),
                    "confidence": 1.0,
                    "_parsed_ts": base + timedelta(hours=4 * i),
                })
        return out

    def test_pooling_triples_n_for_a_one_rule_family(self, monkeypatch):
        """The whole point: one rule on three assets is ONE exam. Scored
        per-asset it is three underpowered ones, and at 16h a 30-day window
        cannot certify an economically-adequate IC at all (P293g)."""
        import analytics.shadow_ic.compute_shadow_ic as mod
        import pandas as pd

        idx = pd.date_range("2026-08-01", periods=400, freq="4h", tz="UTC")
        monkeypatch.setattr(mod, "load_ohlcv", lambda a: pd.DataFrame({
            "timestamp": idx,
            "close": [100.0 + (i % 7) * (1 + len(a)) for i in range(len(idx))]}))

        recs = self._recs("regimebook", ["BTC", "ETH", "SOL"])
        per_asset = mod.compute_per_strategy_ic(recs, horizons_bars=(4,),
                                                pool_assets=False)
        pooled = mod.compute_per_strategy_ic(recs, horizons_bars=(4,),
                                             pool_assets=True)

        assert len(per_asset) == 3, "un-pooled must stay one row per asset"
        assert list(pooled) == [("regimebook", mod.POOLED_KEY)]
        n_pool = pooled[("regimebook", mod.POOLED_KEY)]["n_per_horizon"][4]
        n_each = sum(v["n_per_horizon"][4] for v in per_asset.values())
        assert n_pool == n_each, (
            f"pooling must keep every observation: {n_pool} vs {n_each}")
        assert n_pool > max(v["n_per_horizon"][4] for v in per_asset.values())

    def test_a_per_asset_family_is_NOT_pooled_even_with_the_flag(self, monkeypatch):
        """Merging genuinely per-asset claims into one number is the P294
        defect (a shared name silently merged two exams). The allowlist is
        the guard, so it must actually be consulted."""
        import analytics.shadow_ic.compute_shadow_ic as mod
        import pandas as pd
        idx = pd.date_range("2026-08-01", periods=400, freq="4h", tz="UTC")
        monkeypatch.setattr(mod, "load_ohlcv", lambda a: pd.DataFrame(
            {"timestamp": idx,
             "close": [100.0 + i % 5 for i in range(len(idx))]}))

        recs = self._recs("some_per_asset_thing", ["BTC", "ETH"])
        pooled = mod.compute_per_strategy_ic(recs, horizons_bars=(4,),
                                             pool_assets=True)
        assert len(pooled) == 2, "an unlisted family must stay per-asset"
        assert mod.POOLED_KEY not in {k[1] for k in pooled}

    def test_forward_returns_are_standardized_within_asset_before_pooling(self):
        """Without it a high-vol asset owns the extreme ranks and the pooled
        Spearman mostly measures that one asset while reporting a 3x n."""
        src = (REPO / "analytics" / "shadow_ic"
               / "compute_shadow_ic.py").read_text(encoding="utf-8-sig")
        assert "_by_asset" in src
        assert "standardiz" in src.lower()

    def test_a_constant_asset_series_is_dropped_not_divided_by_zero(self, monkeypatch):
        """A zero-dispersion series carries no rank information; fabricating
        one would be P2 with extra steps."""
        import analytics.shadow_ic.compute_shadow_ic as mod
        import pandas as pd
        idx = pd.date_range("2026-08-01", periods=400, freq="4h", tz="UTC")

        def _ohlcv(a):
            close = ([100.0] * len(idx) if a == "SOL"
                     else [100.0 + i % 5 for i in range(len(idx))])
            return pd.DataFrame({"timestamp": idx, "close": close})

        monkeypatch.setattr(mod, "load_ohlcv", _ohlcv)
        recs = self._recs("regimebook", ["BTC", "SOL"])
        out = mod.compute_per_strategy_ic(recs, horizons_bars=(4,),
                                          pool_assets=True)
        row = out[("regimebook", mod.POOLED_KEY)]
        assert row["n_per_horizon"][4] > 0, "the healthy asset must still score"

    def test_the_dead_families_are_archived_with_a_measured_reason(self):
        """An archive without its measurement is an opinion. These three were
        re-measured on the live volume: cascade 0/486 directional over 9d."""
        from analytics.shadow_ic.compute_shadow_ic import ARCHIVED_FAMILIES
        assert set(ARCHIVED_FAMILIES) == {"cascade", "microstructure", "funding"}
        for fam, reason in ARCHIVED_FAMILIES.items():
            assert any(ch.isdigit() for ch in reason), (
                f"{fam}'s archive reason must carry the measurement, not just "
                f"an assertion")
        assert "ml_factor" not in ARCHIVED_FAMILIES, (
            "ml_factor measured 156/243 directional — it is alive")

    def test_poolable_list_holds_only_one_rule_families(self):
        from analytics.shadow_ic.compute_shadow_ic import (
            POOLABLE_FAMILIES, ARCHIVED_FAMILIES)
        assert not (set(POOLABLE_FAMILIES) & set(ARCHIVED_FAMILIES)), (
            "an archived family must not also be advertised as poolable")
        assert "regimebook" in POOLABLE_FAMILIES
        assert "mlpshadow" not in POOLABLE_FAMILIES, (
            "mlpshadow is a BTC-only exported model, not one rule over assets")


# =============================================================================
# 3. regimebook/SOL, and the live seat
# =============================================================================

class TestSolBookAndTheSeat:

    def test_sol_is_trend_only_and_behaviour_is_unchanged(self):
        from defense.regime_book_shadow import BOOKS_VERSION, book_target
        assert BOOKS_VERSION["SOL"] == "v1_trend_only"
        assert book_target("SOL", "bull", None)[0] == 1.0
        assert book_target("SOL", "peace", 3.0)[0] == 0.0
        assert book_target("SOL", "bear", 3.0)[0] == 0.0

    def test_sol_and_eth_take_the_same_leg_outside_bull(self):
        """The claim being made: SOL runs ETH's certified trend-only book."""
        from defense.regime_book_shadow import book_target
        assert book_target("SOL", "peace", 1.0)[1] == book_target("ETH", "peace", 1.0)[1]

    def test_an_unavailable_record_is_not_served_to_the_seat(self, tmp_path):
        """THE LIVE ONE. main.py's regimebook seat assigns quant_direction
        unconditionally — 0.0 included — so serving an unavailable book lets
        something that CANNOT hold a position flatten the incumbent."""
        from defense.regime_book_shadow import RegimeBookShadow
        h = RegimeBookShadow(data_dir=str(tmp_path))
        h._last_records["SOL"] = {"ts": time.time(), "direction": 0.0,
                                  "leg": "flat_degraded", "available": False}
        assert h.last_direction("SOL") is None

    def test_an_available_book_choosing_flat_IS_served(self, tmp_path):
        """The mirror image, and the reason the guard reads `available` and
        not `direction`: 'be flat' is a position and must reach the seat."""
        from defense.regime_book_shadow import RegimeBookShadow
        h = RegimeBookShadow(data_dir=str(tmp_path))
        h._last_records["ETH"] = {"ts": time.time(), "direction": 0.0,
                                  "leg": "trend_flat", "available": True}
        got = h.last_direction("ETH")
        assert got is not None and got[0] == 0.0

    def test_a_record_predating_the_available_field_is_still_served(self, tmp_path):
        """Old rows have no `available` key. Treating missing as False would
        silently mute every book on the first tick after a rollback."""
        from defense.regime_book_shadow import RegimeBookShadow
        h = RegimeBookShadow(data_dir=str(tmp_path))
        h._last_records["BTC"] = {"ts": time.time(), "direction": 1.0,
                                  "leg": "hold"}
        assert h.last_direction("BTC") is not None

    def test_no_bear_leg_came_back(self, tmp_path):
        from defense.regime_book_shadow import RegimeBookShadow
        h = RegimeBookShadow(data_dir=str(tmp_path))
        h._sol_model = None
        target, leg, version, _ = h._sol_bear_target()
        assert target == 0.0 and version == "v1_trend_only" and "bear" not in leg


# =============================================================================
# 4. CryptoPanic — stop guessing the vendor's reset date
# =============================================================================

class TestQuotaBackoffDoesNotGuessTheResetDate:

    def test_backoff_is_about_a_day_not_a_month(self):
        from data_mgmt.feeds.cryptopanic_feed import quota_backoff_until
        now = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
        until = quota_backoff_until(now)
        assert (until - now) <= timedelta(hours=25), (
            "a month-long lockout assumes a calendar reset; if the plan "
            "resets on its subscription anniversary that is ~19 dark days "
            "per month, forever")
        assert (until - now) >= timedelta(hours=20)

    def test_it_never_probes_past_the_month_boundary(self):
        """The month is a CEILING — the quota cannot fail to have reset by
        then, so there is no reason to wait longer."""
        from data_mgmt.feeds.cryptopanic_feed import quota_backoff_until
        now = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)
        until = quota_backoff_until(now)
        assert until == datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_december_rolls_the_year(self):
        from data_mgmt.feeds.cryptopanic_feed import quota_backoff_until
        until = quota_backoff_until(
            datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc))
        assert until == datetime(2027, 1, 1, tzinfo=timezone.utc)

    def test_it_is_still_far_cheaper_than_the_pre_p293b_hammering(self):
        """P293b's real defect was retrying every 15 minutes for a month
        (~2,900 requests). Daily is ~96x cheaper and cannot regress to that."""
        from data_mgmt.feeds.cryptopanic_feed import QUOTA_REPROBE_SEC
        assert QUOTA_REPROBE_SEC >= 6 * 3600, (
            "anything sub-6h drifts back toward hammering a quota that is "
            "genuinely exhausted for the rest of the month")

    def test_the_consumer_uses_the_helper_not_the_month_start(self):
        """A source pin, because the two helpers differ only in the constant
        they return and a revert would be invisible in behaviour tests that
        only run inside one day."""
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        i = src.index("_quota_exhausted = True")
        block = src[i:i + 400]
        assert "quota_backoff_until(" in block
        assert "_next_month_start_utc(_now)" not in block
