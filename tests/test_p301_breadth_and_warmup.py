"""
================================================================================
HMATS [P301] - breadth funding + persisted v5.1 warmups
================================================================================

Operator: "do both" - fetch the breadth funding so the derivatives exam can
price carry, and persist the v5.1 warmups so three "dead" agents become
measurable.
================================================================================
"""

import importlib
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FUNDING_DIR = REPO / "training" / "training_data" / "coinglass_history"
BREADTH = ("XRP", "ADA", "LTC", "DOGE", "BNB")


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestBreadthFundingIsFetchable:
    """The exam could not price carry for these five, and carry is the
    dominant drag on a long-biased perp book (P296: -59.7% over six years)."""

    def test_fetcher_knows_the_breadth_symbols(self):
        from training.scripts.fetch_binance_funding import SYMBOLS
        for a in BREADTH:
            assert a in SYMBOLS, f"{a} has no Binance symbol mapping"
            assert SYMBOLS[a].endswith("USDT")

    def test_fetcher_refuses_an_unknown_asset(self):
        """[P291] a fetcher that accepts any string writes a file for an asset
        that does not exist and reports success."""
        from training.scripts.fetch_binance_funding import main
        assert main(["--assets", "NOTACOIN"]) == 2

    def test_fetcher_merges_rather_than_overwrites(self):
        """[P266] a partial outage must not silently replace a complete
        series with a truncated one."""
        src = _src(REPO / "training" / "scripts" / "fetch_binance_funding.py")
        assert "drop_duplicates(\"timestamp\", keep=\"last\")" in src
        assert "if out.exists():" in src

    @pytest.mark.skipif(not FUNDING_DIR.exists(), reason="no local archives")
    def test_the_five_archives_landed_with_major_span(self):
        import pandas as pd
        spans = {}
        for a in BREADTH + ("BTC",):
            p = FUNDING_DIR / f"{a}_funding_1d.parquet"
            if not p.exists():
                pytest.skip(f"{a} archive not fetched on this machine")
            spans[a] = len(pd.read_parquet(p))
        # every breadth asset must reach the majors' depth, or the exam is
        # comparing a 6-year book against a partial one
        assert all(spans[a] >= spans["BTC"] * 0.95 for a in BREADTH), spans


class TestWarmupSurvivesRestart:
    """MEASURED on the server: funding reported history_warmup(1/12) and
    regime_warmup(1/30). Those counters read 1 - not 11, not 29 - because the
    deques start empty on every construction, so a warmup needing ~4 and ~10
    days of uninterrupted uptime never completed between deploys. P154/P148/
    P150 class: state that re-arms on restart is not state."""

    @pytest.fixture()
    def isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import strategies._warmup_state as ws
        importlib.reload(ws)
        import strategies.funding_rate_v5_1 as F
        importlib.reload(F)
        return F

    def test_history_survives_a_restart(self, isolated):
        F = isolated
        s = F.FundingRateMeanReversionStrategy()
        for i in range(8):
            s.evaluate("BTC", {"funding_rate_8h": 0.0001 * (i + 1)})
        assert len(s._history["BTC"]) == 8

        fresh = F.FundingRateMeanReversionStrategy()   # a restart
        assert len(fresh._history.get("BTC", [])) == 8, (
            "the warmup restarted at 0 — the counter will read 1/12 forever "
            "at this deploy cadence"
        )
        out = fresh.evaluate("BTC", {"funding_rate_8h": 0.0009})
        assert "9/12" in out.reason, out.reason

    def test_the_regime_strategy_persists_too(self, isolated):
        F = isolated
        s = F.FundingRatePostETFRegimeStrategy()
        for i in range(5):
            s.evaluate("ETH", {"funding_rate_8h": 0.0002 * (i + 1)})
        assert len(F.FundingRatePostETFRegimeStrategy()._history.get("ETH", [])) == 5

    def test_the_two_strategies_do_not_share_a_file(self, isolated):
        """Distinct maxlens and distinct thresholds — one file would let the
        60-day regime history feed the 14-day mean-reversion z-score."""
        F = isolated
        assert (F.FundingRateMeanReversionStrategy._WARMUP_NAME
                != F.FundingRatePostETFRegimeStrategy._WARMUP_NAME)

    def test_a_missing_state_file_is_a_cold_start_not_an_error(self, isolated):
        F = isolated
        s = F.FundingRateMeanReversionStrategy()
        assert s._history == {}

    def test_a_corrupt_state_file_degrades_to_cold_start(self, isolated, tmp_path):
        import strategies._warmup_state as ws
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "funding_mean_reversion.json").write_text("{not json",
                                                       encoding="utf-8")
        assert ws.load("funding_mean_reversion") == {}

    def test_a_stale_history_is_dropped_rather_than_trusted(self, isolated,
                                                            tmp_path):
        """A funding distribution from two months ago is not the current
        regime; restoring it would give the z-score a scale nobody measured."""
        import json
        import strategies._warmup_state as ws
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stale.json").write_text(json.dumps({
            "version": "v5_1_warmup_v1",
            "saved_ts": 0.0,                      # 1970
            "series": {"BTC": [0.1, 0.2, 0.3]},
        }), encoding="utf-8")
        assert ws.load("stale") == {}

    def test_nan_values_never_enter_the_history(self, isolated, tmp_path):
        import json
        import strategies._warmup_state as ws
        import time
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "n.json").write_text(json.dumps({
            "version": "v5_1_warmup_v1",
            "saved_ts": time.time(),
            "series": {"BTC": [0.1, float("nan"), 0.3]},
        }), encoding="utf-8")
        assert ws.load("n") == {"BTC": [0.1, 0.3]}

    def test_persisting_never_raises_into_the_tick(self, isolated, monkeypatch):
        """A warmup that cannot be saved must not take a tick down."""
        F = isolated
        import strategies.funding_rate_v5_1 as mod

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "_warmup_save", boom)
        s = F.FundingRateMeanReversionStrategy()
        out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
        assert out is not None


class TestNoSilentIndexRealignment:
    """[P301] `pd.Series(df["col"], index=other)` ALIGNS on the Series' own
    index instead of replacing it, so a non-overlapping index yields all-NaN.

    Measured: this turned the breadth exam's entire subject - the funding
    carry - into exactly 0.0% for all eight assets, and `fillna(0.0)` made it
    look like a measurement. It survived because a zero is a plausible-looking
    number; it was caught only because a 0.0 six-year funding bill is not.

    The dangerous shape is specifically a PANDAS expression (`df[col]`, or an
    `.astype()` chain on one) passed as `data` alongside `index=`. A plain
    list or a numpy array has no index and is safe, so the scan is written to
    ignore those rather than cry wolf.
    """

    @staticmethod
    def _is_pandas_expr(node):
        import ast
        # df["col"] / df.loc[...] — a subscript on a name or attribute
        if isinstance(node, ast.Subscript):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("to_numpy", "values", "tolist", "to_list"):
                return False        # explicitly de-indexed: the correct form
            if node.func.attr in ("astype", "fillna", "round", "abs", "copy"):
                return TestNoSilentIndexRealignment._is_pandas_expr(
                    node.func.value)
        return False

    def _scan(self, src):
        import ast
        out = []
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return out
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else (f.id if isinstance(f, ast.Name) else None))
            if name not in ("Series", "DataFrame") or not n.args:
                continue
            if not any(k.arg == "index" for k in n.keywords):
                continue
            if self._is_pandas_expr(n.args[0]):
                out.append(n.lineno)
        return out

    def test_the_scan_bites_on_the_exact_bug_that_happened(self):
        """[P174] anti-vacuity, using the real line that produced the zero."""
        bad = 'pd.Series(df["funding_close"].astype(float), index=idx)\n'
        assert self._scan(bad) == [1]
        good = ('pd.Series(df["funding_close"].to_numpy(dtype=float), index=idx)\n'
                "pd.Series(vals, index=idx)\n"
                "pd.Series([1, 2, 3], index=idx)\n"
                "pd.Series(np.zeros(3), index=idx)\n")
        assert self._scan(good) == []

    def test_no_such_site_survives_anywhere(self):
        import io
        import os
        skip = {"venv", "archive", "__pycache__", ".git", "node_modules",
                "models", "training_data"}
        offenders = []
        for dp, dn, fn in os.walk(REPO):
            dn[:] = [d for d in dn if d not in skip]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = Path(dp) / f
                try:
                    src = io.open(p, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                for ln in self._scan(src):
                    offenders.append(f"{p.relative_to(REPO)}:{ln}")
        assert not offenders, (
            "pandas will ALIGN on the source index here, not replace it — "
            "a non-overlapping index yields silent all-NaN:\n  "
            + "\n  ".join(offenders)
        )


class TestMicrostructureStateSurvivesRestart:
    """[P302] Of 6,168 recorded microstructure rows exactly ONE was
    directional. Both of its silences came from RAM-only state, not from a
    quiet market:

        no_prev_price          `_last_price` starts empty every process, so
                               the return since the last tick cannot exist
        z_below_threshold(+0.00)  the lambda history needs >= max(8, 0.3*30)
                               samples and restarts at 0

    At one sample per 4H tick that is ~1.5 days of uninterrupted uptime.
    Same class as the P301 funding warmups and P154/P148/P150 before them.
    """

    @pytest.fixture()
    def isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import strategies._warmup_state as ws
        importlib.reload(ws)
        import strategies.microstructure_v5_1 as M
        importlib.reload(M)
        return M

    @staticmethod
    def _md(px):
        return {"current_price": px, "order_book_imbalance": 0.1,
                "spread_bps": 3.0}

    def test_previous_price_survives_a_restart(self, isolated):
        M = isolated
        s = M.KyleLambdaStrategy()
        assert s.evaluate("BTC", self._md(64000)).reason == "no_prev_price"
        fresh = M.KyleLambdaStrategy()
        assert fresh._last_price.get("BTC") == 64000, (
            "every restart would otherwise burn a tick per asset on "
            "no_prev_price"
        )
        assert fresh.evaluate("BTC", self._md(64100)).reason != "no_prev_price"

    def test_lambda_history_survives_a_restart(self, isolated):
        M = isolated
        s = M.KyleLambdaStrategy()
        for i in range(10):
            s.evaluate("BTC", self._md(64000 + i * 40))
        n = len(s._lambda_history["BTC"])
        assert n >= 8
        assert len(M.KyleLambdaStrategy()._lambda_history.get("BTC", [])) == n

    def test_the_price_key_cannot_collide_with_an_asset(self, isolated):
        """`_last_price` and the lambda history share one file; the price
        rows are suffixed so an asset literally named like the suffix cannot
        overwrite a history."""
        M = isolated
        s = M.KyleLambdaStrategy()
        for i in range(3):
            s.evaluate("BTC", self._md(64000 + i * 40))
        s._persist()
        import strategies._warmup_state as ws
        blob = ws.load("kyle_lambda")
        assert "BTC::price" in blob and "BTC" in blob
        assert blob["BTC::price"] != blob["BTC"]

    def test_a_failed_save_never_raises_into_the_tick(self, isolated,
                                                      monkeypatch):
        M = isolated
        import strategies.microstructure_v5_1 as mod

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "_warmup_save", boom)
        s = M.KyleLambdaStrategy()
        assert s.evaluate("BTC", self._md(64000)) is not None


class TestSeatSurvivesRestart:
    """[P303] `tick()` runs at LOOP level AFTER decide (P248), so the first
    tick of every process had no target to seat and logged "no fresh book
    target exists" - measured live on 2026-08-18 for all three assets right
    after a deploy, while the ledger already held BTC=bear/funding_short/-1.0.

    With regimebook_mode: enforce that costs the CERTIFIED book (P297) a full
    4H tick per deploy, on an engine that deploys often. Same restart class as
    the P301/P302 warmups, on the signal now holding the seat.
    """

    def _ledger(self, tmp_path, name, rec):
        import json
        d = tmp_path / "strategy_shadow"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.jsonl").write_text(json.dumps(rec) + "\n",
                                         encoding="utf-8")

    def test_the_seat_has_a_target_on_the_first_tick(self, tmp_path):
        import time
        from defense.regime_book_shadow import RegimeBookShadow
        self._ledger(tmp_path, "regimebook_BTC",
                     {"ts": time.time(), "asset": "BTC",
                      "direction": -1.0, "leg": "funding_short"})
        h = RegimeBookShadow(data_dir=str(tmp_path))
        got = h.last_direction("BTC")
        assert got is not None, "the seat still loses a tick to every restart"
        assert got[0] == -1.0 and got[1] == "funding_short"

    def test_sibling_exams_are_never_mistaken_for_the_book(self, tmp_path):
        """regimebook_adj / _banded / _volskip are SEPARATE exams; seating one
        would put an unpromoted overlay on the decider slot."""
        import time
        from defense.regime_book_shadow import RegimeBookShadow
        now = time.time()
        self._ledger(tmp_path, "regimebook_adj_BTC",
                     {"ts": now, "asset": "BTC", "direction": 1.0, "leg": "x"})
        self._ledger(tmp_path, "regimebook_volskip_ETH",
                     {"ts": now, "asset": "ETH", "direction": 1.0, "leg": "x"})
        h = RegimeBookShadow(data_dir=str(tmp_path))
        assert h._last_records == {}
        assert h.last_direction("BTC") is None
        assert h.last_direction("ETH") is None

    def test_a_stale_ledger_yields_no_seat_rather_than_a_stale_one(self, tmp_path):
        """[P2] absent is not flat, and old is not fresh - last_direction's
        own 6h bound still governs whatever is restored."""
        import time
        from defense.regime_book_shadow import RegimeBookShadow
        self._ledger(tmp_path, "regimebook_BTC",
                     {"ts": time.time() - 48 * 3600, "asset": "BTC",
                      "direction": -1.0, "leg": "funding_short"})
        h = RegimeBookShadow(data_dir=str(tmp_path))
        assert h.last_direction("BTC") is None

    def test_a_corrupt_ledger_is_a_cold_start_not_a_crash(self, tmp_path):
        from defense.regime_book_shadow import RegimeBookShadow
        d = tmp_path / "strategy_shadow"
        d.mkdir(parents=True, exist_ok=True)
        (d / "regimebook_BTC.jsonl").write_text("{not json\n", encoding="utf-8")
        h = RegimeBookShadow(data_dir=str(tmp_path))
        assert h.last_direction("BTC") is None


class TestCascadeCanDetectACascade:
    """[P304] Every window the CascadeExhaustionGovernor was fed is a UNIFORM
    SHARE of one 24h aggregate: liq_24h/24 as the "1h" volume, liq_24h/6 as
    the "4h", and price_change_4h_pct/4 as the "1h" move. Dividing a 24h
    total by 24 is exactly the operation that erases a burst - and a cascade
    IS a burst.

    Measured 2026-08-18 on the live ledger: a $36.6M day feeds $1.53M against
    a $10M cascade_detect_liq_threshold, so it would take a $240M day to trip,
    and even then it would be reading a sustained average rather than a spike.
    """

    def _runner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import main
        return main.HMATSProductionRunner.__new__(main.HMATSProductionRunner)

    def test_the_average_cannot_reach_the_threshold(self):
        """The arithmetic that makes this a defect and not a tuning choice."""
        from risk.cascade_exhaustion_governor import CascadeExhaustionConfig as CascadeConfig
        thr = CascadeConfig().cascade_detect_liq_threshold
        observed_24h = 36_607_740          # a real reading from the ledger
        assert observed_24h / 24.0 < thr, "premise: the average is under the bar"
        assert observed_24h / 24.0 < thr / 6, (
            "it is not marginally under - it is an order of magnitude under, "
            "which is why the detector has never fired"
        )

    def _age_state(self, tmp_path, asset, hours):
        """Simulate a gap by ageing the persisted stamp."""
        import json
        p = tmp_path / "cascade_liq_prev.json"
        st = json.loads(p.read_text(encoding="utf-8"))
        st[asset]["ts"] -= hours * 3600
        p.write_text(json.dumps(st), encoding="utf-8")

    def test_a_burst_is_visible_as_an_hourly_rate(self, tmp_path, monkeypatch):
        """[P305] The delta is a ~4h figure because ticks are 4H apart, so it
        is returned as a per-HOUR rate. Feeding the raw delta to the 1h slot
        overstated the hourly rate 4x - the same units defect the fix exists
        to correct, committed inside the correction."""
        from risk.cascade_exhaustion_governor import CascadeExhaustionConfig as CascadeConfig
        r = self._runner(tmp_path, monkeypatch)
        assert r._cascade_liq_delta("BTC", 36_607_740) is None, "no previous yet"
        self._age_state(tmp_path, "BTC", 4)
        rate = r._cascade_liq_delta("BTC", 48_607_740)          # +$12M over 4h
        assert rate == pytest.approx(3_000_000, rel=1e-3), (
            "a $12M/4h burst is a $3M/h rate, not a $12M hourly volume"
        )
        assert rate * 4 >= CascadeConfig().cascade_detect_liq_threshold, (
            "scaled to its own window the burst still clears the $10M bar - "
            "which the 24h average ($1.53M) never could"
        )

    def test_an_unusable_interval_is_refused(self, tmp_path, monkeypatch):
        """Too short divides into a huge rate; past 24h the rolling window has
        fully turned over, so the delta measures nothing."""
        r = self._runner(tmp_path, monkeypatch)
        r._cascade_liq_delta("BTC", 36_000_000)
        assert r._cascade_liq_delta("BTC", 48_000_000) is None, "gap too short"
        self._age_state(tmp_path, "BTC", 60)
        assert r._cascade_liq_delta("BTC", 60_000_000) is None, "gap too long"

    def test_the_first_reading_is_absent_not_zero(self, tmp_path, monkeypatch):
        """[P2] None falls back to the old behaviour; a 0.0 would claim 'no
        liquidations happened', which is a measurement nobody made."""
        r = self._runner(tmp_path, monkeypatch)
        assert r._cascade_liq_delta("ETH", 10_000_000) is None

    def test_window_rolloff_never_reports_negative_liquidations(self, tmp_path,
                                                                monkeypatch):
        r = self._runner(tmp_path, monkeypatch)
        r._cascade_liq_delta("BTC", 50_000_000)
        self._age_state(tmp_path, "BTC", 4)
        assert r._cascade_liq_delta("BTC", 40_000_000) == 0.0

    def test_the_previous_reading_survives_a_restart(self, tmp_path, monkeypatch):
        """A per-process cache would reset on every deploy and the delta would
        be None forever - the P301/P302/P303 class that made three other
        signals look dead."""
        r1 = self._runner(tmp_path, monkeypatch)
        r1._cascade_liq_delta("BTC", 36_000_000)
        self._age_state(tmp_path, "BTC", 4)
        r2 = self._runner(tmp_path, monkeypatch)      # a restart
        assert r2._cascade_liq_delta("BTC", 44_000_000) == pytest.approx(
            2_000_000, rel=1e-3)   # $8M over 4h = $2M/h

    def test_arming_is_a_config_decision_and_defaults_off(self):
        """cascade_phase reaches a fusion disable-condition
        (authority_fusion.py:851), so a detector that has never fired is armed
        deliberately or not at all (P141)."""
        import json
        from main import ProductionConfig
        assert ProductionConfig().cascade_real_liquidation_window is False
        live = json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        assert "cascade_real_liquidation_window" not in live

    def test_both_numbers_are_logged_so_the_shadow_can_be_read(self):
        """[P300] an instrument gated behind the condition it is meant to
        measure never reads anything - this one logs every tick regardless."""
        src = (REPO / "main.py").read_text(encoding="utf-8")
        assert "[CASCADE-OBSERVE]" in src
        assert "fed_1h(avg)" in src and "observed_since_last" in src
