"""[P390] KalmanCointegration_SOL_ETH and RelativeStrength warmups survive a
restart — the P358c starvation removed.

P358c measured both as WARMUP-starved: in-memory deques fed ~3 appends per 4H
tick with NO persistence, so the >=50-price (+ >=30-spread) warmups needed
2.8-4.5 days of UNINTERRUPTED uptime and neither strategy has EVER fired.
P390 persists the buffers (and, for Kalman, the recursive filter state theta/P
— which is incrementally updated across calls, never recomputed from the
buffers, so it must travel WITH the spreads it produced) through the existing
strategies/_warmup_state helper (P172).

AUTHORIZED ARMING: the operator, told explicitly this touches an order path,
instructed "fix all" — once warmups are cumulative these strategies can fire
at kraken_quant's existing DECIDE seat. The behavioural tests here therefore
prove BOTH halves (P174): a restarted instance CAN now fire where a cold one
CANNOT, and every warmup gate still binds at 49 samples.

Every test constructs its state under HMATS_DATA_DIR=tmp_path (P294: never
inherit it from the machine, never write into the repo's data/).
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests._guard_pins import assert_live_line, _code_part  # noqa: E402

from agents.kraken_quant_agent import (  # noqa: E402
    KQ_WARMUP_MAX_AGE_SEC,
    KalmanCointegrationStrategy,
    RelativeStrengthStrategy,
)

AGENT_SRC = REPO / "agents" / "kraken_quant_agent.py"
LOGGER = "agents.kraken_quant_agent"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """P294: construct the state, never inherit it (and never pollute data/)."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    yield


# ---------------------------------------------------------------------------
# drivers — market_data in the exact nested shape _convert_market_data emits
# (P366: strategies read market_data['prices'][asset]).
# ---------------------------------------------------------------------------
def _kmd(i, sol, eth):
    return {"timestamp": 1_000_000.0 + i * 14400,
            "prices": {"SOL": float(sol), "ETH": float(eth)}}


def _rmd(i, btc, sol):
    return {"timestamp": 1_000_000.0 + i * 14400,
            "prices": {"BTC": float(btc), "SOL": float(sol)}}


def _drive_kalman(strat, n, start=0):
    """Identical SOL/ETH series: theta=[1,0] is exactly right from tick one,
    so innovations stay ~0 and a later divergence is unambiguous."""
    out = []
    for i in range(start, start + n):
        px = 100.0 * (1 + 0.0005 * i)
        out.append(strat.update(_kmd(i, px, px)))
    return out


def _drive_rs(strat, n, start=0):
    out = []
    for i in range(start, start + n):
        out.append(strat.update(_rmd(i, 50000.0 * (1.001 ** i),
                                     100.0 * (1.001 ** i))))
    return out


def _state_file(tmp_path, name) -> Path:
    from strategies._warmup_state import state_path
    p = Path(state_path(name))
    assert str(p).startswith(str(tmp_path)), (
        "state must live under the fixture's HMATS_DATA_DIR, never the repo")
    return p


def _write_state(name, series):
    from strategies._warmup_state import save
    assert save(name, series) is True


# ---------------------------------------------------------------------------
# 1. round trip through disk on a FRESH object
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_kalman_buffers_and_filter_state_survive(self, tmp_path):
        a = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        _drive_kalman(a, 60)
        assert _state_file(tmp_path, "kq_kalman_sol_eth").exists()

        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert len(b.price_buffer["SOL"]) == 60, "resume at 60, not 0"
        assert list(b.price_buffer["SOL"]) == list(a.price_buffer["SOL"])
        assert list(b.price_buffer["ETH"]) == list(a.price_buffer["ETH"])
        assert list(b.spread_buffer) == pytest.approx(list(a.spread_buffer))
        # THE KALMAN-STATE DECISION: theta/P are a recursive estimate, so
        # they must survive WITH the spreads they produced.
        assert list(b.theta) == pytest.approx(list(a.theta))
        assert np.asarray(b.P).ravel().tolist() == pytest.approx(
            np.asarray(a.P).ravel().tolist())

    def test_rs_buffers_survive(self, tmp_path):
        a = RelativeStrengthStrategy()
        _drive_rs(a, 30)          # below the 50-sample warmup
        assert _state_file(tmp_path, "kq_relative_strength").exists()

        b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 30, "resume at 30, not 0"
        assert list(b.price_buffer["BTC"]) == list(a.price_buffer["BTC"])
        assert list(b.price_buffer["SOL"]) == list(a.price_buffer["SOL"])

    def test_restore_only_prefills_no_double_append(self):
        """The P371 parity rule: restore PRE-FILLS; update() still appends
        exactly once per call, so a restart never double-counts a tick."""
        a = RelativeStrengthStrategy()
        _drive_rs(a, 10)
        b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 10
        _drive_rs(b, 1, start=10)
        assert len(b.price_buffer["BTC"]) == 11


# ---------------------------------------------------------------------------
# 2. the warmup thresholds still GATE — persistence must not loosen them
# ---------------------------------------------------------------------------
class TestWarmupStillGates:
    def test_rs_49_samples_no_signal_path_reached(self):
        a = RelativeStrengthStrategy()
        _drive_rs(a, 48)
        b = RelativeStrengthStrategy()
        assert _drive_rs(b, 1, start=48) == [None]      # 49 < 50
        # the PROOF the warmup gate was the return path: rs is computed only
        # past the gate, and no rs was ever appended.
        assert len(b.rs_buffer) == 0

    def test_kalman_price_gate_still_binds_at_49(self):
        a = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        _drive_kalman(a, 48)
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert _drive_kalman(b, 1, start=48) == [None]  # 49 < 50
        assert len(b.spread_buffer) == 0, "kalman_update must not have run"

    def test_kalman_spread_gate_still_binds_below_30(self):
        a = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        _drive_kalman(a, 55)                            # spreads: 6
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert _drive_kalman(b, 1, start=55) == [None]
        assert 0 < len(b.spread_buffer) < 30
        assert b.state.in_position is False


# ---------------------------------------------------------------------------
# 3. fail directions (P301) — every one a logged cold start, never a hazard
# ---------------------------------------------------------------------------
class TestFailDirections:
    def test_stale_state_over_7_days_is_dropped(self, tmp_path, caplog):
        a = RelativeStrengthStrategy()
        _drive_rs(a, 10)
        p = _state_file(tmp_path, "kq_relative_strength")
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["saved_ts"] -= 8 * 24 * 3600.0
        p.write_text(json.dumps(payload), encoding="utf-8")

        with caplog.at_level(logging.INFO, logger=LOGGER):
            b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 0, "stale history must be dropped"
        assert any("cold start" in r.getMessage() for r in caplog.records), (
            "a cold start must be LOGGED, not silent")

    def test_corrupt_file_is_a_cold_start(self, tmp_path):
        p = _state_file(tmp_path, "kq_kalman_sol_eth")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{{{not json", encoding="utf-8")
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))   # must not raise
        assert len(b.price_buffer["SOL"]) == 0
        assert _drive_kalman(b, 1) == [None]                   # evaluate works

    def test_version_mismatch_is_a_cold_start(self, tmp_path):
        a = RelativeStrengthStrategy()
        _drive_rs(a, 10)
        p = _state_file(tmp_path, "kq_relative_strength")
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["version"] = "some_other_version"
        p.write_text(json.dumps(payload), encoding="utf-8")
        b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 0

    def test_save_failure_is_swallowed_logged_and_update_still_returns(
            self, monkeypatch, caplog):
        import strategies._warmup_state as ws

        def _boom(*a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(ws, "save", _boom)
        s = RelativeStrengthStrategy()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            out = _drive_rs(s, 2)
        assert out == [None, None], "a raising save must not break update()"
        assert any("persist skipped" in r.getMessage()
                   for r in caplog.records), "swallowed-BUT-LOGGED"

    def test_restore_failure_cannot_break_construction(self, monkeypatch):
        import strategies._warmup_state as ws
        monkeypatch.setattr(ws, "load",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("bad disk")))
        s = KalmanCointegrationStrategy(pair=("SOL", "ETH"))   # must not raise
        assert len(s.spread_buffer) == 0

    def test_pair_mismatch_beyond_tolerance_drops_the_whole_restore(self, caplog):
        # [P390b] mismatch 7 > KQ_WARMUP_PAIR_TOLERANCE (5) — a foreign-
        # shaped pair still drops everything
        _write_state("kq_relative_strength",
                     {"px::BTC": [100.0] * 10, "px::SOL": [50.0] * 3})
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 0
        assert len(b.price_buffer["SOL"]) == 0, (
            "a large mismatch must drop the RESTORE, not pair bar i with bar j")
        assert any("DROPPED" in r.getMessage() for r in caplog.records)

    def test_small_pair_desync_truncates_to_the_common_tail(self):
        # [P390b] the per-asset appends are independent (`if asset in
        # prices`), so one missing price desyncs the pair by one for the
        # rest of the warmup — the FIRST live state file showed exactly
        # this (px::SOL 2 vs px::ETH 1). A small desync keeps the common
        # TAIL instead of dropping the restore (which would silently
        # restore the starvation P390 exists to remove).
        _write_state("kq_relative_strength",
                     {"px::BTC": [100.0 + i for i in range(10)],
                      "px::SOL": [50.0 + i for i in range(9)]})
        b = RelativeStrengthStrategy()
        assert len(b.price_buffer["BTC"]) == 9
        assert len(b.price_buffer["SOL"]) == 9
        assert b.price_buffer["BTC"][0] == 101.0, "tail kept, head dropped"

    def test_kalman_live_shape_2v1_restores_with_filter_state(self):
        # the exact first live artifact (2026-08-24 07:56 UTC)
        _write_state("kq_kalman_sol_eth", {
            "px::SOL": [94.27, 94.27], "px::ETH": [2456.71],
            "theta": [1.0, 0.0], "P": [1.0, 0.0, 0.0, 1.0],
        })
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert len(b.price_buffer["SOL"]) == 1
        assert len(b.price_buffer["ETH"]) == 1
        assert list(b.theta) == [1.0, 0.0]

    def test_kalman_partial_filter_state_drops_everything(self):
        """Filter state and its spreads are inseparable: theta with the
        wrong arity must drop the BUFFERS too, not just reset theta —
        restoring converged spreads under a reset filter is a mismatch
        worse than a cold start."""
        _write_state("kq_kalman_sol_eth", {
            "px::SOL": [100.0] * 60, "px::ETH": [100.0] * 60,
            "spread": [0.0] * 20,
            "theta": [1.0],                       # arity 1, not 2
            "P": [1.0, 0.0, 0.0, 1.0],
        })
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert len(b.price_buffer["SOL"]) == 0
        assert len(b.spread_buffer) == 0
        assert list(b.theta) == [1.0, 0.0], "cold-start filter state"

    def test_nan_never_enters_a_restored_buffer(self):
        _write_state("kq_relative_strength", {
            "px::BTC": [100.0] * 9 + [float("nan")],
            "px::SOL": [50.0] * 10,
        })
        b = RelativeStrengthStrategy()
        for buf in (b.price_buffer["BTC"], b.price_buffer["SOL"], b.rs_buffer):
            assert not any(math.isnan(v) for v in buf)
        # the helper drops the NaN -> 9 vs 10 -> a SMALL desync now
        # truncates to the common tail (P390b) instead of dropping all;
        # the invariant under test is only that no NaN got in
        assert len(b.price_buffer["BTC"]) == 9
        assert len(b.price_buffer["SOL"]) == 9

    def test_non_positive_restored_price_drops_the_restore(self):
        _write_state("kq_kalman_sol_eth", {
            "px::SOL": [100.0] * 59 + [0.0], "px::ETH": [100.0] * 60,
            "theta": [1.0, 0.0], "P": [1.0, 0.0, 0.0, 1.0],
        })
        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert len(b.price_buffer["SOL"]) == 0, (
            "a 0.0 price would poison the rolling returns (P2)")

    def test_in_position_is_not_persisted(self, tmp_path):
        a = RelativeStrengthStrategy()
        _drive_rs(a, 10)
        a.state.in_position = True
        a._persist_warmup()
        raw = _state_file(tmp_path, "kq_relative_strength").read_text(
            encoding="utf-8")
        assert "in_position" not in raw, (
            "signal-lifecycle state is not a warmup and must not persist")
        b = RelativeStrengthStrategy()
        assert b.state.in_position is False


# ---------------------------------------------------------------------------
# 4. THE BEHAVIOURAL ARMING PIN (P174): the starvation is actually removed —
#    a restarted instance can FIRE where a cold instance cannot.
# ---------------------------------------------------------------------------
class TestStarvationRemoved:
    def test_kalman_fires_after_a_restart_where_cold_cannot(
            self, tmp_path, monkeypatch):
        from agents.kraken_quant_agent import SignalType
        a = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert all(r is None for r in _drive_kalman(a, 85))
        del a                                     # "the process dies"

        b = KalmanCointegrationStrategy(pair=("SOL", "ETH"))  # the restart
        assert all(r is None for r in _drive_kalman(b, 3, start=85))
        eth = 100.0 * (1 + 0.0005 * 88)
        sig = b.update(_kmd(88, eth * 1.05, eth))   # SOL diverges +5%
        assert sig is not None, (
            "past both warmups after a restart, a 2-sigma spread divergence "
            "must produce a signal — the P358c starvation is removed")
        assert sig.signal_type == SignalType.PAIR_SHORT_A_LONG_B
        assert abs(sig.metadata["zscore"]) > 2.0

        # the COLD CONTROL: identical post-restart ticks, empty state dir —
        # still starved, still silent. This is what P358c measured.
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path / "cold_k"))
        c = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert all(r is None for r in _drive_kalman(c, 3, start=85))
        assert c.update(_kmd(88, eth * 1.05, eth)) is None

    def test_rs_fires_after_a_restart_where_cold_cannot(
            self, tmp_path, monkeypatch):
        from agents.kraken_quant_agent import SignalType

        def _accel(strat, n):
            sig = None
            sol = 100.0 * (1.001 ** 51)
            for j in range(n):
                i = 52 + j
                sol *= 1.01                        # SOL outperformance
                sig = strat.update(_rmd(i, 50000.0 * (1.001 ** i), sol))
                if sig is not None:
                    break
            return sig

        a = RelativeStrengthStrategy()
        assert all(r is None for r in _drive_rs(a, 52))
        del a

        b = RelativeStrengthStrategy()             # the restart
        sig = _accel(b, 12)
        assert sig is not None, "warm restart + SOL outperformance must fire"
        assert sig.signal_type == SignalType.LONG
        assert sig.asset == "SOL"

        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path / "cold_rs"))
        c = RelativeStrengthStrategy()             # the cold control
        assert _accel(c, 12) is None


# ---------------------------------------------------------------------------
# 5. source pins — the wiring cannot silently rot (P170: a mechanism nothing
#    calls is decoration)
# ---------------------------------------------------------------------------
def _live_call_count(src: str, needle: str) -> int:
    return sum(1 for line in src.splitlines()
               if needle in _code_part(line))


class TestSourcePins:
    @pytest.mark.parametrize("cls", [RelativeStrengthStrategy,
                                     KalmanCointegrationStrategy])
    def test_restore_is_called_at_construction(self, cls):
        assert_live_line(
            inspect.getsource(cls.__init__), "self._restore_warmup()",
            why="restore at construction is the whole fix — without it the "
                "persistence is write-only (P211)")

    @pytest.mark.parametrize("cls", [RelativeStrengthStrategy,
                                     KalmanCointegrationStrategy])
    def test_persist_covers_both_mutation_sites_in_update(self, cls):
        n = _live_call_count(inspect.getsource(cls.update),
                             "self._persist_warmup()")
        assert n >= 2, (
            f"{cls.__name__}.update persists at {n} site(s); it needs one "
            "after the price appends (covers the warmup-gate return) AND one "
            "after the derived-state mutation, or a return path loses a tick")

    def test_state_names_and_max_age_are_pinned(self):
        assert RelativeStrengthStrategy._WARMUP_STATE_NAME == \
            "kq_relative_strength"
        k = KalmanCointegrationStrategy(pair=("SOL", "ETH"))
        assert k._warmup_state_name == "kq_kalman_sol_eth"
        # a different pair must never read SOL/ETH state
        k2 = KalmanCointegrationStrategy(pair=("BTC", "ETH"))
        assert k2._warmup_state_name == "kq_kalman_btc_eth"
        assert KQ_WARMUP_MAX_AGE_SEC == 7 * 24 * 3600.0, (
            "7 days is the pipeline bound (P301/P371); moving it is a "
            "decision, not a tidy-up")

    def test_warmup_thresholds_unchanged(self):
        """The gates themselves must not have been loosened by the arming —
        the P358c needles, asserted here at the behaviour-owning classes."""
        assert "< 50" in inspect.getsource(RelativeStrengthStrategy)
        ksrc = inspect.getsource(KalmanCointegrationStrategy)
        assert "< 50" in ksrc and "< 30" in ksrc
