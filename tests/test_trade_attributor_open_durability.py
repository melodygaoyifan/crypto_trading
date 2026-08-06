"""[P185] 38 of 90 closed trades had no recorded entry, and nothing said so.

`TradeAttributor` persisted only CLOSED trades. `_open_trades` lived in memory
and `_load_persisted` restored `_closed_trades` alone, so every position held
across a process restart lost its entry record. The eventual exit fell into the
orphan branch of `record_exit` with entry_price=0.0, direction=0, strategy="",
entry_fee_usd=0.0 — and `net_pnl_usd` subtracts `entry_fee_usd`, so each of
those trades understated its own cost by roughly one taker fee (26bps at
Kraken). The ledger read more profitable than the account.

The P129 comment in core/execution_service.py blamed a swallowed exception in
record_entry and added logging to catch it. That was the wrong suspect: the
call succeeded, its result was simply never written down.

Three things are pinned here:

  1. an open trade survives a restart — the actual fix,
  2. a genuine orphan is still MARKED as one (`entry_recorded=False`), so the
     58%-coverage problem is countable rather than inferred from an empty
     string — this is the P170 lesson applied to a second file,
  3. the entry fee reaches net_pnl_usd across the restart, which is the part
     that moved money-shaped numbers.

Point 1 alone would pass on a system that silently dropped orphan trades
instead of recording them, which would be worse: unmeasured rather than
mismeasured.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.trade_attributor import TradeAttributor, TradeRecord  # noqa: E402


@pytest.fixture
def paths(tmp_path):
    """A JSONL path under tmp_path; the sidecar derives from it."""
    return tmp_path / "trade_attribution.jsonl"


def _entered(attr, asset="BTC", price=70000.0, fee=18.2, notional=7000.0):
    attr.record_entry(asset=asset, price=price, fee=fee, notional=notional,
                      direction=1, strategy="momentum", regime="REGIME_1",
                      mode="NORMAL")


class TestAnOpenTradeSurvivesARestart:
    def test_the_sidecar_is_written_on_entry(self, paths):
        attr = TradeAttributor(persist_path=str(paths))
        _entered(attr)
        sidecar = paths.with_name("trade_attribution_open.json")
        assert sidecar.exists(), (
            "record_entry did not persist the open trade. A restart now "
            "orphans this position, which is exactly P185."
        )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["BTC"]["entry_price"] == 70000.0

    def test_a_second_instance_restores_it(self, paths):
        _entered(TradeAttributor(persist_path=str(paths)))

        restarted = TradeAttributor(persist_path=str(paths))  # the restart
        assert "BTC" in restarted._open_trades, (
            "the open trade did not survive construction of a new "
            "TradeAttributor — the restart path is still lossy"
        )
        rec = restarted._open_trades["BTC"]
        assert rec.entry_price == 70000.0
        assert rec.direction == 1
        assert rec.strategy == "momentum"
        assert rec.entry_recorded is True

    def test_the_exit_after_a_restart_is_not_an_orphan(self, paths):
        _entered(TradeAttributor(persist_path=str(paths)))
        restarted = TradeAttributor(persist_path=str(paths))
        restarted.record_exit(asset="BTC", price=71000.0, fee=18.5,
                              notional=7100.0, gross_pnl=100.0)

        closed = [t for t in restarted._closed_trades if t.exit_type == "FULL"]
        assert len(closed) == 1
        t = closed[0]
        assert t.entry_recorded is True, "the exit still produced an orphan"
        assert t.strategy == "momentum", (
            "attribution metadata was lost across the restart, so this trade "
            "contributes to no per-strategy number"
        )

    def test_the_entry_fee_reaches_net_pnl_across_the_restart(self, paths):
        """The part that moved money-shaped numbers."""
        _entered(TradeAttributor(persist_path=str(paths)))
        restarted = TradeAttributor(persist_path=str(paths))
        restarted.record_exit(asset="BTC", price=71000.0, fee=18.5,
                              notional=7100.0, gross_pnl=100.0)
        t = [x for x in restarted._closed_trades if x.exit_type == "FULL"][0]
        # 100.0 gross - 18.2 entry - 18.5 exit + 0 funding
        assert t.net_pnl_usd == pytest.approx(63.3), (
            f"net_pnl_usd is {t.net_pnl_usd}. Before P185 this was 81.5 — the "
            f"entry fee was silently omitted because the orphan record had "
            f"entry_fee_usd=0.0."
        )

    def test_accrued_funding_survives_too(self, paths):
        attr = TradeAttributor(persist_path=str(paths))
        _entered(attr)
        attr.record_funding("BTC", funding_rate=0.0001, funding_pnl=-1.25)

        restarted = TradeAttributor(persist_path=str(paths))
        assert restarted._open_trades["BTC"].total_funding_pnl == \
            pytest.approx(-1.25), (
            "funding accrued before the restart was lost, so the hold cost of "
            "any multi-day position is understated"
        )


class TestAGenuineOrphanIsStillMarked:
    """The half that keeps the fix from hiding the remaining problem."""

    def test_an_exit_with_no_entry_sets_entry_recorded_false(self, paths):
        attr = TradeAttributor(persist_path=str(paths))
        attr.record_exit(asset="SOL", price=86.31, fee=9.55, notional=5968.0,
                         gross_pnl=-0.04)
        t = attr._closed_trades[-1]
        assert t.entry_recorded is False, (
            "an orphan is no longer flagged. Consumers are back to inferring "
            "it from `entry_time == ''`, which is a legal value — that guess "
            "is what dropped 38 trades from the DRL counterfactual without "
            "the report mentioning it."
        )

    def test_the_flag_round_trips_through_the_jsonl(self, paths):
        attr = TradeAttributor(persist_path=str(paths))
        attr.record_exit(asset="SOL", price=86.31, fee=9.55, notional=5968.0,
                         gross_pnl=-0.04)
        reloaded = TradeAttributor(persist_path=str(paths))
        assert reloaded._closed_trades[-1].entry_recorded is False

    def test_legacy_records_backfill_the_flag_from_entry_time(self, paths):
        """Records written before the field existed must still classify."""
        legacy_orphan = {"trade_id": "SOL_orphan_2026-02-17", "asset": "SOL",
                         "entry_time": "", "is_closed": True,
                         "net_pnl_usd": -9.58}
        legacy_good = {"trade_id": "BTC_20260217_080000", "asset": "BTC",
                       "entry_time": "2026-02-17T08:00:00+00:00",
                       "is_closed": True, "net_pnl_usd": 12.0}
        paths.write_text(json.dumps(legacy_orphan) + "\n"
                         + json.dumps(legacy_good) + "\n", encoding="utf-8")
        attr = TradeAttributor(persist_path=str(paths))
        by_id = {t.trade_id: t for t in attr._closed_trades}
        assert by_id["SOL_orphan_2026-02-17"].entry_recorded is False
        assert by_id["BTC_20260217_080000"].entry_recorded is True

    def test_coverage_is_reported_as_a_number(self, paths):
        attr = TradeAttributor(persist_path=str(paths))
        _entered(attr, asset="BTC")
        attr.record_exit(asset="BTC", price=71000.0, fee=18.5,
                         notional=7100.0, gross_pnl=100.0)
        attr.record_exit(asset="SOL", price=86.31, fee=9.55, notional=5968.0,
                         gross_pnl=-0.04)
        cov = attr.entry_coverage()
        assert cov == {"closed_trades": 2, "with_entry": 1, "orphans": 1,
                       "coverage_pct": 50.0}, (
            f"entry_coverage() returned {cov}. Every per-strategy number from "
            f"this file is computed over with_entry trades only; publishing "
            f"them without this ratio is how 58% read as 100%."
        )


class TestTheOtherWritesThatWereLost:
    def test_force_close_reaches_the_jsonl(self, paths):
        """record_entry's force-close branch closed a trade and never wrote it."""
        attr = TradeAttributor(persist_path=str(paths))
        _entered(attr, asset="BTC", price=70000.0)
        _entered(attr, asset="BTC", price=70500.0)  # forces the first closed

        reloaded = TradeAttributor(persist_path=str(paths))
        forced = [t for t in reloaded._closed_trades
                  if t.exit_type == "FORCE_CLOSE"]
        assert len(forced) == 1, (
            "the force-closed trade is absent from the JSONL. The in-memory "
            "report counted it and the persisted file did not; the two views "
            "of 'closed trades' disagreed and neither said so."
        )

    def test_funding_payments_survive_reload(self, paths):
        """_load_persisted's hand-written copy had already dropped this field."""
        attr = TradeAttributor(persist_path=str(paths))
        _entered(attr)
        attr.record_funding("BTC", funding_rate=0.0001, funding_pnl=-1.25)
        attr.record_exit(asset="BTC", price=71000.0, fee=18.5,
                         notional=7100.0, gross_pnl=100.0)

        reloaded = TradeAttributor(persist_path=str(paths))
        t = [x for x in reloaded._closed_trades if x.exit_type == "FULL"][0]
        assert len(t.funding_payments) == 1, (
            "funding_payments did not survive the reload. Two readers of the "
            "same schema is two places to forget a field — they are one "
            "function now, and this is the test that keeps them one."
        )


class TestTheSidecarFailsLoudlyRatherThanSilently:
    def test_an_unwritable_sidecar_does_not_lose_the_exception(self, paths,
                                                              monkeypatch,
                                                              caplog):
        attr = TradeAttributor(persist_path=str(paths))

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        with caplog.at_level("ERROR"):
            _entered(attr)
        assert any("P185" in r.message or "P185" in r.getMessage()
                   for r in caplog.records), (
            "the sidecar write failed and logged nothing at ERROR. A silent "
            "failure here recreates P185 exactly: trades look tracked, the "
            "next restart orphans them."
        )

    def test_a_corrupt_sidecar_does_not_take_the_process_down(self, paths,
                                                             caplog):
        paths.with_name("trade_attribution_open.json").write_text(
            "{not json", encoding="utf-8")
        with caplog.at_level("ERROR"):
            attr = TradeAttributor(persist_path=str(paths))
        assert attr._open_trades == {}
        assert any("P185" in r.getMessage() for r in caplog.records), (
            "a corrupt sidecar was swallowed. The operator needs to know the "
            "next exits will be orphans."
        )


@pytest.mark.parametrize("reader", ["_load_persisted", "_load_open_trades"])
def test_persisted_records_have_exactly_one_deserializer(reader):
    """Two copies of a field list is how funding_payments got dropped once.

    Both readers consume the same `to_dict()` schema. `_load_persisted` used to
    carry its own field-by-field copy and had already fallen behind by one
    field. Constructing TradeRecord directly in either reader is the drift, so
    that is what this forbids — TradeRecord(...) elsewhere (record_entry, the
    orphan branches, the shadow-ledger backfill) builds from live values, not
    from a serialized dict, and is not the same hazard.
    """
    import inspect
    src = inspect.getsource(getattr(TradeAttributor, reader))
    assert "_record_from_dict" in src, (
        f"{reader} no longer routes through the shared deserializer"
    )
    assert "TradeRecord(" not in src, (
        f"{reader} builds a TradeRecord field-by-field again. That copy will "
        f"drift from the schema the way _load_persisted's did — it was "
        f"already missing funding_payments when P185 found it."
    )
