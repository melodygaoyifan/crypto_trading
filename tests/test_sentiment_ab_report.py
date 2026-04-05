import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import sentiment_ab_report as report_mod


def _write_diag(path: Path, asset: str, ts: str, *, tradeable: bool, source: str = "haiku"):
    payload = {
        "tick_time": ts,
        "asset": asset,
        "components": {
            "det_sentiment": {
                "called": True,
                "consumed": True,
                "output": "{'zscore': 1.5}",
            },
            "llm_sentiment": {
                "called": True,
                "consumed": tradeable,
                "output": str(
                    {
                        "combined_dir": 0.4 if tradeable else 0.0,
                        "conf": 0.7 if tradeable else 0.0,
                        "source": source if tradeable else "none",
                        "headlines": 4 if tradeable else 0,
                        "status": "live_metrics" if tradeable else "unavailable",
                    }
                ),
            },
        },
        "final_intent": {
            "alpha_gate_passed": True,
            "is_actionable": tradeable,
            "veto_active": False,
            "direction": -0.22 if tradeable else 0.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fill(path: Path, ts: str, asset: str, side: str, size: float, realized_pnl: float):
    payload = {
        "timestamp": ts,
        "entry_type": "FILL",
        "asset": asset,
        "data": {
            "side": side,
            "size": size,
            "price": 100.0,
            "realized_pnl": realized_pnl,
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_sentiment_ab_report_returns_insufficient_sample_with_no_treatment_realized(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    diag_dir = data_dir / "diagnostics"
    ledger_dir = data_dir / "shadow_ledger"
    diag_dir.mkdir(parents=True)
    ledger_dir.mkdir(parents=True)

    _write_diag(
        diag_dir / "diag_SOL_2026-03-12T00-00-00+00-00.json",
        "SOL",
        "2026-03-12T00:00:00+00:00",
        tradeable=True,
        source="cryptopanic_metrics",
    )
    (data_dir / "step15_status.json").write_text(
        json.dumps({"realized_pnl": {"closed_trade_count": 0}, "run": {"started_at": "2026-03-12T00:00:00Z"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "DIAGNOSTICS_DIR", diag_dir)
    monkeypatch.setattr(report_mod, "SHADOW_LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(report_mod, "STEP15_STATUS", data_dir / "step15_status.json")
    monkeypatch.setattr(report_mod, "datetime", type("FixedDateTime", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 3, 13, 0, 0, 0, tzinfo=timezone.utc))}))

    report = report_mod.build_report(days=30, association_minutes=30)

    assert report["diagnostics"]["treatment_count"] == 1
    assert report["trades"]["realized_fill_events_total"] == 0
    assert report["verdict"]["status"] == "INSUFFICIENT_REALIZED_SAMPLE"
    assert "treatment_realized_events=0" in report["verdict"]["reason"]
    assert "Current runtime step15_status still reports zero closed trades." in report["warnings"]


def test_sentiment_ab_report_keeps_when_treatment_pnl_beats_baseline(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    diag_dir = data_dir / "diagnostics"
    ledger_dir = data_dir / "shadow_ledger"
    diag_dir.mkdir(parents=True)
    ledger_dir.mkdir(parents=True)

    for i in range(5):
        hour = 1 + i
        _write_diag(
            diag_dir / f"diag_SOL_2026-03-12T{hour:02d}-00-00+00-00.json",
            "SOL",
            f"2026-03-12T{hour:02d}:00:00+00:00",
            tradeable=True,
            source="haiku",
        )

    ledger_path = ledger_dir / "ledger_20260312.jsonl"
    lines = []

    for i in range(5):
        entry_ts = f"2026-03-12T{1 + i:02d}:05:00+00:00"
        exit_ts = f"2026-03-12T{1 + i:02d}:10:00+00:00"
        lines.append(
            json.dumps(
                {
                    "timestamp": entry_ts,
                    "entry_type": "FILL",
                    "asset": "SOL",
                    "data": {"side": "SELL", "size": 1.0, "price": 100.0, "realized_pnl": 0.0},
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "timestamp": exit_ts,
                    "entry_type": "FILL",
                    "asset": "SOL",
                    "data": {"side": "BUY", "size": 1.0, "price": 95.0, "realized_pnl": 10.0},
                }
            )
        )

    for i in range(5):
        entry_ts = f"2026-02-20T{1 + i:02d}:05:00+00:00"
        exit_ts = f"2026-02-20T{1 + i:02d}:10:00+00:00"
        lines.append(
            json.dumps(
                {
                    "timestamp": entry_ts,
                    "entry_type": "FILL",
                    "asset": "BTC",
                    "data": {"side": "SELL", "size": 1.0, "price": 100.0, "realized_pnl": 0.0},
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "timestamp": exit_ts,
                    "entry_type": "FILL",
                    "asset": "BTC",
                    "data": {"side": "BUY", "size": 1.0, "price": 99.0, "realized_pnl": 9.6},
                }
            )
        )

    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / "step15_status.json").write_text(
        json.dumps({"realized_pnl": {"closed_trade_count": 5}, "run": {"started_at": "2026-03-12T00:00:00Z"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "DIAGNOSTICS_DIR", diag_dir)
    monkeypatch.setattr(report_mod, "SHADOW_LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(report_mod, "STEP15_STATUS", data_dir / "step15_status.json")
    monkeypatch.setattr(report_mod, "datetime", type("FixedDateTime", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 3, 13, 0, 0, 0, tzinfo=timezone.utc))}))

    report = report_mod.build_report(days=30, association_minutes=30)

    assert report["trades"]["realized_events"]["treatment"] == 5
    assert report["trades"]["realized_events"]["baseline"] == 5
    assert report["verdict"]["reference_group"] == "baseline"
    assert report["verdict"]["status"] == "KEEP"
    assert report["verdict"]["pnl_delta_pct"] > 0.03
    assert report["warnings"] == []
