from defense.shadow_ledger_jsonl import ShadowLedgerWriter


def test_shadow_ledger_session_fill_summary_tracks_counts_and_timestamps(tmp_path):
    writer = ShadowLedgerWriter(output_dir=str(tmp_path), session_id="TEST", auto_flush=False)
    try:
        writer.record_fill(
            asset="SOL",
            order_id="ORD1",
            fill_id="FILL1",
            side="SELL",
            size=1.0,
            price=100.0,
        )
        writer.record_fill(
            asset="ETH",
            order_id="ORD2",
            fill_id="FILL2",
            side="BUY",
            size=2.0,
            price=200.0,
        )
        writer.record_fill(
            asset="SOL",
            order_id="ORD3",
            fill_id="FILL3",
            side="BUY",
            size=0.5,
            price=90.0,
        )

        summary = writer.get_session_fill_summary()
        sol = writer.get_session_fill_summary("SOL")
        btc = writer.get_session_fill_summary("BTC")

        assert summary["session_id"] == "TEST"
        assert summary["count"] == 3
        assert summary["counts_by_asset"] == {"SOL": 2, "ETH": 1}
        assert summary["last_fill_at"] is not None
        assert summary["last_fill_at_by_asset"]["SOL"] is not None
        assert summary["last_fill_at_by_asset"]["ETH"] is not None

        assert sol["count"] == 2
        assert sol["last_fill_at"] == summary["last_fill_at_by_asset"]["SOL"]
        assert btc["count"] == 0
        assert btc["last_fill_at"] is None
    finally:
        writer.close()
