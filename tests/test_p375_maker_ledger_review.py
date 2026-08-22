"""[P375] Pin the authoritative ledger mode of maker_fill_review.

The live CDE fill ledger (data/fill_quality.jsonl, P290) is the authoritative
source for the maker fill rate — the log-parse mode (P282/P287) rests on the
refuted 0.5/3.0 percentage fee model (P315/P374). These tests pin that ledger
mode (a) uses ONLY CDE-format rows (never mixes pre-P290 Kraken rows, P255),
(b) refuses on an all-stale ledger rather than reading zero as a verdict (P199),
and (c) computes the maker fill fraction from NON-URGENT orders.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "maker_fill_review.py"


def _run(tmp_path, rows):
    f = tmp_path / "fq.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), "--ledger-file", str(f)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120, encoding="utf-8")


def _cde(asset, liq, urgent, slip, fee, px, ct):
    return {"asset": asset, "liquidity": liq, "urgent": urgent,
            "realized_slippage_bps": slip, "fees_usd": fee,
            "fill_avg_price": px, "contracts": ct, "status": "filled"}


def test_all_stale_prep290_rows_refuse_not_zero_verdict(tmp_path):
    # pre-P290 rows have liquidity=None; an all-stale ledger must REFUSE (P199),
    # not report a 0% fill rate — and the message must name the P255 stale-copy
    # trap that produced exactly this false 'zero' in the research.
    stale = [{"asset": "ETH", "order_type": "LIMIT", "fill_ratio": 1.0,
              "slippage_bps": -3.0}] * 5
    out = _run(tmp_path, stale)
    assert out.returncode == 2, out.stdout
    assert "0 CDE-format fills" in out.stdout
    assert "P255" in out.stdout


def test_maker_fill_fraction_uses_nonurgent_only(tmp_path):
    rows = [
        _cde("ETH", "maker", False, -2.0, 1.4, 2428, 5),
        _cde("SOL", "maker", False, -4.0, 0.85, 94, 2),
        _cde("ETH", "taker_cross", False, 2.0, 0.58, 2424, 2),
        # urgent fills are excluded from the maker-fraction denominator
        _cde("SOL", "direct", True, 18.0, 1.4, 99, 3),
        _cde("ETH", "direct", True, 10.0, 1.7, 2426, 6),
    ]
    out = _run(tmp_path, rows)
    assert out.returncode == 0, out.stdout + out.stderr
    # non-urgent = 3, maker = 2 -> f = 0.67 (urgent excluded)
    assert "non-urgent 3" in out.stdout
    assert "maker fill rate f = 0.67" in out.stdout


def test_sol_reprice_progress_is_reported(tmp_path):
    rows = [_cde("SOL", "maker", False, -3.0, 0.85, 94, 2)] * 3
    out = _run(tmp_path, rows)
    assert "3/20 fills toward re-pricing" in out.stdout


def test_cde_rows_are_not_mixed_with_stale_rows(tmp_path):
    rows = [
        {"asset": "ETH", "order_type": "LIMIT", "slippage_bps": -99.0},  # stale, excluded
        _cde("ETH", "maker", False, -2.0, 1.4, 2428, 5),
    ]
    out = _run(tmp_path, rows)
    assert out.returncode == 0
    assert "1 CDE fills" in out.stdout  # the stale row is excluded
