"""[GP0] The statistical-validity layer: shared splits, window-usage ledger,
provenance stamps, standard eval rows, robustness battery.

The point of each test is the failure mode it pins:
  * purged folds that leak train rows into the embargo zone;
  * a ledger that undercounts validation spend (silent re-mining, P243);
  * a provenance stamp that doesn't move when the data moves (P200);
  * an eval row without the overfit gap;
  * a battery that fails to flag an era sign-flip (the composite's exact
    failure shape).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training import splits as SP
from training.provenance import provenance_stamp, file_hash
from training.eval_report import standard_row, robustness_battery


# ---------------------------------------------------------------- splits
def test_purged_folds_respect_embargo():
    for tr, va in SP.purged_folds(1000, 3000, n_splits=4, embargo=42, horizon=4):
        lo, hi = va[0] - 46, va[-1] + 47
        assert not ((tr >= lo) & (tr < hi)).any(), "train rows inside the purge zone"
        assert va[0] >= 1000 and va[-1] < 3000


def test_purged_folds_cover_the_whole_range():
    covered = np.concatenate([va for _, va in SP.purged_folds(1000, 3000, n_splits=4)])
    assert covered.min() == 1000 and covered.max() == 2999
    assert len(np.unique(covered)) == 2000


def test_fold_geometry_matches_supervised_pipeline():
    """One geometry, two implementations — pin them together until the
    pipeline is migrated to the shared module."""
    from training.train_supervised_full import fold_splits as pipeline_fold
    n = 13095
    assert SP.drl_fold_splits(n) == pipeline_fold(n)


# ---------------------------------------------------------------- ledger
@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    p = tmp_path / "window_usage.json"
    monkeypatch.setattr(SP, "LEDGER_PATH", p)
    return p


def test_ledger_records_and_counts_prior_validation_reads(tmp_ledger):
    assert SP.record_window_usage("exp_a", "BTC", 9100, 13000, "validation") == 0
    # overlapping validation read by a DIFFERENT experiment counts
    assert SP.record_window_usage("exp_b", "BTC", 9500, 12000, "validation") == 1
    # same experiment re-reading does not inflate the count
    assert SP.record_window_usage("exp_b", "BTC", 9500, 12000, "validation") == 1
    # design reads never count as validation spend
    assert SP.record_window_usage("exp_c", "BTC", 9100, 13000, "design") == 0
    # disjoint window is free
    assert SP.record_window_usage("exp_d", "BTC", 3000, 9000, "validation") == 0
    data = json.loads(tmp_ledger.read_text(encoding="utf-8"))
    assert len(data["records"]) == 5


def test_ledger_is_per_asset(tmp_ledger):
    SP.record_window_usage("exp_a", "BTC", 9100, 13000, "validation")
    assert SP.record_window_usage("exp_b", "ETH", 9100, 13000, "validation") == 0


# ------------------------------------------------------------- provenance
def test_provenance_stamp_tracks_data_content(tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"aaa")
    s1 = provenance_stamp(data_files=[f])
    f.write_bytes(b"bbb")
    s2 = provenance_stamp(data_files=[f])
    assert s1["data_hashes"]["d.bin"] != s2["data_hashes"]["d.bin"], (
        "a stamp that does not move when the data moves is the P200 hole"
    )
    assert s1["git_commit"], "stamp must carry the code commit"


def test_file_hash_is_stable(tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"same content")
    assert file_hash(f) == file_hash(f)


# ------------------------------------------------------------ eval report
def test_standard_row_reports_overfit_gap():
    train = np.full(200, 0.001)          # steady gains in-sample
    row = standard_row("x", train, cv_sharpe=0.5)
    assert "overfit_gap" in row and row["overfit_gap"] == round(row["train_sharpe"] - 0.5, 3)
    assert "test_pnl_pct" not in row, "test columns must not appear without a spent test window"
    row2 = standard_row("x", train, 0.5, test_seg=np.full(100, -0.001))
    assert row2["test_pnl_pct"] < 0 and "train_test_gap" in row2


def test_battery_flags_era_sign_flip():
    """The composite's exact failure shape: positive in the design window,
    negative pre-design — the battery must name it."""
    def run_fn(params, window, cost_mult):
        pnl = 10.0 if window[0] == 0 else -5.0
        return {"pnl_pct": pnl * (0.5 if cost_mult > 1 else 1.0), "sharpe": 0.5}
    out = robustness_battery(run_fn, {}, {}, {"design": (0, 100), "pre": (100, 200)})
    assert "ERA-FRAGILE" in out["flags"]


def test_battery_flags_cost_fragility():
    def run_fn(params, window, cost_mult):
        return {"pnl_pct": 10.0 - 12.0 * (cost_mult - 1.0), "sharpe": 0.3}
    out = robustness_battery(run_fn, {}, {}, {"w": (0, 100)})
    assert any(f.startswith("COST-FRAGILE") for f in out["flags"])


def test_battery_quiet_on_robust_strategy():
    def run_fn(params, window, cost_mult):
        return {"pnl_pct": 10.0, "sharpe": 0.8}
    out = robustness_battery(run_fn, {"a": 1}, {"a": [2, 3]},
                             {"w1": (0, 100), "w2": (100, 200)})
    assert out["flags"] == []
