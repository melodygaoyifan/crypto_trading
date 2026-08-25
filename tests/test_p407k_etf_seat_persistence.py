"""[P407k] the ETF seat state survives a restart (was RAM-only -> blanked cycle 1)."""
import time
from defense.etf_flow_shadow import EtfFlowShadow


def test_seat_state_survives_a_restart(tmp_path):
    a = EtfFlowShadow(data_dir=str(tmp_path))
    a._seat_state["BTC"] = (1.0, True, time.time())
    a._save_state()
    b = EtfFlowShadow(data_dir=str(tmp_path))   # fresh process
    assert b.seat_direction("BTC") == (1.0, True)  # restored fresh, not None


def test_a_stale_restore_is_not_fresh(tmp_path):
    a = EtfFlowShadow(data_dir=str(tmp_path))
    a._seat_state["BTC"] = (1.0, True, time.time() - 13 * 3600.0)  # >12h old
    a._save_state()
    b = EtfFlowShadow(data_dir=str(tmp_path))
    d, fresh = b.seat_direction("BTC")
    assert d == 1.0 and fresh is False   # age gate rejects it -> seat skipped (P2)


def test_last_direction_persistence_still_works(tmp_path):
    a = EtfFlowShadow(data_dir=str(tmp_path))
    a._last_direction["BTC"] = -1.0
    a._save_state()
    b = EtfFlowShadow(data_dir=str(tmp_path))
    assert b._last_direction.get("BTC") == -1.0   # P402 not broken
