"""
test_sol_usdt_switch_p133.py — SOL pair migration to USDT (P133)
================================================================================

Production diagnostic 2026-04-29: Kraken Spot SOL/USD has 0 trades / 0 volume
in 24h (effectively delisted), while SOL/USDT has $949K quote volume + 24/24
valid OHLCV bars. P133 routes SOL fetches/positions/CRC32-monitor/WS-subs to
SOL/USDT.

Verifies all 5 critical paths consistently use SOL/USDT (not SOL/USD).
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8-sig")


class TestP133SolUsdtSwitch:

    def test_market_data_pipeline_uses_usdt(self):
        src = _read("data_mgmt/market_data_pipeline.py")
        assert '"SOL": "SOL/USDT"' in src, (
            "P133 regression: data_mgmt/market_data_pipeline.py:_fetch_live_data "
            "no longer routes SOL to SOL/USDT. Live fetch will hit the dead "
            "SOL/USD pair → 0 bars → schema FAIL → SOL halt."
        )

    def test_account_sync_uses_usdt(self):
        src = _read("core/account_sync.py")
        assert "'SOL': 'SOL/USDT'" in src, (
            "P133 regression: core/account_sync.py kraken_symbol_map no "
            "longer uses SOL/USDT for SOL valuation lookup."
        )

    def test_kraken_rest_client_uses_usdt(self):
        src = _read("infra/kraken_rest_client.py")
        assert '"SOL": "SOL/USDT"' in src, (
            "P133 regression: infra/kraken_rest_client.py SYMBOL_MAP no "
            "longer routes SOL to SOL/USDT."
        )
        # SOLUSDT alias added for downstream callers
        assert '"SOLUSDT": "SOL/USDT"' in src, (
            "P133 regression: SOLUSDT alias missing from SYMBOL_MAP."
        )

    def test_shadow_ledger_has_sol_usdt_entry(self):
        src = _read("data_mgmt/shadow_ledger_manager.py")
        assert "'SOL/USDT'" in src, (
            "P133 regression: SOL/USDT entry removed from shadow ledger "
            "PAIR_INFO. Order/fee bookkeeping for SOL/USDT will fall back "
            "to defaults — risk of incorrect lot sizing."
        )
        # min_order specifically — Kraken's SOL/USDT min is 0.02
        assert "'min_order': Decimal('0.02')" in src, (
            "P133 regression: SOL/USDT min_order entry changed. Was 0.02 "
            "per exchange.market() probe."
        )

    def test_kraken_integrity_shield_uses_usdt(self):
        src = _read("defense/kraken_integrity_shield.py")
        assert "'SOL/USDT'" in src, (
            "P133 regression: kraken_integrity_shield CRC32 monitor no "
            "longer subscribes to SOL/USDT. Will get 0 messages on dead "
            "SOL/USD pair → spurious silence alarm."
        )

    def test_main_py_uses_usdt(self):
        src = _read("main.py")
        # All 4 SOL symbol references should now use USDT
        assert "'SOL/USDT'" in src or '"SOL/USDT"' in src, (
            "P133 regression: main.py no longer references SOL/USDT."
        )

    def test_kraken_link_uses_usdt(self):
        src = _read("infra/kraken_link.py")
        assert "'SOL/USDT'" in src, (
            "P133 regression: infra/kraken_link.py WS subscription "
            "default no longer includes SOL/USDT."
        )
        # HIGH_VOLATILITY_ASSETS should accept BOTH legacy + new aliases
        assert "'SOL/USDT'" in src and "'SOL/USD'" in src, (
            "P133: HIGH_VOLATILITY_ASSETS should retain SOL/USD for "
            "backward compat AND include SOL/USDT for new pair."
        )

    def test_p133_marker_present(self):
        """Marker comments documenting the why preserve future-grep context."""
        for path in (
            "data_mgmt/market_data_pipeline.py",
            "core/account_sync.py",
            "infra/kraken_rest_client.py",
            "data_mgmt/shadow_ledger_manager.py",
            "defense/kraken_integrity_shield.py",
            "infra/kraken_link.py",
            "main.py",
        ):
            assert "P133" in _read(path), (
                f"P133 marker missing from {path}. Future operator won't "
                f"see why SOL is on USDT vs USD."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
