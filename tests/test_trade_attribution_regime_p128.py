"""
test_trade_attribution_regime_p128.py — regime_at_entry population (P0-3 / P129)
================================================================================

v3 Track A item 1.1: 90/90 historical trade_attribution.jsonl records had
empty regime_at_entry / strategy / mode_at_entry. Audit revealed:

  1. The proper record_entry() call site at execution_service.py:3162 was
     wrapped in `try/except: logger.debug(...)` — silent swallow hid
     whatever exception was preventing it from running.
  2. record_exit() created orphan TradeRecords with default-empty metadata
     when _open_trades[asset] was missing (always, since record_entry never
     ran).

P129 (v3 1.1) hardens both sides:
  A. execution_service.py:3162 — promote logger.debug -> logger.warning
     + add INFO success log + normalize empty strings to "UNKNOWN" /
     "NORMAL" defaults so even if the call succeeds with empty market_data,
     the record gets a real value.
  B. trade_attributor.py:208 record_exit() — accept optional
     regime / strategy / mode kwargs and use them in the orphan branch.
     All 3 callers in execution_service.py updated to pass these.

Combined effect: even if record_entry path stays broken (still being
diagnosed), orphan records will populate metadata. Future audits won't
hit the "100% empty" wall.
"""
from __future__ import annotations

import inspect

import pytest


class TestP129TradeAttributorOrphanMetadata:
    """The orphan branch must accept and use regime/strategy/mode kwargs."""

    def test_record_exit_signature_has_regime_kwargs(self):
        from analytics.trade_attributor import TradeAttributor
        sig = inspect.signature(TradeAttributor.record_exit)
        for kwarg in ("regime", "strategy", "mode"):
            assert kwarg in sig.parameters, (
                f"P129 regression: TradeAttributor.record_exit lost the "
                f"'{kwarg}' kwarg. Orphan records will lose metadata."
            )

    def test_record_exit_orphan_branch_uses_kwargs(self):
        from analytics.trade_attributor import TradeAttributor
        src = inspect.getsource(TradeAttributor.record_exit)
        # The orphan branch creates TradeRecord(...) — must reference the
        # kwargs in that constructor call
        assert "regime_at_entry=regime" in src, (
            "P129: orphan TradeRecord constructor doesn't pass regime kwarg."
        )
        assert "strategy=strategy" in src, (
            "P129: orphan TradeRecord constructor doesn't pass strategy kwarg."
        )
        assert "mode_at_entry=mode" in src, (
            "P129: orphan TradeRecord constructor doesn't pass mode kwarg."
        )

    def test_orphan_record_e2e_populates_metadata(self):
        """Full e2e: call record_exit on a fresh attributor with no open
        trade for asset; assert orphan record has populated metadata."""
        import tempfile
        from pathlib import Path
        from analytics.trade_attributor import TradeAttributor

        with tempfile.TemporaryDirectory() as tmp:
            attr = TradeAttributor(persist_path=str(Path(tmp) / "attr.jsonl"))
            # No record_entry called. record_exit triggers orphan branch.
            attr.record_exit(
                asset="BTC",
                price=70000.0,
                fee=1.0,
                notional=1000.0,
                gross_pnl=10.0,
                exit_type="FULL",
                regime="STEADY_UPTREND",
                strategy="mean_revert",
                mode="NORMAL",
            )
            # Read back the persisted record
            persisted = (Path(tmp) / "attr.jsonl").read_text()
            assert "STEADY_UPTREND" in persisted, (
                f"P129: orphan record persisted without regime. Content: "
                f"{persisted[:300]}"
            )
            assert "mean_revert" in persisted, (
                "P129: orphan record persisted without strategy."
            )

    def test_legacy_caller_without_kwargs_still_works(self):
        """Backward compat: callers that don't pass regime/strategy/mode
        kwargs must still succeed (defaults to empty string)."""
        import tempfile
        from pathlib import Path
        from analytics.trade_attributor import TradeAttributor

        with tempfile.TemporaryDirectory() as tmp:
            attr = TradeAttributor(persist_path=str(Path(tmp) / "attr.jsonl"))
            # Old-style call — no regime/strategy/mode kwargs
            attr.record_exit(
                asset="BTC", price=70000.0, fee=1.0,
                notional=1000.0, gross_pnl=10.0,
                exit_type="FULL",
            )
            # Should not raise


class TestP129ExecutionServiceCallers:
    """All 3 record_exit callers + the record_entry caller must pass
    regime/strategy/mode and use logger.warning (not debug) on failure."""

    def test_record_entry_caller_promoted_to_warning(self):
        from core import execution_service
        src = inspect.getsource(execution_service)
        # The W10 record_entry block must include WARNING (not just debug)
        # AND an INFO success log per P129
        assert "[W10] TradeAttributor record_entry OK" in src, (
            "P129: record_entry success-log marker removed. Operator can't "
            "verify whether the call is being made."
        )
        assert "TradeAttributor record_entry FAILED" in src, (
            "P129: record_entry failure log not promoted to WARNING."
        )

    def test_all_record_exit_callers_pass_regime(self):
        """Source-level: every call to ctx.trade_attributor.record_exit
        must pass regime kwarg within the same call. Per-call check via
        windowed substring search."""
        from pathlib import Path
        src = Path("core/execution_service.py").read_text(encoding="utf-8-sig")
        idx = 0
        misses = []
        while True:
            idx = src.find("ctx.trade_attributor.record_exit(", idx)
            if idx == -1:
                break
            window = src[idx:idx + 800]  # next 800 chars span the full multi-line call
            if "regime=" not in window:
                line_no = src[:idx].count("\n") + 1
                misses.append(line_no)
            idx += 1
        assert not misses, (
            f"P129: record_exit calls at lines {misses} don't pass regime "
            f"kwarg. Orphan records will have empty metadata at those sites."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
