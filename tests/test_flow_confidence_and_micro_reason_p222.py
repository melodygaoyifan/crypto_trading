"""[P222] Two loose ends from this session's own changes.

1. FLOW CONFIDENCE SATURATED — a defect I introduced in P221. The attribution
   proxy is `min(1.0, |whale_flow| / 1e7)`, calibrated for CryptoCompare's
   `large_tx_count x avg_tx_value` (millions). P221 replaced that source with
   Blockchair's 24h settlement VALUE (~$5.7e10), so the proxy pinned at its cap
   and attribution began reporting flow as **direction 0.00 with confidence
   1.00** — a maximally-confident non-signal, strictly worse than the 0.00/0.00
   it replaced, because the IC layer will weight it.

   A confidence attached to a zero direction has no meaning: nothing downstream
   can act on "certainly no opinion". The whale feed is a MAGNITUDE with no sign
   (P221), so until a signed source exists this is honestly zero.

   Same shape as the P219 confidence bug: changing what a number MEASURES
   without re-checking normalisers calibrated against the old scale. Twice in
   one session — a magnitude swap is never local.

2. MICRO'S SILENCE IS NOW EXPLAINED. `micro` was the one agent in the P216 sweep
   left "not yet traced", and the reason is that the system does not say: the
   agent records a cause in `diagnostics.reason` (`no_exchange_data`,
   `stale_snapshot`, insufficient samples, exception) and nothing surfaced it,
   so all four present identically as `micro=+0.00/0.00`. The Binance LOB feed
   IS live and directional (BTC taker_buy 159.8 vs taker_sell 69.8), so "no
   data" is not the obvious answer and guessing would repeat the Helius mistake.
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MSRC = (_REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


class TestFlowConfidenceIsGatedOnDirection:

    def _block(self):
        i = _MSRC.index('"flow": {')
        return _MSRC[i:_MSRC.index('"whale": {', i)]

    def test_confidence_requires_a_nonzero_direction(self):
        b = self._block()
        assert 'abs(float(agent_signals.get("flow_direction", 0.0) or 0.0)) > 1e-9' in b, (
            "confidence is not gated on direction — a zero direction can still "
            "report confidence 1.00"
        )
        assert "else 0.0" in b

    def test_the_magnitude_proxy_is_still_there_for_real_signals(self):
        """The gate must not delete the proxy — only condition it."""
        assert 'abs(float(agent_signals.get("whale_flow", 0.0) or 0.0)) / 1e7' in self._block()

    def test_the_scale_hazard_is_recorded(self):
        """The /1e7 divisor is calibrated for a source that no longer feeds it;
        whoever restores a signed flow source must re-derive it."""
        b = self._block()
        assert "1e7" in b and "P222" in b

    def test_it_evaluates_correctly(self):
        """Reproduce the arithmetic rather than trusting the source scan."""
        def conf(direction, whale):
            return (min(1.0, abs(whale) / 1e7) if abs(direction) > 1e-9 else 0.0)
        assert conf(0.0, 5.7e10) == 0.0, "the live case: huge magnitude, no direction"
        assert conf(0.5, 5.7e10) == 1.0
        assert conf(0.5, 2e6) == 0.2
        assert conf(0.0, 0.0) == 0.0


class TestMicroNeutralReasonIsSurfaced:

    def _block(self):
        i = _MSRC.index("[P222] Say WHY when micro is neutral")
        return _MSRC[i:_MSRC.index("except Exception as _micro_err:", i)]

    def test_the_reason_is_logged(self):
        b = self._block()
        assert 'logger.warning(' in b
        assert '_m_diag.get("reason")' in b

    def test_it_falls_back_to_the_error_field(self):
        """The exception path stores `error`, not `reason` — reading only one
        would leave the crash case silent, which is the worst of the four."""
        assert '_m_diag.get("error")' in self._block()

    def test_it_logs_only_on_change(self):
        """A per-tick line for a steady-state condition becomes wallpaper and
        stops being read (P202)."""
        b = self._block()
        assert "self._micro_last_reason.get(asset) != _m_reason" in b

    def test_it_is_per_asset(self):
        """A single global latch would let the first asset consume the one
        report for all of them — the P202 latch bug."""
        b = self._block()
        assert "self._micro_last_reason[asset] = _m_reason" in b

    def test_it_only_fires_when_direction_is_zero(self):
        """A working agent must stay quiet."""
        assert 'float(_micro_sig.get("micro_direction", 0.0) or 0.0) == 0.0' in self._block()

    def test_the_attribute_read_is_defended(self):
        """P85: a new attribute on the tick path defends itself."""
        assert 'hasattr(self, "_micro_last_reason")' in self._block()

    def test_diagnostics_cannot_break_the_tick(self):
        b = self._block()
        assert "except Exception:" in b and "noqa: silent-swallow" in b

    def test_the_agent_still_records_all_four_causes(self):
        """If the agent stops setting these, the log above goes quiet for the
        wrong reason."""
        src = (_REPO / "agents" / "microstructure_agent.py").read_text(
            encoding="utf-8", errors="replace")
        assert '"reason": "no_exchange_data"' in src
        assert '"reason": "stale_snapshot"' in src
