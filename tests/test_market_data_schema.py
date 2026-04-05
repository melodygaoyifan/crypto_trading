"""
Tests for infra/market_data_schema.py v6 alignment.

Groups:
  1. Config (3 tests) - dual staleness thresholds, alias dict, v6 field list
  2. OHLCVData (3 tests) - validation, to_dict, from_dict
  3. CanonicalMarketData (5 tests) - fail-closed defaults, v6 fields, to_legacy_dict
  4. Validator (6 tests) - aliasing, dual staleness, microstructure extraction
  5. to_legacy_dict primary_asset (3 tests) - parameterization
"""
import pytest
from datetime import datetime, timezone

from infra.market_data_schema import (
    MarketDataConfig,
    OHLCVData,
    CanonicalMarketData,
    MarketDataValidator,
    assert_valid_market_data,
    get_market_data_validator,
    validate_market_data_v6,
    normalize_market_data,
)


# ============================================================================
# HELPERS
# ============================================================================

def _nested_data(**overrides) -> dict:
    """Minimal valid nested-format market data."""
    d = {
        "BTC": {"open": 95000, "high": 96000, "low": 94000, "close": 95500, "volume": 1000},
        "ETH": {"open": 3400, "high": 3500, "low": 3350, "close": 3450, "volume": 500},
        "SOL": {"open": 180, "high": 190, "low": 175, "close": 185, "volume": 2000},
        "timestamp": datetime.now(timezone.utc),
    }
    d.update(overrides)
    return d


# ============================================================================
# GROUP 1: Config (3 tests)
# ============================================================================

class TestConfig:
    def test_dual_staleness_thresholds(self):
        cfg = MarketDataConfig()
        assert cfg.MAX_OHLCV_AGE_SECONDS == 10.0
        assert cfg.MAX_MICRO_AGE_SECONDS == 2.0

    def test_backward_compat_alias(self):
        cfg = MarketDataConfig()
        assert cfg.MAX_DATA_AGE_SECONDS == cfg.MAX_OHLCV_AGE_SECONDS

    def test_field_aliases_present(self):
        cfg = MarketDataConfig()
        assert "dvol" in cfg.FIELD_ALIASES
        assert cfg.FIELD_ALIASES["dvol"] == "dvol_zscore"
        assert cfg.FIELD_ALIASES["price"] == "current_price"
        assert cfg.FIELD_ALIASES["correlation"] == "correlation_btc_sol"


# ============================================================================
# GROUP 2: OHLCVData (3 tests)
# ============================================================================

class TestOHLCVData:
    def test_valid_creation(self):
        o = OHLCVData(open=100, high=110, low=90, close=105, volume=500)
        assert o.close == 105

    def test_invalid_close_raises(self):
        with pytest.raises(ValueError, match="Invalid close"):
            OHLCVData(open=100, high=110, low=90, close=-1, volume=500)

    def test_to_dict_roundtrip(self):
        o = OHLCVData(open=100, high=110, low=90, close=105, volume=500,
                       returns_4h=0.05, volatility_4h=0.03)
        d = o.to_dict()
        assert d["close"] == 105
        assert d["returns_4h"] == 0.05
        o2 = OHLCVData.from_dict(d)
        assert o2.close == o.close


# ============================================================================
# GROUP 3: CanonicalMarketData (5 tests)
# ============================================================================

class TestCanonicalMarketData:
    def _make(self, **kw):
        defaults = dict(
            BTC=OHLCVData(95000, 96000, 94000, 95500, 1000),
            ETH=OHLCVData(3400, 3500, 3350, 3450, 500),
            SOL=OHLCVData(180, 190, 175, 185, 2000),
            timestamp=datetime.now(timezone.utc),
        )
        defaults.update(kw)
        return CanonicalMarketData(**defaults)

    def test_fail_closed_defaults(self):
        c = self._make()
        assert c.vpin is None
        assert c.ofi is None
        assert c.dvol_zscore is None
        assert c.orderbook_depth is None

    def test_v6_execution_fields_default_none(self):
        c = self._make()
        assert c.orderbook_depth_1pct_usd is None
        assert c.estimated_friction_bps is None
        assert c.flash_crash_active is None
        assert c.spread_bps is None

    def test_v6_fields_settable(self):
        c = self._make(orderbook_depth_1pct_usd=500000.0,
                       flash_crash_active=True, spread_bps=3.5)
        assert c.orderbook_depth_1pct_usd == 500000.0
        assert c.flash_crash_active is True
        assert c.spread_bps == 3.5

    def test_correlation_btc_eth_sol(self):
        c = self._make(correlation_btc_eth_sol=0.75)
        assert c.correlation_btc_eth_sol == 0.75

    def test_get_asset(self):
        c = self._make()
        assert c.get_asset("BTC").close == 95500
        assert c.get_asset("SOL").close == 185
        with pytest.raises(ValueError):
            c.get_asset("DOGE")


# ============================================================================
# GROUP 4: Validator (6 tests)
# ============================================================================

class TestValidator:
    def setup_method(self):
        self.v = MarketDataValidator()

    def test_valid_nested(self):
        canonical, err = self.v.validate(_nested_data())
        assert canonical is not None
        assert err == ""

    def test_alias_dvol_to_dvol_zscore(self):
        data = _nested_data(dvol=-2.1)
        canonical, _ = self.v.validate(data)
        assert canonical.dvol_zscore == -2.1

    def test_alias_does_not_overwrite_explicit(self):
        """If both alias and canonical key present, canonical wins."""
        data = _nested_data(dvol=-999.0, dvol_zscore=1.5)
        canonical, _ = self.v.validate(data)
        assert canonical.dvol_zscore == 1.5

    def test_ohlcv_stale_rejects(self):
        data = _nested_data(data_age_seconds=15.0)
        canonical, err = self.v.validate(data)
        assert canonical is None
        assert "OHLCV stale" in err

    def test_micro_stale_nullifies_vpin(self):
        data = _nested_data(
            data_age_seconds=5.0,
            micro_data_age_seconds=3.0,
            vpin=0.99,
        )
        canonical, _ = self.v.validate(data)
        assert canonical is not None
        assert canonical.vpin is None  # micro stale -> None

    def test_missing_vpin_stays_none(self):
        canonical, _ = self.v.validate(_nested_data())
        assert canonical.vpin is None  # not provided -> None, not 0.5


# ============================================================================
# GROUP 5: to_legacy_dict primary_asset (3 tests)
# ============================================================================

class TestToLegacyDict:
    def _make_canonical(self):
        v = MarketDataValidator()
        canonical, _ = v.validate(_nested_data())
        return canonical

    def test_default_primary_sol(self):
        c = self._make_canonical()
        d = c.to_legacy_dict()
        assert d["current_price"] == 185  # SOL close

    def test_primary_btc(self):
        c = self._make_canonical()
        d = c.to_legacy_dict(primary_asset="BTC")
        assert d["current_price"] == 95500

    def test_v6_fields_included_when_set(self):
        v = MarketDataValidator()
        data = _nested_data(orderbook_depth_1pct_usd=500000.0, spread_bps=3.5)
        canonical, _ = v.validate(data)
        d = canonical.to_legacy_dict()
        assert d["orderbook_depth_1pct_usd"] == 500000.0
        assert d["spread_bps"] == 3.5

    def test_v6_fields_absent_when_none(self):
        c = self._make_canonical()
        d = c.to_legacy_dict()
        assert "orderbook_depth" not in d
        assert "flash_crash_active" not in d


# ============================================================================
# GROUP 6: validate_market_data_v6 (5 tests)
# ============================================================================

class TestValidateMarketDataV6:
    """Lightweight flat-dict validator for v6 runtime."""

    def test_valid_flat_dict(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 1.0,
        }
        ok, issues = validate_market_data_v6(md)
        assert ok is True
        assert not any(i.startswith("missing") or i.startswith("no price") for i in issues)

    def test_missing_timestamp_fails(self):
        md = {"btc_price": 95000, "eth_price": 3400, "sol_price": 185}
        ok, issues = validate_market_data_v6(md)
        assert ok is False
        assert any("timestamp" in i for i in issues)

    def test_missing_asset_price_fails(self):
        md = {"btc_price": 95000, "timestamp": 1, "data_age_seconds": 0}
        ok, issues = validate_market_data_v6(md)
        assert ok is False
        assert any("ETH" in i for i in issues) or any("SOL" in i for i in issues)

    def test_alias_resolution(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": 1, "data_age_seconds": 0,
            "dvol": -2.0,  # alias -> dvol_zscore
        }
        ok, issues = validate_market_data_v6(md)
        assert ok is True

    def test_wrong_type_warns(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": 1, "data_age_seconds": 0,
            "vpin": "not_a_float",
        }
        ok, issues = validate_market_data_v6(md)
        assert ok is True  # type mismatch is a warning, not failure
        assert any("vpin" in i for i in issues)


# ============================================================================
# GROUP 7: normalize_market_data (8 tests)
# ============================================================================

class TestNormalizeMarketData:
    """normalize_market_data() flat-dict transformer."""

    def test_flat_input_happy_path(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 1.0,
        }
        out = normalize_market_data(md, primary_asset="SOL")
        assert out["asset"] == "SOL"
        assert out["current_price"] == 185.0
        assert out["btc_price"] == 95000.0
        assert out["eth_price"] == 3400.0
        assert out["sol_price"] == 185.0
        assert out["timestamp"] is not None
        assert out["data_age_seconds"] == 1.0

    def test_primary_asset_switches_current_price(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
        }
        btc_out = normalize_market_data(md, primary_asset="BTC")
        assert btc_out["current_price"] == 95000.0
        assert btc_out["asset"] == "BTC"

        eth_out = normalize_market_data(md, primary_asset="ETH")
        assert eth_out["current_price"] == 3400.0

    def test_nested_dict_extraction(self):
        md = {
            "BTC": {"open": 94000, "high": 96000, "low": 93000, "close": 95500, "volume": 1000},
            "ETH": {"open": 3300, "high": 3500, "low": 3200, "close": 3400, "volume": 500},
            "SOL": {"open": 180, "high": 190, "low": 175, "close": 185, "volume": 2000,
                    "returns_4h": 0.03, "volatility_4h": 0.04},
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
        }
        out = normalize_market_data(md, primary_asset="SOL")
        assert out["btc_price"] == 95500.0
        assert out["sol_price"] == 185.0
        assert out["current_price"] == 185.0
        assert out["sol_volume"] == 2000
        assert out["sol_returns_4h"] == 0.03

    def test_alias_resolution(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
            "dvol": -1.5,
        }
        out = normalize_market_data(md)
        assert out.get("dvol_zscore") == -1.5
        diag = out["_diagnostics"]
        assert any("dvol->dvol_zscore" in a for a in diag["aliased"])

    def test_missing_critical_keys_produce_diagnostics(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
        }
        out = normalize_market_data(md)
        diag = out["_diagnostics"]
        # All exec-critical keys should be missing
        for key in ("orderbook_depth", "estimated_friction_bps",
                     "flash_crash_active", "price_move_pct"):
            assert key in diag["missing"]
            assert out[key] is None  # present but None

    def test_exec_critical_keys_always_present(self):
        md = {
            "btc_price": 95000, "eth_price": 3400, "sol_price": 185,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
            "orderbook_depth": 50000.0, "flash_crash_active": True,
        }
        out = normalize_market_data(md)
        assert out["orderbook_depth"] == 50000.0
        assert out["flash_crash_active"] is True
        assert out["estimated_friction_bps"] is None  # not provided
        diag = out["_diagnostics"]
        assert "orderbook_depth" not in diag["missing"]
        assert "flash_crash_active" not in diag["missing"]

    def test_missing_price_recorded(self):
        md = {
            "btc_price": 95000,
            "timestamp": datetime.now(timezone.utc), "data_age_seconds": 0,
        }
        out = normalize_market_data(md, primary_asset="SOL")
        diag = out["_diagnostics"]
        assert "eth_price" in diag["missing"]
        assert "sol_price" in diag["missing"]
        assert "current_price" in diag["missing"]
        assert out["current_price"] is None

    def test_missing_timestamp_fallback(self):
        md = {"btc_price": 95000, "eth_price": 3400, "sol_price": 185}
        out = normalize_market_data(md)
        assert isinstance(out["timestamp"], datetime)
        assert "timestamp" in out["_diagnostics"]["missing"]


# ============================================================================
# GROUP 8: to_asset_market_data (7 tests)
# ============================================================================

class TestToAssetMarketData:
    """to_asset_market_data() per-asset dict for v6 agents."""

    def _make_canonical(self, **kw):
        v = MarketDataValidator()
        return v.validate(_nested_data(**kw))[0]

    def test_btc_current_price(self):
        c = self._make_canonical()
        d = c.to_asset_market_data("BTC")
        assert d["current_price"] == 95500
        assert d["btc_price"] == 95500
        assert d["asset"] == "BTC"

    def test_eth_current_price(self):
        c = self._make_canonical()
        d = c.to_asset_market_data("ETH")
        assert d["current_price"] == 3450
        assert d["eth_price"] == 3450

    def test_sol_current_price(self):
        c = self._make_canonical()
        d = c.to_asset_market_data("SOL")
        assert d["current_price"] == 185
        assert d["sol_price"] == 185

    def test_four_exec_keys_always_numeric(self):
        """4 exec-critical keys must be numeric/bool, never None."""
        c = self._make_canonical()  # no exec keys set -> defaults
        d = c.to_asset_market_data("BTC")
        assert isinstance(d["orderbook_depth"], (int, float))
        assert isinstance(d["estimated_friction_bps"], (int, float))
        assert isinstance(d["flash_crash_active"], bool)
        assert isinstance(d["price_move_pct"], (int, float))

    def test_friction_computed_from_spread(self):
        c = self._make_canonical(spread_bps=6.0)
        d = c.to_asset_market_data("BTC")
        assert d["estimated_friction_bps"] == 3.0  # spread / 2

    def test_flash_crash_heuristic(self):
        """Flash crash triggers when |price_move| > 8%."""
        v = MarketDataValidator()
        # BTC: open=95000, close=86000 -> (86000-95000)/95000*100 = -9.47% -> flash
        data = _nested_data()
        data["BTC"]["close"] = 86000
        c, _ = v.validate(data)
        d = c.to_asset_market_data("BTC")
        assert d["flash_crash_active"] is True
        assert abs(d["price_move_pct"]) > 8.0

    def test_alias_keys_match(self):
        """dvol and correlation aliases return same values."""
        c = self._make_canonical(dvol_zscore=-1.5, correlation_btc_sol=0.85)
        d = c.to_asset_market_data("BTC")
        assert d["dvol"] == d["dvol_zscore"]
        assert d["correlation"] == d["correlation_btc_sol"]
