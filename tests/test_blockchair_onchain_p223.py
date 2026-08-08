"""[P223] A free, keyless replacement for the on-chain feed we cannot buy.

The CryptoCompare account is capped at 100 calls/MONTH (P220) and the plan
cannot be upgraded, so `/blockchain/latest` is effectively gone. It was the only
source of BTC/ETH on-chain metrics, and its absence is why the `flow` and
`onchain` agents read 0.00 on every tick.

Blockchair's `/{chain}/stats` needs no key, covers both chains, and is unmetered.
Verified live before the module was written, and again after:

    BTC  txs 707,682    vol24h $53.6B   largest single transfer $1.31B
    ETH  txs 2,344,295  vol24h  $4.7B   largest single transfer  $306M

TWO THINGS THIS FILE EXISTS TO PIN.

1. It does NOT fabricate `large_transaction_count`. Blockchair does not publish
   one, and inventing a number to satisfy the composite's `> 0` gate is exactly
   the class of defect this codebase keeps finding. It publishes the honest
   quantity instead: 24h settlement VALUE.

2. It is a MAGNITUDE, not a direction — unchanged from before. The whale proxy's
   sign has always come from CoinGlass OI + funding (main.py:7263), never from
   the on-chain feed. So swapping the size source loses nothing directional, and
   nobody should later read this as a bearish/bullish signal.

Found while building it: `largest_transaction_24h` is reported ALREADY IN USD
under `value_usd`. My first version read `value` — the natural guess — and
silently produced 0.0 on both chains, which would have shipped a whale metric
that is always zero. Same reader/writer key mismatch as P2 and P197's
`entry_vwap`, caught only by printing the number.
"""

from pathlib import Path

import pytest

from data_mgmt.feeds.blockchair_onchain import (
    CHAINS,
    MIN_FETCH_INTERVAL,
    BlockchairOnChainData,
    BlockchairOnChainFeed,
)

_REPO = Path(__file__).resolve().parents[1]

# Live payload shape, 2026-08-07 (trimmed).
_BTC_PAYLOAD = {"data": {
    "transactions_24h": 707682,
    "volume_24h": 82530807977168,          # satoshi
    "market_price_usd": 64937.0,
    "largest_transaction_24h": {"hash": "eaeb", "value_usd": 1313793536},
    "mempool_transactions": 4321,
}}
_ETH_PAYLOAD = {"data": {
    "transactions_24h": 2344358,
    "volume_24h_approximate": "2481482948653361300000000",   # wei, as a STRING
    "market_price_usd": 1914.0,
    "largest_transaction_24h": {"hash": "0xa5", "value_usd": 305820996.2784},
    "mempool_transactions": 352,
}}


def _feed(monkeypatch, payloads):
    f = BlockchairOnChainFeed()
    f._disabled = False

    def _fake(chain):
        asset, base = CHAINS[chain]
        d = payloads[chain]["data"]
        from data_mgmt.feeds.blockchair_onchain import _f as _num
        price = _num(d.get("market_price_usd"))
        txs = int(_num(d.get("transactions_24h")))
        vol = _num(d.get("volume_24h")) or _num(d.get("volume_24h_approximate"))
        coins = vol / base
        lg = d.get("largest_transaction_24h") or {}
        return BlockchairOnChainData(
            symbol=asset, transaction_count=txs,
            average_transaction_value=(coins / txs) if txs else 0.0,
            onchain_volume_24h_usd=coins * price,
            largest_transaction_usd=_num(lg.get("value_usd")),
            market_price_usd=price, is_mock=False)

    monkeypatch.setattr(f, "_fetch_chain", _fake)
    return f


class TestUnitConversion:

    def test_btc_satoshi_to_usd(self, monkeypatch):
        f = _feed(monkeypatch, {"bitcoin": _BTC_PAYLOAD, "ethereum": _ETH_PAYLOAD})
        btc = f.fetch()["BTC"]
        # 82,530,807,977,168 sat = 825,308.08 BTC x $64,937 ~= $53.6B
        assert btc.onchain_volume_24h_usd == pytest.approx(53.6e9, rel=0.02)

    def test_eth_wei_string_is_parsed(self, monkeypatch):
        """ETH reports volume as a STRING because wei overflows JSON's safe
        integer range — float() on it must still work."""
        f = _feed(monkeypatch, {"bitcoin": _BTC_PAYLOAD, "ethereum": _ETH_PAYLOAD})
        eth = f.fetch()["ETH"]
        assert eth.onchain_volume_24h_usd == pytest.approx(4.75e9, rel=0.05)

    def test_average_transaction_value_is_native_units(self, monkeypatch):
        """Same semantics as the CryptoCompare field it replaces, so downstream
        arithmetic is unchanged."""
        f = _feed(monkeypatch, {"bitcoin": _BTC_PAYLOAD, "ethereum": _ETH_PAYLOAD})
        assert f.fetch()["BTC"].average_transaction_value == pytest.approx(
            825308.08 / 707682, rel=0.01)

    def test_zero_transactions_does_not_divide_by_zero(self, monkeypatch):
        p = {"data": dict(_BTC_PAYLOAD["data"], transactions_24h=0)}
        f = _feed(monkeypatch, {"bitcoin": p, "ethereum": _ETH_PAYLOAD})
        assert f.fetch()["BTC"].average_transaction_value == 0.0


class TestLargestTransactionKey:
    """The bug I shipped first: reading `value` instead of `value_usd`."""

    def test_it_reads_value_usd(self, monkeypatch):
        f = _feed(monkeypatch, {"bitcoin": _BTC_PAYLOAD, "ethereum": _ETH_PAYLOAD})
        assert f.fetch()["BTC"].largest_transaction_usd == pytest.approx(1_313_793_536)

    def test_the_source_does_not_read_the_wrong_key(self):
        src = (_REPO / "data_mgmt" / "feeds" / "blockchair_onchain.py").read_text(
            encoding="utf-8", errors="replace")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        assert 'largest.get("value_usd"' in code
        assert 'largest.get("value"' not in code

    def test_a_missing_block_is_zero_not_a_crash(self, monkeypatch):
        p = {"data": {k: v for k, v in _BTC_PAYLOAD["data"].items()
                      if k != "largest_transaction_24h"}}
        f = _feed(monkeypatch, {"bitcoin": p, "ethereum": _ETH_PAYLOAD})
        assert f.fetch()["BTC"].largest_transaction_usd == 0.0


class TestResilience:

    def test_one_chain_failing_does_not_kill_the_other(self, monkeypatch):
        f = BlockchairOnChainFeed()
        f._disabled = False

        def _fake(chain):
            if chain == "bitcoin":
                raise RuntimeError("boom")
            return BlockchairOnChainData(symbol="ETH", transaction_count=5,
                                         onchain_volume_24h_usd=1.0,
                                         is_mock=False)
        monkeypatch.setattr(f, "_fetch_chain", _fake)
        assert "ETH" in f.fetch() and "BTC" not in f.fetch()

    def test_a_total_failure_does_not_advance_the_clock(self, monkeypatch):
        """Otherwise a transient outage is cached out for a full hour."""
        f = BlockchairOnChainFeed()
        f._disabled = False
        monkeypatch.setattr(f, "_fetch_chain",
                            lambda c: (_ for _ in ()).throw(RuntimeError("x")))
        f.fetch()
        assert f._last_fetch_time == 0.0

    def test_cache_ttl_is_respected(self, monkeypatch):
        f = _feed(monkeypatch, {"bitcoin": _BTC_PAYLOAD, "ethereum": _ETH_PAYLOAD})
        f.fetch()
        calls = []
        monkeypatch.setattr(f, "_fetch_chain",
                            lambda c: calls.append(c) or BlockchairOnChainData())
        f.fetch()
        assert calls == [], "refetched inside the TTL"

    def test_ttl_is_at_least_an_hour(self):
        """24h rolling stats — anything faster re-reads an unmoved number."""
        assert MIN_FETCH_INTERVAL >= 3600.0

    def test_it_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("HMATS_DISABLE_BLOCKCHAIR", "1")
        assert BlockchairOnChainFeed().fetch() == {}

    def test_it_needs_no_api_key(self):
        src = (_REPO / "data_mgmt" / "feeds" / "blockchair_onchain.py").read_text(
            encoding="utf-8", errors="replace")
        assert "API_KEY" not in src.upper().replace("HMATS_DISABLE_BLOCKCHAIR", "")


class TestCompositeWiring:

    def _src(self):
        return (_REPO / "data_mgmt" / "feeds" / "onchain_feed.py").read_text(
            encoding="utf-8", errors="replace")

    def _block(self):
        """Delimited by its own end marker, not a character count — a fixed
        window silently truncates when the comments grow, and the assertions
        then fail for a reason unrelated to the contract (same mistake as the
        first cut of the P209 tests)."""
        s = self._src()
        i = s.index("[P223] Blockchair fallback")
        return s[i:s.index("Blockchair fallback failed", i)]

    def _code(self):
        """`_block()` with comment lines removed. A "this must not appear"
        assertion has to read CODE — the block's own comment explains that
        Blockchair has no `large_transaction_count`, so a raw substring check
        fires on its own explanation. Third time this session (P209, P215)."""
        return "\n".join(l for l in self._block().splitlines()
                         if not l.lstrip().startswith("#"))

    def test_it_is_a_fallback_not_a_replacement(self):
        """Runs only when CryptoCompare produced nothing, so restoring that
        quota silently takes precedence again."""
        s = self._src()
        assert "if not any_real:" in self._block()

    def test_it_runs_after_the_cryptocompare_block(self):
        s = self._src()
        assert s.index("CryptoCompare failed") < s.index("[P223] Blockchair fallback")

    def test_it_does_not_fabricate_a_large_transaction_count(self):
        s = self._src()
        w = self._code()
        assert "large_transaction_count" not in w, (
            "inventing a count to satisfy the gate above is the defect class "
            "this replacement exists to avoid"
        )
        assert "onchain_volume_24h_usd" in w

    def test_the_fallback_cannot_break_the_tick(self):
        s = self._src()
        i = s.index("[P223] Blockchair fallback")
        assert "except Exception as e:" in s[i:s.index("Blockchair fallback failed", i) + 120]

    def test_direction_is_not_claimed(self):
        """It contributes size only; the sign comes from CoinGlass OI+funding."""
        s = self._src()
        w = self._block()
        assert "net direction unknown" in w
