"""[P291] Breadth-asset routing readiness — EXPRESSIBLE, deliberately INERT.

Why this exists: the operator holds ~$12.3k of XRP entirely unmanaged (P272)
while the trend/hold mechanism is certified to transfer to exactly that asset
class (P262: trend/hold beat flat 5/5 on the five never-fitted assets; P288:
Donchian-100 XRP +4.77 out-of-selection). If the ~2026-09-15 breadth forward
read passes, widening should be a config flip, not a build — but nothing here
may move a single live order before that read.

PROVENANCE — read-only live probe, 2026-08-17, via
`docker exec hmats-engine` + coinbase-advanced-py `get_products(FUTURE)` then
per-product `get_product`. Raw output (contract_size | price_increment |
base_increment | base_min_size | price | volume_24h):

    XRP  XPP-20DEC30-CDE  cs=500    pinc=0.0001   binc=1 bmin=1  1.0026   9361
    ADA  ADP-20DEC30-CDE  cs=1000   pinc=0.0001   binc=1 bmin=1  0.1776   5997
    LTC  LCP-20DEC30-CDE  cs=5      pinc=0.01     binc=1 bmin=1  44.23     843
    DOGE DOP-20DEC30-CDE  cs=5000   pinc=0.00001  binc=1 bmin=1  0.07014  1651
    BNB  BNB-20DEC30-CDE  cs=1      pinc=0.05     binc=1 bmin=1  605.85    322
    --- incumbents re-read in the SAME probe (the control) ---
    BTC  BIP-20DEC30-CDE  cs=0.01   pinc=5        binc=1 bmin=1  63495  193404
    ETH  ETP-20DEC30-CDE  cs=0.1    pinc=0.5      binc=1 bmin=1  1902.5 132772
    SOL  SLP-20DEC30-CDE  cs=5      pinc=0.01     binc=1 bmin=1  75.52   12658

The incumbent rows reproduce the pre-existing fallback tables EXACTLY. That
agreement is the control: it proves the probe read the same venue fields the
tables encode, so the five new rows were measured the same way — not inferred
from display names or guessed from the majors' pattern.

All 24 CDE 20DEC30 products were listed; only these five of the P262 set were
added. Assets present at the venue but NOT certified by P262 (AAVE, AVAX, BCH,
ENA, HBAR, HYPE, LINK, NEAR, ONDO, PAXG, 1000PEPE, DOT, 1000SHIB, SUI, XLM,
ZCASH) are deliberately absent — venue listing is not evidence.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SET_ASSETS = REPO / "scripts" / "coinbase_set_assets.sh"

# The probe values above, restated as the test's own record. A table entry
# with no row here fails: "verified" must mean a human can see the number
# next to where it came from.
PROBED = {
    "XRP":  {"pid": "XPP-20DEC30-CDE", "cs": 500.0,  "pinc": 0.0001,  "vol24h": 9361},
    "ADA":  {"pid": "ADP-20DEC30-CDE", "cs": 1000.0, "pinc": 0.0001,  "vol24h": 5997},
    "LTC":  {"pid": "LCP-20DEC30-CDE", "cs": 5.0,    "pinc": 0.01,    "vol24h": 843},
    "DOGE": {"pid": "DOP-20DEC30-CDE", "cs": 5000.0, "pinc": 0.00001, "vol24h": 1651},
    "BNB":  {"pid": "BNB-20DEC30-CDE", "cs": 1.0,    "pinc": 0.05,    "vol24h": 322},
}
INCUMBENT_PIDS = {"BIP-20DEC30-CDE", "ETP-20DEC30-CDE", "SLP-20DEC30-CDE"}


class TestContractTablesMatchTheProbe:
    def test_every_breadth_pid_has_a_verified_contract_size(self):
        from exchange.coinbase_adapter import CoinbaseAdapter
        for asset, row in PROBED.items():
            got = CoinbaseAdapter._CONTRACT_SIZE_FALLBACK.get(row["pid"])
            assert got == row["cs"], (
                f"{asset} ({row['pid']}): table says {got}, probe measured "
                f"{row['cs']} — a contract size that disagrees with the venue "
                f"sizes every order wrong by that ratio")

    def test_every_breadth_pid_has_a_verified_price_increment(self):
        from exchange.coinbase_adapter import CoinbaseAdapter
        for asset, row in PROBED.items():
            got = CoinbaseAdapter._PRICE_INCREMENT_FALLBACK.get(row["pid"])
            assert got == row["pinc"], (
                f"{asset} ({row['pid']}): table says {got}, probe measured "
                f"{row['pinc']}")

    def test_no_table_entry_lacks_probe_provenance(self):
        # Drift guard, the OTHER direction: an entry nobody verified is how a
        # guessed contract size enters (P265h: a fabricated unit is a
        # 10-100x order). Every non-incumbent pid must appear in PROBED.
        from exchange.coinbase_adapter import CoinbaseAdapter
        known = INCUMBENT_PIDS | {r["pid"] for r in PROBED.values()}
        for table_name in ("_CONTRACT_SIZE_FALLBACK", "_PRICE_INCREMENT_FALLBACK"):
            table = getattr(CoinbaseAdapter, table_name)
            unverified = set(table) - known
            assert not unverified, (
                f"{table_name} carries {unverified} with no probe row in this "
                f"file — add the probe output to the module docstring or "
                f"remove the entry; never ship an unverified contract unit")

    def test_the_incumbent_control_still_holds(self):
        # The probe re-read BTC/ETH/SOL and reproduced the shipped values.
        # If these ever drift, the probe methodology (not just the new rows)
        # is in question and the breadth values must be re-probed too.
        from exchange.coinbase_adapter import CoinbaseAdapter
        for pid, cs, pinc in (("BIP-20DEC30-CDE", 0.01, 5.0),
                              ("ETP-20DEC30-CDE", 0.1, 0.5),
                              ("SLP-20DEC30-CDE", 5.0, 0.01)):
            assert CoinbaseAdapter._CONTRACT_SIZE_FALLBACK[pid] == cs
            assert CoinbaseAdapter._PRICE_INCREMENT_FALLBACK[pid] == pinc

    def test_p265h_symbol_map_coverage_pin_still_passes(self):
        # The P265h guard is one-directional (SYMBOL_MAP pids must all be in
        # the fallback tables). Adding fallback entries FIRST is the safe
        # order — it pre-satisfies the guard for a later SYMBOL_MAP edit.
        from exchange.coinbase_adapter import CoinbaseAdapter
        from exchange.symbol_mapping import SYMBOL_MAP
        perp_pids = set(SYMBOL_MAP["coinbase"]["perp"].values())
        assert perp_pids
        assert not perp_pids - set(CoinbaseAdapter._CONTRACT_SIZE_FALLBACK)
        assert not perp_pids - set(CoinbaseAdapter._PRICE_INCREMENT_FALLBACK)

    def test_sub_cent_ticks_survive_the_rounding_precision(self):
        # DOGE's tick is 1e-5 and _round_to_tick does round(..., 8). A tick
        # finer than 1e-8 would be silently truncated to 0 and every limit
        # price would round to a wrong multiple.
        from exchange.coinbase_adapter import CoinbaseAdapter
        for pid, inc in CoinbaseAdapter._PRICE_INCREMENT_FALLBACK.items():
            assert inc >= 1e-7, f"{pid} tick {inc} is at/below round(...,8) precision"

    def test_round_to_tick_lands_on_a_real_multiple_for_breadth(self):
        from exchange.coinbase_adapter import CoinbaseAdapter
        a = CoinbaseAdapter(rest_client=object(), paper=True)
        # Seed the cache so no network call is attempted.
        a._price_increment_cache.update(
            {r["pid"]: r["pinc"] for r in PROBED.values()})
        for asset, row in PROBED.items():
            px = a._round_to_tick(row["pid"], 1.23456789)
            steps = px / row["pinc"]
            assert abs(steps - round(steps)) < 1e-6, (
                f"{asset}: {px} is not a multiple of tick {row['pinc']}")


class TestBreadthIsInertWithoutRoutingAndMapping:
    def test_breadth_assets_are_absent_from_symbol_map(self):
        # THE inertness property. SYMBOL_MAP is the resolver the sleeve uses
        # to turn an asset into a product id; until an entry exists there,
        # a breadth asset cannot be routed no matter what the routing state
        # or the config says. Adding those entries is a separate, recorded
        # decision (it is not in this change's ownership).
        from exchange.symbol_mapping import SYMBOL_MAP
        perp = SYMBOL_MAP["coinbase"]["perp"]
        for asset in PROBED:
            assert asset not in perp, (
                f"{asset} gained a SYMBOL_MAP perp entry — breadth routing is "
                f"no longer inert. That is a widening decision and needs its "
                f"own P-entry plus per-asset caps/fractions in the live config")

    def test_to_venue_symbol_refuses_breadth_today(self):
        from exchange.symbol_mapping import to_venue_symbol
        for asset in PROBED:
            with pytest.raises((KeyError, ValueError)):
                to_venue_symbol(asset, "coinbase", "perp")

    def test_live_profile_carries_no_breadth_sizing(self):
        # The third prerequisite. Even with a map entry and routing state, a
        # breadth asset with no fraction/cap entry cannot be sized.
        prof = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8-sig"))
        for key in ("coinbase_target_fraction_by_asset",
                    "coinbase_max_contracts_by_asset"):
            for asset in PROBED:
                assert asset not in (prof.get(key) or {}), (
                    f"{asset} has a {key} entry — sizing exists for an asset "
                    f"whose forward read (~2026-09-15) has not happened")

    def test_unknown_product_still_refuses_rather_than_defaulting(self):
        # P265h semantics must survive the table growing: an asset that is
        # NOT in the tables gets None (-> the sleeve's refusal path), never
        # a borrowed unit from a neighbour.
        from exchange.coinbase_adapter import CoinbaseAdapter

        class _Boom:
            def get_product(self, **kw):
                raise RuntimeError("venue down")

        a = CoinbaseAdapter(rest_client=_Boom(), paper=True)
        assert a._contract_size("AVP-20DEC30-CDE") is None   # AVAX, listed but uncertified
        assert a._contract_size("NOT-A-REAL-PID") is None
        # ...while a verified breadth pid falls back to its probed value.
        assert a._contract_size("XPP-20DEC30-CDE") == 500.0


class TestSetAssetsScript:
    def _run(self, arg):
        # DRY RUN IS MANDATORY. Without it this script scp's a routing state
        # to the live volume and force-recreates the production engine — the
        # first version of this file did exactly that on the home-trio case
        # and restarted live trading. The routing content happened to be
        # identical, which is luck, not a safety property (P186 class).
        import os
        env = dict(os.environ, HMATS_SET_ASSETS_DRY_RUN="1")
        return subprocess.run(["bash", str(SET_ASSETS), arg],
                              capture_output=True, text=True, timeout=60,
                              cwd=str(REPO), env=env)

    def test_the_dry_run_guard_exists_and_precedes_every_side_effect(self):
        # The guard is what makes this whole test class safe to run. If it
        # moves below the scp/ssh, these tests silently start touching
        # production again.
        #
        # [P177/P238] The first version of this pin anchored on the bare
        # string "HMATS_SET_ASSETS_DRY_RUN", whose first occurrence is the
        # COMMENT above the guard — so a falsification probe that moved the
        # actual `if` block below the scp left the comment in place and the
        # pin stayed GREEN. Comments are stripped and the anchor is the
        # executable test itself.
        raw = SET_ASSETS.read_text(encoding="utf-8")
        code = "\n".join("" if ln.lstrip().startswith("#") else ln
                         for ln in raw.splitlines())
        guard = code.index('if [ "${HMATS_SET_ASSETS_DRY_RUN:-0}" = "1" ]')
        assert guard < code.index("scp "), "dry-run guard sits AFTER the scp"
        assert guard < code.index("docker compose"), "guard sits AFTER the restart"
        # ...and the guard must actually exit, not merely print.
        blk = code[guard:guard + 400]
        assert "exit 0" in blk, "the dry-run branch does not exit"

    def test_dry_run_writes_nothing_and_says_so(self):
        r = self._run("BTC")
        assert r.returncode == 0, r.stderr
        assert "DRY RUN" in r.stdout
        assert "ROUTING ->" not in r.stdout

    def test_unknown_asset_is_refused_before_anything_is_written(self):
        r = self._run("BTS")
        assert r.returncode == 2, r.stdout + r.stderr
        assert "REFUSED" in r.stderr
        # The load-bearing half: no routing state was produced.
        assert "ROUTING ->" not in r.stdout

    def test_a_typo_beside_a_valid_asset_refuses_the_whole_list(self):
        # Partial acceptance would route BTC and silently drop the typo'd
        # asset — the operator would believe both were widened.
        r = self._run("BTC,SOLL")
        assert r.returncode == 2
        assert "SOLL" in r.stderr
        assert "ROUTING ->" not in r.stdout

    def test_duplicates_are_refused(self):
        r = self._run("BTC,BTC")
        assert r.returncode == 2
        assert "duplicate" in r.stderr.lower()

    def test_breadth_assets_are_accepted_by_the_validator(self):
        # Accepted = passes validation and reaches the confirmation gate
        # (which then blocks on unmatched confirmation input). It must NOT
        # be refused as unknown.
        r = self._run("XRP")
        assert r.returncode != 2, (
            f"XRP was refused as unknown; stderr={r.stderr}")
        assert "BREADTH WIDENING REQUESTED" in r.stderr

    def test_breadth_requires_typed_confirmation(self):
        # stdin is empty under capture -> confirmation cannot match -> abort
        # with the routing state untouched.
        r = self._run("XRP")
        assert r.returncode == 3, r.stderr
        assert "ABORTED" in r.stderr
        assert "ROUTING ->" not in r.stdout

    def test_the_banner_states_the_three_load_bearing_facts(self):
        r = self._run("BNB")
        err = r.stderr
        assert "322" in err, "BNB's measured 24h volume is not in the banner"
        assert "2026-09-15" in err, "the forward read date is not in the banner"
        assert "P197" in err, "the one-asset-first rule is not in the banner"

    def test_home_trio_does_not_trigger_the_breadth_gate(self):
        # A BTC/ETH/SOL run must be unchanged by this work: no banner, no
        # confirmation prompt, straight through validation. Under dry run it
        # stops before touching anything.
        r = self._run("BTC,ETH,SOL")
        assert "BREADTH WIDENING REQUESTED" not in r.stderr
        assert r.returncode == 0, r.stderr
        assert "DRY RUN" in r.stdout

    def test_empty_string_still_means_inert(self):
        src = SET_ASSETS.read_text(encoding="utf-8")
        assert "'PRE_PHASE_2'" in src or '"PRE_PHASE_2"' in src, (
            "the OFF path lost its inert phase value")

    def test_kraken_assets_stays_the_home_trio(self):
        # Listing breadth assets under kraken_assets would assert a venue
        # fallback that does not exist (they were never Kraken-routed).
        src = SET_ASSETS.read_text(encoding="utf-8")
        m = re.search(r"'kraken_assets':\s*\[k for k in (\w+)", src)
        assert m, "kraken_assets construction changed shape — re-check it"
        assert m.group(1) == "home"

    def test_the_stale_kraken_claim_is_gone(self):
        # P152/P275: "The Kraken spot sleeve trades all 3 regardless" was
        # false from 2026-06-14. Assert the CLAIM is absent, not the words
        # (the correction note legitimately quotes it) — P177 comment trap.
        src = SET_ASSETS.read_text(encoding="utf-8")
        claim_lines = [ln for ln in src.splitlines()
                       if "trades all 3 regardless" in ln
                       and "P291 correction" not in ln
                       and not ln.lstrip().startswith("# The old header")]
        assert all("said" in ln or "falsified" in ln for ln in claim_lines), (
            "the retired Kraken claim is stated as fact somewhere in the file")
