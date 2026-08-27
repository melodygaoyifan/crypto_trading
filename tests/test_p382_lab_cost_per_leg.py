"""[P382] Three labs charged the ROUND-TRIP cost on every LEG — a 2x overcharge
that moved verdicts — and the class gets a mechanism instead of a third audit.

THE DEFECT. `COST_RT` is a ROUND-TRIP cost everywhere in this repo, and a leg is
charged `COST_RT/2 x |dpos|` (P166/P167/P281; `mechanism_lab.pnl_after_cost`,
`strategy_threshold_audit_lab.book_pnl`, `risk_control_audit_lab`). Three labs
written after that convention was settled each carried a PRIVATE `COST_RT` dict
and charged the full RT per unit |dpos|:

    overlay_backtest_lab.py   p = p - turn * COST_RT[asset]            (P377)
    xsmom_backtest_lab.py     pnl[i] -= dot(|new-pos|, COST_RT)       (P373)
    funding_carry_xs_lab.py   pnl[i] -= dot(|new-pos|, COST_RT)       (P374)

The P287 sweep (`test_no_other_full_rt_per_leg_copy_survives_in_training`)
could not see them: its regex keys on `np.abs(np.diff(...)) *` on ONE line, and
these sites assign the turnover to a variable first or charge through `np.dot`.
A guard that matches one SHAPE of a defect is not a guard against the class.

THE FIX has two layers here:
  1. PARITY (behavioural): one identical 0->1->0 round trip is charged through
     each lab's real charge function AND through `mechanism_lab.pnl_after_cost`,
     and all four must agree exactly — 2 legs == exactly one RT. Each lab's
     `backtest`/`run_asset` is also driven END TO END on a synthetic series, so
     the named function cannot be decoration (P170): if the loop stops calling
     it, or charges full RT per leg again, the end-to-end numbers double.
  2. ROSTER (structural): every file under training/ that defines its own
     `COST_*` dict must be a key in COST_DICT_OWNERS, naming HOW its arithmetic
     is covered and carrying a pin that must still match its source (an
     allowlist entry that no longer describes reality is coverage that is
     not, P361). The next lab with private cost arithmetic fails THIS test
     instead of an audit.

Falsification (recorded in the P382 entry): reverting one lab to full-RT-per-
leg turns the parity AND the end-to-end tests red; restored byte-identically.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from _source_scan import code_only  # noqa: E402


def _lab(name):
    return importlib.import_module(f"training.{name}")


# ── 1. parity: one 0->1->0 round trip, four implementations, one number ───────

RT_BPS = 60.0                       # chosen so RT/2 per leg (0.003) is exact at 4dp
RT_FRAC = RT_BPS / 1e4              # 0.006
POS = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # enter, hold, exit = 2 legs
DPOS = np.abs(np.diff(POS, prepend=POS[0]))       # [0,1,0,1,0,0]


class TestOneRoundTripCostsOneRoundTrip:
    def test_mechanism_lab_reference_charges_exactly_one_rt(self):
        ml = _lab("mechanism_lab")
        close = np.full(len(POS), 100.0)
        out = ml.pnl_after_cost(close, POS, RT_BPS, 0, len(POS))
        assert out["turnover_units"] == 2.0
        assert abs(out["cost"] - RT_FRAC) < 1e-12

    def test_overlay_lab_matches_the_reference(self, monkeypatch):
        lab = _lab("overlay_backtest_lab")
        monkeypatch.setitem(lab.COST_RT, "BTC", RT_FRAC)
        charged = float(np.sum(lab.turnover_cost(DPOS, "BTC")))
        assert abs(charged - RT_FRAC) < 1e-12, (
            f"overlay charged {charged} for one 0->1->0 trip; the repo "
            f"convention is RT/2 per leg = {RT_FRAC} (P166/P281)")

    def test_xsmom_lab_matches_the_reference(self, monkeypatch):
        lab = _lab("xsmom_backtest_lab")
        monkeypatch.setitem(lab.COST_RT, "BTC", RT_FRAC)
        charged = sum(lab.turnover_cost(np.array([d]), ("BTC",)) for d in DPOS)
        assert abs(charged - RT_FRAC) < 1e-12

    def test_funding_carry_lab_matches_the_reference(self, monkeypatch):
        lab = _lab("funding_carry_xs_lab")
        monkeypatch.setitem(lab.COST_RT, "BTC", RT_FRAC)
        charged = sum(lab.turnover_cost(np.array([d]), ("BTC",)) for d in DPOS)
        assert abs(charged - RT_FRAC) < 1e-12

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_all_four_agree_at_the_real_cde_constants(self, asset):
        """Same trip, the labs' real per-asset constants, vs mechanism_lab at the
        same RT in bps. mechanism_lab rounds its cost to 4 dp, so the tolerance
        is half a unit in the 4th decimal; the three labs must agree with each
        other to 1e-12 (they carry the identical constant)."""
        ml = _lab("mechanism_lab")
        ov, xs, fc = (_lab("overlay_backtest_lab"), _lab("xsmom_backtest_lab"),
                      _lab("funding_carry_xs_lab"))
        rt_frac = ov.COST_RT[asset]
        assert xs.COST_RT[asset] == rt_frac == fc.COST_RT[asset], (
            "the three labs no longer carry the same measured CDE constant")
        c_ov = float(np.sum(ov.turnover_cost(DPOS, asset)))
        c_xs = sum(xs.turnover_cost(np.array([d]), (asset,)) for d in DPOS)
        c_fc = sum(fc.turnover_cost(np.array([d]), (asset,)) for d in DPOS)
        assert abs(c_ov - rt_frac) < 1e-12 and abs(c_xs - rt_frac) < 1e-12 \
            and abs(c_fc - rt_frac) < 1e-12
        ref = ml.pnl_after_cost(np.full(len(POS), 100.0), POS, rt_frac * 1e4,
                                0, len(POS))["cost"]
        assert abs(ref - c_ov) <= 5e-5

    def test_a_single_leg_is_half_a_round_trip_in_every_lab(self, monkeypatch):
        """The exact property the defect violated, stated on one leg."""
        one_leg = np.array([1.0])
        for name, call in (
            ("overlay_backtest_lab", lambda lab: float(np.sum(lab.turnover_cost(one_leg, "BTC")))),
            ("xsmom_backtest_lab", lambda lab: lab.turnover_cost(one_leg, ("BTC",))),
            ("funding_carry_xs_lab", lambda lab: lab.turnover_cost(one_leg, ("BTC",))),
        ):
            lab = _lab(name)
            monkeypatch.setitem(lab.COST_RT, "BTC", RT_FRAC)
            assert abs(call(lab) - RT_FRAC / 2.0) < 1e-12, name


# ── 2. end to end: the named function is the charge site, not decoration ─────

BIG = 0.5   # 5,000bps RT so the 0.1%-rounded report totals cannot hide a 2x


class TestEndToEndChargeSites:
    def test_overlay_run_asset_charges_half_rt_per_leg(self, monkeypatch):
        """Flat 100 for 300 bars, step to 110, flat: the overlay goes long at
        bar 301 (c[300] > SMA) and exits once the SMA has caught up (c == SMA
        is not >), i.e. exactly 2 legs; hold pays its single entry leg. At a
        flat price after the step, the only PnL is cost."""
        lab = _lab("overlay_backtest_lab")
        n = 800
        c = np.full(n, 100.0); c[300:] = 110.0
        idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
        monkeypatch.setattr(lab, "load_4h", lambda asset: pd.Series(c, index=idx))
        monkeypatch.setitem(lab.COST_RT, "BTC", BIG)
        out = lab.run_asset("BTC")
        # overlay: 2 legs x BIG/2 = BIG = -50.0% ; hold: +10% step - 1 leg = -15.0%
        assert abs(out["overlay"]["total_pct"] - (-50.0)) < 0.15, out["overlay"]
        assert abs(out["hold"]["total_pct"] - (-15.0)) < 0.15, out["hold"]

    def test_xsmom_backtest_charges_half_rt_per_leg(self, monkeypatch):
        """Two assets, A trends up and B is flat, k=1 long-only: the book is
        A from the first rebalance and never changes — exactly one leg.
        Price PnL is removed by comparing a zero-cost run to a BIG-cost run."""
        lab = _lab("xsmom_backtest_lab")
        n = lab.LOOKBACK_BARS + 24 * 40
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        panel = pd.DataFrame({"A": 100.0 + np.arange(n) * 0.01, "B": 100.0}, index=idx)
        monkeypatch.setitem(lab.COST_RT, "A", 0.0)
        monkeypatch.setitem(lab.COST_RT, "B", 0.0)
        free = lab.backtest(panel, ("A", "B"), k=1, long_only=True)["total_pct"]
        monkeypatch.setitem(lab.COST_RT, "A", BIG)
        monkeypatch.setitem(lab.COST_RT, "B", BIG)
        paid = lab.backtest(panel, ("A", "B"), k=1, long_only=True)["total_pct"]
        assert abs((free - paid) - BIG / 2.0 * 100) < 0.15, (free, paid)

    def test_funding_carry_backtest_charges_half_rt_per_leg(self, monkeypatch):
        """Flat prices, A always the lowest funder (0 vs +1bp), k=1 long-only:
        the book is A from day 2 and never changes — exactly one leg, zero
        carry on A, zero price PnL. Total == -BIG/2 exactly."""
        lab = _lab("funding_carry_xs_lab")
        n = 200
        idx = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
        P = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
        F = pd.DataFrame({"A": 0.0, "B": 1e-4}, index=idx)
        monkeypatch.setattr(lab, "UNIVERSE", ("A", "B"))
        monkeypatch.setitem(lab.COST_RT, "A", BIG)
        monkeypatch.setitem(lab.COST_RT, "B", BIG)
        out = lab.backtest(P, F, k=1, long_only=True)
        assert abs(out["total_pct"] - (-BIG / 2.0 * 100)) < 0.15, out


class TestTheChargeSiteIsTheNamedFunction:
    """Source pins, comments + docstrings stripped (P177/P179): the loops must
    charge THROUGH turnover_cost, and the old full-RT expression must be gone."""

    @pytest.mark.parametrize("fname,must,must_not", [
        ("overlay_backtest_lab.py",
         [r"p - turnover_cost\(turn, asset\)"],
         # the charge SITE multiplying the constant directly (turnover_cost's
         # own body legitimately reads `turn * (COST_RT[asset] / 2.0)`)
         [r"p - turn \* COST_RT\[", r"p - turn \* \(COST_RT\["]),
        ("xsmom_backtest_lab.py",
         [r"turnover_cost\(np\.abs\(new - pos\), assets\)"],
         [r"np\.dot\(np\.abs\(new - pos\), costs\)"]),
        ("funding_carry_xs_lab.py",
         [r"turnover_cost\(np\.abs\(new-pos\), UNIVERSE\)"],
         [r"np\.dot\(np\.abs\(new-pos\), costs\)"]),
    ])
    def test_loop_charges_through_turnover_cost(self, fname, must, must_not):
        src = code_only(REPO / "training" / fname, strip_docstrings=True)
        for pat in must:
            assert re.search(pat, src), f"{fname}: charge site no longer calls {pat}"
        for pat in must_not:
            assert not re.search(pat, src), (
                f"{fname}: the pre-P382 direct charge `{pat}` is back — a second "
                f"charge site beside turnover_cost is how the 2x returns")

    @pytest.mark.parametrize("fname", ["overlay_backtest_lab.py",
                                       "xsmom_backtest_lab.py",
                                       "funding_carry_xs_lab.py"])
    def test_turnover_cost_halves_the_round_trip_constant(self, fname):
        src = code_only(REPO / "training" / fname, strip_docstrings=True)
        m = re.search(r"def turnover_cost\(.*?\n(?:    .*\n)+", src)
        assert m, f"{fname}: turnover_cost missing"
        assert re.search(r"COST_RT\[[^\]]+\] / 2\.0", m.group(0)), (
            f"{fname}: turnover_cost no longer halves COST_RT — COST_RT is "
            f"ROUND-TRIP and a unit of |dpos| is one LEG (P166/P281/P382)")


# ── 3. roster: every private COST dict in training/ is owned and pinned ──────

COST_DICT_RE = re.compile(r"^[A-Za-z_]*COST[A-Za-z_]*\s*=\s*\{", re.M)

# file (posix, relative to REPO) -> how its cost arithmetic is covered + a
# source pin that must still match (comments/docstrings stripped). A new file
# defining a COST_* dict fails this roster until someone adds it HERE with a
# parity test or a stated per-RT/per-event reason — that is the point.
COST_DICT_OWNERS = {
    "training/conviction_sizing_lab.py": {
        "covered_by": "charges through sizing_overlay_lab.per_bar_net "
                      "(the P382-corrected per-leg chassis; WS2)",
        "pins": [r"per_bar_net\(close, "]},
    "training/overlay_backtest_lab.py": {
        "covered_by": "this file (P382 parity + end-to-end)",
        "pins": [r"turnover_cost\(turn, asset\)"]},
    "training/xsmom_backtest_lab.py": {
        "covered_by": "this file (P382 parity + end-to-end)",
        "pins": [r"turnover_cost\(np\.abs\(new - pos\), assets\)"]},
    "training/funding_carry_xs_lab.py": {
        "covered_by": "this file (P382 parity + end-to-end)",
        "pins": [r"turnover_cost\(np\.abs\(new-pos\), UNIVERSE\)"]},
    "training/train_supervised_full.py": {
        "covered_by": "the reference: evaluate_segment charges RT/2 per leg "
                      "(P281; tests/test_p287_training.py::TestCostConventionParity)",
        "pins": [r"np\.abs\(np\.diff\(pos_full\)\) \* \(cost_bps / 2\.0\)"]},
    "training/strategy_threshold_audit_lab.py": {
        "covered_by": "book_pnl charges RT/2 per leg at the site (P370)",
        "pins": [r"\(COST_RT_BPS\[asset\] / 2\.0\)"]},
    "training/unread_era_probe.py": {
        "covered_by": "pnl() halves cost_rt per |dpos| unit (P262)",
        "pins": [r"\(cost_rt / 2\.0\)"]},
    "training/risk_control_audit_lab.py": {
        "covered_by": "per-EVENT accounting, not |dpos|: entry is charged "
                      "nothing and the exit charges the full RT once; a "
                      "REDUCE_50 + restore pair charges RT/2 (= one RT on half "
                      "a unit). One RT per unit round trip, verified by "
                      "reading (P369/P370); a state machine, not a formula",
        "pins": [r"size = 1\.0; pnl\[i\] -= 0",        # entry leg free
                 r"held = False; pnl\[i\] -= cost",      # exit charges the RT
                 r"size = 0\.5; pnl\[i\] -= cost / 2"]},  # half-unit reduce
    "training/watchdog_replay_lab.py": {
        "covered_by": "no leg charge: COST_RT_BPS is SUBTRACTED as one round "
                      "trip from a flatten's forward-return value (P369)",
        "pins": [r"- COST_RT_BPS\[asset\]"]},
    "training/scripts/microstructure_probe.py": {
        "covered_by": "no leg charge: one RT compared against gross per trade "
                      "+ required-IC arithmetic (P379)",
        "pins": [r"gross - COST_RT\[asset\] \* 1e4"]},
    "training/scripts/edge_probe_hf.py": {
        "covered_by": "no leg charge: one RT compared against gross per trade "
                      "+ required-IC arithmetic (P375b)",
        "pins": [r"gross - COST_CDE_RT\[asset\]"]},
    "training/scripts/signal_hold_backtest.py": {
        "covered_by": "per-leg RT/2: PER_LEG halves COST_RT per |dpos| unit, "
                      "cost = dpos * per_leg (P386 hold-position backtest)",
        "pins": [r"COST_RT\[a\] / 2\.0 / 1e4"]},
    "training/scripts/microstructure_edge_probe.py": {
        "covered_by": "no leg charge: required-IC arithmetic compares one RT "
                      "against achievable IC/gross (P385b)",
        "pins": [r"COST_RT\[a\] / \(E_ABS_Z"]},
    "training/scripts/breadth_edge_probe.py": {
        "covered_by": "no leg charge: required-IC arithmetic compares one RT "
                      "against achievable IC/gross (P385c)",
        "pins": [r"COST_RT\[tgt\] / \(E_ABS_Z"]},
    "training/scripts/macro_factor_lab.py": {
        "covered_by": "no leg charge: RT enters only the P166 required-IC "
                      "bar (edge >= 2*cost); values pinned equal to "
                      "2 x core.cde_fees.CDE_FEE_BPS by "
                      "tests/test_p392_macro_factor_lab.py (P392)",
        "pins": [r"cost = RT_COST_BPS\[asset\]"]},
    "training/scripts/etf_flow_probe.py": {
        "covered_by": "per-leg RT/2: _hold charges dp * (pl/2.0) so a flip pays "
                      "one full round-trip (P400 ETF-flow lag-1 Rung-0)",
        "pins": [r"dp \* \(pl / 2\.0\)"]},
    "training/scripts/etf_sma_overlay_check.py": {
        "covered_by": "per-leg RT/2: _stats charges dp * (pl/2.0) so a flip pays "
                      "one full round-trip (P404 ETF+SMA200 complementarity check)",
        "pins": [r"dp \* \(pl / 2\.0\)"]},
    "training/scripts/cot_probe.py": {
        "covered_by": "per-leg RT/2: _hold charges dp * (pl/2.0) so a flip pays "
                      "one full round-trip (P402 CFTC COT Rung-0, NOT_EARNED)",
        "pins": [r"dp \* \(pl / 2\.0\)"]},
    "training/scripts/coinmetrics_onchain_probe.py": {
        "covered_by": "per-leg RT/2: _hold charges dpos * (per_leg/2.0) so a flip "
                      "pays one full round-trip (P397 CoinMetrics on-chain Rung-0)",
        "pins": [r"dpos \* \(per_leg / 2\.0\)"]},
    "training/scripts/exchange_flow_probe.py": {
        "covered_by": "per-leg RT/2: _hold charges dpos * (per_leg/2.0) so a flip "
                      "pays one full round-trip (P396 exchange-flow Rung-0)",
        "pins": [r"dpos \* \(per_leg / 2\.0\)"]},
    "training/scripts/newdata_gated_probe.py": {
        "covered_by": "per-leg RT/2: _hold_stats charges dpos * (per_leg/2.0) so a "
                      "flip pays one full round-trip (P396 gated put/call probe)",
        "pins": [r"dpos \* \(per_leg / 2\.0\)"]},
    "training/scripts/metrics_oos_probe.py": {
        "covered_by": "per-leg RT/2: hold_sim charges dpos * (per_leg/2.0) so a "
                      "flip (|dpos|=2) pays one full round-trip and an entry "
                      "from flat pays one leg (P391 OI-positioning OOS gate)",
        "pins": [r"dpos \* \(per_leg / 2\.0\)"]},
}


def _files_defining_cost_dicts():
    found = {}
    for p in (REPO / "training").rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if "/training_data/" in rel or "/reports/" in rel:
            continue
        try:
            src = code_only(p, strip_docstrings=True)
        except Exception:
            continue
        if COST_DICT_RE.search(src):
            found[rel] = src
    return found


class TestPrivateCostDictRoster:
    def test_scanner_is_not_vacuous(self):
        found = _files_defining_cost_dicts()
        assert len(found) >= 5, found.keys()
        for must in ("training/overlay_backtest_lab.py", "training/xsmom_backtest_lab.py",
                     "training/funding_carry_xs_lab.py", "training/train_supervised_full.py"):
            assert must in found, f"scanner no longer sees {must}"

    def test_every_private_cost_dict_is_owned(self):
        found = _files_defining_cost_dicts()
        orphans = sorted(set(found) - set(COST_DICT_OWNERS))
        assert not orphans, (
            f"{orphans} define a private COST_* dict and are not in "
            f"COST_DICT_OWNERS. Either charge through an existing per-leg "
            f"helper (mechanism_lab.pnl_after_cost) or add a parity test and "
            f"register the file here with how it is covered. P382: the last "
            f"three labs that did this charged the ROUND-TRIP cost on every "
            f"LEG and moved three verdicts.")

    def test_every_owner_entry_still_describes_reality(self):
        """An exemption must name something that still exists AND its pin must
        still match (P361): a roster that can hold stale entries is a parking
        spot, not coverage."""
        found = _files_defining_cost_dicts()
        for rel, spec in COST_DICT_OWNERS.items():
            assert rel in found, (
                f"{rel} is in COST_DICT_OWNERS but no longer defines a COST_* "
                f"dict — remove the entry")
            src = found[rel]
            for pat in spec["pins"]:
                assert re.search(pat, src), (
                    f"{rel}: pin `{pat}` no longer matches — its cost "
                    f"arithmetic moved; re-verify the per-leg convention and "
                    f"re-pin (covered_by: {spec['covered_by']})")

    def test_roster_pins_are_real_pins(self):
        # anti-vacuity for the roster itself: a pin that matches nothing
        # anywhere would be decoration; every pin must be a non-trivial regex
        for rel, spec in COST_DICT_OWNERS.items():
            assert spec["covered_by"].strip()
            assert spec["pins"], rel
            for pat in spec["pins"]:
                assert len(pat) >= 12, (rel, pat)
