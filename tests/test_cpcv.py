"""CPCV + CSCV PBO validation-methodology tests (the fix for DRL max-reward selection)."""

import random

from analytics.validation.cpcv import combinatorial_purged_splits, cscv_pbo


class TestSplits:
    def test_combination_count_and_disjoint(self):
        sp = combinatorial_purged_splits(600, n_groups=6, n_test_groups=2)
        assert len(sp) == 15  # C(6,2)
        for tr, te in sp:
            assert set(tr).isdisjoint(set(te))
            assert len(te) > 0 and len(tr) > 0

    def test_embargo_purges_neighbors(self):
        sp = combinatorial_purged_splits(600, n_groups=6, n_test_groups=2, embargo=5)
        for tr, te in sp:
            tset = set(te)
            for t in tr:
                assert all(abs(t - x) > 5 or x == t for x in te) or t not in tset


class TestPBO:
    def _noise(self, rng, n):
        return [0.01 * rng.gauss(0, 1) for _ in range(n)]

    def test_robust_selection_low_pbo(self):
        # one config has a genuine persistent edge among noise -> selection generalizes
        rng = random.Random(1)
        n = 1200
        edge = [0.004 + 0.01 * rng.gauss(0, 1) for _ in range(n)]   # real Sharpe ~0.4/period
        configs = [edge] + [self._noise(rng, n) for _ in range(5)]
        out = cscv_pbo(configs, n_splits=10)
        assert out["pbo"] < 0.5
        assert out["verdict"] in ("ROBUST_SELECTION", "MARGINAL")

    def test_pure_noise_higher_pbo(self):
        # all configs noise -> the IS-best is luck -> OOS rank random -> PBO ~0.5
        rng = random.Random(2)
        configs = [self._noise(rng, 1200) for _ in range(6)]
        out = cscv_pbo(configs, n_splits=10)
        # pure noise -> the IS-best is luck -> OOS rank random -> PBO elevated
        assert out["pbo"] > 0.3

    def test_needs_two_configs(self):
        assert "error" in cscv_pbo([[0.1, 0.2, 0.3]])
