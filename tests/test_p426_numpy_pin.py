"""[P426] The image's numpy must be the numpy CI tests under.

On 2026-10-12 a rebuild let the unpinned pip lines in Dockerfile.engine
re-resolve numpy to 2.5.4 (contourpy 1.4.0 requires numpy>=2.0) against the
<2.0 pin in requirements-runtime.txt. Every shadow TQC predict then raised
"only 0-dimensional arrays can be converted to Python scalars" for 1.5 days,
at WARNING, and nothing else said the runtime had left the tested environment.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from drl.ensemble import action_to_scalar

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = ["Dockerfile.engine", "Dockerfile"]


def _run_block(name: str) -> str:
    """The RUN instruction containing the pip installs, comments stripped."""
    lines = [ln for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("#")]
    blocks, cur = [], None
    for ln in lines:
        if cur is None and ln.startswith('RUN '):
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
        else:
            continue
        if not cur[-1].rstrip().endswith(chr(92)):
            blocks.append("\n".join(cur))
            cur = None
    hits = [b for b in blocks if "requirements-runtime.txt" in b and "pip install" in b]
    assert len(hits) == 1, f"{name}: expected one dependency RUN block, found {len(hits)}"
    return hits[0]


class TestTheImagePinsNumpy:
    def test_the_requirements_file_still_pins_numpy_below_2(self):
        req = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
        pins = [ln.strip() for ln in req.splitlines() if re.match(r"(?i)^numpy", ln.strip())]
        assert len(pins) == 1, pins
        assert "<2" in pins[0], (
            "the numpy pin moved. The Dockerfile constraint is READ from this "
            "line, so this is a deliberate environment change: move CI, the "
            "mypy baseline environment and this test together (P187/P421b).")

    @pytest.mark.parametrize("name", DOCKERFILES)
    def test_the_constraint_is_exported_before_any_pip_install(self, name):
        blk = _run_block(name)
        assert "PIP_CONSTRAINT=/tmp/pip-constraints.txt" in blk
        assert blk.index("export PIP_CONSTRAINT") < blk.index("pip install"), (
            "a pip line before the export re-resolves numpy unconstrained")
        # every pip install shares the RUN (and so the exported variable)
        whole = "\n".join(ln for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()
                          if not ln.lstrip().startswith("#"))
        outside = whole.replace(blk, "")
        stray = [ln for ln in outside.splitlines()
                 if "pip install" in ln and re.search(r"torch|stable-baselines3|sb3|scikit|transformers|ta\b", ln)]
        assert not stray, f"{name}: numeric-stack pip install outside the constrained RUN: {stray}"

    @pytest.mark.parametrize("name", DOCKERFILES)
    def test_the_constraint_comes_from_the_requirements_file_and_cannot_be_empty(self, name):
        blk = _run_block(name)
        assert "grep -iE '^numpy' requirements-runtime.txt > /tmp/pip-constraints.txt" in blk
        assert "test -s /tmp/pip-constraints.txt" in blk, (
            "an empty constraints file constrains nothing and builds green (P159)")

    @pytest.mark.parametrize("name", DOCKERFILES)
    def test_a_violated_pin_fails_the_build(self, name):
        blk = _run_block(name)
        last = blk.strip().splitlines()[-1]
        assert "python -c" in last and "numpy pin violated" in last, (
            "the version assert must be the LAST step, after every install")
        assert "assert int(v.split('.')[0]) < 2" in last


class TestActionToScalar:
    def test_the_sb3_batch_shape_converts_without_any_numpy_warning(self):
        # (1, 1) is what predict() returns for a (1, obs) batch. Under numpy
        # 1.25-2.3 the old float(action[0]) only WARNED; errors-as-failures
        # makes this red on the old code under every numpy.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert action_to_scalar(np.array([[0.25]], dtype=np.float32)) == pytest.approx(0.25)

    @pytest.mark.parametrize("raw,want", [
        (np.array([[3.0]]), 1.0), (np.array([-9.0]), -1.0),
        (np.float32(0.5), 0.5), (0.1, 0.1), ([0.2, 0.9], 0.2),
    ])
    def test_shapes_and_clipping(self, raw, want):
        out = action_to_scalar(raw)
        assert type(out) is float and out == pytest.approx(want)

    def test_an_empty_action_is_a_refusal_not_a_zero(self):
        with pytest.raises(ValueError):
            action_to_scalar(np.array([]))

    def test_the_predict_site_uses_the_helper(self):
        src = (ROOT / "drl" / "ensemble.py").read_text(encoding="utf-8")
        assert "tqc_action = action_to_scalar(action)" in src
        assert "float(np.clip(action[0]" not in src
