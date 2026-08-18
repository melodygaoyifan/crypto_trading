"""[P164] Training features must be computable from the past alone.

`wavelet_denoise` computes its VisuShrink threshold from the whole input array
and reconstructs every sample from every coefficient. Applied to a full history
— which is what `rebuild_pipeline.py` did — each training row becomes a
function of all future rows. Live applies the same function to a trailing
256-bar buffer and takes the last value. Those are different transforms, and
the difference is not subtle:

    on pure random walks (zero predictability by construction)
        batch-denoise delta   -> IC +0.41 vs next-bar return  (Sharpe ~+16)
        rolling-denoise delta -> IC +0.002

The DRL's reported per-fold validation Sharpe (+7 to +17) sits inside the range
the leak produces on noise, against a live IC of +0.052. No date-based split
removes it: the contamination is in every row, which is why CSCV-PBO reported
"ROBUST_SELECTION" while the live account lost money.

`scripts/runtime_parity_check.py` was supposed to cover this and only asserted
that the five denoised column *names* exist in the manifest. These tests assert
the *values*, and they assert causality directly — by mutating the future and
requiring the past not to move.
"""

import numpy as np
import pytest

pywt = pytest.importorskip("pywt")

from training.scripts.wavelet_denoise import (  # noqa: E402
    DENOISE_COLUMNS,
    RUNTIME_MIN_SAMPLES,
    RUNTIME_WINDOW,
    wavelet_denoise,
    wavelet_denoise_causal,
)


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n)) + 50.0


# --- the defect, pinned so it cannot be reintroduced ------------------------
def test_batch_denoise_is_not_causal():
    """Characterisation: prove the original transform leaks.

    If this ever fails, `wavelet_denoise` became causal and this whole module
    can be simplified — but until then it documents why the causal form exists.
    """
    values = _series()
    baseline = wavelet_denoise(values)

    mutated = values.copy()
    mutated[-1] += 1_000.0          # change ONLY the final (future) sample
    after = wavelet_denoise(mutated)

    early = len(values) // 4
    assert not np.allclose(baseline[:early], after[:early]), (
        "expected the batch transform to leak future information into the past"
    )


def test_causal_denoise_is_unaffected_by_the_future():
    """The property that matters: the past must not move when the future does."""
    values = _series()
    baseline = wavelet_denoise_causal(values)

    for future_shock in (1_000.0, -1_000.0):
        mutated = values.copy()
        mutated[-1] += future_shock
        after = wavelet_denoise_causal(mutated)
        assert np.allclose(baseline[:-1], after[:-1], atol=1e-12), (
            "a future sample changed a past denoised value — the feature leaks"
        )


def test_causal_denoise_prefix_stability():
    """Streaming equivalence: growing the series never rewrites history."""
    values = _series(n=300, seed=7)
    full = wavelet_denoise_causal(values)

    for cut in (50, 137, 299):
        prefix = wavelet_denoise_causal(values[:cut])
        assert np.allclose(prefix, full[:cut], atol=1e-12), (
            f"denoised prefix of length {cut} does not match the full run — "
            f"training rows would not equal what live computes at that bar"
        )


# --- train/serve equivalence, the actual contract ---------------------------
def test_causal_matches_the_live_recurrence_exactly():
    """Replays data_mgmt/market_data_pipeline.py:853-866 against the batch build.

    This is the assertion that would have caught the bug: build the column the
    way training does, feed the same series through the live buffer one bar at
    a time, and require the two to be identical.
    """
    from collections import deque

    values = _series(n=600, seed=3)
    trained = wavelet_denoise_causal(values)

    buf = deque(maxlen=RUNTIME_WINDOW)
    served = []
    for v in values:
        buf.append(float(v))
        if len(buf) >= RUNTIME_MIN_SAMPLES:
            served.append(float(wavelet_denoise(np.array(buf))[-1]))
        else:
            served.append(float(buf[-1]))

    assert np.allclose(trained, np.array(served), atol=1e-12), (
        "training-time and runtime denoising disagree — the model would be "
        "served features it was never trained on"
    )


def test_window_longer_than_runtime_buffer_is_not_used():
    """Beyond 256 bars the live buffer forgets; training must forget too."""
    values = _series(n=RUNTIME_WINDOW + 200, seed=11)
    baseline = wavelet_denoise_causal(values)

    mutated = values.copy()
    mutated[0] += 5_000.0           # ancient history, outside the final window
    after = wavelet_denoise_causal(mutated)

    assert np.allclose(baseline[-1], after[-1], atol=1e-12), (
        "a sample older than RUNTIME_WINDOW changed the latest value — "
        "training uses more history than the runtime deque retains"
    )


def test_short_series_passes_through_like_the_runtime():
    """Below min_samples the runtime emits the raw value; training must match."""
    values = _series(n=RUNTIME_MIN_SAMPLES - 1, seed=5)
    out = wavelet_denoise_causal(values)
    assert np.allclose(out, values, atol=1e-12)


def test_output_shape_and_finiteness():
    values = _series(n=512, seed=13)
    out = wavelet_denoise_causal(values)
    assert out.shape == values.shape
    assert np.all(np.isfinite(out))


# --- the pipeline actually uses the causal form -----------------------------
def test_rebuild_pipeline_does_not_import_the_leaky_form():
    """Guard the call site, not just the function."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "training" / "scripts" / "rebuild_pipeline.py"
    # Strip comments: the fix documents the old call by name, and a naive
    # substring check would match that prose instead of live code.
    code = "\n".join(
        line.split("#", 1)[0] for line in src.read_text(encoding="utf-8").splitlines()
    )

    assert "wavelet_denoise_causal(raw_vals)" in code, (
        "rebuild_pipeline must build denoised features causally"
    )
    # the bare form must not be applied to a whole column anywhere
    assert "wavelet_denoise(raw_vals)" not in code, (
        "whole-column wavelet_denoise reintroduces the lookahead leak"
    )


def test_denoise_columns_still_match_the_runtime_map():
    """The five columns are duplicated in market_data_pipeline; keep them equal."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "data_mgmt" / "market_data_pipeline.py"
    text = src.read_text(encoding="utf-8")
    for src_col, dst_col in DENOISE_COLUMNS.items():
        assert f'"{src_col}": "{dst_col}"' in text, (
            f"runtime denoise map is missing {src_col} -> {dst_col}; training "
            f"and serving feature sets have drifted"
        )
