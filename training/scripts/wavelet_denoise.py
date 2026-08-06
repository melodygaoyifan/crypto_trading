#!/usr/bin/env python3
"""
Wavelet denoising for DRL feature pipeline.

Coiflet-4 level-2 soft thresholding (Wavelet-DRL, Neural Computing 2025).
Empirically +18.7% Sharpe improvement on high-noise technical indicators.

[P164] `wavelet_denoise` IS NOT CAUSAL. Do not call it on a full history to
build training features. The VisuShrink threshold (below) is computed from the
whole input array and `waverec` then reconstructs every sample from every
coefficient, so each output sample is a function of ALL input samples —
including future ones.

This module previously documented the batch and rolling calls side by side as
if they were the same transform. They are not, and the gap was the single
largest defect in the system:

    training  wavelet_denoise(entire 8.5y column)   -> each row sees the future
    live      wavelet_denoise(trailing 256)[-1]     -> each row sees only history

Measured on 20 independent pure random walks, where predictability is zero by
construction, the batch path's denoise-delta scored IC +0.41 against next-bar
returns (annualized Sharpe ~+16) while the rolling path scored +0.002. The DRL's
reported per-fold validation Sharpe of +7 to +17 sits inside the range this leak
produces on noise alone, against a live IC of +0.052. No date-based split can
remove it: the contamination is in every row of the parquet, which is why
CSCV-PBO returned "ROBUST_SELECTION" while live lost money.

Usage:
    from training.scripts.wavelet_denoise import (
        wavelet_denoise_causal, DENOISE_COLUMNS,
    )

    # Batch feature build (rebuild_pipeline) — MUST be the causal form:
    df[f'{col}_denoised'] = wavelet_denoise_causal(df[col].values)

    # Runtime (market_data_pipeline) — one step of the same recurrence:
    denoised = wavelet_denoise(recent_256_values)[-1]

`wavelet_denoise` remains exported because the runtime calls it on a trailing
window, which is legitimate. It is only the whole-history application that leaks.
"""

import numpy as np
import pywt

# Runtime contract, mirrored from data_mgmt/market_data_pipeline.py.
# `_wavelet_buffers` is `deque(maxlen=256)` and denoises once `len(buf) >= 8`,
# otherwise it passes the raw value through. `wavelet_denoise_causal` must
# reproduce that exactly — the two are pinned together by
# tests/test_wavelet_causality.py, which fails if either side moves.
RUNTIME_WINDOW = 256
RUNTIME_MIN_SAMPLES = 8


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "coif4",
    level: int = 2,
) -> np.ndarray:
    """Coiflet-4 level-2 soft thresholding denoising.

    Args:
        signal: 1D array of feature values.
        wavelet: Wavelet family (default: coif4).
        level: Decomposition level (default: 2).

    Returns:
        Denoised signal, same length as input.
        Returns original signal if too short for decomposition.
    """
    min_len = 2 ** (level + 1)
    if len(signal) < min_len:
        return signal.copy()

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Soft threshold detail coefficients (keep approximation intact)
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]

    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[: len(signal)]


def wavelet_denoise_causal(
    signal: np.ndarray,
    wavelet: str = "coif4",
    level: int = 2,
    window: int = RUNTIME_WINDOW,
    min_samples: int = RUNTIME_MIN_SAMPLES,
) -> np.ndarray:
    """Causal denoise: output[i] depends only on signal[:i+1].

    [P164] This is the ONLY correct way to build the `*_denoised` training
    features. It replays the live recurrence exactly — for each row, denoise the
    trailing `window` samples and keep the last value — so a model trained on
    this sees precisely what it will see in production.

    The naive `wavelet_denoise(whole_column)` is ~N times cheaper and looks
    equivalent, which is why it was used. It is not equivalent: it leaks the
    entire future into every row.

    Args:
        signal: 1D array, chronologically ordered, oldest first.
        wavelet: Wavelet family (must match runtime).
        level: Decomposition level (must match runtime).
        window: Trailing window, mirrors the runtime deque maxlen.
        min_samples: Below this the runtime passes the raw value through.

    Returns:
        Denoised signal, same length and dtype-compatible with the input.
    """
    values = np.asarray(signal, dtype=float)
    n = len(values)
    out = np.empty(n, dtype=float)

    for i in range(n):
        start = max(0, i + 1 - window)
        buf = values[start : i + 1]
        if len(buf) >= min_samples:
            out[i] = wavelet_denoise(buf, wavelet=wavelet, level=level)[-1]
        else:
            # Runtime passes the raw value through until the buffer fills.
            out[i] = buf[-1]

    return out


# Target columns for denoising (mapped to actual parquet column names).
# These 5 are the highest-noise technical indicators validated by Wavelet-DRL paper.
DENOISE_COLUMNS = {
    "rsi_14": "rsi_14_denoised",
    "macd_12_26": "macd_12_26_denoised",
    "bb_width_20": "bb_width_20_denoised",
    "atr_14": "atr_14_denoised",
    "vol_ratio_s": "vol_ratio_s_denoised",
}

# Ordered list of denoised output column names (for manifest)
DENOISED_FEATURE_NAMES = list(DENOISE_COLUMNS.values())
