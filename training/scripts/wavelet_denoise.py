#!/usr/bin/env python3
"""
Wavelet denoising for DRL feature pipeline.

Coiflet-4 level-2 soft thresholding (Wavelet-DRL, Neural Computing 2025).
Empirically +18.7% Sharpe improvement on high-noise technical indicators.

Usage:
    from scripts.wavelet_denoise import wavelet_denoise, DENOISE_COLUMNS

    # Batch (rebuild_pipeline): denoise entire column
    df[f'{col}_denoised'] = wavelet_denoise(df[col].values)

    # Runtime (main.py): denoise rolling window, take last value
    denoised = wavelet_denoise(recent_256_values)[-1]
"""

import numpy as np
import pywt


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
