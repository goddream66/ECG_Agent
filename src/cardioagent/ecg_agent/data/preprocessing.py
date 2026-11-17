"""Window preprocessing for the AFDB experiment, based on the NeuroECG recipe."""

from math import ceil

import numpy as np
from scipy import signal

from cardioagent.common.signal_tools import resample


PROFILE = "neuroecg-afdb-window-v1"
WINDOW_SECONDS = 10
TARGET_FS = 500
BANDPASS_HZ = (0.5, 40.0)
FILTER_ORDER = 3
SCALE_FLOOR = 1e-6
CLIP_VALUE = 5.0
# Detect suspected dropout even when a 10.24 s AFDB gap is split at a window edge.
FLAT_SECONDS = 1.0


def invalid_spans(values: np.ndarray, fs: int) -> list[tuple[int, int, str]]:
    """Return invalid or suspected dropout spans in window-local sample indices."""
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or fs <= 0:
        raise ValueError("Expected a nonempty single-channel window and positive sampling rate")
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, ~finite, False]))
    spans = [(int(start), int(end), "nonfinite") for start, end in edges.reshape(-1, 2)]

    # A constant run is a dropout candidate even when every sample is finite.
    boundaries = np.r_[0, 1 + np.flatnonzero(values[1:] != values[:-1]), len(values)]
    long_runs = np.flatnonzero(np.diff(boundaries) >= ceil(FLAT_SECONDS * fs))
    for index in long_runs:
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        if finite[start]:
            spans.append((start, end, "suspected_flatline"))
    return sorted(spans)


def preprocess_window(values: np.ndarray, fs: int) -> np.ndarray:
    """Transform one complete, quality-checked, single-channel 10-second window."""
    values = np.asarray(values, dtype=np.float32)
    if fs <= 2 * BANDPASS_HZ[1] or values.shape != (WINDOW_SECONDS * fs,):
        raise ValueError("Expected 10 seconds of one channel at a sampling rate above 80 Hz")
    if not np.isfinite(values).all():
        raise ValueError("Model input contains non-finite samples")

    sos = signal.butter(FILTER_ORDER, BANDPASS_HZ, btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, values)
    samples = resample(filtered, fs, TARGET_FS).astype(np.float32)
    centered = samples - np.median(samples)
    scale = max(float(np.percentile(np.abs(centered), 95)), SCALE_FLOOR)
    result = np.clip(centered / scale, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)
    if result.shape != (TARGET_FS * WINDOW_SECONDS,) or not np.isfinite(result).all():
        raise ValueError("Preprocessing did not produce a finite 5,000-point window")
    return result
