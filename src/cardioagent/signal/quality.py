"""Describe observable signal quality separately from detector and model reliability."""

import numpy as np
from scipy import signal

from cardioagent.common.signal_tools import bandpass


def _ratio(a: float, b: float) -> float | None:
    return round(float(a / b), 6) if b > 0 else None


def quality_features(values: np.ndarray, fs: float, modality: str) -> dict:
    finite = np.isfinite(values)
    out = {
        "nonfinite_fraction": round(float(np.mean(~finite)), 6) if len(values) else None,
        "near_constant": None,
        "extreme_plateau_fraction": None,
        "baseline_power_fraction": None,
        "physiology_band_power_fraction": None,
        "high_frequency_power_fraction": None,
        "longest_flat_run_s": None,
        "impulsive_step_fraction": None,
        "ppg_ac_dc_ratio": None,  # Released PPG is centered; sensor DC is unknown.
    }
    if not len(values) or not finite.all():
        return out
    out["near_constant"] = bool(np.ptp(values) <= 1e-10)
    difference = np.diff(values)
    edges = np.flatnonzero(np.diff(np.r_[False, difference == 0, False]))
    out["longest_flat_run_s"] = round(float((np.max(edges[1::2] - edges[::2]) + 1) / fs), 6) if len(edges) else 0.0
    if out["near_constant"]:
        return out
    at_edge = (values == values.min()) | (values == values.max())
    out["extreme_plateau_fraction"] = round(float(np.mean(at_edge[1:] & (difference == 0))), 6)
    step_scale = np.percentile(np.abs(difference), 90)
    if step_scale > 0:
        out["impulsive_step_fraction"] = round(float(np.mean(np.abs(difference) > 8 * step_scale)), 6)
    freq, power = signal.welch(values, fs, nperseg=min(len(values), round(4 * fs)), detrend="constant")
    low, high = {"ecg": (5, 40), "ppg": (0.45, 5), "pcg": (25, 180)}[modality]
    total = power.sum()
    out.update(
        baseline_power_fraction=_ratio(power[freq < 0.5].sum(), total),
        physiology_band_power_fraction=_ratio(power[(freq >= low) & (freq <= high)].sum(), total),
        high_frequency_power_fraction=_ratio(power[freq > high].sum(), total),
    )
    if modality == "ppg":
        pulse = bandpass(values, fs, low, high)
        out["ppg_pulse_band_amplitude_p95_p05"] = round(float(np.diff(np.percentile(pulse, [5, 95]))[0]), 6)
    elif modality == "pcg":
        envelope = np.abs(signal.hilbert(bandpass(values, fs, low, high)))
        p50, p95 = np.percentile(envelope, [50, 95])
        out["pcg_envelope_p95_to_median"] = _ratio(p95, p50)
    return out
