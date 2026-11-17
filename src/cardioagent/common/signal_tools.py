from fractions import Fraction

import numpy as np
from scipy import signal


def resample(values: np.ndarray, fs: float, target: int) -> np.ndarray:
    ratio = Fraction(target / fs).limit_denominator(10000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator)


def bandpass(values: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = signal.butter(3, [low, min(high, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)


def measure_rate(modality: str, values: np.ndarray, fs: float, start: float) -> dict:
    """Estimate periodic rate and return observations for the selected window."""
    if len(values) / fs < 5:
        return {"status": "insufficient", "reason": "At least 5 seconds are required"}
    if not np.isfinite(values).all() or np.ptp(values) <= 1e-10:
        return {"status": "insufficient", "reason": "Waveform is flat or has non-finite samples"}
    if modality == "ecg":
        filtered = bandpass(values, fs, 5, 25)
        # Select the dominant QRS polarity from robust extreme amplitudes.
        low, high = np.percentile(filtered, [1, 99])
        pulse = filtered if high >= abs(low) else -filtered
        method = "bandpass_qrs_peak_detection"
    elif modality == "ppg":
        pulse = bandpass(values, fs, 0.45, 5)
        method = "bandpass_pulse_peak_detection"
    else:
        return measure_pcg_period(values, fs)

    spread = float(np.percentile(pulse, 99) - np.percentile(pulse, 10))
    peaks, _ = signal.find_peaks(pulse, distance=max(1, round(0.27 * fs)), prominence=0.27 * spread)
    intervals = np.diff(peaks) / fs
    valid = intervals[(intervals >= 0.27) & (intervals <= 2.0)]
    if len(valid) < 3:
        return {"status": "insufficient", "method": method, "reason": "Fewer than 3 plausible beat intervals"}
    median_interval = float(np.median(valid))
    variation = float(np.median(np.abs(valid - median_interval)) / median_interval)
    stable_fraction = float(np.mean(np.abs(intervals - median_interval) < 0.20 * median_interval))
    return {
        "status": "ok",
        "method": method,
        "rate_bpm": round(60 / median_interval, 2),
        "quality": "usable" if variation <= 0.12 and stable_fraction >= 0.7 else "review",
        "interval_mad_ratio": round(variation, 4),
        "stable_interval_fraction": round(stable_fraction, 4),
        "peak_times_s": [round(start + peak / fs, 4) for peak in peaks],
    }


def measure_pcg_period(values: np.ndarray, fs: float) -> dict:
    method = "pcg_envelope_autocorrelation"
    filtered = bandpass(values, fs, 25, 180)
    envelope = resample(np.abs(signal.hilbert(filtered)), fs, 100)
    sos = signal.butter(2, 8, fs=100, output="sos")
    envelope = signal.sosfiltfilt(sos, envelope)
    centered = envelope - np.mean(envelope)
    autocorr = signal.correlate(centered, centered, mode="full", method="fft")[len(centered) - 1:]
    if autocorr[0] <= np.finfo(float).tiny:
        return {"status": "insufficient", "method": method, "reason": "No measurable envelope variation"}
    autocorr /= autocorr[0]
    peaks, _ = signal.find_peaks(autocorr[:202])
    candidates = [int(lag) for lag in peaks if 27 <= lag <= 200 and autocorr[lag] >= 0.18]
    if not candidates:
        return {"status": "insufficient", "method": method, "reason": "No reliable envelope period"}
    lag = max(candidates, key=lambda k: autocorr[k])
    strength = float(autocorr[lag])
    half_period_ambiguous = any(
        abs(other / lag - 0.5) <= 0.06 and autocorr[other] >= 0.85 * strength for other in candidates
    )
    return {
        "status": "ok",
        "method": method,
        "rate_bpm": round(6000 / lag, 2),
        "quality": "usable" if strength >= 0.38 and not half_period_ambiguous else "review",
        "autocorrelation": round(strength, 4),
        "candidate_periods_s": [round(candidate / 100, 2) for candidate in candidates],
    }
