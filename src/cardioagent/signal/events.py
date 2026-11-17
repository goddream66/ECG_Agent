"""Detect candidate events; heuristic scores are not calibrated probabilities."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import signal
from wfdb.processing import xqrs_detect

from cardioagent.common.signal_tools import bandpass, measure_pcg_period
from cardioagent.signal.synchronization import TimeAxis


@dataclass(frozen=True)
class Event:
    event_type: str
    sample_index: int
    time_s: float
    confidence: float | None
    source: str


def ppg_event_samples(values: np.ndarray, fs: float) -> list[tuple[str, int, float | None, str]]:
    filtered = bandpass(values, fs, 0.45, 5)
    onset_wave = bandpass(values, fs, 0.45, 12)
    slope = np.gradient(onset_wave)
    spread = np.percentile(filtered, 95) - np.percentile(filtered, 5)
    peaks, props = signal.find_peaks(filtered, distance=max(1, round(0.27 * fs)), prominence=0.25 * spread)
    events = []
    for i, (peak, prominence) in enumerate(zip(peaks, props["prominences"])):
        score = round(1 - math.exp(-prominence / max(spread, 1e-12)), 4)
        events.append(("systolic_peak_candidate", int(peak), score, "filtered_ppg_peak"))
        # Preceding trough is an onset candidate. Ignore minima at truncated search boundaries.
        lo = max(0, peak - round(0.45 * fs), peaks[i - 1] + 1 if i else 0)
        hi = peak - round(0.025 * fs)
        if hi - lo < 3:
            continue
        foot = lo + int(np.argmin(filtered[lo:hi]))
        if not lo < foot < hi - 1 or filtered[peak] <= filtered[foot]:
            continue
        tangent = foot + int(np.argmax(slope[foot:peak]))
        upstroke_fraction = float(np.mean(slope[foot:tangent + 1] > 0))
        foot_score = round(score * upstroke_fraction, 4)
        events.append(("pulse_foot_candidate", int(foot), foot_score, "preceding_ppg_trough"))
        if slope[tangent] <= 0 or tangent == foot:
            continue
        # Exploratory refinement between the trough and a relative upstroke landmark.
        crossing = int(np.flatnonzero(slope[foot:tangent + 1] >= 0.10 * slope[tangent])[0])
        events.append(("pulse_foot_refined_candidate", round(foot + crossing / 2), foot_score,
                       "trough_to_relative_upslope_midpoint"))
    return events


def _series(events: list[Event], event_type: str) -> dict:
    times = np.array([e.time_s for e in events if e.event_type == event_type])
    intervals = np.diff(times)
    valid = intervals[(intervals >= 0.27) & (intervals <= 2.0)]
    if len(valid) < 3:
        return {"rate_bpm": None, "plausible_interval_fraction": None, "interval_mad_ratio": None}
    median = np.median(valid)
    return {
        "rate_bpm": round(float(60 / median), 2),
        "plausible_interval_fraction": round(float(len(valid) / len(intervals)), 4),
        "interval_mad_ratio": round(float(np.median(np.abs(valid - median)) / median), 4),
    }


def detect_events(modality: str, values: np.ndarray, axis: TimeAxis, first: int) -> dict:
    fs = axis.sample_rate_hz
    result = {
        "status": "insufficient", "events": [], "rate_bpm": None,
        "event_quality": {}, "confidence_semantics": "heuristic_amplitude_score_not_probability",
    }
    if modality == "pcg":
        result["unavailable_events"] = {"S1": "no validated detector/labels in this release",
                                        "S2": "no validated detector/labels in this release"}
    if len(values) / fs < 5 or not np.isfinite(values).all() or np.ptp(values) <= 1e-10:
        return result | {"reason": "Requires at least 5 seconds of finite, non-flat waveform"}
    if modality == "pcg":
        period = measure_pcg_period(values, fs)
        return result | {
            "status": "period_only" if period["status"] == "ok" else "insufficient",
            "rate_bpm": period.get("rate_bpm"), "period_measurement": period,
            "event_quality": {"autocorrelation": period.get("autocorrelation")},
        }

    def event(name: str, index: int, score: float | None, source: str) -> Event:
        index = int(index) + first
        return Event(name, index, round(axis.time_at(index), 8), score, source)

    events = []
    if modality == "ecg":
        main = "r_peak_candidate"
        filtered = bandpass(values, fs, 5, 25)
        low, high = np.percentile(filtered, [1, 99])
        spread = float(high - low)
        if spread <= 1e-10:
            return result | {"reason": "No measurable QRS-band amplitude"}
        # Adapt XQRS's amplitude-based initial threshold to this sensor's signal scale.
        normalized = 2 * (values - np.median(values)) / spread
        peaks = xqrs_detect(normalized, fs, verbose=False)
        scale = np.percentile(np.abs(filtered), 95) + 1e-12
        radius = round(0.04 * fs)
        for center in peaks:
            lo, hi = max(0, center - radius), min(len(values), center + radius + 1)
            peak = lo + int(np.argmax(np.abs(filtered[lo:hi])))
            score = 1 - math.exp(-abs(filtered[peak]) / scale)
            events.append(event(main, peak, round(score, 4), "wfdb_xqrs_robust_scale_local_qrs_extremum"))
    elif modality == "ppg":
        main = "systolic_peak_candidate"
        events.extend(event(*sample) for sample in ppg_event_samples(values, fs))
        result["confidence_semantics"] = "heuristic amplitude and positive upslope scores; not probabilities"
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    events.sort(key=lambda e: e.sample_index)
    stats = _series(events, main)
    return result | {
        "status": "candidates" if events else "insufficient",
        "events": [asdict(e) for e in events],
        "rate_bpm": stats["rate_bpm"],
        "event_quality": {k: v for k, v in stats.items() if k != "rate_bpm"},
    }
