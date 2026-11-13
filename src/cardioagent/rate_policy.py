"""Shared rate measurements and the ECG-first rule policy for the M1.5 audit."""

from collections.abc import Callable

import numpy as np

from .acquisition import run_acquisition
from .recording import Recording
from .signal.events import detect_events
from .signal.quality import quality_features
from .signal.synchronization import TimeAxis


def measure_for_audit(modality: str, values: np.ndarray, fs: float, start: float) -> dict:
    observation = detect_events(modality, values, TimeAxis(fs, len(values), start_s=start), 0)
    rate = observation["rate_bpm"]
    quality = quality_features(values, fs, modality)
    events = observation["event_quality"]
    main_type = {"ecg": "r_peak_candidate", "ppg": "systolic_peak_candidate"}.get(modality)
    scores = [item["confidence"] for item in observation["events"]
              if item["event_type"] == main_type and item["confidence"] is not None]
    layers = {"signal_quality": quality, "event_quality": events, "model_quality": None,
              "event_score_median": round(float(np.median(scores)), 4) if scores else None,
              "pcg_candidate_periods_s": observation.get("period_measurement", {}).get("candidate_periods_s", []),
              "quality_semantics": "heuristic gate; no calibrated probability"}
    if rate is None:
        return {"status": "insufficient", "rate_bpm": None, "quality": "review",
                "quality_reasons": [observation.get("reason", "not enough intervals")], **layers}
    if modality == "pcg":
        reasons = ["weak or ambiguous period"] if observation["period_measurement"]["quality"] != "usable" else []
    else:
        count_type = "r_peak_candidate" if modality == "ecg" else "systolic_peak_candidate"
        count = sum(event["event_type"] == count_type for event in observation["events"])
        expected_count = rate * len(values) / (60 * fs)
        reasons = []
        if events["plausible_interval_fraction"] < 0.85 or events["interval_mad_ratio"] > 0.12:
            reasons.append("irregular detected intervals")
        if not 0.85 <= count / expected_count <= 1.15:
            reasons.append("event count inconsistent with estimated rate")
    if modality == "ecg":
        if quality["baseline_power_fraction"] > 0.70:
            reasons.append("baseline drift dominates")
        if quality["high_frequency_power_fraction"] > 0.35:
            reasons.append("high-frequency power")
    elif modality == "ppg":
        if quality["physiology_band_power_fraction"] < 0.15 or quality["high_frequency_power_fraction"] > 0.45:
            reasons.append("weak or noisy pulsatile band")
    elif quality["physiology_band_power_fraction"] < 0.10 or quality["high_frequency_power_fraction"] > 0.60:
        reasons.append("weak or noisy cardiac acoustic band")
    if quality["longest_flat_run_s"] > 0.5:
        reasons.append("sustained flat samples")
    return {
        "status": "ok", "rate_bpm": rate, "quality": "review" if reasons else "usable",
        "quality_reasons": reasons, **layers,
    }


def observe_rate(record: Recording, modality: str, channel: str) -> dict:
    wave = record.waves[modality]
    end = record.shared_duration
    return {
        "modality": modality, "channel": channel, "window_s": [0, end],
        "evidence_id": f"{modality}/{channel}",
        **measure_for_audit(modality, wave.window(channel, 0, end), wave.fs, 0),
    }


def ecg_first(
    record: Recording, budget: int,
    observe: Callable[[str, str], dict],
    *, strategy: str = "ppg_fallback",
) -> dict:
    return run_acquisition(record, budget, observe, strategy)
