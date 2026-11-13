"""Numeric evidence available to a policy before it selects an action."""

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

import numpy as np
from scipy import signal

from ..signal_tools import bandpass, resample
from ..workflow import preferred_channel, rate_summary


TASKS = {"heart_rate": "ecg", "peripheral_pulse_rate": "ppg"}
SLOTS = ("primary_ecg", "alternate_ecg_1", "alternate_ecg_2", "ppg", "pcg")


def actions_for(record, task: str) -> tuple[dict, str | None]:
    anchor = TASKS[task]
    options = {}
    initial = None
    if anchor == "ecg" and "ecg" in record.waves:
        leads = record.waves["ecg"].channels
        primary = preferred_channel("ecg", leads)
        initial = "ACQUIRE_ECG"
        options[initial] = ("ecg", primary, "primary_ecg")
        choices = [lead for lead in dict.fromkeys(("II", "V3", *leads)) if lead in leads and lead != primary]
        for i, lead in enumerate(choices[:2], 1):
            options[f"VERIFY_ECG:{lead}"] = ("ecg", lead, f"alternate_ecg_{i}")
    for modality in ("ppg", "pcg"):
        if modality in record.waves and (task == "heart_rate" or modality == anchor):
            action = f"ACQUIRE_{modality.upper()}"
            options[action] = (modality, preferred_channel(modality, record.waves[modality].channels), modality)
            if modality == anchor:
                initial = action
    return options, initial


def preview_channel(values: np.ndarray, fs: float, modality: str) -> dict:
    """A rate-free preview. It measures periodic strength without returning a period or rate."""
    began = perf_counter()
    finite = bool(np.isfinite(values).all())
    varying = finite and len(values) >= round(5 * fs) and float(np.ptp(values)) > 1e-10
    features = {"valid": float(varying), "band_fraction": 0., "periodicity": 0., "flat_fraction": 0., "score": 0.}
    if varying:
        low, high = {"ecg": (5, 25), "ppg": (0.45, 5), "pcg": (25, 180)}[modality]
        waveform = bandpass(values, fs, low, high)
        band_fraction = float(np.clip(np.var(waveform) / max(float(np.var(values)), 1e-12), 0, 1))
        periodic_wave = resample(waveform if modality == "ppg" else np.abs(waveform), fs, 40)
        periodic_wave -= periodic_wave.mean()
        autocorr = signal.correlate(periodic_wave, periodic_wave, mode="full", method="fft")[len(periodic_wave)-1:]
        periodicity = float(np.clip(np.max(autocorr[12:81]) / autocorr[0], 0, 1)) if autocorr[0] > 0 else 0.
        flat = np.diff(values) == 0
        edges = np.flatnonzero(np.diff(np.r_[False, flat, False]))
        flat_fraction = float((edges[1::2] - edges[::2]).max() / len(values)) if len(edges) else 0.
        score = (0.8 * periodicity + 0.2 * band_fraction) * max(0., 1 - 4 * flat_fraction)
        features.update(band_fraction=band_fraction, periodicity=periodicity, flat_fraction=flat_fraction, score=score)
    return {k: round(v, 6) for k, v in features.items()} | {"duration_ms": 1000 * (perf_counter() - began)}


def disagreement(evidence: list[dict]) -> float:
    rates = [item["rate_bpm"] for item in evidence if item.get("rate_bpm") is not None]
    return max(rates) - min(rates) if len(rates) > 1 else 0.


@dataclass
class EvidenceState:
    task: str
    duration: float
    options: dict
    evidence: dict
    previews: dict
    remaining: int

    def summary(self) -> dict:
        return rate_summary(list(self.evidence.values()), 0, self.duration)

    def legal(self) -> list[str]:
        result = [a for a in self.options if a not in self.evidence] if self.remaining else []
        if self.summary()["rate_bpm"] is not None:
            result.append("STOP")
        return result + ["ABSTAIN"]

    def numeric(self, summary: dict) -> np.ndarray:
        values = [float(self.task == "peripheral_pulse_rate"), self.remaining / 6,
                  (summary["rate_bpm"] or 0) / 200, min(2., disagreement(list(self.evidence.values())) / 100),
                  float(summary["rate_bpm"] is not None), len(self.evidence) / 5]
        options = {slot: action for action, (_, _, slot) in self.options.items()}
        for slot in SLOTS:
            action = options.get(slot)
            item = self.evidence.get(action, {})
            qc = self.previews.get(action, {})
            event = item.get("event_quality") or {}
            values.extend([float(action is not None), float(bool(item)), float(item.get("quality") == "usable"),
                           (item.get("rate_bpm") or 0) / 200, qc.get("valid", 0), qc.get("score", 0),
                           qc.get("band_fraction", 0), qc.get("periodicity", 0), qc.get("flat_fraction", 0),
                           event.get("interval_mad_ratio") or 0, event.get("plausible_interval_fraction") or 0,
                           event.get("autocorrelation") or 0])
        return np.asarray(values, dtype=np.float32)

    def quality_rank(self, summary: dict) -> float:
        selected = set(summary["evidence_ids"])
        values = []
        for item in self.evidence.values():
            if item["evidence_id"] not in selected:
                continue
            band = item["signal_quality"].get("physiology_band_power_fraction") or 0.
            event = item["event_quality"]
            if item["modality"] == "pcg":
                stability = float(np.clip(event.get("autocorrelation") or 0, 0, 1))
            else:
                stability = (event.get("plausible_interval_fraction") or 0) * (
                    1 - min(1, (event.get("interval_mad_ratio") or 0) / 0.12))
            values.append(0.5 * (band + stability))
        return mean(values) if values else 0.
