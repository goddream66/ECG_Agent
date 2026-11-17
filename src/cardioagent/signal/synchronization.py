"""Map sample indices to a recording time axis without shifting physiological delays."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from cardioagent.common.recording import Recording, Waveform, sample_span


@dataclass(frozen=True)
class TimeAxis:
    sample_rate_hz: float
    sample_count: int
    start_s: float = 0.0
    offset_correction_s: float = 0.0

    @property
    def origin_s(self) -> float:
        return self.start_s + self.offset_correction_s

    @property
    def end_s(self) -> float:
        return self.origin_s + self.sample_count / self.sample_rate_hz

    def time_at(self, index: int) -> float:
        return self.origin_s + index / self.sample_rate_hz

    def select(self, wave: Waveform, channel: str, start: float, end: float) -> tuple[np.ndarray, int]:
        if channel not in wave.channels:
            raise ValueError(f"Unknown channel {channel!r}; available: {wave.channels}")
        first, last = sample_span(self.sample_rate_hz, self.sample_count, self.origin_s, start, end)
        return wave.values[first:last, wave.channels.index(channel)], first


def inspect_axes(record: Recording, documented_common_grid: bool,
                 calibration: dict | None = None) -> tuple[dict[str, TimeAxis], dict]:
    latency = None
    uncertainty = None
    if calibration is not None:
        reference = calibration.get("reference")
        if (calibration.get("record_id") != record.record_id or not isinstance(reference, str) or not reference.strip()
                or calibration.get("method") != "shared_hardware_marker"):
            raise ValueError("Calibration requires the matching record ID and an independent shared hardware marker")
        estimates = calibration.get("modalities", {})
        if not isinstance(estimates, dict) or not set(record.waves) <= set(estimates):
            raise ValueError("Latency and uncertainty estimates are required for each available modality")
        latency = {}
        uncertainty = {}
        for m in record.waves:
            entry = estimates[m]
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid calibration entry for {m}")
            delay, error = entry.get("latency_s"), entry.get("uncertainty_s")
            if (type(delay) not in (int, float) or type(error) not in (int, float)
                    or not math.isfinite(delay) or not math.isfinite(error) or error < 0):
                raise ValueError(f"Invalid latency or uncertainty for {m}")
            latency[m], uncertainty[m] = delay, error
    axes = {m: TimeAxis(w.fs, len(w.values), offset_correction_s=-latency[m] if latency else 0.0)
            for m, w in record.waves.items()}
    grid_matches = (
        len({a.sample_count for a in axes.values()}) == 1
        and all(a.sample_rate_hz == 1000 for a in axes.values())
    )
    if len(axes) == 1:
        status = "single_modality"
    elif documented_common_grid and grid_matches:
        status = "documented_common_grid"
    elif documented_common_grid:
        status = "unexpected_grid"
    else:
        status = "unverified_alignment"
    common_window = [max(a.origin_s for a in axes.values()), min(a.end_s for a in axes.values())]
    if common_window[0] >= common_window[1]:
        raise ValueError("Available time axes do not overlap after latency correction")
    grid_aligned = status == "documented_common_grid"
    return axes, {
        "status": status,
        "sample_grid_timing_allowed": grid_aligned,
        "cross_modality_timing_allowed": grid_aligned and calibration is not None,
        "common_window_s": common_window,
        "device_latency_s": latency,
        "latency_uncertainty_s": uncertainty,
        "latency_calibration_status": "provided" if calibration is not None else "unavailable",
        "calibration_reference": calibration["reference"] if calibration else None,
        "offset_method": "subtract_measured_device_latency" if calibration else "none",
        "axes": {m: asdict(a) | {"end_s": a.end_s} for m, a in axes.items()},
    }
