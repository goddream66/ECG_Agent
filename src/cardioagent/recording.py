from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import wfdb


MODALITIES = ("ecg", "ppg", "pcg")


def sample_span(fs: float, count: int, origin: float, start: float, end: float) -> tuple[int, int]:
    """Sample indices inside the half-open recording time window [start, end)."""
    if not origin <= start < end <= origin + count / fs + 1e-9:
        raise ValueError(f"Window {start:g}–{end:g} s exceeds waveform bounds")
    first = max(0, math.ceil((start - origin) * fs - 1e-8))
    last = min(count, math.ceil((end - origin) * fs - 1e-8))
    return first, last


@dataclass(frozen=True)
class Waveform:
    values: np.ndarray
    fs: float
    channels: tuple[str, ...]
    units: tuple[str, ...]

    @property
    def duration(self) -> float:
        return len(self.values) / self.fs

    def window(self, channel: str, start: float, end: float) -> np.ndarray:
        if channel not in self.channels:
            raise ValueError(f"Unknown channel {channel!r}; choose from {self.channels}")
        first, last = sample_span(self.fs, len(self.values), 0, start, end)
        return self.values[first:last, self.channels.index(channel)]


@dataclass(frozen=True)
class Recording:
    record_id: str
    waves: dict[str, Waveform]
    unavailable: dict[str, str]

    @classmethod
    def read(cls, directory: Path, record_id: str, omit: set[str] | None = None) -> "Recording":
        if not record_id or Path(record_id).name != record_id or record_id in {".", ".."}:
            raise ValueError("record_id must be a WFDB record prefix, without a path")
        waves = {}
        unavailable = {}
        for modality in MODALITIES:
            if modality in (omit or set()):
                unavailable[modality] = "excluded for this run"
                continue
            stem = directory / f"{record_id}_{modality}"
            if not stem.with_suffix(".hea").is_file():
                unavailable[modality] = "WFDB header unavailable"
                continue
            try:
                record = wfdb.rdrecord(str(stem))
                if record.fs <= 0 or record.p_signal is None or len(record.p_signal) == 0:
                    raise ValueError("invalid sampling rate or empty waveform")
                waves[modality] = Waveform(
                    record.p_signal, float(record.fs), tuple(record.sig_name), tuple(record.units)
                )
            except (OSError, ValueError) as exc:
                unavailable[modality] = f"WFDB read failed: {exc}"
        if not waves:
            raise ValueError(f"No readable waveforms for {record_id} in {directory}: {unavailable}")
        return cls(record_id, waves, unavailable)

    @property
    def shared_duration(self) -> float:
        return min(wave.duration for wave in self.waves.values())

    def describe(self) -> dict:
        return {
            "record_id": self.record_id,
            "common_duration_s": self.shared_duration,
            "available": {
                key: {"channels": wave.channels, "sample_rate_hz": wave.fs, "duration_s": wave.duration}
                for key, wave in self.waves.items()
            },
            "unavailable": self.unavailable,
        }
