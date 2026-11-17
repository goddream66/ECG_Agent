"""Release completed AFDB windows only; never load rhythm annotations."""
import json
import math
from pathlib import Path
from collections.abc import Iterator
import numpy as np
from .preprocessing import TARGET_FS, WINDOW_SECONDS
CHANNELS=("ECG1","ECG2")
SOURCE_FS=250

def _arrivals(cache: Path, record: dict, channels: tuple[str, ...], start_s: float,
              end_s: float) -> Iterator[dict[str, tuple[dict, np.ndarray | None]]]:
    """Advance in file order and release selected channels together at each window end."""
    name = record["record_id"]
    if not isinstance(name, str) or len(name) != 5 or any(c not in "0123456789" for c in name):
        raise ValueError(f"Invalid AFDB record ID: {name!r}")
    if record["source_rate_hz"] != SOURCE_FS or end_s > record["source_samples"] / SOURCE_FS + 1e-8:
        raise ValueError(f"{name}: cache does not cover the task span")
    data = np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != TARGET_FS * WINDOW_SECONDS or data.dtype != np.float32:
        raise ValueError(f"{name}: invalid waveform cache shape or dtype")
    width = WINDOW_SECONDS * SOURCE_FS
    next_row = 0
    expected_start = 0
    pending = {}
    with (cache / f"{name}.windows.jsonl").open(encoding="utf-8") as index:
        for line_number, line in enumerate(index):
            row = json.loads(line)
            start = (line_number // len(CHANNELS)) * width
            end = min(start + width, record["source_samples"])
            if end / SOURCE_FS > end_s:
                break
            if (row["record_id"] != name or row["channel"] != CHANNELS[line_number % len(CHANNELS)]
                    or row["start_sample"] != start or row["end_sample"] != end or end <= start):
                raise ValueError(f"{name}: cache index is out of order or misaligned")
            status = row["status"]
            if end - start < width:
                valid_status = status == "incomplete"
            else:
                valid_status = status in ("ok", "insufficient")
            if not valid_status:
                raise ValueError(f"{name}: invalid window status")
            location = row["waveform_row"]
            if status == "ok":
                if type(location) is not int or location != next_row or location >= len(data):
                    raise ValueError(f"{name}: cache row does not match arrival order")
                next_row += 1
            elif location is not None:
                raise ValueError(f"{name}: unavailable window points to a waveform")
            if row["channel"] in channels and start / SOURCE_FS >= start_s:
                values = np.array(data[location], copy=True) if location is not None else None
                pending[row["channel"]] = (row, values)
            if row["channel"] == CHANNELS[-1]:
                expected_start = end
                if pending:
                    yield pending
                    pending = {}
    last_complete = min(record["source_samples"], math.floor(end_s / WINDOW_SECONDS) * width)
    if expected_start < last_complete or (end_s >= record["source_samples"] / SOURCE_FS and expected_start != record["source_samples"]):
        raise ValueError(f"{name}: window index ends before the requested span")


def arrivals(cache: Path, record: dict, channel: str, start_s: float, end_s: float):
    for block in _arrivals(cache, record, (channel,), start_s, end_s):
        yield block[channel]
