"""Build offline AFDB window caches. Usage: python scripts/preprocess_afdb.py --help."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import wfdb

from cardioagent.ecg_agent.data.preprocessing import (
    BANDPASS_HZ, CLIP_VALUE, FILTER_ORDER, FLAT_SECONDS, PROFILE, SCALE_FLOOR,
    TARGET_FS, WINDOW_SECONDS, invalid_spans, preprocess_window,
)


import os
DEFAULT_RAW_ROOT = Path(os.environ["ECG_AGENT_STORAGE"])/"data/afdb_local" if os.environ.get("ECG_AGENT_STORAGE") else Path("D:/ECGagent/ECG/data/raw/afdb_local")
CHANNELS = ("ECG1", "ECG2")
WINDOWS_PER_READ = 60


def write_record(stem: Path, output: Path, duration_s: int | None, *, source_fs: int = 250) -> dict:
    header = wfdb.rdheader(str(stem))
    fs = source_fs
    if header.fs != fs or tuple(header.sig_name) != CHANNELS or not header.sig_len:
        raise ValueError(f"{stem.name}: expected nonempty data with ECG1/ECG2 at {fs} Hz")
    count = min(header.sig_len, duration_s * fs) if duration_s is not None else header.sig_len
    width = WINDOW_SECONDS * fs
    batch_width = WINDOWS_PER_READ * width
    model_rows = []
    counts = {"ok": 0, "insufficient": 0, "incomplete": 0}
    metadata_name = f"{stem.name}.windows.jsonl"
    waveform_name = f"{stem.name}.npy"

    with (output / metadata_name).open("w", encoding="utf-8", newline="\n") as index:
        for batch_start in range(0, count, batch_width):
            batch_end = min(batch_start + batch_width, count)
            wave = wfdb.rdrecord(str(stem), sampfrom=batch_start, sampto=batch_end).p_signal
            if wave is None or wave.shape != (batch_end - batch_start, len(CHANNELS)):
                raise ValueError(f"{stem.name}: unexpected waveform dimensions")
            for offset in range(0, len(wave), width):
                window = wave[offset:offset + width]
                start = batch_start + offset
                end = start + len(window)
                for channel_index, channel in enumerate(CHANNELS):
                    values = window[:, channel_index]
                    spans = invalid_spans(values, fs)
                    row = None
                    if len(values) < width:
                        status = "incomplete"
                    elif spans:
                        status = "insufficient"
                    else:
                        row = len(model_rows)
                        model_rows.append(preprocess_window(values, fs))
                        status = "ok"
                    counts[status] += 1
                    metadata = {
                        "record_id": stem.name,
                        "channel": channel,
                        "start_sample": start,
                        "end_sample": end,
                        "start_s": start / fs,
                        "end_s": end / fs,
                        "status": status,
                        "invalid_spans": [
                            {"start_sample": start + a, "end_sample": start + b, "reason": reason}
                            for a, b, reason in spans
                        ],
                        "waveform_row": row,
                    }
                    index.write(json.dumps(metadata, allow_nan=False) + "\n")

    # Write one record at a time, without a second stacked copy in memory.
    shape = (len(model_rows), WINDOW_SECONDS * TARGET_FS)
    if model_rows:
        cache = np.lib.format.open_memmap(output / waveform_name, mode="w+", dtype=np.float32, shape=shape)
        for row, values in enumerate(model_rows):
            cache[row] = values
        cache.flush()
        del cache
    else:
        np.save(output / waveform_name, np.empty(shape, dtype=np.float32), allow_pickle=False)
    return {
        "record_id": stem.name,
        "source_samples": count,
        "source_record_samples": header.sig_len,
        "source_rate_hz": fs,
        "waveforms": waveform_name,
        "window_index": metadata_name,
        "shape": list(shape),
        "channel_windows_by_status": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this cache")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--record", action="append", help="Five-digit AFDB ID; repeat to select multiple records")
    selection.add_argument("--all", action="store_true", help="Process all local AFDB records with waveforms")
    parser.add_argument("--duration-s", type=int, help="Limit each record to this many seconds from the start")
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be positive")

    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    if not raw_root.is_dir():
        parser.error(f"Raw data directory does not exist: {raw_root}")
    if output.exists() or output.is_relative_to(raw_root):
        parser.error("--output must be a new directory outside the raw data directory")
    records = args.record or sorted(
        path.stem for path in raw_root.glob("*.hea") if path.with_suffix(".dat").is_file()
    )
    if not records:
        parser.error("No local records found")
    for name in records:
        if len(name) != 5 or any(char not in "0123456789" for char in name):
            parser.error(f"Invalid AFDB record ID: {name!r}")
        if any(not (raw_root / f"{name}{suffix}").is_file() for suffix in (".hea", ".dat")):
            parser.error(f"Missing .hea or .dat for {name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".afdb-preprocess-", dir=output.parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        reports = []
        for name in dict.fromkeys(records):
            report = write_record(raw_root / name, staging, args.duration_s)
            reports.append(report)
            print(f"{name}: {report['channel_windows_by_status']}", flush=True)
        manifest = {
            "schema_version": 1,
            "profile": PROFILE,
            "dataset": "AFDB",
            "raw_root": str(raw_root),
            "channels": list(CHANNELS),
            "anatomical_leads": "unknown",
            "output_rate_hz": TARGET_FS,
            "output_units": "normalized",
            "dtype": "float32",
            "window_s": WINDOW_SECONDS,
            "stride_s": WINDOW_SECONDS,
            "bandpass_hz": list(BANDPASS_HZ),
            "butterworth_design_order": FILTER_ORDER,
            "filter": "sosfiltfilt_on_completed_window",
            "normalization": "window_median_center_and_p95_absolute_deviation",
            "normalization_scale_floor": SCALE_FLOOR,
            "clip_range": [-CLIP_VALUE, CLIP_VALUE],
            "flat_run_min_s": FLAT_SECONDS,
            "span_coordinates": "absolute_source_sample_indices_half_open",
            "records": reports,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    print(f"Cache: {output}")


if __name__ == "__main__":
    main()
