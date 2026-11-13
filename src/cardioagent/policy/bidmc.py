"""Import the open BIDMC WFDB subset and make aligned waveform/numeric windows."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import re
from urllib.request import urlopen

import numpy as np
import wfdb

from ..dataset import digest
from ..provenance import experiment_provenance
from ..rate_audit import perturb_recording
from ..recording import Recording, Waveform, sample_span
from .value import capture_profile


SOURCE = "https://physionet.org/content/bidmc/1.0.0/"
FILES = "https://archive.physionet.org/physiobank/database/bidmc/"
RECORDS = tuple(f"bidmc{i:02}" for i in range(1, 54))
OFFSETS = (30, 150, 270, 390)
WINDOW = 60


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    names = [f"{record}{suffix}" for record in RECORDS for suffix in (".hea", ".dat", "n.hea", "n.dat")]

    def fetch(name):
        target = root / name
        if target.is_file() and target.stat().st_size:
            return
        for attempt in range(3):
            try:
                with urlopen(FILES + name, timeout=25) as response:
                    data = response.read()
                    if not data or "text/html" in response.headers.get("Content-Type", ""):
                        raise OSError(f"Unexpected download for {name}")
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
                return
            except OSError:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, _ in enumerate(executor.map(fetch, names), 1):
            if i % 40 == 0 or i == len(names):
                print(f"BIDMC files: {i}/{len(names)}", flush=True)


def _subject_id(header):
    match = re.search(r"matched/(s\d+)/", header.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Missing original subject identifier in {header.name}")
    return f"bidmc:{match[1]}"


def _signals(record, start):
    first, last = sample_span(record.fs, record.sig_len, 0, start, start + WINDOW)
    names = [name.strip(", ").upper() for name in record.sig_name]
    waves = {}
    for modality, allowed in (("ecg", ("II", "I", "III", "V", "AVR")), ("ppg", ("PLETH",))):
        indices = [i for i, name in enumerate(names) if name in allowed and
                   np.isfinite(record.p_signal[first:last, i]).all()]
        if indices:
            waves[modality] = Waveform(record.p_signal[first:last, indices], float(record.fs),
                                       tuple(names[i] for i in indices), tuple(record.units[i] for i in indices))
    return waves


def _reference(numerics, start, name):
    index = [n.strip(", ").upper() for n in numerics.sig_name].index(name)
    first, last = sample_span(numerics.fs, numerics.sig_len, 0, start, start + WINDOW)
    values = numerics.p_signal[first:last, index]
    valid = values[np.isfinite(values) & (values >= 25) & (values <= 240)]
    return float(np.median(valid)) if len(valid) >= .8 * len(values) else None


def capture(root, output, *, paired_hr=False):
    subjects = {record: _subject_id(root / f"{record}n.hea") for record in RECORDS}
    ranked = sorted(set(subjects.values()), key=lambda s: sha256(f"v24:{s}".encode()).digest())
    a, b = round(.6 * len(ranked)), round(.8 * len(ranked))
    splits = {s: group for group, subset in (("train", ranked[:a]), ("validation", ranked[a:b]), ("test", ranked[b:]))
              for s in subset}
    hashes = {f"{r}{suffix}": digest(root / f"{r}{suffix}") for r in RECORDS for suffix in (".hea", ".dat", "n.hea", "n.dat")}
    config = {"schema": 1, "dataset": "bidmc-1.0.0", "source": SOURCE, "window_seconds": WINDOW, "offsets": OFFSETS,
              "subject_splits": splits, "file_hashes": hashes, "paired_hr_all_splits": paired_hr,
              "references": {"heart_rate": "monitor HR from ECG", "peripheral_pulse_rate": "monitor PULSE from PPG"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(experiment_provenance(config)) + "\n")
        for i, record_id in enumerate(RECORDS, 1):
            wave = wfdb.rdrecord(str(root / record_id))
            numerics = wfdb.rdrecord(str(root / f"{record_id}n"))
            subject = subjects[record_id]
            for start in OFFSETS:
                waves = _signals(wave, start)
                case_id = f"{record_id}-{start:03}"
                if "ppg" in waves:
                    rate = _reference(numerics, start, "PULSE")
                    if rate is not None:
                        pr = Recording(case_id, {"ppg": waves["ppg"]}, {})
                        _write_case(stream, pr, "peripheral_pulse_rate", rate, record_id, subject, splits[subject],
                                    (("original", "original"), ("ppg_low_snr", "ecg_ppg_low_snr")))
                if (paired_hr or splits[subject] == "test") and {"ecg", "ppg"} <= waves.keys():
                    rate = _reference(numerics, start, "HR")
                    if rate is not None:
                        hr = Recording(case_id, waves, {})
                        _write_case(stream, hr, "heart_rate", rate, record_id, subject, splits[subject],
                                    tuple((s, s) for s in ("original", "low_snr", "missing_ecg")))
            if i % 10 == 0 or i == len(RECORDS):
                print(f"BIDMC replay: {i}/{len(RECORDS)} recordings", flush=True)
    temporary.replace(output)


def _write_case(stream, record, task, rate, record_id, subject, split, scenarios):
    original = capture_profile(record, task)
    for name, perturbation in scenarios:
        profile = original if name == "original" else capture_profile(
            perturb_recording(record, perturbation), task, original=record, original_profile=original)
        item = {"dataset": "bidmc", "subject_id": subject, "split": split, "record_id": record_id,
                "case_id": record.record_id, "reference_bpm": rate, "scenario": name, "profile": asdict(profile)}
        stream.write(json.dumps(item, allow_nan=False) + "\n")
