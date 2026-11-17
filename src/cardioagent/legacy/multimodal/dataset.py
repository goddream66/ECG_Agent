"""Local import and subject-level split for SensSmartTech 1.0.0."""

from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from cardioagent.common.recording import MODALITIES


DATASET = "senssmarttech-1.0.0"
DATASET_ROOT = Path("data") / DATASET
SOURCE = "https://physionet.org/content/senssmarttech/1.0.0/"
SPLIT_SEED = 20261111
SENSORS = (*MODALITIES, "acc")
ROOT_FILES = {"Demographics.csv", "LICENSE.txt", "README.txt", "RECORDS", "SHA256SUMS.txt"}
RECORD_PATTERN = r"[1-9][0-9]*_[0-9]{2}-[0-9]{2}-[0-9]{2}"






def records_from_table(table: Path) -> list[dict]:
    rows = []
    with table.open(encoding="utf-8-sig", newline="") as stream:
        next(stream)
        for row in csv.DictReader(stream):
            if not row["Subject number"]:
                continue
            subject = int(row["Subject number"])
            record = row["ECG"].removesuffix("_ecg")
            if (
                not re.fullmatch(RECORD_PATTERN, record)
                or record.split("_")[0] != str(subject)
                or any(row[m.upper()] != f"{record}_{m}" for m in SENSORS)
            ):
                raise ValueError(f"Inconsistent SensSmartTech record: {record}")
            hr = float(row["Median heart rate (bpm)"])
            activity = row["Before (B)  / after (A) activity"].strip()
            if not math.isfinite(hr) or hr <= 0 or activity not in ("A", "B"):
                raise ValueError(f"Invalid reference data: {record}")
            rows.append(dict(record_id=record, subject_id=subject, activity=activity, reference_bpm=hr))
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording in Demographics.csv")
    return rows


def split_records(rows: list[dict]) -> dict:
    subjects = sorted(
        {r["subject_id"] for r in rows},
        key=lambda subject: sha256(f"{SPLIT_SEED}:{subject}".encode()).digest(),
    )
    if len(subjects) != 32 or len(rows) != 338 or 1 not in subjects:
        raise ValueError("Expected the full SensSmartTech 1.0.0 release (32 subjects, 338 records)")
    # Subject 1 supplied the V1 development samples and was already inspected.
    subjects.remove(1)
    groups = {"train": sorted([1, *subjects[:19]]), "validation": sorted(subjects[19:25]), "test": sorted(subjects[25:])}
    assignments = {s: group for group, ids in groups.items() for s in ids}
    return {
        "dataset": DATASET, "source": SOURCE, "split_seed": SPLIT_SEED,
        "previously_inspected_subjects": [1], "subject_ids": groups,
        "record_counts": dict(Counter(assignments[r["subject_id"]] for r in rows)),
        "reference": "Demographics.csv: inverse median RR interval per full record (bpm)",
        "records": [dict(r, split=assignments[r["subject_id"]]) for r in rows],
    }


def _import_release(archive: Path, staging: Path) -> dict:
    with ZipFile(archive) as bundle:
        files = {}
        for member in bundle.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) == 2 and parts[1] in ROOT_FILES:
                name = parts[1]
            elif len(parts) == 3 and parts[1] == "WFDB" and re.fullmatch(
                rf"{RECORD_PATTERN}_(ecg|ppg|pcg|acc)\.(hea|dat)", parts[2]
            ):
                name = "/".join(parts[1:])
            else:
                continue
            if name in files:
                raise ValueError(f"Duplicate archive entry: {name}")
            files[name] = member
        if not ROOT_FILES <= files.keys():
            raise ValueError("SensSmartTech release metadata is incomplete")
        checksums = {}
        for line in bundle.read(files["SHA256SUMS.txt"]).decode("utf-8-sig").splitlines():
            if line.strip():
                checksum, name = line.split(maxsplit=1)
                checksums[name] = checksum.lower()
        (staging / "WFDB").mkdir()
        for name, member in files.items():
            dest = staging / name
            checksum = sha256()
            with bundle.open(member) as stream, dest.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
                    output.write(chunk)
            if name != "SHA256SUMS.txt" and checksum.hexdigest() != checksums.get(name):
                raise ValueError(f"Release checksum mismatch: {name}")
    manifest = split_records(records_from_table(staging / "Demographics.csv"))
    expected = {
        f"WFDB/{r['record_id']}_{m}.{extension}"
        for r in manifest["records"] for m in SENSORS for extension in ("hea", "dat")
    }
    if set(files) - ROOT_FILES != expected:
        raise ValueError("WFDB files differ from the records listed in Demographics.csv")
    manifest.update(
        archive_sha256=digest(archive), wfdb_files=len(expected),
        demographics_sha256=checksums["Demographics.csv"],
        release_checksums_sha256=digest(staging / "SHA256SUMS.txt"),
        sample_format="WFDB, 1000 Hz, shared time axis documented by the dataset",
    )
    save_json(staging / "manifest.json", manifest)
    return manifest


def prepare_dataset(archive: Path, root: Path = DATASET_ROOT) -> dict:
    if root.exists():
        manifest = load_manifest(root)
        if digest(archive) != manifest["archive_sha256"]:
            raise ValueError(f"Dataset already exists and is from a different archive: {root}")
        return manifest
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".senssmarttech-", dir=root.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        manifest = _import_release(archive, staging)
        staging.rename(root)
    return manifest


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = split_records(records_from_table(root / "Demographics.csv"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Dataset manifest does not match the fixed subject split and reference table")
    if digest(root / "Demographics.csv") != manifest["demographics_sha256"]:
        raise ValueError("Demographics.csv was changed after import")
    return manifest

from cardioagent.common.io import digest, save_json
