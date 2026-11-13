"""Run interactively in local PowerShell to fetch a bounded PhysioNet pilot."""

import argparse
from collections import Counter
import csv
from getpass import getpass
from io import TextIOWrapper
from itertools import islice
from pathlib import Path
import re
import sys

import requests


URL = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0"
FIELDS = ("patient", "signal_file_name", "median_30s_hr", "strat_fold")


def download(root, subjects, scan_rows):
    if not sys.stdin.isatty():
        raise ValueError("Run this command yourself in an interactive PowerShell window")
    target = root / "metadata.csv"
    if target.exists():
        raise ValueError("metadata.csv already exists. Use that directory for the audit or choose a fresh download directory")
    username = input("PhysioNet username: ").strip()
    password = getpass("PhysioNet password: ")
    counts, patients = Counter(), set()
    with requests.Session() as session:
        session.auth = (username, password)
        del password
        with session.get(f"{URL}/metadata.csv", stream=True, timeout=40) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("PhysioNet returned a page instead of metadata. Verify dataset access in your account")
            response.raw.decode_content = True
            reader = csv.DictReader(TextIOWrapper(response.raw, encoding="utf-8-sig", newline=""))
            if not set(FIELDS) <= set(reader.fieldnames or ()):
                raise ValueError("Unexpected PhysioNet v1.1.0 metadata columns")
            root.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".part")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for row in islice(reader, scan_rows):
                    fold, patient = row["strat_fold"], row["patient"]
                    if fold not in ("8", "9") or counts[fold] >= subjects or patient in patients:
                        continue
                    stem = Path(row["signal_file_name"]).stem
                    if not re.fullmatch(r"p\d{6}", patient) or not re.fullmatch(r"\d+(?:_\d+){3}", stem):
                        continue
                    try:
                        if not 25 <= float(row["median_30s_hr"]) <= 240:
                            continue
                    except ValueError:
                        continue
                    relative = f"{patient[:3]}/{patient}/{stem}"
                    header = session.get(f"{URL}/{relative}.hea", timeout=30)
                    if header.status_code == 404:
                        continue
                    header.raise_for_status()
                    fields = [line.split() for line in header.text.splitlines()[1:] if line and not line.startswith("#")]
                    names = {parts[-1].upper() for parts in fields if parts}
                    if not {"II", "PLETH"} <= names:
                        continue
                    waveform = session.get(f"{URL}/{relative}.dat", timeout=30)
                    if waveform.status_code == 404:
                        continue
                    waveform.raise_for_status()
                    if not waveform.content or "text/html" in waveform.headers.get("Content-Type", "").lower():
                        raise ValueError("Unexpected waveform response from PhysioNet")
                    local = root / relative
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.with_suffix(".hea").write_bytes(header.content)
                    local.with_suffix(".dat").write_bytes(waveform.content)
                    writer.writerow(row)
                    stream.flush()
                    patients.add(patient)
                    counts[fold] += 1
                    print(f"Pilot pairs saved: fold 8 = {counts['8']}, fold 9 = {counts['9']}", flush=True)
                    if min(counts["8"], counts["9"]) >= subjects:
                        break
            if not counts["9"]:
                raise ValueError("No test pairs found before the metadata scan limit")
            temporary.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects-per-fold", type=int, default=8)
    parser.add_argument("--scan-rows", type=int, default=150_000)
    args = parser.parse_args()
    if not 1 <= args.subjects_per_fold <= 100 or args.scan_rows < 1:
        parser.error("Choose 1–100 subjects per fold and a positive scan limit")
    try:
        download(args.output, args.subjects_per_fold, args.scan_rows)
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f"Download stopped: {exc}") from None
