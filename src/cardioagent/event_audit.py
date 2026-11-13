"""Blinded annotation packets and point-by-point event scoring."""

from collections import defaultdict
import csv
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.optimize import linear_sum_assignment
import wfdb

from .dataset import digest, load_manifest, save_json
from .provenance import save_provenance
from .recording import Recording
from .signal.events import detect_events, ppg_event_samples
from .signal.synchronization import TimeAxis
from .v2 import inspect_recording
from .workflow import preferred_channel


EVENT_TYPES = {
    "r_peak": ("ecg", "r_peak_candidate", 0.05),
    "pulse_foot": ("ppg", "pulse_foot_candidate", 0.05),
    "systolic_peak": ("ppg", "systolic_peak_candidate", 0.05),
    "s1": ("pcg", None, 0.05),
    "s2": ("pcg", None, 0.05),
}
MITDB_RECORDS = ("100", "101", "103", "105", "108", "112")
MITDB_BEATS = set("N L R B A a J S V r F e j n E / f Q".split())


def event_matches(reference, detected, tolerance_s: float) -> list[tuple[int, int]]:
    expected = np.asarray(reference, dtype=float)
    actual = np.asarray(detected, dtype=float)
    if not len(expected) or not len(actual):
        return []
    distances = np.abs(expected[:, None] - actual[None, :])
    forbidden = (len(expected) + len(actual) + 1) * tolerance_s
    left, right = linear_sum_assignment(np.where(distances <= tolerance_s + 1e-12, distances, forbidden))
    return [(int(i), int(j)) for i, j in zip(left, right) if distances[i, j] <= tolerance_s + 1e-12]


def event_errors(reference, detected, tolerance_s: float) -> dict:
    matches = event_matches(reference, detected, tolerance_s)
    errors_ms = [1000 * (float(detected[j]) - float(reference[i])) for i, j in matches]
    return {
        "reference_events": len(reference), "detected_events": len(detected), "matched": len(errors_ms),
        "missed": len(reference) - len(errors_ms), "extra": len(detected) - len(errors_ms),
        "matched_errors_ms": errors_ms,
    }


def aggregate_events(rows: list[dict]) -> dict:
    tp = sum(row["matched"] for row in rows)
    reference = sum(row["reference_events"] for row in rows)
    detected = sum(row["detected_events"] for row in rows)
    errors = [e for row in rows for e in row["matched_errors_ms"]]
    return {
        "reference_events": reference, "detected_events": detected, "matched": tp,
        "missed": reference - tp, "extra": detected - tp,
        "precision": tp / detected if detected else None,
        "recall": tp / reference if reference else None,
        "f1": 2 * tp / (reference + detected) if reference + detected else None,
        "timing_mae_ms": float(np.mean(np.abs(errors))) if errors else None,
        "timing_bias_ms": float(np.mean(errors)) if errors else None,
    }


def audit_external_ecg(source: Path, output: Path) -> dict:
    rows = []
    for record_id in MITDB_RECORDS:
        stem = str(source / record_id)
        record = wfdb.rdrecord(stem)
        annotation = wfdb.rdann(stem, "atr")
        channel = record.sig_name.index("MLII")
        values = record.p_signal[:, channel]
        axis = TimeAxis(float(record.fs), len(values))
        candidates = detect_events("ecg", values, axis, 0)["events"]
        reference = [axis.time_at(int(i)) for i, symbol in zip(annotation.sample, annotation.symbol)
                     if symbol in MITDB_BEATS]
        row = event_errors(reference, [e["time_s"] for e in candidates], EVENT_TYPES["r_peak"][2])
        rows.append({"record_id": record_id, "lead": "MLII", **row})
    report = {
        "source": "https://physionet.org/content/mitdb/1.0.0/",
        "reference": "published expert-reviewed .atr beat positions",
        "reference_type": "external ECG-only", "tolerance_ms": 50,
        "metrics": aggregate_events(rows), "records": rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    report["provenance"] = save_provenance(output, {"experiment": "MIT-BIH event validation", "tolerance_ms": 50,
                                                   "inputs": {f"{r}.{ext}": digest(source / f"{r}.{ext}")
                                                              for r in MITDB_RECORDS for ext in ("hea", "dat", "atr")}})
    save_json(output / "external_ecg.json", report)
    return report


def audit_external_aux(source: Path, output: Path) -> dict:
    ppg_zip = source / "ppg-bp" / "PPG-BP_annot.zip"
    ppg_rows = defaultdict(lambda: defaultdict(list))
    semantics_checked = 0
    with ZipFile(ppg_zip) as archive:
        names = set(archive.namelist())
        signals = loadmat(BytesIO(archive.read("PPG-BP_ref1.mat")), simplify_cells=True)["ppg_data"]
        for signal in signals:
            record_id = signal["name"]
            samples = np.asarray(signal["sig"], dtype=float)
            fs = float(signal["fs"])
            detected = defaultdict(list)
            for kind, index, _, _ in ppg_event_samples(samples, fs):
                detected[kind].append(index / fs)
            for reviewer in ("MG", "PC"):
                prefix = f"{reviewer}final_PPG-BP_annot"
                file = f"{prefix}/merged/{record_id}.mat"
                if file not in names:
                    continue
                fields = loadmat(BytesIO(archive.read(file)), simplify_cells=True)["annot"]
                boundaries = np.atleast_1d(fields["os"]["t"]).astype(float)
                peaks = np.atleast_1d(fields["pk"]["t"]).astype(float)
                raw_file = f"{prefix}/all_annot/{record_id}_os.mat"
                raw = loadmat(BytesIO(archive.read(raw_file)), simplify_cells=True)["pk_anns"]["t"]
                if not np.array_equal(boundaries, np.atleast_1d(raw), equal_nan=True):
                    raise ValueError(f"PPG-BP merged onset/offset differs from _os.mat: {reviewer} {record_id}")
                semantics_checked += 1
                if (len(boundaries) != 2 or not np.isfinite(boundaries).all()
                        or not 0 <= boundaries[0] < boundaries[1] < len(samples) / fs):
                    continue
                onset, offset = boundaries
                peak = [p for p in peaks if onset < p < offset]
                if len(peak) != 1:
                    continue
                # Compare the beginning of the annotated pulse and its peak separately.
                for method, kind, ref in (
                    ("pulse_foot_refined_exploratory", "pulse_foot_refined_candidate", [onset]),
                    ("pulse_foot_trough_baseline", "pulse_foot_candidate", [onset]),
                    ("systolic_peak", "systolic_peak_candidate", peak),
                ):
                    inside = [t for t in detected[kind] if abs(t - ref[0]) <= 0.15]
                    ppg_rows[reviewer][method].append({"record_id": record_id,
                                                      "candidates_outside_150ms": len(detected[kind]) - len(inside),
                                                      **event_errors(ref, inside, 0.05)})

    pcg_rows = []
    for label_path in sorted((source / "circor").glob("*.tsv")):
        wave_path = label_path.with_suffix(".wav")
        fs, samples = wavfile.read(wave_path)
        duration = len(samples) / fs
        onsets = {"s1": [], "s2": []}
        with label_path.open(newline="", encoding="utf-8-sig") as stream:
            for start_text, end_text, code in csv.reader(stream, delimiter="\t"):
                if code in ("1", "3"):
                    start, end = float(start_text), float(end_text)
                    if not 0 <= start < end <= duration + 1 / fs:
                        raise ValueError(f"PCG annotation outside recording: {label_path.name}")
                    onsets["s1" if code == "1" else "s2"].append(start)
        pcg_rows.append({"record_id": label_path.stem, "sample_rate_hz": fs, **onsets})
    if not ppg_rows or not pcg_rows:
        raise ValueError("Both PPG-BP annotations and the CirCor waveform/TSV pairs are required")
    report = {
        "ppg": {
            "source": "https://zenodo.org/records/10523285", "sha256": digest(ppg_zip), "tolerance_ms": 50,
            "semantics_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11003363/",
            "reference_fields": {"onset": "first annot.os.t", "offset": "second annot.os.t, excluded",
                                 "systolic_peak": "annot.pk.t strictly between onset and offset"},
            "checked_against_raw_os_files": semantics_checked,
            "exploratory_refinement": "chosen after examining PPG-BP results; requires an independent dataset",
            "scope": "2.1 s finger PPG; one annotated pulse per clip; predictions within 150 ms of each labeled landmark",
            "evaluation_type": "local_landmark_matching",
            "candidate_restriction_ms": 150,
            "continuous_record_detection_f1": None,
            "reviewers": {name: {method: {"metrics": aggregate_events(rows), "records": rows}
                                 for method, rows in methods.items()} for name, methods in ppg_rows.items()},
        },
        "pcg": {
            "source": "https://physionet.org/content/circor-heart-sound/1.0.3/",
            "scope": "published S1/S2 segment onsets; reference inventory only",
            "counts": {kind: sum(len(row[kind]) for row in pcg_rows) for kind in ("s1", "s2")},
            "records": pcg_rows,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    report["provenance"] = save_provenance(output, {"experiment": "PPG-BP and CirCor event audit", "tolerance_ms": 50,
                                                   "ppg_archive_sha256": digest(ppg_zip),
                                                   "circor_inputs": {p.name: digest(p) for p in
                                                                     sorted((source / "circor").glob("*"))
                                                                     if p.suffix in (".wav", ".tsv")}})
    save_json(output / "external_aux_events.json", report)
    return report


def write_csv(path: Path, fields: tuple[str, ...], rows=()) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_records(manifest, cohort, expanded):
    splits = ("test",) if cohort == "test" else ("train", "validation")
    by_subject = defaultdict(list)
    for row in manifest["records"]:
        by_subject[row["subject_id"], row["activity"]].append(row)
    limit = (3 if cohort == "test" else 2) if expanded else 1
    for split in splits:
        subjects = manifest["subject_ids"][split]
        if split != "test":
            subjects = [subject for subject in subjects if subject != 1]
            if not expanded:
                subjects = subjects[:3]
        for subject in subjects:
            for activity in ("B", "A"):
                rows = by_subject[subject, activity]
                n = min(limit, len(rows))
                for index in range(n):
                    yield rows[(2 * index + 1) * len(rows) // (2 * n)]


def export_blind_packet(dataset: Path, output: Path, cohort: str = "development", *, expanded: bool = False) -> dict:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    manifest = load_manifest(dataset)
    output.mkdir(parents=True, exist_ok=False)
    windows = []
    for ref in review_records(manifest, cohort, expanded):
        record = Recording.read(dataset / "WFDB", ref["record_id"])
        start, end = 10.0, 20.0
        window_id = f"window_{len(windows) + 1:02d}"
        target = output / window_id
        target.mkdir()
        fig = Figure(figsize=(16, 7), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1, sharex=True)
        times = None
        values = {}
        for ax, modality in zip(axes, ("ecg", "ppg", "pcg")):
            wave = record.waves[modality]
            channel = preferred_channel(modality, wave.channels)
            samples = wave.window(channel, start, end)
            if wave.fs != 1000 or len(samples) != 10000:
                raise ValueError(f"Unexpected time grid in {ref['record_id']} ({modality})")
            times = start + np.arange(len(samples)) / wave.fs
            values[modality] = samples
            ax.plot(times, samples, linewidth=0.75)
            ax.set_ylabel(f"{modality.upper()} {channel}")
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.grid(axis="x", which="both", alpha=0.25)
        axes[-1].set(xlim=(start, end), xlabel="Seconds from recording start")
        fig.suptitle(f"{window_id} | Unmarked sensor waveforms")
        fig.savefig(target / "waveforms.png", dpi=180)
        np.savetxt(target / "waveforms.csv", np.column_stack([times, *values.values()]),
                   delimiter=",", fmt=("%.3f", "%.8g", "%.8g", "%.8g"), header="time_s,ecg,ppg,pcg", comments="")
        identity = {key: ref[key] for key in ("record_id", "split", "subject_id", "activity")}
        windows.append({"window_id": window_id, **identity, "start_s": start, "end_s": end})

    packet = {"dataset_manifest_sha256": digest(dataset / "manifest.json"), "cohort": cohort, "expanded": expanded, "windows": windows}
    save_json(output / "windows.json", packet)
    event_fields = ("window_id", "event_type", "time_s", "notes")
    for name in ("reviewer_a", "reviewer_b", "consensus"):
        write_csv(output / f"{name}_events.csv", event_fields)
    keys = [
        {"window_id": window["window_id"], "event_type": event_type, "state": "pending"}
        for window in windows for event_type in EVENT_TYPES
    ]
    for name in ("reviewer_a", "reviewer_b"):
        write_csv(output / f"{name}_review.csv", ("window_id", "event_type", "state", "reviewer_id"), keys)
    write_csv(output / "consensus_review.csv",
              ("window_id", "event_type", "state", "reviewer_a", "reviewer_b", "adjudicator"), keys)
    (output / "README.txt").write_text(
        "Review the unmarked waveforms independently. Use recording-relative time in seconds.\n"
        "Use r_peak (R apex), pulse_foot (systolic upstroke onset), systolic_peak (pulse apex), s1/s2 (sound onset).\n"
        "Agree on the fiducial definition before review, and record it with the study protocol.\n"
        "Each reviewer fills their own events and review files. Set state to reviewed or unreadable, with their ID.\n"
        "Use review-event-annotations to get the disagreement queue before preparing consensus.\n"
        "After adjudication put accepted events in consensus_events.csv and record both reviewer IDs in "
        "consensus_review.csv. Add an adjudicator ID for windows/types with flagged disagreements.\n"
        "A reviewed track with zero events requires explicit sign-off. Keep ambiguous tracks unreadable.\n"
        "Reserve all test windows for final evaluation.\n",
        encoding="utf-8",
    )
    return packet


def packet_windows(dataset: Path, packet: Path) -> dict:
    data = json.loads((packet / "windows.json").read_text(encoding="utf-8"))
    if data["dataset_manifest_sha256"] != digest(dataset / "manifest.json"):
        raise ValueError("Annotation packet was exported from a different dataset manifest")
    windows = {window["window_id"]: window for window in data["windows"]}
    source = {row["record_id"]: row for row in load_manifest(dataset)["records"]}
    if len(windows) != len(data["windows"]) or not windows:
        raise ValueError("Invalid or duplicate packet windows")
    for window in windows.values():
        ref = source.get(window["record_id"])
        if not ref or any(window[key] != ref[key] for key in ("split", "subject_id", "activity")):
            raise ValueError(f"Window does not match the subject split: {window['window_id']}")
    splits = {window["split"] for window in windows.values()}
    if "test" in splits and splits != {"test"}:
        raise ValueError("Test and development windows must stay in separate packets")
    return windows


def read_reviews(path: Path, windows: dict) -> dict:
    reviews = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"window_id", "event_type", "state"} <= set(reader.fieldnames or ()):
            raise ValueError(f"Missing required review columns in {path.name}")
        for row in reader:
            key = (row["window_id"], row["event_type"])
            if key[0] not in windows or key[1] not in EVENT_TYPES or key in reviews:
                raise ValueError(f"Invalid or repeated review: {key}")
            if row["state"] not in ("pending", "reviewed", "unreadable"):
                raise ValueError(f"Invalid review state: {key}")
            reviews[key] = row
    return reviews


def read_event_file(path: Path, windows: dict, approved: set) -> dict:
    labels = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"window_id", "event_type", "time_s"} <= set(reader.fieldnames or ()):
            raise ValueError(f"Missing required event columns in {path.name}")
        for row in reader:
            key = (row["window_id"], row["event_type"])
            if key not in approved:
                raise ValueError(f"{path.name} event must belong to a reviewed window/type: {key}")
            window = windows[key[0]]
            if not row.get("time_s"):
                raise ValueError(f"Missing event timestamp in {path.name}: {key}")
            time = float(row["time_s"])
            if not math.isfinite(time) or not window["start_s"] <= time < window["end_s"] or time in labels[key]:
                raise ValueError(f"Invalid or repeated event timestamp: {key} {time}")
            labels[key].append(time)
    return {key: sorted(times) for key, times in labels.items()}


def validate_reviewer_files(packet: Path, windows: dict) -> dict:
    reviewers = {}
    for name in ("reviewer_a", "reviewer_b"):
        path = packet / f"{name}_review.csv"
        if not path.is_file():
            raise ValueError("Reviewer sign-off files are required; copy the review templates from a V2.1 packet")
        reviews = read_reviews(path, windows)
        completed = {key for key, row in reviews.items() if row["state"] == "reviewed"}
        if any(not row.get("reviewer_id", "").strip() for row in reviews.values() if row["state"] != "pending"):
            raise ValueError(f"Reviewer ID required in {path.name}")
        labels = read_event_file(packet / f"{name}_events.csv", windows, completed)
        reviewers[name] = {"reviews": reviews, "completed": completed, "events": labels}
    return reviewers


def compute_inter_rater_metrics(reviewers: dict) -> dict:
    a, b = (reviewers[name] for name in ("reviewer_a", "reviewer_b"))
    common = a["completed"] & b["completed"]
    by_type = defaultdict(list)
    disputes = []
    for key in sorted(common):
        window_id, event_type = key
        first = a["reviews"][key]["reviewer_id"].strip()
        second = b["reviews"][key]["reviewer_id"].strip()
        if first.casefold() == second.casefold():
            raise ValueError(f"Two different reviewer IDs required: {key}")
        times_a, times_b = a["events"].get(key, []), b["events"].get(key, [])
        tolerance = EVENT_TYPES[event_type][2]
        by_type[event_type].append({"window_id": window_id, **event_errors(times_a, times_b, tolerance)})
        matches = event_matches(times_a, times_b, tolerance)
        disagreements = [
            {"reason": "timing_difference", "a_s": times_a[i], "b_s": times_b[j]}
            for i, j in matches if abs(times_a[i] - times_b[j]) > 0.020 + 1e-12
        ]
        paired_a = {i for i, _ in matches}
        paired_b = {j for _, j in matches}
        disagreements += [{"reason": "only_a", "a_s": t} for i, t in enumerate(times_a) if i not in paired_a]
        disagreements += [{"reason": "only_b", "b_s": t} for i, t in enumerate(times_b) if i not in paired_b]
        if disagreements:
            disputes.append({"window_id": window_id, "event_type": event_type, "events": disagreements})
    return {
        "paired_reviews": len(common), "matching_tolerance_ms": 50, "adjudication_threshold_ms": 20,
        "timing_bias": "reviewer B minus reviewer A",
        "agreement": {kind: {"metrics": aggregate_events(rows), "windows": rows} for kind, rows in by_type.items()},
        "adjudication_queue": disputes,
        "completed_by_reviewer": {name: len(item["completed"]) for name, item in reviewers.items()},
    }


def validate_consensus(packet: Path, windows: dict, reviewers: dict, agreement: dict) -> tuple[set, dict]:
    reviews = read_reviews(packet / "consensus_review.csv", windows)
    approved = {key for key, row in reviews.items() if row["state"] == "reviewed"}
    if not approved:
        raise ValueError("No adjudicated windows with two independent reviews")
    disputed = {(row["window_id"], row["event_type"]) for row in agreement["adjudication_queue"]}
    for key in approved:
        row = reviews[key]
        for label in ("a", "b"):
            item = reviewers[f"reviewer_{label}"]
            if (key not in item["completed"] or (row.get(f"reviewer_{label}") or "").strip()
                    != item["reviews"][key]["reviewer_id"].strip()):
                raise ValueError(f"Consensus requires matching reviewer sign-offs: {key}")
        if key in disputed and not row.get("adjudicator", "").strip():
            raise ValueError(f"Disputed events require an adjudicator: {key}")
    labels = read_event_file(packet / "consensus_events.csv", windows, approved)
    return approved, labels


def score_final_detector(dataset: Path, windows: dict, approved: set, labels: dict) -> dict:
    rows = defaultdict(list)
    detector_reports = {}
    unavailable = defaultdict(int)
    for window_id, event_type in sorted(approved):
        modality, candidate, tolerance = EVENT_TYPES[event_type]
        if candidate is None:
            unavailable[event_type] += 1
            continue
        window = windows[window_id]
        if window_id not in detector_reports:
            record = Recording.read(dataset / "WFDB", window["record_id"])
            detector_reports[window_id] = inspect_recording(record, documented_common_grid=True,
                                                            start=window["start_s"], end=window["end_s"])
        observation = next(item for item in detector_reports[window_id]["observations"] if item["modality"] == modality)
        detected = [e["time_s"] for e in observation["events"] if e["event_type"] == candidate]
        stats = event_errors(labels.get((window_id, event_type), []), detected, tolerance)
        rows[event_type].append({"window_id": window_id, **stats})
        if event_type == "pulse_foot":
            refined = [e["time_s"] for e in observation["events"] if e["event_type"] == "pulse_foot_refined_candidate"]
            stats = event_errors(labels.get((window_id, event_type), []), refined, tolerance)
            rows["pulse_foot_refined_exploratory"].append({"window_id": window_id, **stats})
    return {"scored": {event: {"metrics": aggregate_events(items), "windows": items} for event, items in rows.items()},
            "unavailable_detector_windows": dict(unavailable), "tolerance_ms": 50,
            "consensus_events": sum(map(len, labels.values()))}


def review_packet(dataset: Path, packet: Path, output: Path) -> dict:
    windows = packet_windows(dataset, packet)
    report = compute_inter_rater_metrics(validate_reviewer_files(packet, windows))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "inter_rater_report.json", report)
    return report


def score_packet(dataset: Path, packet: Path, output: Path) -> dict:
    windows = packet_windows(dataset, packet)
    reviewers = validate_reviewer_files(packet, windows)
    agreement = compute_inter_rater_metrics(reviewers)
    approved, labels = validate_consensus(packet, windows, reviewers, agreement)
    report = score_final_detector(dataset, windows, approved, labels)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "inter_rater_report.json", agreement)
    for split in sorted({windows[key[0]]["split"] for key in approved}):
        directory = output / "adjudicated" / split
        directory.mkdir(parents=True)
        for modality in ("ecg", "ppg", "pcg"):
            fields = ("record_id", "subject_id", "modality", "event_type", "time_s", "review_status")
            write_csv(directory / f"{modality}_events.csv", fields, (
                {"record_id": windows[window_id]["record_id"], "subject_id": windows[window_id]["subject_id"],
                 "modality": modality, "event_type": kind, "time_s": time, "review_status": "adjudicated"}
                for (window_id, kind), times in sorted(labels.items())
                if windows[window_id]["split"] == split and EVENT_TYPES[kind][0] == modality
                for time in times
            ))
    report["provenance"] = save_provenance(output, {"experiment": "adjudicated event validation",
                                                   "inputs": {p.name: digest(p) for p in packet.glob("*.csv")},
                                                   "window_manifest_sha256": digest(packet / "windows.json")})
    save_json(output / "event_validation.json", report)
    return report
