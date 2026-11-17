"""Local MIMIC-III-Ext-PPG HR evaluation using released WFDB and metadata."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import wfdb

from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.paired_metrics import risk_coverage, summarize_rates
from cardioagent.legacy.multimodal.provenance import save_provenance
from cardioagent.legacy.multimodal.rate_audit import perturb_recording
from cardioagent.common.recording import Recording, Waveform
from cardioagent.legacy.multimodal.policy.audit_v24 import _trials
from cardioagent.legacy.multimodal.policy.conditional_value import ConditionalNetwork, Variant
from cardioagent.legacy.multimodal.policy.selection import run
from cardioagent.legacy.multimodal.policy.stopping import RiskCalibrator, calibration_metrics, fit_calibrator
from cardioagent.legacy.multimodal.policy.value import capture_profile, run_policy


SOURCE = "https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/"
FOLDS = {8: "calibration", 9: "test"}
SCENARIOS = ("original", "low_snr", "missing_ecg")


def local_segments(root, max_subjects, per_subject):
    metadata = root / "metadata.csv"
    if not metadata.is_file():
        raise ValueError("Provide a local MIMIC-III-Ext-PPG v1.1.0 folder containing metadata.csv and WFDB folders")
    headers = {(path.parent.name, path.stem): path for path in root.glob("p??/p??????/*.hea")}
    if not headers:
        raise ValueError("No local WFDB headers found under pXX/pXXXXXX")
    cohorts, seen = defaultdict(lambda: defaultdict(dict)), {}
    required = {"patient", "signal_file_name", "median_30s_hr", "strat_fold"}
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Expected v1.1.0 metadata columns: patient, signal_file_name, median_30s_hr, strat_fold")
        for row in reader:
            patient = row["patient"]
            stem = Path(row["signal_file_name"]).stem
            path = headers.get((patient, stem))
            if path is None:
                continue
            try:
                fold, hr = int(row["strat_fold"]), float(row["median_30s_hr"])
            except ValueError:
                continue
            previous = seen.setdefault(patient, fold)
            if previous != fold:
                raise ValueError("Metadata assigns a patient to multiple folds")
            if fold in FOLDS and 25 <= hr <= 240:
                group = cohorts[FOLDS[fold]][patient]
                if path in group and group[path] != hr:
                    raise ValueError("Conflicting HR references for one waveform")
                group[path] = hr
    if not cohorts.get("test"):
        raise ValueError("No local fold-9 WFDB records match valid ECG-derived heart-rate references")
    for cohort, patients in cohorts.items():
        for patient in sorted(patients, key=lambda p: sha256(p.encode()).digest())[:max_subjects]:
            records = patients[patient]
            for path in sorted(records, key=lambda p: sha256(p.stem.encode()).digest())[:per_subject]:
                yield cohort, patient, path, records[path]


def _record(path):
    record = wfdb.rdrecord(str(path.with_suffix("")))
    names = [name.strip().upper() for name in record.sig_name]
    waves = {}
    for modality, label in (("ecg", "II"), ("ppg", "PLETH")):
        if label not in names:
            return None
        index = names.index(label)
        values = record.p_signal[:, index:index + 1]
        if not np.isfinite(values).all():
            return None
        waves[modality] = Waveform(values, float(record.fs), (label,), (record.units[index],))
    if abs(record.sig_len / record.fs - 30.) > 1 / record.fs:
        return None
    return Recording(path.stem, waves, {})


def audit(root, checkpoint, output, max_subjects, per_subject):
    torch.set_num_threads(2)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = data["models"]["conditional_voi"]
    if not spec["variant"].get("advantage"):
        raise ValueError("This experiment requires a V2.4.1 checkpoint")
    model = ConditionalNetwork(spec["width"], Variant(**spec["variant"]))
    model.load_state_dict(spec["state_dict"])
    model.eval()
    budget, costs, priors, limit = (data[key] for key in ("budget", "costs", "priors", "stop_risk"))
    source_fit = data["calibrators"].get("heart_rate")
    source_fit = RiskCalibrator(**source_fit) if source_fit else None
    profiles, skips = defaultdict(list), Counter()
    for group, patient, path, hr in local_segments(root, max_subjects, per_subject):
        try:
            record = _record(path)
        except (OSError, ValueError):
            record = None
        if record is None:
            skips[group] += 1
            continue
        original = capture_profile(record)
        for scenario in SCENARIOS:
            profile = original if scenario == "original" else capture_profile(
                perturb_recording(record, scenario), original=record, original_profile=original)
            row = {"record_id": record.record_id, "case_id": record.record_id, "subject_id": patient,
                   "reference_bpm": hr, "scenario": scenario}
            profiles[group, scenario].append((row, profile))
    if not profiles.get(("test", "original")):
        raise ValueError("No readable paired ECG/PPG test waveforms were found in fold 9")
    validation = [trial for (group, _), cases in profiles.items() if group == "calibration"
                  for row, profile in cases for trial in _trials(row, run(profile, budget, costs, priors, model=model))]
    target_fit, target_note = None, None
    try:
        target_fit = fit_calibrator(validation, "subject_record")
    except ValueError as exc:
        target_note = str(exc)
    results = {}
    for scenario in SCENARIOS:
        rows, trials = [], []
        methods = {"conditional_voi": None}
        if source_fit:
            methods["source_calibrated"] = source_fit
        if target_fit:
            methods["target_calibrated"] = target_fit
        for row, profile in profiles["test", scenario]:
            row = row.copy()
            for name, fit in methods.items():
                result = run(profile, budget, costs, priors, model=model, calibrator=fit, limit=limit)
                row[name] = result["rate_bpm"]
                if name == "conditional_voi":
                    trials.extend(_trials(row, result))
            for name in ("ecg_only", "channel_eig", "ppg_fallback"):
                row[name] = run_policy(profile, budget, name, costs["waveform"])["rate_bpm"]
            rows.append(row | {"rank": 1.})
        results[scenario] = {"metrics": {name: summarize_rates(rows, name) | {"operating_point": risk_coverage(rows, name, "rank", [0.])[0]}
                                         for name in (*methods, "ecg_only", "channel_eig", "ppg_fallback")},
                             "source_calibration": calibration_metrics(trials, source_fit, "subject_record") if source_fit else None,
                             "target_calibration": calibration_metrics(trials, target_fit, "subject_record") if target_fit else None}
    config = {"source": SOURCE, "reference": "released median_30s_hr derived from ECG", "folds": FOLDS,
              "metadata_sha256": digest(root / "metadata.csv"), "checkpoint_sha256": digest(checkpoint),
              "max_subjects_per_fold": max_subjects, "max_segments_per_subject": per_subject, "stop_risk_limit": limit}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "mimic_hr_aggregate.json", config | {"provenance": save_provenance(output, config),
              "skipped_waveforms": dict(skips), "target_calibration_note": target_note, "results": results})
