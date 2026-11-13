"""Keep published ECG-derived rates and adjudicated references in separate cohorts."""

from dataclasses import replace
from statistics import median

from .dataset import digest
from .recording import Recording, sample_span


REFERENCE_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/"


def evaluation_cases(dataset, manifest: dict, split: str, packet=None) -> tuple[list[dict], dict]:
    rows = [row for row in manifest["records"] if split == "all" or row["split"] == split]
    if packet is None:
        return [row | {"case_id": row["record_id"]} for row in rows], {
            "kind": "published_ecg_derived_rate", "source": REFERENCE_SOURCE,
            "method": "Elgendi on ECG I, II, V3; majority vote; inverse median RR",
            "reference_input_overlap": {"ecg": ["I", "II", "V3"]},
            "interpretation": "agreement with an ECG-derived reference, especially when using II/V3 after I degradation",
        }

    from .event_audit import compute_inter_rater_metrics, packet_windows, validate_consensus, validate_reviewer_files

    windows = packet_windows(dataset, packet)
    reviewers = validate_reviewer_files(packet, windows)
    agreement = compute_inter_rater_metrics(reviewers)
    approved, labels = validate_consensus(packet, windows, reviewers, agreement)
    eligible = {row["record_id"]: row for row in rows}
    cases = []
    for window_id, kind in sorted(approved):
        window = windows[window_id]
        if kind != "r_peak" or window["record_id"] not in eligible:
            continue
        beats = labels.get((window_id, kind), [])
        if len(beats) < 4 or window["end_s"] - window["start_s"] < 5:
            continue
        interval = median(b - a for a, b in zip(beats, beats[1:]))
        cases.append(eligible[window["record_id"]] | {
            "case_id": f"{window['record_id']}-w{len(cases) + 1:03d}", "annotation_window_id": window_id,
            "window_s": [window["start_s"], window["end_s"]],
            "reference_bpm": 60 / interval,
        })
    if not cases:
        raise ValueError(f"No eligible adjudicated R-peak windows for {split}")
    return cases, {
        "kind": "independently_adjudicated_ecg_r_peaks", "source": "two blinded reviewers and adjudication",
        "method": "60 / median RR from same-window adjudicated timestamps",
        "reference_input_overlap": {"ecg": ["source waveform used for manual review"]},
        "interpretation": "independent annotation process; same ECG source waveform and same evaluation window",
        "packet_sha256": {name: digest(packet / name) for name in ("windows.json", "consensus_events.csv",
                          "consensus_review.csv", "reviewer_a_events.csv", "reviewer_b_events.csv",
                          "reviewer_a_review.csv", "reviewer_b_review.csv")},
    }


def load_case_record(dataset, case: dict) -> Recording:
    record = Recording.read(dataset / "WFDB", case["record_id"])
    if "window_s" not in case:
        return record
    start, end = case["window_s"]
    waves = {}
    for modality, wave in record.waves.items():
        first, last = sample_span(wave.fs, len(wave.values), 0, start, end)
        waves[modality] = replace(wave, values=wave.values[first:last])
    return replace(record, waves=waves)
