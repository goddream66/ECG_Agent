"""Check whether arrived gap-local RR adds information to frozen window evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import wfdb

from cardioagent.common.io import digest, save_json, write_jsonl
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.evaluation.metrics import evaluate_record, intersection
from cardioagent.ecg_agent.experiments.episode_review import merge_gaps
from cardioagent.ecg_agent.experiments.gap_repair import LEADS, load_records, relation, temporal_features
from cardioagent.ecg_agent.harness import ToolContext, ToolSession


TOOL = 'detect_beats_featuredb'
PEAK_TOLERANCE_S = 0.075
LINK_THRESHOLDS = (0.65, 0.80, 0.90)
BASE_FEATURES = ('duration_s', 'support_fraction', 'low_only_s', 'unknown_s', 'longest_low_only_s',
                 'longest_unsupported_s', 'dual_usable_s', 'disagreement_s')


def detect(waves, record_id, gap, lead, cursor):
    start, end = gap
    if end > cursor:
        raise ValueError('RR tool received unarrived waveform')
    fs = float(waves[f'{record_id}_fs'])
    data = waves[f'{record_id}_{start:g}_{end:g}']
    if data.shape != (round((end - start) * fs), len(LEADS)):
        raise ValueError('Unexpected waveform dimensions')
    i = LEADS.index(lead)
    signal = Waveform(data[:, i:i+1], fs, (lead,), ('mV',))
    session = ToolSession(ToolContext(signal, start, end), allow_experimental=True)
    began = perf_counter()
    result = session.call(TOOL, {'start_s': start, 'end_s': end, 'lead': lead})
    return dict(status=result['status'], result=result.get('result'), error=result.get('error'),
                wall_s=perf_counter()-began, review_cursor_s=cursor)


def match_peaks(estimated, reference, tolerance=PEAK_TOLERANCE_S):
    """One-to-one tolerance matching on sorted peak times."""
    i = j = 0
    errors = []
    while i < len(estimated) and j < len(reference):
        delta = estimated[i] - reference[j]
        if delta < -tolerance:
            i += 1
        elif delta > tolerance:
            j += 1
        else:
            errors.append(abs(delta))
            i += 1
            j += 1
    return errors


def validate_detector(waves, annotations, records):
    references = {record.record_id: record.reference['intervals'] for record in records}
    comparisons = []
    for name in sorted(key for key in waves.files if not key.endswith('_fs')):
        record_id, start, end = name.split('_')
        gap = float(start), float(end)
        fs = float(waves[f'{record_id}_fs'])
        expected = wfdb.rdann(str(annotations / record_id), 'qrsc').sample / fs
        expected = expected[(expected >= gap[0]) & (expected < gap[1])]
        af_s = sum(b-a for a, b in intersection([gap], references[record_id]))
        state = 'AF/AFL' if af_s == gap[1]-gap[0] else 'mixed' if af_s else 'non-AF/AFL'
        for lead in LEADS:
            measured = detect(waves, record_id, gap, lead, gap[1])
            peaks = (measured['result'] or {}).get('r_peaks_s', [])
            errors = match_peaks(peaks, expected)
            comparisons.append(dict(record_id=record_id, gap_s=gap, lead=lead, state=state,
                                    true_peaks=len(expected), measured_peaks=len(peaks), matched_peaks=len(errors),
                                    errors_s=errors, tool_status=measured['status']))
    return comparisons


def peak_summary(rows):
    groups = dict(overall=rows)
    for key in ('lead', 'record_id', 'state'):
        for value in sorted({row[key] for row in rows}):
            groups[value] = [row for row in rows if row[key] == value]
    output = {}
    for group, values in groups.items():
        total = {key: sum(r[key] for r in values) for key in ('true_peaks', 'measured_peaks', 'matched_peaks')}
        errors = [error for row in values for error in row['errors_s']]
        total.update(windows=len(values), precision=total['matched_peaks']/total['measured_peaks'],
                     recall=total['matched_peaks']/total['true_peaks'],
                     median_error_ms=1000*float(np.median(errors)),
                     p95_error_ms=1000*float(np.quantile(errors, .95)))
        output[group] = total
    return output


def features_with_rr(data):
    rr = np.asarray((data['result'] or {}).get('rr_intervals_ms', []), dtype=float)
    if data['status'] != 'ok' or len(rr) < 4 or not np.all(rr > 0):
        return None
    center = np.median(rr)
    return [float(np.std(rr)/np.mean(rr)), float(np.median(np.abs(np.diff(rr)))/center)]


def read_cases(records, waves):
    cases = []
    for record in records:
        for packet in record.packets:
            gap = tuple(packet['gap_s'])
            measured = detect(waves, record.record_id, gap, record.lead, packet['cursor_s'])
            f = temporal_features(packet)
            cases.append(dict(lead=record.lead, record_id=record.record_id, gap_s=gap, cursor_s=packet['cursor_s'],
                              reference=relation(record, packet), strict=all(packet['gap_support']),
                              features=[f[key] for key in BASE_FEATURES], rr_features=features_with_rr(measured),
                              measurement=measured))
    return cases


def cross_validate(cases):
    labels = np.array([case['reference'] == 'continuous' for case in cases], dtype=int)
    groups = np.array([case['record_id'] for case in cases])
    base = np.asarray([case['features'] for case in cases], dtype=float)
    known = np.asarray([case['rr_features'] is not None for case in cases])
    rr = np.asarray([case['rr_features'] if case['rr_features'] is not None else [np.nan, np.nan]
                     for case in cases])
    scores = {key: np.empty(len(cases)) for key in ('base', 'plus_rr')}
    for train, test in LeaveOneGroupOut().split(base, labels, groups):
        if len(np.unique(labels[train])) != 2:
            raise ValueError('A record holdout leaves one reference class')
        training_rr = train[known[train]]
        filled = np.where(known[:, None], rr, np.median(rr[training_rr], axis=0))
        designs = dict(base=base, plus_rr=np.column_stack((base, filled, known)))
        for key, x in designs.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            model.fit(x[train], labels[train])
            scores[key][test] = model.predict_proba(x[test])[:, 1]
        # An unavailable RR measurement keeps the original diagnostic score.
        scores['plus_rr'][test[~known[test]]] = scores['base'][test[~known[test]]]
    for i, case in enumerate(cases):
        case['oof_score'] = {key: float(v[i]) for key, v in scores.items()}
    return groups, labels, scores


def discrimination(groups, labels, scores):
    rng = np.random.default_rng(20261114)
    record_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    deltas = []
    for draws in rng.integers(len(record_indices), size=(2000, len(record_indices))):
        ix = np.concatenate([record_indices[i] for i in draws])
        if len(np.unique(labels[ix])) == 2:
            deltas.append(average_precision_score(labels[ix], scores['plus_rr'][ix]) -
                          average_precision_score(labels[ix], scores['base'][ix]))
    return dict(methods={key: dict(ap=average_precision_score(labels, value), auc=roc_auc_score(labels, value))
                         for key, value in scores.items()},
                ap_gain_record_bootstrap_95=np.quantile(deltas, [.025, .975]).tolist())


def repair_points(records, cases):
    grouped = {}
    for case in cases:
        grouped.setdefault((case['lead'], case['record_id']), []).append(case)
    output = []
    for lead in LEADS:
        for key in ('base', 'plus_rr'):
            for threshold in LINK_THRESHOLDS:
                totals = Counter()
                positive_records = 0
                for record in (r for r in records if r.lead == lead):
                    items = grouped.get((lead, record.record_id), [])
                    selected = [c for c in items if c['strict'] or (
                        c['features'][1] >= .5 and c['features'][3] == 0 and c['oof_score'][key] >= threshold)]
                    positive_records += any(c['reference'] == 'continuous' and not c['strict'] for c in selected)
                    gaps = {tuple(c['gap_s']) for c in selected}
                    pred = merge_gaps(record.intervals, gaps)
                    scored = evaluate_record(record.reference['task_id'], pred, record.reference, valid=record.valid)
                    totals.update({k: scored[k] for k in ('tp', 'fp', 'fn', 'long_fragment_excess', 'time_fp_s')})
                    totals.update(c['reference'] for c in selected)
                output.append(dict(lead=lead, method=key, threshold=threshold, **totals,
                                   added_correct_record_count=positive_records,
                                   event_f1=2*totals['tp']/(2*totals['tp']+totals['fp']+totals['fn'])))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'waves', 'beat-waves', 'beat-reference', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    records, _ = load_records(args.source)
    with np.load(args.beat_waves) as waves:
        beat_rows = validate_detector(waves, args.beat_reference, records)
    peaks = peak_summary(beat_rows)
    print(json.dumps(peaks, ensure_ascii=False), flush=True)
    with np.load(args.waves) as waves:
        cases = read_cases(records, waves)
    groups, labels, scores = cross_validate(cases)
    results = discrimination(groups, labels, scores)
    points = repair_points(records, cases)
    report = dict(beat_validation=peaks, discrimination=results, repair_points=points,
                  staged_tools=dict(measurements=len(cases), rr_usable=sum(c['rr_features'] is not None for c in cases),
                                    wall_s=sum(c['measurement']['wall_s'] for c in cases)),
                  source_hashes={p.name: digest(p) for p in (args.waves, args.beat_waves, *args.beat_reference.glob('*.qrsc'))},
                  code_sha256=digest(Path(__file__)), thresholds=LINK_THRESHOLDS, peak_tolerance_s=PEAK_TOLERANCE_S,
                  feature_names=BASE_FEATURES, validation='leave-one-record-out, both lead runs held out together')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'comparison.json', report)
    write_jsonl(args.output / 'beat_comparison.jsonl', beat_rows)
    write_jsonl(args.output / 'candidate_rr.jsonl', cases)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print(json.dumps(points, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
