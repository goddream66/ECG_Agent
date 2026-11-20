"""Paired fixed-cursor ECG evidence comparison with record-held-out logistic models."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median
import json

import numpy as np
import wfdb

from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.tools.measurements import _stats, rr_clean_mask
from check_last_window_scores import CHANNELS, decision, unique_cases
from fixed_transition_p34 import judge_view, write_jsonl
from fixed_transition_p35 import digest, rows, ENDED, ONGOING


FEATURE_GROUPS = {'scores': ('scores',), 'traditional': ('traditional',),
                  'fusion': ('scores', 'traditional')}
TRADITIONAL_FIELDS = ('rr_cv_raw', 'rr_cv_clean', 'heart_rate_median_rr_bpm',
                      'rr_removed_fraction', 'p_candidate_abs_median_mv',
                      'p_candidate_to_r_mad_ms', 'qrs_occupancy_median_ms')


def summary_median(values):
    return float(median(values)) if len(values) else None


def aggregate_leads(leads, field):
    return summary_median([v[field] for v in leads if v.get(field) is not None])


def measure_window(stem, fs, end):
    origin = end - 30
    if origin < 0:
        raise ValueError('Need 30 seconds of arrived signal')
    raw = wfdb.rdrecord(stem, sampfrom=round(origin * fs), sampto=round(end * fs))
    if len(raw.p_signal) != round(30 * fs):
        raise ValueError('Incomplete waveform')
    signal = Waveform(raw.p_signal, raw.fs, tuple(raw.sig_name), tuple(raw.units))
    session = ToolSession(ToolContext(signal, origin, end), allow_experimental=True)
    output = []
    for lead in CHANNELS:
        xqrs = session.call('detect_beats_xqrs_proxy', {'lead': lead, 'start_s': origin, 'end_s': end})
        morphology = session.call('get_beat_timeline_proxy', {'lead': lead, 'start_s': end - 10, 'end_s': end})
        peaks = [t for t in xqrs['result'].get('r_peaks_s', []) if end - 10 <= t < end]
        rr = np.diff(peaks) * 1000
        mask = rr_clean_mask(rr)
        raw_stats, clean_stats = _stats(rr), _stats(rr[mask])
        beats = morphology['result'].get('beats', []) if morphology['status'] == 'ok' else []
        amplitudes = [abs(b['p_candidate_amplitude_mv']) for b in beats
                      if b['p_candidate_amplitude_mv'] is not None]
        delays = np.array([(b['r_time_s'] - b['p_candidate_time_s']) * 1000 for b in beats
                           if b['p_candidate_time_s'] is not None])
        occupancies = [b['qrs_suprathreshold_occupancy_ms'] for b in beats
                       if b['qrs_suprathreshold_occupancy_ms'] is not None]
        rr_ok = xqrs['status'] == 'ok' and len(rr) >= 3
        output.append({'lead': lead, 'rr_status': 'ok' if rr_ok else
                       ('error' if xqrs['status'] == 'error' else 'insufficient'),
                       'morphology_status': morphology['status'], 'raw_rr_ms': rr.tolist(),
                       'rr_keep_mask': mask.tolist(), 'raw_rr_stats': raw_stats,
                       'clean_rr_stats': clean_stats, 'rr_cv_raw': raw_stats['cv'] if rr_ok else None,
                       'rr_cv_clean': clean_stats['cv'] if rr_ok and sum(mask) >= 3 else None,
                       'heart_rate_median_rr_bpm': float(60000 / np.median(rr)) if rr_ok else None,
                       'rr_removed_fraction': float(np.mean(~mask)) if rr_ok else None,
                       'p_candidate_abs_median_mv': summary_median(amplitudes),
                       'p_candidate_to_r_mad_ms': float(np.median(abs(delays - np.median(delays))))
                                                 if len(delays) >= 3 else None,
                       'qrs_occupancy_median_ms': summary_median(occupancies)})
    return output


def score_evidence(case):
    result = []
    cursor = case['cursor_s']
    for lead in CHANNELS:
        scores = sorted((r for r in case['scores'] if r['lead'] == lead),
                        key=lambda r: r['window_s'][1])
        values = [max(r['result'].values()) if r['status'] == 'ok' else None for r in scores]
        history = [v for v in values[:4] if v is not None]
        result.append({'id': 'S' + lead[-1], 'status': 'ok' if history else 'insufficient', 'lead': lead,
                       'window_s': [cursor - 60, cursor], 'features': {
                           'af_or_afl_max_mean_earlier_40s': float(np.mean(history)) if history else None,
                           'af_or_afl_max_penultimate_10s': values[-2],
                           'af_or_afl_max_final_10s': values[-1]}})
    return result


def prepare(args):
    prepared, audit, window_cache = [], [], {}
    for case, ids in unique_cases(args.source):
        record, cursor = case['record_id'], case['cursor_s']
        stem = str(args.raw_root / record)
        fs = float(wfdb.rdheader(stem).fs)
        traditional = []
        for i, end in enumerate((cursor - 10, cursor), 1):
            if (record, end) not in window_cache:
                window_cache[record, end] = measure_window(stem, fs, end)
            lead_data = window_cache[record, end]
            features = {field: aggregate_leads(lead_data, field) for field in TRADITIONAL_FIELDS}
            traditional.append({'id': f'T{i}', 'status': 'ok' if any(v is not None for v in features.values())
                                else 'insufficient', 'window_s': [end - 10, end], 'features': features})
            audit.append({'case_id': case['case_id'], 'target_s': [end - 10, end], 'leads': lead_data})
        prepared.append({'case_id': case['case_id'], 'source_case_ids': ids, 'record_id': record,
                         'cursor_s': cursor, 'scores': score_evidence(case), 'traditional': traditional})
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / 'inputs.jsonl', prepared)
    write_jsonl(args.output / 'measurement_audit.jsonl', audit)
    (args.output / 'manifest.json').write_text(json.dumps({
        'score_source_sha256': digest(args.source), 'inputs_sha256': digest(args.output / 'inputs.jsonl'),
        'cases': len(prepared), 'unique_measured_windows': len(window_cache), 'ridge_c': .1,
        'held_out_unit': 'record_id', 'classification_threshold': .5}, indent=2), encoding='utf-8')
    print('Prepared', len(prepared), 'unique decision points', flush=True)


def view(case, group):
    cursor = case['cursor_s']
    return {'cursor_s': 0, **{k: [dict(row, window_s=[t - cursor for t in row['window_s']])
                                for row in case[k]] for k in FEATURE_GROUPS[group]}}


def vector(case, group):
    return [value for key in FEATURE_GROUPS[group] for row in case[key]
            for _, value in sorted(row['features'].items())]


def reference(cases, path):
    keys = {r['case_id']: r['reference'] for r in rows(path)}
    if set(keys) != {i for case in cases for i in case['source_case_ids']}:
        raise ValueError('Answer key does not match frozen decision points')
    result = []
    for case in cases:
        label, = {keys[i] for i in case['source_case_ids']}
        result.append(int(label == 'positive'))
    return np.array(result)


def classical(cases, outcomes):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = [c['record_id'] for c in cases]
    prediction_rows = []
    for group in FEATURE_GROUPS:
        features = np.array([vector(c, group) for c in cases], dtype=float)
        for train, test in LeaveOneGroupOut().split(features, outcomes, records):
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
            model.fit(features[train], outcomes[train])
            for idx, score in zip(test, model.predict_proba(features[test])[:, 1]):
                prediction_rows.append({'case_id': cases[idx]['case_id'], 'group': group,
                                        'decision': ENDED if score >= .5 else ONGOING,
                                        'end_score': float(score), 'held_out_record': records[idx]})
    return prediction_rows


def infer(args, cases):
    chosen = cases[args.part::args.parts]
    dest = args.output / f'qwen-{args.part}-of-{args.parts}.jsonl'
    if dest.exists():
        raise FileExistsError(f'Run output exists: {dest}')
    identity = {'inputs': digest(args.output / 'inputs.jsonl'), 'script': digest(Path(__file__)),
                'model': str(args.model_path.resolve())}
    model = build_chat_model({'provider': 'transformers', 'model_path': str(args.model_path),
                              'max_new_tokens': 512, 'device': 'cuda:0'})
    with dest.open('x', encoding='utf-8') as stream:
        for case in chosen:
            for group in FEATURE_GROUPS:
                result = judge_view(model, view(case, group))
                row = {'case_id': case['case_id'], 'group': group, 'identity': identity,
                       **result, 'decision': decision(result)}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(case['record_id'], case['cursor_s'], group, row['decision'], flush=True)


def summarize(args, cases):
    from sklearn.metrics import roc_auc_score

    outcomes = dict(zip((c['case_id'] for c in cases), reference(cases, args.answer_key).tolist()))
    record_ids = {c['case_id']: c['record_id'] for c in cases}
    expected = {(c['case_id'], group) for c in cases for group in FEATURE_GROUPS}
    logistic, qwen = rows(args.output / 'classical.jsonl'), [
        r for path in sorted(args.output.glob('qwen-*-of-*.jsonl')) for r in rows(path)]
    for predictions in (logistic, qwen):
        if len(predictions) != len(expected) or {(r['case_id'], r['group']) for r in predictions} != expected:
            raise ValueError('Missing or duplicate predictions')
    expected_hash = digest(args.output / 'inputs.jsonl')
    if (any(r['identity']['inputs'] != expected_hash for r in qwen)
            or len({json.dumps(r['identity'], sort_keys=True) for r in qwen}) != 1):
        raise ValueError('Qwen predictions come from different inputs or runs')

    def metrics(group):
        detail = Counter(f"{outcomes[r['case_id']]}:{r['decision']}" for r in group)
        return {'cases': len(group), 'outcomes': dict(detail), 'definitive': sum(
            r['decision'] in (ENDED, ONGOING) for r in group)}

    report = {}
    for group in FEATURE_GROUPS:
        a = [r for r in logistic if r['group'] == group]
        b = [r for r in qwen if r['group'] == group]
        covered = {r['case_id'] for r in b if r['decision'] in (ENDED, ONGOING)}
        if covered and len({outcomes[c] for c in covered}) == 2:
            same_subset_auc = roc_auc_score([outcomes[r['case_id']] for r in a if r['case_id'] in covered],
                                           [r['end_score'] for r in a if r['case_id'] in covered])
        else:
            same_subset_auc = None
        report[group] = {'logistic': metrics(a), 'qwen': metrics(b),
                         'logistic_on_qwen_definitive_cases': metrics([r for r in a if r['case_id'] in covered]),
                         'logistic_auc_on_qwen_definitive_cases': same_subset_auc,
                         'logistic_auc': roc_auc_score([outcomes[r['case_id']] for r in a],
                                                       [r['end_score'] for r in a]),
                         'by_record': {record: {name: metrics([r for r in selected
                             if record_ids[r['case_id']] == record]) for name, selected in
                             (('logistic', a), ('qwen', b))} for record in sorted(set(record_ids.values()))}}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_jsonl(args.output / 'judgment_audit.jsonl', [
        {**r, 'method': method, 'record_id': record_ids[r['case_id']],
         'reference': outcomes[r['case_id']]} for method, predictions in
        (('logistic', logistic), ('qwen', qwen)) for r in predictions])
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_args = modes.add_parser('prepare')
    prepare_args.add_argument('--source', required=True, type=Path)
    prepare_args.add_argument('--raw-root', required=True, type=Path)
    for mode in ('classical', 'summarize'):
        modes.add_parser(mode).add_argument('--answer-key', required=True, type=Path)
    run = modes.add_parser('infer')
    run.add_argument('--model-path', required=True, type=Path)
    run.add_argument('--part', default=0, type=int)
    run.add_argument('--parts', default=1, type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
        return
    cases = rows(args.output / 'inputs.jsonl')
    if args.mode == 'classical':
        write_jsonl(args.output / 'classical.jsonl', classical(cases, reference(cases, args.answer_key)))
    elif args.mode == 'infer':
        if not 0 <= args.part < args.parts:
            parser.error('part must be between zero and parts minus one')
        infer(args, cases)
    else:
        summarize(args, cases)


if __name__ == '__main__':
    main()
