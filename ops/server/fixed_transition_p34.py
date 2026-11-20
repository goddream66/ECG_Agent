"""P3-4: paired fixed-evidence Qwen comparison of original versus XQRS RR.

The preparation file has no reference labels. Answers are stored separately.
This is an exploratory AFDB decision-point test, not an online event run.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import wfdb

import evidence_reasoning_p31 as p31
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.tools.measurements import _stats, rr_clean_mask


ARMS = ('A', 'B', 'C')
SYSTEM = ("An AF/AFL episode was previously confirmed and is currently open. "
          "Judge whether it has ended by the current cursor. AF_AFL_ONGOING means "
          "evidence supports continuation, AF_AFL_ENDED means evidence supports "
          "termination, and UNCERTAIN means the available evidence does not settle it. "
          "Scores are uncalibrated. RR irregularity, a fall in RR variability, or a "
          "low model score alone cannot prove AF/AFL ended. Cite supplied evidence IDs "
          "supporting and opposing termination. No boundary time is requested.")


def case_id(case):
    return sha256(json.dumps([case['record_id'], case['lead'], case['cursor_s']],
                             separators=(',', ':')).encode()).hexdigest()[:16]


def rr_view(row, quality=False):
    result = row.get('result') or {}
    rr = result.get('rr_intervals_ms') or []
    raw = result.get('raw') or _stats(rr)
    out = {'id': row['id'], 'lead': row['lead'], 'window_s': row['window_s'],
           'status': row['status'], 'rr_intervals_ms': rr,
           'mean_rr_ms': raw['mean_ms'], 'rr_cv': raw['cv'],
           'detected_beats': result.get('detected_beats', len(rr) + 1 if rr else 0)}
    if quality:
        out['rr_cleaning_keep_mask'] = result.get('cleaning_keep_mask')
        out['removed_fraction'] = result.get('removed_fraction')
        out['detector_error'] = row.get('error')
    return out


def prepare_case(case, raw_root):
    lead = case['lead'].upper()
    cursor = float(case['cursor_s'])
    stem = str(raw_root / case['record_id'])
    header = wfdb.rdheader(stem)
    xqrs = []
    for i, window in enumerate(case['rr'], 1):
        target_start, target_end = window['window_s']
        origin = target_end - 30
        if origin < 0 or target_end > cursor:
            raise ValueError('XQRS requested future or unavailable history')
        first, last = round(origin * header.fs), round(target_end * header.fs)
        raw = wfdb.rdrecord(stem, sampfrom=first, sampto=last)
        if len(raw.p_signal) != last - first:
            raise ValueError('Incomplete elapsed waveform')
        wave = Waveform(raw.p_signal, raw.fs, tuple(raw.sig_name), tuple(raw.units))
        session = ToolSession(ToolContext(wave, origin, target_end), allow_experimental=True)
        measured = session.call('detect_beats_xqrs_proxy', {
            'lead': lead, 'start_s': origin, 'end_s': target_end})
        peak_times = [x for x in measured['result'].get('r_peaks_s', [])
                      if target_start <= x < target_end]
        rr = (np.diff(peak_times) * 1000).tolist()
        keep = rr_clean_mask(rr)
        xqrs.append({'id': f'R{i}', 'lead': lead, 'window_s': [target_start, target_end],
                     'status': measured['status'], 'error': measured['error'],
                     'result': {'rr_intervals_ms': rr, 'raw': _stats(rr),
                                'detected_beats': len(peak_times),
                                'cleaning_keep_mask': keep.tolist(),
                                'removed_fraction': float(np.mean(~keep)) if len(keep) else None}})
    primary = [{k: row.get(k) for k in ('id', 'status', 'lead', 'window_s', 'result')}
               for row in case['primary']]
    original_rr = [dict(row, id=f'R{i}') for i, row in enumerate(case['rr'], 1)]
    quality = [{k: row.get(k) for k in ('id', 'status', 'lead', 'window_s', 'result')}
               for row in case['quality']]
    return {'case_id': case_id(case), 'record_id': case['record_id'],
            'lead': lead, 'cursor_s': cursor, 'primary_scores': primary,
            'original_rr': original_rr, 'xqrs_rr': xqrs, 'quality_hints': quality}


def prompt_view(case, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    rr = case['original_rr'] if arm == 'A' else case['xqrs_rr']
    view = {'cursor_s': case['cursor_s'], 'primary_scores': case['primary_scores'],
            'rr_measurements': [rr_view(row, quality=arm == 'C') for row in rr],
            'quality_hints': case['quality_hints'] if arm == 'C' else []}
    return view


def infer_one(model, case, arm):
    return judge_view(model, prompt_view(case, arm))


def judge_view(model, view):
    allowed = {e['id'] for entries in view.values() if isinstance(entries, list)
               for e in entries if e['status'] == 'ok'}
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(view, allow_nan=False) +
                 '\nValid evidence IDs: ' + ', '.join(sorted(allowed))}]
    started = perf_counter()
    errors = []
    for attempt in range(2):
        try:
            answer = p31.validate(model.complete(messages, [p31.SCHEMA]), allowed)
            return {'answer': answer, 'attempts': attempt + 1,
                    'elapsed_s': perf_counter() - started}
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc)[:160])
            messages.append({'role': 'user', 'content': 'Invalid output: ' + errors[-1] +
                             '. Submit one valid judgment citing only valid IDs.'})
    return {'error': 'ACTION_ERROR', 'detail': errors, 'attempts': 2,
            'elapsed_s': perf_counter() - started}


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows),
                    encoding='utf-8')


def prepare(cases_root, raw_root, output):
    output.mkdir(parents=True, exist_ok=True)
    prepared, answer_key = [], []
    sources = {}
    for group in ('main', 'hard'):
        source = cases_root / group / 'cases.jsonl'
        sources[group] = sha256(source.read_bytes()).hexdigest()
        for line in source.read_text(encoding='utf-8').splitlines():
            case = json.loads(line)
            prepared.append(dict(prepare_case(case, raw_root), group=group))
            answer_key.append({'case_id': case_id(case), 'group': group,
                               'reference': case['reference']})
    if len({x['case_id'] for x in prepared}) != len(prepared):
        raise ValueError('Duplicate record/lead/cursor cases')
    write_jsonl(output / 'inputs.jsonl', prepared)
    write_jsonl(output / 'answer_key.jsonl', answer_key)
    (output / 'manifest.json').write_text(json.dumps({'sources_sha256': sources,
        'arms': {'A': 'ECGFounder + original 10s XQRS-derived RR',
                 'B': 'ECGFounder + 30s-context XQRS RR',
                 'C': 'B + signal and detector quality information'},
        'model_prompt_sha256': sha256(SYSTEM.encode()).hexdigest()}, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(prepared), 'output': str(output)}))


def infer(inputs, output, model_path, lead):
    output.mkdir(parents=True, exist_ok=True)
    model = build_chat_model({'provider': 'transformers', 'model_path': str(model_path),
                              'device': 'cuda:0', 'max_new_tokens': 512})
    source_hash = sha256(inputs.read_bytes()).hexdigest()
    for line in inputs.read_text(encoding='utf-8').splitlines():
        case = json.loads(line)
        if case['lead'].lower() != lead:
            continue
        for arm in ARMS:
            row = {'case_id': case['case_id'], 'group': case['group'], 'arm': arm,
                   'lead': lead, 'input_sha256': source_hash,
                   **infer_one(model, case, arm)}
            with (output / f'{lead}.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'case_id': case['case_id'], 'arm': arm,
                              'decision': (row.get('answer') or {}).get('decision'),
                              'error': row.get('error')}), flush=True)


def summarize(inputs, key, predictions):
    cases = {row['case_id']: row for row in map(json.loads, inputs.read_text().splitlines())}
    answers = {row['case_id']: row for row in map(json.loads, key.read_text().splitlines())}
    predictions_by_arm = defaultdict(list)
    for path in predictions:
        for row in map(json.loads, path.read_text().splitlines()):
            if row['case_id'] not in cases or row['case_id'] not in answers:
                raise ValueError('Unknown case in predictions')
            predictions_by_arm[row['arm']].append(row)
    output = {}
    for arm, rows in predictions_by_arm.items():
        if len({x['case_id'] for x in rows}) != len(rows):
            raise ValueError('Duplicate prediction for ' + arm)
        detail = Counter()
        for row in rows:
            expected = answers[row['case_id']]['reference']
            actual = (row.get('answer') or {}).get('decision', 'ACTION_ERROR')
            detail[(expected, actual)] += 1
        positive = sum(n for (ref, _), n in detail.items() if ref == 'positive')
        negative = sum(n for (ref, _), n in detail.items() if ref == 'negative')
        output[arm] = {'cases': len(rows), 'confusion': [
            {'reference': expected, 'decision': actual, 'count': n}
            for (expected, actual), n in sorted(detail.items())],
            'true_end_recall': detail[('positive', 'AF_AFL_ENDED')] / positive if positive else None,
            'false_end_rate': detail[('negative', 'AF_AFL_ENDED')] / negative if negative else None,
            'uncertain_rate': sum(n for (_, decision), n in detail.items()
                                  if decision == 'UNCERTAIN') / len(rows) if rows else None,
            'action_error_rate': sum(n for (_, decision), n in detail.items()
                                     if decision == 'ACTION_ERROR') / len(rows) if rows else None,
            'llm_calls': sum(row.get('attempts', 0) for row in rows),
            'total_llm_elapsed_s': sum(row.get('elapsed_s', 0) for row in rows)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'infer', 'summarize'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases-root', type=Path)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--answer-key', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--lead', choices=('ecg1', 'ecg2'))
    parser.add_argument('--predictions', type=Path, nargs='*')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.cases_root or not args.raw_root:
            parser.error('prepare requires --cases-root and --raw-root')
        prepare(args.cases_root, args.raw_root, args.output)
    elif args.mode == 'infer':
        if not args.inputs or not args.model_path or not args.lead:
            parser.error('infer requires --inputs, --model-path and --lead')
        infer(args.inputs, args.output, args.model_path, args.lead)
    else:
        if not args.inputs or not args.answer_key or not args.predictions:
            parser.error('summarize requires --inputs, --answer-key and --predictions')
        args.output.mkdir(parents=True, exist_ok=True)
        results = summarize(args.inputs, args.answer_key, args.predictions)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(results))


if __name__ == '__main__':
    main()
