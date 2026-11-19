"""P3-0: paired, causal evidence-content ablation before active tool selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import fixed_transition_p1 as p1
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.llm.providers import build_chat_model

ARMS = ('A', 'B', 'C', 'D', 'E')
SYSTEM = (p1.SYSTEM + ' Measurements are correlated and local RR or morphology proxies '
          'are not AF diagnoses. A low AF score alone cannot prove an event ended. '
          'Use UNCERTAIN when the available measurements do not resolve continuation '
          'versus transition. Cite only supplied evidence IDs.')


def selected(hard):
    for lead in ('ecg1', 'ecg2'):
        groups = ([(r, 'negative', cursors) for r, cursors in p1.HARD_NEGATIVE[lead].items()]
                  if hard else [(r, label, cursors) for r in p1.CASES
                                for label, cursors in p1.CASES[r].items()])
        for record, reference, cursors in groups:
            if not hard and record == '06453' and lead == 'ecg1':
                continue
            source = (p1.FULL_SOURCE / f'live-{lead}' / f'{lead}-event-adaptive' /
                      'records' / record if hard else p1.SOURCES / lead / 'records' / record)
            if not hard and record == '04126' and lead == 'ecg2':
                source = source.parent / '04126-clean'
            yield lead, record, reference, cursors, source


def _evidence(visible, tool, lead, start, end):
    return next((e for e in visible if e['tool'] == tool and e['lead'] == lead and
                 e['window_s'] == [start, end]), None)


def _compact_existing(e, alias):
    if e is None:
        return {'id': alias, 'status': 'not_measured'}
    result = e.get('result') or {}
    fields = ('af_score', 'afl_score') if e['tool'] == 'score_ecgfounder' else ('score', 'reasons')
    return {'id': alias, 'status': e['status'], 'lead': e['lead'],
            'window_s': e['window_s'], 'result': {k: result.get(k) for k in fields},
            'source_evidence_id': e['evidence_id']}


def _compact_new(result, alias):
    row = result.model_dump() if hasattr(result, 'model_dump') else result
    value = row.get('result') or {}
    fields = (('raw', 'cleaned', 'detected_beats', 'rr_intervals_ms')
              if row['tool'] == 'get_rate_rhythm' else ('detected_beats', 'beats', 'detector_error'))
    return {'id': alias, 'status': row['status'], 'lead': row['lead'],
            'window_s': row['window_s'], 'result': {k: value.get(k) for k in fields},
            'warnings': row.get('warnings', []), 'error': row.get('error'),
            'source_evidence_id': row['evidence_id']}


def prepare_case(raw_root, trace, evidence, cursor, lead, record, reference, source):
    recent, visible, _, _ = p1.view(trace, evidence, cursor)
    primary, backup = lead.upper(), ('ECG2' if lead == 'ecg1' else 'ECG1')
    case = {'record_id': record, 'lead': lead, 'cursor_s': cursor, 'reference': reference,
            'source': str(source), 'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ('window_trace.jsonl', 'evidence.jsonl')},
            'primary': [], 'backup': [], 'quality': [], 'rr': [], 'wave': []}
    for i, w in enumerate(recent, 1):
        a, b = w['start_s'], w['end_s']
        case['primary'].append(_compact_existing(_evidence(visible, 'score_ecgfounder', primary, a, b), f'E{i}'))
        case['backup'].append(_compact_existing(_evidence(visible, 'score_ecgfounder', backup, a, b), f'E{i+6}'))
        case['quality'].append(_compact_existing(_evidence(visible, 'assess_quality', primary, a, b), f'E{i+12}'))
    # Each tool sees at most the current or preceding arrived 10-second window.
    history = HistoricalWindow(raw_root, {'record_id': record, 'start_s': 0, 'end_s': cursor},
                               predict=lambda *_: None)
    for i, w in enumerate(recent[-2:], 1):
        a, b = w['start_s'], w['end_s']
        try:
            session = history.session(a, b, cursor, None, allow_shifted=True)
            args = {'lead': primary, 'start_s': a, 'end_s': b}
            case['rr'].append(_compact_new(session.call('get_rate_rhythm', args), f'E{i+18}'))
            case['wave'].append(_compact_new(session.call('get_beat_timeline_proxy', args), f'E{i+20}'))
        except Exception as exc:
            for key, offset in (('rr', 18), ('wave', 20)):
                case[key].append({'id': f'E{i+offset}', 'status': 'error', 'lead': primary,
                                  'window_s': [a, b], 'error': type(exc).__name__ + ': ' + str(exc)[:180]})
    return case


def payload(case, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    view = {'cursor_s': case['cursor_s'], 'primary_scores': case['primary']}
    if arm in ('B', 'E'):
        view['backup_scores'] = case['backup']
    if arm in ('C', 'E'):
        view['rr_measurements'] = case['rr']
    if arm in ('D', 'E'):
        view['quality_hints'] = case['quality']
        view['local_waveform_proxies'] = case['wave']
    return view


def judge(model, case, arm):
    view = payload(case, arm)
    aliases = {e['id']: e.get('source_evidence_id', e['id']) for value in view.values()
               if isinstance(value, list) for e in value if e['status'] not in ('not_measured', 'error')}
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(view, allow_nan=False) +
                 '\nValid citation IDs: ' + ', '.join(aliases) + '. Cite only these IDs.'}]
    started = time.perf_counter()
    errors = []
    for attempt in range(2):
        try:
            answer = p1.parse(model.complete(messages, [p1.SCHEMA]), aliases)
            return {'answer': answer, 'input_view': view, 'attempts': attempt + 1,
                    'visible_evidence_count': len(aliases),
                    'elapsed_s': round(time.perf_counter() - started, 3)}
        except Exception as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc)[:180])
            messages.append({'role': 'user', 'content': 'The prior output was invalid: ' + errors[-1] +
                             '. Submit one valid judge_transition action. Available IDs: ' +
                             ', '.join(aliases) + '. You may use an empty evidence_ids list.'})
    return {'error': 'ACTION_ERROR', 'error_detail': errors, 'input_view': view,
            'attempts': 2, 'visible_evidence_count': len(aliases),
            'elapsed_s': round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'infer'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hard-only', action='store_true')
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--lead', choices=('ecg1', 'ecg2'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        if not args.raw_root:
            parser.error('--raw-root is required for prepare')
        for lead, record, reference, cursors, source in selected(args.hard_only):
            trace = p1.read_jsonl(source / 'window_trace.jsonl')
            evidence = p1.read_jsonl(source / 'evidence.jsonl')
            for cursor in cursors:
                if not any(w['end_s'] == cursor for w in trace):
                    continue
                row = prepare_case(args.raw_root, trace, evidence, cursor, lead, record,
                                   reference, source)
                with (args.output / 'cases.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({'lead': lead, 'record': record, 'cursor': cursor,
                                  'rr': [x['status'] for x in row['rr']],
                                  'wave': [x['status'] for x in row['wave']]}), flush=True)
    else:
        if not args.model_path or not args.lead:
            parser.error('--model-path and --lead are required for infer')
        model = build_chat_model({'provider': 'transformers', 'model_path': str(args.model_path),
                                  'device': 'cuda:0', 'max_new_tokens': 512})
        for case in p1.read_jsonl(args.output / 'cases.jsonl'):
            if case['lead'] != args.lead:
                continue
            for arm in ARMS:
                row = {k: case[k] for k in ('record_id', 'lead', 'cursor_s', 'reference',
                                           'source', 'source_sha256')}
                row.update({'arm': arm, **judge(model, case, arm)})
                with (args.output / f'{arm}-{args.lead}.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({'lead': args.lead, 'record': case['record_id'],
                                  'cursor': case['cursor_s'], 'arm': arm,
                                  'decision': (row.get('answer') or {}).get('decision'),
                                  'error': row.get('error')}), flush=True)


if __name__ == '__main__':
    main()
