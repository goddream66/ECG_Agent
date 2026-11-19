"""Causal fixed-evidence AF transition probe; evaluation labels never enter prompts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from cardioagent.ecg_agent.llm.providers import build_chat_model

SOURCES = Path('/data/1shared/gaojiajun/ECGagent/runs/afdb-end-phase1-20261119/B')
CASES = {
    '04126': {'positive': (560, 570, 580, 590), 'negative': (500, 520, 540)},
    '04936': {'positive': (1800, 1810, 1820), 'negative': (1700, 1750, 1780)},
    '06453': {'positive': (3080, 3090), 'negative': (3060, 3070)},
}
SCHEMA = {'type': 'function', 'function': {'name': 'judge_transition',
    'description': 'Judge whether an already active AF/AFL event has transitioned.',
    'parameters': {'type': 'object', 'properties': {
        'decision': {'type': 'string', 'enum': ['CONTINUE', 'TRANSITION', 'UNCERTAIN']},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'}},
        'required': ['decision', 'evidence_ids', 'reason'], 'additionalProperties': False}}}
SEARCH = {'type': 'function', 'function': {'name': 'retrieve_prior_evidence',
    'description': 'Retrieve earlier, already arrived evidence before the recent six windows.',
    'parameters': {'type': 'object', 'properties': {
        'focus': {'type': 'string', 'enum': ['last_support', 'prior_negative', 'prior_conflict', 'recent_older']},
        'reason': {'type': 'string'}}, 'required': ['focus', 'reason'],
        'additionalProperties': False}}}
SYSTEM = ('A confirmed AF/AFL event is active. Judge CONTINUE when evidence supports '
          'ongoing AF/AFL, TRANSITION when the available temporal evidence supports a '
          'change away from AF/AFL, or UNCERTAIN when unresolved. A low ECGFounder score '
          'alone is not a rhythm diagnosis. Do not infer from missing measurements. '
          'Cite only supplied short evidence IDs. No end timestamp is requested.')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def compact(e, alias):
    result = e['result']
    if e['tool'] == 'score_ecgfounder':
        result = {k: result.get(k) for k in ('af_score', 'afl_score')}
    elif e['tool'] == 'assess_quality':
        result = {k: result.get(k) for k in ('score', 'reasons')}
    else:
        result = {k: v for k, v in result.items() if k in (
            'raw', 'cleaned', 'detected_beats', 'r_peaks_s', 'rr_intervals_ms',
            'p_candidate_fraction', 'mean_occupancy_ms', 'detector_error')}
    return {'id': alias, 'tool': e['tool'], 'lead': e['lead'],
            'window_s': e['window_s'], 'status': e['status'], 'result': result}


def view(trace, evidence, cursor):
    arrived = [w for w in trace if w['end_s'] <= cursor]
    if not arrived or arrived[-1]['end_s'] != cursor:
        raise ValueError('Cursor must be an arrived window end')
    recent = arrived[-6:]
    first = recent[0]['start_s']
    visible = [e for e in evidence if e['available_at_s'] <= cursor and
               first <= e['window_s'][0] and e['window_s'][1] <= cursor]
    older = [e for e in evidence if e['available_at_s'] <= cursor and
             e['window_s'][1] <= first]
    aliases = {f'E{i}': e['evidence_id'] for i, e in enumerate(visible, 1)}
    return recent, visible, older, aliases


def sequence(recent, visible, aliases):
    id_by_raw = {v: k for k, v in aliases.items()}
    out = []
    for w in recent:
        records = [e for e in visible if e['window_s'] == [w['start_s'], w['end_s']]]
        scores = {e['lead']: {'max_af_afl': round(max(e['result']['af_score'],
                        e['result']['afl_score']), 4), 'id': id_by_raw[e['evidence_id']]}
                  for e in records if e['tool'] == 'score_ecgfounder' and e['status'] == 'ok'}
        qc = {e['lead']: {'reasons': e['result'].get('reasons', []),
                         'id': id_by_raw[e['evidence_id']]}
              for e in records if e['tool'] == 'assess_quality' and e['status'] == 'ok'}
        out.append({'window_s': [w['start_s'], w['end_s']], 'scores': scores,
                    'quality': qc, 'interpretation': w['decision'],
                    'evidence_status': 'measured' if records else 'not_measured'})
    return out


def structured(recent, visible, aliases):
    timeline = sequence(recent, visible, aliases)
    def run_end(kind):
        selected = []
        for row in reversed(timeline):
            if row['interpretation'] != kind:
                break
            selected.append(row['window_s'])
        return [selected[-1][0], selected[0][1]] if selected else None
    return {'timeline': timeline, 'consecutive_support_s': run_end('support'),
            'consecutive_negative_s': run_end('negative'),
            'last_support_s': next((x['window_s'][1] for x in reversed(timeline)
                                    if x['interpretation'] == 'support'), None),
            'first_negative_s': next((x['window_s'][0] for x in timeline
                                      if x['interpretation'] == 'negative'), None),
            'conflict_windows': [x['window_s'] for x in timeline
                                 if x['interpretation'] == 'conflict'],
            'unknown_windows': [x['window_s'] for x in timeline
                                if x['interpretation'] == 'unavailable']}


def parse(reply, aliases):
    calls = reply.message.get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'judge_transition':
        raise ValueError('Expected exactly one transition judgment')
    raw = calls[0]['function']['arguments']
    a = json.loads(raw) if isinstance(raw, str) else raw
    if set(a) != {'decision', 'evidence_ids', 'reason'} or a['decision'] not in (
            'CONTINUE', 'TRANSITION', 'UNCERTAIN') or not isinstance(a['reason'], str) or not a['reason']:
        raise ValueError('Invalid decision fields')
    if not isinstance(a['evidence_ids'], list) or len(set(a['evidence_ids'])) != len(a['evidence_ids']) or not set(a['evidence_ids']) <= aliases.keys():
        raise ValueError('Unavailable or duplicate evidence ID')
    return {**a, 'resolved_evidence_ids': [aliases[k] for k in a['evidence_ids']]}


def retrieve(older, focus, limit=8):
    rows = [e for e in older if e['tool'] == 'score_ecgfounder']
    if focus == 'last_support':
        rows = [e for e in rows if e['status'] == 'ok' and
                max(e['result']['af_score'], e['result']['afl_score']) >= .6]
    elif focus == 'prior_negative':
        rows = [e for e in rows if e['status'] == 'ok' and
                max(e['result']['af_score'], e['result']['afl_score']) < .4]
    elif focus == 'prior_conflict':
        by_window = {}
        for e in rows: by_window.setdefault(tuple(e['window_s']), []).append(e)
        rows = [e for group in by_window.values() if len(group) >= 2 and
                any(max(x['result']['af_score'], x['result']['afl_score']) >= .6 for x in group) and
                any(max(x['result']['af_score'], x['result']['afl_score']) < .4 for x in group)
                for e in group]
    return rows[-limit:]


def judge(model, arm, trace, evidence, cursor):
    recent, visible, older, aliases = view(trace, evidence, cursor)
    payload = ({'recent_windows': [{k:w.get(k) for k in
                    ('start_s','end_s','af_score','afl_score','decision')}
                    for w in recent],
                'evidence': [compact(e, a) for a, e in zip(aliases, visible)]}
               if arm == 'A' else structured(recent, visible, aliases))
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps({'cursor_s': cursor, 'view': payload}, allow_nan=False)}]
    retrieved = []
    started = time.perf_counter()
    try:
        if arm == 'C':
            choice = model.complete(messages, [SEARCH, SCHEMA])
            calls = choice.message.get('tool_calls', [])
            if len(calls) != 1: raise ValueError('Expected one retrieval or judgment')
            fn = calls[0]['function']
            if fn['name'] == 'retrieve_prior_evidence':
                raw = fn['arguments']; args = json.loads(raw) if isinstance(raw, str) else raw
                if set(args) != {'focus', 'reason'} or args['focus'] not in ('last_support','prior_negative','prior_conflict','recent_older'):
                    raise ValueError('Invalid retrieval request')
                retrieved = retrieve(older, args['focus'])
                for e in retrieved: aliases[f'E{len(aliases)+1}'] = e['evidence_id']
                id_by_raw = {v:k for k,v in aliases.items()}
                messages.append({'role':'assistant','content':None,'tool_calls':calls})
                messages.append({'role':'tool','tool_call_id':calls[0].get('id','lookup'),
                    'content':json.dumps([compact(e,id_by_raw[e['evidence_id']]) for e in retrieved],allow_nan=False)})
                reply = model.complete(messages,[SCHEMA])
            elif fn['name'] == 'judge_transition': reply = choice
            else: raise ValueError('Unknown requested action')
        else: reply = model.complete(messages,[SCHEMA])
        answer = parse(reply, aliases)
        return {'answer': answer, 'retrieved_count': len(retrieved), 'visible_count':len(visible),
                'elapsed_s': round(time.perf_counter()-started,3)}
    except Exception as exc:
        return {'error':type(exc).__name__, 'error_detail':str(exc)[:250],
                'retrieved_count':len(retrieved), 'visible_count':len(visible),
                'elapsed_s':round(time.perf_counter()-started,3)}


def rule(trace, cursor):
    rows=[w for w in trace if w['end_s']<=cursor][-2:]
    if len(rows)==2 and all(w['decision']=='negative' for w in rows): return 'TRANSITION'
    if any(w['decision'] in ('conflict','unavailable') for w in rows): return 'UNCERTAIN'
    return 'CONTINUE'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--lead',choices=('ecg1','ecg2'),required=True)
    p.add_argument('--records',nargs='+',choices=tuple(CASES),default=list(CASES))
    p.add_argument('--arms',nargs='+',choices=('A','B','C'),default=['A','B','C'])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    model=build_chat_model({'provider':'transformers','model_path':str(args.model_path),
                            'device':'cuda:0','max_new_tokens':512})
    outputs={arm:(args.output/f'{arm}-{args.lead}.jsonl') for arm in args.arms}
    for rec in args.records:
        source=SOURCES/args.lead/'records'/rec
        if rec=='04126' and args.lead=='ecg2': source=source.parent/'04126-clean'
        trace=read_jsonl(source/'window_trace.jsonl'); evidence=read_jsonl(source/'evidence.jsonl')
        hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest()
                for n in ('window_trace.jsonl','evidence.jsonl')}
        for label,cursors in CASES[rec].items():
            for cursor in cursors:
                if not any(w['end_s']==cursor for w in trace):continue
                if rec=='06453' and args.lead=='ecg1':continue  # no confirmed event in source
                item={'record_id':rec,'lead':args.lead,'cursor_s':cursor,'reference':label,
                      'source':str(source),'source_sha256':hashes,
                      'rule':rule(trace,cursor)}
                for arm in args.arms:
                    row={**item,'arm':arm,**judge(model,arm,trace,evidence,cursor)}
                    with outputs[arm].open('a',encoding='utf-8') as f:
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps({'record_id':rec,'lead':args.lead,'cursor_s':cursor,
                        'arm':arm,'decision':row.get('answer',{}).get('decision'),
                        'error':row.get('error')}),flush=True)


if __name__=='__main__': main()
