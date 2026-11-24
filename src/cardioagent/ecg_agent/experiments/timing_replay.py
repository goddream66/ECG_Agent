"""Paired Qwen replay for the tool-runtime discrepancy at the 05091 transition."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.harness.window_semantics import verify_band


def read_calls(root):
    path = root/'05091-10350/output/ecg2-event-react/records/05091/qwen_calls.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf8') as stream:
        calls = [row for line in stream if (row:=json.loads(line))['cursor_s']==10370
                 and row['tools'][0]['function']['name']=='assess_window']
    if len(calls)!=2:
        raise ValueError('Both archived assessment attempts are required')
    return calls


def without_runtime(call):
    messages = deepcopy(call['messages'])
    data = json.loads(messages[1]['content'])
    rows = [r for r in data['other_measurements'] if r['tool']=='assess_quality']
    if len(rows)!=1:
        raise ValueError('Expected one quality measurement')
    runtime = rows[0]['fields'].pop('duration_ms')
    messages[1]['content'] = json.dumps(data,allow_nan=False)
    return messages, runtime


def replace_feedback(messages, profile, prior_call, raw_scores):
    data=json.loads(messages[1]['content'])
    rows={}
    for r in data['vote_on_this_interval']:
        if r['tool']=='score_ecgfounder':
            fields=raw_scores[tuple(r['window_s']),r['lead']]
            if max(fields['af_score'],fields['afl_score']) != r['fields']['target_score']:
                raise ValueError('Archived score does not match its raw ledger measurement')
            rows[r['id']]=dict(r,fields=fields)
    previous=json.loads(prior_call['message']['tool_calls'][0]['function']['arguments'])
    if previous['finding']!='TARGET_PRESENT' or len(messages)!=3:
        raise ValueError('Unexpected archived repair context')
    try:
        verify_band(dict(interpretation='support', evidence_ids=previous['evidence_ids']),
                    rows,data['window_s'],profile)
    except ValueError as exc:
        updated=deepcopy(messages)
        updated[-1]['content']=f'Rejected assess_window: {str(exc)[:500]}. Correct this stage only.'
        return updated
    raise ValueError('The original first response must be invalid under the same validation semantics')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('control','compact','output'):
        parser.add_argument('--'+arg,required=True,type=Path)
    parser.add_argument('--repair-feedback',action='store_true')
    args=parser.parse_args()
    control,compact=read_calls(args.control),read_calls(args.compact)
    config=json.loads((args.control/'agent_config.json').read_bytes())
    raw_scores={}
    if args.repair_feedback:
        evidence=args.control/'05091-10350/output/ecg2-event-react/records/05091/evidence.jsonl'
        for line in evidence.read_text().splitlines():
            row=json.loads(line)
            if row['tool']=='score_ecgfounder' and row['status']=='ok' and row['available_at_s']<=10370:
                raw_scores[tuple(row['window_s']),row['lead']]=row['result']
    model=LocalQwenModel(config)
    with args.output.open('x',encoding='utf8') as stream:
        for attempt,(a,b) in enumerate(zip(control,compact,strict=True),1):
            if args.repair_feedback and attempt != 2:
                continue
            filtered,ta=without_runtime(a)
            other,tb=without_runtime(b)
            if filtered!=other or a['tools']!=b['tools'] or ta==tb:
                raise ValueError('The two archived requests must differ only in tool runtime')
            views={'control_timing':a['messages'],'compact_timing':b['messages'],'no_timing':filtered}
            if args.repair_feedback:
                views={key: replace_feedback(messages,config['window_band_profile'],control[0],raw_scores)
                       for key,messages in views.items()}
            order=['control_timing','compact_timing','no_timing','no_timing','compact_timing','control_timing']
            for key in order:
                began=perf_counter()
                try:
                    answer=model.complete(views[key],a['tools'])
                    outcome=dict(message=answer.message,usage=answer.usage)
                except ValueError as exc:
                    outcome=dict(parse_error=str(exc))
                result=dict(attempt=attempt,condition=key,repair_feedback=args.repair_feedback,tool_duration_ms={'control_timing':ta,
                    'compact_timing':tb,'no_timing':None}[key],elapsed_s=perf_counter()-began,
                    raw=model.last_raw,**outcome)
                stream.write(json.dumps(result)+'\n')
                stream.flush()


if __name__=='__main__':
    main()
