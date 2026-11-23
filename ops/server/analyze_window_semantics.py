"""Offline-only analysis for the paired 04936 window semantics pilot."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

MODES = ('legacy','masked','explicit','explicit_masked','single','verified')


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]


def group(window, intervals):
    start,end=window
    overlap=sum(max(0,min(end,b)-max(start,a)) for a,b in intervals)
    if overlap == 0:
        return 'outside'
    return 'inside' if overlap == end-start else 'boundary'


def main(root, source):
    reference=json.loads((source/'paired_results.json').read_text())['reference']['intervals']
    profile=json.loads((root/'calibration_audit.json').read_text())
    conditions={}
    cases={}
    comparisons={}
    for mode in (*MODES, *(m for m in ('union_raw','union_max','union_reference') if (root/m/'results.jsonl').exists())):
        results=rows(root/mode/'results.jsonl')
        signatures={(r['source'],r['cursor_s']):r['measurement_digest'] for r in results}
        if len(signatures)!=len(results) or (conditions and signatures!=comparisons):
            raise ValueError('Missing, duplicate or unpaired cases')
        comparisons=signatures
        steps=rows(root/mode/'steps.jsonl')
        errors=Counter(s.get('error_message') for s in steps if s['status']=='stage_rejected')
        counts=defaultdict(Counter)
        actions=defaultdict(Counter)
        model_calls=0
        for r in results:
            kind=group(r['window_s'],reference)
            obs=r['observation']
            interpretation=obs.get('interpretation','error')
            counts[kind][interpretation]+=1
            if r['cursor_s'] in (1590,1670,1810):
                intent=r['intent'] or {}
                actions[int(r['cursor_s'])][intent.get('action',intent.get('status','observation_error'))]+=1
            key=f"{r['source']}/{r['cursor_s']:g}"
            cases.setdefault(key,dict(group=kind, modes={} ))
            cases[key]['modes'][mode]=dict(observation=interpretation, reason=obs['reason'],intent=r['intent'])
            model_calls+=r['counts']['llm_calls']
        with gzip.open(root/mode/'qwen_calls.jsonl.gz','rt',encoding='utf8') as f:
            calls=[json.loads(s) for s in f]
        if len(calls)!=model_calls:
            raise ValueError('Call manifest differs from recorded calls')
        conditions[mode]=dict(groups={k:dict(c) for k,c in counts.items()}, actions=dict(actions), errors=errors,
                              calls=model_calls, generation_seconds=round(sum(c['elapsed_s'] for c in calls),2))
    baseline=defaultdict(Counter)
    for key,c in cases.items():
        arm,ts=key.split('/')
        case_dir=next((source/arm).glob('*/records/04936'))
        lead=arm.split('-')[-1].upper()
        candidates=[r for r in rows(case_dir/'evidence.jsonl') if r['tool']=='score_ecgfounder'
                    and r['status']=='ok' and r['lead']==lead and r['window_s']==[float(ts)-10,float(ts)]]
        if len(candidates)!=1:
            raise ValueError(f'Expected one target-window primary score for {key}')
        pair=candidates[0]['result']
        flag=profile['bins'][lead][bisect_right(profile['edges'],max(pair['af_score'],pair['afl_score']))]['direction']
        c.update(founder=pair, band_only=flag)
        baseline[c['group']][flag]+=1
    swaps={}
    for mode in ('union_raw','union_max'):
        folder=root/(mode+'_swap')
        if not folder.exists():
            continue
        originals={r['cursor_s']:r for r in rows(root/mode/'results.jsonl') if r['source']=='new-fixed-ecg2'}
        swapped=rows(folder/'results.jsonl')
        def first_assessments(path):
            by_time={}
            with gzip.open(path,'rt',encoding='utf8') as stream:
                for line in stream:
                    call=json.loads(line)
                    if call['tools'][0]['function']['name']=='assess_window':
                        by_time.setdefault(call['cursor_s'],call)
            return by_time
        before=first_assessments(root/mode/'qwen_calls.jsonl.gz')
        after=first_assessments(folder/'qwen_calls.jsonl.gz')
        swaps[mode]=[dict(cursor_s=r['cursor_s'],original=originals[r['cursor_s']]['observation'].get('interpretation','error'),
                         swapped=r['observation'].get('interpretation','error'),
                         input_identical=before[r['cursor_s']]['messages']==after[r['cursor_s']]['messages'],
                         raw_response_identical=before[r['cursor_s']]['raw']==after[r['cursor_s']]['raw']) for r in swapped]
    result=dict(subtype_swaps=swaps,reference_intervals=reference,paired_rows=len(cases),score_only=dict(baseline),
                conditions=conditions,cases=cases)
    (root/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    main(**vars(parser.parse_args()))
