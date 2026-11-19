"""Summarize P1 fixed-evidence decisions without using labels in inference."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path, arm):
    return [json.loads(s) for f in path.glob(f'{arm}-*.jsonl')
            for s in f.read_text(encoding='utf-8').splitlines() if s.strip()]


def summarize(rows, *, method):
    positive = [r for r in rows if r['reference'] == 'positive']
    negative = [r for r in rows if r['reference'] in ('negative','hard_negative')]
    def decision(r):
        return r['rule'] if method == 'rule' else (r.get('answer') or {}).get('decision','ERROR')
    valid = sum(decision(r) in ('CONTINUE','TRANSITION','UNCERTAIN') for r in rows)
    return {'cases':len(rows),'positive':len(positive),'negative':len(negative),
            'valid':valid,'transition_tp':sum(decision(r)=='TRANSITION' for r in positive),
            'transition_fp':sum(decision(r)=='TRANSITION' for r in negative),
            'transition_recall':(sum(decision(r)=='TRANSITION' for r in positive)/len(positive)
                                 if positive else None),
            'false_transition_rate':(sum(decision(r)=='TRANSITION' for r in negative)/len(negative)
                                     if negative else None),
            'decisions':dict(Counter(decision(r) for r in rows)),
            'retrieval_calls':sum(r.get('retrieved_count',0)>0 for r in rows)
                               if method != 'rule' else 0,
            'elapsed_s_sum':round(sum(r.get('elapsed_s',0) for r in rows),2)
                             if method != 'rule' else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--hard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={}
    for name,path in (('selected_endings',a.main),('transient_low_af',a.hard)):
        arm_rows={arm:read(path,arm) for arm in 'ABC'}
        keys=[{(r['record_id'],r['lead'],r['cursor_s'],r['reference']) for r in rows}
              for rows in arm_rows.values()]
        if not keys[0] or any(k!=keys[0] for k in keys[1:]):
            raise ValueError(f'Missing or unpaired cases in {name}')
        result[name]={arm:summarize(rows,method=arm) for arm,rows in arm_rows.items()}
        result[name]['rule']=summarize(arm_rows['A'],method='rule')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
