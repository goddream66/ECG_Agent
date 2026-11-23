"""Fit descriptive AF/AFL frequency bins without the pilot patient, then audit held-out records."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

# Set before reviewing pilot outputs. These are empirical direction flags, not clinical probabilities.
EDGES = (.2, .5, .8)


def label(lo, hi, spans):
    overlap = sum(max(0., min(hi,b) - max(lo,a)) for a,b in spans)
    if overlap < 1e-7:
        return 0
    return 1 if abs(hi-lo-overlap) < 1e-7 else None


def collect(folder, cache, record):
    rid = record['record_id']
    scores = np.load(folder/(rid+'.npy'),allow_pickle=False)
    counts = np.zeros((2,4,2),dtype=int)
    for line in (cache/(rid+'.windows.jsonl')).read_text().splitlines():
        row = json.loads(line)
        if row['status'] != 'ok' or row['invalid_spans'] or row['waveform_row'] is None:
            continue
        lo,hi=row['start_s'],row['end_s']
        if not any(a <= lo and hi <= b for a,b in record['valid_intervals']):
            continue
        y = label(lo,hi,record['raw_annotation_af_intervals'])
        if y is None:
            continue
        pair=scores[row['waveform_row']]
        if pair.shape!=(2,) or not np.isfinite(pair).all():
            continue
        lead=('ECG1','ECG2').index(row['channel'])
        counts[lead,np.searchsorted(EDGES,float(max(pair)),side='right'),y] += 1
    return counts


def main(cache,scores,validity,output):
    fingerprint=sha256((cache/'manifest.json').read_bytes()).hexdigest()
    provenance=json.loads((scores/'provenance.json').read_text())
    if fingerprint!=provenance['cache_manifest_sha256']:
        raise ValueError('Cache and model score index differ')
    records={r['record_id']:r for r in json.loads(validity.read_text())['records'].values()}
    ids=sorted(set(records)-{'04936'})
    held=ids[::3]
    train=[r for r in ids if r not in held]
    counts={r:collect(scores,cache,records[r]) for r in ids}
    table={}
    validation={}
    for i,lead in enumerate(('ECG1','ECG2')):
        table[lead]=[]
        # Equal total mass per fitting record/lead, including all eligible windows.
        weighted=sum(counts[r][i]/max(1,counts[r][i].sum()) for r in train)
        raw=sum(counts[r][i] for r in train)
        test=sum(counts[r][i] for r in held)
        for j,(neg,pos) in enumerate(raw.tolist()):
            n_records=sum(bool(counts[r][i,j].sum()) for r in train)
            rate=weighted[j,1]/weighted[j].sum() if weighted[j].sum() else None
            band='indeterminate'
            if n_records>=3 and rate is not None:
                if rate<=.10: band='negative'
                if rate>=.90: band='support'
            table[lead].append(dict(negative=neg,positive=pos,records=n_records,rate=rate,
                                    direction=band,held_counts=test[j].tolist()))
        validation[lead]=dict(correct=sum(int(test[j,1 if t['direction']=='support' else 0])
                              for j,t in enumerate(table[lead]) if t['direction']!='indeterminate'),
                              decisive=sum(int(test[j].sum()) for j,t in enumerate(table[lead]) if t['direction']!='indeterminate'),
                              total=int(test.sum()))
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(dict(edges=EDGES,excluded='04936',fit_records=train,held_records=held,
                          bins=table,held_validation=validation,source=provenance,
                          validity_sha256=sha256(validity.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('cache','scores','validity','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    main(**vars(parser.parse_args()))
