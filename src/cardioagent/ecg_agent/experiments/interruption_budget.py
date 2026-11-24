"""Choose one arrived window for a waveform measurement under a fixed review budget."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter, time_ns

import wfdb

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.harness.staged_review import TEXT, ask, choice, schema
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from .subwindow_change import FEATURES, measure_lead


def choose_window(item, model, emit):
    if set(item) != {'record_id', 'cursor', 'history', 'eligible'}:
        raise ValueError('Expected a waveform-inference packet')
    cursor = item['cursor']
    history = item['history']
    if (type(cursor) is not int or cursor < 70 or len(history) != 6 or
            any(a != cursor-60+10*i or b != a+10 or len(v) != 2 or
                any(x is not None and (type(x) not in (int, float) or
                                        not isfinite(x) or not 0 <= x <= 1) for x in v)
                for i,(a,b,v) in enumerate(history))):
        raise ValueError('Invalid arrived score sequence')
    indices = item['eligible']
    if (not indices or any(type(i) is not int or not 2 <= i < 6 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Invalid review window choices')
    options = {f'W{i-1}':history[i][:2] for i in indices}
    slides = {i: f'W{i-1}' for i in indices}
    lead_history = {
        lead: [dict(time_s=[a,b], score=round(v[j],4) if v[j] is not None else None,
                    **({'candidate':slides[i]} if i in slides else {}))
               for i,(a,b,v) in enumerate(history)]
        for j,lead in enumerate(('ECG1','ECG2'))
    }
    data = dict(window_s=[cursor-10,cursor], cursor_s=cursor, lead_history=lead_history)
    tool = schema('measure_one_window', 'Spend the one waveform review on the chosen candidate.',
                  dict(candidate=choice(options), reason=TEXT), ['candidate','reason'])

    def validate(args):
        key = args['candidate']
        if key not in options:
            raise ValueError('Choose an offered candidate')
        return dict(status='accepted', candidate=key, interval_s=options[key], reason=args['reason'])

    counts = Counter()
    one = ask(model, counts, emit, dict(cursor_s=cursor), data, tool,
              'The candidate windows were observed while the tracker was open. '
              'You can check the waveform in exactly one earlier window, to find a possible short pause in AF/AFL. '
              'Compare marked candidates and choose where a waveform check has the best chance of revealing '
              'a real interruption. Consider signal patterns preceding and following a candidate, when present. '
              'Each score measures its entire ten-second interval on its named lead. Compare temporal changes '
              'within one lead and agreement across leads at the same interval. A missing score supplies no direction. '
              'The tool result will provide two-second beat measurements. Your selection is a review request. '
              'Give a short reason based on the observed scores.',
              validate)
    return one, dict(counts)


def read_wave(raw, name, start, end, cursor):
    if not name.isdecimal() or not 20 <= start < end <= cursor or end-start != 10:
        raise ValueError('Waveform request outside arrived window')
    wave = wfdb.rdrecord(str(raw/name), sampfrom=(start-20)*128, sampto=end*128)
    if wave.fs != 128 or wave.sig_name != ['ECG1','ECG2']:
        raise ValueError('Incompatible LTAFDB waveform')
    return {lead: measure_lead(x,128) for lead,x in zip(wave.sig_name,wave.p_signal.T)}


def receipt(obj):
    return sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('input','raw','output','config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        raise ValueError('Invalid shard')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = json.loads(a.input.read_bytes())
    if set(source) != {'records','packets'}:
        raise ValueError('Invalid input bundle')
    cases = source['packets'][a.shard::a.shards]
    a.output.mkdir(parents=True,exist_ok=True)
    job = dict(revision=a.revision, input=digest(a.input), config=digest(a.config),
               shard=a.shard, shards=a.shards)
    for name in {item['record_id'] for item in cases}:
        if not name.isdecimal() or any(digest(a.raw/(name+'.'+ext)) != source['records'][name][ext]
                                       for ext in ('hea','dat')):
            raise ValueError('Waveform receipt mismatch')
    model = LocalQwenModel(dict(json.loads(a.config.read_bytes()),device='cuda:0'))
    for row in cases:
        name,cursor = row['record_id'],row['cursor']
        key = f'{name}-{cursor}'
        path = a.output/(key+'.json')
        if path.exists():
            old = json.loads(path.read_bytes())
            original = old.pop('receipt')
            if (receipt(old) != original or old['job'] != job or old['key'] != key or
                    digest(a.output/old['log']) != old['log_sha256']):
                raise ValueError('Invalid prior review result')
            continue
        log = a.output/f'{key}.{time_ns()}.jsonl.gz'
        steps = []
        stamp = perf_counter()
        decision,counts = choose_window(row, RecordingChatModel(model,log),
                                        lambda tag,line: steps.append([tag,line]))
        model_wall_s = perf_counter()-stamp
        measurement = None
        if decision['status'] == 'accepted':
            start,end = decision['interval_s']
            stamp = perf_counter()
            try:
                beats = read_wave(a.raw,name,start,end,cursor)
                status = 'ok' if any(any(v is not None for v in lead) for lead in beats.values()) else 'insufficient'
                measurement = dict(status=status, features=FEATURES, leads=beats)
            except (ValueError,RuntimeError,OSError) as error:
                measurement = dict(status='error', reason=str(error))
            measurement.update(interval_s=[start,end], available_at_s=cursor, wall_s=perf_counter()-stamp)
        if not log.exists():
            with gzip.open(log,'wt',encoding='utf8') as f:
                f.write(json.dumps(dict(error='No model trace',key=key))+'\n')
        saved = dict(key=key, job=job, decision=decision, counts=counts, steps=steps,
                     measurement=measurement, model_wall_s=model_wall_s, log=log.name, log_sha256=digest(log))
        saved['receipt'] = receipt(saved)
        tmp = path.with_suffix('.partial')
        save_json(tmp,saved)
        tmp.replace(path)
        print(key, decision.get('candidate','ACTION_ERROR'), flush=True)
    save_json(a.output/'complete.json',dict(job=job,cases=len(cases)))


if __name__ == '__main__':
    main()
