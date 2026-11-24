"""Prepare label-blind brief-event opportunities and replay their causal measurement policy."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, time_ns

import numpy as np

from cardioagent.common.io import digest, save_json, write_jsonl
from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.harness.brief_review import BriefCandidateHistory, LEADS, SHIFT_OPTIONS, review
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
from cardioagent.ecg_agent.schemas import EvidenceContext


def prepare(cache, scores, split_file, output):
    manifest = json.loads((cache/'manifest.json').read_bytes())
    source = json.loads((scores/'provenance.json').read_bytes())
    if digest(cache/'manifest.json') != source['cache_manifest_sha256']:
        raise ValueError('Score cache does not match the waveform manifest')
    split = json.loads(split_file.read_bytes())
    cohorts = dict(fit=split['fit_records'], held=split['held_records'], development=['04936'])
    names = [name for group in cohorts.values() for name in group]
    records = {rec['record_id']: rec for rec in manifest['records']}
    if len(set(names)) != len(names) or set(names) != records.keys():
        raise ValueError('Expected a complete disjoint AFDB partition')
    raw = Path(manifest['raw_root'])
    packets, receipts = [], {}
    for name in names:
        track = BriefCandidateHistory()
        record = records[name]
        index = cache/record['window_index']
        score_path = scores/f'{name}.npy'
        values = np.load(score_path, mmap_mode='r', allow_pickle=False)
        if tuple(values.shape) != (record['shape'][0], 2):
            raise ValueError('Score tensor does not match the waveform cache')
        with index.open(encoding='utf8') as stream:
            for one, two in zip(stream, stream, strict=True):
                pair = [json.loads(one), json.loads(two)]
                first, second = pair
                a, b = first['start_s'], first['end_s']
                if (tuple(row['channel'] for row in pair) != LEADS or
                        (a, b) != (second['start_s'], second['end_s'])):
                    raise ValueError('Misaligned paired input')
                scores_now = {row['channel']: float(max(values[row['waveform_row']]))
                              if row['status']=='ok' else None for row in pair}
                packet = track.observe(a, b, scores_now)
                if packet is not None:
                    packets.append(dict(record_id=name, packet=packet))
        receipts[name] = dict(seconds=record['source_record_samples']/record['source_rate_hz'],
                              dat=digest(raw/f'{name}.dat'), hea=digest(raw/f'{name}.hea'),
                              aligned_scores=digest(score_path), window_index=digest(index))
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output/'packets.jsonl', packets)
    save_json(output/'manifest.json', dict(cohorts=cohorts, source=source, records=receipts,
                                          split_sha256=digest(split_file), packets_sha256=digest(output/'packets.jsonl')))
    print(dict(Counter(r['record_id'] for r in packets)), flush=True)


def run(args):
    manifest = json.loads((args.input/'manifest.json').read_bytes())
    path = args.input/'packets.jsonl'
    if digest(path) != manifest['packets_sha256']:
        raise ValueError('Changed packet list')
    names = set(manifest['cohorts'][args.cohort])
    rows = [r for line in path.read_bytes().splitlines() if (r:=json.loads(line))['record_id'] in names]
    if args.shards <= 0 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard index')
    rows = rows[args.shard::args.shards]
    check = {r['record_id'] for r in rows}
    if args.mode != 'aligned':
        source = code_root()/'third_party'/'ecgfounder'
        for name, field in (('net1d.py','network_sha256'),('tasks.txt','labels_sha256')):
            if digest(source/name) != manifest['source'][field]:
                raise ValueError('Different Founder network or label order')
        for name in check:
            receipt = manifest['records'][name]
            if any(digest(args.raw/f'{name}.{ext}') != receipt[ext] for ext in ('dat', 'hea')):
                raise ValueError(f'Changed raw waveform for {name}')
        predict, model_info = create_ecgfounder_scorer(args.models, args.device)
        if digest(Path(model_info['checkpoint'])) != manifest['source']['checkpoint_sha256']:
            raise ValueError('Different Founder checkpoint')
    config_digest = digest(args.config) if args.mode != 'measure' else None
    job = dict(packets_sha256=digest(path), manifest_sha256=digest(args.input/'manifest.json'),
               config_sha256=config_digest, mode=args.mode, cohort=args.cohort, shard=args.shard,
               shards=args.shards, device=args.device, revision=args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output/'job.json'
    if receipt.exists():
        if json.loads(receipt.read_bytes()) != job:
            raise ValueError('Refusing to resume a different brief-event replay')
    else:
        save_json(receipt, job)
    chat = None
    if args.mode != 'measure':
        config = json.loads(args.config.read_bytes())
        config['device'] = args.device
        chat = LocalQwenModel(config)
    for entry in rows:
        name, packet = entry['record_id'], entry['packet']
        cursor = packet['window_s'][1]
        key = f'{name}-{cursor:g}'
        finished = args.output/(key+'.json')
        if finished.exists():
            old = json.loads(finished.read_bytes())
            fingerprint = old.pop('receipt_sha256')
            if (old.get('key') != key or old.get('job') != job or
                    sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() != fingerprint):
                raise ValueError('Damaged or foreign case receipt')
            if old.get('qwen_log') and digest(args.output/old['qwen_log']) != old['qwen_log_sha256']:
                raise ValueError('The case Qwen call log has changed')
            continue
        started = perf_counter()
        logs = []
        def emit(category, data):
            logs.append(dict(category=category, **data))
        def measure(lead, a, b, now, offset):
            if now != cursor:
                raise ValueError('Review attempted to change its cursor')
            return reader.shifted_score(a, b, now, lead, offset, context)
        if args.mode != 'aligned':
            reader = HistoricalWindow(args.raw, dict(record_id=name, start_s=0,
                                      end_s=manifest['records'][name]['seconds']), predict)
            context = EvidenceContext(f'brief-{args.revision}-{args.mode}', f'brief-{key}', name)
        qwen_log = None
        if args.mode == 'measure':
            a, b = packet['candidate_s']
            result = {option: measure(lead, a, b, cursor, offset)
                      for option, (lead, offset) in SHIFT_OPTIONS.items()}
        else:
            qwen_log = args.output/f'{key}.{time_ns()}.qwen.jsonl.gz'
            result = review(packet, args.mode, RecordingChatModel(chat, qwen_log), measure, emit)
        row = dict(key=key, record_id=name, job=job, window_s=packet['window_s'], result=result, logs=logs,
                   elapsed_s=perf_counter()-started, qwen_log=qwen_log.name if qwen_log else None,
                   qwen_log_sha256=digest(qwen_log) if qwen_log else None)
        row['receipt_sha256'] = sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        tmp = finished.with_suffix('.partial')
        save_json(tmp, row)
        tmp.replace(finished)
        print(key, result.get('decision',{}).get('decision') if args.mode!='measure' else 'measured', flush=True)
    save_json(args.output/'complete.json', dict(job=job, cases=len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('prepare')
    for key in ('cache', 'scores', 'split', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    r = actions.add_parser('run')
    for key in ('input','raw','models','output'):
        r.add_argument('--'+key, type=Path, required=True)
    r.add_argument('--config', type=Path)
    r.add_argument('--device', default='cpu')
    r.add_argument('--revision', required=True)
    r.add_argument('--mode', choices=['aligned','all','active','measure'], required=True)
    r.add_argument('--cohort', choices=['fit','held','development'], required=True)
    r.add_argument('--shards', type=int, default=1)
    r.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.cache, args.scores, args.split, args.output)
    else:
        if args.mode!='measure' and not args.config:
            parser.error('Qwen replay requires --config')
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        run(args)


if __name__ == '__main__':
    main()
