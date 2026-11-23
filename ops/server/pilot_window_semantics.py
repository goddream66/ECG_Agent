"""Paired 04936 pilot. Reuse archived measurement-stage outputs; never replay a full waveform."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from cardioagent.ecg_agent.harness import staged_review as staged
from cardioagent.ecg_agent.harness.measured_event import numeric_fields
from cardioagent.ecg_agent.harness.window_semantics import MODES, current_founder
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.llm.recording import RecordingChatModel

# Frozen before inference. Two boundary-straddling windows are reported separately.
CURSORS = (1560, 1590, 1660, 1670, 1750, 1800, 1810, 1840)
CASES = {'new-fixed-ecg2': CURSORS, 'new-autonomous-ecg1': CURSORS,
         'new-autonomous-ecg2': (1590, 1810)}
LIFECYCLE_CHECKS = (1590, 1670, 1810)


def frozen_inputs(folder, cursors):
    cases = {}
    with gzip.open(folder/'qwen_calls.jsonl.gz', 'rt') as stream:
        for line in stream:
            call = json.loads(line)
            cursor = call['cursor_s']
            if cursor not in cursors or cursor in cases:
                continue
            if len(call['messages']) == 2 and call['tools'][0]['function']['name'] == 'assess_window':
                cases[cursor] = call
    if set(cases) != set(cursors):
        raise ValueError('Missing original first-attempt assessment')
    return cases


def resolve_ids(data, ledger):
    result = {}
    end = data['cursor_s']
    current = [r for r in ledger if r['observed_at_s'] == end]
    for row in data['measurements']:
        alias = row['id']
        possible = ([current[int(alias[1:])-1]] if alias.startswith('E') else ledger)
        matches = {r['evidence_id'] for r in possible if r['available_at_s'] <= end and
                   r['observed_at_s'] <= end and r['status'] == row['status'] and r['lead'] == row['lead']
                   and r['tool'] == row['tool'] and r['window_s'] == row['window_s']
                   and numeric_fields(r) == row['fields']}
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve {alias} at {end} ({len(matches)})')
        result[alias] = matches.pop()
    return result


def main(source, output, config, mode):
    out = output/mode
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(config.read_text())
    model = RecordingChatModel(build_chat_model(cfg), out/'qwen_calls.jsonl.gz')
    with (out/'results.jsonl').open('x') as results, (out/'steps.jsonl').open('x') as steps:
        def emit(kind, obj):
            steps.write(json.dumps(obj)+'\n')
            steps.flush()
        for name, cursors in CASES.items():
            folder = source/name/(name.split('-')[-1]+'-event-react')/'records/04936'
            evidence = [json.loads(s) for s in (folder/'evidence.jsonl').read_text().splitlines()]
            frozen = frozen_inputs(folder, cursors)
            for cursor in cursors:
                call = frozen[cursor]
                data = json.loads(call['messages'][1]['content'])
                rows = {r['id']:r for r in data['measurements']}
                aliases = resolve_ids(data, evidence)
                counts = Counter()
                audit = dict(source=name, cursor_s=cursor)
                obs = staged.assess(model, counts, emit, audit, data, aliases,
                                    lambda k: current_founder(rows[k], data['window_s']),
                                    lambda k: k in aliases, mode, cfg.get('window_band_profile'))
                intent = None
                if cursor in LIFECYCLE_CHECKS and obs['status'] == 'accepted':
                    intent = staged.choose_action(model, counts, emit, audit, data['window_s'],
                                                  obs, data['open_event'], data['candidate'])
                digest = hashlib.sha256(json.dumps(data['measurements'],sort_keys=True).encode()).hexdigest()
                result = dict(**audit, mode=mode, measurement_digest=digest, window_s=data['window_s'],
                              observation=obs, intent=intent, counts=dict(counts))
                results.write(json.dumps(result)+'\n')
                results.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('source','output','config'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    main(**vars(parser.parse_args()))
