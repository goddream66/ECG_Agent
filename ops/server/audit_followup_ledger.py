"""Replay archived agent actions and resolve the measurements for each accepted CLOSE."""
import argparse
import json
from pathlib import Path

from cardioagent.ecg_agent.memory.staged_events import StagedEvents


def rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def audit(directory):
    original = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    traces = list(rows(directory / 'window_trace.jsonl'))
    evidence = {r['evidence_id']: r for r in rows(directory / 'evidence.jsonl')}
    tracker = StagedEvents(original['review_mode'], traces[0]['start_s'])
    for window in traces:
        start, end = window['start_s'], window['end_s']
        if window['decision'] in ('ACTION_ERROR', 'UNREVIEWED'):
            tracker.skip(start, end, retry=window['decision'] == 'ACTION_ERROR')
        else:
            tracker.apply(start, end, window['llm_action'])
    events = tracker.finish(traces[-1]['end_s'])
    if len(events) != len(original['events']) or tracker.intervals != original['intervals']:
        raise ValueError(f'Changed event export in {directory}')
    proofs = []
    for recorded, replayed in zip(original['events'], events):
        for key, value in recorded.items():
            if key == 'end_evidence_ids':
                if not set(value) <= set(replayed[key]):
                    raise ValueError('Lost a previously recorded citation')
            elif replayed[key] != value:
                raise ValueError(f'Changed {key} in {directory}')
        if replayed['final_status'] != 'CLOSED':
            continue
        anchors = replayed['end_evidence_by_role']
        measurements = {}
        previous_end = None
        for role in ('last_support', 'first_against', 'confirming_against'):
            key = anchors[role]
            r = evidence[key]
            a, b = r['window_s']
            if (r['tool'] != 'score_ecgfounder' or r['status'] != 'ok'
                    or r['available_at_s'] > replayed['end_decided_at_s']
                    or previous_end is not None and previous_end > a):
                raise ValueError(f'Invalid close citation {key}')
            measurements[role] = dict(evidence_id=key, window_s=[a,b], lead=r['lead'],
                                      available_at_s=r['available_at_s'])
            previous_end = b
        proofs.append(dict(event_id=replayed['event_id'], decided_at_s=replayed['end_decided_at_s'],
                           estimated_end_s=replayed['estimated_end_s'], measurements=measurements))
    return dict(windows=len(traces), events=len(events), close_proofs=proofs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {d.name: audit(d) for d in args.directories}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
