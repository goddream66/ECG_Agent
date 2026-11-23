"""Replay archived measurements causally, starting with an empty event ledger."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from cardioagent.ecg_agent.harness import autonomous_decision as contract
from cardioagent.ecg_agent.harness.react import CLOSURE_EVIDENCE_TOOLS
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents


def main():
    parser = argparse.ArgumentParser()
    for key in ('source', 'config', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    model = LocalQwenModel(json.loads(args.config.read_text()))
    tracker = AutonomousEvents('fixed', 1550)
    trace = []
    with gzip.open(args.output/'calls.jsonl.gz', 'xt', encoding='utf8') as calls:
        for end in range(1560, 1851, 10):
            snapshot = tracker.snapshot(end-10, end, end == 1850)
            current = [r for r in rows if r['available_at_s'] == end]
            if any(r['window_s'] and r['window_s'][1] > end for r in current):
                raise ValueError('Archived evidence reads future samples')
            ids = {f'E{i}': r['evidence_id'] for i,r in enumerate(current,1)}
            ledger = {r['evidence_id']:r for r in current}
            shown = [dict(id=alias, **{k:ledger[key][k] for k in
                     ('tool','lead','window_s','status','result')}) for alias,key in ids.items()]
            messages = [dict(role='system',content=contract.prompt(snapshot,'final')),
                        dict(role='user',content=json.dumps(dict(state=snapshot,measurements=shown),allow_nan=False))]
            tools = [contract.final_tool(snapshot)]
            entry = dict(cursor_s=end,initial_state=tracker.state,measurement_ids=ids,errors=[])
            for attempt in range(2):
                model.last_raw = ''
                try:
                    reply = model.complete(messages, tools)
                    fn = reply.message['tool_calls'][0]['function']
                    if fn['name'] != 'submit_event_decision':
                        raise ValueError('Only submit_event_decision is available')
                    decision = contract.validate(json.loads(fn['arguments']),ids,set(ledger),ledger,snapshot,CLOSURE_EVIDENCE_TOOLS)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                else:
                    error = None
                calls.write(json.dumps(dict(cursor_s=end,attempt=attempt+1,messages=messages,tools=tools,
                                            raw=model.last_raw,error=error))+'\n')
                if error is None:
                    entry.update(decision=decision,update=tracker.apply(end-10,end,decision))
                    break
                entry['errors'].append(error)
                if attempt == 0:
                    messages[0]['content']=contract.prompt(snapshot,'final',retry=True)
                    messages.append(dict(role='user',content='Rejected: '+error))
            else:
                tracker.skip(end-10,end,retry=True)
            trace.append(entry)
            with (args.output/'windows.jsonl').open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
    events = tracker.finish(1850)
    actions = Counter(t.get('decision',{}).get('event_action','ACTION_ERROR') for t in trace)
    (args.output/'summary.json').write_text(json.dumps(dict(actions=actions, events=events,
        source=str(args.source), source_sha256=hashlib.sha256(source).hexdigest(),total=len(trace)),indent=2))


if __name__ == '__main__':
    main()
