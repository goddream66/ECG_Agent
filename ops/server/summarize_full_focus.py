"""Build one auditable behavior row per arrived window from a completed focus run."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def summarize(folder):
    grouped = defaultdict(list)
    with (folder/'llm_steps.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row.get('phase'):
                grouped[row['cursor_s']].append(row)
    summary = Counter()
    with (folder/'window_trace.jsonl').open() as windows, (folder/'qwen_behavior.jsonl').open('w') as target:
        for line in windows:
            window = json.loads(line)
            steps = grouped.pop(window['cursor_s'], [])
            plans = []
            for i, step in enumerate(steps):
                if step['status'] != 'focus_recorded':
                    continue
                plan = step['focus']
                nxt = steps[i+1] if i+1 < len(steps) else {}
                args = nxt.get('arguments') or {}
                actual = nxt.get('action')
                name = plan['next_tool']
                if name == 'none':
                    match = actual in ('finish_measurements', 'submit_event_decision')
                elif name == 'retrieve_memory':
                    match = actual == name
                else:
                    match = actual == name and all(args.get(key) == plan.get(field)
                        for key, field in [('lead','target_lead'), ('start_s','target_start_s'), ('end_s','target_end_s')])
                plans.append(dict(plan=plan, next_action=actual, next_arguments=args,
                                  execution_status=nxt.get('status'), matches_immediately=match))
                summary['plans'] += 1
                summary['immediate_plan_matches'] += int(match)
            entry = dict(window=[window['start_s'], window['end_s']], action=window['action'],
                         interpretation=window['interpretation'], state=window['state'],
                         reason=window['reason'], plans=plans, steps=steps)
            target.write(json.dumps(entry)+'\n')
            summary['windows'] += 1
            summary['action_'+str(window['action'])] += 1
    (folder/'behavior_summary.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
