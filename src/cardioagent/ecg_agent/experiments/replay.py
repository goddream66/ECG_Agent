from collections import Counter
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
import wfdb
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.memory.evidence import EvidenceLedger
from cardioagent.ecg_agent.memory.events import EventTracker
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.policies.rule_review import decide_event
from cardioagent.ecg_agent.policies.baselines import EventMemory, decide
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.data.afdb import CHANNELS, SOURCE_FS, _arrivals
from cardioagent.common.io import read_jsonl, write_jsonl
from cardioagent.ecg_agent.harness.logging import JsonLogs, write_line

def fixed_evidence_record(rows, task, config, logs, context, *, baseline=None):
    """Reinterpret recorded returns using the frozen old policy and a legacy shadow state."""
    shadow = EventMemory()
    tracker = EventTracker(config)
    ledger = EvidenceLedger(context)
    totals = Counter()
    scores = []
    source_digest = sha256()
    output_digest = sha256()
    source_calls = consumed_calls = 0
    cursor = task['start_s']
    for row in rows:
        start, cursor = row['start_s'], row['cursor_s']
        originals = row.get('tool_calls', [])
        copied = []
        for seq, original in enumerate(originals, 1):
            identity = f'{context.run_id}:{context.task_id}:{context.record_id}:{start}:{cursor}:{seq}'
            result = deepcopy(original)
            result['evidence_id'] = 'ecg:' + sha256(identity.encode()).hexdigest()
            logs.emit('evidence', ledger.append(result, source_evidence_id=original['evidence_id']))
            copied.append(result)
            # Original IDs/timings remain part of source integrity; copied namespace is mapped separately.
            source_digest.update(json.dumps(original, sort_keys=True, allow_nan=False).encode())
        position = 0

        def call(tool, lead):
            nonlocal position, consumed_calls
            if position >= len(copied):
                raise ValueError('Frozen policy requested unrecorded evidence')
            original, result = originals[position], copied[position]
            if result['tool'] != tool or result['lead'] != lead or result['window_s'] != [start, cursor]:
                raise ValueError('Frozen evidence order or coordinates differ')
            output_digest.update(json.dumps(original, sort_keys=True, allow_nan=False).encode())
            consumed_calls += 1
            position += 1
            return result

        if not originals:
            decision, checked, reason = 'unavailable', False, 'incomplete_window'
        else:
            lead = row['channel']
            primary = call('score_ecgfounder', lead)
            secondary = next(x for x in CHANNELS if x != lead)
            decision, checked, reason = decide(call, primary, secondary, .5, shadow, 'adaptive', 4)
        if position != len(copied):
            raise ValueError('Frozen policy did not consume the exact recorded evidence')
        if decision != row['decision'] or reason != row['reason']:
            raise ValueError('Legacy assessment reconstruction differs from frozen trace')
        shadow.observe(start, cursor, decision, checked, [r['evidence_id'] for r in copied])
        source_calls += len(originals)
        totals['tool_calls'] += len(originals)
        totals['extra_tool_calls'] += max(0, len(originals) - 1)
        totals['model_calls'] += sum(r['tool'] == 'score_ecgfounder' and r['status'] == 'ok'
                                     and not r['cached'] for r in originals)
        ids = [r['evidence_id'] for r in copied]
        logs.emit('assessments', ledger.assessment(decision, reason, ids, cursor, checked=checked))
        observation = tracker.observe(start, cursor, decision, checked, ids)
        logs.emit('event_updates', dict(observation, task_id=task['task_id']))
        logs.emit('window_trace', {'task_id': task['task_id'], 'start_s': start, 'end_s': cursor,
                                  'cursor_s': cursor, 'decision': decision, 'reason': reason,
                                  'state': tracker.state, 'evidence_ids': ids,
                                  'af_score': row.get('af_score'), 'afl_score': row.get('afl_score')})
    events = tracker.finish(cursor)
    if source_calls != consumed_calls or source_digest.hexdigest() != output_digest.hexdigest():
        raise ValueError('Fixed-evidence integrity failure')
    return {'task_id': task['task_id'], 'record_id': context.record_id, 'events': events,
            'intervals': tracker.intervals, 'counts': dict(totals), 'config': config.to_dict(),
            'integrity': {'source_calls': source_calls, 'consumed_calls': consumed_calls,
                          'source_sha256': source_digest.hexdigest(), 'consumed_sha256': output_digest.hexdigest(),
                          'identical': True}}
