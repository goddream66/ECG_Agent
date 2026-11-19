"""Current event-agent controller: admitted windows in, causal event updates out."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from math import ceil
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
from cardioagent.ecg_agent.data.preprocessing import invalid_spans, preprocess_window

def replay_event_record(cache, record, task, channel, config, score, logs, context,
                        *, policy='event-adaptive', raw_root=None, reviewer=None, rule_review=None, source_fs=SOURCE_FS):
    """The scorer is invoked only inside a requested current-window model tool."""
    event_control = policy == 'event-react' and getattr(reviewer, 'event_control', False)
    tracker = EventTracker(config, llm_closure=event_control,
                           rule_safety_fallback=event_control and
                           getattr(reviewer, 'rule_safety_fallback', False))
    if policy not in ('event-adaptive', 'legacy-dual', 'event-react'):
        raise ValueError('Unknown event policy')
    if policy == 'event-react' and reviewer is None:
        raise ValueError('event-react requires an explicit LLM reviewer')
    if rule_review is not None and policy != 'event-adaptive':
        raise ValueError('Custom rule review requires event-adaptive replay')
    if policy=='legacy-dual' and config.continue_threshold is not None:
        raise ValueError('Legacy dual policy does not implement a state-dependent score threshold')
    legacy = EventMemory() if policy == 'legacy-dual' else None
    ledger = EvidenceLedger(context)
    totals = Counter()
    began = perf_counter()
    # I/O may load raw samples in advance; tools receive only an owned current-window snapshot.
    # No filtering, peak detection, scoring, or evidence extraction on future samples occurs.
    raw = None
    if policy in ('event-adaptive', 'event-react'):
        raw = wfdb.rdrecord(str(Path(raw_root) / record['record_id']))
        if raw.fs != source_fs or tuple(raw.sig_name) != CHANNELS or raw.sig_len != record['source_record_samples']:
            raise ValueError('Raw data/cache coordinates disagree')
    secondary = next(x for x in CHANNELS if x != channel)
    boundary_review = policy == 'event-react' and getattr(reviewer, 'boundary_tools', False)
    scheduled_evidence = policy == 'event-react' and getattr(reviewer, 'scheduled_evidence', False)
    retained_samples = 120 * source_fs
    cursor = float(task['start_s'])
    for block in _arrivals(Path(cache), record, CHANNELS, task['start_s'], task['end_s'], source_fs=source_fs):
        row = block[channel][0]
        start, cursor = row['start_sample'] / source_fs, row['end_sample'] / source_fs
        primary = None
        evidence_ids = []
        calls = []
        decision_threshold = config.threshold_for(tracker)
        if row['status'] == 'incomplete':
            decision, checked, reason = 'unavailable', False, 'incomplete_window'
        else:
            origin = start
            if raw is not None:
                first = row['start_sample']
                if boundary_review:
                    first = max(ceil(task['start_s'] * source_fs - 1e-8), row['end_sample'] - retained_samples)
                    origin = first / source_fs
                w = Waveform(raw.p_signal[first:row['end_sample']], source_fs,
                             CHANNELS, tuple(raw.units))
            else:
                values = np.full((5000, 2), np.nan, dtype=np.float32)
                for i, lead in enumerate(CHANNELS):
                    if block[lead][1] is not None:
                        values[:, i] = block[lead][1]
                w = Waveform(values, 500, CHANNELS, ('normalized', 'normalized'))

            def model_window(lead, a, b):
                if (a, b) == (start, cursor):
                    selected, values = block[lead]
                    return values, lambda x: score(lead, x, selected['waveform_row'])
                if not boundary_review or not reviewer.live_scoring:
                    return None, lambda _: None
                values = w.window(lead, a-origin, b-origin)
                if len(values) != 10 * source_fs or invalid_spans(values, source_fs):
                    return None, lambda _: None
                return preprocess_window(values, source_fs), lambda x: score(lead, x, None)

            session = ToolSession(ToolContext(w, origin, cursor),
                                  max_calls=1 + config.review_budget * (2 if scheduled_evidence else 1),
                                  cached_model=model_window, evidence_context=context, allow_experimental=boundary_review)

            scheduled = False
            def call_window(tool, args):
                lead, a, b = args['lead'], args['start_s'], args['end_s']
                if scheduled:
                    trigger = 'scheduled_window_evidence'
                elif policy == 'event-react' and primary is not None:
                    trigger = 'llm_review_request'
                elif primary is None:
                    trigger = 'scheduled_primary_score'
                elif tool == 'assess_quality':
                    trigger = 'validate_primary_quality' if lead == channel else 'validate_secondary_quality'
                elif primary['status'] != 'ok':
                    trigger = 'primary_unavailable'
                elif any(x['tool'] == 'assess_quality' and x['lead'] == channel and
                         (x['status'] != 'ok' or x['result'].get('reasons')) for x in session.calls):
                    trigger = 'primary_quality_insufficient'
                elif max(primary['result']['af_score'], primary['result']['afl_score']) < decision_threshold and tracker.active:
                    trigger = 'active_event_negative'
                elif abs(max(primary['result']['af_score'], primary['result']['afl_score']) - decision_threshold) < config.borderline_margin:
                    trigger = 'borderline_score'
                elif tracker.state in ('CANDIDATE', 'PENDING_END', 'UNKNOWN_HOLD'):
                    trigger = 'resolve_' + tracker.state.lower()
                else:
                    trigger = 'periodic_review'
                result = session.call(tool, args)
                fingerprint = sha256(f'{context.manifest_id}:{context.record_id}:{a}:{b}:{lead}:{tool}'.encode()).hexdigest()
                enriched = ledger.append(result, source_evidence_id=session.cache_sources.get(result['evidence_id']),
                                         input_fingerprint=fingerprint)
                logs.emit('evidence', enriched)
                logs.emit('acquisitions', {'evidence_id': result['evidence_id'], 'recorded_at_s': cursor,
                                          'trigger_reason': trigger, 'state_before': tracker.state,
                                          'remaining_calls': session.max_calls - len(session.calls)})
                return result

            def call(tool, lead):
                return call_window(tool, {'lead': lead, 'start_s': start, 'end_s': cursor})

            primary = call('score_ecgfounder', channel)
            if legacy is not None:
                decision, checked, reason = decide(call, primary, secondary, config.threshold,
                                                   legacy, 'dual', config.review_budget)
            elif policy == 'event-react':
                seeded = None
                if scheduled_evidence:
                    scheduled = True
                    try:
                        seeded = decide_event(call, primary, secondary, tracker, config)
                    finally:
                        scheduled = False
                decision, checked, reason = reviewer.review(call, primary, secondary, tracker, config,
                                                            session, logs.emit,
                                                            **({'call_window': call_window} if boundary_review else {}),
                                                            seeded_decision=seeded)
            else:
                review = rule_review if rule_review is not None else decide_event
                decision, checked, reason = review(call, primary, secondary, tracker, config)
            calls = session.calls
            evidence_ids = [r['evidence_id'] for r in calls]
            for result in calls:
                totals['tool_calls'] += 1
                totals[f"calls_{result['tool']}"] += 1
                totals['tool_ms'] += result['duration_ms']
                if result['tool'] == 'score_ecgfounder':
                    totals['model_calls'] += result['status'] == 'ok' and not result['cached']
                else:
                    totals['other_tool_ms'] += result['duration_ms']
            totals['extra_tool_calls'] += max(0, len(calls) - 1)
            if primary['status'] == 'ok':
                totals['scored_seconds'] += cursor - start
        assessment = ledger.assessment(decision, reason, evidence_ids, cursor, checked=checked)
        logs.emit('assessments', dict(assessment, task_id=task['task_id'], decision_threshold=decision_threshold))
        if legacy is not None:
            observation = legacy.observe(start, cursor, decision, checked, evidence_ids)
            state = (observation or {}).get('state', 'idle')
        else:
            observation = tracker.observe(start, cursor, decision, checked, evidence_ids,
                                          close_request=reviewer.pending_close if event_control and primary else None)
            state = tracker.state
        if observation is not None:
            logs.emit('event_updates', dict(deepcopy(observation), task_id=task['task_id']))
        logs.emit('window_trace', {'task_id': task['task_id'], 'start_s': start, 'end_s': cursor,
                                  'cursor_s': cursor, 'decision': decision, 'reason': reason, 'state': state,
                                  'llm_action': deepcopy(reviewer.last_action)
                                      if policy == 'event-react' and hasattr(reviewer, 'last_action') else None,
                                  'decision_threshold': decision_threshold,
                                  'af_score': primary['result'].get('af_score') if primary else None,
                                  'afl_score': primary['result'].get('afl_score') if primary else None,
                                  'evidence_ids': evidence_ids, 'tool_calls': len(calls)})
        totals[f'state_seconds_{state}'] += cursor - start
        if decision in ('unavailable', 'conflict'):
            totals['unknown_seconds'] += cursor - start
    if legacy is not None:
        legacy.close('stream_ended' if legacy.open_interval else 'unresolved_at_end', cursor)
        events, intervals = legacy.events, legacy.intervals
    else:
        events, intervals = tracker.finish(cursor), tracker.intervals
    if policy == 'event-react':
        totals.update(reviewer.counts)
    return {'task_id': task['task_id'], 'record_id': record['record_id'], 'intervals': intervals,
            'events': events, 'config': config.to_dict(), 'channel': channel, 'policy': policy,
            'elapsed_ms': (perf_counter() - began) * 1000, 'counts': dict(totals)}


class ECGEventAgent:
    def __init__(self, config=None):
        self.config = config or EventConfig()
    def run_record(self, cache, record, task, channel, score, logs, context, *, policy='event-adaptive', raw_root=None, reviewer=None):
        return replay_event_record(cache, record, task, channel, self.config, score, logs, context,
                                   policy=policy, raw_root=raw_root, reviewer=reviewer)

def __getattr__(name):
    if name in ('Config', 'ECGAgent'):
        from importlib import import_module
        return getattr(import_module('cardioagent.legacy.ecg_measurement.agent'), name)
    raise AttributeError(name)
