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
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents, merge_spans
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.policies.rule_review import decide_event
from cardioagent.ecg_agent.policies.baselines import EventMemory, decide
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.data.afdb import CHANNELS, SOURCE_FS, _arrivals
from cardioagent.ecg_agent.data.preprocessing import invalid_spans, preprocess_window

def replay_event_record(cache, record, task, channel, config, score, logs, context,
                        *, policy='event-adaptive', raw_root=None, reviewer=None, rule_review=None,
                        source_fs=SOURCE_FS, checkpoint=None):
    """The scorer is invoked only inside a requested current-window model tool."""
    react = policy == 'event-react'
    event_control = react and getattr(reviewer, 'event_control', False)
    review_mode = getattr(reviewer, 'event_review_mode', None) if react else None
    if review_mode:
        tracker = AutonomousEvents(review_mode, float(task['start_s']))
    else:
        tracker = EventTracker(config, llm_closure=event_control,
                               rule_safety_fallback=event_control and getattr(reviewer, 'rule_safety_fallback', False))
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
    if checkpoint is not None and not review_mode:
        raise ValueError('Window checkpoints require an autonomous event ledger')
    previous_ms = checkpoint.restore(ledger, tracker, reviewer, totals) if checkpoint else 0.
    resume_cursor = tracker.cursor if checkpoint else None

    def commit_window():
        if checkpoint:
            checkpoint.commit(logs, tracker, reviewer, totals, previous_ms + (perf_counter() - began) * 1000)
    # I/O may load raw samples in advance; tools receive only an owned current-window snapshot.
    # No filtering, peak detection, scoring, or evidence extraction on future samples occurs.
    raw = None
    if policy in ('event-adaptive', 'event-react'):
        raw = wfdb.rdrecord(str(Path(raw_root) / record['record_id']))
        if raw.fs != source_fs or tuple(raw.sig_name) != CHANNELS or raw.sig_len != record['source_record_samples']:
            raise ValueError('Raw data/cache coordinates disagree')
    secondary = next(x for x in CHANNELS if x != channel)
    boundary_review = react and getattr(reviewer, 'boundary_tools', False)
    llm_interprets = react and getattr(reviewer, 'llm_window_interpretation', False)
    retained_samples = 120 * source_fs
    cursor = resume_cursor if resume_cursor is not None else float(task['start_s'])
    for block in _arrivals(Path(cache), record, CHANNELS, task['start_s'], task['end_s'], source_fs=source_fs):
        row = block[channel][0]
        start, cursor = row['start_sample'] / source_fs, row['end_sample'] / source_fs
        if resume_cursor is not None and cursor <= resume_cursor:
            continue
        if review_mode and tracker.cursor < start:
            tracker.skip(tracker.cursor, start)
        primary = None
        close_request = None
        llm_action = None
        audit = {}
        evidence_ids = []
        calls = []
        decision_threshold = None if review_mode else config.threshold_for(tracker)
        if review_mode:
            snapshot = tracker.snapshot(start, cursor, cursor >= task['end_s'])
            if not tracker.due(cursor) or row['status'] == 'incomplete':
                reason = 'incomplete_window' if row['status'] == 'incomplete' else 'awaiting_appointment'
                tracker.skip(start, cursor)
                totals['unknown_seconds'] += cursor - start
                totals['unreviewed_windows'] += 1
                totals[f'state_seconds_{tracker.state}'] += cursor - start
                logs.emit('window_trace', dict(task_id=task['task_id'], start_s=start, end_s=cursor,
                          cursor_s=cursor, state=tracker.state, decision='UNREVIEWED', action=None,
                          reason=reason, requested_wake_s=tracker.next_review, evidence_ids=[], tool_calls=0))
                commit_window()
                continue
            totals['review_wakes'] += 1
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

            initial_calls = 0 if react else 1
            llm_limit = getattr(reviewer, 'llm_tool_max_calls', None) if react else None
            session = ToolSession(ToolContext(w, origin, cursor),
                                  max_calls=llm_limit if llm_limit is not None else initial_calls + config.review_budget,
                                  cached_model=model_window, evidence_context=context, allow_experimental=boundary_review)

            def call_window(tool, args):
                lead, a, b = args['lead'], args['start_s'], args['end_s']
                if react:
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

            if react:
                decision, checked, reason = reviewer.review(call, None, secondary, tracker, config,
                                                            session, logs.emit,
                                                            window_s=(start, cursor), primary_lead=channel,
                                                            **({'review_snapshot': snapshot} if review_mode else {}),
                                                            **({'call_window': call_window} if boundary_review else {}),
                                                            **({'ledger': ledger} if llm_interprets or getattr(reviewer, 'memory_retrieve', False) else {}))
                llm_action = deepcopy(getattr(reviewer, 'last_action', None))
                if event_control:
                    close_request = reviewer.pending_close
            else:
                primary = call('score_ecgfounder', channel)
                if legacy is not None:
                    decision, checked, reason = decide(call, primary, secondary, config.threshold,
                                                       legacy, 'dual', config.review_budget)
                else:
                    review = rule_review if rule_review is not None else decide_event
                    decision, checked, reason = review(call, primary, secondary, tracker, config)
            calls = session.calls
            if react:
                primary = next((r for r in reversed(calls)
                                if r['tool'] == 'score_ecgfounder' and r['lead'] == channel
                                and r['window_s'] == [start, cursor]), None)
            evidence_ids = [r['evidence_id'] for r in calls]
            for result in calls:
                totals['tool_calls'] += 1
                totals[f"calls_{result['tool']}"] += 1
                totals['tool_ms'] += result['duration_ms']
                if result['tool'] == 'score_ecgfounder':
                    totals['model_calls'] += result['status'] == 'ok' and not result['cached']
                else:
                    totals['other_tool_ms'] += result['duration_ms']
            totals['extra_tool_calls'] += len(calls) - initial_calls
            if (react and any(r['tool'] == 'score_ecgfounder' and r['status'] == 'ok'
                              and r['window_s'] == [start, cursor] for r in calls)
                    or not react and primary['status'] == 'ok'):
                totals['scored_seconds'] += cursor - start
        if review_mode:
            failed = decision == 'ACTION_ERROR'
            submitted = llm_action or {}
            if failed:
                tracker.skip(start, cursor, retry=True)
                totals['action_errors'] += 1
            else:
                update = tracker.apply(start, cursor, submitted)
                logs.emit('event_updates', dict(update, task_id=task['task_id'], action=submitted['event_action']))
                totals['action_' + submitted['event_action']] += 1
                logs.emit('assessments', dict(task_id=task['task_id'], recorded_at_s=cursor,
                          window_s=[start, cursor], **submitted))
            audit = dict(task_id=task['task_id'], start_s=start, end_s=cursor, cursor_s=cursor,
                         requested_wake_s=snapshot['requested_wake_s'], actual_wake_s=cursor,
                         next_review_s=tracker.next_review, decision=decision, state=tracker.state,
                         action=submitted.get('action'), reason=reason, llm_action=llm_action,
                         evidence_ids=submitted.get('evidence_ids', []), acquired_evidence_ids=evidence_ids,
                         tool_calls=len(calls))
            logs.emit('window_trace', audit)
            totals[f'state_seconds_{tracker.state}'] += cursor - start
            commit_window()
            continue
        acquired_ids = evidence_ids
        if llm_interprets:
            metadata = llm_action or {}
            evidence_ids = metadata.get('evidence_ids', [])
            audit = dict(decision_source=metadata.get('decision_source', 'incomplete_window'),
                         **({'evidence_focus': deepcopy(reviewer.evidence_focus)}
                            if getattr(reviewer, 'investigation_checkpoint', False) else {}),
                         interpretation=decision, action=metadata.get('action'),
                         submitted_event_id=metadata.get('event_id'), submitted_end_s=metadata.get('end_s'),
                         reason=metadata.get('reason', reason), acquired_evidence_ids=acquired_ids)
        assessment = ledger.assessment(decision, reason, evidence_ids, cursor, checked=checked)
        assessment.update(audit)
        logs.emit('assessments', dict(assessment, task_id=task['task_id'], decision_threshold=decision_threshold))
        if legacy is not None:
            observation = legacy.observe(start, cursor, decision, checked, evidence_ids)
            state = (observation or {}).get('state', 'idle')
        else:
            observation = tracker.observe(start, cursor, decision, checked, evidence_ids,
                                          close_request=close_request)
            state = tracker.state
        if observation is not None:
            observation.update(audit)
            update = observation.get('event_update')
            if update and update['kind'].startswith('rule_safety'):
                update['decision_source'] = 'rule_fallback'
            elif update and audit:
                update.update(audit)
            logs.emit('event_updates', dict(deepcopy(observation), task_id=task['task_id']))
        logs.emit('window_trace', {**audit, 'task_id': task['task_id'], 'start_s': start, 'end_s': cursor,
                                  'cursor_s': cursor, 'decision': decision, 'reason': reason, 'state': state,
                                  'llm_action': llm_action,
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
        if review_mode and cursor < task['end_s']:
            tracker.skip(cursor, task['end_s'])
            cursor = task['end_s']
        events, intervals = tracker.finish(cursor), tracker.intervals
    extra = {}
    if review_mode:
        unknown = merge_spans(tracker.unknown)
        totals['unknown_seconds'] = sum(b-a for a,b in unknown)
        supported = merge_spans(tracker.support)
        extra = dict(review_mode=review_mode, event_control='qwen_open_close',
                     event_boundary_projection='midpoint_of_bracket', burden_intervals=supported,
                     supported_seconds=sum(b-a for a,b in supported),
                     conflict_intervals=merge_spans(tracker.conflicts),
                     unknown_intervals=unknown, pending_questions=deepcopy(tracker.questions))
    if policy == 'event-react':
        totals.update(reviewer.counts)
    return {'task_id': task['task_id'], 'record_id': record['record_id'], 'intervals': intervals,
            'events': events, 'config': config.to_dict(), 'channel': channel, 'policy': policy,
            'elapsed_ms': previous_ms + (perf_counter() - began) * 1000, 'counts': dict(totals), **extra}


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
