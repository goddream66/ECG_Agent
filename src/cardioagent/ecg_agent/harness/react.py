"""Bounded ReAct review runner; the provider chooses tools, existing rules judge evidence."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from cardioagent.ecg_agent.policies.rule_review import decide_event

PROMPT = Path(__file__).resolve().parents[1] / 'prompts/react_review_v1.txt'
ALLOWED = frozenset(('score_ecgfounder', 'assess_quality'))
BOUNDARY_TOOLS = frozenset(('detect_beats', 'get_rate_rhythm', 'get_feature_profile',
                            'get_p_wave_proxy', 'get_qrs_proxy', 'get_beat_timeline_proxy'))
BOUNDARY_PROMPT = """Boundary evidence is enabled. Choose a range wholly inside the retained signal.
Fractional seconds are allowed. For possible rhythm changes query surrounding beats on either lead.
Use get_beat_timeline_proxy for aligned raw RR, P candidates and QRS morphology proxies.
Quality warnings and missing measurements express uncertainty. These measurements alone cannot establish AF termination.
For a possible event end, use finish_review.proposed_end_s, with measured evidence IDs and an explanation.
The runtime records the suggestion. It uses its separate event lifecycle to close tracked events.
"""
EVENT_CONTROL_PROMPT = """Boundary evidence is enabled. Choose a range wholly inside the retained arrived signal.
Fractional seconds are allowed. For a possible rhythm change query surrounding beats on either lead.
Use get_beat_timeline_proxy for aligned raw RR, P candidates and QRS morphology proxies.
There are two distinct final actions for a confirmed active event:
- keep_event_open: end only this window's review. In reason, state either that available
  evidence supports continued AF/AFL or that the end remains uncertain, and say why.
  Do not use this action to report a confirmed event end.
- end_confirmed_event: request closure of the named event. First obtain successful local
  boundary evidence on arrived signal, then provide event_id, estimated end_s, actual
  evidence_ids and a reason explaining why they support a rhythm transition.
If you conclude the event has ended and have the required local evidence, call
end_confirmed_event, not keep_event_open. If evidence is insufficient, keep the event
open and explain what remains unresolved. A low score, poor quality or one proxy alone
does not establish termination. Decision time is not automatically the event end time.
If no confirmed active event exists, use keep_event_open to finish review; never
call end_confirmed_event. This action does not create or confirm an event.
"""
UNIFIED_PROMPT = """Boundary evidence is enabled. Use only arrived retained signal.
Measure when useful, then use submit_event_decision as the only final action.
KEEP_OPEN means evidence supports continued AF/AFL; DEFER means termination remains
uncertain; CLOSE requests closure of an already confirmed event. A low score,
poor quality, or one proxy alone does not establish termination. For CLOSE,
cite successful local boundary evidence and estimate the event end separately
from this decision's cursor. If no confirmed event exists, do not choose CLOSE.
"""


class FinishReview(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    reason: str = Field(min_length=1, max_length=4096)
    evidence_ids: list[str] = Field(max_length=32)


class FinishBoundaryReview(FinishReview):
    proposed_end_s: float | None = Field(default=None, ge=0, allow_inf_nan=False,
                                         description='Optional proposed end of the confirmed active event, in seconds')

class CloseEvent(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    event_id: int = Field(ge=1)
    end_s: float = Field(ge=0, allow_inf_nan=False)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)
    reason: str = Field(min_length=1, max_length=4096)


class EventDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    decision: Literal['KEEP_OPEN', 'DEFER', 'CLOSE']
    event_id: int | None
    end_s: float | None = Field(allow_inf_nan=False)
    evidence_ids: list[str] = Field(max_length=32)
    reason: str = Field(min_length=1, max_length=4096)


class ReActReview:
    def __init__(self, model, *, max_rounds=5, history_windows=6, boundary_tools=False,
                 live_scoring=False, event_control=False, scheduled_evidence=False,
                 unified_event_decision=False):
        if type(max_rounds) is not int or not 1 <= max_rounds <= 16:
            raise ValueError('max_rounds must be 1..16')
        if type(history_windows) is not int or not 0 <= history_windows <= 24:
            raise ValueError('history_windows must be 0..24')
        if event_control and not boundary_tools:
            raise ValueError('Event control requires boundary tools')
        if unified_event_decision and not event_control:
            raise ValueError('Unified event decision requires event control')
        self.model, self.max_rounds, self.history_windows = model, max_rounds, history_windows
        self.boundary_tools = boundary_tools
        self.live_scoring = boundary_tools and live_scoring
        self.event_control = event_control
        self.unified_event_decision = unified_event_decision
        self.scheduled_evidence = scheduled_evidence
        self.pending_close = None
        self.history = []
        self.identity = None
        self.counts = {'llm_calls': 0, 'llm_ms': 0., 'llm_prompt_tokens': 0,
                       'llm_completion_tokens': 0}

    def review(self, call, primary, secondary, tracker, config, session, emit, *,
               call_window=None, seeded_decision=None):
        self.pending_close = None
        start, end = primary['window_s']
        c = session.evidence_context
        identity = {'run_id': c.run_id, 'task_id': c.task_id, 'record_id': c.record_id}
        if self.identity is not None and self.identity != identity:
            raise ValueError('Create a fresh reviewer for each run/task/record')
        self.identity = identity
        state = {'identity': identity, 'window_s': [start, end], 'primary_lead': primary['lead'],
                 'available_leads': list(session.context.waveform.channels),
                 'secondary_lead': secondary, 'state': tracker.state,
                 'active_event': deepcopy(tracker.active), 'recent_windows': deepcopy(self.history),
                 'consecutive_support': tracker.support_count,
                 'consecutive_negative': tracker.negative_count,
                 'unknown_start_s': tracker.unknown_start,
                 'primary_evidence': deepcopy(primary),
                 'remaining_tool_calls': session.max_calls - len(session.calls)}
        if seeded_decision is not None:
            state['scheduled_window_interpretation'] = {
                'decision': seeded_decision[0], 'checked': seeded_decision[1],
                'reason': seeded_decision[2]}
        if self.unified_event_decision:
            state['already_acquired_evidence'] = [
                {'evidence_id': r['evidence_id'], 'tool': r['tool'], 'lead': r['lead'],
                 'window_s': deepcopy(r['window_s']), 'status': r['status'],
                 'result': deepcopy(r['result'])}
                for r in session.calls if r['available_at_s'] <= end]
        # Active events may contain hours of intervals; expose bounded recent spans only.
        if state['active_event']:
            for key, value in state['active_event'].items():
                if isinstance(value, list):
                    state['active_event'][key] = value[-6:]
        allowed = ALLOWED | BOUNDARY_TOOLS if self.boundary_tools else ALLOWED
        prompt = PROMPT.read_text(encoding='utf-8')
        if self.boundary_tools:
            if call_window is None:
                raise ValueError('Boundary review requires a retained-window dispatcher')
            state['retained_window_s'] = [session.context.origin_s, end]
            state['ecgfounder_scope'] = 'retained_10s_window' if self.live_scoring else 'current_window_only'
            prompt = prompt.replace(
                'Choose which current-window lead to score or check for quality. Use exact lead names.\n'
                'All measurement arguments must match current start_s/end_s. Never request future signal.\n',
                'Use exact lead names and respect ecgfounder_scope. Never request future signal.\n'
            ) + '\n' + (UNIFIED_PROMPT if self.unified_event_decision else
                         EVENT_CONTROL_PROMPT if self.event_control else BOUNDARY_PROMPT)
            if self.event_control:
                prompt = prompt.replace(
                    'Use finish_review when acquired evidence is enough OR further useful checks cannot be made.\n'
                    'Stopping review does not mean the event has ended. The existing rule interpreter and\n'
                    'event lifecycle determine support/negative/conflict/unavailable and event boundaries.\n'
                    'Return a short action justification and actual evidence IDs in finish_review.\n',
                    'Choose a final action only after considering acquired evidence and the active event.\n'
                    'The rule interpreter determines each window interpretation; the event-end action\n'
                    'controls closure of a confirmed event. Cite actual evidence IDs.\n')
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(state, allow_nan=False)}]
        tools = [s for s in session.registry.function_schemas(include_experimental=self.boundary_tools)
                 if s['function']['name'] in allowed]
        finish_schema = FinishBoundaryReview if self.boundary_tools and not self.event_control else FinishReview
        if self.unified_event_decision:
            prompt = prompt + ('\nAfter measuring, use submit_event_decision as the sole final action. '
                'KEEP_OPEN means evidence supports continuation; DEFER means the end is unresolved; '
                'CLOSE requests termination only with local boundary evidence. For non-CLOSE, '
                'end_s must be null; event_id may name the current event. For CLOSE, cite the active event ID, actual '
                'evidence IDs, and an estimated boundary distinct from decision time.\n')
            messages[0]['content'] = prompt
            tools.append({'type': 'function', 'function': {'name': 'submit_event_decision',
                'description': 'Submit one explicit event-end decision and finish this review.',
                'parameters': EventDecision.model_json_schema()}})
        else:
            tools.append({'type': 'function', 'function': {'name': 'keep_event_open' if self.event_control else 'finish_review',
                      'description': ('End this window review while leaving the confirmed event open. '
                                      'Explain continued support or uncertainty; never use to confirm an end.'
                                      if self.event_control else
                                      'Stop this window review, with evidence references; not event termination.'),
                      'parameters': finish_schema.model_json_schema()}})
        if self.event_control and not self.unified_event_decision:
            tools.append({'type': 'function', 'function': {'name': 'end_confirmed_event',
                          'description': 'Request closure of the named confirmed active event using local boundary evidence.',
                          'parameters': CloseEvent.model_json_schema()}})
        stop = 'round_limit'
        failed = False
        for step in range(self.max_rounds):
            began = perf_counter()
            audit = dict(identity, cursor_s=end, step=step)
            self.counts['llm_calls'] += 1
            try:
                try:
                    available_tools = ([t for t in tools if t['function']['name'] == 'submit_event_decision']
                                       if self.unified_event_decision and step == self.max_rounds - 1
                                       else tools)
                    reply = self.model.complete(deepcopy(messages), deepcopy(available_tools))
                finally:
                    self.counts['llm_ms'] += (perf_counter() - began) * 1000
                for key in ('prompt_tokens', 'completion_tokens'):
                    value = reply.usage.get(key, 0)
                    if type(value) is int and value >= 0:
                        self.counts['llm_' + key] += value
                msg = reply.message
                calls = msg.get('tool_calls', [])
                if msg.get('role') != 'assistant' or len(calls) != 1:
                    raise ValueError('Exactly one assistant tool call required')
                fn = calls[0]['function']; name = fn['name']
                args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
                if not isinstance(args, dict):
                    raise ValueError('Tool arguments must be an object')
                audit = dict(identity, cursor_s=end, step=step, action=name, arguments=deepcopy(args))
                if name == 'submit_event_decision' and self.unified_event_decision:
                    submitted = EventDecision.model_validate(args)
                    known = {r['evidence_id']: r for r in session.calls}
                    visible_history = {evidence_id for window in self.history
                                       for evidence_id in window['evidence_ids']}
                    if (len(set(submitted.evidence_ids)) != len(submitted.evidence_ids) or
                            not set(submitted.evidence_ids) <= known.keys() | visible_history):
                        raise ValueError('Event decision cites unavailable or duplicate evidence')
                    if submitted.decision == 'CLOSE':
                        close = CloseEvent.model_validate({
                            'event_id': submitted.event_id, 'end_s': submitted.end_s,
                            'evidence_ids': submitted.evidence_ids, 'reason': submitted.reason})
                        active = tracker.active
                        if (active is None or active['confirmed_at_s'] is None
                                or close.event_id != active['event_id']
                                or not active['reported_intervals'][-1][0] < close.end_s <= end
                                or not any(k in known and known[k]['tool'] in BOUNDARY_TOOLS
                                           and known[k]['status'] == 'ok'
                                           and known[k]['window_s'][0] <= close.end_s <= known[k]['window_s'][1]
                                           for k in close.evidence_ids)):
                            raise ValueError('Close needs confirmed event and successful local boundary evidence')
                        self.pending_close = close.model_dump()
                    elif (submitted.end_s is not None or
                          (submitted.event_id is not None and
                           (tracker.active is None or submitted.event_id != tracker.active['event_id']))):
                        raise ValueError('Non-close decision has end time or wrong event ID')
                    emit('llm_steps', dict(audit, status='decision_submitted',
                                           decision=submitted.decision,
                                           evidence_ids=submitted.evidence_ids))
                    stop = 'model_decision_' + submitted.decision.lower()
                    break
                if name == 'end_confirmed_event' and self.event_control:
                    close = CloseEvent.model_validate(args)
                    active = tracker.active
                    known = {r['evidence_id']: r for r in session.calls}
                    if (active is None or active['confirmed_at_s'] is None
                            or close.event_id != active['event_id']
                            or not active['reported_intervals'][-1][0] < close.end_s <= end
                            or len(set(close.evidence_ids)) != len(close.evidence_ids)
                            or not set(close.evidence_ids) <= known.keys()
                            or not any(known[k]['tool'] in BOUNDARY_TOOLS and known[k]['status'] == 'ok'
                                       and known[k]['window_s'][0] <= close.end_s <= known[k]['window_s'][1]
                                       for k in close.evidence_ids)):
                        raise ValueError('Event closure requires an active event and successful local boundary evidence')
                    self.pending_close = close.model_dump()
                    emit('llm_steps', dict(audit, status='close_requested',
                                           evidence_ids=close.evidence_ids))
                    stop = 'model_end_confirmed_event'
                    break
                if name == ('keep_event_open' if self.event_control else 'finish_review'):
                    finish = finish_schema.model_validate(args)
                    known = {r['evidence_id']: r for r in session.calls}
                    if not set(finish.evidence_ids) <= known.keys():
                        raise ValueError('Finish cites unknown evidence')
                    recorded = dict(audit, status='stopped', reason=finish.reason)
                    if self.boundary_tools:
                        recorded['evidence_ids'] = finish.evidence_ids
                    if self.boundary_tools and not self.event_control and finish.proposed_end_s is not None:
                        proposed = finish.proposed_end_s
                        active = tracker.active
                        if (not active or active['confirmed_at_s'] is None
                                or not max(session.context.origin_s, active['estimated_start_s']) < proposed <= end
                                or not any(known[k]['tool'] in BOUNDARY_TOOLS and known[k]['status'] == 'ok'
                                           and known[k]['window_s'][0] <= proposed < known[k]['window_s'][1]
                                           for k in finish.evidence_ids)):
                            raise ValueError('Proposed end needs an active event and successful local evidence at the boundary')
                        recorded.update(proposed_end_s=proposed, proposed_for_event_id=active['event_id'])
                    emit('llm_steps', recorded)
                    stop = 'model_keep_event_open' if self.event_control else 'model_finish'
                    break
                if name not in allowed:
                    raise ValueError('Tool is not enabled for this review')
                if len(session.calls) >= session.max_calls:
                    stop = 'tool_budget'; break
                # Validate before dispatch; even rejected requests receive a bounded audit entry.
                parsed = session.registry._tools[name].arguments.model_validate(args)
                if not session.context.origin_s <= parsed.start_s < parsed.end_s <= end:
                    raise ValueError('Measurement is outside the retained elapsed waveform')
                current_only = not self.boundary_tools or (name == 'score_ecgfounder' and not self.live_scoring)
                if current_only and (parsed.start_s, parsed.end_s) != (start, end):
                    raise ValueError('Only the arrived current window is retained')
                result = call_window(name, parsed.model_dump()) if self.boundary_tools else call(name, parsed.lead)
                emit('llm_steps', dict(audit, status='observed', evidence_id=result['evidence_id']))
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': f'call-{step}', 'type': 'function', 'function': {
                        'name': name, 'arguments': json.dumps(args, allow_nan=False)}}]})
                messages.append({'role': 'tool', 'tool_call_id': f'call-{step}',
                                 'content': json.dumps(result, allow_nan=False)})
                messages.append({'role': 'user', 'content': json.dumps({
                    'remaining_tool_calls': session.max_calls - len(session.calls)})})
            except Exception as exc:
                # No invented fallback diagnosis on backend/schema/timeout failure.
                stop, failed = 'model_or_action_error', True
                emit('llm_steps', dict(audit, status='error', error_type=type(exc).__name__))
                break
        acquired = {(r['tool'], r['lead']): deepcopy(r) for r in session.calls
                    if r['window_s'] == [start, end]}
        def existing(tool, lead):
            return deepcopy(acquired.get((tool, lead)))
        if failed:
            decision = ('unavailable', False, stop)
        else:
            decision = decide_event(existing, primary, secondary, tracker, config)
        emit('llm_steps', dict(identity, cursor_s=end, status='review_complete', stop_reason=stop,
                              decision=decision[0], interpreter_reason=decision[2],
                              remaining_tool_calls=session.max_calls-len(session.calls)))
        acquired_summary = []
        for result in session.calls:
            item = {key: result[key] for key in ('tool', 'lead', 'status')}
            if self.boundary_tools:
                item.update(window_s=deepcopy(result['window_s']), evidence_id=result['evidence_id'])
            if not self.boundary_tools or result['tool'] not in BOUNDARY_TOOLS:
                item['result'] = deepcopy(result['result'])
            acquired_summary.append(item)
        self.history.append({'window_s': [start, end], 'decision': decision[0],
                             'af_score': primary['result'].get('af_score'),
                             'afl_score': primary['result'].get('afl_score'),
                             'acquired': acquired_summary,
                             'evidence_ids': [r['evidence_id'] for r in session.calls]})
        self.history = self.history[-self.history_windows:] if self.history_windows else []
        return decision
