"""Bounded ReAct review runner; the provider chooses tools, existing rules judge evidence."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from pydantic import BaseModel, ConfigDict, Field
from cardioagent.ecg_agent.policies.rule_review import decide_event

PROMPT = Path(__file__).resolve().parents[1] / 'prompts/react_review_v1.txt'
ALLOWED = frozenset(('score_ecgfounder', 'assess_quality'))


class FinishReview(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(max_length=32)


class ReActReview:
    def __init__(self, model, *, max_rounds=5, history_windows=6):
        if type(max_rounds) is not int or not 1 <= max_rounds <= 16:
            raise ValueError('max_rounds must be 1..16')
        if type(history_windows) is not int or not 0 <= history_windows <= 24:
            raise ValueError('history_windows must be 0..24')
        self.model, self.max_rounds, self.history_windows = model, max_rounds, history_windows
        self.history = []
        self.identity = None
        self.counts = {'llm_calls': 0, 'llm_ms': 0., 'llm_prompt_tokens': 0,
                       'llm_completion_tokens': 0}

    def review(self, call, primary, secondary, tracker, config, session, emit):
        start, end = session.context.origin_s, session.context.cursor_s
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
        # Active events may contain hours of intervals; expose bounded recent spans only.
        if state['active_event']:
            for key, value in state['active_event'].items():
                if isinstance(value, list):
                    state['active_event'][key] = value[-6:]
        messages = [{'role': 'system', 'content': PROMPT.read_text(encoding='utf-8')},
                    {'role': 'user', 'content': json.dumps(state, allow_nan=False)}]
        tools = [s for s in session.registry.function_schemas() if s['function']['name'] in ALLOWED]
        tools.append({'type': 'function', 'function': {'name': 'finish_review',
                      'description': 'Stop this window review, with evidence references; not event termination.',
                      'parameters': FinishReview.model_json_schema()}})
        stop = 'round_limit'
        failed = False
        for step in range(self.max_rounds):
            began = perf_counter()
            audit = dict(identity, cursor_s=end, step=step)
            self.counts['llm_calls'] += 1
            try:
                try:
                    reply = self.model.complete(deepcopy(messages), deepcopy(tools))
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
                if name == 'finish_review':
                    finish = FinishReview.model_validate(args)
                    known = {r['evidence_id'] for r in session.calls}
                    if not set(finish.evidence_ids) <= known:
                        raise ValueError('Finish cites unknown evidence')
                    emit('llm_steps', dict(audit, status='stopped', reason=finish.reason))
                    stop = 'model_finish'
                    break
                if name not in ALLOWED:
                    raise ValueError('Tool is not enabled for this review')
                if len(session.calls) >= session.max_calls:
                    stop = 'tool_budget'; break
                # Validate before dispatch; even rejected requests receive a bounded audit entry.
                parsed = session.registry._tools[name].arguments.model_validate(args)
                if (parsed.start_s, parsed.end_s) != (start, end):
                    raise ValueError('Only the arrived current window is retained')
                result = call(name, parsed.lead)
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
        acquired = {(r['tool'], r['lead']): deepcopy(r) for r in session.calls}
        def existing(tool, lead):
            return deepcopy(acquired.get((tool, lead)))
        if failed:
            decision = ('unavailable', False, stop)
        else:
            decision = decide_event(existing, primary, secondary, tracker, config)
        emit('llm_steps', dict(identity, cursor_s=end, status='review_complete', stop_reason=stop,
                              decision=decision[0], interpreter_reason=decision[2],
                              remaining_tool_calls=session.max_calls-len(session.calls)))
        self.history.append({'window_s': [start, end], 'decision': decision[0],
                             'af_score': primary['result'].get('af_score'),
                             'afl_score': primary['result'].get('afl_score'),
                             'acquired': [{'tool': r['tool'], 'lead': r['lead'], 'status': r['status'],
                                           'result': deepcopy(r['result'])} for r in session.calls],
                             'evidence_ids': [r['evidence_id'] for r in session.calls]})
        self.history = self.history[-self.history_windows:] if self.history_windows else []
        return decision
