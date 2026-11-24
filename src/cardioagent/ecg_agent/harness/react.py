"""Bounded ReAct review with selectable rule or model interpretation."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from cardioagent.ecg_agent.policies.rule_review import decide_event
from .event_decision import WindowEventDecision, overlaps_current, validate_grounding
from . import active_acquisition, autonomous_decision, split_event, measured_event, staged_review

PROMPT = Path(__file__).resolve().parents[1] / 'prompts/react_review_v1.txt'
WINDOW_PROMPT = PROMPT.with_name('react_window_decision_v2.txt')
ALLOWED = frozenset(('score_ecgfounder', 'assess_quality'))
BOUNDARY_TOOLS = frozenset(('detect_beats', 'get_rate_rhythm', 'get_feature_profile',
                            'get_p_wave_proxy', 'get_qrs_proxy', 'get_beat_timeline_proxy',
                            'detect_beats_xqrs_proxy'))
# A second, unvalidated peak train can inform review but cannot by itself
# satisfy the existing local-boundary requirement for event closure.
CLOSURE_EVIDENCE_TOOLS = BOUNDARY_TOOLS - {'detect_beats_xqrs_proxy'}
BOUNDARY_PROMPT = """Boundary evidence is enabled. Choose a range wholly inside the retained signal.
Fractional seconds are allowed. For possible rhythm changes query surrounding beats on either lead.
Use get_beat_timeline_proxy for aligned raw RR, P candidates and QRS morphology proxies.
Use detect_beats_xqrs_proxy only as an experimental second peak detector; inspect raw peaks,
RR cleaning mask and detector status before interpreting any rhythm change.
Quality warnings and missing measurements express uncertainty. These measurements alone cannot establish AF termination.
For a possible event end, use finish_review.proposed_end_s, with measured evidence IDs and an explanation.
The runtime records the suggestion. It uses its separate event lifecycle to close tracked events.
"""
EVENT_CONTROL_PROMPT = """Boundary evidence is enabled. Choose a range wholly inside the retained arrived signal.
Fractional seconds are allowed. For a possible rhythm change query surrounding beats on either lead.
Use get_beat_timeline_proxy for aligned raw RR, P candidates and QRS morphology proxies.
Use detect_beats_xqrs_proxy only as an experimental second peak detector; compare its raw
peak train with other available measurements before interpreting RR changes.
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
detect_beats_xqrs_proxy is an unvalidated local second peak detector. Its raw peaks,
RR mask and failure status are review evidence, not an AF end label.
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


class WriteMemory(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    evidence_id: str = Field(min_length=1, description='Exact ID of a successful measurement shown in this review')


class RetrieveMemory(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    scope: Literal['evidence', 'events'] = 'evidence'
    start_s: float = Field(default=0., ge=0, allow_inf_nan=False)
    end_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    lead: str | None = Field(default=None, min_length=1, description='Evidence scope only')
    tool: str | None = Field(default=None, min_length=1, description='Evidence scope only')
    event_id: int | None = Field(default=None, ge=1, description='Events scope only')
    limit: int = Field(default=3, ge=1, le=6)


class EvidenceFocus(BaseModel):
    """An LLM-authored investigation question, not a rule-based rhythm label."""
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    issue: Literal['possible_onset', 'possible_end', 'possible_conflict',
                   'quality_issue', 'insufficient_evidence', 'no_specific_issue']
    observation: str = Field(min_length=1, max_length=768)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    unresolved_question: str = Field(min_length=1, max_length=512)
    next_tool: str = Field(min_length=1, max_length=64,
                           description='Name of a permitted measurement tool, retrieve_memory, or none')
    target_lead: str | None = None
    target_start_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    target_end_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expected_information: str = Field(min_length=1, max_length=512)


def _retrieve_memory(query, ledger, tracker, start, cursor):
    end_s = cursor if query.end_s is None else query.end_s
    if not query.start_s < end_s <= cursor:
        raise ValueError('Requested memory interval must end at or before the review cursor')
    if query.scope == 'events':
        if query.lead is not None or query.tool is not None:
            raise ValueError('Lead and tool filters require evidence scope')
        return tracker.retrieve(query.start_s, end_s, available_through_s=start,
                                limit=query.limit, event_id=query.event_id)
    if query.event_id is not None:
        raise ValueError('Event ID filter requires events scope')
    return ledger.retrieve(query.start_s, end_s, available_through_s=start,
                           limit=query.limit, lead=query.lead, tool=query.tool)


def _tool_exchange(messages, step, name, args, result):
    call_id = f'call-{step}'
    messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(args, allow_nan=False)}}]})
    messages.append({'role': 'tool', 'tool_call_id': call_id,
                     'content': json.dumps(result, allow_nan=False)})


def factual_history(windows):
    """Expose measured facts and past decisions while keeping the full audit ledger intact."""
    summaries = []
    for window in windows:
        measurements = []
        for source in window['acquired']:
            item = {k: v for k, v in source.items() if k != 'result'}
            result = source.get('result') or {}
            for key in ('af_score', 'afl_score', 'rate_bpm', 'rr_cv', 'raw', 'cleaned',
                        'removed_fraction', 'detector_status'):
                if key in result:
                    item[key] = result[key]
            if source.get('retained_by_model'):
                item['result'] = deepcopy(result)
            if source['tool'] == 'assess_quality':
                item['quality'] = result
            measurements.append(item)
        action = window.get('llm_action') or {}
        summaries.append(dict(window_s=window['window_s'], decision=window['decision'],
                              action=action.get('action', action.get('decision')),
                              acquired=measurements, evidence_ids=window['evidence_ids']))
    return summaries


class ReActReview:
    def __init__(self, model, *, max_rounds=5, history_windows=6, boundary_tools=False,
                 live_scoring=False, event_control=False,
                 unified_event_decision=False, separate_final_decision=False,
                 rule_safety_fallback=False, memory_write=False, memory_retrieve=False,
                 llm_window_interpretation=False, llm_tool_max_calls=None,
                 investigation_checkpoint=False, fact_history=False, event_review_mode=None,
                 compact_final_evidence=True, split_event_review=False, measured_event_review=False, staged_event_review=False,
                 window_assessment_mode='legacy', window_band_profile=None, end_range_prompt=False,
                 confirm_end_with_followup=False, compare_rhythm_leads=False, bound_window_choices=False,
                 direction_only_reference=False, persistent_event_evidence=False, adaptive_acquisition=False,
                 evidence_menu=False, focused_evidence_menu=False, boundary_pair_menu=False,
                 advisory_reference=False):
        if type(max_rounds) is not int or not 1 <= max_rounds <= 16:
            raise ValueError('max_rounds must be 1..16')
        if type(history_windows) is not int or not 0 <= history_windows <= 24:
            raise ValueError('history_windows must be 0..24')
        if event_control and not boundary_tools:
            raise ValueError('Event control requires boundary tools')
        if unified_event_decision and not event_control:
            raise ValueError('Unified event decision requires event control')
        if separate_final_decision and not unified_event_decision:
            raise ValueError('Separated final decision requires unified event decision')
        if llm_window_interpretation and not separate_final_decision:
            raise ValueError('LLM window interpretation requires the separate final decision phase')
        if llm_tool_max_calls is not None and (type(llm_tool_max_calls) is not int or llm_tool_max_calls < 1):
            raise ValueError('llm_tool_max_calls must be a positive integer')
        if investigation_checkpoint and not llm_window_interpretation:
            raise ValueError('Investigation checkpoints require LLM window interpretation')
        if event_review_mode not in (None, 'fixed', 'autonomous'):
            raise ValueError('event_review_mode must be fixed or autonomous')
        if event_review_mode and (not llm_window_interpretation or rule_safety_fallback or investigation_checkpoint):
            raise ValueError('Autonomous events require separate LLM interpretation without rule fallback or focus checkpoint')
        self.event_review_mode = event_review_mode
        from .window_semantics import MODES
        if window_assessment_mode not in MODES or (window_assessment_mode != 'legacy' and not staged_event_review):
            raise ValueError('Window assessment experiment requires a valid staged mode')
        self.window_band_profile = window_band_profile
        self.window_assessment_mode = window_assessment_mode
        if type(end_range_prompt) is not bool or (end_range_prompt and window_assessment_mode != 'union_reference'):
            raise ValueError('The experimental end-range prompt requires union_reference mode')
        self.end_range_prompt = end_range_prompt
        if type(confirm_end_with_followup) is not bool or (confirm_end_with_followup and window_assessment_mode!='union_reference'):
            raise ValueError('End follow-up requires the union_reference window assessment')
        self.confirm_end_with_followup = confirm_end_with_followup
        if type(compare_rhythm_leads) is not bool or (compare_rhythm_leads and not confirm_end_with_followup):
            raise ValueError('Paired-lead follow-up requires the evidence-confirmed event policy')
        self.compare_rhythm_leads = compare_rhythm_leads
        if type(bound_window_choices) is not bool or (bound_window_choices and not confirm_end_with_followup):
            raise ValueError('Bounded window choices require the evidence-confirmed policy')
        self.bound_window_choices = bound_window_choices
        if type(direction_only_reference) is not bool or (direction_only_reference and window_assessment_mode != 'union_reference'):
            raise ValueError('Direction-only presentation requires the union reference profile')
        self.direction_only_reference = direction_only_reference
        if type(persistent_event_evidence) is not bool or (
            persistent_event_evidence and not (staged_event_review and confirm_end_with_followup)
        ):
            raise ValueError('Persistent event evidence requires staged end follow-up')
        self.persistent_event_evidence = persistent_event_evidence
        if type(adaptive_acquisition) is not bool or (
            adaptive_acquisition and not (persistent_event_evidence and event_review_mode == 'fixed'
                                          and separate_final_decision)
        ):
            raise ValueError('Adaptive acquisition requires fixed review with persistent staged evidence')
        self.adaptive_acquisition = adaptive_acquisition
        if type(evidence_menu) is not bool or (evidence_menu and not adaptive_acquisition):
            raise ValueError('The evidence menu requires adaptive acquisition')
        self.evidence_menu = evidence_menu
        if type(focused_evidence_menu) is not bool or (focused_evidence_menu and not evidence_menu):
            raise ValueError('The focused evidence menu requires the evidence menu')
        self.focused_evidence_menu = focused_evidence_menu
        if type(boundary_pair_menu) is not bool or (boundary_pair_menu and not focused_evidence_menu):
            raise ValueError('The boundary pair menu requires focused evidence acquisition')
        self.boundary_pair_menu = boundary_pair_menu
        if type(advisory_reference) is not bool or (
            advisory_reference and not (focused_evidence_menu and bound_window_choices
                                        and compare_rhythm_leads and window_assessment_mode == 'union_reference')
        ):
            raise ValueError('Advisory reference requires the bounded paired-lead observation policy')
        self.advisory_reference = advisory_reference
        self.staged_event_review = staged_event_review
        self.measured_event_review = measured_event_review or staged_event_review
        self.split_event_review = split_event_review or self.measured_event_review
        self.compact_final_evidence = compact_final_evidence
        self.investigation_checkpoint = investigation_checkpoint
        self.fact_history = fact_history
        self.evidence_focus = None
        self.llm_window_interpretation = llm_window_interpretation
        self.llm_tool_max_calls = llm_tool_max_calls
        self.model, self.max_rounds, self.history_windows = model, max_rounds, history_windows
        self.boundary_tools = boundary_tools
        self.live_scoring = boundary_tools and live_scoring
        self.event_control = event_control
        self.unified_event_decision = unified_event_decision
        self.separate_final_decision = separate_final_decision
        self.rule_safety_fallback = rule_safety_fallback
        self.memory_write = memory_write and history_windows > 0
        self.memory_retrieve = memory_retrieve
        self.pending_close = None
        self.last_action = None
        self.history = []
        self.identity = None
        self.counts = {'llm_calls': 0, 'llm_ms': 0., 'llm_prompt_tokens': 0,
                       'llm_completion_tokens': 0}

    def _legacy_context(self, session, tracker, identity, start, end, primary, primary_lead, secondary, call_window, allowed):
        state = {'identity': identity, 'window_s': [start, end], 'primary_lead': primary_lead,
                 'available_leads': list(session.context.waveform.channels),
                 'secondary_lead': secondary, 'state': tracker.state,
                 'active_event': deepcopy(tracker.active),
                 'consecutive_support': tracker.support_count,
                 'consecutive_negative': tracker.negative_count,
                 'unknown_start_s': tracker.unknown_start,
                 'primary_evidence': deepcopy(primary),
                 'remaining_tool_calls': session.max_calls - len(session.calls)}
        if self.unified_event_decision:
            state['already_acquired_evidence'] = [
                {'evidence_id': r['evidence_id'], 'tool': r['tool'], 'lead': r['lead'],
                 'window_s': deepcopy(r['window_s']), 'status': r['status'],
                 'result': deepcopy(r['result'])}
                for r in session.calls if r['available_at_s'] <= end]
        state['recent_windows'] = factual_history(self.history) if self.fact_history else deepcopy(self.history)
        # Active events may contain hours of intervals; expose bounded recent spans only.
        if state['active_event']:
            for key, value in state['active_event'].items():
                if isinstance(value, list):
                    state['active_event'][key] = value[-6:]
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
        if primary is None:
            prompt += (
                '\nThis review starts with no current-window measurements. Decide which tools, leads '
                'and elapsed intervals to measure, or finish if no measurement is needed. '
                'To supply current AF/AFL evidence to the window interpreter, request score_ecgfounder '
                'for the current window on a lead you choose. Use assess_quality when useful. '
                'An unscored window is unknown, and historical tool results apply to their own intervals.\n')
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(state, allow_nan=False)}]
        tools = [s for s in session.registry.function_schemas(include_experimental=self.boundary_tools)
                 if s['function']['name'] in allowed]
        finish_schema = FinishBoundaryReview if self.boundary_tools and not self.event_control else FinishReview
        final_schema = None
        if self.unified_event_decision:
            prompt = prompt + ('\nAfter measuring, use submit_event_decision as the sole final action. '
                'KEEP_OPEN means evidence supports continuation; DEFER means the end is unresolved; '
                'CLOSE requests termination only with local boundary evidence. For non-CLOSE, '
                'end_s must be null; event_id may name the current event. For CLOSE, cite the active event ID, actual '
                'evidence IDs, and an estimated boundary distinct from decision time.\n')
            final_schema = EventDecision
            if self.llm_window_interpretation:
                prompt = WINDOW_PROMPT.read_text(encoding='utf-8')
                final_schema = WindowEventDecision
            messages[0]['content'] = prompt
            tools.append({'type': 'function', 'function': {'name': 'submit_event_decision',
                'description': ('Submit the current-window rhythm interpretation and event action (KEEP_OPEN, DEFER, or CLOSE).' if self.llm_window_interpretation else 'Submit one explicit event-end decision and finish this review.'),
                'parameters': final_schema.model_json_schema()}})
            if self.separate_final_decision:
                prompt += ('\nMeasurement phase: request a measurement or call finish_measurements. '
                           'A separate final phase will always ask for exactly one event decision. '
                           'Do not submit an event decision during measurement.\n')
                messages[0]['content'] = prompt
                tools.append({'type': 'function', 'function': {'name': 'finish_measurements',
                    'description': 'Finish measurements now and proceed to the separate mandatory decision phase.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}})
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
        if self.memory_write:
            messages[0]['content'] += (
                '\nWhile reviewing, you may call write_memory with the exact evidence_id of a successful '
                'measurement shown in this review if later windows need its details. Select useful results '
                'such as RR or beat morphology. Scores and quality are recorded automatically. '
                'The runtime retains the selected tool result. '
                'Writing uses one interaction turn, no measurement budget, and does not finish review.\n')
            tools.append({'type': 'function', 'function': {'name': 'write_memory',
                          'description': 'Keep a measured result in recent window memory for later reviews.',
                          'parameters': WriteMemory.model_json_schema()}})
        if self.memory_retrieve:
            messages[0]['content'] += (
                '\nRecent window summaries are supplied. Use retrieve_memory when you need a previous measurement or an earlier event to '
                'resolve this review. Query absolute seconds and use the lead/tool filters for '
                'evidence or event_id for events. Historical scores describe their own intervals. '
                'Retrieval uses one interaction turn and no measurement budget. Cite returned '
                'evidence IDs where relevant and obtain current local boundary evidence for a close.\n')
            tools.append({'type': 'function', 'function': {'name': 'retrieve_memory',
                          'description': 'Find previously acquired measurements or tracked events '
                                         'in this recording and task, newest first.',
                          'parameters': RetrieveMemory.model_json_schema()}})
        if self.investigation_checkpoint:
            messages[0]['content'] += (
                '\nINVESTIGATION CHECKPOINT. After the first successful current rhythm '
                'measurement, you may record one evidence focus for this window. '
                'Describe your own observable concern or no_specific_issue; never invent '
                'a conflict merely to justify extra measurements. Cite actual evidence IDs. '
                'Propose ONE next tool and interval only when it can resolve a named question. '
                'You retain the choice to finish, retrieve memory, or gather more evidence. '
                'The checkpoint records your hypothesis; it does not assign a rhythm label '
                'and it never triggers EventTracker transitions.\n')
            tools.append({'type': 'function', 'function': {
                'name': 'record_evidence_focus',
                'description': 'Record a grounded suspected change, the unresolved question, '
                               'and one high-information next test (or none). Not a diagnosis.',
                'parameters': EvidenceFocus.model_json_schema()}})
        return state, messages, tools, finish_schema, final_schema

    def review(self, call, primary, secondary, tracker, config, session, emit, *,
               call_window=None, ledger=None, window_s=None, primary_lead=None, review_snapshot=None):
        self.pending_close = None
        self.last_action = None
        self.evidence_focus = None
        if primary is not None:
            window_s, primary_lead = primary['window_s'], primary['lead']
        if window_s is None or primary_lead is None:
            raise ValueError('Review requires current window coordinates and a primary lead')
        start, end = window_s
        c = session.evidence_context
        identity = {'run_id': c.run_id, 'task_id': c.task_id, 'record_id': c.record_id}
        if self.identity is not None and self.identity != identity:
            raise ValueError('Create a fresh reviewer for each run/task/record')
        if (self.memory_retrieve or self.llm_window_interpretation) and ledger is not None and ledger.context != c:
            raise ValueError('Memory retrieval requires this review record and task ledger')
        if self.memory_retrieve and ledger is None:
            raise ValueError('Memory retrieval requires this review record and task ledger')
        self.identity = identity
        stable = None
        allowed = ALLOWED | BOUNDARY_TOOLS if self.boundary_tools else ALLOWED
        if self.event_review_mode:
            if review_snapshot is None or ledger is None or call_window is None:
                raise ValueError('Autonomous review needs a scheduler snapshot, evidence ledger and retained window')
            state = dict(review_snapshot, identity=identity, primary_lead=primary_lead,
                         available_leads=list(session.context.waveform.channels),
                         remaining_tool_calls=session.max_calls-len(session.calls),
                         retained_window_s=[session.context.origin_s,end],
                         ecgfounder_scope='retained_10s_window' if self.live_scoring else 'current_window_only',
                         recent_windows=factual_history(self.history) if self.fact_history else deepcopy(self.history))
            if self.split_event_review and tracker.active:
                for field in ('confirmed_support_intervals','opposing_intervals','unknown_intervals'):
                    state['open_event'][field] = deepcopy(tracker.active[field][-3:])
            if self.adaptive_acquisition:
                primary = call_window('score_ecgfounder', dict(lead=primary_lead, start_s=start, end_s=end),
                                      trigger_reason='adaptive_primary')
                stable = active_acquisition.stable_action(primary, state, self.window_band_profile)
                allowed = allowed & active_acquisition.SUPPLEMENTARY_TOOLS
                state['remaining_tool_calls'] = session.max_calls-len(session.calls)
                state['already_acquired_evidence'] = [{
                    key: primary[key] for key in ('evidence_id', 'lead', 'window_s', 'status', 'result')
                }]
            contract = split_event if self.split_event_review else autonomous_decision
            if self.measured_event_review:
                contract = measured_event
            acquisition = {key: value for key, value in state.items() if key != 'end_review_memory'}
            if self.adaptive_acquisition:
                acquisition['recent_windows'] = [
                    {k: entry[k] for k in ('window_s', 'af_score', 'afl_score') if k in entry}
                    for entry in self.history[-3:]
                ]
            messages = [{'role':'system', 'content':contract.prompt(acquisition, 'acquire')},
                        {'role':'user', 'content':json.dumps(acquisition, allow_nan=False)}]
            tools = [tool for tool in session.registry.function_schemas(include_experimental=True)
                     if tool['function']['name'] in allowed]
            if self.split_event_review:
                tools = deepcopy(tools)
                for tool in tools:
                    fn = tool['function']
                    fn['description'] += ' ' + split_event.TOOL_GUIDE[fn['name']]
            if self.adaptive_acquisition:
                messages[0]['content'] = active_acquisition.acquisition_prompt()
            elif self.staged_event_review:
                messages[0]['content'] = staged_review.SYSTEM + '\n' + autonomous_decision._fragments()['acquire']['en'] + '\nAfter acquisition you will submit the window assessment, choose a lifecycle action and optionally investigate a boundary. An observed onset is optional.'
                if self.compare_rhythm_leads:
                    messages[0]['content'] += (' First inspect the current primary ECGFounder scores. A new high score in background '
                        'or a sudden low score in an open event can be transient. To check a suspected change, you can '
                        'call score_ecgfounder on the OTHER available ECG lead over the SAME ten seconds. '
                        'Compare max(AF,AFL) across leads; the two scores on one lead are subtypes of the same target. '
                        'Measure another lead when that comparison could change your decision. If the result is already '
                        'available, reuse it. End measurement when you have the useful evidence.')
            tools += [contract.final_tool(state),
                      {'type':'function','function':{'name':'finish_measurements',
                       'description':'Finish this round of measurement and submit the observation and event action.',
                       'parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
            for enabled, name, schema, desc in (
                    (self.memory_write, 'write_memory', WriteMemory, 'Keep a successful measurement in recent memory.'),
                    (self.memory_retrieve, 'retrieve_memory', RetrieveMemory, 'Retrieve earlier measurements or events.')):
                if enabled:
                    tools.append({'type':'function','function':{'name':name, 'description':desc,
                                                                'parameters':schema.model_json_schema()}})
        else:
            state, messages, tools, finish_schema, final_schema = self._legacy_context(
                session, tracker, identity, start, end, primary, primary_lead, secondary, call_window, allowed)
        if self.evidence_menu and stable is None:
            staged_review.acquire_from_menu(
                self, state, session, call_window, emit, dict(identity, cursor_s=end, step=0))
        visible_history = {key for window in self.history for key in window.get('evidence_ids', [])}
        retained_citations = visible_history if self.memory_write else set()
        retrieved_ids = set()
        writable_evidence = set()
        written = set()
        if self.memory_write:
            shown = {row['evidence_id'] for row in state.get('already_acquired_evidence', [])}
            if primary is not None:
                shown.add(primary['evidence_id'])
            writable_evidence = {r['evidence_id'] for r in session.calls
                                 if r['evidence_id'] in shown and r['status'] == 'ok'
                                 and r['available_at_s'] <= end}
        stop = 'round_limit'
        failed = False
        measurement_done = stable is not None or self.evidence_menu
        acquisition_errors = 0
        acquisition_failed = False
        decision_retries = 0
        aliases = {}
        final_context_added = False
        total_rounds = self.max_rounds + (1 if self.separate_final_decision else 0)
        recovery_turn = False
        recovery_used = False
        for step in range(total_rounds + (2 if self.measured_event_review else 0)):
            final_phase = self.separate_final_decision and (measurement_done or step >= self.max_rounds - 1)
            budget_exhausted = len(session.calls) >= session.max_calls
            if (self.separate_final_decision and budget_exhausted
                    and not self.memory_retrieve and not (writable_evidence - written)):
                final_phase = True
            if recovery_turn:
                final_phase = False
                recovery_turn = False
                measurement_done = True  # allow one acquisition turn before returning to final
            if final_phase and not final_context_added:
                aliases = {f'E{i}': row['evidence_id'] for i, row in enumerate(session.calls, 1)}
                if self.measured_event_review:
                    history_keys = sorted(key for key in visible_history if ledger.get(key)['tool'] == 'score_ecgfounder')
                    historical = dict.fromkeys(history_keys + sorted(retrieved_ids))
                    current_ids = set(aliases.values())
                    aliases.update({f'H{i}': key for i, key in enumerate(historical, 1) if key not in current_ids})
                    if self.staged_event_review:
                        state['recent_window_votes'] = {}
                        for prior in self.history:
                            observed = (prior.get('llm_action') or {}).get('window_assessment', {})
                            if observed.get('interpretation') in ('support', 'negative'):
                                for key in observed['directional_ids']:
                                    state['recent_window_votes'][key] = observed['interpretation']
                        self.last_action = staged_review.submit(self, state, aliases, ledger, session, emit,
                            dict(identity, cursor_s=end, step=step), call_window, tools, stable=stable)
                        failed = self.last_action['action']=='ACTION_ERROR'
                        stop = 'ACTION_ERROR' if failed else 'stable_primary' if stable else 'staged_review_complete'
                        break
                    messages = measured_event.final_context(dict(state, remaining_tool_calls=session.max_calls-len(session.calls)), aliases, ledger)
                    tools = [contract.final_tool(state, aliases) if t['function']['name']=='submit_event_decision'
                             else t for t in tools]
                elif self.event_review_mode:
                    messages[0]['content'] = contract.prompt(state, 'final')
                    entries = [{'id': alias, 'evidence_id': key} for alias, key in aliases.items()]
                    for entry, row in zip(entries, session.calls):
                        entry.update({k: row[k] for k in ('tool', 'lead', 'window_s', 'status')})
                        if not self.compact_final_evidence:
                            entry['result'] = row['result']
                        elif row['tool'] == 'score_ecgfounder' and row['status'] == 'ok':
                            entry['result'] = {k: row['result'].get(k) for k in ('af_score', 'afl_score')}
                    messages.append({'role': 'user', 'content': json.dumps({'phase': 'final_decision',
                                                                            'measurement_ids': entries})})
                else:
                    messages.append({'role': 'user', 'content': json.dumps({
                        'phase': 'final_decision',
                        'active_event_id': (tracker.active or {}).get('event_id'),
                        **({'evidence_focus': deepcopy(self.evidence_focus)} if self.investigation_checkpoint else {}),
                        'evidence_aliases': [{
                            'id': alias, 'tool': row['tool'], 'lead': row['lead'],
                            'window_s': row['window_s'], 'status': row['status'],
                            'result': row['result'], **({'warnings': row['warnings'], 'error': row['error'],
                            'available_at_s': row['available_at_s']} if self.llm_window_interpretation else {})}
                            for alias, row in zip(aliases, session.calls)],
                        'instruction': 'Submit one decision. You may cite these short IDs; '
                                       'the runtime resolves them to immutable evidence IDs.'}, allow_nan=False)})
                final_context_added = True
            began = perf_counter()
            audit = dict(identity, cursor_s=end, step=step)
            if self.llm_window_interpretation:
                audit['phase'] = ('evidence_recovery' if recovery_used and not final_phase else
                                  'retry' if decision_retries else 'final' if final_phase else 'acquisition')
            self.counts['llm_calls'] += 1
            try:
                try:
                    if self.separate_final_decision:
                        available_tools = [t for t in tools if
                            (t['function']['name'] == 'submit_event_decision') == final_phase]
                        if budget_exhausted and not final_phase:
                            available_tools = [t for t in available_tools if
                                t['function']['name'] not in allowed]
                        if self.investigation_checkpoint and not final_phase:
                            rhythm_observed = any(r['tool'] != 'assess_quality' and
                                                  overlaps_current(r, start, end) for r in session.calls)
                            if not rhythm_observed or self.evidence_focus is not None:
                                available_tools = [t for t in available_tools if
                                                   t['function']['name'] != 'record_evidence_focus']
                    else:
                        available_tools = ([t for t in tools if t['function']['name'] == 'submit_event_decision']
                                           if self.unified_event_decision and step == self.max_rounds - 1
                                           else tools)
                        if budget_exhausted and self.memory_retrieve:
                            available_tools = [t for t in available_tools if t['function']['name'] not in allowed]
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
                audit.update(action=name, arguments=deepcopy(args))
                if (self.separate_final_decision or name in ('write_memory', 'retrieve_memory')) and name not in {
                        t['function']['name'] for t in available_tools}:
                    raise ValueError('Tool is not available in this phase')
                if name == 'record_evidence_focus' and self.investigation_checkpoint:
                    try:
                        focus = EvidenceFocus.model_validate(args)
                        known = {r['evidence_id']: r for r in session.calls}
                        if (len(focus.evidence_ids) != len(set(focus.evidence_ids)) or
                                not all(k in known and overlaps_current(known[k], start, end)
                                        for k in focus.evidence_ids)):
                            raise ValueError('Cite distinct successful current-window evidence IDs')
                        if focus.next_tool not in ({t['function']['name'] for t in tools} & allowed) | {'retrieve_memory', 'none'}:
                            raise ValueError('Next tool must be permitted, retrieve_memory or none')
                        if focus.next_tool == 'retrieve_memory' and not self.memory_retrieve:
                            raise ValueError('Memory retrieval is not enabled')
                        if focus.target_lead is not None and focus.target_lead not in state['available_leads']:
                            raise ValueError('Next test lead is unavailable')
                        if ((focus.target_start_s is None) != (focus.target_end_s is None)):
                            raise ValueError('Specify both target times or neither')
                        if focus.next_tool in allowed and (focus.target_lead is None or focus.target_start_s is None):
                            raise ValueError('Specify the lead and interval for the proposed measurement')
                        if focus.next_tool == 'score_ecgfounder' and (
                                focus.target_end_s - focus.target_start_s != 10 or
                                (not self.live_scoring and (focus.target_start_s, focus.target_end_s) != (start, end))):
                            raise ValueError('Planned ECGFounder interval is outside the current scoring scope')
                        if focus.next_tool in allowed and focus.target_start_s is not None and not (
                                session.context.origin_s <= focus.target_start_s <
                                focus.target_end_s <= end):
                            raise ValueError('Next test lies outside arrived retained signal')
                    except ValueError as exc:
                        outcome = {'status': 'focus_rejected', 'error_message': str(exc)[:500]}
                    else:
                        self.evidence_focus = focus.model_dump()
                        outcome = {'status': 'focus_recorded', 'focus': deepcopy(self.evidence_focus)}
                    emit('llm_steps', dict(audit, **outcome))
                    _tool_exchange(messages, step, name, args, outcome)
                    if outcome['status'] == 'focus_rejected':
                        raise ValueError(outcome['error_message'])
                    acquisition_errors = 0
                    continue
                if name == 'write_memory':
                    try:
                        key = WriteMemory.model_validate(args).evidence_id
                        if key not in writable_evidence:
                            raise ValueError('Use an ID of a successful measurement shown in this review')
                    except ValueError as exc:
                        outcome = {'status': 'memory_write_rejected', 'error_message': str(exc)[:500]}
                    else:
                        outcome = {'status': 'memory_already_written' if key in written else 'memory_written',
                                   'evidence_id': key}
                        written.add(key)
                    emit('llm_steps', dict(audit, **outcome))
                    _tool_exchange(messages, step, name, args, outcome)
                    if outcome['status'] != 'memory_write_rejected':
                        acquisition_errors = 0
                    continue
                if name == 'retrieve_memory':
                    try:
                        query = RetrieveMemory.model_validate(args)
                        matches = _retrieve_memory(query, ledger, tracker, start, end)
                    except ValueError as exc:
                        outcome = {'status': 'memory_retrieval_rejected', 'error_message': str(exc)[:500]}
                        emit('llm_steps', dict(audit, **outcome))
                    else:
                        evidence_ids = [row['evidence_id'] for row in matches if query.scope == 'evidence']
                        event_ids = [row['event_id'] for row in matches if query.scope == 'events']
                        retrieved_ids.update(evidence_ids)
                        if self.measured_event_review and query.scope == 'events':
                            state.setdefault('retrieved_events', []).extend(matches)
                        outcome = {'status': 'memory_retrieved', 'scope': query.scope, 'matches': matches}
                        emit('llm_steps', dict(audit, status='memory_retrieved', count=len(matches),
                                               evidence_ids=evidence_ids, event_ids=event_ids))
                    _tool_exchange(messages, step, name, args, outcome)
                    if outcome['status'] == 'memory_retrieved':
                        acquisition_errors = 0
                    continue
                if name == 'finish_measurements' and self.separate_final_decision:
                    if args:
                        raise ValueError('finish_measurements takes no arguments')
                    emit('llm_steps', dict(audit, status='measurements_finished'))
                    measurement_done = True
                    continue
                if name == 'submit_event_decision' and self.unified_event_decision:
                    if self.event_review_mode:
                        validate_event = split_event.event if self.split_event_review else autonomous_decision.validate
                        visible = set(aliases.values()) | visible_history | retrieved_ids
                        if self.measured_event_review:
                            validate_event = measured_event.event
                            visible = set(aliases.values())
                        self.last_action = validate_event(
                            args, aliases, visible,
                            ledger, state, CLOSURE_EVIDENCE_TOOLS)
                        if self.split_event_review:
                            scheduling = split_event.appointment(self.model,state,self.last_action,self.counts,emit,audit)
                            self.last_action.update(scheduling)
                        self.last_action['action'] = self.last_action['event_action']
                        self.last_action['decision_source'] = 'llm'
                        stop = 'model_decision_' + self.last_action['action'].lower()
                        emit('llm_steps', dict(audit, status='decision_submitted', submission=deepcopy(self.last_action)))
                        break
                    submitted = final_schema.model_validate(args)
                    action = submitted.action if self.llm_window_interpretation else submitted.decision
                    if aliases:
                        submitted.evidence_ids = [aliases.get(key, key) for key in submitted.evidence_ids]
                    known = {r['evidence_id']: r for r in session.calls}
                    if (len(set(submitted.evidence_ids)) != len(submitted.evidence_ids) or
                            not set(submitted.evidence_ids) <= known.keys() | visible_history | retrieved_ids):
                        raise ValueError('Event decision cites unavailable or duplicate evidence')
                    if self.llm_window_interpretation:
                        validate_grounding(submitted, known, ledger, start, end, tracker.active, CLOSURE_EVIDENCE_TOOLS)
                    if action == 'CLOSE':
                        close = CloseEvent.model_validate({
                            'event_id': submitted.event_id, 'end_s': submitted.end_s,
                            'evidence_ids': submitted.evidence_ids, 'reason': submitted.reason})
                        active = tracker.active
                        if (active is None or active['confirmed_at_s'] is None
                                or close.event_id != active['event_id']
                                or not active['reported_intervals'][-1][0] < close.end_s <= end
                                or not any(k in known and known[k]['tool'] in CLOSURE_EVIDENCE_TOOLS
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
                                           decision=action,
                                           evidence_ids=submitted.evidence_ids))
                    self.last_action = {'decision': action, 'reason': submitted.reason,
                                        'evidence_ids': list(submitted.evidence_ids)}
                    if self.llm_window_interpretation:
                        self.last_action.update(submitted.model_dump(), decision_source='llm')
                    stop = 'model_decision_' + action.lower()
                    break
                if name == 'end_confirmed_event' and self.event_control:
                    close = CloseEvent.model_validate(args)
                    active = tracker.active
                    known = {r['evidence_id']: r for r in session.calls}
                    if (active is None or active['confirmed_at_s'] is None
                            or close.event_id != active['event_id']
                            or not active['reported_intervals'][-1][0] < close.end_s <= end
                            or len(set(close.evidence_ids)) != len(close.evidence_ids)
                            or not set(close.evidence_ids) <= known.keys() | retained_citations | retrieved_ids
                            or not any(k in known and known[k]['tool'] in CLOSURE_EVIDENCE_TOOLS
                                       and known[k]['status'] == 'ok'
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
                    if not set(finish.evidence_ids) <= known.keys() | retained_citations | retrieved_ids:
                        raise ValueError('Finish cites unknown evidence')
                    recorded = dict(audit, status='stopped', reason=finish.reason)
                    if self.boundary_tools:
                        recorded['evidence_ids'] = finish.evidence_ids
                    if self.boundary_tools and not self.event_control and finish.proposed_end_s is not None:
                        proposed = finish.proposed_end_s
                        active = tracker.active
                        if (not active or active['confirmed_at_s'] is None
                                or not max(session.context.origin_s, active['estimated_start_s']) < proposed <= end
                                or not any(k in known and known[k]['tool'] in CLOSURE_EVIDENCE_TOOLS
                                           and known[k]['status'] == 'ok'
                                           and known[k]['window_s'][0] <= proposed < known[k]['window_s'][1]
                                           for k in finish.evidence_ids)):
                            raise ValueError('Proposed end needs an active event and successful local evidence at the boundary')
                        recorded.update(proposed_end_s=proposed, proposed_for_event_id=active['event_id'])
                    emit('llm_steps', recorded)
                    stop = 'model_keep_event_open' if self.event_control else 'model_finish'
                    break
                if name not in allowed:
                    raise ValueError('Tool is not enabled for this review')
                if budget_exhausted:
                    stop = 'tool_budget'; break
                # Validate before dispatch; even rejected requests receive a bounded audit entry.
                parsed = session.registry._tools[name].arguments.model_validate(args)
                if not session.context.origin_s <= parsed.start_s < parsed.end_s <= end:
                    raise ValueError('Measurement is outside the retained elapsed waveform')
                current_only = not self.boundary_tools or (name == 'score_ecgfounder' and not self.live_scoring)
                if current_only and (parsed.start_s, parsed.end_s) != (start, end):
                    raise ValueError('Only the arrived current window is retained')
                result = call_window(name, parsed.model_dump()) if self.boundary_tools else call(name, parsed.lead)
                if self.memory_write and result['status'] == 'ok' and result['available_at_s'] <= end:
                    writable_evidence.add(result['evidence_id'])
                acquisition_errors = 0
                emit('llm_steps', dict(audit, status='observed', evidence_id=result['evidence_id']))
                _tool_exchange(messages, step, name, args, result)
                messages.append({'role': 'user', 'content': json.dumps({
                    'remaining_tool_calls': session.max_calls - len(session.calls)})})
            except Exception as exc:
                emit('llm_steps', dict(audit, status='error', error_type=type(exc).__name__,
                                       error_message=str(exc)[:500]))
                if self.separate_final_decision:
                    if not final_phase:
                        acquisition_errors += 1
                        retry = (isinstance(exc, (ValueError, KeyError, TypeError))
                                 and acquisition_errors < 2 and step + 1 < self.max_rounds - 1)
                        if not retry:
                            acquisition_failed = measurement_done = True
                        status = 'acquisition_retry' if retry else 'ACQUISITION_FAILURE'
                        emit('llm_steps', dict(audit, status=status, consecutive_errors=acquisition_errors,
                                               error_message=str(exc)[:500]))
                        instruction = ('Choose one listed tool for this phase.' if retry else
                                       'Proceed to final decision using available evidence.')
                        messages.append({'role': 'user', 'content': f'{status}: {str(exc)[:500]}. {instruction}'})
                        continue
                    if (self.measured_event_review and isinstance(exc, measured_event.BoundaryEvidenceNeeded)
                            and not recovery_used and not decision_retries and not budget_exhausted):
                        recovery_used = recovery_turn = True
                        final_context_added = False
                        messages[0]['content'] = contract.prompt(state, 'acquire')
                        messages.append({'role':'user', 'content':'Boundary rejected: '+str(exc)+
                                         ' One acquisition turn remains. Choose one measurement, retrieval, or finish_measurements.'})
                        emit('llm_steps', dict(audit, status='evidence_recovery_requested'))
                        continue
                    decision_retries += 1
                    if decision_retries <= 1 and self.event_review_mode:
                        messages[0]['content'] = contract.prompt(state, 'final', retry=True)
                        detail = 'Rejected: ' + str(exc)[:500]
                        if self.measured_event_review:
                            detail += ' Repair with submit_event_decision only. Previous submission: ' + json.dumps(audit.get('arguments', {}))
                        messages.append({'role': 'user', 'content': detail})
                        continue
                    if decision_retries <= 1:
                        messages.append({'role': 'user', 'content':
                            'Final action rejected: ' + str(exc)[:500] +
                            '. Retry exactly once with submit_event_decision. '
                            'Use numeric active_event_id=' +
                            str((tracker.active or {}).get('event_id')) +
                            ' (or null if no active event). '
                            'For KEEP_OPEN or DEFER set end_s to null; cite only available evidence IDs.'})
                        continue
                    stop, failed = 'ACTION_ERROR', True
                    self.last_action = {'decision': 'ACTION_ERROR', 'reason': str(exc)[:500],
                                        'evidence_ids': []}
                    break
                # Preserve the original policy's failure semantics for paired comparison.
                stop, failed = 'model_or_action_error', True
                break
        if self.separate_final_decision and self.last_action is None:
            stop, failed = 'ACTION_ERROR', True
            self.last_action = {'decision': 'ACTION_ERROR',
                                'reason': 'No valid final event decision', 'evidence_ids': []}
        acquired = {(r['tool'], r['lead']): deepcopy(r) for r in session.calls
                    if r['window_s'] == [start, end]}
        def existing(tool, lead):
            return deepcopy(acquired.get((tool, lead)))
        primary = existing('score_ecgfounder', primary_lead)
        if self.llm_window_interpretation:
            if failed:
                self.pending_close = None
                self.last_action.update(action='ACTION_ERROR', interpretation=None if self.event_review_mode else 'unavailable',
                                        event_id=(tracker.active or {}).get('event_id'), end_s=None,
                                        decision_source='llm_action_error')
                decision = ('ACTION_ERROR' if self.event_review_mode else 'unavailable', False, 'ACTION_ERROR')
            else:
                checked = any(overlaps_current(r, start, end) for r in session.calls)
                if self.event_review_mode:
                    decision = (self.last_action['assessment'], checked, self.last_action['reason'])
                else:
                    decision = (self.last_action['interpretation'], checked, self.last_action['reason'])
        elif failed:
            decision = ('unavailable', False, stop)
        else:
            decision = decide_event(existing, primary, secondary, tracker, config)
        emit('llm_steps', dict(identity, cursor_s=end, status='review_complete', stop_reason=stop,
                              decision=decision[0], interpreter_reason=decision[2], acquisition_failed=acquisition_failed,
                              llm_action=deepcopy(self.last_action),
                              remaining_tool_calls=session.max_calls-len(session.calls)))
        acquired_summary = []
        for result in session.calls:
            item = {key: result[key] for key in ('tool', 'lead', 'status')}
            if self.boundary_tools:
                item.update(window_s=deepcopy(result['window_s']), evidence_id=result['evidence_id'])
            if (not self.boundary_tools or result['tool'] not in BOUNDARY_TOOLS
                    or result['evidence_id'] in written
                    or (self.fact_history and result['tool'] in {'get_rate_rhythm', 'detect_beats', 'assess_quality'})):
                item['result'] = deepcopy(result['result'])
            if self.event_review_mode and result['evidence_id'] in written:
                item['retained_by_model'] = True
            acquired_summary.append(item)
        self.history.append({'window_s': [start, end], 'decision': decision[0],
                             'llm_action': deepcopy(self.last_action),
                             'af_score': primary['result'].get('af_score') if primary else None,
                             'afl_score': primary['result'].get('afl_score') if primary else None,
                             'acquired': acquired_summary,
                             'evidence_ids': [r['evidence_id'] for r in session.calls]})
        self.history = self.history[-self.history_windows:] if self.history_windows else []
        return decision
