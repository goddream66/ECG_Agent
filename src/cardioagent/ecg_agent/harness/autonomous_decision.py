"""Contract used by both event-ledger arms. Only the ledger executes decisions."""
from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


PROMPT_PATH = Path(__file__).resolve().parents[1] / 'prompts/autonomous_event_v1.json'


@lru_cache(maxsize=1)
def _fragments():
    return json.loads(PROMPT_PATH.read_text(encoding='utf-8'))['fragments']


def prompt(snapshot, phase, *, retry=False):
    keys = ['core', 'open' if snapshot['open_event'] else 'no_open']
    if snapshot['pending_question']:
        keys.append('resume')
    keys += [phase, 'stream_end' if snapshot['stream_end'] else snapshot['review_mode'] + '_time']
    if retry:
        keys.append('retry')
    return '\n\n'.join(_fragments()[key]['en'] for key in keys)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True, allow_inf_nan=False)


class Point(StrictModel):
    kind: Literal['localized']
    time_s: float = Field(ge=0)


class Bracket(StrictModel):
    kind: Literal['bracketed']
    range_s: list[float] = Field(min_length=2, max_length=2)


class UnobservedOnset(StrictModel):
    kind: Literal['onset_unobserved']
    first_supported_interval_s: list[float] = Field(min_length=2, max_length=2)


class Pending(StrictModel):
    question: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)


class EvidenceAttribution(StrictModel):
    evidence_id: str = Field(min_length=1)
    role: Literal['SUPPORTS_TARGET', 'AGAINST_TARGET', 'CONTEXT_ONLY']


class Decision(StrictModel):
    assessment: Literal['SUPPORTS_TARGET', 'AGAINST_TARGET', 'UNCERTAIN']
    uncertainty_cause: Literal['missing', 'unusable', 'conflicting', 'insufficient'] | None
    event_action: Literal['OPEN', 'RECORD_ONLY', 'DEFER_DECISION', 'CLOSE']
    boundary: Point | Bracket | UnobservedOnset | None
    evidence_ids: list[str] = Field(max_length=32)
    evidence_attributions: list[EvidenceAttribution] = Field(max_length=32)
    pending: Pending | None
    next_review_s: float | None = Field(ge=0)
    reason: str = Field(min_length=1, max_length=4096)

    @model_validator(mode='after')
    def check_fields(self, info: ValidationInfo):
        if (self.assessment == 'UNCERTAIN') != (self.uncertainty_cause is not None):
            raise ValueError('Give an uncertainty cause exactly when assessment is UNCERTAIN')
        open_followup = self.event_action == 'OPEN' and (info.context or {}).get('measured_contract')
        if not open_followup and (self.event_action == 'DEFER_DECISION') != (self.pending is not None):
            raise ValueError('Only a deferred question requires pending details')
        if (self.event_action in ('OPEN', 'CLOSE')) != (self.boundary is not None):
            raise ValueError('OPEN and CLOSE require a boundary; other actions use null')
        if self.event_action == 'OPEN' and self.assessment != 'SUPPORTS_TARGET':
            raise ValueError('Opening requires positive rhythm evidence')
        if self.event_action == 'CLOSE' and self.assessment != 'AGAINST_TARGET':
            raise ValueError('Closing requires evidence against continued AF/AFL')
        return self


class IdleDecision(Decision):
    event_action: Literal['OPEN', 'RECORD_ONLY', 'DEFER_DECISION']


class TrackingDecision(Decision):
    event_action: Literal['CLOSE', 'RECORD_ONLY', 'DEFER_DECISION']
    boundary: Point | Bracket | None


class ResumeIdle(IdleDecision):
    pending_resolution: Literal['resolved', 'withdrawn', 'still_pending']


class ResumeTracking(TrackingDecision):
    pending_resolution: Literal['resolved', 'withdrawn', 'still_pending']


def submission_type(snapshot):
    if snapshot['pending_question']:
        return ResumeTracking if snapshot['open_event'] else ResumeIdle
    return TrackingDecision if snapshot['open_event'] else IdleDecision


def final_tool(snapshot):
    schema = submission_type(snapshot)
    return {'type': 'function', 'function': {'name': 'submit_event_decision',
            'description': 'Record the observed rhythm, event action and next review.',
            'parameters': schema.model_json_schema()}}


def bounds(boundary):
    if isinstance(boundary, Point):
        return boundary.time_s, boundary.time_s
    if isinstance(boundary, Bracket):
        return tuple(boundary.range_s)
    return tuple(boundary.first_supported_interval_s)


def is_rhythm(row):
    return (row['status'] == 'ok' and bool(row['window_s'])
            and row['tool'] not in ('assess_quality', 'assess_template_quality')
            and not row['result'].get('detector_error'))


def overlap(span, start, end):
    return span[0] < end and start < span[1]


def timed_beats(row):
    data = row['result']
    times = data.get('r_peaks_s') or [beat.get('r_time_s') for beat in data.get('beats', [])]
    return [t for t in times if isinstance(t, (int, float)) and row['window_s'][0] <= t < row['window_s'][1]]


def validate(args, aliases, visible_ids, ledger, snapshot, boundary_tools, *, measured_contract=False):
    decision = submission_type(snapshot).model_validate(args, context=dict(measured_contract=measured_contract))
    new_followup = measured_contract and decision.event_action == 'OPEN' and decision.pending is not None
    if snapshot['pending_question'] and not new_followup and (decision.pending_resolution == 'still_pending') != (decision.pending is not None):
        raise ValueError('Choose still_pending when deferring the existing question, otherwise resolved or withdrawn')
    ids = [aliases.get(key, key) for key in decision.evidence_ids]
    if len(ids) != len(set(ids)) or not set(ids) <= visible_ids:
        raise ValueError('Cite distinct measurements visible in this review')
    roles = {aliases.get(item.evidence_id, item.evidence_id): item.role
             for item in decision.evidence_attributions}
    if len(roles) != len(decision.evidence_attributions) or set(ids) != roles.keys():
        raise ValueError('Assign exactly one role to each cited measurement')
    rows = [ledger.get(key) for key in ids]
    start, end = snapshot['window_s']
    if any(r['available_at_s'] > end or (r['window_s'] and r['window_s'][1] > end) for r in rows):
        raise ValueError('Evidence must already have arrived')
    directional = {'SUPPORTS_TARGET': [], 'AGAINST_TARGET': []}
    current_spans = {role: [] for role in directional}
    measured_roles = {}
    for key, row in zip(ids, rows):
        role = roles[key]
        if role == 'CONTEXT_ONLY':
            continue
        if not is_rhythm(row):
            raise ValueError('Quality and unsuccessful rhythm measurements require CONTEXT_ONLY')
        measurement = (row['tool'], row['lead'], tuple(row['window_s']))
        if measured_roles.setdefault(measurement, role) != role:
            raise ValueError('Repeated copies of the same measurement cannot have opposite roles')
        directional[role].append(row)
        a, b = row['window_s']
        if overlap((a,b), start, end):
            current_spans[role].append([max(start,a), min(end,b)])
    if decision.assessment in current_spans and not current_spans[decision.assessment]:
        raise ValueError('Directional observation needs a matching rhythm attribution in the assessed interval')
    conflict = any(overlap(a, *b) for a in current_spans['SUPPORTS_TARGET'] for b in current_spans['AGAINST_TARGET'])
    if conflict and decision.uncertainty_cause != 'conflicting':
        raise ValueError('Opposite roles overlapping in this interval require UNCERTAIN with cause conflicting')
    if not conflict and decision.uncertainty_cause == 'conflicting':
        raise ValueError('Conflict needs two distinct rhythm measurements with overlapping opposite roles')
    if snapshot['stream_end']:
        if decision.next_review_s is not None:
            raise ValueError('At stream end use next_review_s=null')
    elif decision.next_review_s is None or decision.next_review_s <= end:
        raise ValueError('Choose an absolute next_review_s later than the cursor')
    elif snapshot['review_mode'] == 'fixed' and decision.next_review_s != snapshot['scheduled_review_s']:
        raise ValueError('Use the displayed scheduled_review_s in fixed mode')
    elif decision.next_review_s > snapshot['latest_review_s']:
        raise ValueError('Requested review is beyond the displayed retention horizon')
    if decision.boundary is not None:
        a, b = bounds(decision.boundary)
        if a < snapshot['earliest_boundary_s'] or b > end or b < a:
            raise ValueError('Boundary is outside the admissible episode timeline')
        rhythm = [r for r in rows if is_rhythm(r)]
        local = [r for r in rhythm if r['tool'] in boundary_tools
                 and any(r['result'].get(k) for k in ('beats', 'r_peaks_s', 'peak'))]
        if measured_contract and not isinstance(decision.boundary, UnobservedOnset):
            from .measured_event import validate_boundary
            validate_boundary(decision.boundary, decision.event_action, directional)
        elif isinstance(decision.boundary, Point):
            if not any(times and min(times) < a < max(times) for times in map(timed_beats, local)):
                raise ValueError('A point needs a local measured change with beats before and after it; otherwise bracket it')
        elif isinstance(decision.boundary, UnobservedOnset):
            if b <= a or not any(r['window_s'][0] <= a < b <= r['window_s'][1]
                                 for r in directional['SUPPORTS_TARGET']):
                raise ValueError('Cite positive rhythm evidence covering the first supported interval')
        else:
            # A local measurement spanning the change, or separate measured
            # intervals anchoring each side, can support a bracket.
            spans = sorted(set(tuple(r['window_s']) for r in rhythm))
            anchored = any(x <= a <= y <= z <= b <= w for x, y in spans for z, w in spans)
            if b <= a or not (anchored or any(r['window_s'][0] <= a < b <= r['window_s'][1] for r in local)):
                raise ValueError('Bracket needs a local measurement or separate rhythm measurements anchoring both sides')
        end_estimate = (a + b) / 2 if measured_contract else a
        if decision.event_action == 'CLOSE' and end_estimate <= snapshot['open_event']['estimated_start_s']:
            raise ValueError('End must follow the registered onset')
    payload = decision.model_dump()
    payload['evidence_ids'] = ids
    payload['evidence_attributions'] = [dict(evidence_id=key, role=roles[key]) for key in ids]
    payload['support_intervals'] = current_spans['SUPPORTS_TARGET']
    payload['opposing_intervals'] = current_spans['AGAINST_TARGET']
    if isinstance(decision.boundary, UnobservedOnset):
        a, b = bounds(decision.boundary)
        payload['first_support_available_at_s'] = min(r['available_at_s'] for r in directional['SUPPORTS_TARGET']
                                                       if r['window_s'][0] <= a < b <= r['window_s'][1])
    return payload
