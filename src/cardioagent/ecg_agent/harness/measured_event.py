"""AF/AFL event contract for the v1.6 paired experiment."""
import json
from math import isfinite

from . import autonomous_decision as legacy
from . import split_event

FOUNDER = 'score_ecgfounder'
PAIR = ('af_score', 'afl_score')
DIRECTIONS = ('SUPPORTS_TARGET', 'AGAINST_TARGET')
OTHER_ROLES = ('CONTEXT_ONLY', 'NEUTRAL', 'INDETERMINATE')


class BoundaryEvidenceNeeded(ValueError):
    """The model may spend one remaining measurement turn to resolve this error."""


def prompt(state, phase, *, retry=False):
    parts = [legacy._fragments()['core']['en']]
    if state['open_event']:
        parts.append('An event is open. If you suspect an end, compare the latest credible support with current contrary evidence. Retrieve earlier scores if needed. A change between AF and AFL continues the episode.')
    else:
        parts.append('There is no open event. OPEN when you judge current evidence warrants registering AF or AFL. If onset was never observed, use onset_unobserved and give the actual first supported interval.')
    if state['pending_question']:
        parts.append(legacy._fragments()['resume']['en'])
    parts.append('Founder scores are uncalibrated. Consider BOTH absolute values. Merely exceeding the other subtype score gives no positive evidence. To weigh against the union explain why both subtypes are disfavored. No score threshold forces an action. Quality informs reliability. Regular RR does not exclude AFL. Beat-derived tools share peaks and cannot assign target direction in this experiment.')
    if phase == 'acquire':
        parts.append(legacy._fragments()['acquire']['en'])
        parts.append('Choose a check that could change your judgment. The beat timeline already contains RR and morphology proxies. For a suspected boundary, compare distinct arrived intervals.')
    else:
        parts.append('Submit submit_event_decision once. Cite measurement IDs. The runtime binds both actual Founder scores together. Assign one role per measurement. NEUTRAL means a usable observation with no target direction, INDETERMINATE means you cannot interpret it, CONTEXT_ONLY gives background. Only successful Founder pairs can have directional roles. Describe the decisive values and times in reason. UNCERTAIN needs a cause. DEFER_DECISION needs a question and the evidence you would seek. OPEN may also carry a follow-up. RECORD_ONLY and CLOSE have no pending field. Scheduling is separate.')
        parts.append('Only OPEN and CLOSE use boundary. For bracketed, set range_s to [start of the earlier directional window, end of the later directional window]; the two windows must not overlap. OPEN goes from against to support; CLOSE from support to against. These aggregate windows cannot establish an exact second. If the end lacks anchors, defer with a concrete boundary question.')
    if state['stream_end']:
        parts.append('The recording has ended. An unobserved end remains right-censored.')
    if retry:
        parts.append('Repair only the rejected submission using the displayed evidence. The only callable function in this phase is submit_event_decision.')
    return '\n\n'.join(parts)


def final_tool(state):
    tool = split_event.final_tool(state)
    schema = tool['function']['parameters']
    props = schema['$defs']['EvidenceAttribution']['properties']
    props.pop('field')
    props['role']['enum'] = list(DIRECTIONS + OTHER_ROLES)
    props['fields'] = {'type': 'array', 'items': {'type': 'string'},
                       'description': 'Optional keys from the displayed measured fields. Founder always binds both subtype scores.'}
    boundary = schema['properties']['boundary']
    boundary['anyOf'] = [option for option in boundary['anyOf'] if option.get('$ref') != '#/$defs/Point']
    schema['$defs'].pop('Point', None)
    return tool


def numeric_fields(row):
    """Expose actual finite values, retaining their path into the immutable result."""
    fields = {}
    def visit(data, prefix):
        if type(data) in (int, float) and isfinite(data):
            fields[prefix] = data
        elif isinstance(data, dict):
            for key, value in data.items():
                visit(value, f'{prefix}.{key}' if prefix else key)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            for i in sorted({0, len(data)-1}):
                visit(data[i], f'{prefix}.{i}')
    visit(row['result'], '')
    return fields


def final_context(state, aliases, ledger):
    rows = []
    for alias, key in aliases.items():
        row = ledger.get(key)
        rows.append(dict(id=alias, tool=row['tool'], lead=row['lead'], window_s=row['window_s'],
                         status=row['status'], warnings=row.get('warnings', []),
                         fields=numeric_fields(row), direction_allowed=row['tool']==FOUNDER and row['status']=='ok'))
    data = {k: v for k, v in state.items() if k != 'recent_windows'}
    data['recent_decisions'] = [{k: w.get(k) for k in ('window_s', 'action', 'decision')}
                                for w in state.get('recent_windows', [])]
    for entry in rows:
        if entry['tool'] == FOUNDER:
            entry['overlapping_quality_ids'] = [q['id'] for q in rows if q['tool']=='assess_quality'
                and q['lead']==entry['lead'] and q['window_s'] and entry['window_s']
                and legacy.overlap(entry['window_s'], *q['window_s'])]
    data['measurements'] = rows
    return [dict(role='system', content=prompt(state,'final')),
            dict(role='user', content=json.dumps(data, allow_nan=False))]


def event(args, aliases, visible, ledger, state, boundary_tools):
    args = dict(args)
    ignored_schedule = args.pop('next_review_s', None)
    citations = args.pop('citations', None)
    if not isinstance(citations, list) or len(citations) > 64:
        raise ValueError('Supply a list of at most 64 citations')
    grouped = {}
    measured = []
    for item in citations:
        if not isinstance(item, dict) or set(item) - {'evidence_id','role','fields','field'}:
            raise ValueError('Use evidence_id, role and optional fields')
        ref = item.get('evidence_id')
        key = aliases.get(ref, ref) if isinstance(ref, str) else None
        if key not in visible:
            raise ValueError('Select an ID in the displayed evidence catalog')
        role = item.get('role')
        if role not in DIRECTIONS + OTHER_ROLES:
            raise ValueError('Select a listed evidence role')
        row = ledger.get(key)
        values = numeric_fields(row)
        paths = item.get('fields', [item['field']] if 'field' in item else [])
        if not isinstance(paths, list) or not all(isinstance(p,str) and p in values for p in paths):
            raise ValueError(f'Use displayed fields from {ref} or cite its ID without fields')
        if role in DIRECTIONS:
            if row['tool'] != FOUNDER or row['status'] != 'ok' or not all(p in values for p in PAIR):
                raise ValueError('Only a usable Founder AF and AFL score pair can carry target direction')
            paths = list(dict.fromkeys([*PAIR, *paths]))
        if key in grouped and role != grouped[key]['role']:
            raise ValueError('Assign one combined-target role per measurement')
        entry = grouped.setdefault(key, dict(evidence_id=key, role=role, fields=[]))
        for path in paths:
            if path not in entry['fields']:
                entry['fields'].append(path)
                measured.append(dict(evidence_id=key, field=path, value=values[path]))
    if set(args) & {'evidence_ids', 'evidence_attributions'}:
        raise ValueError('Use citations for event evidence')
    for field in ('pending','uncertainty_cause','boundary'):
        args.setdefault(field, None)
    args.update(evidence_ids=list(grouped), evidence_attributions=[
        dict(evidence_id=key, role=g['role'] if g['role'] in DIRECTIONS else 'CONTEXT_ONLY')
        for key, g in grouped.items()], next_review_s=None if state['stream_end'] else
        state['scheduled_review_s'] or state['cursor_s']+10)
    payload = legacy.validate(args, aliases, visible, ledger, state, boundary_tools, measured_contract=True)
    payload.update(measured_facts=measured, evidence_attributions=list(grouped.values()), event_valid=True)
    if ignored_schedule is not None:
        payload['ignored_event_schedule_s'] = ignored_schedule
    return payload


def validate_boundary(boundary, action, directional):
    if isinstance(boundary, legacy.Point):
        raise BoundaryEvidenceNeeded('Beat times do not localize AF/AFL to a second. Use a measured bracket, unobserved onset, or defer.')
    a, b = boundary.range_s
    before, after = DIRECTIONS if action == 'CLOSE' else reversed(DIRECTIONS)
    spans = lambda role: [r['window_s'] for r in directional[role]]
    for x, y in spans(before):
        for z, w in spans(after):
            if a == x and x < y <= z < w and w == b and not any(
                legacy.overlap((x,y), *span) for span in spans(after)
            ) and not any(legacy.overlap((z,w), *span) for span in spans(before)):
                return
    raise BoundaryEvidenceNeeded('Use the outer endpoints of distinct ordered, nonconflicting before and after Founder measurements with opposite target roles. Retrieve or measure the missing anchor, report unobserved onset, or defer.')

