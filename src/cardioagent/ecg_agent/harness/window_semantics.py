"""Window-stage interventions. Event actions and the immutable measurement ledger stay separate."""
from bisect import bisect_right
from copy import deepcopy
from math import isfinite

LABELS = {
    'EVIDENCE_FAVORS_AF_OR_AFL': 'support',
    'EVIDENCE_WEIGHS_AGAINST_BOTH': 'negative',
    'CURRENT_MEASUREMENTS_DISAGREE': 'conflict',
    'INSUFFICIENT_CURRENT_EVIDENCE': 'unavailable',
}
DIRECTIONS = {
    'support': 'TOWARD_TARGET',
    'negative': 'AWAY_FROM_TARGET',
    'conflict': 'MIXED',
    'unavailable': 'UNDETERMINED',
}
MODES = ('legacy', 'masked', 'explicit', 'explicit_masked', 'single', 'verified')
INSTRUCTION = (
    'What do the measurements for THIS interval say about the presence of AF or AFL? '
    'Read both absolute Founder scores. One being larger than the other does not establish target presence. '
    'High support for either subtype can favor the target; two weak scores can weigh against it. '
    'Low scores alone do not prove a normal rhythm. Use insufficient when evidence cannot distinguish. '
    'A regular RR is compatible with AFL and with non-target rhythms. Compatibility alone gives no positive evidence. '
    'Cite the measurements, explain their direction, then choose the finding with that same meaning. '
    'The word support always refers to target presence. Evidence favoring absence weighs against both. '
    'Previous decisions are not new measurements. Do not decide whether to open or close an event here.'
)


def window_request(data, tool, instruction, mode, profile=None):
    if mode not in MODES:
        raise ValueError(f'Unknown window assessment mode {mode}')
    if mode in ('masked', 'explicit_masked', 'single', 'verified'):
        # Positive allowlist excludes notes, pending questions, event IDs, prior votes and actions.
        data = {key: data[key] for key in ('window_s', 'cursor_s', 'primary_lead', 'measurements')}
    if mode in ('explicit', 'explicit_masked', 'single', 'verified'):
        tool = deepcopy(tool)
        schema = tool['function']['parameters']
        schema['properties'] = {
            'evidence_ids': schema['properties']['evidence_ids'],
            'reason': schema['properties']['reason'],
            'measurement_direction': {'type': 'string', 'enum': list(DIRECTIONS.values())},
            'finding': {'type': 'string', 'enum': list(LABELS)},
        }
        if mode in ('single', 'verified'):
            del schema['properties']['measurement_direction']
        schema['required'] = list(schema['properties'])
        instruction = INSTRUCTION
    if mode == 'verified':
        if not profile:
            raise ValueError('Verified mode needs a frozen AFDB empirical reference')
        data = deepcopy(data)
        for row in data['measurements']:
            if current_founder(row, data['window_s']):
                row['reference_band'] = band(row, profile)
        instruction += (' Reference bands summarize other AFDB records. For a directional finding, cite a '
                        'current Founder measurement. Each current Founder citation must match the chosen direction. Weak or conflicting evidence '
                        'can remain insufficient. Use the waveform time, not the lifecycle state.')
    return data, tool, instruction


def canonical_arguments(args, mode):
    if mode in ('legacy', 'masked'):
        return args
    label = LABELS.get(args['finding'])
    if label is None or (mode not in ('single','verified') and DIRECTIONS[label] != args['measurement_direction']):
        raise ValueError('Measurement direction and finding disagree. Re-read the current measurements.')
    return dict(interpretation=label, evidence_ids=args['evidence_ids'], reason=args['reason'])


def current_founder(row, window):
    fields = row.get('fields', {})
    return (row.get('tool') == 'score_ecgfounder' and row.get('status') == 'ok' and
            row.get('direction_allowed') and window[0] < row['window_s'][1] <= window[1] and
            row['window_s'][0] < window[1] and
            all(type(fields.get(k)) in (int, float) and isfinite(fields[k]) for k in ('af_score','afl_score')))


def band(row, profile):
    pair = row['fields']
    index = bisect_right(profile['edges'], max(pair['af_score'], pair['afl_score']))
    return profile['bins'][row['lead']][index]


def verify_band(observation, rows, window, profile):
    direction = observation['interpretation']
    if direction not in ('support', 'negative'):
        return
    flags = [band(rows[k], profile)['direction'] for k in observation['evidence_ids']
             if k in rows and current_founder(rows[k], window)]
    if not flags or any(flag != direction for flag in flags):
        raise ValueError('Each current Founder measurement cited as directional must match the finding. Use uncertainty if needed.')
