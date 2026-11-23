"""Window-stage interventions. Event actions and the immutable measurement ledger stay separate."""
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
MODES = ('legacy', 'masked', 'explicit', 'explicit_masked')
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


def window_request(data, tool, instruction, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown window assessment mode {mode}')
    if mode in ('masked', 'explicit_masked'):
        # Positive allowlist excludes notes, pending questions, event IDs, prior votes and actions.
        data = {key: data[key] for key in ('window_s', 'cursor_s', 'primary_lead', 'measurements')}
    if mode in ('explicit', 'explicit_masked'):
        tool = deepcopy(tool)
        schema = tool['function']['parameters']
        schema['properties'] = {
            'evidence_ids': schema['properties']['evidence_ids'],
            'reason': schema['properties']['reason'],
            'measurement_direction': {'type': 'string', 'enum': list(DIRECTIONS.values())},
            'finding': {'type': 'string', 'enum': list(LABELS)},
        }
        schema['required'] = list(schema['properties'])
        instruction = INSTRUCTION
    return data, tool, instruction


def canonical_arguments(args, mode):
    if mode in ('legacy', 'masked'):
        return args
    label = LABELS.get(args['finding'])
    if label is None or DIRECTIONS[label] != args['measurement_direction']:
        raise ValueError('Measurement direction and finding disagree. Re-read the current measurements.')
    return dict(interpretation=label, evidence_ids=args['evidence_ids'], reason=args['reason'])


def current_founder(row, window):
    fields = row.get('fields', {})
    return (row.get('tool') == 'score_ecgfounder' and row.get('status') == 'ok' and
            row.get('direction_allowed') and window[0] < row['window_s'][1] <= window[1] and
            row['window_s'][0] < window[1] and
            all(type(fields.get(k)) in (int, float) and isfinite(fields[k]) for k in ('af_score','afl_score')))
