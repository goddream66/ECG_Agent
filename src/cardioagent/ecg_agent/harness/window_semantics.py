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
UNION_MODES = ('union_raw', 'union_max', 'union_reference')
UNION_LABELS = {'TARGET_PRESENT': 'support', 'TARGET_ABSENT': 'negative',
                'CANNOT_TELL': 'unavailable', 'CONFLICTING_MEASUREMENTS': 'conflict'}
UNION_TASK = (
    'Answer ONE binary target question about the current interval: which is better supported, '
    'TARGET PRESENT (atrial fibrillation OR atrial flutter), or TARGET ABSENT (neither)? '
    'AF and AFL are interchangeable members of one target class. Never choose between their names. '
    'A change from one target subtype to the other stays target-present. '
    'Choose TARGET_PRESENT if current reliable measurements favor the target class; choose TARGET_ABSENT '
    'if they favor neither target subtype. TARGET_ABSENT does not require proving normal sinus rhythm or '
    'naming an alternative diagnosis. You are reporting the direction of the evidence, not diagnostic certainty. '
    'When both absolute scores are weak, a small difference between them gives no evidence of target presence. '
    'The union score is the larger absolute score; it is uncalibrated. There is no mandatory cutoff. '
    'Use CANNOT_TELL when missing, ambiguous or unreliable evidence prevents even a directional preference. '
    'Use CONFLICTING_MEASUREMENTS for opposing current rhythm measurements. Beat and RR proxies provide '
    'quality or rhythm context only. Use the current interval for your vote; historical measurements '
    'can show a change. Cite the IDs and explain your target versus non-target choice in one sentence.'
)
MODES = ('legacy', 'masked', 'explicit', 'explicit_masked', 'single', 'verified') + UNION_MODES
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
    if mode in ('masked', 'explicit_masked', 'single', 'verified') + UNION_MODES:
        # Positive allowlist excludes notes, pending questions, event IDs, prior votes and actions.
        data = {key: data[key] for key in ('window_s', 'cursor_s', 'primary_lead', 'measurements')}
    if mode in ('explicit', 'explicit_masked', 'single', 'verified') + UNION_MODES:
        tool = deepcopy(tool)
        schema = tool['function']['parameters']
        schema['properties'] = {
            'evidence_ids': schema['properties']['evidence_ids'],
            'reason': schema['properties']['reason'],
            'measurement_direction': {'type': 'string', 'enum': list(DIRECTIONS.values())},
            'finding': {'type': 'string', 'enum': list(LABELS)},
        }
        if mode in ('single', 'verified') + UNION_MODES:
            del schema['properties']['measurement_direction']
        schema['required'] = list(schema['properties'])
        instruction = INSTRUCTION
    if mode in UNION_MODES:
        data = deepcopy(data)
        rhythm, context = [], []
        for row in data.pop('measurements'):
            if row['tool'] != 'score_ecgfounder':
                context.append(row)
                continue
            fields = row['fields']
            scores = [fields.get(k) for k in ('af_score', 'afl_score')]
            valid = row['status']=='ok' and all(type(v) in (float,int) and isfinite(v) for v in scores)
            row['fields'] = {'target_score': max(scores) if valid else None}
            if mode == 'union_raw':
                row['fields'].update(af_score=scores[0], afl_score=scores[1])
            rhythm.append(row)
        timeline = sorted(rhythm, key=lambda r:(r['window_s'],r['lead'],r['id']))
        data['rhythm_measurements_by_time'] = timeline
        if mode == 'union_reference':
            if not profile:
                raise ValueError('Union reference needs an independently fitted profile')
            data.pop('rhythm_measurements_by_time')
            current, previous = [], []
            for entry in timeline:
                score = entry['fields']['target_score']
                if score is not None:
                    entry['empirical_reference'] = profile['bins'][entry['lead']][bisect_right(profile['edges'],score)]
                (current if entry['window_s'][1]>data['window_s'][0] else previous).append(entry)
            data['vote_on_this_interval'] = current
            data['earlier_measurements_for_comparison_only'] = previous
        data['other_measurements'] = context
        tool['function']['parameters']['properties']['finding']['enum'] = list(UNION_LABELS)
        instruction = UNION_TASK
        if mode == 'union_reference':
            instruction += (' Each reference gives the fraction of target-positive windows among similar scores '
                            'in other records, and the direction of that measurement. Negative weighs toward '
                            'TARGET_ABSENT; support weighs toward TARGET_PRESENT. Neither label claims certainty. '
                            'A nonzero score alone gives no positive evidence. Old positive readings do not '
                            'make a current negative reading positive. If you override or abstain, cite a '
                            'specific current conflict, reliability problem or ambiguous measurement.')
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
    if mode in UNION_MODES:
        return dict(interpretation=UNION_LABELS[args['finding']], evidence_ids=args['evidence_ids'], reason=args['reason'])
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


def available_findings(rows, window, profile):
    current=[r for r in rows.values() if current_founder(r,window)]
    flags={band(r,profile)['direction'] for r in current}
    allowed=['CANNOT_TELL']
    if {'support','negative'}<=flags:
        allowed.append('CONFLICTING_MEASUREMENTS')
    elif 'support' in flags:
        allowed.append('TARGET_PRESENT')
    elif 'negative' in flags:
        allowed.append('TARGET_ABSENT')
    return allowed
