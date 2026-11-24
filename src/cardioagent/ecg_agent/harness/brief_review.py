"""Review an elapsed, isolated positive window without backdating its publication."""

from collections import Counter, deque
from copy import deepcopy
from math import isfinite

from .staged_review import TEXT, ask, choice, schema


LEADS = ('ECG1', 'ECG2')
SHIFT_OPTIONS = {f'{side}_{lead}': (lead, offset) for side, offset in (('left', -5), ('right', 5))
                 for lead in LEADS}
DECISIONS = ('REPORT_PAST_EVENT', 'DISMISS', 'UNCERTAIN')
QUESTION = (
    'An isolated ten-second window had higher AF-or-AFL model scores than the preceding and following '
    'windows. Decide whether to publish that earlier window as a distinct brief episode. The current '
    'window has already arrived. A score excursion can reflect a brief rhythm, artifact, or fluctuation '
    'inside a longer rhythm. Two lead measurements and overlapping windows are correlated. An elevated '
    'score does not establish the precise onset or duration. REPORT_PAST_EVENT publishes the estimated '
    'candidate interval now. DISMISS means the evidence favors rejecting a separate brief episode. '
    'Use UNCERTAIN when the available measurements cannot distinguish the alternatives. Cite observed '
    'measurement IDs and give a short reason. The time of this decision stays at the current cursor.'
)


class BriefCandidateHistory:
    """Hold four arrived paired measurements. A missing or discontinuous window breaks the pattern."""

    def __init__(self):
        self.windows = deque(maxlen=4)
        self.cursor = None

    def observe(self, start, end, scores):
        if not all(isfinite(x) for x in (start, end)) or not start < end:
            raise ValueError('Invalid window coordinates')
        if self.cursor is not None:
            if start < self.cursor:
                raise ValueError('Duplicate or out-of-order window')
            if start != self.cursor:
                self.windows.clear()
        self.cursor = end
        values = [scores.get(lead) for lead in LEADS]
        if end-start != 10 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not isfinite(v) or not 0 <= v <= 1 for v in values):
            self.windows.clear()
            return None
        direction = 'high' if min(values) >= .5 else 'low' if max(values) < .5 else 'mixed'
        self.windows.append((start, end, values, direction))
        if [w[3] for w in self.windows] != ['low', 'low', 'high', 'low']:
            return None
        evidence = []
        for a, b, pair, _ in self.windows:
            for lead, score in zip(LEADS, pair, strict=True):
                evidence.append(dict(id=f'E{len(evidence)+1}', lead=lead, window_s=[a,b],
                                     available_at_s=b, target_score=score))
        return dict(window_s=[start,end], candidate_s=list(self.windows[2][:2]), measurements=evidence)


def review(packet, mode, model, measure, emit):
    """Publish, dismiss or defer one historical finding. All optional reads happen via the supplied tool."""
    if mode not in ('aligned', 'all', 'active'):
        raise ValueError('Unknown brief-event review mode')
    if set(packet) != {'window_s', 'candidate_s', 'measurements'}:
        raise ValueError('Unexpected brief-event input fields')
    data = deepcopy(packet)
    counts = Counter()
    cursor = data['window_s'][1]
    if any(set(row) != {'id','lead','window_s','available_at_s','target_score'}
           for row in data['measurements']):
        raise ValueError('Unexpected measurement fields')
    if any(not r['window_s'][0] < r['window_s'][1] <= r['available_at_s'] <= cursor
           for r in data['measurements']):
        raise ValueError('Evidence must have arrived by the decision cursor')
    requested = []
    available = dict(SHIFT_OPTIONS)
    audit = dict(cursor_s=cursor, candidate_s=data['candidate_s'])

    def acquire(option):
        lead, offset = available.pop(option)
        a, b = data['candidate_s']
        if b+offset > cursor:
            raise ValueError('Shifted measurement has not arrived')
        result = measure(lead, a, b, cursor, offset)
        if (result['lead'] != lead or result['window_s'] != [a+offset, b+offset]
                or result['available_at_s'] != cursor):
            raise ValueError('Tool result does not belong to the requested lead, interval and cursor')
        requested.append(option)
        score = None
        if result['status'] == 'ok':
            fields = result['result']
            values = (fields['af_score'], fields['afl_score'])
            if any(not isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError('Invalid shifted score')
            score = max(values)
        data['measurements'].append(dict(id=f'S{len(requested)}', lead=lead, window_s=result['window_s'],
                                         available_at_s=cursor, status=result['status'], target_score=score))
        emit('measurements', dict(**audit, option=option, displayed_id=f'S{len(requested)}', receipt=result))

    if mode == 'all':
        for option in list(available):
            acquire(option)
    elif mode == 'active':
        for step in range(2):
            options = ['finish', *available]
            tool = schema('select_brief_measurement', 'Select an arrived ten-second measurement or finish.',
                          dict(option=choice(options), reason=TEXT), ['option','reason'])

            def select(args):
                if args['option'] not in options:
                    raise ValueError('Select an available measurement or finish')
                return args['option']

            selection = ask(model, counts, emit, audit, dict(data, remaining_measurements=2-step), tool,
                            'Review the earlier brief candidate. Left and right shift the candidate by -5 '
                            'and +5 seconds respectively. Pick a lead and shift only when its measurement '
                            'could help distinguish an episode from a score excursion. Choose finish if '
                            'current evidence is sufficient or extra overlapping samples cannot resolve it.',
                            select)
            if not isinstance(selection, str) or selection == 'finish':
                break
            acquire(selection)
    ids = {row['id'] for row in data['measurements'] if row['target_score'] is not None}
    final = schema('review_brief_candidate', 'Judge the distinct, already elapsed candidate interval.',
                   dict(decision=choice(DECISIONS), evidence_ids=dict(type='array', items=TEXT), reason=TEXT),
                   ['decision', 'evidence_ids', 'reason'])

    def validate(args):
        citations = args['evidence_ids']
        if (args['decision'] not in DECISIONS or not isinstance(citations, list) or not citations
                or any(not isinstance(key, str) or key not in ids for key in citations)):
            raise ValueError('Select a listed decision and cite available, measured IDs')
        return dict(status='accepted', **args)

    answer = ask(model, counts, emit, audit, data, final, QUESTION, validate)
    write = None
    if answer.get('decision') == 'REPORT_PAST_EVENT':
        write = dict(interval=data['candidate_s'], t_recorded=cursor, kind='retrospective_brief',
                     evidence_ids=list(answer['evidence_ids']))
    return dict(candidate_s=data['candidate_s'], decided_at_s=cursor, decision=answer,
                requests=requested, write=write, counts=dict(counts))
