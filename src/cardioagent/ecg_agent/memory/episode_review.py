"""Build bounded, causal evidence packets from the window and event ledgers."""

from collections import Counter, deque
from itertools import groupby, zip_longest


def window_memory(trace, evidence):
    expected = set(trace['evidence_ids'])
    span = [trace['start_s'], trace['end_s']]
    cursor = trace['cursor_s']
    if expected != {item['evidence_id'] for item in evidence} or span[1] != cursor:
        raise ValueError('Window trace and evidence IDs disagree')
    leads = {lead: {'score': None, 'score_status': 'not_measured', 'quality': 'not_measured'}
             for lead in ('ECG1', 'ECG2')}
    for item in evidence:
        if (item['window_s'] != span or item['available_at_s'] > cursor
                or item['task_id'] != trace['task_id']):
            raise ValueError('Evidence does not belong to the arrived task window')
        lead = leads[item['lead']]
        if item['tool'] == 'score_ecgfounder':
            lead['score_status'] = item['status']
            lead['score_id'] = item['evidence_id']
            if item['status'] == 'ok':
                lead['score'] = [item['result']['af_score'], item['result']['afl_score']]
        elif item['tool'] == 'assess_quality':
            reasons = item['result'].get('reasons')
            lead['quality_id'] = item['evidence_id']
            lead['quality_reasons'] = reasons
            if item['status'] != 'ok' or reasons is None:
                lead['quality'] = 'measurement_failed'
            else:
                lead['quality'] = 'warning' if reasons else 'no_warning'
    return {'window_s': span, 'decision': trace['decision'], 'reason': trace['reason'],
            'threshold': trace['decision_threshold'], 'leads': leads}


class EpisodeMemory:
    def __init__(self, event_id):
        self.event_id = event_id
        self.interval = None
        self.confirmed_at = None
        self.first_seen_at = None
        self.decisions = Counter()
        self.quality = {lead: Counter() for lead in ('ECG1', 'ECG2')}

    def observe(self, window, update):
        if self.first_seen_at is None:
            self.first_seen_at = window['window_s'][1]
        self.decisions[window['decision']] += 1
        for lead, counters in self.quality.items():
            counters[window['leads'][lead]['quality']] += 1
        if update:
            self.interval = update['interval']
            if update['kind'] == 'confirm':
                self.confirmed_at = update['recorded_at_s']

    def summary(self):
        return {'event_id': self.event_id, 'interval': self.interval,
                'first_seen_at_s': self.first_seen_at, 'confirmed_at_s': self.confirmed_at,
                'observed_window_counts': dict(self.decisions),
                'quality_window_counts': {lead: dict(counts) for lead, counts in self.quality.items()}}


def _packet(left, right, left_closed, recent, cursor, threshold, gap_limit):
    a, b = left['interval'][1], right['interval'][0]
    if not 0 < b - a <= gap_limit:
        return None
    windows = [win for win in recent if a - 20 <= win['window_s'][0] < b + 20]
    gap = [win for win in windows if a <= win['window_s'][0] < b]
    if (not gap or gap[0]['window_s'][0] != a or gap[-1]['window_s'][1] != b
            or any(x['window_s'][1] != y['window_s'][0] for x, y in zip(gap, gap[1:]))):
        raise ValueError('Gap windows must cover the whole candidate gap')
    if left_closed['closed_at_s'] > cursor:
        raise ValueError('Left event has not closed at review time')
    index = {}
    compact = []
    support = []
    for win in windows:
        entry = {**win, 'leads': {}}
        choices = []
        for lead, data in win['leads'].items():
            item = dict(data)
            for key in ('score_id', 'quality_id'):
                if key in item:
                    alias = f'e{len(index) + 1}'
                    index[alias] = item[key]
                    item[key] = alias
            entry['leads'][lead] = item
            if (item['score'] is not None and max(item['score']) >= threshold
                    and item['quality'] == 'no_warning'):
                choices.append([item['score_id'], item['quality_id']])
        if a <= win['window_s'][0] < b:
            support.append(choices)
        compact.append(entry)
    return {'cursor_s': cursor, 'gap_s': [a, b], 'start_threshold': threshold,
            'left': dict(left, closed_at_s=left_closed['closed_at_s'],
                         termination_reason=left_closed['termination_reason']),
            'right': right, 'windows': compact, 'evidence_index': index, 'gap_support': support}


def review_packets(traces, updates, evidence, events, *, start_threshold, gap_limit=60):
    """Consume ledger rows in arrival order. The final event index supplies closure metadata only."""
    final = {event['event_id']: event for event in events}
    recent = deque()
    current = previous_confirmed = None
    evidence_groups = iter(groupby(evidence, key=lambda row: row['available_at_s']))
    evidence_at, records = next(evidence_groups, (None, ()))
    for trace, state in zip_longest(traces, updates):
        if trace is None or state is None or state['recorded_at_s'] != trace['cursor_s']:
            raise ValueError('Window and event update streams disagree')
        cursor = trace['cursor_s']
        if evidence_at is not None and evidence_at < cursor:
            raise ValueError('Evidence out of order')
        if evidence_at == cursor:
            arrived = list(records)
            evidence_at, records = next(evidence_groups, (None, ()))
        else:
            arrived = []
        win = window_memory(trace, arrived)
        recent.append(win)
        while recent[0]['window_s'][1] <= cursor - gap_limit - 40:
            recent.popleft()
        event_id = state['event_id']
        if event_id is None:
            continue
        if current is None or current.event_id != event_id:
            current = EpisodeMemory(event_id)
        update = state['event_update']
        current.observe(win, update)
        if update and update['kind'] == 'confirm':
            right = current.summary()
            if previous_confirmed is not None:
                left = previous_confirmed.summary()
                packet = _packet(left, right, final[left['event_id']], recent, cursor,
                                 start_threshold, gap_limit)
                if packet is not None:
                    yield packet
            previous_confirmed = current
    if evidence_at is not None:
        raise ValueError('Unreferenced evidence after last arrival')
