"""Episode and appointment ledger for model-owned OPEN/CLOSE in a causal replay."""
from copy import deepcopy


def merge_spans(spans):
    merged = []
    for a, b in sorted(spans):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def gaps(start, end, covered):
    cursor = start
    for a, b in merge_spans(covered):
        if a > cursor:
            yield [cursor, min(end, a)]
        cursor = max(cursor, b)
    if cursor < end:
        yield [cursor, end]


def representative(boundary):
    if boundary['kind'] == 'localized':
        return boundary['time_s']
    if boundary['kind'] == 'bracketed':
        a, b = boundary['range_s']
        return (a + b) / 2
    return boundary['first_supported_interval_s'][0]


def subtract_spans(spans, exclusions):
    cuts = merge_spans(exclusions)
    result = []
    i = 0
    for a, b in merge_spans(spans):
        while i < len(cuts) and cuts[i][1] <= a:
            i += 1
        while i < len(cuts) and cuts[i][0] < b:
            left, right = cuts[i]
            if a < left:
                result.append([a, left])
            a = max(a, right)
            if right >= b:
                break
            i += 1
        if a < b:
            result.append([a, b])
    return result


class AutonomousEvents:
    def __init__(self, mode, start):
        self.mode = mode
        self.start = self.cursor = start
        self.next_review = None
        self.pending = self.active = self.last_change = None
        self.events = []
        self.questions = []
        self.unregistered_support = []
        self.unknown = []
        self.conflicts = []

    @property
    def state(self):
        return 'OPEN' if self.active else 'IDLE'

    @property
    def intervals(self):
        return [span for event in self.events for span in event['reported_intervals']]

    @property
    def support(self):
        positive = merge_spans(self.unregistered_support + [span for event in self.events
                               for span in event['confirmed_support_intervals']])
        return subtract_spans(positive, self.unknown)

    def due(self, end):
        return self.next_review is None or end >= self.next_review

    def snapshot(self, start, end, stream_end):
        active = None
        if self.active:
            active = {k: self.active[k] for k in ('event_id', 'estimated_start_s', 'confirmed_at_s', 'onset_boundary')}
        earliest = self.start
        if self.events:
            previous = self.events[-1]
            earliest = previous['estimated_start_s'] if active else previous['estimated_end_s']
        return dict(window_s=[start, end], assessment_window_s=[start, end], cursor_s=end,
                    open_event=active, earliest_boundary_s=earliest,
                    last_executed_change=deepcopy(self.last_change), pending_question=deepcopy(self.pending),
                    requested_wake_s=self.next_review, actual_wake_s=end,
                    review_mode=self.mode, stream_end=stream_end,
                    scheduled_review_s=None if stream_end or self.mode != 'fixed' else end + 10,
                    latest_review_s=end + 120)

    def retrieve(self, start_s, end_s, *, available_through_s, limit, event_id=None):
        if self.cursor > available_through_s:
            raise ValueError('Requested snapshot predates ledger state')
        found = []
        for event in reversed(self.events):
            if event_id is not None and event['event_id'] != event_id:
                continue
            if event['estimated_start_s'] >= end_s or (event['estimated_end_s'] or self.cursor) <= start_s:
                continue
            found.append(deepcopy({k: v[-6:] if isinstance(v, list) else v for k, v in event.items()}))
            if len(found) == limit:
                break
        return found

    def _coverage(self, start, end, supports, opposes, action=None, boundary=None):
        def clip(spans, lo, hi):
            return merge_spans((max(lo, a), min(hi, b)) for a,b in spans)

        positive = clip(supports, start, end)
        negative = clip(opposes, start, end)
        conflicts = merge_spans((max(a,c), min(b,d)) for a,b in positive for c,d in negative)
        self.conflicts.extend(conflicts)
        positive = subtract_spans(positive, conflicts)
        negative = subtract_spans(negative, conflicts)
        if boundary:
            point = representative(boundary)
            lo, hi = boundary.get('range_s', [point, point])
            if action == 'OPEN':
                positive = clip(positive, max(start, hi), end)
                negative = clip(negative, start, min(end, lo))
            else:
                positive = clip(positive, start, min(end, lo))
                negative = clip(negative, max(start, hi), end)
        unknown = list(gaps(start, end, positive + negative))
        self.unknown.extend(unknown)
        if self.active:
            self.active['unknown_intervals'].extend(deepcopy(unknown))
            self.active['confirmed_support_intervals'].extend(positive)
            self.active['opposing_intervals'].extend(negative)
        else:
            self.unregistered_support.extend(positive)

    def skip(self, start, end, *, retry=False):
        self.cursor = end
        self._coverage(start, end, [], [])
        if retry:
            self.next_review = None  # first arriving batch retries; no model appointment

    def apply(self, start, end, payload):
        if end <= self.cursor:
            raise ValueError('Duplicate or out-of-order event submission')
        self.cursor = end
        action = payload['event_action']
        boundary = payload['boundary']
        update = None
        if action == 'OPEN':
            onset = representative(boundary)
            self.active = dict(event_id=len(self.events)+1, estimated_start_s=onset,
                               onset_boundary=deepcopy(boundary), confirmed_at_s=end,
                               first_seen_at_s=payload.get('first_support_available_at_s', end),
                               estimated_end_s=None, end_decided_at_s=None, right_censored=False,
                               confirmed_support_intervals=[], unknown_intervals=[], opposing_intervals=[],
                               start_evidence_ids=payload['evidence_ids'][:], reported_intervals=[], final_status='OPEN')
            self.events.append(self.active)
            update = dict(kind='confirm', interval=[onset,end])
        self._coverage(start, end, payload['support_intervals'], payload['opposing_intervals'], action, boundary)
        if action == 'CLOSE':
            offset = representative(boundary)
            event = self.active
            support_end = boundary['range_s'][0] if boundary['kind'] == 'bracketed' else offset
            event['confirmed_support_intervals'] = merge_spans(
                (a, min(b, support_end)) for a, b in event['confirmed_support_intervals'])
            span = [event['estimated_start_s'], offset]
            event.update(end_boundary=deepcopy(boundary), estimated_end_s=offset, end_decided_at_s=end,
                         termination_reason='llm_close', final_status='CLOSED', reported_intervals=[span],
                         end_evidence_ids=payload['evidence_ids'][:])
            update = dict(kind='llm_close', interval=span)
        if boundary and boundary['kind'] == 'bracketed':
            self.unknown.append(boundary['range_s'][:])
            self.active['unknown_intervals'].append(boundary['range_s'][:])
        if update:
            update.update(recorded_at_s=end, event_id=self.active['event_id'],
                          evidence_ids=payload['evidence_ids'][:], reason=payload['reason'])
            self.last_change = deepcopy(update)
        if action == 'CLOSE':
            self.active = None
        self.next_review = payload['next_review_s']
        question_change = None
        if payload['pending']:
            if self.pending and payload.get('pending_resolution') in ('resolved', 'withdrawn'):
                self.pending.update(status=payload['pending_resolution'], resolution=payload['reason'], reviewed_at_s=end)
                self.pending = None
            if self.pending is None:
                self.pending = dict(question_id=len(self.questions)+1, first_asked_at_s=end,
                                    original_question=payload['pending']['question'],
                                    event_id=self.active['event_id'] if self.active else None)
                self.questions.append(self.pending)
            if self.active:
                self.pending['event_id'] = self.active['event_id']
            self.pending.update(payload['pending'], status='pending', reviewed_at_s=end,
                                next_review_s=self.next_review, evidence_ids=payload['evidence_ids'][:])
            question_change = deepcopy(self.pending)
        elif self.pending:
            self.pending.update(status=payload['pending_resolution'], resolution=payload['reason'], reviewed_at_s=end)
            question_change, self.pending = deepcopy(self.pending), None
        return dict(recorded_at_s=end, state=self.state, event_update=update, question_update=question_change)

    def finish(self, cursor):
        if self.active:
            event = self.active
            event.update(final_status='RIGHT_CENSORED', right_censored=True, termination_reason='stream_end',
                         reported_intervals=[[event['estimated_start_s'], cursor]])
        if self.pending:
            self.pending['status'] = 'unresolved_at_stream_end'
        return deepcopy(self.events)
