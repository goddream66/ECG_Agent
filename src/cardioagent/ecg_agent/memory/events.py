"""Causal event identity, evidence spans, boundaries and append-only updates."""
from copy import deepcopy
import math
from cardioagent.ecg_agent.policies.event_rules import EventConfig, EventRules

class EventTracker:
    def __init__(self, config=None):
        self.config = config or EventConfig()
        self.rules = EventRules(self.config)
        self.state = 'IDLE'
        self.events = []
        self.active = None
        self.support_count = self.negative_count = 0
        self.since_check = 0
        self.cursor = None
        self.unknown_start = self.negative_start = self.support_start = None
        self.intervals = []

    def _close(self, reason, cursor):
        event = self.active
        event['termination_reason'] = reason
        event['closed_at_s'] = cursor
        event['final_status'] = 'CLOSED' if event['confirmed_at_s'] is not None else 'REJECTED'
        event['right_censored'] = reason == 'stream_end' and event['confirmed_at_s'] is not None
        self.active = None
        self.state = 'IDLE'
        self.support_count = self.negative_count = self.since_check = 0
        self.unknown_start = self.negative_start = self.support_start = None

    def observe(self, start, end, decision, checked=False, evidence_ids=()):
        if decision not in ('support', 'negative', 'conflict', 'unavailable'):
            raise ValueError('Unknown evidence interpretation')
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
            raise ValueError('Invalid observation interval')
        if self.cursor is not None and start != self.cursor:
            raise ValueError('Observations must cover contiguous arrived windows')
        self.cursor = end
        previous = self.state
        update = None
        closed_id = None
        if self.active is None and decision == 'support':
            self.active = {'event_id': len(self.events) + 1, 'candidate_start_s': start,
                           'first_seen_at_s': end, 'confirmed_at_s': None, 'estimated_end_s': None,
                           'end_decided_at_s': None, 'confirmed_support_intervals': [],
                           'counter_evidence_intervals': [], 'unknown_intervals': [],
                           'reported_intervals': [], 'termination_reason': None, 'right_censored': False}
            self.events.append(self.active)
            self.state = 'CANDIDATE'
        event = self.active
        if event is not None:
            if decision == 'support':
                if not self.support_count:
                    self.support_start = start
                self.support_count += 1
                self.negative_count = 0
                self.negative_start = self.unknown_start = None
                self.since_check = 0 if checked else self.since_check + 1
                event['confirmed_support_intervals'].append([start, end])
                event['last_supported_at_s'] = end
                if event['confirmed_at_s'] is None and self.rules.confirms(self.support_count):
                    event['confirmed_at_s'] = end
                    # The confirmed run starts after any interruption of a candidate.
                    event['estimated_start_s'] = self.support_start
                    interval = [self.support_start, end]
                    self.intervals.append(interval)
                    event['reported_intervals'].append(interval)
                    update = {'kind': 'confirm', 'interval': interval.copy(), 'recorded_at_s': end}
                elif event['confirmed_at_s'] is not None:
                    interval = event['reported_intervals'][-1]
                    interval[1] = end
                    update = {'kind': 'extend', 'interval': interval.copy(), 'recorded_at_s': end}
                self.state = 'ACTIVE' if event['confirmed_at_s'] is not None else 'CANDIDATE'
            elif decision == 'negative':
                self.support_count = 0
                self.unknown_start = None
                event['counter_evidence_intervals'].append([start, end])
                if event['confirmed_at_s'] is None:
                    closed_id = event['event_id']
                    self._close('candidate_rejected', end)
                else:
                    if not self.negative_count:
                        self.negative_start = start
                    self.negative_count += 1
                    self.state = 'PENDING_END'
                    if self.rules.ends(self.negative_count):
                        event['estimated_end_s'] = self.negative_start
                        event['end_decided_at_s'] = end
                        # Last support is never extended into unobserved/unknown end evidence.
                        closed_id = event['event_id']
                        self._close('confirmed_end', end)
            else:
                self.support_count = self.negative_count = 0
                self.negative_start = None
                if self.unknown_start is None:
                    self.unknown_start = start
                event['unknown_intervals'].append([start, end])
                self.state = 'UNKNOWN_HOLD'
                if self.rules.unknown_expired(self.unknown_start, end):
                    closed_id = event['event_id']
                    self._close('unknown_timeout' if event['confirmed_at_s'] is not None else 'candidate_timeout', end)
        return {'window_s': [start, end], 'recorded_at_s': end, 'decision': decision,
                'previous_state': previous, 'state': self.state,
                'event_id': self.active['event_id'] if self.active else closed_id,
                'evidence_ids': list(evidence_ids), 'event_update': update,
                'stop_review': True, 'legal_actions': ['WAIT_NEXT']}

    def finish(self, cursor):
        if self.cursor is not None and cursor < self.cursor:
            raise ValueError('Stream end precedes observation')
        if self.active is not None:
            self._close('stream_end', cursor)
        return deepcopy(self.events)
