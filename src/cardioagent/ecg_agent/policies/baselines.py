"""Frozen baseline decisions and their original memory semantics."""
from .labels import score_label
POLICIES = ("two-window", "dual", "adaptive")
BORDERLINE_MARGIN = .10
RECHECK_WINDOWS = 3

class EventMemory:
    def __init__(self):
        self.active = None
        self.events = []
        self.intervals = []
        self.support_start = None
        self.support_count = 0
        self.open_interval = False
        self.last_checked_count = 0

    def close(self, status, cursor):
        if self.active is not None:
            self.active.update(final_status=status, closed_at_s=cursor)
        self.active = None
        self.reset_support()

    def reset_support(self):
        self.support_start = None
        self.support_count = 0
        self.open_interval = False
        self.last_checked_count = 0

    def observe(self, start, cursor, decision, checked, evidence_ids):
        if self.active is None:
            if decision in ('negative', 'unavailable'):
                return None
            self.active = {'event_id': len(self.events) + 1, 'candidate_start_s': start,
                           'first_seen_at_s': cursor, 'confirmations': [], 'observations': []}
            self.events.append(self.active)
        observation = {'event_id': self.active['event_id'], 'window_s': [start, cursor], 'recorded_at_s': cursor,
                       'decision': decision, 'evidence_ids': evidence_ids, 'event_update': None}
        self.active['observations'].append(observation)
        if decision == 'negative':
            confirmed = bool(self.active['confirmations'])
            self.close('ended' if confirmed else 'dismissed', cursor)
            observation['state'] = 'ended' if confirmed else 'dismissed'
            return observation
        if decision != 'support':
            self.reset_support()
            observation['state'] = 'needs_review'
        else:
            if self.support_count == 0:
                self.support_start = start
            self.support_count += 1
            self.active['last_supported_at_s'] = cursor
            if checked:
                self.last_checked_count = self.support_count
            if self.open_interval:
                self.intervals[-1][1] = cursor
                update = 'extend'
            elif self.support_count >= 2 and checked:
                self.intervals.append([self.support_start, cursor])
                self.open_interval = True
                update = 'confirm'
            else:
                observation['state'] = 'pending'
                return observation
            observation['state'] = 'confirmed'
            observation['event_update'] = {'kind': update, 'interval': self.intervals[-1].copy(),
                                           'recorded_at_s': cursor}
            if update == 'confirm':
                self.active['confirmations'].append(observation['event_update'])
        return observation


def decide(call, primary, secondary, threshold, memory, policy, budget):
    primary_label = score_label(primary, threshold)
    if policy == 'two-window':
        return {True: 'support', False: 'negative', None: 'unavailable'}[primary_label], True, 'primary_score'
    if policy == 'dual':
        alternate = score_label(call('score_ecgfounder', secondary), threshold)
        if primary_label is None or alternate is None:
            return 'unavailable', False, 'missing_channel'
        if primary_label != alternate:
            return 'conflict', False, 'channel_disagreement'
        return ('support' if primary_label else 'negative'), True, 'channel_agreement'

    if primary_label is None:
        return 'unavailable', False, 'primary_unavailable'
    if primary_label is False:
        if memory.active is not None:
            quality = call('assess_quality', primary['lead'])
            if quality['status'] != 'ok' or quality['result']['reasons']:
                return 'unavailable', False, 'primary_quality_insufficient'
        return 'negative', False, 'primary_below_threshold'

    spent = 0

    def review(tool, lead):
        nonlocal spent
        if spent >= budget:
            return None
        spent += 1
        return call(tool, lead)

    primary_quality = review('assess_quality', primary['lead'])
    if primary_quality['status'] != 'ok' or primary_quality['result']['reasons']:
        return 'conflict', False, 'primary_quality_insufficient'
    score = max(primary['result']['af_score'], primary['result']['afl_score'])
    borderline = score < threshold + BORDERLINE_MARGIN
    recheck_due = memory.open_interval and memory.support_count + 1 - memory.last_checked_count >= RECHECK_WINDOWS
    needs_check = borderline or recheck_due or (memory.active is not None and not memory.open_interval)
    if not needs_check:
        return 'support', False, 'track_or_wait_for_next_window'
    other = review('score_ecgfounder', secondary)
    if other is None:
        return 'conflict', False, 'review_budget_exhausted'
    alternate = score_label(other, threshold)
    if alternate is not None:
        quality = review('assess_quality', secondary)
        if quality is None:
            return 'conflict', False, 'review_budget_exhausted'
        if quality['status'] != 'ok' or quality['result']['reasons']:
            return 'conflict', False, 'secondary_quality_insufficient'
        if alternate:
            return 'support', True, 'usable_channels_agree'
    rr = review('get_rate_rhythm', primary['lead'])
    reason = 'channel_disagreement' if alternate is False else 'secondary_unavailable'
    if rr is None:
        return 'conflict', False, f'{reason}_budget_exhausted'
    if rr['status'] != 'ok':
        return 'conflict', False, f'{reason}_rr_insufficient'
    return 'conflict', False, reason
