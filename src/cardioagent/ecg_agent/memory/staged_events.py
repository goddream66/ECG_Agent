"""Apply model-owned episode intents independently of boundary refinement."""
from copy import deepcopy
from .autonomous_events import AutonomousEvents, representative


class StagedEvents(AutonomousEvents):
    def __init__(self, mode, start):
        super().__init__(mode, start)
        self.candidate = None
        self.candidates = []
        self.next_id = 1

    def snapshot(self, start, end, stream_end):
        state = super().snapshot(start, end, stream_end)
        state.update(candidate=deepcopy(self.candidate), next_event_id=self.next_id,
                     last_closed_end_s=self.events[-1]['estimated_end_s'] if self.events and not self.active
                     else self.events[-2]['estimated_end_s'] if len(self.events)>1 else self.start)
        if self.active and 'end_review_memory' in self.active:
            state['end_review_memory'] = deepcopy(self.active['end_review_memory'])
        return state

    def apply(self, start, end, payload):
        if end <= self.cursor:
            raise ValueError('Duplicate or out-of-order staged observation')
        intent = payload['lifecycle']
        action = intent.get('action') if intent['status']=='accepted' else None
        revision = payload.get('refinement') or {}
        bracket = revision.get('boundary') if revision.get('status')=='located' else None
        journal = dict(payload, event_action='RECORD_ONLY', boundary=None)
        if action=='CREATE_CANDIDATE':
            self.candidate = dict(event_id=self.next_id, detected_at_s=end, first_support=payload['support_intervals'][0],
                                  evidence_ids=payload['evidence_ids'][:], status='candidate')
            self.next_id += 1
            self.candidates.append(self.candidate)
            self.candidate['onset_refinement']=deepcopy(revision)
        elif action=='DISMISS_CANDIDATE':
            self.candidate.update(status='dismissed', decided_at_s=end)
            self.candidate = None
        if action=='CONFIRM_EVENT':
            candidate=self.candidate
            first=candidate['first_support'] if candidate else payload['support_intervals'][0]
            ident=candidate['event_id'] if candidate else self.next_id
            detected=candidate['detected_at_s'] if candidate else end
            if candidate:
                bracket = bracket or candidate['onset_refinement'].get('boundary')
                candidate.update(status='confirmed', decided_at_s=end)
                self.candidate=None
            else:
                self.next_id+=1
            journal.update(event_action='OPEN',boundary=bracket or
                           dict(kind='onset_unobserved',first_supported_interval_s=first),
                           first_support_available_at_s=detected)
        elif action=='REQUEST_CLOSE' and bracket:
            journal.update(event_action='CLOSE',boundary=bracket)
        updated=super().apply(start,end,journal)
        if journal['event_action']=='CLOSE' and revision.get('confirmation_evidence_id'):
            # The current window's citations and the three temporal anchors have different roles.
            anchors = dict(last_support=revision['before_id'], first_against=revision['after_id'],
                           confirming_against=revision['confirmation_evidence_id'])
            event = self.events[-1]
            event['end_evidence_by_role'] = anchors
            pair = revision.get('support_pair_ids', [])
            event['end_evidence_ids'] = list(dict.fromkeys([*anchors.values(), *pair, *event['end_evidence_ids']]))
            updated['event_update']['end_evidence_by_role'] = anchors.copy()
            if pair:
                event['last_support_pair_ids'] = pair[:]
                updated['event_update']['last_support_pair_ids'] = pair[:]
        if action=='CONFIRM_EVENT':
            self.active.update(event_id=ident, detected_at_s=detected,
                               onset_status='bracketed' if bracket else 'unlocalized')
            self.last_change['event_id']=updated['event_update']['event_id']=ident
        elif action=='REFINE_ONSET' and self.candidate and bracket:
            self.candidate['onset_refinement']=deepcopy(revision)
        elif action=='REFINE_ONSET' and bracket:
            self.active.update(onset_status='bracketed',onset_boundary=deepcopy(bracket),
                               estimated_start_s=representative(bracket))
            self.unknown.append(bracket['range_s'][:])
            self.active['unknown_intervals'].append(bracket['range_s'][:])
            updated['event_update']=dict(kind='onset_revision',event_id=self.active['event_id'],
                                         recorded_at_s=end,boundary=deepcopy(bracket))
            self.last_change=deepcopy(updated['event_update'])
        if self.active and 'end_review_memory' in payload:
            self.active['end_review_memory'] = deepcopy(payload['end_review_memory'])
            if action == 'CONFIRM_EVENT' and payload.get('supported_tail_export'):
                last = max(min(end, right) for _, right in payload['support_intervals'])
                self.active['published_through_s'] = last
                self.active['supported_tail_export'] = True
                updated['event_update']['interval'][1] = last
                self.last_change = deepcopy(updated['event_update'])
            if action == 'CONTINUE_EVENT' and payload.get('interpretation') == 'support':
                # Publish the event's provisional span. Coverage still records gaps separately.
                coverage = self.active['confirmed_support_intervals']
                supported_end = coverage[-1][1] if coverage else start
                published = self.active.get('published_through_s', self.active['confirmed_at_s'])
                if start < supported_end <= end and supported_end > published:
                    span = [self.active['estimated_start_s'], supported_end]
                    self.active['published_through_s'] = supported_end
                    updated['event_update'] = dict(kind='extend', event_id=self.active['event_id'],
                        interval=span, recorded_at_s=end, provisional=True,
                        evidence_ids=payload['evidence_ids'][:], reason=payload['reason'])
        updated.update(lifecycle=intent,refinement=revision,candidate=deepcopy(self.candidate))
        return updated

    def finish(self, cursor):
        if self.active and self.active.get('supported_tail_export'):
            # The replay ended with this episode open. Export only the span supported so far.
            cursor = min(cursor, self.active['published_through_s'])
        return super().finish(cursor)

