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
        updated.update(lifecycle=intent,refinement=revision,candidate=deepcopy(self.candidate))
        return updated

