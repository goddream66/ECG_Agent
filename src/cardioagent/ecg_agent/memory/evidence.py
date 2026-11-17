"""Append-only, namespaced evidence and separate decision interpretations."""
from copy import deepcopy
from hashlib import sha256
import json


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False,
                             separators=(',', ':')).encode()).hexdigest()


class EvidenceLedger:
    def __init__(self, context):
        self.context = context
        self._items = {}

    def append(self, result, *, source_evidence_id=None, input_fingerprint=None):
        key = result['evidence_id']
        if key in self._items:
            raise ValueError('Duplicate evidence ID')
        if result['window_s'] and result['window_s'][1] > result['available_at_s']:
            raise ValueError('Evidence precedes signal availability')
        item = dict(deepcopy(result), run_id=self.context.run_id, task_id=self.context.task_id,
                    record_id=self.context.record_id, manifest_id=self.context.manifest_id,
                    observed_at_s=result['available_at_s'], source_evidence_id=source_evidence_id,
                    input_fingerprint=input_fingerprint, anatomical_lead='unknown')
        self._items[key] = item
        return deepcopy(item)

    def assessment(self, decision, reason, evidence_ids, cursor, *, checked=False):
        for key in evidence_ids:
            if key not in self._items or self._items[key]['available_at_s'] > cursor:
                raise ValueError('Decision cites missing or future evidence')
        return {'decision': decision, 'decision_reason': reason, 'evidence_ids': list(evidence_ids),
                'evidence_role': {'support': 'support', 'negative': 'oppose'}.get(decision, 'unknown'),
                'recorded_at_s': cursor, 'checked': checked, 'policy_version': 'event-v1'}

    def get(self, key):
        return deepcopy(self._items[key])
