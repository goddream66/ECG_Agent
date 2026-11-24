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
        self._receipt_order = {}

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
        self._receipt_order[key] = len(self._items)
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

    def presentation_order(self, keys):
        """Order visible receipts without using namespaced hashes as semantic input."""
        def coordinates(key):
            row = self._items[key]
            window = row['window_s'] or (float('inf'), float('inf'))
            return (*window, row['lead'] or '', row['tool'], row['available_at_s'], self._receipt_order[key])
        return sorted(keys,key=coordinates)

    def retrieve(self, start_s, end_s, *, available_through_s, limit, lead=None, tool=None):
        """Return the newest acquired evidence overlapping a half-open time interval."""
        matches = []
        for key in reversed(self._items):
            item = self._items[key]
            span = item['window_s']
            if (span is None or span[1] <= start_s or span[0] >= end_s
                    or item['available_at_s'] > available_through_s
                    or (lead is not None and item['lead'] != lead)
                    or (tool is not None and item['tool'] != tool)):
                continue
            matches.append(deepcopy({field: item[field] for field in (
                'evidence_id', 'tool', 'lead', 'window_s', 'available_at_s',
                'status', 'result', 'warnings', 'error')}))
            if len(matches) == limit:
                break
        return matches
