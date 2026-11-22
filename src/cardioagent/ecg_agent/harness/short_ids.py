"""Record-scoped display IDs. Canonical evidence stays in the ledger."""
import json
import re
from pathlib import Path

HASH = re.compile(r'ecg:[0-9a-f]{64}(?![0-9a-f])')


class EvidenceAliases:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.forward = {}
        self.reverse = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                alias, key = json.loads(line)
                if (alias != f'E{len(self.forward)+1}' or key in self.forward
                        or not HASH.fullmatch(key)):
                    raise ValueError('Invalid evidence alias journal')
                self.forward[key] = alias
                self.reverse[alias] = key

    def display(self, key):
        if key not in self.forward:
            if not HASH.fullmatch(key):
                raise ValueError('Expected canonical evidence ID')
            alias = f'E{len(self.forward)+1}'
            if self.path:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps([alias, key])+'\n')
            self.forward[key] = alias
            self.reverse[alias] = key
        return self.forward[key]

    def messages(self, messages):
        # Substitution in serialized JSON also covers nested tool arguments and feedback.
        encoded = json.dumps(messages, ensure_ascii=False)
        return json.loads(HASH.sub(lambda m: self.display(m.group()), encoded))

    def resolve(self, key):
        if key not in self.reverse:
            raise ValueError('Use a short evidence ID shown in the review')
        return self.reverse[key]

    def arguments(self, args):
        out = dict(args)
        if 'evidence_id' in out:
            out['evidence_id'] = self.resolve(out['evidence_id'])
        if isinstance(out.get('evidence_ids'), list):
            out['evidence_ids'] = [self.resolve(k) for k in out['evidence_ids']]
        return out

