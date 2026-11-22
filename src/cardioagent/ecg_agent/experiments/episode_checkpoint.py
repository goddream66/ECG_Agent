"""Commit a causal prefix of one autonomous replay, including tool and model logs."""
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path

from cardioagent.ecg_agent.memory.evidence import canonical_hash


LOGS = ('acquisitions.jsonl', 'evidence.jsonl', 'event_updates.jsonl', 'assessments.jsonl',
        'window_trace.jsonl', 'llm_steps.jsonl', 'qwen_calls.jsonl.gz')


class EpisodeCheckpoint:
    def __init__(self, target, job):
        self.work = target.parent / '.episode-work' / target.name
        self.folder = self.work / 'result'
        self.marker = self.work / 'checkpoint.json'
        self.job_hash = canonical_hash(job)
        self.saved = {}
        self.hashes = {name: sha256() for name in LOGS}
        self.offsets = {name: 0 for name in LOGS}
        self.execution = {}
        self.stack = ExitStack()

    def __enter__(self):
        self.work.mkdir(parents=True, exist_ok=True)
        lock = self.stack.enter_context((self.work/'writer.lock').open('a+b'))
        try:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.marker.exists():
                document = json.loads(self.marker.read_text(encoding='utf-8'))
                self.saved = document['payload']
                if canonical_hash(self.saved) != document['digest'] or self.saved['job_hash'] != self.job_hash:
                    raise ValueError('Episode checkpoint provenance or digest mismatch')
                self._restore_files()
            else:
                if self.folder.exists() and any(self.folder.iterdir()):
                    raise ValueError('Unidentified episode staging directory')
                self.folder.mkdir(exist_ok=True)
                self._save({'job_hash': self.job_hash, 'files': {}})
            return self.work
        except BaseException:
            self.stack.close()
            raise

    def _hash_prefix(self, stream, name, size):
        while size:
            block = stream.read(min(size, 1024*1024))
            if not block:
                raise ValueError(f'Truncated episode checkpoint log: {name}')
            self.hashes[name].update(block)
            size -= len(block)

    def _restore_files(self):
        for name in LOGS:
            path = self.folder/name
            size, digest = self.saved['files'].get(name, (0, self.hashes[name].hexdigest()))
            if not path.exists():
                if size:
                    raise ValueError(f'Missing episode checkpoint log: {name}')
                continue
            with path.open('rb') as stream:
                self._hash_prefix(stream, name, size)
            if self.hashes[name].hexdigest() != digest:
                raise ValueError(f'Corrupt episode checkpoint log: {name}')
            self.offsets[name] = size
        # Only uncommitted audit tails can be discarded after all prefixes are verified.
        for name, size in self.offsets.items():
            path = self.folder/name
            if path.exists() and path.stat().st_size > size:
                with path.open('r+b') as stream:
                    stream.truncate(size)

    def _save(self, payload):
        temp = self.marker.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as stream:
            json.dump({'digest': canonical_hash(payload), 'payload': payload}, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.marker)

    def commit(self, logs, tracker, reviewer, counts, elapsed_ms):
        for stream in logs.files.values():
            stream.flush()
        files = {}
        for name in LOGS:
            path = self.folder/name
            if path.exists():
                size = path.stat().st_size
                with path.open('rb') as stream:
                    stream.seek(self.offsets[name])
                    self._hash_prefix(stream, name, size-self.offsets[name])
                self.offsets[name] = size
                files[name] = [size, self.hashes[name].hexdigest()]
        self._save(dict(job_hash=self.job_hash, files=files, cursor=tracker.cursor,
                        counts=dict(counts), history=reviewer.history, model_counts=reviewer.counts,
                        elapsed_ms=elapsed_ms, execution=self.execution))

    def restore(self, ledger, tracker, reviewer, counts):
        if 'cursor' not in self.saved:
            return 0.
        def rows(name):
            with (self.folder/name).open(encoding='utf-8') as stream:
                for line in stream:
                    yield json.loads(line)
        for item in rows('evidence.jsonl'):
            ledger.append(item, source_evidence_id=item['source_evidence_id'],
                          input_fingerprint=item['input_fingerprint'])
        for row in rows('window_trace.jsonl'):
            a, b = row['start_s'], row['cursor_s']
            if tracker.cursor < a:
                tracker.skip(tracker.cursor, a)
            if row['decision'] in ('ACTION_ERROR', 'UNREVIEWED'):
                tracker.skip(a, b, retry=row['decision'] == 'ACTION_ERROR')
            else:
                tracker.apply(a, b, row['llm_action'])
        if tracker.cursor != self.saved['cursor']:
            raise ValueError('Decision journal does not reach saved cursor')
        counts.update(self.saved['counts'])
        reviewer.counts.update(self.saved['model_counts'])
        reviewer.history = self.saved['history']
        return self.saved['elapsed_ms']

    def __exit__(self, typ, value, traceback):
        if typ is None and self.marker.exists():
            self.marker.unlink()
        self.stack.close()
