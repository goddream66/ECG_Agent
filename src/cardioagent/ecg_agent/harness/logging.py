"""Append-only file sink; decision logic only receives emit()."""
import json
from pathlib import Path

def write_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


class JsonLogs:
    def __init__(self, folder, *, append=False):
        self.mode = 'a' if append else 'w'
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files = {name: (self.folder / f'{name}.jsonl').open(self.mode, encoding='utf-8')
                      for name in ('evidence', 'acquisitions', 'assessments', 'event_updates', 'window_trace')}

    def emit(self, name, row):
        if name == 'llm_steps' and name not in self.files:
            self.files[name] = (self.folder / f'{name}.jsonl').open(self.mode, encoding='utf-8')
        write_line(self.files[name], row)

    def close(self):
        for stream in self.files.values():
            stream.close()
