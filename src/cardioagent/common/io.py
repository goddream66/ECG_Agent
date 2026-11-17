"""Dataset-independent JSON and fingerprint helpers."""
from hashlib import file_digest
import json
from pathlib import Path

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path,key):
    rows={}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            value=row.get(key)
            if not isinstance(value,str) or not value or value in rows:
                raise ValueError(f'Missing/duplicate {key} in {path}')
            rows[value]=row
    if not rows:raise ValueError(f'Empty JSONL: {path}')
    return rows


def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
