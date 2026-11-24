"""Record exactly the messages, schemas, raw response and latency of each model call."""
import gzip
from contextlib import nullcontext
import json
import time


class RecordingChatModel:
    def __init__(self, model, output):
        self.model = model
        self.output = output
        self.last_raw = None

    def prefix_scope(self):
        call=getattr(self.model,'prefix_scope',None)
        return call() if call else nullcontext()

    def complete(self, messages, tools):
        started = time.monotonic()
        record = dict(cursor_s=json.loads(messages[1]['content'])['window_s'][1],
                      messages=messages, tools=tools)
        self.model.last_raw = None
        try:
            response = self.model.complete(messages, tools)
            record.update(message=response.message, usage=response.usage)
            return response
        except Exception as exc:
            record['error'] = str(exc)
            raise
        finally:
            self.last_raw = getattr(self.model, 'last_raw', None)
            record.update(raw=self.last_raw, elapsed_s=time.monotonic()-started)
            with gzip.open(self.output, 'at', encoding='utf8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
