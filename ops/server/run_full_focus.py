"""Run a continuous AFDB review, writing Qwen requests and accepted actions to disk."""
import argparse
import gzip
import json
from pathlib import Path
import time
from collections import Counter
from cardioagent.ecg_agent.agent import replay_event_record
from cardioagent.ecg_agent.harness.execution import build_model
from cardioagent.ecg_agent.harness.logging import JsonLogs
from cardioagent.ecg_agent.harness.react import ReActReview
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.schemas import EvidenceContext


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


class Recorder:
    def __init__(self, model, output):
        self.model = model
        self.output = output
        self.last_raw = None

    def complete(self, messages, tools):
        started = time.monotonic()
        record = dict(cursor_s=json.loads(messages[1]['content'])['window_s'][1],
                      messages=messages, tools=tools)
        self.model.last_raw = None
        self.model.last_timing = {}
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
            if self.model.last_timing:
                record["timing"] = self.model.last_timing
            with gzip.open(self.output, 'at', encoding='utf8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')


class ContinuousLogs(JsonLogs):
    def __init__(self, folder):
        super().__init__(folder)
        self.windows = 0
        self.actions = Counter()
        self.status = Counter()
        self.started = time.monotonic()

    def emit(self, name, row):
        super().emit(name, row)
        if name == 'llm_steps':
            self.status[row['status']] += 1
        if name == 'window_trace':
            self.actions[str(row.get('action'))] += 1
            self.windows += 1
            for stream in self.files.values():
                stream.flush()
            save(self.folder/'progress.json', dict(cursor_s=row['cursor_s'],
                 windows=self.windows, state=row['state'], actions=self.actions,
                 steps=self.status, elapsed_s=time.monotonic()-self.started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    data = args.storage/'data'
    cache = data/'afdb_preprocessed_server'
    manifest = json.loads((cache/'manifest.json').read_text())
    record = next(r for r in manifest['records'] if r['record_id']==args.record)
    if record['source_samples'] != record['source_record_samples']:
        raise ValueError('Incomplete source cache')
    end = record['source_samples']/record['source_rate_hz']
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    task = dict(task_id=f'full-{args.record}',start_s=0,end_s=end)
    save(args.output/'experiment.json',dict(task=task,record=record,llm=config))
    scorer, info, load_ms = build_model('live',args.storage/'models',2,'cpu')
    model=Recorder(build_chat_model(config),args.output/'qwen_calls.jsonl.gz')
    provider_keys={'provider','model_path','device','max_new_tokens'}
    reviewer=ReActReview(model,live_scoring=True,**{k:v for k,v in config.items() if k not in provider_keys})
    logs=ContinuousLogs(args.output)
    try:
        result=replay_event_record(cache,record,task,'ECG1',EventConfig(),lambda lead,x,loc:scorer(x),
                logs,EvidenceContext(args.output.name,task['task_id'],args.record),
                policy='event-react',raw_root=data/'afdb_local',reviewer=reviewer)
        if logs.windows!=(record['source_samples']+2499)//2500:
            raise RuntimeError('Incomplete window coverage')
        result.update(founder=info,founder_load_ms=load_ms)
        save(args.output/'result.json',result)
    finally:
        logs.close()


if __name__=='__main__':
    main()
