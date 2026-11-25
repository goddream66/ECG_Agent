"""Evaluate CardioAgent on reference-derived ECG-Scroll window labels."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from math import isfinite, nextafter
from pathlib import Path
from time import perf_counter

from cardioagent.ecg_agent.experiments.full_reference_memory import BatchQwen, atomic_json
from cardioagent.ecg_agent.harness.staged_review import schema
from cardioagent.ecg_agent.llm.providers import parse_qwen_tool_reply
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents, merge_spans


BATCH_WINDOWS = 60
TASK_GOALS = {
    'episode_detection': 'Locate every continuous AF episode.',
    'burden_quantification': 'Track the AF episodes whose combined duration will give AF burden.',
}
SYSTEM = """You track AF events in an ECG stream. {task}
Each line gives a 30-second window's reference-derived rhythm label and AF fraction.
AF fraction is the part of that window labelled AF. It does not locate the change inside the window.
Only windows in this batch, the preceding carried window, and the event ledger are available.
Use submit_event_actions once. Return an ordered list of pairs [action, window_index].
OPEN sets the start to the beginning of that window. CLOSE sets the end to the beginning of that
window, usually the first window after AF ends. Alternate OPEN and CLOSE using the current ledger
state. You may open and close several events in a batch. Keep an ongoing event open across batches.
Use an empty list when no state change is warranted. Set wait=true if you want to revisit uncertainty
after the next batch. On the last batch, the record end bounds any event still open.
Use only the window numbers shown. Read the labels and their order before submitting."""
TOOLS = [schema('submit_event_actions', 'Commit event openings and closings ordered by window index.',
                {'actions': {'type': 'array', 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2,
                                                       'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}},
                 'wait': {'type': 'boolean'}}, ['actions'])]


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class WindowQwen(BatchQwen):
    def render_batch(self, task, observation):
        return self.adapter.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM.format(task=TASK_GOALS[task])},
             {'role': 'user', 'content': observation}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def infer(self, rendered):
        import torch

        if len(rendered) > self.limit:
            return [output for i in range(0, len(rendered), self.limit)
                    for output in self.infer(rendered[i:i+self.limit])]
        try:
            t = self.adapter.tokenizer
            inputs = t(rendered, padding=True, add_special_tokens=False, return_tensors='pt').to(self.adapter.device)
            with torch.inference_mode():
                result = self.adapter.model.generate(
                    **inputs, do_sample=False, max_new_tokens=768, pad_token_id=t.pad_token_id)
            return t.batch_decode(result[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            if len(rendered) == 1:
                raise
        self.limit = max(1, len(rendered) // 2)
        torch.cuda.empty_cache()
        return self.infer(rendered)


def parse(raw):
    reply = parse_qwen_tool_reply(raw)
    calls = reply.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'submit_event_actions':
        raise ValueError('Expected exactly one submit_event_actions')
    args = json.loads(calls[0]['function']['arguments'])
    if (not isinstance(args, dict) or set(args) - {'actions', 'wait'}
            or not isinstance(args.get('actions'), list) or type(args.get('wait', False)) is not bool):
        raise ValueError('Invalid action list or wait field')
    return args


def make_chunks(task, label):
    start, end = task['t0'], task['t1']
    if not (isfinite(start) and isfinite(end) and 0 <= start < end):
        raise ValueError(f'Invalid span for {task["id"]}')
    chunks = []
    i = 0
    while (lo := start + i * 30) < end:
        hi = min(end, lo + 30)
        evidence = label(task['record'], task['pn_dir'], lo, hi)
        if set(evidence) != {'dominant_rhythm', 'af_frac', 'confidence'}:
            raise ValueError(f'Missing rhythm evidence for {task["id"]} chunk {i}')
        chunks.append(dict(i=i, t0=lo, t1=hi, label=evidence))
        i += 1
    return chunks


class Review:
    def __init__(self, task, chunks, dest):
        self.task, self.chunks, self.dest = task, chunks, dest
        self.tracker = AutonomousEvents('fixed', task['t0'])
        self.index = 0
        self.waiting = False
        self.counts = Counter()
        self.seconds = 0.0

    def request(self):
        batch = self.chunks[self.index:self.index + BATCH_WINDOWS]
        cursor = batch[-1]['t1']
        active = self.tracker.active
        memory = dict(open_event=None if active is None else {
            k: active[k] for k in ('event_id', 'estimated_start_s', 'confirmed_at_s')},
            last_change=self.tracker.last_change, waiting=self.waiting)
        rows = [f'cursor_s={cursor} final_batch={cursor == self.task["t1"]}', f'ledger={encoded(memory)}']
        for row in self.chunks[max(0, self.index - 1):self.index + len(batch)]:
            rows.append(f'i={row["i"]} t={row["t0"]:.0f}-{row["t1"]:.0f}s label={encoded(row["label"])}')
        return '\n'.join(rows), batch, cursor

    def commit(self, raw, observation, batch, cursor):
        tracker = self.tracker
        errors, updates = [], []
        self.counts['calls'] += 1
        try:
            payload = parse(raw)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
            payload = None
            self.counts['invalid_batches'] += 1
        prior = self.index - 2
        allowed = range(max(0, self.index - 1), self.index + len(batch))
        actions = payload['actions'] if payload else []
        for action in actions:
            if (not isinstance(action, list) or len(action) != 2 or action[0] not in ('OPEN', 'CLOSE')
                    or type(action[1]) is not int or action[1] not in allowed or action[1] <= prior):
                errors.append(f'Invalid or unordered action {action!r}')
                continue
            kind, index = action
            row = self.chunks[index]
            point = round(row['t0'], 1)
            active = tracker.active
            previous_end = tracker.events[-1]['estimated_end_s'] if tracker.events and not active else None
            if ((kind == 'OPEN' and (active or previous_end is not None and point < previous_end))
                    or (kind == 'CLOSE' and (not active or point <= active['estimated_start_s']))):
                errors.append(f'{kind} at window {index} is incompatible with the current event state')
                continue
            prior = index
            evidence = f'window:{index}'
            # Each action is applied in ECG time order. The batch cursor is the time at which
            # the model could actually submit it, including when the boundary is in a carried window.
            clock = min(cursor, row['t1'])
            if clock <= tracker.cursor:
                clock = nextafter(tracker.cursor, cursor)
            update = tracker.apply(tracker.cursor, clock, dict(
                event_action=kind, boundary=dict(kind='localized', time_s=point, evidence_id=evidence,
                                                source='scroll_window_edge'),
                pending=None, next_review_s=None, support_intervals=[], opposing_intervals=[],
                evidence_ids=[evidence], reason='Qwen event action from Scroll window labels'))['event_update']
            update = dict(update, recorded_at_s=cursor, window_index=index)
            tracker.last_change = deepcopy(update)
            event = tracker.events[-1]
            if kind == 'OPEN':
                event.update(confirmed_at_s=cursor, first_seen_at_s=cursor)
            else:
                event['end_decided_at_s'] = cursor
            updates.append(update)
            self.counts[kind.lower()] += 1
        if tracker.cursor < cursor:
            tracker.skip(tracker.cursor, cursor)
        self.waiting = payload['wait'] if payload and 'wait' in payload else False
        self.counts['waits'] += self.waiting
        self.counts['invalid_actions'] += len(errors) - int(payload is None)
        self.index += len(batch)
        return dict(cursor_s=cursor, observation=observation, raw=raw, parsed=payload,
                    validation_errors=errors, event_updates=updates)

    @property
    def complete(self):
        return self.index == len(self.chunks)

    def result(self, verifier, provenance, log_sha):
        events = self.tracker.finish(self.task['t1'])
        intervals = [p for event in events for p in event['reported_intervals']]
        if self.task['task'] == 'episode_detection':
            prediction = dict(intervals=intervals)
        else:
            span = self.task['t1'] - self.task['t0']
            predicted_s = sum(b-a for a,b in merge_spans(intervals))
            prediction = dict(value=round(predicted_s/span, 4), kind='af_burden')
        return dict(id=self.task['id'], record=self.task['record'], dataset=self.task['pn_dir'],
                    task=self.task['task'], duration_s=self.task['t1']-self.task['t0'],
                    prediction=prediction, score=verifier(self.task['verifier'], prediction, self.task['answer']),
                    events=events, counts=dict(self.counts), chunks=len(self.chunks),
                    inference_seconds_attributed=self.seconds, logs_sha256=log_sha, provenance=provenance)


def run(args):
    from ecgscroll.tools import classify_rhythm
    from ecgscroll.verify import score

    tasks = json.loads(args.input.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))
    provenance = dict(input_sha256=sha(args.input), config_sha256=sha(args.config),
                      source_sha256=sha(Path(__file__)), scroll_revision=args.scroll_revision,
                      task_sha256=tasks['task_sha256'])
    reviews, complete = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks['cases']:
        dest = args.output / task['id']
        result = dest / 'result.json'
        logs = dest / 'qwen_calls.jsonl.gz'
        if result.exists():
            prior = json.loads(result.read_bytes())
            if prior['provenance'] != provenance or prior['logs_sha256'] != sha(logs):
                raise ValueError(f'Completed task provenance mismatch {task["id"]}')
            complete.append(task['id'])
            continue
        dest.mkdir(parents=True, exist_ok=True)
        reviews.append(Review(task, make_chunks(task, classify_rhythm), dest))
    if reviews:
        started = perf_counter()
        model = WindowQwen(config, args.microbatch, 'Scroll')
        print(f'Model loaded in {perf_counter()-started:.2f}s. Tasks ready: {len(reviews)}', flush=True)
        for review in reviews:
            (review.dest / 'qwen_calls.partial').unlink(missing_ok=True)
        steps = 0
        while reviews:
            stride = model.limit
            for first in range(0, len(reviews), stride):
                selected = reviews[first:first+stride]
                inputs = [review.request() for review in selected]
                rendered = [model.render_batch(review.task['task'], obs[0])
                            for review, obs in zip(selected, inputs, strict=True)]
                started = perf_counter()
                outputs = model.infer(rendered)
                share = (perf_counter()-started) / len(selected)
                for review, raw, inp in zip(selected, outputs, inputs, strict=True):
                    review.seconds += share
                    row = review.commit(raw, *inp)
                    path = review.dest / 'qwen_calls.partial'
                    with gzip.open(path, 'at', encoding='utf-8') as stream:
                        stream.write(encoded(row)+'\n')
                    if review.complete:
                        log = path.with_name('qwen_calls.jsonl.gz')
                        path.replace(log)
                        receipt = review.result(score, provenance, sha(log))
                        atomic_json(review.dest / 'result.json', receipt)
                        complete.append(review.task['id'])
                        print(f'{review.task["id"]}: score={receipt["score"]:.4f}, {receipt["counts"]}', flush=True)
            reviews = [review for review in reviews if not review.complete]
            steps += 1
            if steps % 5 == 0:
                print(f'Batches per remaining task: {steps}, complete={len(complete)}, remaining={len(reviews)}', flush=True)
    atomic_json(args.output / 'complete.json', dict(tasks=complete, provenance=provenance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'config', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--scroll-revision', required=True)
    parser.add_argument('--microbatch', type=int, default=8)
    run(parser.parse_args())
