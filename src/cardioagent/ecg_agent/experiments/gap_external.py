"""Frozen gap-repair validation on LTAFDB ECG-Scroll tasks."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from typing import Callable

import numpy as np
import wfdb

from cardioagent.common.intervals import union_seconds, validate_intervals
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.agent import replay_event_record
from cardioagent.ecg_agent.data.offline import write_record
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, intersection
from cardioagent.ecg_agent.evaluation.references import prepare_validity
from cardioagent.ecg_agent.experiments.episode_review import merge_gaps
from cardioagent.ecg_agent.experiments.gap_budget import BUDGET, accepts, apply_observation, next_action
from cardioagent.ecg_agent.experiments.gap_repair import LEADS, paired_delta, read_json
from cardioagent.ecg_agent.experiments.original_window_review import SCORE, action_plan, gap_windows
from cardioagent.ecg_agent.experiments.qwen_gap_relationship import reference_relation
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.memory.episode_review import review_packets
from cardioagent.ecg_agent.memory.evidence import canonical_hash
from cardioagent.ecg_agent.models.ecgfounder import ecgfounder_execution, load_ecgfounder
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.schemas import EvidenceContext


SOURCE_FS = 128
METHODS = ('no_repair', 'strict', 'temporal_67%', 'temporal_fixed', 'temporal_feasible')
CONFIG = code_root() / 'configs/ecg/policy_hysteresis060040.json'
RHYTHM_LABELS = ('AB', 'AFIB', 'AFL', 'B', 'IVR', 'N', 'SBR', 'SVTA', 'T', 'VT')
FIXED_CODE = (
    'scripts/experiment_afdb_quality.py',
    'third_party/ecgfounder/net1d.py',
    'third_party/ecgfounder/tasks.txt',
)


def code_hashes():
    root = code_root()
    files = (*sorted((root / 'src/cardioagent/ecg_agent').rglob('*.py')),
             *sorted((root / 'src/cardioagent/common').rglob('*.py')),
             *(root / name for name in FIXED_CODE))
    return {path.relative_to(root).as_posix(): digest(path) for path in files}


class PacketLogs:
    """Retain the same fields required by the frozen packet generator."""

    def __init__(self):
        self.rows = {key: [] for key in ('window_trace', 'event_updates', 'evidence')}

    def emit(self, name, row):
        if name in self.rows:
            self.rows[name].append(row)

    def packets(self, result, threshold):
        return list(review_packets(self.rows['window_trace'], self.rows['event_updates'], self.rows['evidence'],
                                   result['events'], start_threshold=threshold))


def prepare(raw, tasks, models, output, *, smoke=False, await_waveforms=False):
    rows = list(read_jsonl(tasks, 'id').values())
    rows = [row for row in rows if row['pn_dir'] == 'ltafdb']
    episodic = {row['record']: row for row in rows if row['task'] == 'episode_detection'}
    burdens = {row['record']: row for row in rows if row['task'] == 'burden_quantification'
               and row['answer']['kind'] == 'af_burden'}
    if len(rows) != 168 or len(episodic) != 84 or episodic.keys() != burdens.keys():
        raise ValueError('Expected all 84 paired full-record ECG-Scroll AF localization and burden tasks')
    if smoke:
        episodic = {'00': episodic['00']}
    source_hashes = {name.removeprefix('./'): sha for sha, name in
                     (line.split(maxsplit=1) for line in (raw.parent / 'SHA256SUMS.txt').read_text().splitlines() if line.strip())}
    output.mkdir(parents=True, exist_ok=False)
    adapted = output / 'channels'
    adapted.mkdir()
    public, truth, files, adapted_hashes = [], [], {}, {}
    for name, row in sorted(episodic.items()):
        if not re.fullmatch(r'[0-9]{2,3}', name):
            raise ValueError(f'Invalid LTAFDB record identifier: {name}')
        for ext in ('hea', 'dat', 'atr'):
            basename = f'{name}.{ext}'
            source = raw / basename
            deferred = ext == 'dat' and await_waveforms and not source.exists()
            if not deferred and (not source.is_file() or digest(source) != source_hashes[basename]):
                raise ValueError(f'PhysioNet source hash mismatch: {basename}')
            files[basename] = source_hashes[basename]
        header = wfdb.rdheader(str(raw / name))
        if header.fs != SOURCE_FS or header.n_sig != 2 or header.units != ['mV', 'mV']:
            raise ValueError(f'{name}: expected two 128 Hz millivolt channels')
        if set(header.file_name) != {f'{name}.dat'}:
            raise ValueError(f'{name}: unexpected WFDB signal file layout')
        lines = (raw / f'{name}.hea').read_text(encoding='ascii').splitlines()
        for i, lead in enumerate(LEADS, start=1):
            # PhysioNet uses non-unique channel descriptions. Use positional names.
            lines[i] = ' '.join(lines[i].split()[:8] + [lead])
        adapted_header = adapted / f'{name}.hea'
        adapted_header.write_text('\n'.join(lines)+'\n', encoding='ascii')
        adapted_hashes[name] = digest(adapted_header)
        (adapted / f'{name}.dat').symlink_to((raw / f'{name}.dat').resolve())
        if tuple(wfdb.rdheader(str(adapted / name)).sig_name) != LEADS:
            raise ValueError(f'{name}: channel adaptation failed')
        task_id = row['id']
        start = float(row['t0'])
        end = min(float(row['t1']), header.sig_len / SOURCE_FS)
        burden = burdens[name]
        tau = row['verifier']['params']['tau']
        if (start != 0 or abs(header.sig_len / SOURCE_FS - row['t1']) > .005 or end <= start
                or (row['t0'], row['t1']) != (burden['t0'], burden['t1']) or tau != .5):
            raise ValueError(f'{name}: task span does not fit source coordinate grid')
        original = validate_intervals(row['answer']['intervals'], start, float(row['t1']))
        intervals = [[max(start, a), min(end, b)] for a, b in original
                     if min(end, b) > max(start, a)]
        public.append(dict(task_id=task_id, record_id=name, start_s=start, end_s=end, source_database='ltafdb'))
        truth.append(dict(task_id=task_id, start_s=start, end_s=end, intervals=intervals,
                          burden=float(burden['answer']['value']), tau=tau,
                          burden_floor=burden['verifier']['params']['floor']))
    write_jsonl(output / 'tasks.jsonl', public)
    write_jsonl(output / 'references.jsonl', truth)
    manifest = dict(records=len(public), raw_root=str(raw.resolve()), raw_files=files,
                    adapted_headers=adapted_hashes, code=code_hashes(),
                    source_tasks_sha256=digest(tasks), tasks_sha256=digest(output / 'tasks.jsonl'),
                    references_sha256=digest(output / 'references.jsonl'), config_sha256=digest(CONFIG),
                    checkpoint_sha256=digest(models / '1_lead_ECGFounder.pth'),
                    model_input={'channels': LEADS, 'source_fs': SOURCE_FS, 'window_s': 10, 'model_fs': 500},
                    methods=METHODS, budget=BUDGET, gap_limit_s=60, threshold=.6,
                    temporal_min_support=2/3, temporal_max_low_s=10, temporal_unknown_allowed_s=0,
                    scoring='original benchmark intervals and greedy one-to-one matching',
                    validity_rhythms=RHYTHM_LABELS, smoke=smoke, waveforms_verified_before_use=True)
    save_json(output / 'manifest.json', manifest)


def gpu_model(models, device, batch_size):
    import torch

    info = ecgfounder_execution(device)
    target = torch.device(info['device'])
    torch.set_num_threads(2)
    if target.type == 'cuda':
        torch.cuda.set_device(target)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    started = perf_counter()
    model, labels, _ = load_ecgfounder(models)
    model = model.to(target)
    if target.type == 'cuda':
        torch.cuda.synchronize(target)
    indices = [labels.index(name) for name in ('ATRIAL FIBRILLATION', 'ATRIAL FLUTTER')]
    info.update(load_s=perf_counter()-started, torch_version=torch.__version__, batch_size=batch_size)

    def batch(waves):
        out = np.empty((len(waves), 2), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(waves), batch_size):
                data = torch.from_numpy(waves[i:i+batch_size].copy()).to(target)
                logits, _ = model(data[:, None, :])
                out[i:i+len(data)] = logits[:, indices].sigmoid().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError('ECGFounder returned non-finite scores')
        return out

    return batch, info


def live_repair(packet, reader, context, method):
    """Only the chosen action is actually executed and admitted to the packet."""
    state = deepcopy(packet)
    todo = [dict(window_s=[a,b], lead=lead, tool=tool, evidence_id=None)
            for a,b,lead,tool in action_plan(gap_windows(state), state['start_threshold'])]
    used = []
    with_time = perf_counter()
    while len(used) < BUDGET and not accepts(state, 'temporal_67%'):
        item = next_action(state, todo, 'temporal_67%', method, BUDGET-len(used))
        if item is None:
            break
        todo.remove(item)
        a, b = item['window_s']
        session = reader.session(a, b, packet['cursor_s'], context)
        measured = session.call(item['tool'], dict(lead=item['lead'], start_s=a, end_s=b))
        used.append(measured)
        apply_observation(state, measured)
    return dict(link=accepts(state, 'temporal_67%'), elapsed_s=perf_counter()-with_time, actions=used)


def repair_predictions(intervals, packets, reader, context):
    decisions = []
    picked = defaultdict(set)
    for packet in packets:
        gap = tuple(packet['gap_s'])
        strict = all(packet['gap_support'])
        temporal = accepts(packet, 'temporal_67%')
        fixed = live_repair(packet, reader, context, 'fixed')
        feasible = live_repair(packet, reader, context, 'feasible_adaptive')
        for method, link in zip(METHODS, (False, strict, temporal, fixed['link'], feasible['link'])):
            if link:
                picked[method].add(gap)
        decisions.append(dict(gap_s=gap, cursor_s=packet['cursor_s'], left=packet['left']['interval'],
                              right=packet['right']['interval'], strict=strict, temporal_67=temporal,
                              temporal_fixed=fixed, temporal_feasible=feasible))
    return dict(predictions={method: merge_gaps(intervals, picked[method]) for method in METHODS}, decisions=decisions)


def score_record(task, prediction, reference, validity):
    expected = reference['intervals']
    results = {}
    for method, intervals in prediction['predictions'].items():
        scored = evaluate_record(task['task_id'], intervals, reference, valid=validity)
        counts = Counter()
        decisions = prediction['decisions']
        for case in decisions:
            action = case.get(method)
            if method == 'temporal_67%':
                action = case['temporal_67']
            if isinstance(action, dict):
                calls = action['actions']
                counts['calls'] += len(calls)
                counts['model_calls'] += sum(item['tool'] == SCORE and item['status'] == 'ok' for item in calls)
                counts['repair_wall_s'] += action['elapsed_s']
                counts['model_tool_s'] += sum(item['duration_ms']/1000 for item in calls if item['tool'] == SCORE)
                counts['other_tool_s'] += sum(item['duration_ms']/1000 for item in calls if item['tool'] != SCORE)
                action = action['link']
            if not action:
                continue
            gap = case['gap_s']
            label = reference_relation(gap, case['left'], case['right'], expected, validity)
            counts['correct_links'] += label == 'continuous'
            counts['false_links'] += label == 'separate'
            counts['unscored_links'] += label == 'unscored'
            known = intersection([gap], validity)
            true_s = union_seconds(intersection(known, expected))
            counts['recovered_af_s'] += true_s
            counts['added_non_af_s'] += max(0, union_seconds(known)-true_s)
        for key in ('correct_links', 'false_links', 'unscored_links', 'recovered_af_s', 'added_non_af_s', 'calls',
                    'model_calls', 'repair_wall_s', 'model_tool_s', 'other_tool_s'):
            scored[key] = counts[key]
        scored['record_id'] = task['record_id']
        scored['candidates'] = len(decisions)
        results[method] = scored
    return results


def verified_result(artifacts, manifest_id, task):
    body = read_json(artifacts / 'result.json')
    if (body['manifest_id'] != manifest_id or body['task'] != task or set(body['results']) != set(LEADS)
            or any(set(methods) != set(METHODS) for methods in body['results'].values())
            or body['predictions_sha256'] != digest(artifacts / 'predictions.json')
            or body['validity_sha256'] != digest(artifacts / 'validity.json')):
        raise ValueError(f'{task["record_id"]}: incompatible saved record')
    return body


def execute(task, output, manifest, batch, review: Callable, *, crop_s=None):
    name = task['record_id']
    adapted = output / 'channels'
    if crop_s:
        task = {**task, 'end_s': min(task['end_s'], crop_s)}
    artifacts = output / ('smoke' if crop_s else 'records') / name
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_id = canonical_hash(manifest)
    raw = Path(manifest['raw_root'])
    if ((adapted / f'{name}.dat').resolve() != (raw / f'{name}.dat').resolve()
            or digest(adapted / f'{name}.hea') != manifest['adapted_headers'][name]
            or any(digest(raw / f'{name}.{ext}') != manifest['raw_files'][f'{name}.{ext}'] for ext in ('dat', 'atr'))):
        raise ValueError(f'{name}: frozen waveform or rhythm labels changed')
    target = artifacts / 'result.json'
    if target.exists():
        return verified_result(artifacts, manifest_id, task)
    config = EventConfig(**read_json(CONFIG))
    began = perf_counter()
    with TemporaryDirectory(prefix='ltafdb-', dir=artifacts) as temp:
        cache = Path(temp)
        cached = write_record(adapted / name, cache, crop_s, source_fs=SOURCE_FS)
        values = np.load(cache / cached['waveforms'], mmap_mode='r')
        preproc_s = perf_counter()-began
        start = perf_counter()
        scores = batch(values)
        del values
        forward_s = perf_counter()-start
        reader = HistoricalWindow(adapted, task, lambda values: batch(values[None,:])[0].tolist(), source_fs=SOURCE_FS)
        predictions = {}
        for lead in LEADS:
            context = EvidenceContext(output.name + '-' + lead, task['task_id'], name, manifest_id)
            logs = PacketLogs()
            # Index access occurs in the already-arrived window's score tool only.
            raw_result = replay_event_record(cache, cached, task, lead, config, lambda _, __, i: scores[i].tolist(),
                                             logs, context, raw_root=adapted, rule_review=review, source_fs=SOURCE_FS)
            packets = logs.packets(raw_result, config.threshold)
            predictions[lead] = repair_predictions(raw_result['intervals'], packets, reader, context)
        write_jsonl(cache / 'task.jsonl', [task])
        availability = prepare_validity(cache / 'task.jsonl', raw, cache, artifacts / 'validity.json',
                                        rhythm_labels=tuple(manifest['validity_rhythms']),
                                        scope='LTAFDB external evaluation')[task['task_id']]
        unknown_labels = {note.lstrip('(').rstrip(')') for note in availability['observed_rhythm_labels']} - set(RHYTHM_LABELS)
        if unknown_labels:
            raise ValueError(f'{name}: unexpected reference rhythm labels {unknown_labels}')
        validity = availability['valid_intervals']
        references = read_jsonl(output / 'references.jsonl', 'task_id')
        reference = references[task['task_id']]
        if crop_s:
            end = task['end_s']
            intervals = intersection(reference['intervals'], [[0., end]])
            reference = {**reference, 'end_s': end, 'intervals': intervals, 'burden': union_seconds(intervals)/end}
        results = {lead: score_record(task, rows, reference, validity) for lead, rows in predictions.items()}
        save_json(artifacts / 'predictions.json', predictions)
        body = dict(manifest_id=manifest_id, task=task, results=results, preproc_s=preproc_s, forward_s=forward_s,
                    cached_forward_count=len(scores), predictions_sha256=digest(artifacts / 'predictions.json'),
                    validity_sha256=digest(artifacts / 'validity.json'),
                    total_s=perf_counter()-began)
        save_json(artifacts / 'result.tmp', body)
        (artifacts / 'result.tmp').replace(target)
        return body


def aggregate_run(output, tasks, manifest):
    rows = defaultdict(list)
    timings = Counter()
    manifest_id = canonical_hash(manifest)
    for name, task in sorted(tasks.items()):
        artifacts = output / 'records' / name
        body = verified_result(artifacts, manifest_id, task)
        for key in ('preproc_s', 'forward_s', 'total_s', 'cached_forward_count'):
            timings[key] += body[key]
        for lead, methods in body['results'].items():
            for method, result in methods.items():
                rows[lead, method].append(result)
    report = []
    for (lead, method), values in rows.items():
        metrics = aggregate(values)
        extras = ('candidates', 'correct_links', 'false_links', 'unscored_links', 'recovered_af_s', 'added_non_af_s',
                  'calls', 'model_calls', 'repair_wall_s', 'model_tool_s', 'other_tool_s')
        metrics.update({key: sum(row[key] for row in values) for key in extras})
        report.append(dict(lead=lead, method=method, metrics=metrics, per_record=values,
                           paired_vs_strict=paired_delta(values, rows[lead, 'strict']),
                           paired_vs_fixed=paired_delta(values, rows[lead, 'temporal_fixed'])))
    save_json(output / 'comparison.json', dict(records=len(tasks), manifest_id=manifest_id,
                                              shared_scan=timings, results=report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--await-waveforms', action='store_true', help='Work while verified waveform files are arriving')
    args = parser.parse_args()
    if args.prepare:
        if args.raw is None or args.tasks is None:
            parser.error('--prepare requires --raw and --tasks')
        prepare(args.raw, args.tasks, args.models, args.output, smoke=args.smoke,
                await_waveforms=args.await_waveforms)
        return
    if not 0 <= args.shard < args.shards or args.batch_size < 1:
        parser.error('Expected a valid shard and a positive batch size')
    manifest = read_json(args.output / 'manifest.json')
    if code_hashes() != manifest['code']:
        raise ValueError('External validation code was modified after freezing the manifest')
    if (digest(CONFIG) != manifest['config_sha256'] or digest(args.models / '1_lead_ECGFounder.pth') != manifest['checkpoint_sha256']
            or digest(args.output / 'tasks.jsonl') != manifest['tasks_sha256']
            or digest(args.output / 'references.jsonl') != manifest['references_sha256']):
        raise ValueError('Frozen data, configuration or model no longer agrees with the manifest')
    tasks = read_jsonl(args.output / 'tasks.jsonl', 'record_id')
    if args.smoke != manifest['smoke'] or len(tasks) != manifest['records']:
        raise ValueError('Requested evaluation does not match the frozen cohort')
    if args.aggregate:
        if args.smoke:
            parser.error('The smoke prefix cannot be aggregated as an external evaluation')
        aggregate_run(args.output, tasks, manifest)
        return
    from scripts.experiment_afdb_quality import quality_as_hint
    batch, info = gpu_model(args.models, args.device, args.batch_size)
    print(json.dumps(info), flush=True)
    ids = sorted(tasks) if not args.smoke else ['00']
    pending = ids[args.shard::args.shards]
    deadline = perf_counter() + 10800
    while pending:
        name = next((item for item in pending if (args.output / 'channels' / f'{item}.dat').is_file()), None)
        if name is None:
            if not args.await_waveforms or perf_counter() >= deadline:
                raise FileNotFoundError(f'Frozen waveform inputs missing: {pending}')
            sleep(10)
            continue
        pending.remove(name)
        result = execute(tasks[name], args.output, manifest, batch, quality_as_hint, crop_s=600 if args.smoke else None)
        print(f'{name}: {result["cached_forward_count"]} scores, {result["forward_s"]:.2f}s GPU, '
              f'{result["total_s"]:.2f}s total', flush=True)


if __name__ == '__main__':
    main()
