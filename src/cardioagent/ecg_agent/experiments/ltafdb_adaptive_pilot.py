"""Freeze an external ECG-Scroll clip panel and measure its baseline Founder scores."""

import argparse
import json
from hashlib import sha256
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.data.afdb import CHANNELS, _arrivals
from cardioagent.ecg_agent.data.offline import cache_manifest, write_record
from cardioagent.ecg_agent.evaluation.metrics import intersection
from cardioagent.ecg_agent.experiments.gap_external import SOURCE_FS
from cardioagent.ecg_agent.harness.execution import build_model


WINDOW_S = 120
STRATA = ('short_event', 'event_end', 'background')
SELECTION_SEED = 'cardioagent-ltafdb-adaptive-20261124-v1'
CONTINUOUS_SEED = 'cardioagent-ltafdb-continuous-20261124-v1'
TRANSITION_SEED = 'cardioagent-ltafdb-grid-transition-20261124-v1'
TRANSITION_STRATA = ('short_inside', 'long_end_inside', 'persistent_target', 'target_absent')


def record_candidates(task, intervals, valid):
    def centered(time):
        a = 10 * floor((time - WINDOW_S / 2) / 10)
        return a, a + WINDOW_S

    for a, b in intervals:
        if 5 <= b - a < 30:
            yield 'short_event', *centered((a + b) / 2), [[a, b]]
        if b - a >= 65:
            start, end = centered(b)
            yield 'event_end', start, end, [[start, b]]
    for v0, v1 in valid:
        free_start = max(v0, task['start_s'])
        for a, b in [*intervals, (min(v1, task['end_s']), min(v1, task['end_s']))]:
            free_end = min(a, v1, task['end_s'])
            if free_end - free_start >= WINDOW_S + 20:
                yield 'background', *centered((free_start + free_end) / 2), []
            free_start = max(free_start, b)
            if free_start >= v1:
                break


def select_clips(tasks, references, validity, per_stratum=4):
    candidates = {stratum: [] for stratum in STRATA}
    for key, task in tasks.items():
        name = task['record_id']
        valid = validity[name]['valid_intervals']
        intervals = references[key]['intervals']
        for stratum, a, b, expected in record_candidates(task, intervals, valid):
            if (task['start_s'] <= a < b <= task['end_s']
                    and any(v0 <= a < b <= v1 for v0, v1 in valid)
                    and intersection(intervals, [[a, b]]) == expected):
                rank = sha256(f'{SELECTION_SEED}:{stratum}:{name}:{a}'.encode()).digest()
                candidates[stratum].append((rank, name, a, b, expected, key))
    selected, used = [], set()
    for stratum in STRATA:
        count = 0
        for _, name, a, b, expected, source_id in sorted(candidates[stratum]):
            if name in used:
                continue
            task_id = f'ltafdb-adaptive-{name}'
            task = dict(task_id=task_id, record_id=name, source_database='ltafdb', start_s=a, end_s=b)
            reference = dict(task_id=task_id, start_s=a, end_s=b, intervals=expected,
                             burden=union_seconds(expected) / WINDOW_S, tau=.5, burden_floor=.02)
            selected.append((task, reference, stratum, source_id))
            used.add(name)
            count += 1
            if count == per_stratum:
                break
        if count != per_stratum:
            raise ValueError(f'Only {count} unique eligible records for {stratum}')
    return selected


def continuous_sections(tasks, validity, excluded, count=4, seconds=1200):
    """Rank records and aligned positions without using rhythm annotations or detector scores."""
    def rank(*parts):
        return sha256(':'.join((CONTINUOUS_SEED,*parts)).encode()).digest()

    eligible = []
    for key, task in tasks.items():
        name = task['record_id']
        if name in excluded:
            continue
        spans = []
        for left, right in validity[name]['valid_intervals']:
            lo = 10 * ceil(max(left, task['start_s']) / 10)
            hi = 10 * floor((min(right, task['end_s'])-seconds) / 10)
            if hi >= lo:
                spans.append(range(lo, hi+1, 10))
        positions = sum(map(len,spans))
        if positions:
            index = int.from_bytes(rank(name,'position'),'big') % positions
            for span in spans:
                if index < len(span):
                    eligible.append((rank(name,'record'), name, span[index], key))
                    break
                index -= len(span)
    chosen = sorted(eligible)[:count]
    if len(chosen) != count or len({row[1] for row in chosen}) != count:
        raise ValueError('Insufficient distinct eligible records for continuous evaluation')
    return [(key, dict(task_id=f'ltafdb-adaptive-{name}', record_id=name, source_database='ltafdb',
                       start_s=start, end_s=start+seconds)) for _, name, start, key in chosen]


def transition_sections(tasks, references, validity, excluded, per_stratum=2, seconds=300):
    """Select evaluation blocks on a pre-aligned grid, retaining every reference in each block."""
    def rank(*parts):
        return sha256(':'.join((TRANSITION_SEED, *parts)).encode()).digest()

    candidates = {kind: [] for kind in TRANSITION_STRATA}
    for key, task in tasks.items():
        name = task['record_id']
        if name in excluded:
            continue
        intervals = references[key]['intervals']
        offset = 10 * (int.from_bytes(rank(name, 'grid'), 'big') % (seconds // 10))
        for left, right in validity[name]['valid_intervals']:
            first = offset + seconds * ceil((max(left, task['start_s']) - offset) / seconds)
            limit = floor(min(right, task['end_s']))
            for a in range(first, limit - seconds + 1, seconds):
                b = a + seconds
                inside = [(x, y) for x, y in intervals if x < b and a < y]
                if any(a < x < y < b and 5 <= y - x < 30 for x, y in inside):
                    kind = 'short_inside'
                elif any(a < y < b and y - x >= 60 for x, y in inside):
                    kind = 'long_end_inside'
                elif any(x <= a < b <= y for x, y in inside):
                    kind = 'persistent_target'
                elif not inside:
                    kind = 'target_absent'
                else:
                    continue
                candidates[kind].append((rank(kind, name), rank(kind, name, str(a)), name, key, a))
    chosen, used = [], set()
    for kind in TRANSITION_STRATA:
        count = 0
        for _, _, name, key, a in sorted(candidates[kind]):
            if name in used:
                continue
            task = dict(task_id=f'ltafdb-adaptive-{name}', record_id=name, source_database='ltafdb',
                        start_s=a, end_s=a + seconds)
            chosen.append((key, task, kind))
            used.add(name)
            count += 1
            if count == per_stratum:
                break
        if count != per_stratum:
            raise ValueError(f'Only {count} eligible new records in stratum {kind}')
    return chosen


def prepare(source, output, *, exclude_selection=None, transition_exclusion=None):
    if transition_exclusion is not None and exclude_selection is None:
        raise ValueError('Both prior selection files are required for the transition panel')
    manifest = json.loads((source / 'manifest.json').read_bytes())
    for file, field in [('tasks.jsonl', 'tasks_sha256'), ('references.jsonl', 'references_sha256')]:
        if digest(source / file) != manifest[field]:
            raise ValueError(f'External source has changed: {file}')
    tasks = read_jsonl(source / 'tasks.jsonl', 'task_id')
    references = read_jsonl(source / 'references.jsonl', 'task_id')
    validity = {}
    for task in tasks.values():
        name = task['record_id']
        bundle = source / 'records' / name
        path = bundle / 'validity.json'
        if digest(path) != json.loads((bundle / 'result.json').read_bytes())['validity_sha256']:
            raise ValueError(f'External validity receipt changed for {name}')
        validity[name] = json.loads(path.read_bytes())['records'][task['task_id']]
    raw, adapted = Path(manifest['raw_root']), source / 'channels'
    excluded = set()
    if exclude_selection is None:
        clips = select_clips(tasks, references, validity)
    else:
        for path in (exclude_selection, transition_exclusion):
            if path is None:
                continue
            prior = json.loads(path.read_bytes())
            if prior['source_manifest_sha256'] != digest(source/'manifest.json'):
                raise ValueError('The prior selection must refer to the same source')
            excluded.update(row['record_id'] for row in prior['clips'])
        if transition_exclusion is None:
            sections = [(key, task, 'continuous') for key, task in continuous_sections(tasks, validity, excluded)]
        else:
            sections = transition_sections(tasks, references, validity, excluded)
        clips = []
        for key, task, stratum in sections:
            a, b = task['start_s'], task['end_s']
            events = intersection(references[key]['intervals'], [[a,b]])
            reference = dict(task_id=task['task_id'], start_s=a, end_s=b, intervals=events,
                             burden=union_seconds(events)/(b-a), tau=.5, burden_floor=.02)
            clips.append((task, reference, stratum, key))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ltafdb-adaptive-', dir=output.parent) as temp:
        staging = Path(temp) / 'prepared'
        staging.mkdir()
        records, selection = [], []
        for task, _, stratum, source_id in clips:
            name = task['record_id']
            if (digest(adapted / f'{name}.hea') != manifest['adapted_headers'][name]
                    or (adapted / f'{name}.dat').resolve() != (raw / f'{name}.dat').resolve()
                    or any(digest(raw / f'{name}.{suffix}') != manifest['raw_files'][f'{name}.{suffix}']
                           for suffix in ('dat', 'atr'))):
                raise ValueError(f'External record source has changed: {name}')
            records.append(write_record(adapted / name, staging, task['end_s'], source_fs=SOURCE_FS))
            selection.append(dict(record_id=name, start_s=task['start_s'], end_s=task['end_s'],
                                  selection=stratum, source_task_id=source_id))
            print(f'{name}: 128 Hz cache through {task["end_s"]} s', flush=True)
        save_json(staging / 'manifest.json', cache_manifest(adapted, records, 'LTAFDB'))
        write_jsonl(staging / 'tasks.jsonl', [task for task, _, _, _ in clips])
        evaluation = staging / 'evaluation'
        evaluation.mkdir()
        write_jsonl(evaluation / 'references.jsonl', [reference for _, reference, _, _ in clips])
        protocol = (dict(seed=CONTINUOUS_SEED, seconds_per_clip=1200, exclusion_sha256=digest(exclude_selection),
                         excluded_records=sorted(excluded)) if exclude_selection is not None
                    else dict(seed=SELECTION_SEED, seconds_per_clip=WINDOW_S))
        if transition_exclusion is not None:
            protocol.update(seed=TRANSITION_SEED, seconds_per_clip=300,
                            second_exclusion_sha256=digest(transition_exclusion))
        save_json(evaluation / 'selection.json', dict(
            **protocol, clips=selection,
            source_manifest_sha256=digest(source / 'manifest.json'),
            tasks_sha256=digest(staging / 'tasks.jsonl'), references_sha256=digest(evaluation / 'references.jsonl')))
        staging.rename(output)


def score_baseline(cache, task_path, models, device, output):
    task, = read_jsonl(task_path, 'task_id').values()
    manifest = json.loads((cache / 'manifest.json').read_bytes())
    record, = (row for row in manifest['records'] if row['record_id'] == task['record_id'])
    predictor, model, load_ms = build_model('live', models, 2, device)
    rows = []
    began = perf_counter()
    for block in _arrivals(cache, record, CHANNELS, task['start_s'], task['end_s'], source_fs=SOURCE_FS):
        bounds = block[CHANNELS[0]][0]
        scores = {lead: predictor(values) if values is not None else None for lead, (_, values) in block.items()}
        rows.append(dict(start_s=bounds['start_s'], end_s=bounds['end_s'], scores=scores))
    if len(rows) * 10 != task['end_s'] - task['start_s']:
        raise ValueError('Incomplete baseline time coverage')
    elapsed_s = perf_counter() - began
    save_json(output, dict(rows=rows, model=model, model_load_ms=load_ms, wall_s=elapsed_s,
                           founder_calls=sum(value is not None for row in rows for value in row['scores'].values()),
                           cache_sha256=digest(cache / 'manifest.json'), task_sha256=digest(task_path),
                           checkpoint_sha256=digest(Path(model['checkpoint'])),
                           waveform_files={name: digest(cache / name)
                                           for name in (record['waveforms'], record['window_index'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('prepare')
    build.add_argument('--source', required=True, type=Path)
    build.add_argument('--output', required=True, type=Path)
    build.add_argument('--exclude-selection', type=Path, help='Draw continuous segments on other records.')
    build.add_argument('--transition-exclusion', type=Path,
                       help='Exclude an additional previous panel and evaluate strata on a fixed time grid.')
    baseline = sub.add_parser('baseline')
    baseline.add_argument('--cache', required=True, type=Path)
    baseline.add_argument('--task', required=True, type=Path)
    baseline.add_argument('--models', required=True, type=Path)
    baseline.add_argument('--device', default='cuda:0')
    baseline.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.transition_exclusion is not None and args.exclude_selection is None:
            parser.error('--transition-exclusion needs the original --exclude-selection as well')
        prepare(args.source, args.output, exclude_selection=args.exclude_selection,
                transition_exclusion=args.transition_exclusion)
    else:
        score_baseline(args.cache, args.task, args.models, args.device, args.output)


if __name__ == '__main__':
    main()
