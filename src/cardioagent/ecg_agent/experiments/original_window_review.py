"""Compare no new evidence, all original-window measurements, and a budgeted rule."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, intersection, paired_bootstrap
from cardioagent.ecg_agent.experiments.analysis import jsonlines, trace_data
from cardioagent.ecg_agent.experiments.episode_review import collect, merge_gaps
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.memory.evidence import EvidenceLedger, canonical_hash
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
from cardioagent.ecg_agent.schemas import EvidenceContext


QUALITY = 'assess_quality'
SCORE = 'score_ecgfounder'


def gap_windows(packet):
    a, b = packet['gap_s']
    return [w for w in packet['windows'] if a <= w['window_s'][0] < b]


def action_plan(windows, threshold):
    actions = []
    for window in windows:
        for lead, item in window['leads'].items():
            missing = [tool for tool, key in ((SCORE, 'score_id'), (QUALITY, 'quality_id')) if key not in item]
            plausible = (item['score'] is None or max(item['score']) >= threshold) and item['quality'] in (
                'not_measured', 'no_warning') and item['score_status'] in ('not_measured', 'ok')
            for tool in missing:
                actions.append((not plausible, len(missing), *window['window_s'], lead, tool))
    return [(a, b, lead, tool) for _, _, a, b, lead, tool in sorted(actions)]


def support_refs(packet, evidence):
    threshold = packet['start_threshold']
    refs = []
    for window in gap_windows(packet):
        a, b = window['window_s']
        qualifying = []
        for lead, item in window['leads'].items():
            score, quality = item['score'], item['quality']
            score_id = packet['evidence_index'].get(item.get('score_id'))
            quality_id = packet['evidence_index'].get(item.get('quality_id'))
            extra_score = evidence.get((a, b, lead, SCORE))
            extra_quality = evidence.get((a, b, lead, QUALITY))
            if extra_score and extra_score['status'] == 'ok':
                score = [extra_score['result']['af_score'], extra_score['result']['afl_score']]
                score_id = extra_score['evidence_id']
            if extra_quality and extra_quality['status'] == 'ok' and extra_quality['result'].get('reasons') == []:
                quality, quality_id = 'no_warning', extra_quality['evidence_id']
            if score is not None and max(score) >= threshold and quality == 'no_warning':
                qualifying.extend((score_id, quality_id))
        if not qualifying:
            return []
        refs.extend(qualifying)
    return refs


def acquire(packet, actions, reader, ledger, *, budget=None):
    extra = {}
    windows = {}
    for key in actions:
        if budget is not None and (len(extra) >= budget or support_refs(packet, extra)):
            break
        a, b, lead, tool = key
        if (a, b) not in windows:
            windows[a, b] = reader.session(a, b, packet['cursor_s'], ledger.context)
        raw = windows[a, b].call(tool, {'start_s': a, 'end_s': b, 'lead': lead})
        extra[key] = ledger.append(raw, input_fingerprint=canonical_hash([ledger.context.manifest_id, *key]))
    ids = support_refs(packet, extra)
    return ids, extra


def evaluate(jobs, output, refs, validity, methods):
    rows = {(ch, method): [] for ch in ('ECG1', 'ECG2') for method in methods}
    categories = Counter()
    for channel, task, folder, packet_path, _, _ in jobs:
        key = task['task_id']
        reference = refs[key]
        valid = validity[key]['valid_intervals']
        clipped_af = intersection(reference['intervals'], valid)
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        _, unknown, _ = trace_data(jsonlines(folder / 'window_trace.jsonl'))
        updates = list(jsonlines(folder / 'event_updates.jsonl'))
        decisions = list(jsonlines(output / 'decisions' / packet_path.name))
        for item in decisions:
            a, b = item['gap_s']
            if union_seconds(intersection([(a, b)], valid)) < b - a - 1e-8:
                category = 'annotation_insufficient'
            elif any(c <= a and b <= d for c, d in reference['intervals']):
                category = 'same_reference_af'
            else:
                category = 'other_gap'
            categories[channel, category, 'candidates'] += 1
            for method in methods[1:]:
                categories[channel, category, method] += bool(item[method]['support_ids'])
            categories[channel, category, 'repairable_within_budget'] += item.get('within_budget', False)
        for method in methods:
            selected = {tuple(d['gap_s']) for d in decisions if method != 'A' and d[method]['support_ids']}
            pred = merge_gaps(result['intervals'], selected)
            row = evaluate_record(key, pred, reference, valid=valid, unknown=unknown,
                                  events=result['events'], updates=updates)
            added = intersection(selected, valid)
            true_s = union_seconds(intersection(added, clipped_af))
            row.update(record_id=task['record_id'], merges=len(selected), added_true_s=true_s,
                       added_non_af_s=union_seconds(added) - true_s,
                       extra_tool_calls=sum(d.get(method, {}).get('calls', 0) for d in decisions),
                       model_calls=sum(d.get(method, {}).get('forwards', 0) for d in decisions))
            rows[channel, method].append(row)
    summaries = []
    for (channel, method), values in rows.items():
        metrics = aggregate(values)
        for field in ('merges', 'added_true_s', 'added_non_af_s', 'extra_tool_calls', 'model_calls'):
            metrics[field] = sum(row[field] for row in values)
        summaries.append(dict(channel=channel, method=method, metrics=metrics, rows=values,
                              paired_vs_b0=paired_bootstrap(values, rows[channel, 'B0']) if method in ('B', 'D') else None))
    return summaries, [dict(channel=ch, category=cat, count=n, metric=metric)
                       for (ch, cat, metric), n in categories.items()]


def parity(jobs, raw_root, raw_cache, scores, predict, provenance, tolerance):
    rows = []
    for _, task, _, _, _, _ in jobs:
        name = task['record_id']
        if any(row['record_id'] == name for row in rows):
            continue
        reader = HistoricalWindow(raw_root, task, predict)
        cached = np.load(scores / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        seen = set()
        for win in jsonlines(raw_cache / f'{name}.windows.jsonl'):
            lead = win['channel']
            start, end = win['start_s'], win['end_s']
            if (lead in seen or win['status'] != 'ok' or start < task['start_s'] or end > task['end_s']):
                continue
            context = EvidenceContext('parity', task['task_id'], name, provenance)
            result = reader.session(start, end, end, context).call(SCORE, {'lead': lead, 'start_s': start, 'end_s': end})
            if result['status'] != 'ok':
                raise ValueError(f'Parity score unavailable: {name} {lead}: {result}')
            expected = cached[win['waveform_row']]
            actual = [result['result']['af_score'], result['result']['afl_score']]
            error = max(abs(float(x) - y) for x, y in zip(expected, actual))
            flip = any((max(expected) >= t) != (max(actual) >= t) for t in (.4, .6))
            rows.append(dict(record_id=name, lead=lead, window_s=[start, end], max_error=error,
                             threshold_flip=flip, within_tolerance=error <= tolerance))
            seen.add(lead)
            if len(seen) == 2:
                break
        if len(seen) != 2:
            raise ValueError(f'No complete parity window on both leads: {name}')
    return rows


def execute(jobs, output, raw_root, predict, manifest_id, method):
    for channel, task, _, packet_path, _, _ in jobs:
        name = task['record_id']
        forwards = 0

        def counted_predict(values):
            nonlocal forwards
            forwards += 1
            return predict(values)

        context = EvidenceContext(output.name + '-' + channel + '-' + method, task['task_id'], name, manifest_id)
        ledger = EvidenceLedger(context)
        reader = HistoricalWindow(raw_root, task, counted_predict)
        rows = []
        evidence = []
        previous = (list(jsonlines(output / 'decisions' / packet_path.name)) if method == 'B' else None)
        for index, packet in enumerate(jsonlines(packet_path)):
            ids = support_refs(packet, {})
            if bool(ids) != all(packet['gap_support']):
                raise ValueError('B0 differs from the frozen evidence rule')
            row = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'], B0=dict(support_ids=ids))
            if previous is not None:
                row = previous[index]
            began = perf_counter()
            before = forwards
            actions = action_plan(gap_windows(packet), packet['start_threshold'])
            selected, observed = (ids, {}) if ids else acquire(packet, actions, reader, ledger,
                                                              budget=2 if method == 'B' else None)
            row[method] = dict(support_ids=selected, calls=len(observed), forwards=forwards-before,
                               wall_s=perf_counter()-began, failures=sum(x['status'] != 'ok' for x in observed.values()),
                               evidence_ids=[x['evidence_id'] for x in observed.values()])
            if method == 'D' and selected and not ids:
                usable = [key for key, item in observed.items() if item['status'] == 'ok']
                row['within_budget'] = any(support_refs(packet, {key: observed[key] for key in keys})
                                           for size in (1, 2) for keys in combinations(usable, size))
            evidence.extend(observed.values())
            rows.append(row)
        write_jsonl(output / 'evidence' / f'{method}-{packet_path.name}', evidence)
        write_jsonl(output / 'decisions' / packet_path.name, rows)
        print(f'{method} {channel} {name}: {len(rows)} gaps, {forwards} forwards', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'raw-root', 'cache', 'scores', 'models', 'references', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--parity-tolerance', type=float, default=.001)
    args = parser.parse_args()
    started = perf_counter()
    tasks = read_jsonl(args.source / 'tasks.jsonl', 'task_id')
    if args.records:
        tasks = {key: task for key, task in tasks.items() if task['record_id'] in args.records}
        if {task['record_id'] for task in tasks.values()} != set(args.records):
            raise ValueError('Requested records missing from replay')
    if any({'answer', 'intervals', 'burden'} & task.keys() or task['source_database'] != 'afdb' for task in tasks.values()):
        raise ValueError('Expected unlabeled AFDB tasks')
    model_hash = digest(args.models / '1_lead_ECGFounder.pth')
    cache_hash = digest(args.cache / 'manifest.json')
    score_manifest = json.loads((args.scores / 'provenance.json').read_text())
    model_source = code_root() / 'third_party/ecgfounder'
    expected_scores = {'checkpoint_sha256': model_hash, 'cache_manifest_sha256': cache_hash,
                       'network_sha256': digest(model_source / 'net1d.py'),
                       'labels_sha256': digest(model_source / 'tasks.txt')}
    if any(score_manifest.get(key) != value for key, value in expected_scores.items()):
        raise ValueError('Cached model or preprocessing manifest mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    for child in ('packets', 'decisions', 'evidence'):
        (args.output / child).mkdir()
    jobs = list(collect(args.source, tasks, args.output, ('ECG1', 'ECG2')))
    provenance = dict(expected_scores, experiment_sha256=digest(Path(__file__)),
                      historical_window_sha256=digest(code_root() / 'src/cardioagent/ecg_agent/harness/historical_window.py'),
                      score_manifest_sha256=digest(args.scores / 'provenance.json'), source=str(args.source.resolve()),
                      source_sha256=digest(args.source / 'comparison.json'), device=args.device, budget=2,
                      packets_sha256={j[3].name: digest(j[3]) for j in jobs}, parity_tolerance=args.parity_tolerance,
                      raw_files={f'{name}.{ext}': digest(args.raw_root / f'{name}.{ext}')
                                 for name in sorted({task['record_id'] for task in tasks.values()}) for ext in ('hea', 'dat')})
    save_json(args.output / 'inputs.json', provenance)
    manifest_id = canonical_hash(provenance)
    print('Candidates:', sum(j[4] for j in jobs), 'B0 eligible:', sum(j[5] for j in jobs), flush=True)
    predict, _ = create_ecgfounder_scorer(args.models, args.device)
    checks = parity(jobs, args.raw_root, args.cache, args.scores, predict, manifest_id, args.parity_tolerance)
    save_json(args.output / 'parity.json', {'windows': checks})
    if any(not row['within_tolerance'] or row['threshold_flip'] for row in checks):
        raise ValueError('Original-grid parity check failed; see parity.json')
    execute(jobs, args.output, args.raw_root, predict, manifest_id, 'D')
    refs = read_jsonl(args.references, 'task_id')
    valid = json.loads((args.source / 'evaluation_validity.json').read_text(encoding='utf-8'))['records']
    methods = ('A', 'B0', 'D')
    summaries, categories = evaluate(jobs, args.output, refs, valid, methods)
    by = {(r['channel'], r['method']): r['metrics'] for r in summaries}
    # Exploratory gate: no extra non-AF coverage on either channel, some added true time.
    b0 = [by[ch, 'B0'] for ch in ('ECG1', 'ECG2')]
    full = [by[ch, 'D'] for ch in ('ECG1', 'ECG2')]
    gate = (all(d['added_non_af_s'] <= b['added_non_af_s'] + 1e-8 for b, d in zip(b0, full))
            and any(d['added_true_s'] > b['added_true_s'] for b, d in zip(b0, full)))
    if gate:
        execute(jobs, args.output, args.raw_root, predict, manifest_id, 'B')
        summaries, categories = evaluate(jobs, args.output, refs, valid, (*methods, 'B'))
    save_json(args.output / 'comparison.json', dict(results=summaries, categories=categories,
              full_measurement_gate_passed=gate, references_sha256=digest(args.references), wall_s=perf_counter()-started))
    lines = ['# AFDB original-window supplementary measurements', '',
             '| Lead | Method | F1 | Time recall | Non-AF minutes | Long fragments | Merges | Added true s vs A | Added non-AF s vs A | Calls | Model forwards |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        m = row['metrics']
        recall = 'NA' if m['time_recall'] is None else f"{m['time_recall']:.4f}"
        lines.append(f"| {row['channel']} | {row['method']} | {m['micro_episode_f1']:.4f} | {recall} | "
                     f"{m['non_af_alarm_minutes']:.3f} | {m['long_fragment_excess']} | {m['merges']} | "
                     f"{m['added_true_s']:.2f} | {m['added_non_af_s']:.2f} | {m['extra_tool_calls']} | {m['model_calls']} |")
    lines.extend(['', f'Full-measurement gate passed: {gate}', 'No LLM or shifted-window scoring was used.'])
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
