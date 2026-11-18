"""Measure window-shift sensitivity and compare two budgeted review policies."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.evaluation.metrics import intersection
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.episode_review import collect
from cardioagent.ecg_agent.experiments.original_window_review import (
    QUALITY, SCORE, acquire, action_plan, evaluate, gap_windows, support_refs,
)
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.memory.evidence import EvidenceLedger, canonical_hash
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
from cardioagent.ecg_agent.schemas import EvidenceContext


BUDGET = 2
OFFSETS = (-5, 5)


def shift_options(packet, reader):
    for window in gap_windows(packet):
        a, b = window['window_s']
        for lead, item in window['leads'].items():
            if item['score'] is None:
                continue
            for offset in OFFSETS:
                cursor = packet['cursor_s']
                if (reader.task['start_s'] <= a + offset < b + offset <= cursor
                        and cursor - a - offset <= reader.retention_s):
                    yield (a, b, lead, offset), item['score']


def choose_probe(packet, reader, actions):
    """Probe a low score where one original measurement can still add support."""
    unresolved = [window for window, refs in zip(gap_windows(packet), packet['gap_support'], strict=True)
                  if not refs]
    if len(unresolved) != 1:
        return None
    window = unresolved[0]
    a, b = window['window_s']
    threshold = packet['start_threshold']
    followups = []
    for key in actions:
        if key[:2] != (a, b):
            continue
        item = window['leads'][key[2]]
        if ((key[3] == SCORE and item['quality'] == 'no_warning')
                or (key[3] == QUALITY and item['score'] is not None and max(item['score']) >= threshold)):
            followups.append(key)
    if not followups:
        return None
    gap_start, gap_end = packet['gap_s']
    probes = [(key, score) for key, score in shift_options(packet, reader)
              if key[:2] == (a, b) and max(score) < threshold]
    if not probes:
        return None

    def preference(probe):
        key, _ = probe
        offset = key[3]
        overlap = min(b + offset, gap_end) - max(a + offset, gap_start)
        return (-overlap, key[2], offset)

    return min(probes, key=preference)[0], followups


def review(packet, reader, ledger, *, enable_shift):
    ids = support_refs(packet, {})
    if ids:
        return ids, [], None, []
    actions = action_plan(gap_windows(packet), packet['start_threshold'])
    budget = BUDGET
    evidence = []
    probe_result = None
    probe = choose_probe(packet, reader, actions) if enable_shift else None
    if probe is not None:
        (a, b, lead, offset), followups = probe
        score = reader.shifted_score(a, b, packet['cursor_s'], lead, offset, ledger.context)
        probe_result = ledger.append(score, input_fingerprint=canonical_hash([
            ledger.context.manifest_id, 'shifted', a, b, lead, offset]))
        evidence.append(probe_result)
        budget -= 1
        if score['status'] == 'ok' and max(score['result']['af_score'], score['result']['afl_score']) >= packet['start_threshold']:
            actions = followups
    ids, original = acquire(packet, actions, reader, ledger, budget=budget)
    return ids, evidence + list(original.values()), probe_result, list(original)


def run_policies(jobs, args, predict, manifest_id):
    for channel, task, _, path, count, _ in jobs:
        if not count:
            for folder in ('decisions', 'sensitivity'):
                write_jsonl(args.output / folder / path.name, [])
            continue
        name = task['record_id']
        forwards = 0

        def counted_predict(values):
            nonlocal forwards
            forwards += 1
            return predict(values)

        reader = HistoricalWindow(args.raw_root, task, counted_predict)
        ledgers = {tag: EvidenceLedger(EvidenceContext(f'{args.output.name}-{channel}-{tag}',
                   task['task_id'], name, manifest_id)) for tag in ('B', 'B_shift', 'diagnostic')}
        evidence = {tag: [] for tag in ledgers}
        diagnostics = []
        decisions = []
        for packet in jsonlines(path):
            ids = support_refs(packet, {})
            if bool(ids) != all(packet['gap_support']):
                raise ValueError('B0 differs from frozen evidence rule')
            row = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'], B0={'support_ids': ids})
            for tag in ('B', 'B_shift'):
                before = forwards
                started = perf_counter()
                selected, measured, probe, steps = review(packet, reader, ledgers[tag], enable_shift=tag == 'B_shift')
                row[tag] = dict(support_ids=selected, calls=len(measured), forwards=forwards-before,
                                wall_s=perf_counter()-started, steps=steps,
                                probe_id=probe['evidence_id'] if probe else None,
                                evidence_ids=[item['evidence_id'] for item in measured])
                evidence[tag].extend(measured)
            decisions.append(row)
            for (a, b, lead, offset), old in shift_options(packet, reader):
                ledger = ledgers['diagnostic']
                observed = reader.shifted_score(a, b, packet['cursor_s'], lead, offset, ledger.context)
                item = ledger.append(observed, input_fingerprint=canonical_hash([
                    manifest_id, 'shifted', a, b, lead, offset]))
                evidence['diagnostic'].append(item)
                diagnostics.append(dict(gap_s=packet['gap_s'], window_s=[a, b], shifted_s=item['window_s'],
                                        cursor_s=packet['cursor_s'], lead=lead, offset_s=offset, original=old,
                                        threshold=packet['start_threshold'], evidence_id=item['evidence_id'],
                                        status=item['status'], shifted=[item['result']['af_score'],
                                        item['result']['afl_score']] if item['status'] == 'ok' else None))
        write_jsonl(args.output / 'decisions' / path.name, decisions)
        write_jsonl(args.output / 'sensitivity' / path.name, diagnostics)
        for tag, items in evidence.items():
            write_jsonl(args.output / 'evidence' / f'{tag}-{path.name}', items)
        print(f'{channel} {name}: {count} candidate gaps, {len(diagnostics)} diagnostic scores', flush=True)


def seconds(spans):
    return sum(b - a for a, b in spans)


def summarize(jobs, output, refs, validity):
    counts = Counter()
    for channel, task, _, path, _, _ in jobs:
        key = task['task_id']
        for row in jsonlines(output / 'sensitivity' / path.name):
            counts[channel, 'requested'] += 1
            if row['status'] != 'ok':
                counts[channel, 'unavailable'] += 1
                continue
            threshold = row['threshold']
            before, after = max(row['original']), max(row['shifted'])
            counts[channel, 'scored'] += 1
            counts[channel, 'score_delta_sum'] += after - before
            counts[channel, 'score_abs_delta_sum'] += abs(after - before)
            counts[channel, 'downward_flips'] += after < threshold <= before
            if before < threshold <= after:
                counts[channel, 'upward_flips'] += 1
                shifted = [row['shifted_s']]
                valid = validity[key]['valid_intervals']
                af = intersection(shifted, intersection(valid, refs[key]['intervals']))
                within_gap = intersection(shifted, [row['gap_s']])
                counts[channel, 'up_flips_overlapping_reference_af_outside_gap'] += (
                    seconds(af) - seconds(intersection(af, within_gap)) > 1e-8)
        for row in jsonlines(output / 'decisions' / path.name):
            counts[channel, 'policy_probes'] += row['B_shift']['probe_id'] is not None
            counts[channel, 'different_first_original_actions'] += bool(
                row['B']['steps'] and row['B_shift']['steps'] and row['B']['steps'][0] != row['B_shift']['steps'][0])
    return {ch: {metric: value for (lead, metric), value in counts.items() if lead == ch} for ch in ('ECG1', 'ECG2')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'raw-root', 'models', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--records', nargs='+')
    args = parser.parse_args()
    began = perf_counter()
    tasks = read_jsonl(args.source / 'tasks.jsonl', 'task_id')
    if args.records:
        tasks = {key: row for key, row in tasks.items() if row['record_id'] in args.records}
        if {row['record_id'] for row in tasks.values()} != set(args.records):
            raise ValueError('Requested records missing from replay')
    if any({'answer', 'intervals', 'burden'} & task.keys() or task['source_database'] != 'afdb' for task in tasks.values()):
        raise ValueError('Expected unlabeled AFDB tasks')
    args.output.mkdir(parents=True, exist_ok=False)
    for child in ('packets', 'decisions', 'evidence', 'sensitivity'):
        (args.output / child).mkdir()
    jobs = list(collect(args.source, tasks, args.output, ('ECG1', 'ECG2')))
    comparison = json.loads((args.source / 'comparison.json').read_text(encoding='utf-8'))
    expected = comparison['provenance']
    raw_files = {f'{task["record_id"]}.{ext}': digest(args.raw_root / f'{task["record_id"]}.{ext}')
                 for task in tasks.values() for ext in ('hea', 'dat')}
    model_files = {'checkpoint_sha256': digest(args.models / '1_lead_ECGFounder.pth'),
                   'network_sha256': digest(code_root() / 'third_party/ecgfounder/net1d.py'),
                   'labels_sha256': digest(code_root() / 'third_party/ecgfounder/tasks.txt')}
    if (any(expected.get(key) != value for key, value in model_files.items())
            or any(expected['records'][name.split('.')[0]][{'hea': 'header_sha256', 'dat': 'signal_sha256'}[name[-3:]]] != value
                   for name, value in raw_files.items())):
        raise ValueError('Model or waveform differs from the original replay')
    provenance = dict(model_files, budget=BUDGET, offsets=list(OFFSETS), raw_files=raw_files,
                      experiment_sha256=digest(Path(__file__)), source=str(args.source.resolve()),
                      source_sha256=digest(args.source / 'comparison.json'),
                      source_files={name: digest(code_root() / 'src/cardioagent/ecg_agent' / name) for name in (
                          'harness/historical_window.py', 'data/preprocessing.py', 'tools/ecgfounder.py',
                          'schemas.py', 'experiments/original_window_review.py')},
                      packets_sha256={job[3].name: digest(job[3]) for job in jobs})
    save_json(args.output / 'inputs.json', provenance)
    print(f'Candidates: {sum(job[4] for job in jobs)}, B0 eligible: {sum(job[5] for job in jobs)}', flush=True)
    predict, model_info = create_ecgfounder_scorer(args.models)
    run_policies(jobs, args, predict, canonical_hash(provenance))
    refs = read_jsonl(args.references, 'task_id')
    validity = json.loads((args.source / 'evaluation_validity.json').read_text(encoding='utf-8'))['records']
    summaries, categories = evaluate(jobs, args.output, refs, validity, ('A', 'B0', 'B', 'B_shift'))
    sensitivity = summarize(jobs, args.output, refs, validity)
    save_json(args.output / 'comparison.json', dict(results=summaries, categories=categories,
              sensitivity=sensitivity, model_info=model_info, references_sha256=digest(args.references),
              wall_s=perf_counter()-began))
    lines = ['# AFDB shifted-window review', '', 'A: original. B0: frozen evidence. B: original-window rule.',
             'B_shift: shift as a routing hint. Both budgeted policies have two total calls per gap.', '',
             '| Lead | Method | Event F1 | Merges | Added true s vs A | Added non-AF s vs A | Calls | Forwards |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        m = row['metrics']
        lines.append(f"| {row['channel']} | {row['method']} | {m['micro_episode_f1']:.4f} | {m['merges']} | "
                     f"{m['added_true_s']:.2f} | {m['added_non_af_s']:.2f} | {m['extra_tool_calls']} | {m['model_calls']} |")
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(sensitivity, flush=True)


if __name__ == '__main__':
    main()
