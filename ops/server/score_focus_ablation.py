"""Offline scoring of a completed segment using independent AFDB reference intervals."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from cardioagent.ecg_agent.evaluation.metrics import greedy_pairs, intersection, ratio, union_seconds


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def score(folder, validity):
    result = json.loads((folder / 'result.json').read_text())
    ref = validity['records'][result['task_id']]
    target = ref['raw_annotation_af_intervals']
    windows = read_rows(folder / 'window_trace.jsonl')
    evidence = read_rows(folder / 'evidence.jsonl')
    steps = read_rows(folder / 'llm_steps.jsonl')
    measured = defaultdict(set)
    for item in evidence:
        if item['status'] == 'ok' and item['window_s']:
            measured[item['available_at_s']].add(tuple(item['window_s']))
    acquisition_failures = {s['cursor_s'] for s in steps if s.get('status') == 'ACQUISITION_FAILURE'}
    confusion, outcomes, missed = Counter(), Counter(), Counter()
    for row in windows:
        a, b = row['start_s'], row['end_s']
        action_error = row['action'] == 'ACTION_ERROR'
        outcome = 'ACTION_ERROR' if action_error else row['interpretation']
        outcomes[outcome] += 1
        if union_seconds(intersection([[a, b]], ref['valid_intervals'])) < b - a - 1e-7:
            continue
        positive = union_seconds(intersection([[a, b]], target)) >= (b-a)/2
        support = outcome == 'support'
        if positive:
            confusion['tp' if support else 'fn'] += 1
            if not support:
                if action_error or b in acquisition_failures:
                    category = 'execution_failure_observed'
                elif any(x < b and a < y <= b for x, y in measured[b]):
                    category = 'no_support_despite_current_measurement'
                else:
                    category = 'no_successful_current_measurement'
                missed[category] += 1
        elif support:
            confusion['fp'] += 1
        else:
            confusion['tn_negative' if outcome == 'negative' else 'non_target_without_decision'] += 1
    calls = parsed = input_tokens = output_tokens = usage_count = 0
    with gzip.open(folder / 'qwen_calls.jsonl.gz', 'rt') as stream:
        for line in stream:
            call = json.loads(line)
            calls += 1
            parsed += int(len((call.get('message') or {}).get('tool_calls', [])) == 1)
            usage = call.get('usage') or {}
            if 'prompt_tokens' in usage and 'completion_tokens' in usage:
                usage_count += 1
                input_tokens += usage['prompt_tokens']
                output_tokens += usage['completion_tokens']
    pred = result['intervals']
    pairs = greedy_pairs(pred, target)
    tp, fp, fn = (confusion[k] for k in ('tp', 'fp', 'fn'))
    n = len(windows)
    return dict(windows=n, reference=target, predictions=pred, window_outcomes=outcomes,
                window_target='at least 50% reference overlap in fully annotated window',
                confusion=confusion, missed=missed, precision=ratio(tp,tp+fp),
                recall=ratio(tp,tp+fn), window_f1=ratio(2*tp,2*tp+fp+fn),
                event_f1=ratio(2*len(pairs),len(pred)+len(target)), matched_pairs=pairs,
                matched_boundary_errors_s=[[pred[i][0]-target[j][0],pred[i][1]-target[j][1]] for i,j in pairs],
                measurement_results=len(evidence), measurement_success=ratio(sum(r['status']=='ok' for r in evidence), len(evidence)),
                calls=calls, parsed=parsed, format_success=ratio(parsed,calls), usage_count=usage_count,
                calls_per_window=ratio(calls,n), recorded_input_tokens_per_window=ratio(input_tokens,n),
                recorded_output_tokens_per_window=ratio(output_tokens,n),
                seconds_per_window=ratio(result['elapsed_ms']/1000,n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('validity', type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.folder,json.loads(args.validity.read_text())), indent=2))
