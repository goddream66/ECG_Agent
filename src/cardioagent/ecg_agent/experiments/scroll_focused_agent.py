"""Let Qwen request local ECG-Scroll diagnostic windows and commit supported episodes."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from time import perf_counter

from cardioagent.ecg_agent.experiments.full_reference_memory import BatchQwen, atomic_json
from cardioagent.ecg_agent.experiments.scroll_window_agent import make_chunks, encoded, sha
from cardioagent.ecg_agent.harness.staged_review import schema
from cardioagent.ecg_agent.llm.providers import parse_qwen_tool_reply
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents, merge_spans


ROUTE = (
    'Choose a measurement that can localize AF episodes before recording them. The candidate is a search '
    'range; it can contain several episodes separated by a gap. AF fraction describes the occupied part '
    'of a window, and a nonzero fraction can be a short real episode. Use available routes only.\n'
    'If short_1 is available for a small isolated region, use it to locate short episodes. '
    'With internal partial windows, prefer zoom_partial to find brief breaks; partial_5 is a cheaper '
    'option if a break is large enough to appear as an entire empty five-second window. '
    'If partial_1 is available, consider it when the partially occupied windows may hide very short spells or gaps. '
    'With only partial edges, use edges_5 for coarse localization or zoom_edges for a small episode '
    'whose boundaries need more precision. Zoom routes repeatedly split only mixed-label windows. '
    'Use coarse when no refinement is needed, for example, a fully occupied candidate or a few seconds '
    'of uncertainty across a long episode. The coarse_iou_upper_estimate is the occupied fraction of the '
    'candidate; low values signal that recording the whole candidate may not meet IoU 0.5. '
    'Call choose_review once.'
)
COMMIT = (
    'You are writing AF event intervals to an evidence ledger. The window labels are based on reference '
    'rhythm annotations. Use the highest resolution option supported by the returned measurements. '
    'The refined option preserves genuine separate runs when a measured gap separates them. '
    'A short supported interval is still an event. Choose drop only if there is no episode worth '
    'recording. Submit an available option using select_intervals. Do not invent boundary times.'
)


class FocusQwen(BatchQwen):
    def review(self, observations, stage):
        name, field, instruction = ('choose_review', 'route', ROUTE) if stage == 'route' else (
            'select_intervals', 'option', COMMIT)
        tokenizer = self.adapter.tokenizer

        def render(observation):
            allowed = {'type': 'string', 'enum': observation['available']}
            tools = [schema(name, 'Select exactly one listed value.', {field: allowed}, [field])]
            return tokenizer.apply_chat_template([
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': encoded(observation)},
            ], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        def decode(raw, observation):
            try:
                calls = parse_qwen_tool_reply(raw).get('tool_calls', [])
                fn, = calls
                args = json.loads(fn['function']['arguments'])
                if (fn['function']['name'] != name or set(args) != {field}
                        or args[field] not in observation['available']):
                    raise ValueError('Invalid or unavailable choice')
                return dict(raw=raw, choice=args[field])
            except (ValueError, KeyError, TypeError) as error:
                return dict(raw=raw, error=str(error))

        replies = [decode(raw, item) for raw,item in zip(
            self.forward([render(item) for item in observations]), observations, strict=True)]
        failed = [i for i,reply in enumerate(replies) if 'error' in reply]
        corrections = [dict(observations[i], correction='The previous call had an invalid choice. '
                            'Use one value from the tool enum.') for i in failed]
        if corrections:
            recovered = self.forward([render(item) for item in corrections])
            for i,raw in zip(failed, recovered, strict=True):
                replies[i] = dict(decode(raw, observations[i]), first_attempt=replies[i])
        return iter(replies)


@dataclass
class Region:
    preceding: dict | None
    positive: list
    following: dict | None

    @property
    def start(self):
        return self.positive[0]['t0']

    @property
    def end(self):
        return self.positive[-1]['t1']

    @property
    def cursor(self):
        return self.following['t1'] if self.following else self.end


def regions(chunks):
    preceding, positive = None, []
    for row in chunks:
        if row['label']['af_frac'] > 0:
            positive.append(row)
        else:
            if positive:
                yield Region(preceding, positive, row)
            preceding, positive = row, []
    if positive:
        yield Region(preceding, positive, None)


def partial(row):
    return row['label']['af_frac'] < 1


def estimated_time(row):
    return (row['t1']-row['t0'])*row['label']['af_frac']


def scan_summary(region):
    rows = region.positive
    candidates = [row for row in rows if partial(row)]
    options = ['coarse']
    if partial(rows[0]) or partial(rows[-1]):
        options.append('edges_5')
        options.append('zoom_edges')
    if any(partial(row) for row in rows[1:-1]):
        # Measured positive time on both sides of a partial interior window
        # leaves an unlocalized break. A single coarse span cannot express it.
        options = ['zoom_partial', 'partial_5']
        if len(candidates) <= 6:
            options.append('partial_1')
    expected = sum(estimated_time(row) for row in rows)
    if candidates and len(rows) <= 2 and expected < 15:
        options.append('short_1')
    upper_iou = expected / (region.end-region.start)
    # Any episode inside the coarse candidate is at most the observed occupied time.
    # Allow a margin for the three-decimal label quantization.
    if 'coarse' in options and len(options) > 1 and upper_iou < .49:
        options.remove('coarse')
    query_count = sum(partial(row) for row in (rows if len(rows) == 1 else (rows[0], rows[-1])))
    return dict(observed_through_s=region.cursor, candidate=[region.start, region.end],
                preceding_zero_s=None if region.preceding is None else region.preceding['t0'],
                following_zero_s=None if region.following is None else region.following['t0'],
                n_windows=len(rows), first_fraction=rows[0]['label']['af_frac'],
                last_fraction=rows[-1]['label']['af_frac'], n_partial=len(candidates),
                n_interior_partial=sum(partial(row) for row in rows[1:-1]),
                partial_windows=[[row['t0'], row['label']['af_frac']]
                                 for row in (candidates if len(candidates) <= 12 else candidates[:6]+candidates[-6:])],
                approximate_af_seconds=round(expected, 2),
                coarse_iou_upper_estimate=round(upper_iou, 3),
                measurement_cost={key: cost for key, cost in (
                    ('coarse', 0), ('edges_5', 6*query_count), ('partial_5', 6*len(candidates)),
                    ('partial_1', 30*len(candidates)), ('short_1', 30*len(candidates)),
                    ('zoom_edges', f'{2*query_count} initial queries, then split mixed windows'),
                    ('zoom_partial', f'{2*len(candidates)} initial queries, then split mixed windows')
                ) if key in options},
                available=options)


def refine(region, choice, label, task):
    expanded, measurements = [], []

    def measure(start, end):
        if end > region.cursor:
            raise ValueError('Requested window is ahead of the available cursor')
        item = dict(t0=start, t1=end, label=label(task['record'], task['pn_dir'], start, end))
        measurements.append(item)
        return item

    def zoom(row):
        fraction = row['label']['af_frac']
        start, end = row['t0'], row['t1']
        if not 0 < fraction < 1 or end-start <= .25:
            expanded.append(row)
            return
        midpoint = start+(end-start)/2
        zoom(measure(start, midpoint))
        zoom(measure(midpoint, end))

    for i, row in enumerate(region.positive):
        edge = i in (0, len(region.positive)-1)
        if not partial(row) or choice == 'coarse' or (choice in ('edges_5', 'zoom_edges') and not edge):
            expanded.append(row)
            continue
        if choice in ('zoom_edges', 'zoom_partial'):
            zoom(row)
            continue
        step = 1 if choice in ('short_1', 'partial_1') else 5
        for offset in range(0, 30, step):
            start = row['t0'] + offset
            end = min(start+step, row['t1'])
            if start >= end:
                break
            expanded.append(measure(start, end))
    spans = merge_spans((row['t0'], row['t1']) for row in expanded if row['label']['af_frac'] > 0)
    return spans, measurements


def publish(tracker, region, spans, method):
    updates = []
    cursor = region.cursor
    for a, b in spans:
        if not tracker.cursor <= a < b <= cursor:
            raise ValueError('Out-of-order interval publication')
        for action, time, tick in (('OPEN', a, a+(b-a)/2), ('CLOSE', b, b)):
            evidence = f'review:{region.start:g}:{method}'
            reply = tracker.apply(tracker.cursor, tick, dict(
                event_action=action, boundary=dict(kind='localized', time_s=time, evidence_id=evidence,
                                                  source='scroll_focused_window'),
                pending=None, next_review_s=None, support_intervals=[], opposing_intervals=[],
                evidence_ids=[evidence], reason='Qwen selected an observed interval option'))
            update = dict(reply['event_update'], recorded_at_s=cursor)
            tracker.last_change = deepcopy(update)
            field = 'confirmed_at_s' if action == 'OPEN' else 'end_decided_at_s'
            tracker.events[-1][field] = cursor
            if action == 'OPEN':
                tracker.events[-1]['first_seen_at_s'] = cursor
            updates.append(update)
    if tracker.cursor < cursor:
        tracker.skip(tracker.cursor, cursor)
    return updates


def decisions(model, items, stage, size):
    for i in range(0, len(items), size):
        yield from model.review(items[i:i+size], stage)


def preview_options(options):
    return {name: dict(interval_count=len(spans), total_s=round(sum(b-a for a,b in spans), 3),
                       examples=spans if len(spans)<=8 else spans[:4]+spans[-4:])
            for name,spans in options.items()}


def evaluate(model, task, label, score, folder, microbatch):
    chunks = make_chunks(task, label)
    candidates = list(regions(chunks))
    summaries = [scan_summary(region) for region in candidates]
    ambiguous = [s for s in summaries if len(s['available']) > 1]
    choices = iter(decisions(model, ambiguous, 'route', microbatch))
    routes = [next(choices) if len(s['available']) > 1 else dict(choice=s['available'][0], deterministic=True)
              for s in summaries]
    comparisons, commits = [], []
    for summary, route, region in zip(summaries, routes, candidates, strict=True):
        selected_route = route.get('choice', summary['available'][0])
        coarse = [[region.start, region.end]]
        refined, measurements = refine(region, selected_route, label, task)
        options = {'coarse': coarse} if 'coarse' in summary['available'] else {}
        if measurements and (refined != coarse or not options):
            options['refined'] = refined
        observation = dict(observed_through_s=region.cursor, scan=summary, options=options,
                           route=selected_route, measurements=len(measurements),
                           available=list(options)+['drop'])
        comparisons.append((observation, measurements))
        commits.append(dict(observation, options=preview_options(options)))
    chosen = decisions(model, commits, 'commit', microbatch)
    tracker = AutonomousEvents('fixed', task['t0'])
    counts = Counter(coarse_windows=len(chunks), candidate_regions=len(candidates), available_partial_windows=0)
    folder.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    log = folder/'qwen_calls.jsonl.gz'
    coarse_for_selected = []
    with gzip.open(log, 'wt', encoding='utf-8') as out:
        for region, route, (observation, measurements), final in zip(
                candidates, routes, comparisons, chosen, strict=True):
            counts['available_partial_windows'] += observation['scan']['n_partial']
            counts['route_'+observation['route']] += 1
            retries = int('first_attempt' in route) + int('first_attempt' in final)
            counts['model_calls'] += 1 + int(not route.get('deterministic')) + retries
            counts['retries'] += retries
            counts['deterministic_routes'] += int(bool(route.get('deterministic')))
            if not route.get('deterministic') and route.get('choice') not in (None, 'coarse'):
                counts['optional_refinement'] += int('coarse' in observation['scan']['available'])
            counts['coarse_pruned'] += int('coarse' not in observation['scan']['available'])
            counts['invalid_route'] += int('error' in route)
            counts['invalid_commit'] += int('error' in final)
            counts['extra_diagnostics'] += len(measurements)
            option = final.get('choice', 'drop')
            spans = observation['options'].get(option, [])
            if spans:
                coarse_for_selected.append([region.start, region.end])
            counts['selected_'+option] += 1
            updates = publish(tracker, region, spans, option)
            out.write(encoded(dict(observation=observation, prompt_options=preview_options(observation['options']),
                                   route=route, evidence=measurements,
                                   selection=final, event_updates=updates))+'\n')
    result = tracker.finish(task['t1'])
    intervals = [span for event in result for span in event['reported_intervals']]
    score_one = lambda spans: score(task['verifier'], answer(task, spans), task['answer'])
    return dict(id=task['id'], dataset=task['pn_dir'], record=task['record'], task=task['task'],
                prediction=answer(task, intervals), score=score_one(intervals),
                same_selected_coarse_score=score_one(coarse_for_selected), counts=dict(counts),
                logs_sha256=sha(log), seconds=perf_counter()-started, events=result)


def answer(task, spans):
    if task['task']=='episode_detection':
        return dict(intervals=spans)
    time = sum(b-a for a,b in merge_spans(spans))
    return dict(value=round(time/(task['t1']-task['t0']), 4), kind='af_burden')


def main():
    import argparse
    from ecgscroll.tools import classify_rhythm
    from ecgscroll.verify import score

    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('input', 'config', 'output'):
        parser.add_argument('--'+field, required=True, type=Path)
    parser.add_argument('--microbatch', type=int, default=20)
    args = parser.parse_args()
    tasks = json.loads(args.input.read_bytes())['cases']
    config = json.loads(args.config.read_bytes())
    prov = dict(input_sha256=sha(args.input), config_sha256=sha(args.config), code_sha256=sha(Path(__file__)))
    model = FocusQwen(config, args.microbatch, 'Scroll')
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        folder = args.output/task['id']
        receipt = folder/'result.json'
        if receipt.exists():
            previous = json.loads(receipt.read_bytes())
            if previous['provenance'] != prov or previous['logs_sha256'] != sha(folder/'qwen_calls.jsonl.gz'):
                raise ValueError('Input or log changed for '+task['id'])
            continue
        started = perf_counter()
        result = evaluate(model, task, classify_rhythm, score, folder, args.microbatch)
        result.update(provenance=prov, total_s=perf_counter()-started)
        atomic_json(receipt, result)
        print(task['id'], round(result['score'], 4), round(result['same_selected_coarse_score'], 4),
              result['counts'], flush=True)
    atomic_json(args.output/'complete.json', dict(tasks=[task['id'] for task in tasks], provenance=prov))


if __name__ == '__main__':
    main()
