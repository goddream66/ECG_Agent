"""Rule-based AF/AFL review over synchronized, completed offline windows."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import wfdb

from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.common.intervals import validate_intervals
from cardioagent.ecg_agent.data.afdb import CHANNELS, SOURCE_FS, TARGET_FS, _arrivals


POLICIES = ('two-window', 'dual', 'adaptive')
BORDERLINE_MARGIN = 0.10
RECHECK_WINDOWS = 3








def replay_review(cache, record, task, channel, threshold, predict, emit, raw_root: Path, policy, budget):
    began = perf_counter()
    start_s, end_s = float(task['start_s']), float(task['end_s'])
    validate_intervals([], start_s, end_s)
    stem = raw_root / record['record_id']
    if policy == 'adaptive':
        header = wfdb.rdheader(str(stem))
        if header.fs != SOURCE_FS or tuple(header.sig_name) != CHANNELS or header.sig_len != record['source_record_samples']:
            raise ValueError('Raw AFDB header does not match cached window coordinates')
    secondary = next(lead for lead in CHANNELS if lead != channel)
    memory = EventMemory()
    counts, totals = Counter(), Counter()
    scored_seconds = 0.0
    review_seconds = 0.0
    cursor = start_s
    for block in _arrivals(cache, record, CHANNELS, start_s, end_s):
        row, _ = block[channel]
        start, cursor = row['start_sample'] / SOURCE_FS, row['end_sample'] / SOURCE_FS
        counts[row['status']] += 1
        if row['status'] == 'incomplete':
            observation = memory.observe(start, cursor, 'unavailable', False, [])
            emit({'task_id': task['task_id'], 'start_s': start, 'end_s': cursor,
                  'cursor_s': cursor, 'status': 'incomplete', 'decision': 'unavailable',
                  'reason': 'incomplete_window', 'tool_calls': [], 'memory': observation})
            review_seconds += cursor - start
            continue
        if policy == 'adaptive':
            raw = wfdb.rdrecord(str(stem), sampfrom=row['start_sample'], sampto=row['end_sample'])
            if raw.p_signal is None or raw.p_signal.shape != (row['end_sample'] - row['start_sample'], len(CHANNELS)):
                raise ValueError('Unexpected raw AFDB window shape')
            waveform = Waveform(raw.p_signal, SOURCE_FS, CHANNELS, tuple(raw.units))
        else:
            samples = np.full((10 * TARGET_FS, len(CHANNELS)), np.nan, dtype=np.float32)
            for i, lead in enumerate(CHANNELS):
                if block[lead][1] is not None:
                    samples[:, i] = block[lead][1]
            waveform = Waveform(samples, TARGET_FS, CHANNELS, ('normalized',) * len(CHANNELS))

        def cached_model(lead, a, b):
            if (a, b) != (start, cursor):
                return None, predict
            return block[lead][1], predict

        session = ToolSession(ToolContext(waveform, start, cursor), max_calls=1 + budget,
                              cached_model=cached_model)

        def call(tool, lead):
            return session.call(tool, {'lead': lead, 'start_s': start, 'end_s': cursor})

        primary = call('score_ecgfounder', channel)
        decision, checked, reason = decide(call, primary, secondary, threshold, memory, policy, budget)
        ids = [item['evidence_id'] for item in session.calls]
        observation = memory.observe(start, cursor, decision, checked, ids)
        scored_seconds += cursor - start if primary['status'] == 'ok' else 0
        review_seconds += cursor - start if decision in ('conflict', 'unavailable') else 0
        totals['tool_calls'] += len(session.calls)
        totals['review_calls'] += len(session.calls) - 1
        totals['model_calls'] += sum(item['tool'] == 'score_ecgfounder' and item['status'] == 'ok'
                                     and not item['cached'] for item in session.calls)
        totals['tool_ms'] += sum(item['duration_ms'] for item in session.calls)
        emit({'task_id': task['task_id'], 'channel': channel, 'start_s': start, 'end_s': cursor,
              'cursor_s': cursor, 'status': primary['status'], 'decision': decision, 'reason': reason,
              'af_score': primary['result'].get('af_score'), 'afl_score': primary['result'].get('afl_score'),
              'tool_calls': session.calls, 'memory': observation})
    memory.close('stream_ended' if memory.open_interval else 'unresolved_at_end', cursor)
    return {'task_id': task['task_id'], 'intervals': validate_intervals(memory.intervals, start_s, end_s),
            'scored_seconds': scored_seconds, 'unscored_seconds': end_s - start_s - scored_seconds,
            'window_status_counts': dict(counts), 'events': memory.events, 'review_seconds': review_seconds,
            'elapsed_ms': (perf_counter() - began) * 1000,
            **{key: totals[key] for key in ('tool_calls', 'model_calls', 'review_calls', 'tool_ms')}}

from cardioagent.ecg_agent.policies.baselines import EventMemory, decide, score_label
