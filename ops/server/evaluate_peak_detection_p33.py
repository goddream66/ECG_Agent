"""P3-3: causal, paired beat-detector audit on AFDB.

Inference receives waveform only. Annotation access is confined to this offline
evaluation script and its separate metrics output; do not pass metrics to an agent.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import wfdb

from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext, ToolSession


METHODS = {
    'original_10s': ('detect_beats', 10),
    'featuredb_10s': ('detect_beats_featuredb', 10),
    'xqrs_10s': ('detect_beats_xqrs_proxy', 10),
    'xqrs_30s': ('detect_beats_xqrs_proxy', 30),
    'xqrs_60s': ('detect_beats_xqrs_proxy', 60),
}
TOLERANCE_S = .15


def match_peaks(reference, predicted, tolerance=TOLERANCE_S):
    """Sorted one-to-one matching with explicit pairs for RR error accounting."""
    i = j = 0
    pairs = []
    while i < len(reference) and j < len(predicted):
        difference = predicted[j] - reference[i]
        if abs(difference) <= tolerance:
            pairs.append((i, j, abs(difference)))
            i += 1
            j += 1
        elif difference < 0:
            j += 1
        else:
            i += 1
    rr_errors = [abs((predicted[b] - predicted[a]) - (reference[d] - reference[c]))
                 for (c, a, _), (d, b, _) in zip(pairs[:-1], pairs[1:])
                 if d == c + 1 and b == a + 1]
    return {'tp': len(pairs), 'fp': len(predicted) - len(pairs),
            'fn': len(reference) - len(pairs),
            'localization_errors_ms': [1000 * p[2] for p in pairs],
            'rr_errors_ms': [1000 * e for e in rr_errors]}


def windows(record, duration_s):
    if record == '04936':
        return [1780., 1790., 7910., 7920.]
    # Fixed, label-independent sampling for the corrected-annotation records.
    return sorted({float(10 * round(t / 10)) for t in np.linspace(70, duration_s - 20, 12)})


def measure(raw_root, record, target_start, lead, name, span):
    header = wfdb.rdheader(str(raw_root / record))
    fs = float(header.fs)
    origin = target_start + 10 - span
    if origin < 0:
        raise ValueError('No elapsed context for requested span')
    first, last = round(origin * fs), round((target_start + 10) * fs)
    waveform = wfdb.rdrecord(str(raw_root / record), sampfrom=first, sampto=last)
    if len(waveform.p_signal) != last - first:
        raise ValueError('Incomplete waveform')
    wave = Waveform(waveform.p_signal, fs, tuple(waveform.sig_name), tuple(waveform.units))
    session = ToolSession(ToolContext(wave, origin, target_start + 10), allow_experimental=True)
    started = perf_counter()
    tool = session.call(name, {'lead': lead, 'start_s': origin, 'end_s': target_start + 10})
    elapsed_ms = 1000 * (perf_counter() - started)
    result = tool.get('result') or {}
    predicted = sorted(float(t) for t in result.get('r_peaks_s', [])
                       if target_start <= t < target_start + 10)
    return {'record_id': record, 'lead': lead, 'target_window_s': [target_start, target_start + 10],
            'measurement_window_s': [origin, target_start + 10], 'method': next(
                label for label, spec in METHODS.items() if spec == (name, span)),
            'status': tool['status'], 'error': tool['error'], 'peak_times_s': predicted,
            'all_detected_beats': len(result.get('r_peaks_s', [])),
            'raw_rr_ms': result.get('rr_intervals_ms',
                                    [1000 * x for x in result.get('rr_intervals_s', [])]),
            'rr_keep_mask': result.get('cleaning_keep_mask'), 'tool_wall_ms': elapsed_ms,
            'available_at_s': tool['available_at_s']}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['annotation_kind'], row['method'])].append(row)
        groups[(row['annotation_kind'], row['record_id'], row['lead'], row['method'])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        tp = sum(x['tp'] for x in group)
        fp = sum(x['fp'] for x in group)
        fn = sum(x['fn'] for x in group)
        loc = [e for x in group for e in x['localization_errors_ms']]
        rr = [e for x in group for e in x['rr_errors_ms']]
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        output.append({'group': key, 'windows': len(group), 'tp': tp, 'fp': fp, 'fn': fn,
                       'precision': tp / (tp + fp) if tp + fp else None,
                       'recall': tp / (tp + fn) if tp + fn else None, 'f1': f1,
                       'median_localization_ms': median(loc) if loc else None,
                       'p95_localization_ms': float(np.quantile(loc, .95)) if loc else None,
                       'rr_pairs': len(rr), 'median_rr_error_ms': median(rr) if rr else None,
                       'p95_rr_error_ms': float(np.quantile(rr, .95)) if rr else None,
                       'median_tool_wall_ms': median(x['tool_wall_ms'] for x in group),
                       'statuses': dict(Counter(x['status'] for x in group))})
    return output


def run(raw_root, output, records):
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    metrics = []
    manifest = {'source': 'PhysioNet AFDB 1.0.0', 'tolerance_s': TOLERANCE_S,
                'methods': METHODS, 'records': {}}
    for record in records:
        header = wfdb.rdheader(str(raw_root / record))
        annotation_kind = 'qrs' if record == '04936' else 'qrsc'
        if not (raw_root / f'{record}.{annotation_kind}').is_file():
            raise FileNotFoundError(f'{record}.{annotation_kind}')
        annotated = wfdb.rdann(str(raw_root / record), annotation_kind).sample / header.fs
        manifest['records'][record] = {'annotation_kind': annotation_kind,
            'annotation_is_manually_corrected': annotation_kind == 'qrsc',
            'files_sha256': {ext: sha256((raw_root / f'{record}.{ext}').read_bytes()).hexdigest()
                             for ext in ('hea', 'dat', annotation_kind)}}
        for target_start in windows(record, header.sig_len / header.fs):
            reference = annotated[(annotated >= target_start) & (annotated < target_start + 10)]
            for lead in header.sig_name:
                for method, (tool, span) in METHODS.items():
                    row = measure(raw_root, record, target_start, lead, tool, span)
                    row['method'] = method
                    measurements.append(row)
                    metrics.append({k: row[k] for k in ('record_id', 'lead', 'target_window_s',
                                                        'method', 'status', 'tool_wall_ms')} | {
                        'annotation_kind': annotation_kind,
                        **match_peaks(reference, row['peak_times_s'])})
    for name, rows in (('measurements.jsonl', measurements), ('metrics.jsonl', metrics)):
        (output / name).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows),
                                   encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summarize(metrics), indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'measurement_rows': len(measurements), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', nargs='+', default=['04936', '05091', '07859'])
    args = parser.parse_args()
    run(args.raw_root, args.output, args.records)


if __name__ == '__main__':
    main()
