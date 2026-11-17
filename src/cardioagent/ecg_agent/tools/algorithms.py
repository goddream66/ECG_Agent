"""Window-local ECG tools. No annotations or other modalities are accepted."""
from time import perf_counter
import numpy as np
from cardioagent.signal.events import detect_events
from cardioagent.signal.quality import quality_features
from cardioagent.signal.synchronization import TimeAxis


def inspect(values, fs):
    began = perf_counter()
    quality = quality_features(values, fs, 'ecg')
    reasons = []
    if quality['nonfinite_fraction'] or quality['near_constant']:
        reasons.append('nonfinite_or_flat')
    for key, limit in [('longest_flat_run_s', .5), ('baseline_power_fraction', .70), ('high_frequency_power_fraction', .35)]:
        if quality.get(key) is not None and quality[key] > limit:
            reasons.append(key)
    band = quality.get('physiology_band_power_fraction') or 0.
    score = 0. if reasons else float(band)
    return {'signal_quality': quality, 'reasons': reasons, 'score': score,
            'duration_ms': (perf_counter()-began)*1000}


def measure(values, fs, start, commit_start, end):
    began = perf_counter()
    try:
        observation = detect_events('ecg', values, TimeAxis(fs, len(values), start_s=start), 0)
        peaks = [r['time_s'] for r in observation['events'] if commit_start <= r['time_s'] < end]
        rr = np.diff([r['time_s'] for r in observation['events']])
        median = float(np.median(rr)) if len(rr) else None
        rate = 60/median if median and median > 0 else None
        cv = float(np.std(rr)/np.mean(rr)) if len(rr)>1 and np.mean(rr)>0 else None
        # Irregularity is an observation, never a signal-quality rejection criterion.
        return {'r_peaks_s': peaks, 'rate_bpm': rate, 'rr_cv': cv,
                'rr_intervals_s': rr.tolist(), 'detector_status': observation['status'],
                'detector_error': None, 'duration_ms': (perf_counter()-began)*1000}
    except (ValueError, IndexError, FloatingPointError) as exc:
        return {'r_peaks_s': [], 'rate_bpm': None, 'rr_cv': None, 'rr_intervals_s': [],
                'detector_status': 'failed', 'detector_error': str(exc),
                'duration_ms': (perf_counter()-began)*1000}
