"""Window-local XQRS evidence inspired by ECG-Scroll's rhythm measurement.

Unlike ECG-Scroll's whole-record peak cache, this detector receives only the
elapsed slice supplied by ToolSession. Its results are experimental until
checked against independent beat annotations in the target recordings.
"""

import numpy as np
from wfdb import processing

from cardioagent.ecg_agent.tools.measurements import _stats, rr_clean_mask


VERSION = '0.1.0'


def detect_beats_xqrs_proxy(session, args):
    x = np.asarray(session.window(args), dtype=np.float64)
    fs = float(session.context.waveform.fs)
    start, end = session.sampled_window(args)
    base = {'detector': 'WFDB XQRS', 'sampled_window_s': [start, end],
            'r_peaks_s': [], 'rr_intervals_ms': [], 'rr_start_times_s': [],
            'cleaning_keep_mask': [], 'raw': _stats([]), 'cleaned': _stats([])}
    warnings = [
        'Experimental local XQRS: beat positions require independent annotation validation',
        'Short windows and their boundaries may affect detection; RR irregularity is not an AF diagnosis',
        'Cleaned RR may remove true abnormal intervals; original RR and mask are retained',
    ]
    if not np.isfinite(x).all() or np.ptp(x) <= 1e-10:
        return {**base, 'reason': 'missing_or_flat_signal'}, 'insufficient', warnings
    try:
        detector = processing.XQRS(sig=x, fs=fs)
        detector.detect(verbose=False)
        indices = np.asarray(detector.qrs_inds, dtype=np.int64)
    except (ValueError, ArithmeticError, RuntimeError, IndexError) as exc:
        return {**base, 'detector_error': str(exc)}, 'error', warnings
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(x)) or np.any(np.diff(indices) <= 0):
        return {**base, 'detector_error': 'invalid_peak_indices'}, 'error', warnings
    times = start + indices / fs
    rr_ms = np.diff(times) * 1000
    keep = rr_clean_mask(rr_ms)
    payload = {**base, 'r_peaks_s': times.tolist(),
               'rr_intervals_ms': rr_ms.tolist(),
               'rr_start_times_s': times[:-1].tolist(),
               'cleaning_keep_mask': keep.tolist(),
               'raw': _stats(rr_ms), 'cleaned': _stats(rr_ms[keep]),
               'detected_beats': int(len(indices)),
               'removed_fraction': float(np.mean(~keep)) if len(keep) else None}
    return payload, ('ok' if len(rr_ms) >= 3 else 'insufficient'), warnings
