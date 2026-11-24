"""Experimental two-second waveform measurements within the last arrived ten seconds."""
import numpy as np
from wfdb.processing import xqrs_detect


FEATURES = ('rr_ratio', 'rr_deviation', 'morphology_distance', 'qrs_amplitude_ratio',
            'qrs_count', 'observed_age_ratio')


def measure_lead(wave, fs):
    """Use 20 s of prior context and 10 s of target. Return five retrospective cells."""
    if type(fs) is not int or fs < 50 or np.shape(wave) != (30*fs,):
        raise ValueError('Expected a 30 s arrived lead at known integer sampling rate')
    if not np.isfinite(wave).all() or np.ptp(wave)==0:
        return [None]*5
    peaks = np.asarray(xqrs_detect(wave, fs, verbose=False), dtype=int)
    left, right = round(.08*fs), round(.12*fs)
    peaks = peaks[(peaks >= left) & (peaks + right <= len(wave))]
    past = peaks[peaks + right <= 20*fs]
    if len(past) < 5:
        return [None]*5
    def segments(pp):
        x = np.array([wave[k-left:k+right] for k in pp])
        return x - np.median(x, axis=1, keepdims=True)
    template = np.median(segments(past), axis=0)
    amp = np.ptp(template)
    typical_rr = np.median(np.diff(past))
    if amp < 1e-8 or typical_rr<=0:
        return [None]*5
    rows = []
    for start in range(20*fs, 30*fs, 2*fs):
        end = start + 2*fs
        inside = peaks[(peaks>=start)&(peaks<end)]
        before = peaks[peaks<end]
        intervals = np.diff(peaks)[(peaks[1:]>=start)&(peaks[1:]<end)] / typical_rr
        waveforms = segments(inside) if len(inside) else None
        dist = np.sqrt(np.mean((waveforms-template)**2, axis=1))/amp if len(inside) else None
        rows.append([float(np.median(intervals)) if len(intervals) else None,
                     float(np.max(np.abs(intervals-1))) if len(intervals) else None,
                     float(np.max(dist)) if len(inside) else None,
                     float(np.median(np.ptp(waveforms,axis=1))/amp) if len(inside) else None,
                     int(len(inside)), float((end-before[-1])/typical_rr) if len(before) else None])
    return rows
