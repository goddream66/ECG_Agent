"""Window-local adapters for FeatureDB and ecg_signal_quality; no network access.

See third_party/physio_tools/source.json for pinned sources and adaptation notes.
These measurements are experimental and are not AF/AFL diagnoses.
"""
from math import ceil
import numpy as np
from scipy import signal

VERSION = '1.0.0'
QUALITY_VERSION = '1.1.0'

def template_correlation_passes(score):
    return bool(score is not None and np.isfinite(score) and score > .8)

WARNINGS = ['Experimental measurement; no independently validated diagnostic accuracy',
            'RR and morphology abnormalities must not be treated as noise automatically']


def _input(session, args):
    wave = session.context.waveform
    unit = wave.units[wave.channels.index(args.lead)].lower().strip().replace('μ', 'u').replace('µ', 'u')
    factors = {'mv': 1., 'uv': .001, 'v': 1000.}
    if unit not in factors:
        raise ValueError('FeatureDB and template quality require physical mV/uV/V units')
    x = np.asarray(session.window(args), dtype=float) * factors[unit]
    first = ceil((args.start_s-session.context.origin_s)*wave.fs-1e-8)
    origin = session.context.origin_s+first/wave.fs
    return x, wave.fs, origin


def _detect(x, fs):
    """FeatureDB simple_qrs_detector filter chain and NeuroKit peak detector.

    Resampling uses explicit sample times; returned coordinates are mapped to the
    original rate, rounded, deduplicated and checked. No HTTP variant is imported.
    """
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise RuntimeError('Install the ecg-tools extra (neurokit2==0.2.12)') from exc
    y = x.copy()
    if fs > 100:
        b, a = signal.iirnotch(50, 50, fs)
        y = signal.filtfilt(b, a, y)
    target_times = np.arange(0., (len(y)-1)/fs+1e-12, 1/500.)
    y = np.interp(target_times, np.arange(len(y))/fs, y)
    for band in ((.5, 45), (3, 40)):
        b, a = signal.butter(4, band, btype='bandpass', fs=500)
        y = signal.filtfilt(b, a, y)
    _, info = nk.ecg_peaks(y, sampling_rate=500)
    indices = np.unique(np.rint(np.asarray(info['ECG_R_Peaks'])*fs/500).astype(int))
    return indices[(indices >= 0) & (indices < len(x))]


def _peaks(session, args, x, fs, origin):
    key = ('featuredb_r_peaks_v1', args.lead, args.start_s, args.end_s)
    executed = key not in session.internal_cache
    if executed:
        session.internal_cache[key] = _detect(x, fs)
    indices = np.asarray(session.internal_cache[key], dtype=int)
    parent = next((c['evidence_id'] for c in session.calls
                   if c['status']=='ok' and c['lead']==args.lead
                   and c['window_s']==[args.start_s,args.end_s]
                   and c['tool'] in ('detect_beats_featuredb','extract_average_beat_featuredb','assess_template_quality')), None)
    return indices, {'r_peaks_samples': indices.tolist(),
                     'r_peaks_s': (origin+indices/fs).tolist(),
                     'r_peak_evidence_id': parent,
                     'detector_executed': executed,
                     'detector': 'FeatureDB adapted filters + NeuroKit2 0.2.12',
                     'internal_sample_rate_hz': 500,
                     'notch_50hz_applied': fs>100}


def _invalid(x):
    if not np.isfinite(x).all(): return 'missing_or_nonfinite_signal'
    if not len(x) or np.ptp(x)<=1e-10: return 'flat_signal'
    return None


def _failure(reason):
    return {'reason':reason}, 'insufficient', WARNINGS.copy()


def detect_beats(session, args):
    x, fs, origin = _input(session,args)
    reason = _invalid(x)
    if reason: return _failure(reason)
    indices, data = _peaks(session,args,x,fs,origin)
    rr = np.diff(indices)/fs*1000
    data.update(sample_rate_hz=fs, sample_origin_s=origin, rr_intervals_ms=rr.tolist(),
                rr_start_times_s=(origin+indices[:-1]/fs).tolist(),
                rr_valid_mask=[True]*len(rr), detected_beats=len(indices),
                mean_rr_ms=float(np.mean(rr)) if len(rr) else None,
                median_rr_ms=float(np.median(rr)) if len(rr) else None,
                rate_from_median_rr_bpm=float(60000/np.median(rr)) if len(rr) else None,
                rr_cv=float(np.std(rr)/np.mean(rr)) if len(rr)>1 else None)
    return data, 'ok' if len(rr)>=2 else 'insufficient', WARNINGS.copy()


def average_beat(session,args):
    x,fs,origin = _input(session,args)
    reason = _invalid(x)
    if reason:return _failure(reason)
    peaks,data = _peaks(session,args,x,fs,origin)
    if len(peaks)<3:return _failure('insufficient_r_peaks')
    # FeatureDB extract_avg_wave uses a median-RR-sized beat with 35% before R.
    period=int(np.median(np.diff(peaks)))
    ahead=int(round(.35*period));tail=int(round(.8*period))
    b,a=signal.butter(4,(.75,35),btype='bandpass',fs=fs)
    filtered=signal.filtfilt(b,a,x)
    valid=[int(p) for p in peaks[1:-1] if p-ahead>=0 and p+tail<=len(x)]
    data.update(used_r_peaks_samples=valid, used_beats=len(valid),
                discarded_beats=len(peaks)-len(valid), padding_used=False)
    if len(valid)<2:return data|{'reason':'insufficient_complete_beats'},'insufficient',WARNINGS.copy()
    mean=np.mean([filtered[p-ahead:p+tail] for p in valid],axis=0)
    data.update(average_beat_mv=mean.tolist(), sample_rate_hz=fs,
                r_index=ahead, relative_time_s=((np.arange(len(mean))-ahead)/fs).tolist(),
                median_rr_ms=float(period/fs*1000))
    return data,'ok',WARNINGS+['Averaging can hide beat-to-beat differences; preserve original waveform']


def template_quality(session,args):
    x,fs,origin = _input(session,args)
    reason = _invalid(x)
    if reason:
        return {'reason':reason,'template_correlation':None,
                'quality_label':'poor' if reason=='flat_signal' else 'unknown'},'insufficient',WARNINGS.copy()
    peaks,data = _peaks(session,args,x,fs,origin)
    rr=np.diff(peaks)
    # Upstream prescreen flags are reported separately, never a rhythm veto.
    data['prescreen_flags']={'fewer_than_six_peaks':len(peaks)<6,
        'peak_to_peak_over_5mv':bool(np.ptp(x)>5),
        'fewer_than_five_samples_over_01mv':bool(np.count_nonzero(abs(x)>.1)<5),
        'max_rr_over_three_mean_rr':bool(np.max(rr)>3*np.mean(rr)) if len(rr) else None}
    radius=int(round(.4*fs))
    valid=[int(p) for p in peaks if p-radius>=0 and p+radius<=len(x)]
    data.update(used_beats=len(valid), excluded_boundary_beats=len(peaks)-len(valid),
                template_correlation=None, quality_label='unknown')
    if len(valid)<3:return data|{'reason':'insufficient_complete_beats'},'insufficient',WARNINGS.copy()
    beats=np.asarray([x[p-radius:p+radius] for p in valid])
    template=np.mean(beats,axis=0);template-=np.mean(template)
    coefficients=[]
    for beat in beats:
        centered=beat-np.mean(beat)
        denom=np.linalg.norm(centered)*np.linalg.norm(template)
        coefficients.append(float(np.clip(np.dot(centered,template)/denom,-1,1)) if denom>1e-12 else None)
    finite=np.asarray([c for c in coefficients if c is not None])
    data['beat_correlations']=coefficients
    if len(finite)<3:return data|{'reason':'degenerate_template'},'insufficient',WARNINGS.copy()
    groups=0;remaining=finite.copy()
    # Historical order-sensitive group heuristic retained for audit only; never gates quality.
    while len(remaining):
        remaining=abs(remaining-remaining[0]);remaining=remaining[remaining>.1];groups+=1
    score=float(np.mean(finite))
    data.update(template_correlation=score, coefficient_group_count=groups,
                template_consistent=bool(template_correlation_passes(score)),
                quality_label='template_consistent' if template_correlation_passes(score) else 'template_inconsistent',
                heuristic_parameters={'correlation_threshold':.8,'group_count_gate_enabled':False})
    return data,'ok',WARNINGS+['Template correlation is not a calibrated probability or a global signal-quality diagnosis',
                             'Group count is audit-only and never gates quality; same R-peak detector as FeatureDB measurements']
