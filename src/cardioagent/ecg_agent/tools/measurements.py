"""ECG measurements adapted from ECG-Scroll; see third_party/licenses/ECG-Scroll-MIT.txt.

The original whole-record cache and ground-truth tools are intentionally absent.
Only elapsed window-local peaks are used. Proxies are explicitly experimental.
"""
import numpy as np
from cardioagent.ecg_agent.schemas import BeatArgs


def _beats(session,args):
    key=(args.lead,args.start_s,args.end_s)
    if key not in session.internal_cache:
        import cardioagent.ecg_agent.tools.algorithms as legacy
        start, end = session.sampled_window(args)
        session.internal_cache[key]=legacy.measure(session.window(args),session.context.waveform.fs,
                                                   start,start,end)
    return session.internal_cache[key]


def quality(session,args):
    import cardioagent.ecg_agent.tools.algorithms as legacy
    data=legacy.inspect(session.window(args),session.context.waveform.fs)
    return data, 'ok', ['Signal-quality score is heuristic, not a probability']


def beats(session,args):
    # Preserve existing per-chunk rate semantics, including bounded lookback RR.
    import cardioagent.ecg_agent.tools.algorithms as legacy
    start, end = session.sampled_window(args)
    data=legacy.measure(session.window(args),session.context.waveform.fs,start,
                        args.commit_start_s if args.commit_start_s is not None else start,end)
    return data, ('error' if data['detector_error'] else 'ok' if data['rate_bpm'] is not None else 'insufficient'), []


def rr_clean_mask(rr_ms):
    """ECG-Scroll local cleaning rule, exposed as a mask; original intervals survive."""
    rr=np.asarray(rr_ms,dtype=float)
    mask=(rr>=300)&(rr<=2000)
    indices=np.flatnonzero(mask); clean=rr[mask]
    if len(clean)>=4:
        for i,index in enumerate(indices):
            neighbors=np.r_[clean[max(0,i-2):i],clean[i+1:min(len(clean),i+3)]]
            local=float(np.median(neighbors))
            if clean[i]>1.8*local or clean[i]<.5*local: mask[index]=False
    return mask


def _stats(rr):
    return {'mean_ms':float(np.mean(rr)) if len(rr) else None,
            'sd_ms':float(np.std(rr)) if len(rr)>1 else None,
            'cv':float(np.std(rr)/np.mean(rr)) if len(rr)>1 and np.mean(rr)>0 else None,
            'rate_from_mean_bpm':float(60000/np.mean(rr)) if len(rr) and np.mean(rr)>0 else None}


def rhythm(session,args):
    measured=_beats(session,args)
    rr=np.array(measured['rr_intervals_s'])*1000
    mask=rr_clean_mask(rr)
    return {'r_peaks_s':measured['r_peaks_s'],'rr_intervals_ms':rr.tolist(),
            'rr_start_times_s':measured['r_peaks_s'][:-1],
            'cleaning_keep_mask':mask.tolist(),'raw':_stats(rr),'cleaned':_stats(rr[mask]),
            'removed_fraction':float(np.mean(~mask)) if len(rr) else None}, ('ok' if len(rr)>=3 else 'insufficient'), [
                'Cleaned RR can remove true abnormal intervals; use raw RR for event assessment',
                'RR irregularity alone is not a rhythm diagnosis']


def profile(session,args):
    measured=_beats(session,args)
    times=np.array(measured['r_peaks_s'])
    count=min(args.bins,max(1,int((args.end_s-args.start_s)//5)))
    edges=np.linspace(args.start_s,args.end_s,count+1)
    bins=[]
    for a,b in zip(edges[:-1],edges[1:]):
        rr=np.diff(times[(times>=a)&(times<b)])*1000
        bins.append({'window_s':[float(a),float(b)],'rr_cv':_stats(rr)['cv'] if len(rr)>=3 else None,
                     'rr_intervals':len(rr)})
    valid=[i for i in bins if i['rr_cv'] is not None]
    peak=max(valid,key=lambda i:i['rr_cv']) if valid else None
    return {'bins':bins,'peak':peak}, ('ok' if valid else 'insufficient'), [
        'A coarse profile is not free: its cost includes any underlying beat detection',
        'Missing sub-bin measurements are null, not zero']


def _mv(session,args):
    wave=session.context.waveform
    unit=wave.units[wave.channels.index(args.lead)].strip().lower().replace('μ','u').replace('µ','u')
    factors={'mv':1.,'uv':.001,'v':1000.}
    if unit not in factors: raise ValueError('Morphology proxies require physical mV/uV/V units')
    values=session.window(args)*factors[unit]
    if not np.isfinite(values).all() or np.ptp(values)<=1e-10:
        raise ValueError('Morphology requires finite, non-flat ECG')
    times=np.array(_beats(session,args)['r_peaks_s'])
    start, _ = session.sampled_window(args)
    indices=np.rint((times-start)*wave.fs).astype(int)
    return values,wave.fs,times,indices


def _output(values,summary,assumptions):
    return values|summary, ('ok' if values.get('measured_beats',0) else 'insufficient'), [
        'Experimental waveform proxy; not a validated clinical interval or diagnosis',*assumptions]


def p_wave(session,args):
    x,fs,times,indices=_mv(session,args); rows=[]
    start, _ = session.sampled_window(args)
    lo,hi,base=max(1,round(.25*fs)),max(1,round(.05*fs)),max(1,round(.02*fs))
    for time,r in zip(times,indices):
        a,b=r-lo,r-hi
        if a<base or b<=a:continue
        dev=x[a:b]-np.median(x[a-base:a]); j=int(np.argmax(abs(dev)))
        rows.append({'r_time_s':float(time),'p_candidate_time_s':float(start+(a+j)/fs),
                     'p_peak_to_r_peak_ms':float((r-a-j)/fs*1000),'amplitude_mv':float(dev[j])})
    return _output({'beats':rows,'measured_beats':len(rows)},
                   {'p_candidate_fraction':float(np.mean([abs(r['amplitude_mv'])>.05 for r in rows])) if rows else None},
                   ['Fixed search 250..50 ms before R; peak-to-peak distance is not PR interval'])


def qrs(session,args):
    x,fs,times,indices=_mv(session,args); amplitude=abs(x-np.median(x))
    threshold=.3*np.percentile(amplitude,99); radius=max(1,round(.12*fs)); rows=[]
    for time,r in zip(times,indices):
        if r-radius<0 or r+radius>len(x):continue
        value=float(np.sum(amplitude[r-radius:r+radius]>threshold)/fs*1000)
        rows.append({'r_time_s':float(time),'suprathreshold_occupancy_ms':value})
    return _output({'beats':rows,'measured_beats':len(rows)},
                   {'mean_occupancy_ms':float(np.mean([r['suprathreshold_occupancy_ms'] for r in rows])) if rows else None},
                   ['Threshold occupancy is not QRS onset-to-offset duration; no PVC classification'])


def beat_timeline(session, args):
    """Join ECG-Scroll-style local RR, P and QRS evidence on detected R timestamps."""
    measured = _beats(session, args)
    times = measured['r_peaks_s']
    rows = []
    details = {'sampled_window_s': session.sampled_window(args), 'detected_beats': len(times), 'beats': rows}
    if measured['detector_error']:
        return {**details, 'detector_error': measured['detector_error']}, 'error', []
    warnings = [
        'Experimental waveform proxy; use surrounding beats to review a possible rhythm transition',
        'Atrial deflection candidates and QRS occupancy do not establish rhythm type or an AF end',
    ]
    if len(times) < 2:
        return details, 'insufficient', warnings

    p_by_r = {row['r_time_s']: row for row in p_wave(session, args)[0]['beats']}
    qrs_by_r = {row['r_time_s']: row for row in qrs(session, args)[0]['beats']}
    for index, time in enumerate(times):
        p = p_by_r.get(time)
        qrs_sample = qrs_by_r.get(time)
        rows.append({
            'r_time_s': time,
            'rr_from_previous_ms': None if index == 0 else (time - times[index - 1]) * 1000,
            'p_candidate_time_s': p['p_candidate_time_s'] if p else None,
            'p_candidate_amplitude_mv': p['amplitude_mv'] if p else None,
            'qrs_suprathreshold_occupancy_ms': qrs_sample['suprathreshold_occupancy_ms'] if qrs_sample else None,
        })
    return details, 'ok', warnings


def st(session,args):
    x,fs,times,indices=_mv(session,args); before=round(.04*fs); after=round(.12*fs); rows=[]
    for time,r in zip(times,indices):
        if r-before<0 or r+after>=len(x):continue
        rows.append({'r_time_s':float(time),'r120_minus_rminus40_mv':float(x[r+after]-x[r-before])})
    return _output({'beats':rows,'measured_beats':len(rows)},
                   {'mean_deviation_mv':float(np.mean([r['r120_minus_rminus40_mv'] for r in rows])) if rows else None},
                   ['Uses R+120 ms minus R-40 ms; J point is not located; not a J+80 ms measurement'])


def t_wave(session,args):
    x,fs,times,indices=_mv(session,args); rows=[]
    start, _ = session.sampled_window(args)
    for time,r in zip(times,indices):
        a,b=r+round(.15*fs),r+round(.40*fs); base=r-round(.04*fs)
        if base<0 or b>=len(x) or b<=a:continue
        dev=x[a:b]-x[base]; j=int(np.argmax(abs(dev)))
        rows.append({'r_time_s':float(time),'t_candidate_time_s':float(start+(a+j)/fs),'amplitude_mv':float(dev[j])})
    return _output({'beats':rows,'measured_beats':len(rows)},
                   {'mean_amplitude_mv':float(np.mean([r['amplitude_mv'] for r in rows])) if rows else None},
                   ['Fixed R+150..400 ms search; no validated T-wave delineation'])


def qt(session,args):
    x,fs,times,indices=_mv(session,args); rows=[]
    rr=np.diff(times); mean_rr=float(np.mean(rr)) if len(rr) else None
    for time,r in zip(times,indices):
        onset=r-round(.05*fs);a,b=r+round(.15*fs),r+round(.50*fs);base=r-round(.04*fs)
        if onset<0 or base<0 or b>=len(x):continue
        seg=abs(x[a:b]-x[base])
        if len(seg)<4 or np.max(seg)<=1e-12:continue
        end=int(np.argmax(seg)); threshold=.15*np.max(seg)
        while end<len(seg) and seg[end]>threshold:end+=1
        # A T tail hitting the search boundary is unavailable, never a fabricated endpoint.
        if end==len(seg):continue
        value=float((a+end-onset)/fs*1000)
        rows.append({'r_time_s':float(time),'qt_proxy_ms':value})
    average=float(np.mean([r['qt_proxy_ms'] for r in rows])) if rows else None
    return _output({'beats':rows,'measured_beats':len(rows)},
        {'mean_qt_proxy_ms':average,'qtc_bazett_proxy_ms':float(average/np.sqrt(mean_rr)) if average is not None and mean_rr and mean_rr>0 else None},
        ['Assumed onset R-50 ms; threshold-estimated T end; no abnormal-value range filtering',
         'QTc uses mean RR and is an experimental proxy'])
