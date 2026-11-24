"""Localize a confirmed event end using four arrived ten-second windows and an optional centered pair."""
from math import isfinite
from collections import Counter
from .staged_review import TEXT, ask, choice, schema


def disagreement(values):
    return max(abs(values[i] - values[i+1]) for i in (2, 4))


def estimate_end(onset, grid_end, cursor, aligned, calibration, acquire=None):
    """acquire(start,end,cursor) returns the two measured target scores, or unavailable values."""
    if not all(isfinite(t) for t in (onset, grid_end, cursor)) or not onset < grid_end <= cursor:
        raise ValueError('Invalid confirmed event interval')
    base = dict(estimated_end_s=grid_end, recorded_at_s=cursor, requests=0, source='grid')
    def usable(values, count):
        return (values is not None and len(values)==count and
                all(v is not None and isfinite(v) and 0 <= v <= 1 for v in values))
    if grid_end < 20 or grid_end+20 > cursor or not usable(aligned,8):
        return base
    values = list(aligned)
    source = 'aligned'
    if acquire is not None and disagreement(values) >= calibration['min_disagreement']:
        start,end=grid_end-5,grid_end+5
        extra=acquire(start,end,cursor)
        base.update(requests=2, requested_interval_s=[start,end])
        if usable(extra,2):
            values.extend(extra)
            source='shifted'
    model=calibration[source]
    change=model['intercept']+sum(w*x for w,x in zip(model['weights'],values,strict=True))
    if not isfinite(change):
        raise ValueError('Invalid boundary calibration')
    estimate=min(cursor, grid_end+max(-15.,min(15.,change)))
    if estimate<=onset:
        return dict(base, source='grid_invalid_estimate')
    return dict(base, estimated_end_s=estimate, source=source)


def review_offset(onset, grid_end, cursor, aligned, calibration, model, acquire, emit):
    """Let Qwen decide whether the already elapsed, centered pair is worth measuring."""
    prior=estimate_end(onset,grid_end,cursor,aligned,calibration)
    if prior['source']!='aligned':
        return dict(prior, choice='ineligible', counts={})
    windows=[dict(window_s=[grid_end+d,grid_end+d+10],ECG1=aligned[2*i],ECG2=aligned[2*i+1])
             for i,d in enumerate((-20,-10,0,10))]
    data=dict(window_s=[grid_end+10,cursor],estimated_end_s=prior['estimated_end_s'],
              grid_end_s=grid_end,observations=windows,optional_window_s=[grid_end-5,grid_end+5])
    tool=schema('choose_offset_measurement','Choose whether to request one centered ten-second window on both leads.',
                dict(action=choice(['use_aligned','measure_centered_pair']),reason=TEXT),['action','reason'])
    def validate(args):
        if args['action'] not in ('use_aligned','measure_centered_pair'):
            raise ValueError('Unknown acquisition action')
        return dict(status='accepted',**args)
    counts=Counter()
    decision=ask(model,counts,emit,dict(cursor_s=cursor),data,tool,
                 'The event is already closed. Your choice only affects its estimated end time. '
                 'Each score is max(AF,AFL) on the indicated lead. Decide whether a pair centered on '
                 'the provisional grid boundary can reduce the timing uncertainty enough to justify two '
                 'extra measurements. If the aligned evidence suffices, use it. The runtime computes '
                 'the final numerical end estimate from whichever measurements you actually acquire.',
                 validate)
    action=decision.get('action','ACTION_ERROR')
    result=prior
    if action=='measure_centered_pair':
        result=estimate_end(onset,grid_end,cursor,aligned,dict(calibration,min_disagreement=0),acquire)
    return dict(result,choice=action,decision=decision,counts=dict(counts))
