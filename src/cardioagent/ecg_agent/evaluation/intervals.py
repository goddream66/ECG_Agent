"""Frozen ECG-Scroll interval scoring; attribution retained."""
import math
from cardioagent.common.intervals import validate_intervals, union_seconds

def iou(a,b):
    intersection=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    union=a[1]-a[0]+b[1]-b[0]-intersection
    return intersection/union if union>0 else 0.


def interval_metrics(predicted,reference,tau=.5):
    if not math.isfinite(tau) or not 0<tau<=1:raise ValueError('tau must be in (0,1]')
    used=set();tp=0
    for interval in predicted:
        best,best_iou=-1,tau
        for j,target in enumerate(reference):
            if j in used:continue
            value=iou(interval,target)
            if value>=best_iou:best,best_iou=j,value
        if best>=0:used.add(best);tp+=1
    fp,fn=len(predicted)-tp,len(reference)-tp
    denominator=2*tp+fp+fn
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/len(predicted) if predicted else None,
            'recall':tp/len(reference) if reference else None,
            'f1':2*tp/denominator if denominator else 1.}


def burden_score(predicted,reference,floor=.02):
    if not all(math.isfinite(x) for x in (predicted,reference,floor)) or floor<=0:
        raise ValueError('Finite values and positive floor required')
    return max(0.,1-abs(predicted-reference)/max(abs(reference),floor))


def burden_metrics(intervals, support, reference):
    duration = reference['end_s'] - reference['start_s']
    target = float(reference['burden'])
    if not 0 <= target <= 1:
        raise ValueError('Reference burden must lie in [0,1]')
    floor = float(reference.get('burden_floor', .02))
    predicted = union_seconds(intervals) / duration
    supported = None if support is None else union_seconds(support) / duration
    return dict(predicted_burden=predicted, reference_burden=target, burden_floor=floor,
                burden_score=burden_score(predicted, target, floor),
                burden_absolute_error_pp=100 * abs(predicted - target),
                supported_burden=supported,
                supported_burden_score=None if supported is None else burden_score(supported, target, floor),
                supported_burden_absolute_error_pp=None if supported is None else 100 * abs(supported - target))


def mean_burdens(rows):
    report = {}
    for name in ('burden_score', 'burden_absolute_error_pp',
                 'supported_burden_score', 'supported_burden_absolute_error_pp'):
        values = [r[name] for r in rows if r.get(name) is not None]
        report['mean_' + name] = sum(values) / len(values) if values else None
    report['supported_burden_records'] = sum(r.get('supported_burden') is not None for r in rows)
    return report
