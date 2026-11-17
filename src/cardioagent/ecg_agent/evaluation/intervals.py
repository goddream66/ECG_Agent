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
