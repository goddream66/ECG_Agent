"""ECG-Scroll interval verifiers adapted under MIT; see THIRD_PARTY_NOTICES.md.

Keep the original prediction-order greedy matching and burden score for comparison.
"""
import math


def validate_intervals(intervals, start, end):
    if not all(math.isfinite(t) for t in (start,end)) or not 0<=start<end:
        raise ValueError('Task bounds must be finite and ordered')
    if not isinstance(intervals,list):raise ValueError('intervals must be a list')
    result=[]
    for row in intervals:
        if not isinstance(row,(list,tuple)) or len(row)!=2:
            raise ValueError('Each interval must contain two times')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in row):
            raise ValueError('Interval times must be finite numbers')
        a,b=map(float,row)
        if not start<=a<b<=end:raise ValueError('Interval exceeds task bounds or has invalid duration')
        result.append([a,b])
    return result


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


def union_seconds(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)


def burden_score(predicted,reference,floor=.02):
    if not all(math.isfinite(x) for x in (predicted,reference,floor)) or floor<=0:
        raise ValueError('Finite values and positive floor required')
    return max(0.,1-abs(predicted-reference)/max(abs(reference),floor))
