"""Pure interval validation and duration; no labels or evaluation I/O."""
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


def union_seconds(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)
