"""Offline interval evaluation. References never enter the inference manifest."""
from pathlib import Path
from statistics import mean
from ...dataset import digest,save_json
from ...provenance import save_provenance
from .tasks import read_jsonl
from .intervals import validate_intervals,interval_metrics,union_seconds,burden_score


def score_files(references,predictions,output):
    references,predictions,output=map(Path,(references,predictions,output))
    truth=read_jsonl(references,'task_id');pred=read_jsonl(predictions,'task_id')
    if set(truth)!=set(pred):
        raise ValueError(f'Prediction task IDs must match references exactly: missing={len(set(truth)-set(pred))}, extra={len(set(pred)-set(truth))}')
    rows=[]
    for task_id,ref in truth.items():
        start,end=float(ref['start_s']),float(ref['end_s'])
        expected=validate_intervals(ref['intervals'],start,end)
        actual=validate_intervals(pred[task_id]['intervals'],start,end)
        metrics=interval_metrics(actual,expected,float(ref['tau']))
        burden=union_seconds(actual)/(end-start)
        target=float(ref['burden'])
        if not 0<=target<=1:raise ValueError('Reference burden must lie in [0,1]')
        rows.append({'task_id':task_id,'reference_positive':bool(expected),
            'duration_s':end-start,**metrics,'predicted_burden':burden,'reference_burden':target,
            'burden_absolute_error_pp':abs(burden-target)*100,
            'burden_score':burden_score(burden,target,float(ref['burden_floor']))})
    positives=[r for r in rows if r['reference_positive']];negatives=[r for r in rows if not r['reference_positive']]
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    denominator=2*tp+fp+fn
    report={'experiment':'ecg_scroll_interval_evaluation','tasks':len(rows),
        'positive_tasks':len(positives),'negative_tasks':len(negatives),
        'macro_episode_f1':mean(r['f1'] for r in rows),
        'positive_macro_episode_f1':mean(r['f1'] for r in positives) if positives else None,
        'negative_macro_episode_f1':mean(r['f1'] for r in negatives) if negatives else None,
        'micro_episode_f1':2*tp/denominator if denominator else 1.,'tp':tp,'fp':fp,'fn':fn,
        'mean_burden_score':mean(r['burden_score'] for r in rows),
        'mean_burden_absolute_error_pp':mean(r['burden_absolute_error_pp'] for r in rows),
        'false_events_per_recording_hour':fp/(sum(r['duration_s'] for r in rows)/3600),
        'rows':rows,'limitations':[
            'Scores final intervals only; no online latency, costs or memory performance is inferred.',
            'Preserves upstream prediction-order greedy IoU matching; no prediction sorting or merging for event F1.',
            'Burden is derived from interval union, not a separate model guess.',
            'Both-empty episode score is 1; positive/negative groups are reported separately.',
            'Small cropped runs are engineering pilots, not full benchmark or held-out clinical validation.']}
    output.mkdir(parents=True,exist_ok=False)
    report['provenance']=save_provenance(output,{'experiment':report['experiment'],
        'references_sha256':digest(references),'predictions_sha256':digest(predictions)})
    save_json(output/'interval_scores.json',report)
    return report
