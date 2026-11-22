"""Offline interval evaluation. References never enter the inference manifest."""
from pathlib import Path
from statistics import mean
from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.provenance import save_provenance
from cardioagent.common.io import read_jsonl
from cardioagent.common.intervals import validate_intervals
from cardioagent.ecg_agent.evaluation.intervals import burden_metrics, interval_metrics, mean_burdens


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
        support = pred[task_id].get('burden_intervals')
        supported = None if support is None else validate_intervals(support, start, end)
        rows.append({'task_id':task_id,'reference_positive':bool(expected),
            'duration_s':end-start,**metrics,**burden_metrics(actual, supported, {**ref,'start_s':start,'end_s':end})})
    positives=[r for r in rows if r['reference_positive']];negatives=[r for r in rows if not r['reference_positive']]
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    denominator=2*tp+fp+fn
    report={'experiment':'ecg_scroll_interval_evaluation','tasks':len(rows),
        'positive_tasks':len(positives),'negative_tasks':len(negatives),
        'macro_episode_f1':mean(r['f1'] for r in rows),
        'positive_macro_episode_f1':mean(r['f1'] for r in positives) if positives else None,
        'negative_macro_episode_f1':mean(r['f1'] for r in negatives) if negatives else None,
        'micro_episode_f1':2*tp/denominator if denominator else 1.,'tp':tp,'fp':fp,'fn':fn,
        **mean_burdens(rows),
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
