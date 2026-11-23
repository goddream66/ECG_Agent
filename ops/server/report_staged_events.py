"""Build linked observation/episode views and offline staged metrics from a finished paired run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def read_rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_rows(path, rows):
    with path.open('w',encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def ratio(n,d):
    return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def match_alerts(episodes, intervals, field):
    """Match one report to one reference when its actual decision cursor falls inside it."""
    matched={}
    for e in sorted(episodes,key=lambda item:item[field]):
        for i,(start,end) in enumerate(intervals):
            if i not in matched and start<=e[field]<end:
                matched[i]=e
                break
    return matched


def analyze(folder, reference):
    windows=read_rows(folder/'window_trace.jsonl')
    results=json.loads((folder/'result.json').read_text(encoding='utf8'))
    journal=read_rows(folder/'llm_steps.jsonl')
    observations=read_rows(folder/'assessments.jsonl')
    measurements={r['evidence_id']:r for r in read_rows(folder/'evidence.jsonl')}
    identity=next(r for r in journal if 'run_id' in r)
    prefix='/'.join(identity[k] for k in ('run_id','task_id','record_id'))
    candidates={e['event_id']:e for e in results['candidates']}
    confirmed={e['event_id']:e for e in results['events']}
    cursor_to_obs={r['cursor_s']:r for r in observations if r.get('stage')=='window'}
    items=[]
    associations=defaultdict(list)
    intentions=Counter()
    related_id=None
    pending_errors=Counter()
    retrievals=Counter()
    for w in windows:
        end=w['cursor_s']
        p=w.get('llm_action') or {}
        raw=cursor_to_obs.get(end)
        if raw is None:
            if (p.get('window_assessment') or {}).get('status')=='accepted':
                raise ValueError('Window journal missing an accepted observation')
            continue
        obs=p['window_assessment']
        if raw['assessment']!=obs['assessment'] or raw['evidence_ids']!=obs['evidence_ids']:
            raise ValueError('Window journal and final trace disagree')
        finding_id=f'{prefix}/w/{end:g}'
        intent=p['lifecycle']
        action=intent.get('action') if intent['status']=='accepted' else None
        intentions[action or 'LIFECYCLE_ERROR']+=1
        if action in ('CREATE_CANDIDATE','CONFIRM_EVENT') and related_id is None:
            table=candidates if action=='CREATE_CANDIDATE' else confirmed
            time_field='detected_at_s' if action=='CREATE_CANDIDATE' else 'confirmed_at_s'
            related_id=next(i for i,e in table.items() if e[time_field]==end)
        if related_id is not None:
            associations[related_id].append(finding_id)
        items.append(dict(finding_id=finding_id,window_s=raw['window_s'],recorded_at_s=end,
                          interpretation=obs['interpretation'], reason=obs['reason'],record_status='saved',
                          linked_episode_id=related_id,evidence=[{k:measurements[e][k] for k in
                          ('evidence_id','lead','window_s','tool','status','available_at_s')} for e in obs['evidence_ids']]))
        if action=='DISMISS_CANDIDATE' or (action=='REQUEST_CLOSE' and p['refinement']['status']=='located'):
            related_id=None
    episodes=[]
    for i in sorted(candidates.keys()|confirmed.keys()):
        e=confirmed.get(i,{})
        c=candidates.get(i,{})
        end_boundary=e.get('end_boundary')
        episodes.append(dict(event_id=i,episode_key=f'{prefix}/e/{i}',candidate_created_at_s=c.get('detected_at_s'),
            detected_at_s=e.get('detected_at_s',c.get('detected_at_s')),confirmed_at_s=e.get('confirmed_at_s'),
            status=e.get('final_status',c.get('status')),onset_status=e.get('onset_status') or
            ('bracketed' if (c.get('onset_refinement') or {}).get('status')=='located' else 'unlocalized'),
            onset_boundary=e.get('onset_boundary') or (c.get('onset_refinement') or {}).get('boundary'),
            onset_estimate_s=e.get('estimated_start_s') if e.get('onset_status')=='bracketed' else None,
            scoring_start_s=e.get('estimated_start_s'),end_status='bracketed' if end_boundary else
            'right_censored' if e.get('right_censored') else 'unlocalized',end_boundary=end_boundary,
            end_decided_at_s=e.get('end_decided_at_s'),window_finding_ids=associations[i]))
    for item in items:
        ident=item['linked_episode_id']
        item['linked_episode_key']=f'{prefix}/e/{ident}' if ident is not None else None
    save_rows(folder/'window_findings.jsonl',items)
    save_rows(folder/'episode_ledger.jsonl',episodes)
    truth=reference['intervals']
    detections=match_alerts(episodes,truth,'detected_at_s')
    formal=[e for e in episodes if e['confirmed_at_s'] is not None]
    confirmations=match_alerts(formal,truth,'confirmed_at_s')
    localized=[e for e in episodes if e['onset_status']=='bracketed']
    boundary_errors=[abs(confirmations[i]['onset_estimate_s']-truth[i][0]) for i in confirmations
                     if confirmations[i]['onset_estimate_s'] is not None]
    window_pos=[x for x in items if x['interpretation']=='support']
    def overlaps(w,interval):
        a,b=(w['window_s'] if 'window_s' in w else (w['start_s'],w['cursor_s']))
        return a<interval[1] and interval[0]<b
    for r in journal:
        if r['status']=='stage_rejected':
            pending_errors[r['phase']]+=1
        if r['status'] in ('memory_retrieved','memory_written'):
            retrievals[r['status']]+=1
    # A bracket was proposed versus an accurate bracket: score the latter separately.
    bracket_hits=sum(e['onset_boundary']['range_s'][0]<=truth[i][0]<=e['onset_boundary']['range_s'][1]
                     for i,e in confirmations.items() if e['onset_estimate_s'] is not None)
    return dict(bracket_contains_onset=ratio(bracket_hits,len(boundary_errors)), wakes=results['counts']['review_wakes'],saved=len(items),
        window_votes=dict(Counter(r['interpretation'] for r in items)),lifecycle=dict(intentions),
        stage_rejected_attempts=dict(pending_errors),memory=dict(retrievals),episodes=len(episodes),
        candidates=len(candidates),confirmed=len(formal),closed=sum(e['status']=='CLOSED' for e in episodes),
        unlocalized=sum(e['onset_status']=='unlocalized' for e in episodes),
        detection=ratio(len(detections),len(truth)),confirmation=ratio(len(confirmations),len(truth)),
        candidate_confirmation=ratio(sum(c['status']=='confirmed' for c in candidates.values()),len(candidates)),
        localization=ratio(len(localized),len(episodes)),unlocalized_rate=ratio(len(episodes)-len(localized),len(episodes)),
        unmatched_reports=len(episodes)-len(detections),matched_onset_error_n=len(boundary_errors),
        onset_mae_s=mean(boundary_errors) if boundary_errors else None,
        detection_delays_s=[e['detected_at_s']-truth[i][0] for i,e in detections.items()],
        support_inside=sum(any(overlaps(x,t) for t in truth) for x in window_pos),
        support_outside=sum(not any(overlaps(x,t) for t in truth) for x in window_pos),
        positive_reference_windows=sum(any(overlaps(w,t) for t in truth) for w in windows),
        first_support_cursor_s=min((x['recorded_at_s'] for x in window_pos),default=None),
        episode_records=episodes)


def main(root):
    paired=json.loads((root/'paired_results.json').read_text())
    output={}
    for arm in paired['arms']:
        folder=next((root/arm['job']).glob('*/records/04936/result.json')).parent
        output[arm['job']]=analyze(folder,paired['reference'])
    (root/'staged_metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)


