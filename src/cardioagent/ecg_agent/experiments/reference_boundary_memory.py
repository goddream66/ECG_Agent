"""Give Qwen arrived reference boundaries in an isolated event-ledger diagnostic."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from cardioagent.common.io import digest, save_json, write_jsonl
from cardioagent.ecg_agent.harness import measured_event
from cardioagent.ecg_agent.harness.autonomous_decision import Point
from cardioagent.ecg_agent.harness.staged_review import TEXT, ask, choice, schema
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.memory.evidence import EvidenceLedger, canonical_hash
from cardioagent.ecg_agent.memory.staged_events import StagedEvents
from cardioagent.ecg_agent.schemas import EvidenceContext


def boundary_decision(model, counts, emit, ledger, trace, tracker, kind, ref, waveform):
    cursor = trace['cursor_s']
    index = 0 if kind=='onset' else 1
    time_s = ref['interval_s'][index]
    if time_s>cursor:
        raise ValueError('Cannot release a future reference boundary')
    released = ref['interval_s'] if kind=='offset' else [time_s,None]
    proof = dict(tool='diagnostic_reference', lead=None, window_s=None, status='ok',
                 available_at_s=cursor, result=dict(origin='LTAFDB reference annotation',
                 available_interval_s=released, boundary=kind, time_s=time_s))
    item = ledger.append(dict(proof,evidence_id='annotation:'+canonical_hash(proof)),
                         input_fingerprint=ref['sha256'])
    alias = 'REF_START' if kind=='onset' else 'REF_END'
    data = dict(window_s=[trace['start_s'],cursor], cursor_s=cursor, requested_boundary=kind,
                open_event=tracker.snapshot(trace['start_s'],cursor,False)['open_event'],
                reference_evidence=dict(id=alias,**item['result']), waveform_observations=waveform)
    tool = schema('write_event_boundary','Submit the located boundary and its source reference to the experiment ledger.',
                  dict(boundary=choice([kind]), time_s=dict(type='number',minimum=0,maximum=cursor),
                       reference_id=choice([alias]), reason=TEXT),
                  ['boundary','time_s','reference_id','reason'])

    def validate(args):
        if args['boundary']!=kind or args['reference_id']!=alias:
            raise ValueError('Use the displayed reference ID and requested boundary kind')
        point = Point(kind='localized',time_s=args['time_s'])
        if point.time_s!=time_s:
            raise ValueError('Use the exact boundary time supplied by this reference')
        boundary = dict(point.model_dump(), source='LTAFDB_reference_diagnostic',evidence_id=item['evidence_id'])
        ledger.assessment('reference_boundary',args['reason'],[item['evidence_id']],cursor,checked=True)
        return dict(status='located',boundary=boundary,reason=args['reason'])

    located = ask(model,counts,emit,dict(cursor_s=cursor),data,tool,
                  'This isolated experiment provides a reference annotation as explicit point-time evidence. '
                  'Read the requested onset or offset, submit its exact time in seconds and cite the displayed source ID. '
                  'The waveform observations are context. A missing opposite boundary is still unknown.',
                  validate)
    return located,item


def apply_trace(tracker,row,payload):
    start,cursor = row['start_s'],row['cursor_s']
    if start!=tracker.cursor:
        raise ValueError('Trace is not contiguous')
    if row['decision'] in ('ACTION_ERROR','UNREVIEWED'):
        tracker.skip(start,cursor,retry=row['decision']=='ACTION_ERROR')
        return None
    return tracker.apply(start,cursor,payload)


def replay(data,model,emit):
    trace = data['trace']
    start = trace[0]['start_s']
    case = data['reference']
    tracker = StagedEvents('fixed',start)
    original = StagedEvents('fixed',start)
    ledger = EvidenceLedger(EvidenceContext(run_id='reference-boundary-diagnostic',
                            task_id=case['task_id'],record_id=case['record_id'],manifest_id=case['sha256']))
    planned = {row:kind for kind,row in case['decision_cursors'].items()}
    counts = Counter()
    results, updates, receipts, normal_rejections = {},[],[],{}
    for row in trace:
        payload = deepcopy(row['llm_action'])
        apply_trace(original,row,deepcopy(payload))
        cursor = row['cursor_s']
        if cursor in planned:
            kind = planned[cursor]
            visible = [m for m in data['observations'] if m['span_s'][1]<=cursor and
                       m['available_at_s']<=cursor and m['span_s'][1]>cursor-30]
            finding,item = boundary_decision(model,counts,emit,ledger,row,tracker,kind,case,visible)
            results[kind] = finding
            receipts.append(item)
            if finding['status']=='located':
                # Normal measured evidence validation requires an interval bracket from arrived signal tools.
                try:
                    measured_event.validate_boundary(Point(kind='localized',time_s=finding['boundary']['time_s']),
                                                     'OPEN' if kind=='onset' else 'CLOSE',{})
                except measured_event.BoundaryEvidenceNeeded as exc:
                    normal_rejections[kind] = str(exc)
                # Only this standalone entry point submits the reference point to the standard event ledger.
                payload.update(refinement=finding, evidence_ids=[*payload['evidence_ids'],item['evidence_id']])
        state = apply_trace(tracker,row,payload)
        if state and state['event_update']:
            updates.append(state['event_update'])
    recovered = tracker.retrieve(trace[0]['start_s'],tracker.cursor,available_through_s=tracker.cursor,limit=5,event_id=1)
    closed = recovered[0] if len(recovered)==1 else {}
    exact = (closed.get('final_status')=='CLOSED' and
             [closed.get('estimated_start_s'),closed.get('estimated_end_s')]==case['interval_s'] and
             len(receipts)==2 and
             receipts[0]['evidence_id'] in closed['start_evidence_ids'] and
             receipts[1]['evidence_id'] in closed['end_evidence_ids'])
    return dict(diagnostic_only=True, reference=case, model_submissions=results, normal_gate_rejections=normal_rejections,
                retrieved_event=recovered, original_events=original.events, event_intervals=tracker.intervals,
                exact_reference_written_and_retrieved=exact, updates=updates, evidence=receipts, counts=dict(counts))


if __name__=='__main__':
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    model = RecordingChatModel(LocalQwenModel(dict(json.loads(args.config.read_bytes()),device='cuda:0')),
                               args.output/'qwen_calls.jsonl.gz')
    steps = []
    report = replay(json.loads(args.input.read_bytes()),model,lambda tag,row:steps.append(dict(tag=tag,**row)))
    save_json(args.output/'result.json',dict(input_sha256=digest(args.input),config_sha256=digest(args.config),**report))
    write_jsonl(args.output/'reference_evidence.jsonl',report['evidence'])
    write_jsonl(args.output/'event_updates.jsonl',report['updates'])
    write_jsonl(args.output/'llm_steps.jsonl',steps)
    print('reference_written_and_retrieved',report['exact_reference_written_and_retrieved'],flush=True)
