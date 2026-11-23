"""Decision and appointment contracts for the split event-review experiment."""
import json
from math import isfinite
from time import perf_counter
from . import autonomous_decision as legacy


TOOL_GUIDE = {
    'score_ecgfounder': 'AF and AFL each range 0 to 1. For orientation 0.1 is near the low end, 0.5 is mid-scale and 0.9 is near the high end. A high value for either favors the combined target. Two low values weaken target support. These are uncalibrated model outputs, with no mandatory cutoff for OPEN or CLOSE.',
    'assess_quality': 'Score is a heuristic band-power fraction, or zero on a flag. Flat run >0.5 s, baseline fraction >0.70 or high-frequency fraction >0.35 triggers a flag. High score describes signal characteristics only.',
    'detect_beats': 'Peak times and RR are seconds, rate is bpm. CV is SD/mean: 0.02 means 2% relative spread, 0.20 means 20%. Count depends on duration and detector errors. There is no AF-specific high count.',
    'detect_beats_xqrs_proxy': 'Local XQRS times in seconds; RR in ms. removed_fraction ranges 0 to 1: 0.8 means 80% removed, so inspect raw intervals and detector coverage.',
    'get_rate_rhythm': 'Compare raw and cleaned RR in ms. CV 0.02 vs 0.20 means 2% vs 20% relative variation. Larger means more variable detected intervals. RR outside 300..2000 ms fails the cleaning filter. Regular RR is compatible with AFL.',
    'get_feature_profile': 'Each time bin gives RR CV and number of intervals. Peak is the highest observed CV bin, not a rhythm-change point. Null means too few intervals.',
    'get_p_wave_proxy': 'Search 250..50 ms before R. Fraction 0..1 counts candidates with absolute amplitude >0.05 mV. A fraction of 1 means all eligible beats passed this amplitude test. It does not establish sinus rhythm.',
    'get_qrs_proxy': 'Occupancy is ms above a local amplitude threshold in about 240 ms around R. Larger means more time above that threshold. Consistent occupancy alone gives no AF/AFL direction.',
    'get_beat_timeline_proxy': 'Rows align beat time (s), RR (ms), candidate P amplitude (mV), and QRS occupancy (ms). Compare before/after beats for a boundary. A peak timestamp locates a beat, not the rhythm change. Rows reuse the peaks of related tools.',
}


def prompt(state, phase, *, retry=False):
    fragments = legacy._fragments()
    parts = [fragments['core']['en'], fragments['open' if state['open_event'] else 'no_open']['en']]
    if state['pending_question']:
        parts.append(fragments['resume']['en'])
    if phase == 'acquire':
        parts.append(fragments['acquire']['en'])
    else:
        parts.append("""Submit exactly one event decision. First assess this interval. Name the decisive result and its lead, interval and value in reason. Give a directional citation its numeric result field (for example afl_score); the ledger attaches the measured value. Cite each useful measurement once in citations; omit redundant tools. Use CONTEXT_ONLY for quality, isolated RR, stable QRS and nonspecific P proxies. If a proxy has directional meaning explain the specific finding and the before/after comparison. Multiple tools sharing the same beats do not confirm each other.
No open event: OPEN only when current evidence favors AF or AFL and warrants recording an episode. If onset was not observed use onset_unobserved with the first supported interval. Mid-scale Founder scores alone are ambiguous; check the other lead or leave a specific question pending when needed.
Event already open: compare the last target-supporting interval with the latest usable evidence. If you judge AF/AFL ended, CLOSE with a measured point or a bracket anchored by evidence on both sides. Cite the earlier support and current contrary evidence at their actual times. A direct change from AF to AFL remains one event. If rhythm evidence now weighs against the target but the end boundary is unclear, use AGAINST_TARGET + DEFER_DECISION, naming the boundary question and needed measurement. If the current rhythm itself is unclear, use UNCERTAIN and its cause. RECORD_ONLY records a routine observation. Choose DEFER_DECISION when you have a specific unresolved question requiring follow-up.
Only OPEN and CLOSE supply boundary. Only DEFER_DECISION supplies pending. Only UNCERTAIN supplies uncertainty_cause. Leave inapplicable fields out. Appointments are handled separately.""")
    if state['stream_end']:
        parts.append('Stream has ended. A missing measured termination leaves the open event right-censored.')
    if retry:
        parts.append('Repair the rejected event fields using the same measurements.')
    return '\n\n'.join(parts)


def final_tool(state):
    tool = legacy.final_tool(state)
    schema = tool['function']['parameters']
    props = schema['properties']
    props['citations'] = props.pop('evidence_attributions')
    definition = schema['$defs']['EvidenceAttribution']
    definition['properties']['field'] = {'type':'string','description':'For a directional citation, exact numeric result field, e.g. afl_score, raw.cv or beats.0.amplitude_mv. Omit for CONTEXT_ONLY.'}
    for key in ('evidence_ids','next_review_s'):
        del props[key]
    schema['required'] = [k for k in ('assessment','event_action','citations','reason','pending_resolution') if k in props]
    tool['function']['description'] = 'Commit the ECG observation and event action. Use only fields applicable to that action.'
    return tool


def event(args, aliases, visible, ledger, state, boundary_tools):
    args = dict(args)
    for key in ('next_review_s','evidence_ids','evidence_attributions'):
        if key in args:
            raise ValueError(f'Unexpected event field {key}; use citations and omit scheduling')
    cite = args.pop('citations', None)
    if not isinstance(cite, list) or any(not isinstance(c,dict) or 'evidence_id' not in c for c in cite):
        raise ValueError('Each citation needs evidence_id and role')
    facts=[]
    clean=[]
    for item in cite:
        item=dict(item)
        path=item.pop('field',None)
        clean.append(item)
        key=aliases.get(item['evidence_id'],item['evidence_id'])
        if key not in visible:
            raise ValueError('Cite a visible evidence ID')
        if item.get('role')=='CONTEXT_ONLY':
            continue
        row=ledger.get(key)
        value=row['result']
        if not isinstance(path,str) or not path:
            raise ValueError('Directional citation requires the numeric field that supports it')
        try:
            for part in path.split('.'):
                value=value[int(part)] if isinstance(value,list) and part.isdecimal() else value[part]
        except (IndexError,TypeError,KeyError):
            raise ValueError(f'No measured field {path} in {item["evidence_id"]}') from None
        if type(value) not in (float,int) or not isfinite(value):
            raise ValueError('Directional field must identify a finite measured number')
        if row['tool']=='score_ecgfounder' and path not in ('af_score','afl_score'):
            raise ValueError('Cite af_score or afl_score')
        facts.append(dict(evidence_id=key,field=path,value=value))
    for key in ('pending','uncertainty_cause','boundary'):
        args.setdefault(key,None)
    # Adapt to the unchanged legacy event validator. Scheduling is resolved separately.
    args.update(evidence_attributions=clean,evidence_ids=[c['evidence_id'] for c in cite],
                next_review_s=None if state['stream_end'] else state['scheduled_review_s'] or state['cursor_s']+10)
    payload=legacy.validate(args,aliases,visible,ledger,state,boundary_tools)
    payload['measured_facts']=facts
    payload['event_valid']=True
    return payload


def appointment(model,state,payload,counts,emit,audit):
    cursor = state['cursor_s']
    if state['stream_end'] or state['review_mode']=='fixed':
        return dict(next_review_s=state['scheduled_review_s'],schedule_source='runtime',schedule_valid=None)
    limit = state['latest_review_s']-cursor
    tool={'type':'function','function':{'name':'schedule_review','description':'Schedule the next review after a delay.',
          'parameters':{'type':'object','properties':{'delay_s':{'type':'number','exclusiveMinimum':0,'maximum':limit}},
                        'required':['delay_s'],'additionalProperties':False}}}
    messages=[dict(role='system',content='Your event decision is accepted. Return schedule_review with a positive delay in seconds. For the next 10-second batch use 10. Weigh the risk of missing a short change.'),
              dict(role='user',content=json.dumps(dict(window_s=state['window_s'],max_delay_s=limit,
                   assessment=payload['assessment'],action=payload['event_action'],reason=payload['reason'],pending=payload['pending'])))]
    for attempt in (1,2):
        began=perf_counter()
        counts['llm_calls']+=1
        counts['schedule_calls']=counts.get('schedule_calls',0)+1
        try:
            reply=model.complete(messages,[tool])
            for key in ('prompt_tokens','completion_tokens'):
                counts['llm_'+key]+=reply.usage.get(key,0)
            calls=reply.message.get('tool_calls',[])
            if len(calls)!=1 or calls[0]['function']['name']!='schedule_review':
                raise ValueError('Submit one schedule_review')
            args=calls[0]['function']['arguments']
            args=json.loads(args) if isinstance(args,str) else args
            delay=args.get('delay_s') if isinstance(args,dict) and set(args)=={'delay_s'} else None
            if type(delay) not in (int,float) or not isfinite(delay) or not 0<delay<=limit or cursor+delay<=cursor:
                raise ValueError(f'delay_s must be greater than zero and at most {limit}')
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as exc:
            error=str(exc)[:500]
            emit('llm_steps',dict(audit,phase='appointment',attempt=attempt,status='schedule_rejected',error_message=error))
            messages.append(dict(role='user',content=error+'. Retry the appointment only.'))
        else:
            emit('llm_steps',dict(audit,phase='appointment',attempt=attempt,status='scheduled',delay_s=delay))
            return dict(next_review_s=cursor+delay,schedule_source='model',schedule_valid=True)
        finally:
            counts['llm_ms']+=(perf_counter()-began)*1000
    return dict(next_review_s=cursor+10,schedule_source='next_batch_recovery',schedule_valid=False)
