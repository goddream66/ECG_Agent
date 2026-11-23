"""Three independent model submissions for the streaming AF/AFL experiment."""
from copy import deepcopy
from math import isfinite
import json
from time import perf_counter
from . import measured_event, split_event, window_semantics
from .autonomous_decision import overlap

ASSESSMENTS = {'support':'SUPPORTS_TARGET', 'negative':'AGAINST_TARGET',
               'conflict':'UNCERTAIN', 'unavailable':'UNCERTAIN'}
TEXT = {'type':'string','minLength':1}
SYSTEM = ('Track the combined AF or AFL target. Either subtype counts. Scores are uncalibrated. '
          'Use only arrived measurements. Regular RR can occur in AFL. Quality and beat proxies provide context.')


def schema(name, description, properties, required):
    return dict(type='function',function=dict(name=name,description=description,
                parameters=dict(type='object',properties=properties,required=required,additionalProperties=False)))


def choice(values):
    return dict(type='string',enum=list(values))


def ask(model, counts, emit, audit, data, tool, instruction, validate, extras=(), measure=None):
    name = tool['function']['name']
    messages=[dict(role='system',content=SYSTEM+' '+instruction),
              dict(role='user',content=json.dumps(data,allow_nan=False))]
    failures=0
    used_measure=False
    while failures<2:
        began=perf_counter()
        counts['llm_calls']+=1
        log=dict(audit,phase=name,attempt=failures+1)
        try:
            available=[tool]+([] if used_measure else list(extras))
            try:
                reply=model.complete(messages,available)
            finally:
                counts['llm_ms']+=(perf_counter()-began)*1000
            for key in ('prompt_tokens','completion_tokens'):
                counts['llm_'+key]+=reply.usage.get(key,0)
            calls=reply.message.get('tool_calls',[])
            if len(calls)!=1:
                raise ValueError('Submit exactly one offered tool')
            fn=calls[0]['function']
            args=fn['arguments']
            args=json.loads(args) if isinstance(args,str) else args
            log.update(action=fn['name'],arguments=deepcopy(args))
            if not isinstance(args,dict):
                raise ValueError('Arguments must be an object')
            if fn['name']!=name:
                if used_measure or fn['name'] not in {t['function']['name'] for t in extras}:
                    raise ValueError('Choose one offered function')
                used_measure=True
                row=measure(fn['name'],args)
                emit('llm_steps',dict(log,status='observed',evidence_id=row['evidence_id']))
                messages.append(dict(role='user',content=json.dumps(row)))
                continue
            props=tool['function']['parameters']
            if set(args)-props['properties'].keys() or set(props['required'])-args.keys():
                raise ValueError('Expected fields: '+', '.join(props['required']))
            if not isinstance(args.get('reason'),str) or not args['reason'].strip():
                raise ValueError('Supply your reason')
            result=validate(args)
            emit('llm_steps',dict(log,status='stage_committed',result=result))
            return result
        except (ValueError,RuntimeError,TypeError,KeyError,OSError) as exc:
            failures+=1
            error=str(exc)[:500]
            emit('llm_steps',dict(log,status='stage_rejected',error_message=error))
            messages.append(dict(role='user',content=f'Rejected {log.get("action",name)}: {error}. Correct this stage only.'))
    return dict(status='error',reason=error)


def assess(model, counts, emit, audit, data, aliases, usable, visible_now, mode):
    rows = {r['id']:r for r in data['measurements']}
    a,b = data['window_s']
    def interpret(args):
        label=args['interpretation']
        if label not in ASSESSMENTS or not isinstance(args['evidence_ids'],list):
            raise ValueError('Choose a listed interpretation and a list of evidence IDs')
        ids=list(dict.fromkeys(args['evidence_ids']))
        for key in ids:
            if key not in rows or not visible_now(key):
                raise ValueError(f'{key} is not visible at this cursor')
        supporting=[key for key in ids if usable(key) and overlap(rows[key]['window_s'],a,b)]
        if label in ('support','negative') and not supporting:
            raise ValueError('Cite a current usable Founder pair for a directional observation')
        if label=='conflict' and len({(rows[k]['lead'],tuple(rows[k]['window_s'])) for k in supporting})<2:
            raise ValueError('Conflict requires two distinct overlapping current rhythm measurements')
        spans=[[max(a,rows[k]['window_s'][0]),min(b,rows[k]['window_s'][1])] for k in supporting]
        return dict(status='accepted',interpretation=label,assessment=ASSESSMENTS[label],
                    evidence_ids=[aliases[k] for k in ids],directional_ids=[aliases[k] for k in supporting],
                    support_intervals=spans if label=='support' else [],
                    opposing_intervals=spans if label=='negative' else [],reason=args['reason'])
    tool=schema('assess_window','Record your judgment of this interval.',
                dict(interpretation=choice(ASSESSMENTS),evidence_ids=dict(type='array',items=choice(rows) if rows else TEXT),reason=TEXT),
                ['interpretation','evidence_ids','reason'])
    instruction=('Judge the current interval only. Cite the measurements you used. The runtime identifies auxiliary tools as context. '
            'support favors either subtype, negative weighs against both, conflict means opposed current measurements, unavailable includes insufficient evidence.')
    observation_data, tool, instruction = window_semantics.window_request(data, tool, instruction, mode)
    def validate_observation(args):
        return interpret(window_semantics.canonical_arguments(args, mode))
    return ask(model,counts,emit,audit,observation_data,tool,instruction,validate_observation)


def choose_action(model, counts, emit, audit, window, obs, active, candidate):
    a,b = window
    entity = active or candidate
    ident = entity['event_id'] if entity else None
    if active:
        actions=['CONTINUE_EVENT','HOLD_EVENT','REQUEST_CLOSE','REFINE_ONSET','NO_EVENT_CHANGE']
    elif candidate:
        actions=['CONFIRM_EVENT','REFINE_ONSET','HOLD_EVENT','DISMISS_CANDIDATE','NO_EVENT_CHANGE']
    else:
        actions=['CREATE_CANDIDATE','CONFIRM_EVENT','NO_EVENT_CHANGE']
    def lifecycle(args):
        action=args['action']
        if action not in actions or args['event_id']!=ident or (ident is not None and type(args['event_id']) is not int):
            raise ValueError(f'Select one of {actions} and use event_id={ident}')
        if action in ('CONFIRM_EVENT','CREATE_CANDIDATE') and obs['interpretation']!='support':
            raise ValueError('Registering an episode requires a supported observation')
        if action=='REQUEST_CLOSE' and obs['interpretation']!='negative':
            raise ValueError('Request closure when current evidence weighs against both subtypes')
        return dict(status='accepted',**args)
    tool=schema('choose_event_action','Update the episode ledger independently of its boundary.',
                dict(action=choice(actions),event_id={'enum':[ident]},reason=TEXT),['action','event_id','reason'])
    return ask(model,counts,emit,audit,
               dict(window_s=[a,b],assessment=obs,open_event=active,candidate=candidate,event_id=ident),
               tool,'CREATE_CANDIDATE records a provisional finding. CONFIRM_EVENT registers an episode now, even on first detection. '
               'You may hold or leave the ledger unchanged. CONFIRM_EVENT never requires a known onset. REQUEST_CLOSE starts an end investigation.',lifecycle)


def submit(reviewer, state, aliases, ledger, session, emit, audit, call_window, tools):
    data=json.loads(measured_event.final_context(state,aliases,ledger)[1]['content'])
    rows={r['id']:r for r in data['measurements']}
    a,b=state['window_s']
    def usable(key):
        r=rows.get(key,{})
        return (r.get('direction_allowed') and all(type(r['fields'].get(k)) in (int,float) and isfinite(r['fields'][k]) for k in measured_event.PAIR)
                and r['window_s'][1]<=b and ledger.get(aliases[key])['available_at_s']<=b)
    obs = assess(reviewer.model, reviewer.counts, emit, audit, data, aliases, usable,
                 lambda key: ledger.get(aliases[key])['available_at_s'] <= b,
                 getattr(reviewer, 'window_assessment_mode', 'legacy'))
    if obs['status']=='error':
        return dict(action='ACTION_ERROR',decision='ACTION_ERROR',evidence_ids=[],reason=obs['reason'],window_assessment=obs)
    emit('assessments',dict(audit,stage='window',window_s=[a,b],**obs))
    active=state['open_event']
    candidate=state['candidate']
    intent = choose_action(reviewer.model, reviewer.counts, emit, audit, [a,b], obs, active, candidate)
    emit('llm_steps',dict(audit,stage='lifecycle',status='lifecycle_result',intent=intent))
    action=intent.get('action')
    refinement=dict(status='not_requested')
    if action in ('CREATE_CANDIDATE','CONFIRM_EVENT','REFINE_ONSET','REQUEST_CLOSE'):
        def refine(args):
            if args['status']=='unlocalized':
                return dict(status='unlocalized',reason=args['reason'])
            left,right=args.get('before_id'),args.get('after_id')
            if args['status']!='located' or not all(usable(k) for k in (left,right)):
                raise ValueError('Choose usable before_id and after_id or unlocalized')
            x,y=rows[left]['window_s']
            z,w=rows[right]['window_s']
            if y>z or x<state['last_closed_end_s'] or (action=='REQUEST_CLOSE' and
                    (x<active['estimated_start_s'] or (x+w)/2<=active['estimated_start_s'])):
                raise ValueError('Anchors must be ordered, non-overlapping and within this episode timeline')
            if action!='REQUEST_CLOSE' and active and z>active['confirmed_at_s']:
                raise ValueError('An onset anchor cannot first appear after episode confirmation')
            current=set(obs['directional_ids'])
            if action in ('REQUEST_CLOSE','CONFIRM_EVENT','CREATE_CANDIDATE') and aliases[right] not in current and right!='B1':
                raise ValueError('Use a current directional anchor for this action or leave unlocalized')
            expected = ('support','negative') if action=='REQUEST_CLOSE' else ('negative','support')
            previous = state.get('recent_window_votes',{})
            for key, direction in zip((left,right),expected):
                known = previous.get(aliases[key])
                if known and known != direction:
                    raise ValueError(f'{key} has an opposite previously committed rhythm judgment')
            return dict(status='located',boundary=dict(kind='bracketed',range_s=[x,w]),
                        before_id=aliases[left],after_id=aliases[right],reason=args['reason'])
        props=dict(status=choice(['unlocalized','located']),before_id=TEXT,after_id=TEXT,reason=TEXT)
        tool=schema('refine_boundary','Name ordered measurements on opposite sides or leave the boundary unlocalized.',
                    props,['status','reason'])
        extras=[t for t in tools if t['function']['name'] in split_event.TOOL_GUIDE and
                len(session.calls)<session.max_calls]
        def measure(name,args):
            parsed=session.registry._tools[name].arguments.model_validate(args)
            if not session.context.origin_s<=parsed.start_s<parsed.end_s<=b:
                raise ValueError('Outside retained arrived interval')
            row=call_window(name,parsed.model_dump())
            aliases['B1']=row['evidence_id']
            rows['B1']=dict(window_s=row['window_s'],lead=row['lead'],fields=measured_event.numeric_fields(row),
                            direction_allowed=name==measured_event.FOUNDER and row['status']=='ok')
            return dict(id='B1',**row)
        direction='before supports either subtype and after weighs against both' if action=='REQUEST_CLOSE' else 'before weighs against both and after supports either'
        refinement=ask(reviewer.model,reviewer.counts,emit,audit,
                       dict(window_s=[a,b],event_action=action,open_event=active,measurements=list(rows.values())),
                       tool,'Localize only if '+direction+'. You may measure one retained interval. '
                       'Use unlocalized if no pair supports the change. The ledger already saved the observation and lifecycle intent.',refine,extras,measure)
    payload=dict(**obs,window_assessment=obs,lifecycle=intent,refinement=refinement,
                 event_action=action or 'LIFECYCLE_ERROR',action=action or 'LIFECYCLE_ERROR',
                 boundary=refinement.get('boundary'),pending=None,decision_source='llm_staged')
    payload.update(split_event.appointment(reviewer.model,state,payload,reviewer.counts,emit,audit))
    return payload

