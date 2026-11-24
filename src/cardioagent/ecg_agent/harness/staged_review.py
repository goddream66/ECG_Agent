"""Three independent model submissions for the streaming AF/AFL experiment."""
from copy import deepcopy
from math import isfinite
import json
from time import perf_counter
from . import measured_event, split_event, window_semantics
from .autonomous_decision import overlap
from .end_review import advance_end_memory, end_evidence
from . import active_acquisition
from .boundary_pairs import onset_pairs
from .compact_history import TABLE_GUIDE, table_view

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


def ask(model, counts, emit, audit, data, tool, instruction, validate, extras=(), measure=None, *, compact_history=False):
    name = tool['function']['name']
    if compact_history:
        view = table_view(data)
        if any(view[key] is not value for key, value in data.items()):
            data = view
            instruction += TABLE_GUIDE
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


def assess(model, counts, emit, audit, data, aliases, usable, visible_now, mode, profile=None, compare_leads=False,
           bound_choices=False, direction_only=False, *, advisory_reference=False, compact_history=False):
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
    observation_data, tool, instruction = window_semantics.window_request(data, tool, instruction, mode, profile)
    if direction_only:
        observation_data, instruction = window_semantics.direction_only_view(observation_data, instruction)
    advisory = window_semantics.advisory_pair(rows, [a,b], profile) if advisory_reference else ()
    if bound_choices:
        allowed=window_semantics.available_findings(rows,[a,b],profile)
        if advisory and 'TARGET_PRESENT' not in allowed:
            allowed.append('TARGET_PRESENT')
        tool['function']['parameters']['properties']['finding']['enum']=allowed
        observation_data['available_findings']=allowed
        instruction += (' Choose one available_finding. You may abstain from a directional finding.'
                        if advisory else
                        ' Choose one available_finding. These are the observations supported by the current '
                        'measurement contract. A numeric rise or fall inside an indeterminate band is still '
                        'indeterminate. You may abstain from a directional finding.')
    if compare_leads:
        flags={window_semantics.band(row,profile)['direction'] for row in rows.values()
               if row['window_s']==[a,b] and window_semantics.current_founder(row,[a,b])}
        opposing={'support','negative'} <= flags
        instruction += (' Check all measured leads on the same time interval. If one has a positive and another '
                        'a negative empirical direction, report CONFLICTING_MEASUREMENTS or CANNOT_TELL. '
                        'Missing measurements and the two subtypes of the same lead do not create lead disagreement.')
        if direction_only and bound_choices:
            observation_data, instruction = window_semantics.paired_observation_view(observation_data, instruction)
    if advisory:
        observation_data['advisory_pair_ids'] = advisory
        instruction += (' The two listed co-timed lead measurements permit a tentative target-present finding '
                        'even though at least one reference band is inconclusive. The empirical bands are '
                        'derived from other records. Cite both aligned IDs and explain why the current '
                        'combined AF-or-AFL evidence is persuasive, or choose CANNOT_TELL. The pair can share '
                        'artifacts. Do not infer target presence from two different subtype names alone.')
    def validate_observation(args):
        if bound_choices and args.get('finding') not in allowed:
            raise ValueError(f'Choose one of the available findings: {allowed}')
        canonical = window_semantics.canonical_arguments(args, mode)
        if compare_leads and opposing and canonical['interpretation'] in ('support','negative'):
            raise ValueError('Current measured rhythm directions disagree; acknowledge the conflict or uncertainty.')
        observation = interpret(canonical)
        if mode in ('verified','union_reference'):
            citations = canonical['evidence_ids']
            if (advisory and canonical['interpretation'] == 'support' and set(advisory).issubset(citations)
                    and not any(window_semantics.band(rows[key], profile)['direction'] == 'negative'
                                for key in citations if window_semantics.current_founder(rows[key], [a,b]))):
                observation['support_basis'] = 'paired_advisory_reference'
                observation['support_pair_ids'] = [aliases[key] for key in advisory]
            else:
                window_semantics.verify_band(canonical, rows, data['window_s'], profile)
        return observation
    return ask(model,counts,emit,audit,observation_data,tool,instruction,validate_observation,compact_history=compact_history)


def choose_action(model, counts, emit, audit, window, obs, active, candidate, transition=None, followup=False,
                  *, corroborated_onset=None, compact_history=False):
    a,b = window
    entity = active or candidate
    ident = entity['event_id'] if entity else None
    if active:
        actions=['CONTINUE_EVENT','HOLD_EVENT','REQUEST_CLOSE','REFINE_ONSET','NO_EVENT_CHANGE']
    elif candidate:
        actions=['CONFIRM_EVENT','REFINE_ONSET','HOLD_EVENT','DISMISS_CANDIDATE','NO_EVENT_CHANGE']
    else:
        actions=['CREATE_CANDIDATE','CONFIRM_EVENT','NO_EVENT_CHANGE']
    if followup:
        if obs['interpretation']!='support':
            actions=[v for v in actions if v not in ('CREATE_CANDIDATE','CONFIRM_EVENT','CONTINUE_EVENT')]
        elif active:
            actions.remove('REQUEST_CLOSE')
    if corroborated_onset is False and 'CONFIRM_EVENT' in actions:
        actions.remove('CONFIRM_EVENT')
    def lifecycle(args):
        action=args['action']
        if action not in actions or args['event_id']!=ident or (ident is not None and type(args['event_id']) is not int):
            raise ValueError(f'Select one of {actions} and use event_id={ident}')
        if action in ('CONFIRM_EVENT','CREATE_CANDIDATE') and obs['interpretation']!='support':
            raise ValueError('Registering an episode requires a supported observation')
        if action=='REQUEST_CLOSE' and obs['interpretation']!='negative' and not (transition and obs['assessment']=='UNCERTAIN'):
            raise ValueError('Request closure when current evidence weighs against both subtypes')
        return dict(status='accepted',**args)
    tool=schema('choose_event_action','Update the episode ledger independently of its boundary.',
                dict(action=choice(actions),event_id={'enum':[ident]},reason=TEXT),['action','event_id','reason'])
    data = dict(window_s=[a,b],assessment=obs,open_event=active,candidate=candidate,event_id=ident)
    instruction = ('CREATE_CANDIDATE records a provisional finding. CONFIRM_EVENT registers an episode now, even on first detection. '
                   'You may hold or leave the ledger unchanged. CONFIRM_EVENT never requires a known onset. REQUEST_CLOSE starts an end investigation.')
    if transition is not None:
        data.update(transition)
        instruction += (' Compare the earlier and current evidence about the combined target. For an open event, '
                        'choose REQUEST_CLOSE when a change from target-present to target-absent warrants '
                        'checking its end, even if the exact boundary remains unknown. This requests an investigation. '
                        'An accepted pair of before/after measurements is still required to close. '
                        'CONTINUE_EVENT means evidence of ongoing target, HOLD_EVENT means unresolved. '
                        'A prior open event itself provides no current rhythm evidence.')
    if followup and active:
        instruction += (' A single contrary window may start an end review. If support recovers, '
                        'continue the same event and withdraw the suspicion. If the contrary direction '
                        'persists, REQUEST_CLOSE lets you inspect and confirm the earlier boundary. '
                        'If the evidence remains ambiguous, hold the event.')
    if corroborated_onset is not None and not active:
        data['onset_corroborated'] = corroborated_onset
        instruction += (' A first supportive reading can be registered as a candidate. To confirm the episode, '
                        'seek corroboration from another lead now or from an adjacent observation on the same lead.')
    return ask(model,counts,emit,audit,data,tool,instruction,lifecycle,compact_history=compact_history)


def acquire_from_menu(reviewer, state, session, call_window, emit, audit):
    while len(session.calls) < session.max_calls:
        if reviewer.focused_evidence_menu and active_acquisition.decisive_pair(
                session.calls, state['window_s'], reviewer.window_band_profile['edges']):
            emit('llm_steps', dict(audit, phase='select_evidence', status='concordant_evidence_stop'))
            return
        options = active_acquisition.measurement_menu(state, session.calls, live_scoring=reviewer.live_scoring)
        if not options:
            return
        # Summaries expose measured values and availability without repeating raw waveform proxies.
        observations = [dict(tool=row['tool'], lead=row['lead'], window_s=row['window_s'],
                             status=row['status'], fields=measured_event.numeric_fields(row))
                        for row in session.calls]
        if reviewer.focused_evidence_menu:
            for summary, row in zip(observations, session.calls, strict=True):
                if row['tool'] == 'assess_quality':
                    summary['artifact_flags'] = active_acquisition.quality_flags(row['result'])
        pending = state.get('pending_question')
        data = dict(window_s=state['window_s'], open_event=state['open_event'] is not None,
                    candidate=state['candidate'] is not None, pending=pending and pending['question'],
                    measurements=observations, options=options, remaining=session.max_calls-len(session.calls))
        tool = schema('select_evidence', 'Choose an unmeasured observation or finish.',
                      dict(option=choice([*options, 'finish']), reason=TEXT), ['option', 'reason'])

        def select(args):
            if args['option'] not in options and args['option'] != 'finish':
                raise ValueError('Choose one offered option')
            return args

        instruction = ('Take the option that would most change your next event decision. For a possible onset or end, '
                       'compare the unmeasured lead on the current interval first if it can corroborate or contradict '
                       'the change. A single high score can create a candidate only. A signal quality check can help '
                       'identify an artifact. A shifted score is useful near a boundary, but can share samples with '
                       'another measurement. Finish if the current measurements already answer the question.')
        if reviewer.focused_evidence_menu:
            instruction += (' When co-timed readings conflict and artifact is plausible, quality checks can help '
                            'describe the signal. A quality flag is about signal integrity. '
                            'An unflagged regular RR cannot by itself rule out atrial flutter.')
        selected = ask(reviewer.model, reviewer.counts, emit, audit, data, tool, instruction, select)
        if selected.get('status') == 'error' or selected['option'] == 'finish':
            return
        args = options[selected['option']].copy()
        name = args.pop('tool')
        row = call_window(name, args, trigger_reason='adaptive_selected')
        emit('llm_steps', dict(audit, phase='select_evidence', status='observed', action=name,
                               arguments=args, option=selected['option'], evidence_id=row['evidence_id']))


def transition_context(data, profile):
    if profile is None:
        return None
    rhythm=[]
    for row in data['measurements']:
        if not row.get('window_s') or not window_semantics.current_founder(row, [row['window_s'][0],data['cursor_s']]):
            continue
        rhythm.append(dict(id=row['id'], window_s=row['window_s'], lead=row['lead'],
                           target_score=max(row['fields'][k] for k in measured_event.PAIR),
                           reference=window_semantics.band(row,profile)))
    return dict(rhythm_by_time=sorted(rhythm,key=lambda r:(r['window_s'],r['lead'],r['id'])),
                pending_question=data.get('pending_question'))


def retain_end_evidence(state, transition, aliases, ledger, observation):
    window = state['window_s']
    current = []
    for row in transition['rhythm_by_time']:
        if row['window_s'] != window:
            continue
        key = aliases[row['id']]
        if ledger.get(key)['available_at_s'] <= window[1]:
            current.append(dict(evidence_id=key, window_s=window, lead=row['lead'],
                                direction=row['reference']['direction'], target_score=row['target_score']))
    if observation.get('support_basis') == 'paired_advisory_reference':
        # The validated interpretation belongs to the pair. Preserve the actual band on each lead.
        paired_ids = observation['support_pair_ids']
        if len(current) != 2 or set(paired_ids) != {row['evidence_id'] for row in current}:
            raise ValueError('Both arrived measurements must be present for an advisory support anchor')
        current = [dict(current[0], direction='support', basis='paired_advisory_reference',
                        measurements=current)]
    memory = advance_end_memory(state.get('end_review_memory', {}), window, current)
    by_id = {key: alias for alias, key in aliases.items()}
    proof = {}
    for role, field in (('before', 'support'), ('first_against', 'first_against'),
                        ('later_against', 'later_against')):
        item = memory[field]
        proof[role] = None
        if item:
            alias = by_id.get(item['evidence_id'], 'event_' + role)
            aliases[alias] = item['evidence_id']
            proof[role] = dict(id=alias, window_s=item['window_s'], lead=item['lead'],
                               direction=item['direction'], target_score=item['target_score'])
            if item.get('basis') == 'paired_advisory_reference':
                proof[role]['basis'] = item['basis']
                proof[role]['measurements'] = []
                for index, row in enumerate(item['measurements']):
                    key = row['evidence_id']
                    short = alias if key == item['evidence_id'] else by_id.get(key, f'event_{role}_{index+1}')
                    aliases[short] = key
                    proof[role]['measurements'].append(dict(id=short, lead=row['lead'],
                        target_score=row['target_score'], reference_direction=row['direction']))
    return memory, dict(proof, can_confirm=bool(proof['later_against']))


def localize_onset(reviewer, state, transition, aliases, emit, audit):
    active = state['open_event']
    causal = [row for row in transition['rhythm_by_time'] if row['id'] in aliases]
    options = onset_pairs(causal, state['window_s'], state['last_closed_end_s'],
                          active['confirmed_at_s'] if active else None)
    if not options:
        return dict(status='unlocalized', reason='No arrived measurements bracket the onset.')
    tool = schema('select_onset_boundary', 'Select a measured onset bracket or leave it unlocalized.',
                  dict(option=choice([*options, 'unlocalized']), reason=TEXT), ['option', 'reason'])

    def select(args):
        option, reason = args['option'], args['reason']
        if option == 'unlocalized':
            return dict(status='unlocalized', reason=reason)
        if option not in options:
            raise ValueError('Select a displayed event bracket')
        pair = options[option]
        return dict(status='located', boundary=dict(kind='bracketed', range_s=pair['range_s']),
                    before_id=aliases[pair['before']['id']], after_id=aliases[pair['after']['id']], reason=reason)

    return ask(reviewer.model, reviewer.counts, emit, audit,
               dict(window_s=state['window_s'], options=options), tool,
               'Select the pair that best localizes the onset of AF or AFL. Each pair has non-overlapping '
               'arrived support and contrary windows. A negative-before/positive-after pair on the same lead '
               'spanning the clearest recent change can help localize onset. Check earlier competing evidence '
               'and select unlocalized if the measured change does not localize this episode. '
               'Only the bracket is grounded. Its midpoint is an estimate.', select)



def submit(reviewer, state, aliases, ledger, session, emit, audit, call_window, tools, *, stable=None):
    data=json.loads(measured_event.final_context(state,aliases,ledger)[1]['content'])
    if reviewer.focused_evidence_menu:
        for row in data['measurements']:
            if row['tool'] == 'assess_quality':
                row['artifact_flags'] = active_acquisition.quality_flags(ledger.get(aliases[row['id']])['result'])
    rows={r['id']:r for r in data['measurements']}
    a,b=state['window_s']
    def usable(key):
        r=rows.get(key,{})
        return (r.get('direction_allowed') and all(type(r['fields'].get(k)) in (int,float) and isfinite(r['fields'][k]) for k in measured_event.PAIR)
                and r['window_s'][1]<=b and ledger.get(aliases[key])['available_at_s']<=b)
    if stable is None:
        obs = assess(reviewer.model, reviewer.counts, emit, audit, data, aliases, usable,
                     lambda key: ledger.get(aliases[key])['available_at_s'] <= b,
                     getattr(reviewer, 'window_assessment_mode', 'legacy'), getattr(reviewer, 'window_band_profile', None),
                     reviewer.compare_rhythm_leads, reviewer.bound_window_choices, reviewer.direction_only_reference,
                     advisory_reference=reviewer.advisory_reference, compact_history=reviewer.compact_history)
    else:
        direction, _ = stable
        key = state['already_acquired_evidence'][0]['evidence_id']
        obs = dict(status='accepted', interpretation=direction, assessment=ASSESSMENTS[direction],
                   evidence_ids=[key], directional_ids=[key],
                   support_intervals=[[a,b]] if direction == 'support' else [],
                   opposing_intervals=[[a,b]] if direction == 'negative' else [],
                   reason='The confident primary measurement agrees with the current event state.')
    if obs['status']=='error':
        return dict(action='ACTION_ERROR',decision='ACTION_ERROR',evidence_ids=[],reason=obs['reason'],window_assessment=obs)
    emit('assessments',dict(audit,stage='window',window_s=[a,b],**obs))
    active=state['open_event']
    candidate=state['candidate']
    profile = reviewer.window_band_profile if getattr(reviewer,'window_assessment_mode','legacy')=='union_reference' else None
    transition = transition_context(data,profile)
    followup = reviewer.confirm_end_with_followup
    memory = None
    if reviewer.persistent_event_evidence:
        memory, proof = retain_end_evidence(state, transition, aliases, ledger, obs)
    if followup and active:
        transition['end_evidence'] = proof if memory is not None else end_evidence(
            transition['rhythm_by_time'], [a,b], active)
    corroborated = None
    if reviewer.evidence_menu and not active:
        corroborated = (obs.get('support_basis') == 'paired_advisory_reference'
                        or active_acquisition.onset_supported(transition['rhythm_by_time'], [a,b]))
    intent = (choose_action(reviewer.model, reviewer.counts, emit, audit, [a,b], obs, active, candidate, transition, followup,
                            corroborated_onset=corroborated, compact_history=reviewer.compact_history)
              if stable is None else dict(status='accepted', action=stable[1], event_id=active['event_id'] if active else None,
                                          reason=obs['reason']))
    emit('llm_steps',dict(audit,stage='lifecycle',status='lifecycle_result',intent=intent))
    action=intent.get('action')
    refinement=dict(status='not_requested')
    if followup and action=='REQUEST_CLOSE':
        proof=transition['end_evidence']
        if proof['can_confirm'] and obs['interpretation']!='conflict':
            def confirm(args):
                if args['decision']=='WAIT':
                    return dict(status='unlocalized',reason=args['reason'])
                if args['decision']!='CONFIRM_END':
                    raise ValueError('Choose WAIT or CONFIRM_END')
                left, first, last=(proof[k] for k in ('before','first_against','later_against'))
                result = dict(status='located',reason=args['reason'],
                            boundary=dict(kind='bracketed',range_s=[left['window_s'][0],first['window_s'][1]]),
                            before_id=aliases[left['id']],after_id=aliases[first['id']],
                            confirmation_evidence_id=aliases[last['id']])
                if left.get('measurements'):
                    result['support_pair_ids'] = [aliases[row['id']] for row in left['measurements']]
                return result
            tool=schema('confirm_event_end','Decide whether later evidence confirms the suspected end.',
                        dict(decision=choice(['CONFIRM_END','WAIT']),reason=TEXT),['decision','reason'])
            refinement=ask(reviewer.model,reviewer.counts,emit,audit,
                           dict(window_s=[a,b],open_event=active,assessment=obs,**transition),tool,
                           'Decide if the combined AF or AFL event has ended. Compare the last supported '
                           'interval, first contrary interval and later confirmation. Confirm if these '
                           'measurements support a sustained change. WAIT if recovery, poor reliability '
                           'or conflicting evidence makes the change doubtful. Confirmation time is now; '
                           'the boundary uses the earlier change, even when the exact second is unknown.',confirm,
                           compact_history=reviewer.compact_history)
        else:
            refinement=dict(status='unlocalized',reason='No current pair of successive contrary intervals after observed support.')
    elif reviewer.boundary_pair_menu and action in ('CREATE_CANDIDATE','CONFIRM_EVENT','REFINE_ONSET'):
        refinement = localize_onset(reviewer, state, transition, aliases, emit, audit)
    elif action in ('CREATE_CANDIDATE','CONFIRM_EVENT','REFINE_ONSET','REQUEST_CLOSE'):
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
                if profile is not None:
                    if window_semantics.band(rows[key],profile)['direction'] != direction:
                        raise ValueError(f'{key} does not provide the required measured direction')
                elif known and known != direction:
                    raise ValueError(f'{key} has an opposite previously committed rhythm judgment')
            return dict(status='located',boundary=dict(kind='bracketed',range_s=[x,w]),
                        before_id=aliases[left],after_id=aliases[right],reason=args['reason'])
        props=dict(status=choice(['unlocalized','located']),before_id=TEXT,after_id=TEXT,reason=TEXT)
        tool=schema('refine_boundary','Name ordered measurements on opposite sides or leave the boundary unlocalized.',
                    props,['status','reason'])
        extras = [] if reviewer.evidence_menu else [
            t for t in tools if t['function']['name'] in split_event.TOOL_GUIDE and len(session.calls)<session.max_calls
        ]
        def measure(name,args):
            parsed=session.registry._tools[name].arguments.model_validate(args)
            if not session.context.origin_s<=parsed.start_s<parsed.end_s<=b:
                raise ValueError('Outside retained arrived interval')
            row=call_window(name,parsed.model_dump())
            aliases['B1']=row['evidence_id']
            rows['B1']=dict(id='B1',tool=name,status=row['status'],window_s=row['window_s'],lead=row['lead'],fields=measured_event.numeric_fields(row),
                            direction_allowed=name==measured_event.FOUNDER and row['status']=='ok')
            return dict(id='B1',**row)
        direction='before supports either subtype and after weighs against both' if action=='REQUEST_CLOSE' else 'before weighs against both and after supports either'
        boundary_data=dict(window_s=[a,b],event_action=action,open_event=active,measurements=list(rows.values()))
        if transition is not None:
            boundary_data.update(transition)
        instruction = ('Localize only if '+direction+'. You may measure one retained interval. '
                       'Use unlocalized if no pair supports the change. The ledger already saved the observation and lifecycle intent.')
        if reviewer.end_range_prompt and action=='REQUEST_CLOSE':
            instruction += (' Here located means a TIME RANGE between two observed sides of a plausible '
                            'end, even when its exact second is unknown. Before and after refer to the '
                            'combined rhythm target: present earlier, absent later. Consider temporary '
                            'score dips, conflicting leads or a return of support before claiming an end. '
                            'If a change is supported, select its closest credible ordered before_id and '
                            'after_id; the runtime records the bracket. If the same event might plausibly '
                            'still be in progress, use unlocalized and name what prevents localization.')
        refinement=ask(reviewer.model,reviewer.counts,emit,audit,boundary_data,
                       tool,instruction,refine,extras,measure)
    payload=dict(**obs,window_assessment=obs,lifecycle=intent,refinement=refinement,
                 event_action=action or 'LIFECYCLE_ERROR',action=action or 'LIFECYCLE_ERROR',
                 boundary=refinement.get('boundary'),pending=None,decision_source='llm_staged')
    if stable:
        payload['decision_source'] = 'stable_primary'
    if memory is not None:
        payload['end_review_memory'] = memory
    if reviewer.focused_evidence_menu:
        payload['supported_tail_export'] = True
    if transition is not None:
        prior = state.get('pending_question')
        if action=='REQUEST_CLOSE' and refinement['status']!='located':
            payload['pending'] = dict(question='Did the combined target end? Identify a measured before/after pair.',
                                      needed_evidence='Earlier target support and later contrary evidence on arrived ECG')
        elif prior and not (action=='REQUEST_CLOSE' or (action=='CONTINUE_EVENT' and obs['interpretation']=='support')):
            payload['pending'] = {k:prior[k] for k in ('question','needed_evidence') if k in prior}
        if prior:
            payload['pending_resolution'] = 'still_pending' if payload['pending'] else ('resolved' if action=='REQUEST_CLOSE' else 'withdrawn')
    payload.update(split_event.appointment(reviewer.model,state,payload,reviewer.counts,emit,audit))
    return payload

