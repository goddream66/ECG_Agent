"""ReAct integration tests with real ToolSession and deterministic fake model responses."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import numpy as np
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.schemas import ToolContext, EvidenceContext
from cardioagent.ecg_agent.harness import ToolSession
from cardioagent.ecg_agent.harness.logging import JsonLogs
from cardioagent.ecg_agent.harness.react import ReActReview
from cardioagent.ecg_agent.llm.contracts import ModelReply
from cardioagent.ecg_agent.llm.providers import EndpointModel, parse_qwen_tool_reply
from cardioagent.ecg_agent.memory.events import EventTracker
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.experiments.runner import configured_variants


def reply(name, args):
    return ModelReply({'role': 'assistant', 'tool_calls': [{'id': 'a', 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(args)}}]}, {'prompt_tokens': 10})


class ScriptModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []

    def complete(self, messages, tools):
        self.inputs.append(deepcopy(messages))
        return next(self.actions)


class ReActTests(unittest.TestCase):
    def setup_review(self, actions, budget=4, max_rounds=5, identity='record1'):
        t = np.arange(2500) / 250
        values = np.column_stack([np.sin(2*np.pi*8*t)]*2)
        session = ToolSession(ToolContext(Waveform(values, 250, ('ECG1','ECG2'), ('mV','mV')),0,10),
            cached_model=lambda lead,a,b:(np.ones(5000,dtype=np.float32),lambda x:(.9,.1)), max_calls=1+budget,
            evidence_context=EvidenceContext('run','task',identity))
        primary=session.call('score_ecgfounder', {'lead':'ECG1','start_s':0.,'end_s':10.})
        model=ScriptModel(actions); review=ReActReview(model,max_rounds=max_rounds)
        logs=[]
        def call(name,lead):
            return session.call(name,{'lead':lead,'start_s':0.,'end_s':10.})
        def run():
            return review.review(call,primary,'ECG2',EventTracker(),EventConfig(review_budget=max(1,budget)),
                                 session,lambda name,row:logs.append(deepcopy(row)))
        return review,session,model,logs,run

    def test_missing_quality_stays_unknown_without_hidden_calls(self):
        r,s,m,logs,run=self.setup_review([reply('finish_review',{'reason':'wait','evidence_ids':[]})])
        self.assertEqual(run()[0],'unavailable')
        self.assertEqual(len(s.calls),1)
        self.assertEqual(logs[-1]['stop_reason'],'model_finish')

    def test_quality_call_then_finish_uses_real_observation(self):
        args={'lead':'ECG1','start_s':0.,'end_s':10.}
        r,s,m,logs,run=self.setup_review([reply('assess_quality',args),reply('finish_review',{
            'reason':'obtained quality','evidence_ids':[]})])
        run()
        self.assertEqual([x['tool'] for x in s.calls],['score_ecgfounder','assess_quality'])
        tool_msg=next(x for x in m.inputs[1] if x['role']=='tool')
        self.assertEqual(json.loads(tool_msg['content'])['evidence_id'],s.calls[-1]['evidence_id'])
        self.assertEqual(r.counts['llm_calls'],2)
        self.assertEqual(r.counts['llm_prompt_tokens'],20)

    def test_future_window_never_executes(self):
        r,s,m,logs,run=self.setup_review([reply('score_ecgfounder',{
            'lead':'ECG2','start_s':10.,'end_s':20.})])
        self.assertEqual(run()[0],'unavailable');self.assertEqual(len(s.calls),1)
        self.assertEqual(logs[-1]['stop_reason'],'model_or_action_error')

    def test_unrequested_lead_remains_absent(self):
        r,s,m,logs,run=self.setup_review([reply('finish_review',{'reason':'stop','evidence_ids':[]})])
        run(); self.assertFalse(any(x['lead']=='ECG2' for x in s.calls))
        self.assertNotIn('secondary_evidence',json.loads(m.inputs[0][1]['content']))

    def test_forbidden_tool_and_forged_citation(self):
        for item in [reply('read_annotations',{}),reply('finish_review',{
            'reason':'invented','evidence_ids':['not-observed']})]:
            with self.subTest(item=item):
                r,s,m,logs,run=self.setup_review([item])
                self.assertEqual(run()[0],'unavailable');self.assertEqual(len(s.calls),1)

    def test_budget_prevents_more_tool_execution(self):
        r,s,m,logs,run=self.setup_review([reply('assess_quality',{
            'lead':'ECG1','start_s':0.,'end_s':10.})],budget=0)
        self.assertEqual(run()[0],'unavailable');self.assertEqual(len(s.calls),1)
        self.assertEqual(logs[-1]['stop_reason'],'tool_budget')

    def test_repeated_calls_are_cached_and_round_limited(self):
        action=reply('score_ecgfounder',{'lead':'ECG1','start_s':0.,'end_s':10.})
        r,s,m,logs,run=self.setup_review([action]*3,max_rounds=2)
        run();self.assertEqual(r.counts['llm_calls'],2)
        self.assertEqual(len(s.calls),3);self.assertTrue(s.calls[-1]['cached'])
        self.assertEqual(logs[-1]['stop_reason'],'round_limit')

    def test_backend_and_malformed_response_fail_to_unknown(self):
        for item in [ModelReply({'role':'assistant','content':'normal'}),
                     ModelReply({'role':'assistant','tool_calls':[{},{}]})]:
            r,s,m,logs,run=self.setup_review([item]);self.assertEqual(run()[0],'unavailable')
        r,s,m,logs,run=self.setup_review([]);self.assertEqual(run()[0],'unavailable')

    def test_model_cannot_mutate_tool_evidence(self):
        r,s,m,logs,run=self.setup_review([])
        def bad(messages,tools):
            messages.clear();tools.clear();raise RuntimeError('offline')
        m.complete=bad; before=deepcopy(s.calls)
        run();self.assertEqual(s.calls,before)

    def test_memory_namespace_and_bounds(self):
        r,s,m,logs,run=self.setup_review([reply('finish_review',{'reason':'wait','evidence_ids':[]})])
        run();self.assertEqual(len(r.history),1)
        s.evidence_context=EvidenceContext('run','task','another')
        with self.assertRaises(ValueError):run()
        self.assertNotIn('answer',json.loads(m.inputs[0][1]['content']))

    def test_qwen_native_parser(self):
        raw='<tool_call>{"name":"finish_review","arguments":{"reason":"done","evidence_ids":[]}}</tool_call>'
        self.assertEqual(parse_qwen_tool_reply(raw)['tool_calls'][0]['function']['name'],'finish_review')
        for invalid in ['normal ECG',raw+raw,'<tool_call>{}</tool_call>']:
            with self.assertRaises(ValueError):parse_qwen_tool_reply(invalid)

    def test_endpoint_protocol(self):
        context=MagicMock(); context.__enter__.return_value.read.return_value=json.dumps({
            'choices':[{'message':reply('finish_review',{}).message}],'usage':{'completion_tokens':3}}).encode()
        with patch('cardioagent.ecg_agent.llm.providers.urlopen',return_value=context) as request:
            response=EndpointModel({'model':'Qwen3-8B','base_url':'http://localhost:8000/v1'}).complete([],[])
        payload=json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['tool_choice'],'required')
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(response.usage['completion_tokens'],3)

    def test_explicit_variant_preserves_old_defaults(self):
        c=EventConfig()
        self.assertEqual(len(configured_variants('live',c)),2)
        self.assertEqual(configured_variants('live',c,['event-react'])[0][2],'event-react')
        with self.assertRaises(ValueError):configured_variants('fixed',c,['event-react'])

    def test_trace_files_are_separate_and_legacy_files_unchanged(self):
        with TemporaryDirectory() as d:
            logs=JsonLogs(d)
            self.assertFalse((Path(d)/'llm_steps.jsonl').exists())
            logs.emit('llm_steps',{'status':'test'});logs.close()
            self.assertEqual(json.loads((Path(d)/'llm_steps.jsonl').read_text())['status'],'test')

    def test_record_runner_integrates_two_windows_and_checkpoint(self):
        from tests.runtime.test_event_checkpoints import EventCheckpointTests
        from cardioagent.ecg_agent.experiments.runner import run_job
        with TemporaryDirectory() as d:
            job = EventCheckpointTests().make_job(Path(d))
            job.update(policy='event-react', llm={'provider':'test'})
            script=[]
            for start in (0.,10.):
                script += [reply('assess_quality',{'lead':'ECG1','start_s':start,'end_s':start+10}),
                           reply('finish_review',{'reason':'reviewed','evidence_ids':[]})]
            model=ScriptModel(script)
            raw=SimpleNamespace(fs=250,sig_name=('ECG1','ECG2'),sig_len=5000,
                                p_signal=np.ones((5000,2)),units=('mV','mV'))
            with patch('cardioagent.ecg_agent.llm.providers.build_chat_model',return_value=model), \
                 patch('cardioagent.ecg_agent.agent.wfdb.rdrecord',return_value=raw):
                result,resumed=run_job(job)
                self.assertFalse(resumed)
                self.assertEqual(result['counts']['llm_calls'],4)
                self.assertEqual(result['counts']['calls_score_ecgfounder'],2)
                self.assertEqual(result['counts']['calls_assess_quality'],2)
                self.assertEqual(result['execution']['actual_model_calls'],0)
                self.assertEqual(len(json.loads(model.inputs[2][1]['content'])['recent_windows']),1)
                before=len(model.inputs)
                self.assertTrue(run_job(job)[1])
                self.assertEqual(len(model.inputs),before)
            folder=Path(job['target'])
            self.assertTrue((folder/'llm_steps.jsonl').exists())
            evidence=[json.loads(x) for x in (folder/'evidence.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['lead']=='ECG1' for x in evidence))
            self.assertEqual(len({x['evidence_id'] for x in evidence}),4)

    def test_boundary_mode_executes_fractional_lookback_and_logs_end_proposal(self):
        from tests.runtime.test_ecg_toolkit import measured

        class BoundaryModel:
            def complete(self, messages, tools):
                state = json.loads(messages[1]['content'])
                assert state['retained_window_s'] == [0, 30]
                results = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
                if not results:
                    return reply('get_beat_timeline_proxy', {'lead': 'ECG2', 'start_s': 18.25, 'end_s': 25.40})
                return reply('finish_review', {'reason': 'local transition cue', 'proposed_end_s': 22.35,
                                               'evidence_ids': [results[0]['evidence_id']]})

        review = ReActReview(BoundaryModel(), boundary_tools=True)
        tracker = EventTracker()
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250, ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32), lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        logs = []
        with patch('cardioagent.ecg_agent.tools.algorithms.measure', side_effect=measured):
            review.review(lambda *_: self.fail('Expected window callback'), primary, 'ECG2', tracker,
                          EventConfig(), session, lambda _, row: logs.append(row), call_window=session.call)
        result = session.calls[1]
        self.assertEqual(result['window_s'], [18.25, 25.40])
        self.assertEqual(result['status'], 'ok')
        proposal = next(row for row in logs if 'proposed_end_s' in row)
        self.assertEqual(proposal['proposed_end_s'], 22.35)
        self.assertEqual(proposal['cursor_s'], 30.)
        self.assertEqual(proposal['evidence_ids'], [result['evidence_id']])
        self.assertEqual(tracker.state, 'ACTIVE')

    def test_qwen_close_event_changes_reported_interval(self):
        from tests.runtime.test_ecg_toolkit import measured

        class ClosingModel:
            def complete(self, messages, tools):
                names = {tool['function']['name'] for tool in tools}
                assert 'end_confirmed_event' in names
                assert 'keep_event_open' in names
                results = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
                if not results:
                    return reply('get_beat_timeline_proxy', {
                        'lead': 'ECG2', 'start_s': 18.25, 'end_s': 25.40})
                return reply('end_confirmed_event', {'event_id': 1, 'end_s': 22.35,
                                             'evidence_ids': [results[0]['evidence_id']],
                                             'reason': 'locally reviewed rhythm transition'})

        review = ReActReview(ClosingModel(), boundary_tools=True, event_control=True)
        tracker = EventTracker(llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32),
                                                       lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        logs = []
        with patch('cardioagent.ecg_agent.tools.algorithms.measure', side_effect=measured):
            decision = review.review(lambda *_: self.fail('Expected window callback'), primary,
                                     'ECG2', tracker, EventConfig(), session,
                                     lambda _, row: logs.append(row), call_window=session.call)
        self.assertEqual(review.pending_close['end_s'], 22.35)
        self.assertEqual(logs[-1]['stop_reason'], 'model_end_confirmed_event')
        observation = tracker.observe(20, 30, *decision[:2],
                                      [x['evidence_id'] for x in session.calls],
                                      close_request=review.pending_close)
        self.assertEqual(tracker.intervals, [[0, 22.35]])
        self.assertEqual(observation['event_update']['kind'], 'llm_close')
        self.assertEqual(tracker.events[0]['end_decided_at_s'], 30)

    def test_qwen_close_event_requires_real_local_boundary_evidence(self):
        model = ScriptModel([reply('end_confirmed_event', {'event_id': 1, 'end_s': 22.35,
                                                   'evidence_ids': ['invented'], 'reason': 'guess'})])
        review = ReActReview(model, boundary_tools=True, event_control=True)
        tracker = EventTracker(llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32),
                                                       lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        decision = review.review(lambda *_: None, primary, 'ECG2', tracker, EventConfig(),
                                 session, lambda *_: None, call_window=session.call)
        self.assertEqual(decision[0], 'unavailable')
        self.assertIsNone(review.pending_close)
        self.assertEqual(tracker.state, 'ACTIVE')

    def test_event_control_can_wait_without_closing(self):
        model = ScriptModel([reply('keep_event_open', {
            'reason': 'more evidence needed ' * 40, 'evidence_ids': []})])
        review = ReActReview(model,
            boundary_tools=True, event_control=True)
        tracker = EventTracker(llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32),
                                                       lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        logs = []
        review.review(lambda *_: None, primary, 'ECG2', tracker, EventConfig(), session,
                      lambda _, row: logs.append(row), call_window=session.call)
        self.assertIsNone(review.pending_close)
        self.assertEqual(logs[-1]['stop_reason'], 'model_keep_event_open')
        self.assertEqual(tracker.state, 'ACTIVE')
        self.assertIn('end_confirmed_event', model.inputs[0][0]['content'])
        self.assertIn('keep_event_open', model.inputs[0][0]['content'])
        self.assertNotIn('Use finish_review when acquired evidence is enough', model.inputs[0][0]['content'])
        self.assertNotIn('event boundaries.', model.inputs[0][0]['content'])

    def test_unified_decision_reads_scheduled_evidence_and_defers(self):
        class DecisionModel(ScriptModel):
            def complete(self, messages, tools):
                self.final_tools = [t['function']['name'] for t in tools]
                return super().complete(messages, tools)

        model = DecisionModel([reply('submit_event_decision', {
            'decision': 'DEFER', 'event_id': 1, 'end_s': None,
            'evidence_ids': ['previous-evidence'], 'reason': 'end unresolved'})])
        review = ReActReview(model, max_rounds=1, boundary_tools=True, event_control=True,
                             unified_event_decision=True)
        review.history = [{'window_s': [10, 20], 'evidence_ids': ['previous-evidence']}]
        tracker = EventTracker(llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32),
                                                       lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        logs = []
        review.review(lambda *_: None, primary, 'ECG2', tracker, EventConfig(),
                      session, lambda _, row: logs.append(row), call_window=session.call)
        state = json.loads(model.inputs[0][1]['content'])
        self.assertEqual(state['already_acquired_evidence'][0]['evidence_id'], primary['evidence_id'])
        self.assertEqual(logs[-1]['stop_reason'], 'model_decision_defer')
        self.assertIsNone(review.pending_close)
        self.assertNotIn('keep_event_open', model.inputs[0][0]['content'])
        self.assertEqual(model.final_tools, ['submit_event_decision'])

    def test_unified_close_uses_local_boundary_evidence(self):
        from tests.runtime.test_ecg_toolkit import measured

        class Model:
            def complete(self, messages, tools):
                seen = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
                if not seen:
                    return reply('get_beat_timeline_proxy', {
                        'lead': 'ECG2', 'start_s': 18.25, 'end_s': 25.40})
                return reply('submit_event_decision', {
                    'decision': 'CLOSE', 'event_id': 1, 'end_s': 22.35,
                    'evidence_ids': [seen[0]['evidence_id']], 'reason': 'local transition'})

        review = ReActReview(Model(), boundary_tools=True, event_control=True,
                             unified_event_decision=True)
        tracker = EventTracker(llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        t = np.arange(7500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 30), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000, dtype=np.float32),
                                                       lambda _: (.9,.1)),
                              evidence_context=EvidenceContext('run','task','record1'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':20.,'end_s':30.})
        with patch('cardioagent.ecg_agent.tools.algorithms.measure', side_effect=measured):
            decision = review.review(lambda *_: None, primary, 'ECG2', tracker, EventConfig(),
                                     session, lambda *_: None, call_window=session.call)
        self.assertEqual(review.pending_close['end_s'], 22.35)
        tracker.observe(20, 30, *decision[:2], [x['evidence_id'] for x in session.calls],
                        close_request=review.pending_close)
        self.assertEqual(tracker.intervals, [[0, 22.35]])

    def test_separate_decision_retries_and_resolves_short_evidence_id(self):
        class DecisionModel(ScriptModel):
            def complete(self, messages, tools):
                self.tool_sets = getattr(self, 'tool_sets', [])
                self.tool_sets.append([t['function']['name'] for t in tools])
                return super().complete(messages, tools)

        model = DecisionModel([
            reply('finish_measurements', {}),
            reply('submit_event_decision', {'decision': 'DEFER', 'event_id': None,
                  'end_s': 10., 'evidence_ids': ['E1'], 'reason': 'uncertain'}),
            reply('submit_event_decision', {'decision': 'DEFER', 'event_id': None,
                  'end_s': None, 'evidence_ids': ['E1'], 'reason': 'uncertain'}),
        ])
        review = ReActReview(model, boundary_tools=True, event_control=True,
                             unified_event_decision=True, separate_final_decision=True)
        t = np.arange(2500) / 250
        wave = Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2), 250,
                        ('ECG1','ECG2'), ('mV','mV'))
        session = ToolSession(ToolContext(wave, 0, 10), allow_experimental=True,
                              cached_model=lambda *_: (np.ones(5000), lambda _:(.9,.1)),
                              evidence_context=EvidenceContext('run','task','record'))
        primary = session.call('score_ecgfounder', {'lead':'ECG1','start_s':0.,'end_s':10.})
        logs=[]
        review.review(lambda *_: None, primary, 'ECG2', EventTracker(llm_closure=True),
                      EventConfig(), session, lambda _,row: logs.append(row),
                      call_window=session.call)
        self.assertEqual(review.counts['llm_calls'], 3)
        self.assertEqual(model.tool_sets[1], ['submit_event_decision'])
        self.assertEqual(model.tool_sets[2], ['submit_event_decision'])
        self.assertIn('active_event_id', model.inputs[1][-1]['content'])
        self.assertIn('Final action rejected', model.inputs[2][-1]['content'])
        self.assertEqual(review.last_action['decision'], 'DEFER')
        self.assertEqual(review.last_action['evidence_ids'], [primary['evidence_id']])
        self.assertEqual(logs[-1]['stop_reason'], 'model_decision_defer')

    def test_separate_decision_failure_is_action_error(self):
        model = ScriptModel([reply('finish_measurements', {}),
            reply('submit_event_decision', {'decision':'DEFER', 'event_id':None,
                  'end_s':10., 'evidence_ids':[], 'reason':'invalid'}),
            reply('submit_event_decision', {'decision':'DEFER', 'event_id':None,
                  'end_s':10., 'evidence_ids':[], 'reason':'invalid'})])
        review = ReActReview(model, boundary_tools=True, event_control=True,
                             unified_event_decision=True, separate_final_decision=True)
        t=np.arange(2500)/250
        wave=Waveform(np.column_stack([np.sin(2*np.pi*8*t)]*2),250,
                      ('ECG1','ECG2'),('mV','mV'))
        session=ToolSession(ToolContext(wave,0,10),allow_experimental=True,
                            cached_model=lambda *_:(np.ones(5000),lambda _:(.9,.1)),
                            evidence_context=EvidenceContext('run','task','record'))
        primary=session.call('score_ecgfounder',{'lead':'ECG1','start_s':0.,'end_s':10.})
        logs=[]
        review.review(lambda *_:None,primary,'ECG2',EventTracker(llm_closure=True),
                      EventConfig(),session,lambda _,row:logs.append(row),call_window=session.call)
        self.assertEqual(review.last_action['decision'],'ACTION_ERROR')
        self.assertEqual(logs[-1]['stop_reason'],'ACTION_ERROR')

    def test_rule_safety_closure_is_attributed_separately(self):
        for fallback, expected in ((False,None),(True,'rule_safety_close')):
            tracker=EventTracker(EventConfig(),llm_closure=True,rule_safety_fallback=fallback)
            tracker.observe(0,10,'support')
            tracker.observe(10,20,'support')
            tracker.observe(20,30,'negative')
            update=tracker.observe(30,40,'negative',evidence_ids=['counter'])
            self.assertEqual(tracker.events[0]['termination_reason'],expected)
            if fallback:
                self.assertEqual(update['event_update']['kind'],'rule_safety_close')
                self.assertEqual(update['event_update']['evidence_ids'],['counter'])

    def test_record_runner_applies_model_close_to_saved_prediction(self):
        from tests.runtime.test_event_checkpoints import EventCheckpointTests
        from cardioagent.ecg_agent.experiments.runner import run_job

        class ControlledReviewer:
            boundary_tools = True
            event_control = True
            live_scoring = False
            counts = {'llm_calls': 2}
            pending_close = None

            def review(self, call, primary, secondary, tracker, config, session, emit, **kwargs):
                if tracker.active is None:
                    self.pending_close = None
                    return 'support', False, 'first_window'
                self.pending_close = {'event_id': tracker.active['event_id'],
                                      'end_s': 12.35, 'evidence_ids': ['verified-boundary'],
                                      'reason': 'local transition'}
                return 'negative', True, 'model_reviewed'

        with TemporaryDirectory() as d:
            job = EventCheckpointTests().make_job(Path(d))
            job.update(policy='event-react', mode='cached',
                       llm={'provider':'test', 'boundary_tools':True, 'event_control':True},
                       config=EventConfig(enter_windows=1).to_dict())
            raw = SimpleNamespace(fs=250, sig_name=('ECG1','ECG2'), sig_len=5000,
                                  p_signal=np.ones((5000,2)), units=('mV','mV'))
            with patch('cardioagent.ecg_agent.llm.providers.build_chat_model', return_value=object()), \
                 patch('cardioagent.ecg_agent.harness.react.ReActReview', return_value=ControlledReviewer()), \
                 patch('cardioagent.ecg_agent.agent.wfdb.rdrecord', return_value=raw):
                result, _ = run_job(job)
            self.assertEqual(result['intervals'], [[0., 12.35]])
            self.assertEqual(result['events'][0]['termination_reason'], 'llm_close')
            updates = [json.loads(line) for line in (Path(job['target'])/'event_updates.jsonl').read_text().splitlines()]
            self.assertEqual(updates[-1]['event_update']['kind'], 'llm_close')

    def test_scheduled_window_evidence_allows_confirmed_event(self):
        from tests.runtime.test_event_checkpoints import EventCheckpointTests
        from cardioagent.ecg_agent.experiments.runner import run_job

        with TemporaryDirectory() as d:
            job = EventCheckpointTests().make_job(Path(d))
            job.update(policy='event-react', mode='cached',
                       llm={'provider':'test', 'boundary_tools':True,
                            'scheduled_evidence':True, 'event_control':True},
                       config=EventConfig(enter_windows=1).to_dict())
            model = ScriptModel([reply('keep_event_open', {'reason':'keep open', 'evidence_ids':[]})]*2)
            t = np.arange(5000) / 250
            raw = SimpleNamespace(fs=250, sig_name=('ECG1','ECG2'), sig_len=5000,
                                  p_signal=np.column_stack([np.sin(2*np.pi*8*t)]*2),
                                  units=('mV','mV'))
            with patch('cardioagent.ecg_agent.llm.providers.build_chat_model', return_value=model), \
                 patch('cardioagent.ecg_agent.agent.wfdb.rdrecord', return_value=raw):
                result, _ = run_job(job)
            self.assertEqual(result['events'][0]['confirmed_at_s'], 10.)
            self.assertTrue(result['events'][0]['right_censored'])
            first_state = json.loads(model.inputs[0][1]['content'])
            self.assertEqual(first_state['scheduled_window_interpretation']['decision'], 'support')
            acquisitions = [json.loads(line) for line in
                            (Path(job['target'])/'acquisitions.jsonl').read_text().splitlines()]
            self.assertIn('scheduled_window_evidence', {row['trigger_reason'] for row in acquisitions})

    def test_boundary_runner_scores_a_fractional_window_without_reading_future(self):
        from tests.runtime.test_event_checkpoints import EventCheckpointTests
        from cardioagent.ecg_agent.experiments import runner

        with TemporaryDirectory() as d:
            job = EventCheckpointTests().make_job(Path(d))
            job.update(mode='live', policy='event-react', llm={'provider': 'test', 'boundary_tools': True})
            model = ScriptModel([
                reply('finish_review', {'reason': 'wait', 'evidence_ids': []}),
                reply('score_ecgfounder', {'lead': 'ECG2', 'start_s': 6.1, 'end_s': 16.1}),
                reply('get_beat_timeline_proxy', {'lead': 'ECG2', 'start_s': 16.25, 'end_s': 21.40}),
            ])
            t = np.arange(5000) / 250
            wave = np.column_stack([np.sin(2*np.pi*8*t)]*2)
            raw = SimpleNamespace(fs=250, sig_name=('ECG1','ECG2'), sig_len=5000,
                                  p_signal=wave, units=('mV','mV'))
            with patch('cardioagent.ecg_agent.llm.providers.build_chat_model', return_value=model), \
                 patch('cardioagent.ecg_agent.agent.wfdb.rdrecord', return_value=raw), \
                 patch.object(runner, '_MODEL', return_value=(.9, .1)) as score:
                result, _ = runner.run_job(job)
            self.assertEqual(score.call_count, 3)
            self.assertEqual(score.call_args.args[0].shape, (5000,))
            evidence = [json.loads(line) for line in (Path(job['target'])/'evidence.jsonl').read_text().splitlines()]
            self.assertEqual(evidence[-1]['window_s'], [6.1, 16.1])
            self.assertEqual(evidence[-1]['available_at_s'], 20.)
            self.assertEqual(result['counts']['tool_calls'], 3)

    def test_history_is_bounded_and_tool_failure_is_unknown(self):
        action=reply('assess_quality',{'lead':'ECG1','start_s':0.,'end_s':10.})
        r,s,m,logs,run=self.setup_review([action,reply('finish_review',{'reason':'done','evidence_ids':[]})])
        s.registry._tools['assess_quality']=replace(s.registry._tools['assess_quality'],
                                                   handler=lambda *a: ({},'error',['failed']))
        # Primary is usable but required QC failed; missing second lead cannot rescue it.
        self.assertEqual(run()[0],'unavailable')
        r.history=[{'dummy':i} for i in range(30)]
        r.history_windows=0
        self.assertEqual(run()[0],'unavailable')
        self.assertEqual(r.history,[])


if __name__=='__main__':unittest.main()
