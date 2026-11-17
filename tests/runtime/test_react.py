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
