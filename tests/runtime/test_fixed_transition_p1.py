"""Causal views and short evidence IDs for the fixed transition probe."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'ops/server/fixed_transition_p1.py'
spec = importlib.util.spec_from_file_location('fixed_transition_p1', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Model:
    def __init__(self, decision='TRANSITION'):
        self.decision = decision
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append(messages)
        names = [t['function']['name'] for t in tools]
        if 'retrieve_prior_evidence' in names:
            name = 'retrieve_prior_evidence'
            args = {'focus':'last_support','reason':'inspect prior evidence'}
        else:
            name = 'judge_transition'
            args = {'decision':self.decision,'evidence_ids':['E1'],
                    'reason':'observed change'}
        return type('Reply',(),{'message':{'tool_calls':[{'id':'call1','function':{
            'name':name,'arguments':json.dumps(args)}}]},'usage':{}})()


def evidence(start, score, raw_id):
    return {'evidence_id':raw_id,'tool':'score_ecgfounder','lead':'ECG1',
            'window_s':[start,start+10],'available_at_s':start+10,
            'status':'ok','result':{'af_score':score,'afl_score':0.}}


class FixedTransitionTests(unittest.TestCase):
    def setUp(self):
        self.trace=[{'start_s':i*10,'end_s':(i+1)*10,
                     'af_score':.9 if i<5 else .1,'afl_score':0.,
                     'decision':'support' if i<5 else 'negative',
                     'state':'ACTIVE','llm_action':{'decision':'CLOSE'}}
                    for i in range(8)]
        self.evidence=[evidence(i*10,.9 if i<5 else .1,f'raw{i}')
                       for i in range(8)]
        self.evidence.append(evidence(80,.9,'future'))

    def test_view_excludes_future_and_hides_prior_llm_action(self):
        recent,visible,older,aliases=module.view(self.trace,self.evidence,80)
        self.assertEqual(len(recent),6)
        self.assertEqual(len(visible),6)
        self.assertEqual(len(older),2)
        self.assertNotIn('future',aliases.values())
        m=Model()
        result=module.judge(m,'A',self.trace,self.evidence,80)
        self.assertEqual(result['answer']['resolved_evidence_ids'],['raw2'])
        text=json.dumps(m.messages[0])
        self.assertNotIn('llm_action',text)
        self.assertNotIn('future',text)

    def test_structured_uses_same_six_windows_and_retrieval_older_only(self):
        m=Model()
        result=module.judge(m,'C',self.trace,self.evidence,80)
        self.assertEqual(result['retrieved_count'],2)
        self.assertEqual(result['answer']['resolved_evidence_ids'],['raw2'])
        self.assertNotIn('future',json.dumps(m.messages[-1]))
        recent,visible,older,aliases=module.view(self.trace,self.evidence,80)
        self.assertEqual(module.structured(recent,visible,aliases)['consecutive_negative_s'],[50,80])

    def test_rule_requires_two_consecutive_negative_windows(self):
        self.assertEqual(module.rule(self.trace,80),'TRANSITION')
        changed=[dict(w) for w in self.trace]
        changed[-2]['decision']='conflict'
        self.assertEqual(module.rule(changed,80),'UNCERTAIN')


if __name__=='__main__':unittest.main()
