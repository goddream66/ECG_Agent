from unittest import TestCase

from cardioagent.ecg_agent.harness.offset_localization import estimate_end, review_offset
from tests.runtime.test_brief_review import Model
from tests.runtime.test_react import reply


class OffsetTest(TestCase):
    config={'aligned':{'weights':[0]*8,'intercept':1},'shifted':{'weights':[0]*10,'intercept':-3},
            'min_disagreement':.2}

    def test_future_and_unavailable(self):
        calls=[]
        def acquire(a,b,t):
            calls.append((a,b,t))
            return [None,.5]
        vals=[0,0,.8,.5,.2,.1,0,0]
        self.assertEqual(estimate_end(5,50,69,vals,self.config,acquire)['estimated_end_s'],50)
        self.assertEqual(calls,[])
        result=estimate_end(5,50,70,vals,self.config,acquire)
        self.assertEqual(calls,[(45,55,70)])
        self.assertEqual((result['estimated_end_s'],result['recorded_at_s'],result['requests']),(51,70,2))

    def test_selected_measurement_and_positive_duration(self):
        vals=[.8,.8,.95,.4,.4,.1,0,0]
        call=lambda *args:[.6,.5]
        self.assertEqual(estimate_end(20,50,70,vals,self.config,call)['estimated_end_s'],47)
        self.assertEqual(estimate_end(49,50,70,vals,self.config,call)['estimated_end_s'],50)
        self.assertEqual(estimate_end(20,50,70,[.5]*8,self.config,call)['requests'],0)

    def test_qwen_can_stop_without_scoring_or_request_the_arrived_pair(self):
        for action,n in [('measure_centered_pair',2),('use_aligned',0)]:
            calls=[]
            model=Model([reply('choose_offset_measurement',dict(action=action,reason='Consult timing.'))])
            def measure(a,b,c):
                calls.append((a,b,c))
                return [.55,.6]
            r=review_offset(20,50,70,[.7]*8,self.config,model,measure,lambda *a:None)
            self.assertEqual(r['requests'],n)
            self.assertEqual(calls,[(45,55,70)] if n else [])
            self.assertEqual(r['recorded_at_s'],70)
