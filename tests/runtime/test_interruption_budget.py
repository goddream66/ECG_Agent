from unittest import TestCase

from cardioagent.ecg_agent.experiments.interruption_budget import choose_window, read_wave
from tests.runtime.test_brief_review import Model
from tests.runtime.test_react import reply


class MeasureBudgetTests(TestCase):
    case = dict(record_id='12', cursor=100, eligible=[2,4,5],
                history=[[s,s+10,[.9,.1 if s==80 else .8]] for s in range(40,100,10)])

    def test_selects_arrived_window(self):
        model = Model([reply('measure_one_window',dict(candidate='W3',reason='Verify ECG2 decrease.'))])
        decision,counts=choose_window(self.case,model,lambda *args:None)
        self.assertEqual(decision['interval_s'],[80,90])
        self.assertEqual(counts['llm_calls'],1)

    def test_rejects_future_and_offline_truth(self):
        for row in (dict(self.case,cursor=90),dict(self.case,reference='AF')):
            with self.assertRaises(ValueError):
                choose_window(row,None,None)
        with self.assertRaises(ValueError):
            read_wave(None,'12',80,90,85)

    def test_bounded_retry_rejects_ineligible(self):
        invalid=reply('measure_one_window',dict(candidate='W2',reason='na'))
        decision,counts=choose_window(self.case,Model([invalid,invalid]),lambda *args:None)
        self.assertEqual(decision['status'],'error')
        self.assertEqual(counts['llm_calls'],2)
