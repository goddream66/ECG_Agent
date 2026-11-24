import copy
import json
import unittest
from collections import Counter

from cardioagent.ecg_agent.harness.compact_history import TABLE_GUIDE, table_view
from cardioagent.ecg_agent.harness.staged_review import ask, schema, TEXT
from cardioagent.ecg_agent.harness.measured_event import numeric_fields
from tests.runtime.test_react import reply, ScriptModel


class CompactHistoryTests(unittest.TestCase):
    def test_runtime_fluctuations_leave_physio_evidence_and_original_cost_intact(self):
        for tool in ('assess_quality','detect_beats'):
            record = dict(tool=tool,result=dict(duration_ms=1.2306,score=.89,beat=dict(duration_ms=100)))
            a = numeric_fields(record)
            record['result']['duration_ms'] = 1.2316
            self.assertEqual(a,numeric_fields(record))
            self.assertEqual(a,{'score':.89,'beat.duration_ms':100})
            self.assertEqual(record['result']['duration_ms'],1.2316)
        self.assertEqual(numeric_fields(dict(tool='other_measurement',result=dict(duration_ms=120))),
                         {'duration_ms':120})

    def test_prompt_retains_order_timing_measurements_and_ledger_attribution(self):
        rows = [dict(id=f'H{i}', tool='score_ecgfounder', lead=f'ECG{1+i%2}', window_s=[10*i,10*i+10],
                     status='ok', warnings=['Scores are uncalibrated'], direction_allowed=True,
                     fields={'target_score': .4999999 if i == 3 else .8001},
                     overlapping_quality_ids=['Q1'] if i == 3 else []) for i in range(8)]
        data = dict(window_s=[80,90], earlier_measurements_for_comparison_only=rows,
                    vote_on_this_interval=[dict(rows[0], id='E1', window_s=[80,90])])
        original = copy.deepcopy(data)
        model = ScriptModel([reply('read',dict(reason='read'))])
        ask(model, Counter(), lambda *_: None, {}, data, schema('read','read',dict(reason=TEXT),['reason']),
            'Use current evidence.', lambda args: args, compact_history=True)
        messages = model.inputs[0]
        self.assertIn(TABLE_GUIDE, messages[0]['content'])
        actual = json.loads(messages[1]['content'])
        table = actual['earlier_measurements_for_comparison_only']
        recovered = [dict(table['shared'], **dict(zip(table['columns'], values, strict=True)))
                     for values in table['rows']]
        self.assertEqual(recovered, rows)
        self.assertEqual(actual['vote_on_this_interval'], data['vote_on_this_interval'])
        self.assertEqual(original, data)

    def test_mixed_and_short_history_preserve_original_representation(self):
        mixed = [dict(id='a', score=.8), dict(id='b', status='failed'), dict(id='c', score=None)]
        for timeline in (mixed, mixed[:1], []):
            self.assertIs(table_view(dict(rhythm_by_time=timeline))['rhythm_by_time'], timeline)
