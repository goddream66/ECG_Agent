from unittest import TestCase

from cardioagent.ecg_agent.memory.evidence import EvidenceLedger
from cardioagent.ecg_agent.schemas import EvidenceContext


class EvidenceOrderingTests(TestCase):
    def test_namespace_does_not_change_display_order_and_same_window_ties_use_arrival_order(self):
        data=(([20,30],'ECG2'),([10,20],'ECG2'),([10,20],'ECG1'),([10,20],'ECG1'))
        ledgers=[]
        for ids in ('zbca','acbz'):
            ledger=EvidenceLedger(EvidenceContext('run','task','record'))
            for i,((a,b),lead) in enumerate(data):
                ledger.append(dict(evidence_id=ids[i], window_s=[a,b],lead=lead,available_at_s=30,
                                   tool='score_ecgfounder',result=dict(af_score=i/10)))
            order=ledger.presentation_order(set(ids))
            self.assertEqual([ledger.get(k)['result']['af_score'] for k in order],[.2,.3,.1,0.])
            ledgers.append(order)
        self.assertNotEqual(ledgers[0],ledgers[1])
