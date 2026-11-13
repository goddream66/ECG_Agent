import unittest
from cardioagent.policy.qc_continuous import distribution
from cardioagent.policy.value import Profile

class ContinuousQCTests(unittest.TestCase):
    def test_subject_replication_does_not_change_distribution(self):
        p=Profile('heart_rate',10,{},None,{}, {})
        from cardioagent.policy.qc_continuous import features
        rows=[{'subject':str(i),'x':features(p,{}),'sample':{'score':i/5}} for i in range(5)]
        a=distribution({'heart_rate/ppg':rows},p,{},'ppg')
        b=distribution({'heart_rate/ppg':rows+rows[:1]*100},p,{},'ppg')
        self.assertEqual(a,b)
    def test_insufficient_subjects_falls_back(self):
        p=Profile('heart_rate',10,{},None,{}, {})
        self.assertIsNone(distribution({},p,{},'ppg'))
