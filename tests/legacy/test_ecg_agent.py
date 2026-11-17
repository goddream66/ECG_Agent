import unittest
from unittest.mock import patch
import numpy as np
from cardioagent.ecg_agent import Config, ECGAgent
from cardioagent.recording import Waveform


def wave(values=None):
    return Waveform(np.ones((3000,2)) if values is None else values,100.,('II','V5'),('mV','mV'))


def qc(values,fs):
    return {'score':float(values[0]),'reasons':[],'duration_ms':0.}


def measurement(values,fs,start,commit_start,end):
    return {'r_peaks_s':list(np.arange(commit_start+.5,end,1.)), 'rate_bpm':60.,'rr_cv':.01,
            'rr_intervals_s':[1.]*5,'duration_ms':0.,'detector_status':'candidates','detector_error':None}


class ECGAgentTests(unittest.TestCase):
    def test_future_suffix_cannot_change_first_commit(self):
        a=np.ones((6000,2));b=a.copy();b[3000:]*=100
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',qc), patch('cardioagent.ecg_agent.tools.algorithms.measure',measurement):
            x=ECGAgent().run(wave(a))['ledger'][0]
            y=ECGAgent().run(wave(b))['ledger'][0]
        for key in ('rate_bpm','r_peaks_s','trace','lead','status'):self.assertEqual(x[key],y[key])

    def test_bad_primary_uses_alternative_and_budget_is_enforced(self):
        def bad_qc(values,fs):
            return {'score':1.,'reasons':['flat'] if values[0]==1 else [],'duration_ms':0.}
        x=np.ones((3000,2));x[:,1]=2
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',bad_qc),patch('cardioagent.ecg_agent.tools.algorithms.measure',measurement):
            result=ECGAgent().run(wave(x))['ledger'][0]
            limited=ECGAgent(Config(measurement_budget=1)).run(wave(x))['ledger'][0]
        self.assertEqual(result['lead'],'V5');self.assertEqual(result['cost']['measurements'],2)
        self.assertEqual(limited['status'],'review_required');self.assertEqual(limited['cost']['measurements'],1)

    def test_irregularity_triggers_verification_and_conflict_requires_review(self):
        def irregular(values,fs,start,commit_start,end):
            return measurement(values,fs,start,commit_start,end)|{'rr_cv':.3,'rate_bpm':60. if values[0]==1 else 90.}
        x=np.ones((3000,2));x[:,1]=2
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',qc),patch('cardioagent.ecg_agent.tools.algorithms.measure',irregular):
            result=ECGAgent().run(wave(x))['ledger'][0]
        self.assertEqual(result['reason'],'lead_disagreement');self.assertIsNone(result['rate_bpm'])
        self.assertEqual(result['r_peaks_s'],[])

    def test_irregularity_alone_does_not_discard_single_lead(self):
        def irregular(*args):return measurement(*args)|{'rr_cv':.3}
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',qc),patch('cardioagent.ecg_agent.tools.algorithms.measure',irregular):
            result=ECGAgent().run(Waveform(np.ones((3000,1)),100.,('II',),('mV',)))['ledger'][0]
        self.assertEqual(result['status'],'measured')

    def test_flat_signal_requires_review(self):
        result=ECGAgent().run(wave(np.zeros((3000,2))))['ledger'][0]
        self.assertEqual(result['status'],'review_required')
        self.assertIsNone(result['rate_bpm'])

    def test_cursor_and_config_validation(self):
        with self.assertRaises(ValueError):Config(chunk_s=float('nan'))
        with self.assertRaises(ValueError):Config(measurement_budget=0)
        agent=ECGAgent()
        with self.assertRaises(ValueError):agent.step(wave(np.ones((6000,2))))
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',qc),patch('cardioagent.ecg_agent.tools.algorithms.measure',measurement):agent.step(wave())
        with self.assertRaises(ValueError):agent.step(wave())

    def test_bounded_block_stream_matches_prefix_stream(self):
        raw = np.ones((6500,2))
        with patch('cardioagent.ecg_agent.tools.algorithms.inspect',qc),patch('cardioagent.ecg_agent.tools.algorithms.measure',measurement):
            bounded=ECGAgent().run(wave(raw))['ledger']
            agent=ECGAgent()
            for end in (3000,6000,6500):agent.step(wave(raw[:end]))
        for a,b in zip(bounded,agent.ledger):
            for key in ('window_s','rate_bpm','r_peaks_s','trace'):self.assertEqual(a[key],b[key])
