import unittest
from unittest.mock import patch
import numpy as np
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext,ToolSession,build_registry

ARGS={'lead':'ECG1','start_s':100.,'end_s':110.}
TOOLS=('detect_beats_featuredb','extract_average_beat_featuredb','assess_template_quality')
def session(unit='mV',missing=False,future=0.,enabled=True,fs=250.):
    t=np.arange(int(20*fs))/fs
    x=np.zeros(len(t))
    for p in np.arange(.5,20,1):x+=np.exp(-((t-p)/.025)**2)
    if unit=='uV':x*=1000
    x[int(10*fs):]+=future
    if missing:x[100]=np.nan
    return ToolSession(ToolContext(Waveform(x[:,None],fs,('ECG1',),(unit,)),100.,110.),allow_experimental=enabled)

class PhysioToolsTests(unittest.TestCase):
    def test_default_gate(self):
        s=session(enabled=False)
        for name in TOOLS:self.assertEqual(s.call(name,ARGS)['error']['code'],'experimental_disabled')
        self.assertTrue(set(TOOLS).isdisjoint(x['name'] for x in build_registry().schemas()))

    def test_real_detector_known_peaks_and_absolute_times(self):
        r=session().call(TOOLS[0],ARGS)
        self.assertEqual(r['status'],'ok',r)
        peaks=np.array(r['result']['r_peaks_s'])
        for p in np.arange(100.5,110.,1.):self.assertLess(np.min(abs(peaks-p)),.04)
        self.assertAlmostEqual(r['result']['rate_from_median_rr_bpm'],60.,delta=1.)
        self.assertTrue(all(100<=p<110 for p in peaks))

    def test_units_equivalent(self):
        for name in TOOLS:
            a=session().call(name,ARGS);b=session('uV').call(name,ARGS)
            self.assertEqual(a['status'],b['status'])
            self.assertEqual(a['result']['r_peaks_samples'],b['result']['r_peaks_samples'])
            if name==TOOLS[1]:np.testing.assert_allclose(a['result']['average_beat_mv'],b['result']['average_beat_mv'],atol=1e-8)

    def test_missing_is_not_filled_or_detected(self):
        with patch('cardioagent.ecg_agent.tools.physio._detect') as detect:
            for name in TOOLS:self.assertEqual(session(missing=True).call(name,ARGS)['status'],'insufficient')
            detect.assert_not_called()

    def test_future_does_not_change_tools(self):
        for name in TOOLS:self.assertEqual(session(future=0).call(name,ARGS)['result'],session(future=999).call(name,ARGS)['result'])

    def test_shared_detector_has_dependency_and_no_duplicate_cost(self):
        s=session()
        with patch('cardioagent.ecg_agent.tools.physio._detect',return_value=np.arange(125,2500,250)) as detect:
            first=s.call(TOOLS[0],ARGS);second=s.call(TOOLS[2],ARGS)
            self.assertEqual(detect.call_count,1)
            self.assertEqual(second['result']['r_peak_evidence_id'],first['evidence_id'])
            self.assertFalse(second['result']['detector_executed'])
            self.assertGreater(second['result']['template_correlation'],.99)

    def test_flat_unknown_units_and_degenerate_templates(self):
        s=session();s.context.waveform.values[:]=0
        r=s.call(TOOLS[2],ARGS);self.assertEqual(r['status'],'insufficient');self.assertIsNone(r['result']['template_correlation'])
        self.assertEqual(session('normalized').call(TOOLS[0],ARGS)['status'],'error')
        with patch('cardioagent.ecg_agent.tools.physio._detect',return_value=np.array([125])):
            self.assertEqual(session().call(TOOLS[2],ARGS)['status'],'insufficient')

    def test_cache_failure_budget_and_future_request(self):
        s=session();a=s.call(TOOLS[0],ARGS);a['result']['r_peaks_samples'].clear()
        b=s.call(TOOLS[0],ARGS);self.assertTrue(b['cached']);self.assertTrue(b['result']['r_peaks_samples'])
        self.assertEqual(s.call(TOOLS[0],ARGS|{'end_s':111.})['error']['code'],'unavailable_window')
        with patch('cardioagent.ecg_agent.tools.physio._detect',side_effect=RuntimeError('detector failed')):
            self.assertEqual(session().call(TOOLS[0],ARGS)['status'],'error')
        s=session();s.max_calls=1;s.call(TOOLS[0],ARGS)
        self.assertEqual(s.call(TOOLS[2],ARGS)['error']['code'],'budget_exhausted')

    def test_average_has_coverage_no_padding(self):
        r=session().call(TOOLS[1],ARGS)
        self.assertEqual(r['status'],'ok');self.assertFalse(r['result']['padding_used'])
        self.assertGreaterEqual(r['result']['used_beats'],2)
        self.assertTrue(np.isfinite(r['result']['average_beat_mv']).all())


    def test_sampling_rates_and_notch_metadata(self):
        for fs in (100.,125.,500.):
            r=session(fs=fs).call(TOOLS[0],ARGS)
            self.assertEqual(r['status'],'ok',r)
            self.assertEqual(r['result']['notch_50hz_applied'],fs>100)
            for p in np.arange(100.5,110.,1.):
                self.assertLess(np.min(abs(np.array(r['result']['r_peaks_s'])-p)),.05)

    def test_prescreen_flags_do_not_veto_template_correlation(self):
        s=session();s.context.waveform.values[:]*=10
        r=s.call(TOOLS[2],ARGS)
        self.assertEqual(r['status'],'ok')
        self.assertTrue(r['result']['prescreen_flags']['peak_to_peak_over_5mv'])
        self.assertTrue(r['result']['template_consistent'])
