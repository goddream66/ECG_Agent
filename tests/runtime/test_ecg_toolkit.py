from cardioagent.common.paths import code_root
import json
import unittest
from unittest.mock import patch
import numpy as np
from cardioagent.recording import Waveform
from cardioagent.ecg_agent.toolkit import ToolContext, ToolSession, build_registry
from cardioagent.ecg_agent.toolkit.measurements import rr_clean_mask


def context(unit='mV', suffix=0.):
    t=np.arange(6000)/100
    values=np.sin(2*np.pi*t)[:,None]
    values[3000:]+=suffix
    return ToolContext(Waveform(values,100.,('II',),(unit,)),0.,30.)


def measured(values,fs,start,commit_start,end):
    times=np.arange(start+.5,end,1.)
    return {'r_peaks_s':times.tolist(),'rr_intervals_s':np.diff(times).tolist(),
            'rate_bpm':60.,'rr_cv':0.,'detector_status':'candidates','detector_error':None,'duration_ms':0.}


class ToolRuntimeTests(unittest.TestCase):
    def test_cached_model_tool_only_accesses_elapsed_retained_windows(self):
        calls = []

        def cached_model(lead, start, end):
            calls.append((lead, start, end))
            return np.ones(5000, dtype=np.float32), lambda _: (0.8, 0.1)

        wave = context().waveform
        session = ToolSession(ToolContext(wave, 10., 20.), cached_model=cached_model)
        for a, b in [(0., 10.), (11., 21.)]:
            result = session.call('score_ecgfounder', {'lead': 'II', 'start_s': a, 'end_s': b})
            self.assertEqual(result['error']['code'], 'unavailable_window')
        self.assertEqual(calls, [])
        result = session.call('score_ecgfounder', {'lead': 'II', 'start_s': 10., 'end_s': 20.})
        self.assertEqual(result['result'], {'af_score': 0.8, 'afl_score': 0.1})
        self.assertEqual(calls, [('II', 10., 20.)])

    def test_default_allowlist_excludes_diagnostic_and_proxy_tools(self):
        registry=build_registry();names={t['name'] for t in registry.schemas()}
        self.assertEqual(names,{'assess_quality','detect_beats','get_rate_rhythm','get_feature_profile'})
        self.assertEqual(len(registry.schemas(True)),12)
        self.assertNotIn('classify_rhythm',{t['name'] for t in registry.schemas(True)})
        json.dumps(registry.function_schemas(True),allow_nan=False)

    def test_validation_precedes_execution(self):
        session=ToolSession(context())
        with patch('cardioagent.ecg_agent.tools.algorithms.measure') as detector:
            cases=[('detect_beats',{'lead':'II','start_s':0.,'end_s':31.},'unavailable_window'),
                   ('detect_beats',{'lead':'II','start_s':0.,'end_s':30.,'label':'AF'},'invalid_arguments'),
                   ('detect_beats',{'lead':'PPG','start_s':0.,'end_s':30.},'unknown_lead'),
                   ('detect_beats',{'lead':'II','start_s':0.,'end_s':float('nan')},'invalid_arguments'),
                   ('classify_rhythm',{},'unknown_tool')]
            for name,args,code in cases:self.assertEqual(session.call(name,args)['error']['code'],code)
            detector.assert_not_called()

    def test_budget_and_experimental_gate(self):
        session=ToolSession(context(),max_calls=1)
        args={'lead':'II','start_s':0.,'end_s':30.}
        self.assertEqual(session.call('get_qt_proxy',args)['error']['code'],'experimental_disabled')
        self.assertEqual(session.call('assess_quality',args)['error']['code'],'budget_exhausted')

    def test_cache_is_isolated_from_caller_mutation(self):
        session=ToolSession(context());args={'lead':'II','start_s':0.,'end_s':30.}
        with patch('cardioagent.ecg_agent.tools.algorithms.measure',measured):
            a=session.call('get_rate_rhythm',args)
            a['result']['rr_intervals_ms'][0]=9.
            b=session.call('get_rate_rhythm',args)
        self.assertTrue(b['cached']);self.assertEqual(b['result']['rr_intervals_ms'][0],1000.)
        self.assertEqual(session.calls[0]['result']['rr_intervals_ms'][0],1000.)
        self.assertNotEqual(a['evidence_id'],b['evidence_id'])

    def test_future_signal_never_reaches_detector(self):
        received=[]
        def detector(values,*args):received.append(values.copy());return measured(values,*args)
        args={'lead':'II','start_s':0.,'end_s':30.}
        with patch('cardioagent.ecg_agent.tools.algorithms.measure',detector):
            a=ToolSession(context()).call('get_rate_rhythm',args)
            b=ToolSession(context(suffix=1000.)).call('get_rate_rhythm',args)
        np.testing.assert_array_equal(received[0],received[1]);self.assertEqual(len(received[0]),3000)
        self.assertEqual(a['result'],b['result'])

    def test_cleaning_preserves_raw_intervals_and_exposes_removals(self):
        mask=rr_clean_mask([1000.,1000.,2500.,1000.,1000.])
        self.assertEqual(mask.tolist(),[True,True,False,True,True])

    def test_proxy_units_and_semantics(self):
        args={'lead':'II','start_s':0.,'end_s':30.}
        with patch('cardioagent.ecg_agent.tools.algorithms.measure',measured):
            unknown=ToolSession(context('NU'),allow_experimental=True).call('get_st_proxy',args)
            session=ToolSession(context(),allow_experimental=True)
            for spec in build_registry().schemas(True):
                result=session.call(spec['name'],args)
                self.assertNotEqual(result['status'],'error',result)
                json.dumps(result,allow_nan=False)
                if spec['experimental']:self.assertIn('Experimental',result['warnings'][0])
        self.assertEqual(unknown['status'],'error')

    def test_short_window_is_insufficient_not_normal_zero(self):
        out=ToolSession(context()).call('get_feature_profile',{'lead':'II','start_s':0.,'end_s':2.})
        self.assertEqual(out['status'],'insufficient');self.assertEqual(out['result'],{})

    def test_runtime_snapshots_elapsed_signal_and_discards_future(self):
        c=context();session=ToolSession(c)
        self.assertEqual(len(session.context.waveform.values),3000)
        before=session.context.waveform.values.copy()
        c.waveform.values[:]=123.
        np.testing.assert_array_equal(session.context.waveform.values,before)

    def test_catalog_is_generated_from_registry(self):
        from pathlib import Path
        saved=json.loads((code_root()/'configs/ecg/tools.schema.json').read_text())
        self.assertEqual(saved['tools'],build_registry(include_cached_model=True).schemas(True))

    def test_execution_errors_are_structured_and_not_replayed_as_success(self):
        args={'lead':'II','start_s':0.,'end_s':30.}
        failed={'rate_bpm':None,'r_peaks_s':[],'detector_error':'detector unavailable'}
        session=ToolSession(context())
        with patch('cardioagent.ecg_agent.tools.algorithms.measure',return_value=failed) as detector:
            a=session.call('detect_beats',args)
            b=session.call('detect_beats',args)
        self.assertEqual(a['error']['code'],'tool_failed')
        self.assertEqual(b['error']['code'],'tool_failed')
        self.assertEqual(detector.call_count,2)
