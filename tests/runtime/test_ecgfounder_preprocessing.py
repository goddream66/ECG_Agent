import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import butter,filtfilt,iirnotch,medfilt
from cardioagent.ecg_agent.data.ecgfounder_preprocessing import preprocess_window,load_profile
from cardioagent.ecg_agent.data.preprocessing import PROFILE as OLD_PROFILE

class FounderPreprocessingTests(unittest.TestCase):
    def wave(self,fs=500):
        t=np.arange(fs*10)/fs
        return .2*np.sin(2*np.pi*1.3*t)+.05*np.sin(2*np.pi*50*t)+.1*np.sin(2*np.pi*.2*t)

    def test_matches_upstream_chain_at_500_hz(self):
        x=self.wave();b,a=iirnotch(50,30,500);expected=filtfilt(b,a,x)
        b,a=butter(4,[.67,40],btype='bandpass',fs=500);expected=filtfilt(b,a,expected)
        expected-=medfilt(expected,201);expected=(expected-expected.mean())/(expected.std()+1e-8)
        np.testing.assert_allclose(preprocess_window(x,500),expected.astype(np.float32),atol=1e-6)

    def test_shape_scale_and_original_profile_unchanged(self):
        for fs in (250,500,1000):
            a=preprocess_window(self.wave(fs),fs)
            self.assertEqual(a.shape,(5000,));self.assertEqual(a.dtype,np.float32)
            self.assertAlmostEqual(float(a.mean()),0.,places=5);self.assertAlmostEqual(float(a.std()),1.,places=5)
        self.assertEqual(OLD_PROFILE,'neuroecg-afdb-window-v1')
        self.assertIsNone(load_profile()['clipping'])

    def test_rejects_missing_flat_and_incomplete(self):
        for x in (np.zeros(5000),np.full(5000,np.nan),np.zeros(4999)):
            with self.assertRaises(ValueError):preprocess_window(x,500)
