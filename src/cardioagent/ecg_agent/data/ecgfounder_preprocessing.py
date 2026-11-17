"""Separate ECGFounder window profile, adapted from upstream util.filter_bandpass.

MIT upstream notice: third_party/ecgfounder/LICENSE. No whole-record filtering.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly
from cardioagent.common.paths import code_root

PROFILE = 'ecgfounder-filter-zscore-window-v1'


def profile_path():
    return code_root()/'configs/ecg/ecgfounder_preprocessing.json'


def load_profile():
    config=json.loads(profile_path().read_text(encoding='utf-8'))
    if config['profile']!=PROFILE or config['target_fs']!=500 or config['window_seconds']!=10:
        raise ValueError('Unsupported ECGFounder preprocessing profile')
    return config


def preprocess_window(values, fs):
    c=load_profile();x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(fs) or fs<=80 or not float(fs).is_integer():
        raise ValueError('Expected integral source sampling rate above 80 Hz')
    if x.shape!=(int(fs)*c['window_seconds'],) or not np.isfinite(x).all():
        raise ValueError('Expected one finite complete 10-second physical-unit window')
    if np.ptp(x)<=1e-10:raise ValueError('Flat ECG cannot be z-score normalized')
    # At native 500 Hz no resampling occurs. Other rates are mapped explicitly first.
    if fs!=c['target_fs']:
        from math import gcd
        divisor=gcd(int(fs),c['target_fs'])
        x=resample_poly(x,c['target_fs']//divisor,int(fs)//divisor)
    if x.shape!=(5000,):raise ValueError('Unexpected resampled window length')
    b,a=iirnotch(c['notch_hz'],c['notch_q'],c['target_fs'])
    x=filtfilt(b,a,x)
    b,a=butter(c['filter_order'],c['bandpass_hz'],btype='bandpass',fs=c['target_fs'])
    x=filtfilt(b,a,x)
    kernel=int(c['baseline_window_seconds']*c['target_fs'])+1
    if kernel%2==0:kernel+=1
    x=x-medfilt(x,kernel_size=kernel)
    std=float(np.std(x))
    if not np.isfinite(std) or std<=1e-12:raise ValueError('Degenerate filtered ECG')
    x=(x-np.mean(x))/(std+c['zscore_epsilon'])
    output=x.astype(np.float32)
    if not np.isfinite(output).all():raise ValueError('Nonfinite model input')
    return output
