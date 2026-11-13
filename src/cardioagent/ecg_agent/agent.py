"""Deterministic ECG-only agent; streaming decisions see elapsed samples only."""
from dataclasses import asdict, dataclass
from time import perf_counter
import math
import numpy as np
from ..recording import Waveform
from .toolkit import ToolContext, ToolSession


@dataclass(frozen=True)
class Config:
    chunk_s: float = 30.
    lookback_s: float = 1.
    measurement_budget: int = 2
    policy: str = 'verify'
    lead_disagreement_bpm: float = 5.

    def __post_init__(self):
        if not math.isfinite(self.chunk_s) or self.chunk_s < 5:
            raise ValueError('chunk_s must be finite and >= 5')
        if not math.isfinite(self.lookback_s) or not 0 <= self.lookback_s <= self.chunk_s:
            raise ValueError('lookback_s must lie in [0, chunk_s]')
        if type(self.measurement_budget) is not int or self.measurement_budget < 1:
            raise ValueError('measurement_budget must be a positive integer')
        if self.policy not in ('primary', 'quality', 'verify'):
            raise ValueError('policy must be primary, quality, or verify')
        if not math.isfinite(self.lead_disagreement_bpm) or self.lead_disagreement_bpm <= 0:
            raise ValueError('lead disagreement threshold must be positive')


class ECGAgent:
    def __init__(self, config=None):
        self.config = config or Config()
        self.cursor_s = 0.
        self.ledger = []
        self._channels = None
        self._fs = None

    def step(self, elapsed: Waveform, origin_s=0.):
        """Accept an elapsed ECG block with absolute origin; no future or labels."""
        if (not math.isfinite(elapsed.fs) or elapsed.fs <= 60 or elapsed.values.ndim != 2
                or elapsed.values.shape[1] != len(elapsed.channels) or not elapsed.channels
                or len(set(elapsed.channels)) != len(elapsed.channels)):
            raise ValueError('ECG requires unique channels, a 2-D signal and fs > 60 Hz')
        if not math.isfinite(origin_s) or not 0 <= origin_s <= self.cursor_s:
            raise ValueError('Block origin must be finite and at or before cursor')
        end = origin_s + elapsed.duration
        if self._channels is not None and (elapsed.channels != self._channels or elapsed.fs != self._fs):
            raise ValueError('Stream sampling rate and channel schema must remain constant')
        if not self.cursor_s < end <= self.cursor_s + self.config.chunk_s + 1/elapsed.fs:
            raise ValueError('Prefix must advance by at most one configured chunk')
        self._channels, self._fs = elapsed.channels, elapsed.fs
        began = perf_counter()
        start, context_start = self.cursor_s, max(origin_s, self.cursor_s-self.config.lookback_s)
        trace, previews, observations = [], {}, {}
        session = ToolSession(ToolContext(elapsed, origin_s, end),
                              max_calls=max(8, 2*len(elapsed.channels)+self.config.measurement_budget))
        primary = next((c for c in ('MLII','II','I') if c in elapsed.channels), elapsed.channels[0])
        channels = [primary]+[c for c in elapsed.channels if c != primary]

        def qc(channel):
            if channel not in previews:
                response = session.call('assess_quality', {'lead':channel,'start_s':context_start,'end_s':end})
                previews[channel] = response['result'] if response['status']=='ok' else {
                    'score':0.,'reasons':[response['error']['code'] if response['error'] else 'insufficient_window'],
                    'signal_quality':{},'duration_ms':response['duration_ms']}
                trace.append({'action': 'QC', 'lead':channel, 'available_until_s':end,
                              'reasons':previews[channel]['reasons']})
            return previews[channel]

        def measure(channel):
            q = qc(channel)
            response = session.call('detect_beats', {'lead':channel,'start_s':context_start,
                                                     'end_s':end,'commit_start_s':start})
            item = response['result'] or {'r_peaks_s':[],'rate_bpm':None,'rr_cv':None,
                'rr_intervals_s':[],'detector_status':'insufficient' if response['status']=='insufficient' else 'failed',
                'detector_error':response['error'],'duration_ms':response['duration_ms']}
            observations[channel] = item | {'lead': channel, 'signal_usable':not q['reasons'],
                'eligible':not q['reasons'] and item['rate_bpm'] is not None and len(item['r_peaks_s']) >= 4}
            trace.append({'action':'MEASURE','lead':channel,'window_s':[context_start,end],
                          'rate_bpm':item['rate_bpm'],'eligible':observations[channel]['eligible']})

        if self.config.policy == 'quality':
            order = sorted(channels, key=lambda c:qc(c)['score'], reverse=True)
        else:
            order = channels
        measure(order[0])
        if self.config.policy == 'verify':
            # Independently inspect alternatives. Acquire one when the anchor is weak,
            # irregular, or another lead has substantially stronger physiological power.
            first = observations[primary]
            others = sorted(channels[1:], key=lambda c:qc(c)['score'], reverse=True)
            weak = not first['eligible'] or (first['rr_cv'] or 0) > .12
            better = others and previews[others[0]]['score'] > qc(primary)['score'] + .15
            for lead in others:
                if len(observations) >= self.config.measurement_budget or not (weak or better):
                    break
                measure(lead)
                if observations[lead]['eligible']:
                    break
        eligible = [r for r in observations.values() if r['eligible']]
        conflict = len(eligible)>1 and np.ptp([r['rate_bpm'] for r in eligible]) > self.config.lead_disagreement_bpm
        selected = max(eligible, key=lambda r:previews[r['lead']]['score']) if eligible and not conflict else None
        reason = 'lead_disagreement' if conflict else ('measurement_available' if selected else 'insufficient_evidence')
        trace.append({'action':'COMMIT' if selected else 'ABSTAIN','reason':reason})
        entry = {'window_s':[start,end], 'committed_at_s':end, 'lead':selected['lead'] if selected else None,
                 'rate_bpm':selected['rate_bpm'] if selected else None,
                 'r_peaks_s':selected['r_peaks_s'] if selected else [],
                 'status':'measured' if selected else 'review_required', 'reason':reason,
                 'observations':observations, 'previews':previews, 'trace':trace,
                 'tool_calls':session.calls,
                 'cost':{'measurements':len(observations),'qc_calls':len(previews),
                         'tool_ms':sum(r['duration_ms'] for r in observations.values())+sum(r['duration_ms'] for r in previews.values()),
                         'analysis_ms':(perf_counter()-began)*1000}}
        self.cursor_s = end
        self.ledger.append(entry)
        return entry

    def run(self, ecg: Waveform):
        if self.ledger:
            raise ValueError('Use a fresh agent for each recording')
        if not len(ecg.values):
            raise ValueError('ECG waveform is empty')
        samples = max(1, round(self.config.chunk_s*ecg.fs))
        for end in range(samples, len(ecg.values)+samples, samples):
            # Copy only the current chunk and bounded lookback, independent of total duration.
            first = max(0, end-samples-round(self.config.lookback_s*ecg.fs))
            block = Waveform(ecg.values[first:min(end,len(ecg.values))].copy(), ecg.fs, ecg.channels, ecg.units)
            self.step(block, origin_s=first/ecg.fs)
        return {'schema':1, 'agent':'ecg_agent_v1', 'controller':'deterministic_rules',
                'config':asdict(self.config),'duration_s':self.cursor_s,'ledger':self.ledger,
                'semantics':'ECG beat/rate measurements and review decisions; no disease diagnosis or calibrated confidence'}
