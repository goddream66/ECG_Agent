"""Run existing ECG tools on a bounded, already arrived AFDB window."""

from pathlib import Path

import wfdb

from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.data.afdb import CHANNELS, SOURCE_FS
from cardioagent.ecg_agent.data.preprocessing import WINDOW_SECONDS, invalid_spans, preprocess_window
from cardioagent.ecg_agent.harness.runtime import ToolSession
from cardioagent.ecg_agent.schemas import ShiftedScoreArgs, ToolContext


class HistoricalWindow:
    def __init__(self, raw_root, task, predict, *, retention_s=120):
        name = task['record_id']
        if len(name) != 5 or not name.isascii() or not name.isdecimal():
            raise ValueError('Expected an AFDB record ID')
        self.stem = str(Path(raw_root) / name)
        self.task = task
        self.predict = predict
        self.retention_s = retention_s

    def session(self, start, end, cursor, context, *, allow_shifted=False):
        if (end - start != WINDOW_SECONDS
                or not self.task['start_s'] <= start < end <= cursor <= self.task['end_s']
                or cursor - start > self.retention_s):
            raise ValueError('Historical window is outside the arrived retained task span')
        samples = [time * SOURCE_FS for time in (start, end, cursor)]
        if any(abs(value - round(value)) > 1e-6 for value in samples):
            raise ValueError('Historical window must align with the source sample grid')
        first, _, arrived = map(round, samples)
        raw = wfdb.rdrecord(self.stem, sampfrom=first, sampto=arrived)
        if (raw.fs != SOURCE_FS or tuple(raw.sig_name) != CHANNELS
                or raw.p_signal.shape != (arrived - first, len(CHANNELS))):
            raise ValueError('Historical signal does not match AFDB coordinates')
        wave = Waveform(raw.p_signal, raw.fs, tuple(raw.sig_name), tuple(raw.units))

        def model_window(lead, a, b):
            if (a, b) != (start, end):
                return None, self.predict
            values = wave.window(lead, 0, end - start)
            if invalid_spans(values, SOURCE_FS):
                return None, self.predict
            return preprocess_window(values, SOURCE_FS), self.predict

        return ToolSession(ToolContext(wave, start, cursor), max_calls=4,
                           cached_model=model_window, evidence_context=context, allow_experimental=allow_shifted)

    def shifted_score(self, start, end, cursor, lead, offset, context):
        args = ShiftedScoreArgs(lead=lead, start_s=start+offset, end_s=end+offset, offset_s=offset)
        if not self.task['start_s'] <= start < end <= cursor:
            raise ValueError('Original window is outside the arrived task span')
        return self.session(args.start_s, args.end_s, cursor, context, allow_shifted=True).call(
            'score_ecgfounder_shifted', args.model_dump())
