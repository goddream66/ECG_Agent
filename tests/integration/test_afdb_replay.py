import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cardioagent.ecg_agent.benchmarks.replay import replay_task
from cardioagent.ecg_agent.benchmarks.review_replay import EventMemory, decide, replay_review


class AFDBReplayTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = Path(folder.name)
        self.record = {"record_id": "00001", "source_rate_hz": 250, "source_samples": 12500}
        self.task = {"task_id": "test", "start_s": 0, "end_s": 50}

    def build_cache(self, future=0.8):
        rows, waves = [], []
        for step, score in enumerate((0.9, 0.7, None, 0.6, future)):
            for channel in ("ECG1", "ECG2"):
                start, end = step * 2500, (step + 1) * 2500
                rows.append({"record_id": "00001", "channel": channel,
                             "start_sample": start, "end_sample": end,
                             "status": "insufficient" if score is None else "ok",
                             "waveform_row": None if score is None else len(waves),
                             "invalid_spans": []})
                if score is not None:
                    waves.append(np.full(5000, score if channel == "ECG1" else 0.1, dtype=np.float32))
        np.save(self.cache / "00001.npy", np.stack(waves))
        (self.cache / "00001.windows.jsonl").write_text(
            "\n".join(map(json.dumps, rows)), encoding="utf-8"
        )

    def replay(self, end, predict):
        trace = []
        result = replay_task(self.cache, self.record, self.task | {"end_s": end},
                             "ECG1", 0.5, predict, trace.append)
        for row in trace:
            row.pop("model_ms", None)
        return result, trace

    def test_future_rows_are_unavailable_and_prefix_is_invariant(self):
        def predict(values):
            self.assertEqual(values.shape, (5000,))
            self.assertIsNone(values.base)
            return values[0], 0

        self.build_cache()
        original, original_trace = self.replay(39, predict)
        self.build_cache(future=np.nan)
        result, trace = self.replay(39, predict)
        self.assertEqual((original, original_trace), (result, trace))
        self.assertEqual([row["cursor_s"] for row in trace], [10, 20, 30])
        self.assertEqual(result["scored_seconds"], 20)
        self.assertEqual(result["unscored_seconds"], 19)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.replay(50, predict)

    def test_missing_evidence_does_not_merge_across_gap(self):
        self.build_cache()
        result, trace = self.replay(50, lambda values: (values[0], 0))
        self.assertEqual(result["intervals"], [[0, 20], [30, 50]])
        self.assertIsNone(trace[2]["candidate"])
        self.assertIsNone(trace[2]["event_update"])
        for row in trace:
            if update := row["event_update"]:
                self.assertEqual(update["interval"][1], row["cursor_s"])
                self.assertEqual(update["recorded_at_s"], row["cursor_s"])

    def test_index_cannot_point_to_future_waveform(self):
        self.build_cache()
        index = self.cache / "00001.windows.jsonl"
        rows = list(map(json.loads, index.read_text(encoding="utf-8").splitlines()))
        rows[0]["waveform_row"] = 2
        index.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "arrival order"):
            self.replay(50, lambda _: self.fail("Future waveform reached predictor"))

    def test_two_window_replay_keeps_confirmation_time_and_no_gap_merge(self):
        self.build_cache()
        trace = []
        result = replay_review(self.cache, self.record, self.task, 'ECG1', 0.5,
                               lambda values: (values[0], 0), trace.append,
                               self.cache, 'two-window', 4)
        self.assertEqual(result['intervals'], [[0, 20], [30, 50]])
        self.assertEqual(trace[0]['memory']['state'], 'pending')
        self.assertEqual(trace[1]['memory']['event_update']['recorded_at_s'], 20)
        self.assertEqual(trace[2]['memory']['state'], 'needs_review')
        self.assertEqual(trace[4]['memory']['event_update']['interval'], [30, 50])
        self.assertEqual(result['model_calls'], 4)

    def test_adaptive_waits_then_checks_and_retains_channel_conflict(self):
        calls = []

        def call(tool, lead):
            calls.append((tool, lead))
            payload = {'reasons': []} if tool == 'assess_quality' else {'af_score': 0.2, 'afl_score': 0}
            return {'status': 'ok', 'result': payload}

        primary = {'lead': 'ECG1', 'status': 'ok', 'result': {'af_score': 0.8, 'afl_score': 0}}
        memory = EventMemory()
        decision, checked, _ = decide(call, primary, 'ECG2', 0.5, memory, 'adaptive', 4)
        self.assertEqual(calls, [('assess_quality', 'ECG1')])
        memory.observe(0, 10, decision, checked, ['first'])
        calls.clear()
        decision, checked, reason = decide(call, primary, 'ECG2', 0.5, memory, 'adaptive', 4)
        state = memory.observe(10, 20, decision, checked, ['second'])
        self.assertEqual(reason, 'channel_disagreement')
        self.assertEqual(state['state'], 'needs_review')
        self.assertEqual(calls, [('assess_quality', 'ECG1'), ('score_ecgfounder', 'ECG2'),
                                 ('assess_quality', 'ECG2'), ('get_rate_rhythm', 'ECG1')])
        self.assertEqual(memory.intervals, [])
        calls.clear()
        decision, checked, reason = decide(call, primary, 'ECG2', 0.5, memory, 'adaptive', 1)
        self.assertEqual((decision, checked, reason), ('conflict', False, 'review_budget_exhausted'))
        self.assertEqual(calls, [('assess_quality', 'ECG1')])
