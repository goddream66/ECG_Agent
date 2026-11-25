import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cardioagent.ecg_agent.experiments.scroll_focused_agent import evaluate


class ScrollCoarseOnlyTests(unittest.TestCase):
    def test_ambiguous_windows_only_allow_coarse_label_evidence(self):
        calls = []
        fractions = {0: 0.1, 30: 0.2, 60: 0, 90: 0.1}

        def label(record, database, start, end):
            calls.append((record, database, start, end))
            fraction = fractions[start]
            return dict(dominant_rhythm='N', af_frac=fraction, confidence=1)

        class Model:
            def review(self, observations, stage):
                self_test.assertEqual(stage, 'commit_coarse')
                for observation, choice in zip(observations, ('coarse', 'drop'), strict=True):
                    self_test.assertEqual(observation['available'], ['coarse', 'drop'])
                    self_test.assertEqual(observation['measurements'], 0)
                    self_test.assertEqual(observation['scan']['measurement_cost'], {'coarse': 0})
                    self_test.assertFalse('answer' in json.dumps(observation))
                    yield dict(choice=choice)

        def score(verifier, prediction, reference):
            self.assertEqual(prediction, {'intervals': [[0, 60]]})
            return 0.5

        self_test = self
        task = dict(id='record-ep-full', pn_dir='afdb', record='record', task='episode_detection',
                    t0=0, t1=95, verifier={}, answer={'intervals': [[5, 8], [42, 48], [92, 92.5]]})
        with TemporaryDirectory() as tmp, patch(
                'cardioagent.ecg_agent.experiments.scroll_focused_agent.refine',
                side_effect=AssertionError('coarse-only mode attempted a finer measurement')):
            result = evaluate(Model(), task, label, score, Path(tmp), 2, coarse_only=True)
            with gzip.open(Path(tmp)/'qwen_calls.jsonl.gz', 'rt', encoding='utf-8') as log:
                receipts = [json.loads(line) for line in log]
        self.assertEqual(calls, [('record', 'afdb', start, end) for start,end in ((0,30),(30,60),(60,90),(90,95))])
        self.assertEqual(result['counts']['extra_diagnostics'], 0)
        self.assertEqual(result['counts']['model_calls'], 2)
        self.assertEqual(result['counts']['selected_drop'], 1)
        self.assertTrue(all(row['evidence'] == [] for row in receipts))
        self.assertTrue(all(update['recorded_at_s'] == 90 for update in receipts[0]['event_updates']))


if __name__ == '__main__':
    unittest.main()
