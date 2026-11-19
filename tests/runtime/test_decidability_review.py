"""The reference is joined only after a completed blind review."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
from analyze_decidability_review import analyze


class DecidabilityReviewTests(unittest.TestCase):
    def test_blind_review_join_and_pending_exclusion(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / 'review'
            review.mkdir()
            keys = [{'case_id': 'case-001', 'group': 'main', 'record_id': 'a',
                     'lead': 'ecg1', 'cursor_s': 10, 'reference': 'positive'},
                    {'case_id': 'case-002', 'group': 'hard', 'record_id': 'b',
                     'lead': 'ecg2', 'cursor_s': 20, 'reference': 'negative'}]
            (root / 'decidability_answer_key.jsonl').write_text(''.join(json.dumps(row) + '\n'
                for row in keys), encoding='utf-8')
            paired = [dict(group='main', record_id='a', lead='ecg1', cursor_s=10,
                           raw_decision='AF_AFL_ONGOING', aligned_decision='AF_AFL_ENDED',
                           reasoned_decision='AF_AFL_ENDED'),
                      dict(group='hard', record_id='b', lead='ecg2', cursor_s=20,
                           raw_decision='AF_AFL_ENDED', aligned_decision='AF_AFL_ONGOING',
                           reasoned_decision='AF_AFL_ONGOING')]
            (root / 'paired.jsonl').write_text(''.join(json.dumps(row) + '\n'
                for row in paired), encoding='utf-8')
            with (review / 'review.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['case_id', 'review_status',
                    'decidability', 'waveform_judgment', 'reviewer', 'evidence_basis'])
                writer.writeheader()
                writer.writerow(dict(case_id='case-001', review_status='reviewed',
                    decidability='existing_sufficient', waveform_judgment='AF_AFL_ENDED',
                    reviewer='independent', evidence_basis='waveform'))
                writer.writerow(dict(case_id='case-002', review_status='pending'))
            result = analyze(review, root)
            self.assertEqual(result['counts']['pending'], 1)
            self.assertEqual(result['counts']['existing_sufficient'], 1)
            self.assertEqual(result['by_arm']['raw']['existing_sufficient_reference_errors'], 1)
            self.assertEqual(result['by_arm']['aligned']['existing_sufficient_reference_errors'], 0)
            self.assertEqual(len(result['cases']), 1)


if __name__ == '__main__':
    unittest.main()
