"""Two-stage blinding, chronological release, and sealed-review checks."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
from prepare_decidability_review_v2 import audit_stage_a, build_stage_a, build_stage_b, read_csv, \
    release_next, write_csv
from analyze_decidability_review_v2 import analyze


def case(cursor, reference):
    return {'record_id': 'a', 'lead': 'ecg1', 'cursor_s': cursor,
            'reference': reference,
            'primary': [{'id': 'E1', 'status': 'ok', 'lead': 'ECG1',
                         'window_s': [cursor - 10, cursor],
                         'result': {'af_score': .2, 'afl_score': .1}}],
            'backup': [{'id': 'E2', 'status': 'not_measured'}]}


class ReviewV2Tests(unittest.TestCase):
    def test_two_stage_review_and_no_future_release(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            (source / 'main').mkdir(parents=True)
            (source / 'hard').mkdir()
            (source / 'main' / 'cases.jsonl').write_text('\n'.join(
                json.dumps(x) for x in [case(30, 'positive'), case(20, 'negative')])
                + '\n', encoding='utf-8')
            (source / 'hard' / 'cases.jsonl').write_text('', encoding='utf-8')
            output = root / 'review'
            self.assertEqual(build_stage_a(source, output), 2)
            stage_a = output / 'stage_a'
            self.assertTrue((stage_a / 'cases/case-001.json').exists())
            self.assertFalse((stage_a / 'cases/case-002.json').exists())
            self.assertEqual(len(read_csv(stage_a / 'review.csv')), 1)
            self.assertEqual(audit_stage_a(output)['released_stage_a_cases'], 1)
            packet = json.loads((stage_a / 'cases/case-001.json').read_text())
            self.assertEqual(packet['cursor_s'], 20)
            self.assertNotIn('reference', packet)
            self.assertLessEqual(packet['measurements'][0]['window_s'][1], 20)
            self.assertEqual(release_next(output, 'stage-a')['waiting_for_review'],
                             'case-001')
            with self.assertRaises(ValueError):
                build_stage_b(output, root, lambda *args: None)
            rows = read_csv(stage_a / 'review.csv')
            fields = list(rows[0])
            rows[0].update(review_status='reviewed', decidability='sufficient',
                           judgment='AF_AFL_ONGOING', confidence_1to5='5',
                           evidence_ids='E1', rationale='score evidence', reviewer='r1')
            write_csv(stage_a / 'review.csv', fields, rows)
            self.assertEqual(release_next(output, 'stage-a')['released_case'],
                             'case-002')
            rows = read_csv(stage_a / 'review.csv')
            rows[1].update(review_status='reviewed', decidability='insufficient',
                           judgment='UNCERTAIN', confidence_1to5='3',
                           evidence_ids='', rationale='ambiguous', reviewer='r1')
            write_csv(stage_a / 'review.csv', fields, rows)
            self.assertTrue(release_next(output, 'stage-a')['complete'])
            def fake_render(raw_root, record, cursor, path, case_id):
                path.write_bytes(b'PNG')
            self.assertEqual(build_stage_b(output, root, fake_render), 2)
            stage_b = output / 'stage_b'
            self.assertTrue((stage_b / 'waveforms/case-001.png').exists())
            self.assertFalse((stage_b / 'waveforms/case-002.png').exists())
            b_rows = read_csv(stage_b / 'review.csv')
            b_fields = list(b_rows[0])
            b_rows[0].update(review_status='reviewed', decidability='sufficient',
                             judgment='AF_AFL_ONGOING', confidence_1to5='4',
                             waveform_basis='rhythm', next_evidence_needed='none',
                             reviewer='r2')
            write_csv(stage_b / 'review.csv', b_fields, b_rows)
            self.assertEqual(release_next(output, 'stage-b')['released_case'],
                             'case-002')
            b_rows = read_csv(stage_b / 'review.csv')
            b_rows[1].update(review_status='reviewed', decidability='sufficient',
                             judgment='AF_AFL_ENDED', confidence_1to5='4',
                             waveform_basis='rhythm', next_evidence_needed='none',
                             reviewer='r2')
            write_csv(stage_b / 'review.csv', b_fields, b_rows)
            paired = [dict(group='main', record_id='a', lead='ecg1', cursor_s=cursor,
                           raw_decision='AF_AFL_ONGOING',
                           aligned_decision='AF_AFL_ONGOING',
                           reasoned_decision='AF_AFL_ONGOING') for cursor in (20, 30)]
            results = root / 'results'
            results.mkdir()
            (results / 'paired.jsonl').write_text(''.join(json.dumps(row) + '\n'
                for row in paired), encoding='utf-8')
            report = analyze(output, results)
            self.assertEqual(report['n_cases'], 2)
            self.assertEqual(report['counts']['score_sufficient'], 1)
            self.assertEqual(report['counts']['waveform_adds_information'], 1)
            rows[0]['rationale'] = 'edited after sealing'
            write_csv(stage_a / 'review.csv', fields, rows)
            with self.assertRaisesRegex(ValueError, 'changed after'):
                analyze(output, results)


if __name__ == '__main__':
    unittest.main()
