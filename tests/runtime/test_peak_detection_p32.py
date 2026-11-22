from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
from evaluate_peak_detection_p32 import match


class PeakAuditTests(unittest.TestCase):
    def test_one_to_one_and_missed_extra(self):
        result = match([1.0, 2.0], [1.01, 1.03, 3.0])
        self.assertEqual(result['matched'], 1)
        self.assertEqual(result['unmatched_reference'], 1)
        self.assertEqual(result['unmatched_candidate'], 2)


if __name__ == '__main__':
    unittest.main()
