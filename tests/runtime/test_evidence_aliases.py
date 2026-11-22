import tempfile
import unittest
from pathlib import Path
from cardioagent.ecg_agent.harness.short_ids import EvidenceAliases


class TestEvidenceAliases(unittest.TestCase):
    def test_stable_across_windows_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/'ids.jsonl'
            ids = EvidenceAliases(file)
            a, b = 'ecg:'+'a'*64, 'ecg:'+'b'*64
            msg = [{'content': a+' '+b+' '+a}]
            self.assertEqual(ids.messages(msg), [{'content':'E1 E2 E1'}])
            self.assertIn(a, msg[0]['content'])
            again = EvidenceAliases(file)
            self.assertEqual(again.display(a),'E1')
            self.assertEqual(again.arguments({'evidence_ids':['E2','E1']}),{'evidence_ids':[b,a]})
            with self.assertRaises(ValueError):
                again.arguments({'evidence_id':'E3'})

