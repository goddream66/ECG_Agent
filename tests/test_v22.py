import json
from types import SimpleNamespace
import unittest

import numpy as np

from cardioagent.active_agent import active_agent
from cardioagent.costs import Costs
from cardioagent.paired_metrics import pareto_frontier, risk_coverage, summarize_rates
from cardioagent.recording import Recording, Waveform


class V22Tests(unittest.TestCase):
    def test_accounting_distinguishes_failed_execution_and_cache(self):
        costs = Costs()
        costs.waveform({"status": "failed"})
        costs.waveform({"status": "ok"}, cached=True)
        report = costs.report(0.2)
        self.assertEqual([report[k] for k in ("waveform_requests", "waveform_tool_calls", "waveform_failures",
                                             "waveform_cache_hits", "action_budget_units")], [2, 1, 1, 1, 2])

    def test_agent_can_choose_auxiliary_or_stop_with_same_initial_state(self):
        wave = Waveform(np.zeros((10000, 1)), 1000, ("I",), ("mV",))
        record = Recording("hidden-id", {"ecg": wave, "ppg": wave, "pcg": wave}, {})
        initial = []

        class Client:
            def __init__(self, actions):
                self.actions = iter(actions)
                self.seen = []

            def invoke(self, messages):
                payload = json.loads(messages[-1][1])
                self.seen.append(payload)
                return SimpleNamespace(content=next(self.actions), usage_metadata={"input_tokens": 20, "output_tokens": 4})

        def observe(modality, _channel):
            return {"status": "ok", "quality": "usable", "rate_bpm": 85}

        for mode, replies, expected in (
            ("single_active", ["STOP"], ["ACQUIRE_ECG", "STOP"]),
            ("multi_active", ["ACQUIRE_PPG", "ACQUIRE_PPG", "STOP", "STOP"], ["ACQUIRE_ECG", "ACQUIRE_PPG", "STOP"]),
        ):
            client = Client(replies)
            report = active_agent(record, 3, observe, client, mode, 4)
            initial.append(client.seen[0])
            self.assertEqual([a["action"] for a in report["actions"]], expected)
            self.assertEqual(report["cost"]["llm_invocations"], len(replies))
            self.assertEqual(report["cost"]["input_tokens"], 20 * len(replies))
            self.assertEqual(report["cost"]["coordinator_recheck_calls"], 0)
        self.assertEqual(initial[0], initial[1])
        self.assertNotIn("reference_bpm", initial[0])
        self.assertNotIn("hidden-id", json.dumps(initial[0]))

    def test_agent_can_abstain_at_waveform_budget_limit(self):
        wave = Waveform(np.zeros((10000, 1)), 1000, ("I",), ("mV",))
        record = Recording("case", {"ecg": wave, "ppg": wave}, {})
        client = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="ABSTAIN", usage_metadata=None))
        report = active_agent(record, 1, lambda *_: {"status": "ok", "quality": "usable", "rate_bpm": 85},
                              client, "single_active", 1)
        self.assertIsNone(report["rate_summary"]["rate_bpm"])
        self.assertEqual(report["cost"]["waveform_tool_calls"], 1)

    def test_pareto_and_risk_coverage_keep_refusals_in_denominator(self):
        rows = [
            {"subject_id": 1, "reference_bpm": 80, "rate": 81, "rank": 2},
            {"subject_id": 1, "reference_bpm": 80, "rate": None, "rank": 0},
            {"subject_id": 2, "reference_bpm": 80, "rate": 83, "rank": 2},
        ]
        metric = summarize_rates(rows, "rate")
        self.assertAlmostEqual(metric["within_1_bpm_of_all"], 1 / 3, places=5)
        self.assertEqual(metric["p95_absolute_error_bpm"], 2.9)
        point, = risk_coverage(rows, "rate", "rank")
        self.assertEqual((point["minimum_rank"], point["coverage"], point["mae_bpm"], point["n"], point["subjects"]),
                         (2, 2 / 3, 2., 2, 2))
        self.assertAlmostEqual(point["unconditional_failure"], 2 / 3)
        self.assertEqual(pareto_frontier({"a": {"calls": 1, "risk": 0.2}, "b": {"calls": 2, "risk": 0.2}}), ["a"])
