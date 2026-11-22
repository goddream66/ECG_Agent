import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from uuid import UUID
from unittest.mock import patch

import numpy as np

from cardioagent.ecg_agent.evaluation.metrics import evaluate_record
from cardioagent.ecg_agent.experiments.episode_checkpoint import EpisodeCheckpoint
from cardioagent.ecg_agent.experiments.runner import run_job
from cardioagent.ecg_agent.harness.autonomous_decision import validate
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents
from tests.runtime import test_event_checkpoints as fixtures
from tests.runtime.test_ecg_toolkit import measured
from tests.runtime.test_react import ScriptModel, reply


ROOT = Path(__file__).resolve().parents[2]
FINISH = reply('finish_measurements', {})


def measure(start, tool='score_ecgfounder', lead='ECG1'):
    return reply(tool, dict(lead=lead, start_s=start, end_s=start + 10))


def submit(assessment, action, ids, next_s, **kwargs):
    return reply('submit_event_decision', dict(dict(assessment=assessment, event_action=action,
                 reason='Decision based on cited measurements', uncertainty_cause=None,
                 evidence_ids=ids, pending=None, boundary=None, next_review_s=next_s), **kwargs))


class EpisodeTests(unittest.TestCase):
    def replay(self, mode, actions, n, *, future=None, interrupt_at=None):
        with TemporaryDirectory() as tmp:
            job = fixtures.EventCheckpointTests().make_job(Path(tmp))
            cfg = ROOT / f'configs/ecg/react_qwen_episode_{mode}.json'
            job.update(policy='event-react', llm=json.loads(cfg.read_text()))
            job['task']['end_s'] = n*10
            job['record'].update(source_samples=n*2500, source_record_samples=n*2500)
            rows = [dict(record_id='00001', channel=ch, start_sample=i*2500, end_sample=(i+1)*2500,
                         status='ok', waveform_row=2*i+j, invalid_spans=[])
                    for i in range(n) for j,ch in enumerate(('ECG1','ECG2'))]
            cache, scores = Path(job['cache']), Path(job['scores'])
            (cache/'00001.windows.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            np.save(cache/'00001.npy', np.ones((2*n,5000), dtype=np.float32))
            np.save(scores/'00001.npy', np.tile([.1,.95], (2*n,1)))
            wave = np.sin(np.arange(n*2500)/8.)
            if future is not None:
                wave[2500:] = future
            raw = SimpleNamespace(fs=250, sig_name=('ECG1','ECG2'), sig_len=n*2500,
                                  p_signal=np.column_stack((wave,wave)), units=('mV','mV'))
            model = ScriptModel(actions)
            with (patch('cardioagent.ecg_agent.llm.providers.build_chat_model', return_value=model),
                  patch('cardioagent.ecg_agent.agent.wfdb.rdrecord', return_value=raw),
                  patch('cardioagent.ecg_agent.harness.runtime.uuid4', side_effect=[UUID(int=i+1) for i in range(n)]),
                  patch('cardioagent.ecg_agent.tools.algorithms.measure', side_effect=measured),
                  patch('cardioagent.ecg_agent.harness.react.decide_event') as rule):
                if interrupt_at is not None:
                    commit = EpisodeCheckpoint.commit
                    def interrupt(cp, logs, tracker, *args):
                        commit(cp, logs, tracker, *args)
                        if tracker.cursor == interrupt_at:
                            # Simulate an incomplete write after the last committed batch.
                            with (cp.folder/'qwen_calls.jsonl.gz').open('ab') as stream:
                                stream.write(b'incomplete call')
                            raise InterruptedError('simulated worker restart')
                    with patch.object(EpisodeCheckpoint, 'commit', interrupt):
                        with self.assertRaises(InterruptedError):
                            run_job(job)
                    recorded = len(model.inputs)
                    self.assertGreater(recorded, 0)
                result, _ = run_job(job)
                rule.assert_not_called()
            folder = Path(job['target'])
            def read(name):
                return [json.loads(s) for s in (folder/(name+'.jsonl')).read_text().splitlines()]
            with gzip.open(folder/'qwen_calls.jsonl.gz', 'rt', encoding='utf-8') as stream:
                requests = [json.loads(line) for line in stream]
            return result, read('window_trace'), read('event_updates'), read('llm_steps'), requests

    def test_autonomous_episode_pending_sleep_and_bracket(self):
        actions = [measure(0), FINISH, submit('SUPPORTS_TARGET','OPEN',['E1'],20,
                    boundary=dict(kind='onset_unobserved',first_supported_interval_s=[0,10])),
                   FINISH, submit('UNCERTAIN','DEFER_DECISION',[],34,uncertainty_cause='missing',
                                 pending=dict(question='Did AF stop?',expected_evidence='Compare later leads')),
                   reply('retrieve_memory', dict(scope='events',start_s=0.,end_s=20.)),
                   measure(30), measure(30,'detect_beats','ECG2'), FINISH,
                   submit('AGAINST_TARGET','CLOSE',['E1','E2'],None,pending_resolution='resolved',
                          boundary=dict(kind='bracketed',range_s=[32,36]))]
        result, trace, updates, steps, requests = self.replay('autonomous',actions,4)
        self.assertEqual(result['intervals'], [[0,34]])
        self.assertEqual(result['burden_intervals'], [[0,10]])
        self.assertEqual(result['unknown_intervals'], [[10,36]])
        self.assertEqual([r['action'] for r in trace], ['OPEN','DEFER_DECISION',None,'CLOSE'])
        self.assertEqual((trace[2]['tool_calls'],trace[3]['requested_wake_s']), (0,34))
        self.assertEqual(result['events'][0]['end_decided_at_s'], 40)
        self.assertEqual(result['pending_questions'][0]['status'], 'resolved')
        self.assertTrue(any(r['status']=='memory_retrieved' for r in steps))
        self.assertFalse(any(r['cursor_s']==30 for r in requests))
        self.assertEqual(len(requests),len(actions))
        for r in requests:
            self.assertNotIn('KEEP_OPEN',r['messages'][0]['content'])
            names = [t['function']['name'] for t in r['tools']]
            if names == ['submit_event_decision']:
                actions_allowed = r['tools'][0]['function']['parameters']['properties']['event_action']['enum']
                self.assertIn('OPEN' if r['cursor_s']==10 else 'CLOSE', actions_allowed)
        metrics = evaluate_record('task',result['intervals'],dict(start_s=0,end_s=40,
               intervals=[[0,34]],burden=.25), burden_intervals=result['burden_intervals'],
               unknown=result['unknown_intervals'], events=result['events'],updates=updates)
        self.assertEqual(metrics['burden_absolute_error_pp'],0)
        self.assertEqual(metrics['predicted_af_s'],10)
        self.assertEqual(metrics['end_delay_sum_s'],6)

    def test_onset_midpoint_only_affects_localization(self):
        actions = [measure(0), measure(0,'detect_beats','ECG2'), FINISH,
                   submit('SUPPORTS_TARGET','OPEN',['E1','E2'],None,
                          boundary=dict(kind='bracketed', range_s=[2,4]))]
        result, _, _, _, _ = self.replay('fixed',actions,1)
        self.assertEqual(result['intervals'], [[3,10]])
        self.assertEqual(result['burden_intervals'], [[4,10]])
        self.assertEqual(result['unknown_intervals'], [[0,4]])
        self.assertIsNone(result['events'][0]['end_decided_at_s'])

    def test_fixed_retry_and_continued_tracking(self):
        actions = [measure(0),FINISH,submit('SUPPORTS_TARGET','OPEN',['E1'],25,
                    boundary=dict(kind='onset_unobserved',first_supported_interval_s=[0,10])),
                   submit('SUPPORTS_TARGET','OPEN',['E1'],20,
                    boundary=dict(kind='onset_unobserved',first_supported_interval_s=[0,10])),
                   measure(10), FINISH, submit('SUPPORTS_TARGET','RECORD_ONLY',['E1'],30),
                   FINISH,submit('AGAINST_TARGET','CLOSE',['forged'],None,boundary=dict(kind='localized',time_s=25)),
                   submit('AGAINST_TARGET','CLOSE',[],None,boundary=dict(kind='localized',time_s=25))]
        result, trace, _, steps, requests = self.replay('fixed',actions,3)
        self.assertEqual([r['action'] for r in trace],['OPEN','RECORD_ONLY','ACTION_ERROR'])
        self.assertEqual(result['events'][0]['final_status'],'RIGHT_CENSORED')
        self.assertIsNone(result['events'][0]['end_decided_at_s'])
        self.assertEqual(result['supported_seconds'],20)
        self.assertEqual(result['unknown_intervals'],[[20,30]])
        self.assertEqual([r['phase'] for r in steps if r['status']=='error'],['final','final','retry'])
        self.assertIn('scheduled_review_s',requests[3]['messages'][-1]['content'])

    def test_no_automatic_open_and_future_samples_do_not_change_first_window(self):
        actions = [measure(0), FINISH,submit('SUPPORTS_TARGET','RECORD_ONLY',['E1'],20),
                   measure(10),FINISH,submit('SUPPORTS_TARGET','RECORD_ONLY',['E1'],None)]
        first = self.replay('fixed',actions,2)
        second = self.replay('fixed',actions,2,future=100)
        self.assertEqual(first[0]['events'],[])
        self.assertEqual(first[0]['burden_intervals'],[[0,20]])
        self.assertEqual(first[1][0],second[1][0])
        self.assertEqual(first[4][0]['messages'],second[4][0]['messages'])

    def test_restart_recovers_pending_event_memory_and_executed_prefix_once(self):
        actions = [measure(0), FINISH, submit('SUPPORTS_TARGET','OPEN',['E1'],20,
                    boundary=dict(kind='onset_unobserved',first_supported_interval_s=[0,10])),
                   FINISH, submit('UNCERTAIN','DEFER_DECISION',[],35,uncertainty_cause='missing',
                                 pending=dict(question='Still AFL?',expected_evidence='Check other lead')),
                   reply('retrieve_memory', dict(scope='evidence',start_s=0.,end_s=10.)),
                   measure(30),FINISH,submit('SUPPORTS_TARGET','RECORD_ONLY',['E1'],None,
                                             pending_resolution='withdrawn')]
        clean = self.replay('autonomous',actions,4)
        resumed = self.replay('autonomous',actions,4,interrupt_at=20)
        for key in ('events','intervals','pending_questions','unknown_intervals','burden_intervals'):
            self.assertEqual(clean[0][key],resumed[0][key])
        self.assertEqual(clean[1],resumed[1])
        self.assertEqual(len(resumed[-1]),len(actions))
        self.assertEqual(resumed[0]['counts']['llm_calls'],len(actions))
        self.assertTrue(any(r['status']=='memory_retrieved' and r['count']==1 for r in resumed[3]))

    def test_boundary_citations_and_conflict_are_checked_against_actual_measurements(self):
        tracker = AutonomousEvents('fixed',0)
        snapshot = tracker.snapshot(0,10,False)
        rows = {'k': dict(tool='score_ecgfounder',lead='ECG1',window_s=[0,10],available_at_s=10,
                          status='ok',result={'af_score':.7,'afl_score':.2})}
        args = dict(assessment='SUPPORTS_TARGET',event_action='OPEN',evidence_ids=['E1'],
                    reason='Rhythm supports AF',next_review_s=20,pending=None,uncertainty_cause=None,
                    boundary=dict(kind='localized',time_s=9.5))
        for changes, error in [
                ({}, 'local measured change'),
                ({'evidence_ids':['E2']},'visible'),
                ({'event_action':'CLOSE'},'Input should be'),
                (dict(assessment='UNCERTAIN',event_action='RECORD_ONLY',uncertainty_cause='conflicting',
                      boundary=None),'two distinct')]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,error):
                validate(args | changes, {'E1':'k'}, {'k'}, rows,snapshot,{'detect_beats'})
        rows['k']['available_at_s'] = 11
        with self.assertRaisesRegex(ValueError,'arrived'):
            validate(args,{'E1':'k'},{'k'},rows,snapshot,{'detect_beats'})

    def test_sleeping_to_end_leaves_question_unresolved(self):
        actions = [FINISH,submit('UNCERTAIN','DEFER_DECISION',[],21,uncertainty_cause='missing',
                                  pending=dict(question='New episode?',expected_evidence='later ECG'))]
        result, trace, _, _, requests = self.replay('autonomous',actions,2)
        self.assertEqual(trace[-1]['decision'],'UNREVIEWED')
        self.assertEqual(result['events'],[])
        self.assertEqual(result['pending_questions'][0]['status'],'unresolved_at_stream_end')
        self.assertEqual(len(requests),2)
