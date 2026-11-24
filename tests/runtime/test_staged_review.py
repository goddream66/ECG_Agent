import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from tests.runtime import test_autonomous_events as fixture
from tests.runtime.test_react import reply

def obs(label, ids):
    return reply('assess_window',dict(interpretation=label,evidence_ids=ids,reason='Observed evidence'))
def act(action,event_id):
    return reply('choose_event_action',dict(action=action,event_id=event_id,reason='Episode decision'))
def boundary(status,**args):
    return reply('refine_boundary',dict(status=status,reason='Earlier and later measurements',**args))


class StagedReviewTests(unittest.TestCase):
    def test_advisory_pair_can_support_opening_only_after_both_current_leads_are_cited(self):
        def tool_selection(option):
            return reply('select_evidence', dict(option=option, reason='Check the paired rhythm evidence'))

        def vote(ids):
            return reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=ids,
                                              reason='Both brief co-timed recordings favor the combined target.'))

        config = json.loads((fixture.ROOT/'configs/ecg/react_qwen_review_pilot.json').read_text())
        config.update(persistent_event_evidence=True, adaptive_acquisition=True, evidence_menu=True,
                      focused_evidence_menu=True, boundary_pair_menu=True, llm_tool_max_calls=4)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            for enabled, ids, valid in ((True, ['E1','E2'], True), (True, ['E1'], False),
                                        (False, ['E1','E2'], False)):
                path.write_text(json.dumps(dict(config, advisory_reference=enabled)))
                actions = [tool_selection('compare_current_leads'), tool_selection('finish'), vote(ids)]
                actions.append(act('CONFIRM_EVENT', None) if valid else vote(ids))
                result, trace, _, _, _ = fixture.EpisodeTests().replay(
                    'fixed', actions, 1, config_file=path, score_pairs=[[.77,.09]])
                self.assertEqual(trace[0]['action'], 'CONFIRM_EVENT' if valid else 'ACTION_ERROR')
                self.assertEqual(result['intervals'], [[0,10]] if valid else [])
                if valid:
                    self.assertEqual(trace[0]['llm_action']['support_basis'], 'paired_advisory_reference')

    def test_failed_refinement_preserves_confirmation_and_observation(self):
        bad=boundary('located',before_id='future',after_id='E1')
        inputs=[fixture.measure(0),fixture.FINISH,obs('support',['E1']),act('CONFIRM_EVENT',None),bad,bad,
                fixture.measure(10),fixture.FINISH,obs('negative',['E1']),act('REQUEST_CLOSE',1),
                boundary('located',before_id='H1',after_id='E1')]
        result,trace,_,steps,_=fixture.EpisodeTests().replay('fixed',inputs,2,split='v17')
        self.assertEqual(trace[0]['state'],'OPEN')
        self.assertEqual(trace[0]['llm_action']['refinement']['status'],'error')
        self.assertEqual(result['events'][0]['end_decided_at_s'],20)
        self.assertEqual(result['events'][0]['detected_at_s'],10)
        self.assertEqual(result['events'][0]['onset_status'],'unlocalized')
        self.assertEqual(sum(s['status']=='stage_rejected' for s in steps),2)

    def test_candidate_and_invalid_lifecycle_survive_restart(self):
        inputs=[fixture.measure(0),fixture.FINISH,obs('support',['E1']),act('CREATE_CANDIDATE',None),boundary('unlocalized'),
                fixture.measure(10),fixture.FINISH,obs('support',['E1']),act('CONFIRM_EVENT',2),act('CONFIRM_EVENT',2),
                fixture.measure(20),fixture.FINISH,obs('support',['E1']),act('CONFIRM_EVENT',1),boundary('unlocalized')]
        result,trace,*_=fixture.EpisodeTests().replay('fixed',inputs,3,split='v17',interrupt_at=20)
        self.assertEqual(trace[1]['action'],'LIFECYCLE_ERROR')
        self.assertEqual(trace[1]['decision'],'SUPPORTS_TARGET')
        self.assertEqual(result['candidates'][0]['status'],'confirmed')
        self.assertEqual(result['events'][0]['detected_at_s'],10)
        self.assertEqual(result['events'][0]['confirmed_at_s'],30)
        self.assertEqual(result['supported_seconds'],30)

    def test_request_alone_cannot_close_and_bad_reference_cannot_create(self):
        invalid=obs('support',['future'])
        inputs=[fixture.FINISH,invalid,invalid,fixture.measure(10),fixture.FINISH,obs('support',['E1']),
                act('CONFIRM_EVENT',None),boundary('unlocalized'), fixture.measure(20),fixture.FINISH,
                obs('negative',['E1']),act('REQUEST_CLOSE',1),boundary('unlocalized')]
        result,trace,*_=fixture.EpisodeTests().replay('fixed',inputs,3,split='v17')
        self.assertEqual(trace[0]['action'],'ACTION_ERROR')
        self.assertEqual(trace[2]['action'],'REQUEST_CLOSE')
        self.assertTrue(result['events'][0]['right_censored'])

