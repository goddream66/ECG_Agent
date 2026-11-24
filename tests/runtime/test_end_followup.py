import unittest
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cardioagent.ecg_agent.harness.end_review import end_evidence
from tests.runtime import test_autonomous_events as fixture
from tests.runtime.test_react import reply
from tests.runtime.test_react import ScriptModel
from cardioagent.ecg_agent.harness.staged_review import assess, retain_end_evidence
from tests.runtime.test_staged_review import act, boundary


class EndFollowupTests(unittest.TestCase):
    def test_accepted_aligned_positive_survives_inconclusive_other_lead(self):
        def step(a, directions, observation, prior=None, enabled=True):
            span = [a,a+10]
            rows = [dict(id=lead, lead=lead, window_s=span, target_score=score,
                         reference=dict(direction=direction))
                    for lead, (score,direction) in zip(('ECG1','ECG2'),directions)]
            ids = {row['id']: f"{a}-{row['id']}" for row in rows}
            ledger = dict.fromkeys(ids.values(), dict(available_at_s=a+10))
            return retain_end_evidence(dict(window_s=span, end_review_memory=prior or {}),
                dict(rhythm_by_time=rows), ids, ledger,
                dict(interpretation=observation, directional_ids=[ids['ECG1']]), accepted_support_memory=enabled)

        mixed = [(.9202,'support'),(.6240,'indeterminate')]
        self.assertIsNone(step(0, mixed, 'support', enabled=False)[0]['support'])
        self.assertIsNone(step(0, mixed, 'unavailable')[0]['support'])
        self.assertIsNone(step(0, [mixed[0],(.32,'negative')], 'conflict')[0]['support'])
        memory, _ = step(0, mixed, 'support')
        self.assertEqual(memory['support']['evidence_id'], '0-ECG1')
        lows = [(.1,'negative'),(.2,'negative')]
        memory, proof = step(10, lows, 'negative', memory)
        self.assertFalse(proof['can_confirm'])
        _, proof = step(20, lows, 'negative', memory)
        self.assertTrue(proof['can_confirm'])
    def run_persistent_episode(self, observations, score_pairs, *, interrupt_at=None, adaptive=False, menu=False,
                               focused=False, pairs=False, advisory=False, joint=False):
        name = 'react_qwen_review_pilot.json' if advisory else 'react_qwen_followup_close.json'
        cfg_path = fixture.ROOT/'configs/ecg'/name
        config = dict(json.loads(cfg_path.read_text()), persistent_event_evidence=True,
                      adaptive_acquisition=adaptive or menu or focused or pairs,
                      evidence_menu=menu or focused or pairs, focused_evidence_menu=focused or pairs,
                      boundary_pair_menu=pairs, advisory_reference=advisory, acquire_or_assess=joint)
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'event_evidence.json'
            cfg.write_text(json.dumps(config))
            return fixture.EpisodeTests().replay('fixed', observations, len(score_pairs), config_file=cfg,
                                                 score_pairs=score_pairs, interrupt_at=interrupt_at)

    def test_advisory_open_can_close_after_restart_without_relabeling_measurements(self):
        def pick(option):
            return reply('select_evidence', dict(option=option, reason='compare the two leads'))

        def vote(finding):
            return reply('assess_window', dict(finding=finding, evidence_ids=['E1','E2'], reason='paired scores'))

        calls = [pick('compare_current_leads'), pick('finish'), vote('TARGET_PRESENT'), act('CONFIRM_EVENT', None)]
        for _ in range(2):
            calls.extend([pick('compare_current_leads'), vote('TARGET_ABSENT'), act('REQUEST_CLOSE',1)])
        calls.append(reply('confirm_event_end', dict(decision='CONFIRM_END', reason='two later contrary intervals')))
        result, trace, updates, _, _ = self.run_persistent_episode(
            calls, [[.77,.09],[.12,.05],[.13,.04]], interrupt_at=20, pairs=True, advisory=True)
        event, = result['events']
        self.assertEqual(event['end_decided_at_s'], 30)
        self.assertEqual(event['end_boundary']['range_s'], [0,20])
        pair = trace[0]['llm_action']['support_pair_ids']
        self.assertEqual(set(event['last_support_pair_ids']), set(pair))
        self.assertEqual(set(updates[-1]['event_update']['last_support_pair_ids']), set(pair))
        self.assertTrue(set(pair).issubset(event['end_evidence_ids']))
        saved = trace[0]['llm_action']['end_review_memory']['support']['measurements']
        self.assertEqual([row['direction'] for row in saved], ['indeterminate','indeterminate'])

    def test_support_survives_ambiguity_and_context_eviction_on_resume(self):
        def window(start, finding, action, ident):
            return [fixture.measure(start), fixture.FINISH,
                    reply('assess_window', dict(finding=finding, evidence_ids=['E1'], reason='measured')),
                    act(action, ident)]

        actions = window(0, 'TARGET_PRESENT', 'CONFIRM_EVENT', None) + [boundary('unlocalized')]
        for start in range(10, 80, 10):
            actions += window(start, 'CANNOT_TELL', 'HOLD_EVENT', 1)
        actions += window(80, 'TARGET_ABSENT', 'REQUEST_CLOSE', 1)
        actions += window(90, 'TARGET_ABSENT', 'REQUEST_CLOSE', 1)
        actions += [reply('confirm_event_end', dict(decision='CONFIRM_END', reason='sustained negative'))]
        result, trace, *_ = self.run_persistent_episode(
            actions, [[.99,.1]] + [[.62,.1]]*7 + [[.1,.02]]*2, interrupt_at=90
        )
        self.assertEqual(trace[-2]['state'], 'OPEN')
        event, = result['events']
        self.assertEqual(event['end_decided_at_s'], 100)
        self.assertEqual(event['end_boundary']['range_s'], [0,90])
        self.assertEqual(event['end_evidence_by_role']['last_support'], event['start_evidence_ids'][0])

    def test_supported_continuation_publishes_without_extra_model_action(self):
        actions = [fixture.measure(0), fixture.FINISH,
                   reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1'], reason='measured')),
                   act('CONFIRM_EVENT', None), boundary('unlocalized'),
                   fixture.measure(10), fixture.FINISH,
                   reply('assess_window', dict(finding='TARGET_ABSENT', evidence_ids=['E1'], reason='low')),
                   act('REQUEST_CLOSE', 1),
                   fixture.measure(20), fixture.FINISH,
                   reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1'], reason='recovered')),
                   act('CONTINUE_EVENT', 1)]
        result, _, updates, _, requests = self.run_persistent_episode(actions, [[.99,.1],[.12,.1],[.97,.1]])
        extension = updates[-1]['event_update']
        self.assertIsNone(updates[-2]['event_update'])
        self.assertEqual((extension['kind'], extension['event_id'], extension['interval'],
                          extension['recorded_at_s'], extension['provisional']), ('extend', 1, [0,30], 30, True))
        self.assertEqual(len(requests), len(actions))
        self.assertEqual(result['events'][0]['opposing_intervals'], [[10,20]])

    def test_adaptive_route_preserves_model_control_of_event_changes(self):
        def observe(finding):
            return reply('assess_window', dict(finding=finding, evidence_ids=['E1'], reason='primary score'))

        actions = [fixture.FINISH, observe('TARGET_PRESENT'), act('CONFIRM_EVENT', None), boundary('unlocalized'),
                   fixture.FINISH, observe('TARGET_ABSENT'), act('REQUEST_CLOSE', 1),
                   fixture.FINISH, observe('TARGET_ABSENT'), act('REQUEST_CLOSE', 1),
                   reply('confirm_event_end', dict(decision='CONFIRM_END', reason='two later low windows'))]
        result, trace, updates, _, requests = self.run_persistent_episode(
            actions, [[.12,.03],[.99,.1],[.99,.1],[.12,.02],[.13,.02]], adaptive=True
        )
        self.assertEqual([row['action'] for row in trace],
                         ['NO_EVENT_CHANGE','CONFIRM_EVENT','CONTINUE_EVENT','REQUEST_CLOSE','REQUEST_CLOSE'])
        self.assertEqual([row['llm_action']['decision_source'] for row in trace],
                         ['stable_primary','llm_staged','stable_primary','llm_staged','llm_staged'])
        self.assertEqual({row['cursor_s'] for row in requests}, {20,40,50})
        self.assertEqual(result['counts']['model_calls'], 5)
        self.assertEqual(result['intervals'], [[10,30]])
        self.assertEqual(updates[2]['event_update']['interval'], [10,30])

    def test_evidence_menu_executes_only_the_chosen_unmeasured_lead(self):
        actions = [
            reply('select_evidence', dict(option='compare_current_leads', reason='check new high')),
            reply('select_evidence', dict(option='finish', reason='corroborated')),
            reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1','E2'], reason='both high')),
            act('CONFIRM_EVENT', None), boundary('unlocalized'),
        ]
        result, _, _, steps, requests = self.run_persistent_episode(actions, [[.94,.05]], menu=True)
        self.assertEqual(result['intervals'], [[0,10]])
        self.assertEqual(result['counts']['model_calls'], 2)
        choices = [json.loads(request['messages'][1]['content'])['options']
                   for request in requests if request['tools'][0]['function']['name'] == 'select_evidence']
        self.assertEqual(choices[0]['compare_current_leads']['lead'], 'ECG2')
        self.assertNotIn('compare_current_leads', choices[1])
        observed = [row['option'] for row in steps if row.get('phase') == 'select_evidence'
                    and row['status'] == 'observed']
        self.assertEqual(observed, ['compare_current_leads'])

    def test_focused_menu_stops_after_concordant_measurement(self):
        actions = [
            reply('select_evidence', dict(option='compare_current_leads', reason='check new high')),
            reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1','E2'], reason='both high')),
            act('CONFIRM_EVENT', None), boundary('unlocalized'),
        ]
        result, _, _, steps, requests = self.run_persistent_episode(actions, [[.94,.05]], focused=True)
        self.assertEqual(result['intervals'], [[0,10]])
        self.assertEqual(len(requests), 4)
        self.assertTrue(any(row['status'] == 'concordant_evidence_stop' for row in steps))

    def test_joint_measurement_rebuilds_visible_citations_and_preserves_close(self):
        actions = []
        for finding, event_action, event_id in (
            ('TARGET_PRESENT', 'CONFIRM_EVENT', None),
            ('TARGET_ABSENT', 'REQUEST_CLOSE', 1),
            ('TARGET_ABSENT', 'REQUEST_CLOSE', 1),
        ):
            actions.extend([
                reply('request_more_evidence', dict(option='compare_current_leads', reason='check change')),
                reply('assess_window', dict(finding=finding, evidence_ids=['E1','E2'], reason='both leads')),
                act(event_action, event_id),
            ])
        actions.append(reply('confirm_event_end', dict(decision='CONFIRM_END', reason='two contrary intervals')))
        result, _, _, steps, requests = self.run_persistent_episode(
            actions, [[.94,.01],[.12,.05],[.16,.08]], pairs=True, joint=True
        )
        offered = [[t['function']['name'] for t in row['tools']] for row in requests]
        self.assertEqual(offered.count(['assess_window','request_more_evidence']), 3)
        self.assertEqual(offered.count(['assess_window']), 3)
        self.assertNotIn(['finish_measurements'], offered)
        self.assertEqual(len(actions), len(requests))
        acquired = [row for row in steps if row.get('phase')=='request_more_evidence' and row['status']=='observed']
        self.assertEqual(len(acquired), 3)
        self.assertEqual(result['events'][0]['end_decided_at_s'], 30)
        self.assertEqual(result['intervals'], [[0,10]])

    def test_rejected_joint_request_keeps_a_terminal_assessment(self):
        bad = reply('request_more_evidence', dict(option='future_window', reason='not an available option'))
        actions = [bad, bad,
                   reply('assess_window', dict(finding='CANNOT_TELL', evidence_ids=['E1'], reason='single lead')),
                   act('NO_EVENT_CHANGE', None)]
        result, trace, _, steps, requests = self.run_persistent_episode(
            actions, [[.94,.01]], pairs=True, joint=True
        )
        self.assertEqual(len(actions), len(requests))
        self.assertEqual(result['counts']['model_calls'], 1)
        self.assertEqual(trace[0]['action'], 'NO_EVENT_CHANGE')
        self.assertEqual(sum(row['status']=='stage_rejected' for row in steps), 2)

    def test_focused_export_ends_at_last_supported_interval(self):
        actions = [
            reply('select_evidence', dict(option='compare_current_leads', reason='check new high')),
            reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1','E2'], reason='both high')),
            act('CONFIRM_EVENT', None), boundary('unlocalized'),
            reply('select_evidence', dict(option='compare_current_leads', reason='check possible end')),
            reply('assess_window', dict(finding='TARGET_ABSENT', evidence_ids=['E1','E2'], reason='both low')),
            act('REQUEST_CLOSE', 1),
        ]
        result, _, updates, _, _ = self.run_persistent_episode(actions, [[.94,.05],[.1,.04]], focused=True)
        self.assertEqual(result['intervals'], [[0,10]])
        self.assertTrue(result['events'][0]['right_censored'])
        self.assertIsNone(result['events'][0]['end_decided_at_s'])
        self.assertIsNone(updates[-1]['event_update'])

    def test_onset_menu_prefers_nonoverlapping_shifted_pair_and_filters_conflicts(self):
        from cardioagent.ecg_agent.harness.boundary_pairs import onset_pairs

        def evidence(key, a, b, direction, score, lead='ECG1'):
            return dict(id=key, window_s=[a,b], lead=lead, target_score=score,
                        reference={'direction': direction})
        samples = [
            evidence('left',0,10,'negative',.15),
            evidence('shifted_left',5,15,'negative',.12),
            evidence('other_left',10,20,'negative',.1,'ECG2'),
            evidence('conflicting_left',10,20,'support',.9),
            evidence('shifted_right',15,25,'support',.95),
            evidence('right',20,30,'support',.91),
            evidence('future',25,35,'support',.99),
        ]
        options = onset_pairs(samples, [20,30], 0)
        self.assertEqual(options['pair_1']['range_s'], [5,25])
        offered = {row[side]['id'] for row in options.values() for side in ('before', 'after')}
        self.assertTrue({'shifted_left','shifted_right'} <= offered)
        self.assertFalse({'conflicting_left','other_left','future'} & offered)

    def test_onset_choice_preserves_window_evidence_and_model_authority(self):
        actions = [
            reply('select_evidence', dict(option='compare_current_leads', reason='check new high')),
            reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1','E2'], reason='both high')),
            act('CONFIRM_EVENT', None),
            reply('select_onset_boundary', dict(option='pair_1', reason='negative then positive')),
        ]
        result, trace, _, _, requests = self.run_persistent_episode(actions, [[.12,.04],[.94,.05]], pairs=True)
        self.assertEqual(result['intervals'], [[10,20]])
        refinement = trace[-1]['llm_action']['refinement']
        self.assertEqual(refinement['boundary']['range_s'], [0,20])
        self.assertNotEqual(refinement['before_id'], refinement['after_id'])
        self.assertEqual(len(requests), 4)


    def test_single_new_high_can_create_candidate_and_adjacent_high_can_confirm(self):
        sample = [
            reply('select_evidence', dict(option='finish', reason='use the primary measurement')),
            reply('assess_window', dict(finding='TARGET_PRESENT', evidence_ids=['E1'], reason='high')),
        ]
        actions = sample + [act('CREATE_CANDIDATE', None), boundary('unlocalized')]
        actions += sample + [act('CONFIRM_EVENT', 1), boundary('unlocalized')]
        result, trace, _, _, requests = self.run_persistent_episode(actions, [[.94,.05],[.93,.06]], menu=True)
        self.assertEqual([row['action'] for row in trace], ['CREATE_CANDIDATE','CONFIRM_EVENT'])
        schemas = [request['tools'][0]['function']['parameters']['properties']['action']['enum']
                   for request in requests if request['tools'][0]['function']['name'] == 'choose_event_action']
        self.assertNotIn('CONFIRM_EVENT', schemas[0])
        self.assertIn('CONFIRM_EVENT', schemas[1])
        self.assertEqual(result['intervals'], [[0,20]])

    def test_paired_view_marks_only_opposite_cotimed_leads(self):
        from cardioagent.ecg_agent.harness.window_semantics import paired_observation_view
        def row(key,lead,span,meaning):
            return dict(id=key,lead=lead,window_s=span,empirical_reference={'evidence_for_this_window':meaning})
        rows=[row('E1','ECG1',[10,20],'FAVORS_AF_OR_AFL'),row('E2','ECG2',[10,20],'WEIGHS_AGAINST_AF_AND_AFL'),
              row('H1','ECG2',[0,10],'WEIGHS_AGAINST_AF_AND_AFL')]
        data=dict(window_s=[10,20],vote_on_this_interval=rows[:2],earlier_measurements_for_comparison_only=rows[2:])
        view,_=paired_observation_view(data,'')
        self.assertEqual(view['opposing_same_interval_measurement_ids'],[['E1','E2']])
        self.assertNotIn('evidence_for_this_window',view['vote_on_this_interval'][0]['empirical_reference'])
        self.assertIn('evidence_for_this_window',rows[0]['empirical_reference'])
    def test_direction_only_view_preserves_score_timing_direction_and_archive(self):
        from cardioagent.ecg_agent.harness.window_semantics import direction_only_view, REFERENCE_DESCRIPTION
        row = dict(id='H1', lead='ECG2', window_s=[10,20],fields={'target_score':.855},
                   empirical_reference=dict(direction='support',target_fraction=.9829))
        data = dict(vote_on_this_interval=[],earlier_measurements_for_comparison_only=[row])
        view, prompt = direction_only_view(data, REFERENCE_DESCRIPTION)
        actual, = view['earlier_measurements_for_comparison_only']
        self.assertEqual(actual,dict(row,empirical_reference=dict(evidence_for_this_window='FAVORS_AF_OR_AFL')))
        self.assertIn('target_fraction',row['empirical_reference'])
        self.assertIn('same lead',prompt)
    def test_indeterminate_measurement_never_offers_an_unexecutable_direction(self):
        from cardioagent.ecg_agent.harness.window_semantics import available_findings
        profile=json.loads((fixture.ROOT/'configs/ecg/react_qwen_followup_close.json').read_text())['window_band_profile']
        row=dict(id='E1',tool='score_ecgfounder',lead='ECG2',window_s=[10,20],direction_allowed=True,
                 status='ok',fields=dict(af_score=.667,afl_score=.61))
        self.assertEqual(available_findings({'E1':row},[10,20],profile),['CANNOT_TELL'])
        row['fields']['af_score']=.99
        self.assertIn('TARGET_PRESENT',available_findings({'E1':row},[10,20],profile))
        self.assertEqual(available_findings({'E1':row},[20,30],profile),['CANNOT_TELL'])
    def test_disagreeing_measured_leads_cannot_be_hidden_by_selective_citation(self):
        cfg=json.loads((fixture.ROOT/'configs/ecg/react_qwen_followup_leads.json').read_text())
        rows=[dict(id=k,tool='score_ecgfounder',lead=lead,window_s=[10,20],direction_allowed=True,
                   status='ok',fields=dict(af_score=af,afl_score=.05))
              for k,lead,af in [('E1','ECG1',.98),('E2','ECG2',.10)]]
        data=dict(window_s=[10,20],cursor_s=20,primary_lead='ECG1',measurements=rows)
        model=ScriptModel([reply('assess_window',dict(finding='TARGET_PRESENT',evidence_ids=['E1'],reason='one lead high')),
                           reply('assess_window',dict(finding='CONFLICTING_MEASUREMENTS',evidence_ids=['E1','E2'],reason='opposing directions'))])
        steps=[]
        out=assess(model,Counter(),lambda _,s:steps.append(s),{},data,{'E1':'hash1','E2':'hash2'},
                   lambda x:True,lambda x:True,'union_reference',cfg['window_band_profile'],True)
        self.assertEqual([s['status'] for s in steps],['stage_rejected','stage_committed'])
        self.assertEqual(out['interpretation'],'conflict')
    def test_overlaps_gaps_conflicts_and_future(self):
        def row(a,b,d):
            return dict(id=f'{a}-{b}-{d}',window_s=[a,b], reference=dict(direction=d))
        support=row(0,10,'support')
        neg=row(10,20,'negative')
        later=row(20,30,'negative')
        def eligible(seq):
            return end_evidence(seq,[20,30],dict(estimated_start_s=0))['can_confirm']
        self.assertTrue(eligible([later,support,neg]))
        for seq in ([support,neg,neg],[support,later],[support,row(15,25,'negative'),later],
                    [support,neg,row(20,30,'support'),later],[support,neg,row(30,40,'negative')]):
            self.assertFalse(eligible(seq))

    def test_full_runtime_preserves_identity_and_projects_earlier_boundary(self):
        def observe(result):
            return reply('assess_window', dict(finding=result,evidence_ids=['E1'],reason='current evidence'))
        cfg=fixture.ROOT/'configs/ecg/react_qwen_followup_close.json'
        for recovers in (True,False):
            actions=[fixture.measure(0),fixture.FINISH,observe('TARGET_PRESENT'),act('CONFIRM_EVENT',None),
                     boundary('unlocalized'),fixture.measure(10),fixture.FINISH,observe('TARGET_ABSENT'),
                     act('REQUEST_CLOSE',1),fixture.measure(20),fixture.FINISH]
            if recovers:
                actions.extend([observe('TARGET_PRESENT'),act('CONTINUE_EVENT',1)])
            else:
                actions.extend([observe('TARGET_ABSENT'),act('REQUEST_CLOSE',1),
                                reply('confirm_event_end',dict(decision='CONFIRM_END',reason='continued contrary measurements'))])
            result,trace,*_=fixture.EpisodeTests().replay('fixed',actions,3,config_file=cfg,
                              score_pairs=[[.99,.1],[.1,.02],[.97 if recovers else .12,.03]])
            event,=result['events']
            self.assertEqual(trace[1]['state'],'OPEN')
            self.assertEqual(event['final_status'],'RIGHT_CENSORED' if recovers else 'CLOSED')
            self.assertEqual(event['end_decided_at_s'],None if recovers else 30)
            if not recovers:
                self.assertEqual(event['estimated_end_s'],10)
                refinement=trace[2]['llm_action']['refinement']
                anchors=event['end_evidence_by_role']
                self.assertEqual(anchors,dict(last_support=refinement['before_id'],
                                             first_against=refinement['after_id'],
                                             confirming_against=refinement['confirmation_evidence_id']))
                self.assertEqual(len(set(anchors.values())),3)
                self.assertTrue(set(anchors.values()).issubset(event['end_evidence_ids']))
