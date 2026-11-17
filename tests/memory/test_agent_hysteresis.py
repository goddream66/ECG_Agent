import unittest
from cardioagent.ecg_agent.memory.events import EventConfig, EventTracker
from cardioagent.ecg_agent.policies.rule_review import decide_event
from cardioagent.ecg_agent.experiments.runner import configured_variants


class AgentHysteresisTests(unittest.TestCase):
    def config(self):return EventConfig(threshold=.6,continue_threshold=.4)

    def test_validation_and_old_serialization(self):
        self.assertNotIn('continue_threshold',EventConfig().to_dict())
        self.assertEqual(EventConfig(**self.config().to_dict()),self.config())
        for value in (0,.7,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):EventConfig(threshold=.6,continue_threshold=value)

    def test_candidate_and_unknown_do_not_use_lower_threshold_before_confirmation(self):
        c=self.config();t=EventTracker(c)
        self.assertEqual(c.threshold_for(t),.6)
        t.observe(0,10,'support');self.assertEqual(c.threshold_for(t),.6)
        t.observe(10,20,'unavailable');self.assertEqual(t.state,'UNKNOWN_HOLD')
        self.assertEqual(c.threshold_for(t),.6)
        t.observe(20,30,'unavailable');self.assertIsNone(t.active)
        self.assertEqual(c.threshold_for(t),.6)

    def test_confirmed_pending_end_and_unknown_keep_lower_threshold(self):
        c=self.config();t=EventTracker(c)
        t.observe(0,10,'support');t.observe(10,20,'support')
        self.assertEqual(c.threshold_for(t),.4)
        t.observe(20,30,'negative');self.assertEqual(t.state,'PENDING_END')
        self.assertEqual(c.threshold_for(t),.4)
        t.observe(30,40,'support');t.observe(40,50,'unavailable')
        self.assertEqual(c.threshold_for(t),.4)
        t.observe(50,60,'support');t.observe(60,70,'negative');t.observe(70,80,'negative')
        self.assertIsNone(t.active);self.assertEqual(c.threshold_for(t),.6)

    def call(self,primary,secondary,tracker):
        calls=[]
        def response(tool,lead):
            calls.append((tool,lead))
            return {'status':'ok','lead':lead,'result':{'reasons':[]} if tool=='assess_quality' else {'af_score':secondary,'afl_score':0.}}
        p={'status':'ok','lead':'ECG1','result':{'af_score':primary,'afl_score':0.}}
        return decide_event(response,p,'ECG2',tracker,self.config()),calls

    def test_both_channels_use_start_threshold_for_candidate(self):
        t=EventTracker(self.config());t.observe(0,10,'support')
        decision,calls=self.call(.65,.55,t)
        self.assertEqual(decision[0],'conflict');self.assertIn(('score_ecgfounder','ECG2'),calls)

    def test_both_channels_use_continue_threshold_and_boundaries(self):
        t=EventTracker(self.config());t.observe(0,10,'support');t.observe(10,20,'support')
        t.since_check=2
        decision,calls=self.call(.55,.4,t)
        self.assertEqual(decision[0],'support');self.assertIn(('score_ecgfounder','ECG2'),calls)
        t.since_check=2
        self.assertEqual(self.call(.399,.399,t)[0][0],'negative')

    def test_end_to_end_score_sequence_preserves_event_until_two_low_windows(self):
        c=self.config();t=EventTracker(c)
        for i,value in enumerate((.7,.8,.55,.45,.2,.2)):
            decision,_=self.call(value,value,t)
            t.observe(i*10,(i+1)*10,decision[0],checked=decision[1])
        self.assertEqual(t.intervals,[[0,40]])
        self.assertEqual(t.events[0]['end_decided_at_s'],60)

    def test_variant_selection_is_explicit_and_does_not_silently_change_baselines(self):
        c=self.config()
        self.assertEqual([x[0] for x in configured_variants('live',c,['event-adaptive'])],['event-adaptive'])
        with self.assertRaises(ValueError):configured_variants('live',c)
        with self.assertRaises(ValueError):configured_variants('live',c,['missing'])
        self.assertEqual(len(configured_variants('live',EventConfig())),2)


if __name__=='__main__':unittest.main()
