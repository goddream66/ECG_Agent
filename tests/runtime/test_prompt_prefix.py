from unittest import TestCase

from cardioagent.ecg_agent.llm.prefix_cache import PromptPrefix, common_prefix
from cardioagent.ecg_agent.llm.providers import LocalQwenModel


class PromptPrefixTests(TestCase):
    def test_exact_and_partial_prefixes_leave_a_generation_token(self):
        self.assertEqual(common_prefix([1,2,3],[1,2,3,4]),3)
        self.assertEqual(common_prefix([1,2,3],[1,2,3]),2)
        self.assertEqual(common_prefix([1,2,3],[1,4,3]),1)
        self.assertEqual(common_prefix([1,2],[]),0)

    def test_mismatched_suffix_is_removed_and_cache_is_consumed(self):
        class Cache:
            def crop(self,n): self.length=n

        old=list(range(200))
        scope=PromptPrefix(256)
        scope.tokens=old[:]
        scope.cache=Cache()
        past,amount=scope.consume(old[:180]+[1000,1001])
        self.assertEqual((past.length,amount),(180,180))
        self.assertIsNone(scope.cache)
        self.assertEqual(scope.tokens,[])

    def test_exception_resets_review_scope(self):
        model=LocalQwenModel.__new__(LocalQwenModel)
        model.prefix_cache_tokens=256
        model._prompt_prefix=None
        with self.assertRaisesRegex(RuntimeError,'test'):
            with model.prefix_scope():
                self.assertIsNotNone(model._prompt_prefix)
                raise RuntimeError('test')
        self.assertIsNone(model._prompt_prefix)
