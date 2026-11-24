"""Reuse only token-identical prompt prefixes within one window review."""


def common_prefix(old, new):
    """Leave a new input token for the first generation step."""
    bound=min(len(old),max(0,len(new)-1))
    for i in range(bound):
        if old[i]!=new[i]:
            return i
    return bound


class PromptPrefix:
    def __init__(self, max_tokens):
        self.max_tokens=max_tokens
        self.tokens=[]
        self.cache=None

    def consume(self, tokens):
        length=common_prefix(self.tokens,tokens)
        previous,self.cache=self.cache,None
        self.tokens=[]
        if previous is None or length<128:
            return None,0
        previous.crop(length)
        return previous,length

    def store(self, tokens, generated):
        from transformers import DynamicCache
        cache=generated.past_key_values
        if type(cache) is not DynamicCache or any(layer.is_sliding for layer in cache.layers):
            return
        size=min(self.max_tokens,len(tokens),cache.get_seq_length())
        if size<128:
            return
        # A normal crop is a view. Keep compact independent buffers so a long tool transcript's
        # full KV backing storage is released before the next signal-processing tool call.
        states=tuple((k[:,:,:size,:].clone(),v[:,:,:size,:].clone()) for k,v in cache.to_legacy_cache())
        self.cache=DynamicCache.from_legacy_cache(states)
        self.tokens=tokens[:size]
