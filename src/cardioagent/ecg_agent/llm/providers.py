"""Qwen adapters: OpenAI-compatible endpoint or lazy local Transformers execution."""
from copy import deepcopy
from contextlib import contextmanager
import json
import os
import re
from urllib.request import Request, urlopen
from .contracts import ModelReply


class EndpointModel:
    def __init__(self, config):
        self.config = dict(config)

    def complete(self, messages, tools):
        c = self.config
        body = dict(model=c['model'], messages=messages, tools=tools, tool_choice='required',
                    temperature=0, max_tokens=c.get('max_new_tokens', 512),
                    chat_template_kwargs={'enable_thinking': False})
        headers = {'Content-Type': 'application/json'}
        token = os.environ.get(c.get('api_key_env', 'ECG_LLM_API_KEY'))
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = Request(c['base_url'].rstrip('/') + '/chat/completions',
                      data=json.dumps(body, allow_nan=False).encode(), headers=headers)
        with urlopen(req, timeout=c.get('timeout_s', 60)) as response:
            data = json.load(response)
        return ModelReply(data['choices'][0]['message'], data.get('usage', {}))


class LocalQwenModel:
    def __init__(self, config):
        # Loading only happens when this explicit provider is selected.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = dict(config)
        device = config['device']
        if device != 'cpu' and not re.fullmatch(r'cuda:\d+', device):
            raise ValueError('Select cpu or one explicit cuda:N device')
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'], local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32 if device == 'cpu' else torch.bfloat16).to(device).eval()
        self.device = device
        self.prefix_cache_tokens=config.get('prefix_cache_tokens',0)
        if type(self.prefix_cache_tokens) is not int or not 0<=self.prefix_cache_tokens<=4096:
            raise ValueError('prefix_cache_tokens must be between 0 and 4096')
        self._prompt_prefix=None

    @contextmanager
    def prefix_scope(self, *, limit=None):
        from .prefix_cache import PromptPrefix
        limit=self.prefix_cache_tokens if limit is None else limit
        if self._prompt_prefix is not None or type(limit) is not int or not 0<=limit<=4096:
            raise ValueError('Invalid or nested prompt prefix scope')
        if limit:
            self._prompt_prefix=PromptPrefix(limit)
        try:
            yield
        finally:
            self._prompt_prefix=None

    def complete(self, messages, tools):
        import torch
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True, enable_thinking=False)
        scope=self._prompt_prefix
        tokens=inputs['input_ids'][0].tolist() if scope is not None else None
        inputs=inputs.to(self.device)
        budget=self.config.get('max_new_tokens',512)
        reused=0
        with torch.inference_mode():
            if scope is None:
                output=self.model.generate(**inputs,do_sample=False,max_new_tokens=budget)
            else:
                previous,reused=scope.consume(tokens)
                if previous is None:
                    generated=self.model.generate(**inputs,do_sample=False,max_new_tokens=budget,
                                                  return_dict_in_generate=True)
                else:
                    generated=self.model.generate(**inputs,do_sample=False,max_new_tokens=budget,
                                                  past_key_values=previous,return_dict_in_generate=True)
                output=generated.sequences
                try:
                    scope.store(tokens,generated)
                except torch.cuda.OutOfMemoryError:
                    # The generation already succeeded. Release the cache and keep its original output.
                    scope.cache=None
                    scope.tokens=[]
        n = inputs['input_ids'].shape[-1]
        raw = self.tokenizer.decode(output[0, n:], skip_special_tokens=True)
        self.last_raw = raw
        usage={'prompt_tokens':n,'completion_tokens':output.shape[-1]-n}
        if scope is not None:
            usage['prefix_cache_reused_tokens']=reused
        return ModelReply(parse_qwen_tool_reply(raw),usage)


def parse_qwen_tool_reply(raw):
    # Qwen's native Hermes tool-call format. Never execute generated Python/text.
    matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.S)
    if len(matches) != 1:
        raise ValueError('Expected exactly one complete tool call')
    item = json.loads(matches[0])
    if set(item) != {'name', 'arguments'} or not isinstance(item['arguments'], dict):
        raise ValueError('Invalid Qwen tool-call object')
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'local-call', 'type': 'function', 'function': {
            'name': item['name'], 'arguments': json.dumps(item['arguments'], allow_nan=False)}}]}


_MODELS = {}


def build_chat_model(config):
    """Reuse the loaded local model per worker; never share patient conversations."""
    c = deepcopy(config)
    provider = c.pop('provider')
    c.pop('max_rounds', None)
    c.pop('llm_tool_max_calls', None)
    c.pop('llm_window_interpretation', None)
    c.pop('investigation_checkpoint', None)
    c.pop('event_review_mode', None)
    c.pop('fact_history', None)
    c.pop('compact_final_evidence', None)
    c.pop('split_event_review', None)
    c.pop('measured_event_review', None)
    c.pop('staged_event_review', None)
    c.pop('selective_lifecycle', None)
    c.pop('chronological_history_ids', None)
    c.pop('acquire_or_assess', None)
    c.pop('window_assessment_mode', None)
    c.pop('window_band_profile', None)
    c.pop('separate_final_decision', None)
    c.pop('rule_safety_fallback', None)
    c.pop('history_windows', None)
    c.pop('memory_write', None)
    c.pop('memory_retrieve', None)
    c.pop('boundary_tools', None)
    c.pop('event_control', None)
    c.pop('unified_event_decision', None)
    c.pop('scheduled_evidence', None)  # Ignore the retired option in archived configurations.
    key = json.dumps([provider, c], sort_keys=True)
    if provider == 'openai-compatible':
        return EndpointModel(c)
    if provider != 'transformers':
        raise ValueError('Unknown LLM provider')
    if key not in _MODELS:
        _MODELS[key] = LocalQwenModel(c)
    return _MODELS[key]
