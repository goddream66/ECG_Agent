"""Qwen adapters: OpenAI-compatible endpoint or lazy local Transformers execution."""
from copy import deepcopy
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

    def complete(self, messages, tools):
        import torch
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True, enable_thinking=False).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False,
                                         max_new_tokens=self.config.get('max_new_tokens', 512))
        n = inputs['input_ids'].shape[-1]
        raw = self.tokenizer.decode(output[0, n:], skip_special_tokens=True)
        self.last_raw = raw
        return ModelReply(parse_qwen_tool_reply(raw), {'prompt_tokens': n,
                          'completion_tokens': output.shape[-1] - n})


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
    c.pop('history_windows', None)
    c.pop('boundary_tools', None)
    c.pop('event_control', None)
    c.pop('unified_event_decision', None)
    c.pop('scheduled_evidence', None)
    key = json.dumps([provider, c], sort_keys=True)
    if provider == 'openai-compatible':
        return EndpointModel(c)
    if provider != 'transformers':
        raise ValueError('Unknown LLM provider')
    if key not in _MODELS:
        _MODELS[key] = LocalQwenModel(c)
    return _MODELS[key]
