"""Bounded Qwen measurement selection from a frozen event-review packet."""

import json
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field


PROMPT = Path(__file__).resolve().parents[1] / 'prompts/shifted_evidence_review_v1.txt'


class ReviewAction(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: str = Field(pattern=r'^(STOP|A[1-9][0-9]*)$')


class ShiftedEvidenceReviewer:
    def __init__(self, model):
        self.model = model
        self.prompt = PROMPT.read_text(encoding='utf-8')

    def choose(self, state, actions):
        menu = {f'A{i}': action for i, action in enumerate(actions, 1)}
        tools = [{'type': 'function', 'function': {'name': 'choose_measurement',
                  'description': 'Select one listed action ID or STOP.',
                  'parameters': ReviewAction.model_json_schema()}}]
        messages = [{'role': 'system', 'content': self.prompt},
                    {'role': 'user', 'content': json.dumps(dict(state, actions=menu),
                                                          allow_nan=False, separators=(',', ':'))}]
        result = {'action': None, 'status': 'error', 'usage': {}}
        began = perf_counter()
        try:
            reply = self.model.complete(messages, tools)
            result['usage'] = reply.usage
            calls = reply.message.get('tool_calls', [])
            if reply.message.get('role') != 'assistant' or len(calls) != 1:
                raise ValueError('Expected one assistant tool call')
            fn = calls[0]['function']
            if fn['name'] != 'choose_measurement':
                raise ValueError('Unexpected tool')
            args = fn['arguments']
            selection = (ReviewAction.model_validate_json(args) if isinstance(args, str)
                         else ReviewAction.model_validate(args)).action
            if selection != 'STOP' and selection not in menu:
                raise ValueError('Action is outside the current allowlist')
            result.update(action=selection, status='ok')
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:300])
        result['wall_s'] = perf_counter() - began
        result['request'] = dict(state, actions=menu)
        return (menu.get(result['action']), result)
