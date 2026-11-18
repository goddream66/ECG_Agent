"""Provider-neutral direct event relationship judgment."""

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PROMPT = Path(__file__).resolve().parents[1] / 'prompts/gap_relationship_v1.txt'
RELATIONS = ('continuous', 'separate', 'uncertain')


class GapJudgment(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    relation: Literal['continuous', 'separate', 'uncertain']
    evidence: list[str]
    support: list[str]
    reason: str = Field(min_length=1, max_length=280)


class GapRelationshipReviewer:
    def __init__(self, model):
        self.model = model
        self.prompt = PROMPT.read_text(encoding='utf-8')

    def judge(self, state):
        tools = [{'type': 'function', 'function': {'name': 'submit_gap_judgment',
                  'description': 'Judge the event relationship and select supporting observation IDs.',
                  'parameters': GapJudgment.model_json_schema()}}]
        messages = [{'role': 'system', 'content': self.prompt}, {'role': 'user',
                     'content': json.dumps(state, allow_nan=False, separators=(',', ':'))}]
        began = perf_counter()
        result = dict(status='error', relation='uncertain', evidence=[], support=[], usage={})
        try:
            reply = self.model.complete(messages, tools)
            result['usage'] = reply.usage
            result['response'] = reply.message
            calls = reply.message.get('tool_calls', [])
            if reply.message.get('role') != 'assistant' or len(calls) != 1:
                raise ValueError('Expected one assistant tool call')
            call = calls[0]['function']
            if call['name'] != 'submit_gap_judgment':
                raise ValueError('Unexpected tool')
            args = call['arguments']
            judgment = GapJudgment.model_validate_json(args) if isinstance(args, str) else GapJudgment.model_validate(args)
            ids = {item['id'] for item in state['observations']}
            if not set(judgment.evidence + judgment.support) <= ids:
                raise ValueError('Unknown observation ID')
            if judgment.relation != 'uncertain' and not (judgment.evidence or judgment.support):
                raise ValueError('Decisive judgment needs an observation')
            result.update(judgment.model_dump(), status='ok')
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:320])
        result['wall_s'] = perf_counter() - began
        return result
