"""One structured model call for an already observed event pair."""

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PROMPT = Path(__file__).resolve().parents[1] / 'prompts/episode_review_v1.txt'


class MissingEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    start_s: float
    lead: Literal['ECG1', 'ECG2']
    tool: Literal['score_ecgfounder', 'assess_quality']


class EpisodeRecommendation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['keep_separate', 'merge', 'link_only', 'insufficient']
    evidence_ids: list[str] = Field(max_length=24)
    reason: str = Field(min_length=1, max_length=400)
    missing_evidence: MissingEvidence | None = None


def validate_recommendation(recommendation, packet):
    ids = set(recommendation.evidence_ids)
    if not ids <= packet['evidence_index'].keys():
        raise ValueError('unknown_evidence_reference')
    if recommendation.action in ('merge', 'link_only') and not ids:
        raise ValueError('missing_evidence_reference')
    if packet['sustained_opposition'] and recommendation.action in ('merge', 'link_only'):
        raise ValueError('link_despite_sustained_opposition')
    if recommendation.action == 'merge' and not all(
        any(set(pair) <= ids for pair in choices) for choices in packet['gap_support']
    ):
        raise ValueError('merge_without_full_gap_support')
    request = recommendation.missing_evidence
    if request is not None:
        a, b = packet['gap_s']
        matches = [win for win in packet['windows'] if win['window_s'][0] == request.start_s
                   and a <= request.start_s < b]
        if not matches:
            raise ValueError('requested_window_outside_gap')
        lead = matches[0]['leads'][request.lead]
        field = 'score_id' if request.tool == 'score_ecgfounder' else 'quality_id'
        if field in lead:
            raise ValueError('requested_evidence_already_acquired')


class EpisodeReviewer:
    def __init__(self, model):
        self.model = model
        self.prompt = PROMPT.read_text(encoding='utf-8')

    def review(self, packet):
        payload = {key: value for key, value in packet.items() if key != 'evidence_index'}
        messages = [{'role': 'system', 'content': self.prompt},
                    {'role': 'user', 'content': json.dumps(payload, allow_nan=False, separators=(',', ':'))}]
        tools = [{'type': 'function', 'function': {'name': 'submit_episode_review',
                  'description': 'Submit one evidence-based recommendation for this event pair.',
                  'parameters': EpisodeRecommendation.model_json_schema()}}]
        result = {'action': 'insufficient', 'status': 'ok', 'usage': {}}
        started = perf_counter()
        try:
            response = self.model.complete(messages, tools)
            result['usage'] = {name: response.usage.get(name, 0)
                               for name in ('prompt_tokens', 'completion_tokens')}
            calls = response.message.get('tool_calls', [])
            if len(calls) != 1 or response.message.get('role') != 'assistant':
                raise ValueError('expected_one_assistant_tool_call')
            fn = calls[0]['function']
            if fn['name'] != 'submit_episode_review':
                raise ValueError('wrong_tool_name')
            args = fn['arguments']
            decision = EpisodeRecommendation.model_validate_json(args) if isinstance(args, str) else (
                EpisodeRecommendation.model_validate(args))
            result['proposal'] = decision.model_dump()
            validate_recommendation(decision, packet)
            result.update(action=decision.action,
                          referenced_evidence_ids=[packet['evidence_index'][key]
                                                   for key in decision.evidence_ids])
        except Exception as exc:
            result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:400])
        result['wall_s'] = perf_counter() - started
        return result
