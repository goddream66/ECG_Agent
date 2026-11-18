"""Typed, provider-neutral ECG tool contracts."""
from dataclasses import dataclass
from typing import Callable, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from cardioagent.common.recording import Waveform


class WindowArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    lead: str = Field(min_length=1, description='Exact ECG lead name in the current block')
    start_s: float = Field(ge=0, description='Absolute start time, seconds')
    end_s: float = Field(gt=0, description='Absolute end time, seconds, no later than cursor')

    @model_validator(mode='after')
    def ordered(self):
        if self.end_s <= self.start_s:
            raise ValueError('end_s must exceed start_s')
        return self


class BeatArgs(WindowArgs):
    commit_start_s: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def commit_inside(self):
        if self.commit_start_s is not None and not self.start_s <= self.commit_start_s < self.end_s:
            raise ValueError('commit_start_s must be inside measurement window')
        return self


class CachedScoreArgs(WindowArgs):
    @model_validator(mode='after')
    def complete_window(self):
        if self.end_s - self.start_s != 10:
            raise ValueError('ECGFounder requires one complete 10-second cache window')
        return self


class ShiftedScoreArgs(CachedScoreArgs):
    offset_s: Literal[-5, 5] = Field(description='Actual window minus original window start, in seconds')

    @model_validator(mode='after')
    def original_start(self):
        if self.start_s - self.offset_s < 0:
            raise ValueError('Original window starts before the recording')
        return self


class ProfileArgs(WindowArgs):
    bins: int = Field(default=6, ge=1, le=12)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    schema_version: int = 1
    evidence_id: str
    tool: str
    tool_version: str
    status: Literal['ok', 'insufficient', 'error']
    window_s: list[float] | None
    available_at_s: float
    lead: str | None
    result: dict
    warnings: list[str]
    error: dict | None
    duration_ms: float = Field(ge=0)
    cached: bool = False


@dataclass(frozen=True)
class EvidenceContext:
    run_id: str
    task_id: str
    record_id: str
    manifest_id: str = ''

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.run_id, self.task_id, self.record_id)):
            raise ValueError('Evidence identity requires nonempty run/task/record IDs')


@dataclass(frozen=True)
class ToolContext:
    waveform: Waveform
    origin_s: float
    cursor_s: float


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[WindowArgs]
    handler: Callable
    experimental: bool = False
    version: str = '1.0.0'
    source: str = 'CardioAgent'

    def schema(self):
        return {'name': self.name, 'description': self.description,
                'parameters': self.arguments.model_json_schema(),
                'experimental': self.experimental, 'version': self.version, 'source': self.source,
                'output_schema': ToolResult.model_json_schema()}
