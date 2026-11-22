"""Contracts for the LLM-owned window interpretation and event action."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WindowEventDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    action: Literal['KEEP_OPEN', 'DEFER', 'CLOSE']
    interpretation: Literal['support', 'negative', 'conflict', 'unavailable']
    event_id: int | None = Field(ge=1)
    end_s: float | None = Field(ge=0, allow_inf_nan=False)
    evidence_ids: list[str] = Field(max_length=32)
    reason: str = Field(min_length=1, max_length=4096)

    @model_validator(mode='after')
    def compatible(self):
        if (self.action == 'KEEP_OPEN') != (self.interpretation == 'support'):
            raise ValueError('KEEP_OPEN requires support and support requires KEEP_OPEN')
        if self.action == 'CLOSE' and self.interpretation != 'negative':
            raise ValueError('CLOSE requires negative')
        return self


def overlaps_current(row, start, end):
    span = row['window_s']
    return (row['status'] == 'ok' and row['available_at_s'] <= end and span is not None
            and span[0] < end and start < span[1] <= end)


def validate_grounding(submission, known, ledger, start, end, active, boundary_tools):
    if submission.event_id != (active['event_id'] if active else None):
        raise ValueError('Use the current event ID or null when idle')
    rows = []
    for key in submission.evidence_ids:
        row = known.get(key)
        if row is None:
            if ledger is None:
                raise ValueError('Cannot resolve historical citation')
            row = ledger.get(key)
        if row['available_at_s'] > end or (row['window_s'] and row['window_s'][1] > end):
            raise ValueError('Citation has not arrived')
        if row['status'] != 'ok' and submission.interpretation != 'unavailable':
            raise ValueError('Failed measurements can only explain unavailable evidence')
        rows.append(row)
    current = [r for r in rows if r['evidence_id'] in known and overlaps_current(r, start, end)]
    rhythm = [r for r in current if r['tool'] != 'assess_quality']
    if submission.interpretation != 'unavailable' and not rhythm:
        raise ValueError('Cite current successful rhythm evidence or defer as unavailable')
    if submission.interpretation == 'conflict' and len({
            (r['tool'], r['lead'], tuple(r['window_s'])) for r in rhythm}) < 2:
        raise ValueError('Conflict needs distinct current rhythm measurements')
    if submission.action == 'CLOSE':
        for local in rows:
            if (local['evidence_id'] not in known or local['tool'] not in boundary_tools
                    or submission.end_s is None
                    or not local['window_s'][0] <= submission.end_s < local['window_s'][1]):
                continue
            data = local['result']
            if data.get('detector_error') or not (
                    data.get('r_peaks_s') or data.get('beats') or data.get('peak')):
                continue
            if any((r['lead'], r['tool']) != (local['lead'], local['tool']) for r in rhythm):
                return
        raise ValueError('CLOSE needs a measured local boundary and corroborating current rhythm evidence')
