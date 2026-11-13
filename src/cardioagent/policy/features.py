"""Versioned numerical state with explicit availability flags."""

import numpy as np

from .complementarity import WIDTH, context_features
from .state import SLOTS


def numeric(state, summary):
    values = state.numeric(summary).tolist()
    by_slot = {slot: action for action, (_, _, slot) in state.options.items()}
    for slot in SLOTS:
        action = by_slot.get(slot)
        item = state.evidence.get(action, {})
        event = item.get("event_quality") or {}
        signal = item.get("signal_quality") or {}
        values.extend([action in state.previews, item.get("rate_bpm") is not None,
                       item.get("status") == "insufficient", item.get("status") == "failed",
                       *[event.get(k) is not None for k in ("interval_mad_ratio", "plausible_interval_fraction", "autocorrelation")]])
        for value in (item.get("event_score_median"), signal.get("baseline_power_fraction"),
                      signal.get("high_frequency_power_fraction"), signal.get("physiology_band_power_fraction")):
            values.extend((0. if value is None else value, value is not None))
    return np.asarray(values + context_features(state), dtype=np.float32)


def ablate(values, *, task=True, context=True):
    if task and context:
        return values
    result = values.copy()
    if not task:
        result[..., 0] = 0.
    if not context:
        result[..., -WIDTH:] = 0.
    return result
