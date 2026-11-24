"""Route stable observations directly and reserve extra measurements for review."""

from math import isfinite


SUPPLEMENTARY_TOOLS = frozenset({"score_ecgfounder", "assess_quality", "get_rate_rhythm"})


def stable_action(primary, state, profile):
    """A stable batch can maintain the current state but cannot open or close it."""
    if state['candidate'] or state['pending_question'] or round(state['cursor_s'] / 10) % 8 == 0:
        return None
    if primary['status'] != 'ok':
        return None
    scores = [primary['result'].get(k) for k in ('af_score', 'afl_score')]
    if any(type(v) not in (int, float) or not isfinite(v) for v in scores):
        return None
    if state['open_event'] and max(scores) >= profile['edges'][-1]:
        return 'support', 'CONTINUE_EVENT'
    if not state['open_event'] and max(scores) < profile['edges'][0]:
        return 'negative', 'NO_EVENT_CHANGE'
    return None


def acquisition_prompt():
    return (
        "Review the arrived ECG for the combined AF-or-AFL target. The current primary ECGFounder "
        "measurement is already available. Choose one additional measurement only if it can change "
        "the window assessment or an open onset/end question. For a new high or low score, compare "
        "the other lead over the same interval. For possible artifact, choose assess_quality. "
        "For ambiguous window placement, request a shifted ten-second score within the retained "
        "interval and no later than the cursor. RR describes timing and cannot alone diagnose AF/AFL. "
        "Do not repeat a finished measurement. Use finish_measurements as soon as more measurements "
        "are unlikely to help. A separate stage will request your judgment and event action."
    )
