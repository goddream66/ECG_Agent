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


def measurement_menu(state, calls, *, live_scoring):
    """Offer only distinct measurements of arrived, retained samples."""
    start, end = state['window_s']
    primary = state['primary_lead']
    alternatives = [lead for lead in state['available_leads'] if lead != primary]
    other = alternatives[0] if alternatives else None
    seen = {(row['tool'], row['lead'], tuple(row['window_s'] or ())) for row in calls}
    options = {}

    def offer(name, tool, lead, left, right):
        if lead and (tool, lead, (left, right)) not in seen:
            options[name] = dict(tool=tool, lead=lead, start_s=left, end_s=right)

    offer('compare_current_leads', 'score_ecgfounder', other, start, end)
    offer('check_primary_quality', 'assess_quality', primary, start, end)
    offer('check_other_quality', 'assess_quality', other, start, end)
    offer('check_primary_rr', 'get_rate_rhythm', primary, start, end)
    if live_scoring and start-5 >= state['retained_window_s'][0]:
        offer('check_shifted_primary', 'score_ecgfounder', primary, start-5, end-5)
    return options


def onset_supported(timeline, window):
    """An onset can be corroborated by another lead or a consecutive window."""
    start, end = window
    current = {row['lead'] for row in timeline if row['window_s'] == [start, end]
               and row['reference']['direction'] == 'support'}
    if len(current) >= 2:
        return True
    previous = [2*start-end, start]
    return any(row['lead'] in current and row['window_s'] == previous
               and row['reference']['direction'] == 'support' for row in timeline)
