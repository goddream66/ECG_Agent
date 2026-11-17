"""State-aware rule review; only explicitly acquired results enter decisions."""
from cardioagent.ecg_agent.policies.labels import score_label


def decide_event(call, primary, secondary, tracker, config):
    label = score_label(primary, config.threshold)
    used = 0

    def review(tool, lead):
        nonlocal used
        if used >= config.review_budget:
            return None
        used += 1
        return call(tool, lead)

    def usable(quality):
        return quality is not None and quality['status'] == 'ok' and not quality['result'].get('reasons', ['missing'])

    def other_result():
        other = review('score_ecgfounder', secondary)
        if other is None:
            return None, 'review_budget_exhausted'
        other_label = score_label(other, config.threshold)
        if other_label is None:
            return None, 'secondary_unavailable'
        if not usable(review('assess_quality', secondary)):
            return None, 'secondary_quality_or_budget_insufficient'
        return other_label, 'secondary_usable'

    if label is None:
        other, reason = other_result()
        return ('unavailable', False, reason) if other is None else (
            'support' if other else 'negative', True, 'single_usable_secondary')
    score = max(primary['result']['af_score'], primary['result']['afl_score'])
    if not label and tracker.active is None and score <= config.threshold - config.borderline_margin:
        return 'negative', False, 'primary_negative_without_event'
    quality = review('assess_quality', primary['lead'])
    if not usable(quality):
        other, reason = other_result()
        return ('unavailable', False, reason) if other is None else (
            'support' if other else 'negative', True, 'primary_unusable_secondary_only')
    borderline = abs(score - config.threshold) < config.borderline_margin
    needs_check = (borderline or tracker.state in ('CANDIDATE', 'UNKNOWN_HOLD', 'PENDING_END')
                   or (tracker.active is not None and not label)
                   or (tracker.state == 'ACTIVE' and tracker.since_check + 1 >= config.recheck_windows))
    if not needs_check:
        return 'support' if label else 'negative', False, 'usable_primary_stop_review'
    other, reason = other_result()
    if other is None:
        return 'unavailable', False, reason
    if label != other:
        return 'conflict', False, 'usable_channel_disagreement'
    return 'support' if label else 'negative', True, 'usable_channels_agree'
