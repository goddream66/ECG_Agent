"""Conservative review routing for offline MC-MED labels, not rhythm diagnosis."""

POLICY = 'hard-invalid-skip-correlation-review-v2'


def summarize_quality(spans, responses):
    """Only missing/flat spans veto scoring; template flags request human review."""
    quality = next((r for r in responses if r['tool'] == 'assess_template_quality'), None)
    reasons = sorted({reason for _, _, reason in spans})
    if spans:
        status = 'unusable_for_auto_label'
    elif quality is None or quality['status'] != 'ok':
        status = 'review_required'
        reasons.append('template_quality_unavailable')
    else:
        result = quality['result']
        if result.get('template_consistent') is not True:
            reasons.append('template_correlation_low_or_unknown')
        reasons.extend('prescreen:' + key for key, value in result.get('prescreen_flags', {}).items()
                       if value is True)
        status = 'review_required' if reasons else 'no_quality_flag'
    return {'policy': POLICY, 'status': status, 'skip_founder': bool(spans),
            'reasons': reasons,
            'quality_evidence_id': quality['evidence_id'] if quality else None,
            'limitations': 'Template inconsistency is not proof of noise; no rhythm veto applied.'}
