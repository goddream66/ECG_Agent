"""Frozen AF/AFL score interpretation, shared by current and baseline policies."""

def score_label(result, threshold):
    if result is None or result['status'] != 'ok':
        return None
    return max(result['result']['af_score'], result['result']['afl_score']) >= threshold
