"""ECGFounder scoring on a cache window already admitted by ToolSession."""

import numpy as np


def score_ecgfounder(session, args):
    payload = {'af_score': None, 'afl_score': None}
    if session.cached_model is None:
        return payload, 'insufficient', ['No ECGFounder cache attached']
    values, predict = session.cached_model(args.lead, args.start_s, args.end_s)
    if values is None:
        return payload, 'insufficient', ['No usable cached samples for the requested channel/window']
    if values.shape != (5000,) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Invalid ECGFounder cache samples')
    af, afl = map(float, predict(values.copy()))
    if not (0 <= af <= 1 and 0 <= afl <= 1):
        raise ValueError('Expected finite AF/AFL scores in [0,1]')
    return {'af_score': af, 'afl_score': afl}, 'ok', ['Scores are uncalibrated']
