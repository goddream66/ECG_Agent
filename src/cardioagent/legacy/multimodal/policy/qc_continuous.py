"""V2.4.2 continuous, subject-balanced QC distributions (opt-in)."""
from collections import Counter, defaultdict
import numpy as np
from cardioagent.legacy.multimodal.policy.targets import subsets


def features(profile, observed):
    anchor = observed.get(profile.initial, {})
    signal, event = anchor.get('signal_quality') or {}, anchor.get('event_quality') or {}
    values = [anchor.get('quality') == 'usable', anchor.get('quality') == 'review']
    for value in [signal.get(k) for k in ('baseline_power_fraction', 'high_frequency_power_fraction', 'physiology_band_power_fraction')] + [event.get(k) for k in ('interval_mad_ratio', 'plausible_interval_fraction', 'autocorrelation')] + [anchor.get('event_score_median')]:
        valid = value is not None and np.isfinite(value)
        values.extend([float(np.clip(value, -3, 3)) if valid else 0., float(valid)])
    rates = [v['rate_bpm'] for v in observed.values() if v.get('rate_bpm') is not None]
    values.extend([min(np.ptp(rates) / 30, 3) if len(rates) > 1 else 0., len(rates) > 1])
    return values


def fit(cases, budget):
    groups = defaultdict(list)
    for row, profile in cases:
        if row['split'] != 'train':
            raise ValueError('Continuous QC requires training subjects')
        initial = (profile.initial,) if profile.initial else ()
        for extra in subsets(tuple(a for a in profile.options if a not in initial)):
            actions = initial + extra
            if len(actions) >= budget:
                continue
            x = features(profile, {a: profile.measurements[a] for a in actions})
            for a, (_, _, slot) in profile.options.items():
                if a not in actions:
                    groups[f'{profile.task}/{slot}'].append({'x': x, 'subject': row['subject_id'], 'sample': {k:v for k,v in profile.previews[a].items() if k != 'duration_ms'}})
    return dict(groups)


def distribution(groups, profile, observed, slot, min_subjects=5):
    rows = groups.get(f'{profile.task}/{slot}', [])
    if len({r['subject'] for r in rows}) < min_subjects:
        return None
    x = np.asarray([r['x'] for r in rows])
    scale = np.maximum(np.std(x, axis=0), .15)
    distance = np.mean(((x - features(profile, observed)) / scale) ** 2, axis=1)
    counts = Counter(r['subject'] for r in rows)
    base = np.array([1 / counts[r['subject']] for r in rows])
    # Smooth shrinkage to subject-balanced marginal; no test labels or hidden QC.
    weights = base * (.2 + .8 * np.exp(-distance / 2))
    order = np.argsort([r['sample']['score'] for r in rows], kind='stable')
    cdf = np.cumsum(weights[order]); cdf /= cdf[-1]
    return [rows[order[min(np.searchsorted(cdf, q), len(rows)-1)]]['sample'] for q in (.125,.375,.625,.875)]
