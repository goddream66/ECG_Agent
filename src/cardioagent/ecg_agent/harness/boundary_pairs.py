"""Build causal, measured alternatives for model-owned onset localization."""

from collections import defaultdict


def onset_pairs(timeline, window, earliest, confirmed_at=None, *, limit=4):
    """Select ordered, non-overlapping observations with opposite directions."""
    start, cursor = window
    grouped = defaultdict(set)
    available = []
    for row in timeline:
        a, b = row['window_s']
        if not earliest <= a < b <= cursor:
            continue
        grouped[a, b].add(row['reference']['direction'])
        available.append(row)
    negative, positive = [], []
    for row in available:
        a, b = row['window_s']
        direction = row['reference']['direction']
        other = 'support' if direction == 'negative' else 'negative'
        if other in grouped[a, b]:
            continue
        if direction == 'negative':
            negative.append(row)
        if direction == 'support' and (
            b > start if confirmed_at is None else a < confirmed_at <= b
        ):
            positive.append(row)
    candidates = []
    used = set()
    for left in negative:
        a, b = left['window_s']
        for right in positive:
            c, d = right['window_s']
            if not b <= c <= b + 10:
                continue
            key = a, d, left['lead'], right['lead']
            if key in used:
                continue
            used.add(key)
            rank = (left['lead'] != right['lead'], c-b, abs(c-start), -min(
                1.0-left['target_score'], right['target_score']))
            candidates.append((rank, dict(range_s=[a,d], before=left, after=right)))
    candidates.sort(key=lambda item: item[0])
    return {f'pair_{i}': item for i, (_, item) in enumerate(candidates[:limit], 1)}
