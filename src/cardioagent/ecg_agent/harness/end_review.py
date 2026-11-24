"""Causal evidence available for confirming an end after a suspected score dip."""
from collections import defaultdict


def end_evidence(timeline, window, active):
    """Find a supported side, first contrary window and a distinct later confirmation.

    Use full, adjacent assessment windows. Co-timed leads cannot count as two
    time points. A gap or mixed directions interrupts the confirming sequence.
    """
    start, end = window
    width = end - start
    if width <= 0:
        raise ValueError('Invalid assessment window')
    by_time = defaultdict(list)
    for row in timeline:
        a, b = row['window_s']
        if (a >= active['estimated_start_s'] and b <= end and
                abs(b-a-width) < 1e-6 and abs((end-b)/width-round((end-b)/width)) < 1e-6):
            by_time[a,b].append(row)
    before = first = latest = previous_end = None
    sequence = []
    for span, rows in sorted(by_time.items()):
        if previous_end != span[0]:
            before = first = latest = None
        previous_end = span[1]
        directions = {r['reference']['direction'] for r in rows}
        direction = directions.pop() if len(directions)==1 else 'conflict'
        ids = [r['id'] for r in rows]
        sequence.append(dict(window_s=list(span), direction=direction, ids=ids))
        anchor = dict(window_s=list(span), id=ids[0])
        if direction == 'support':
            before, first, latest = anchor, None, None
        elif direction == 'negative' and before:
            if first is None:
                first = anchor
            else:
                latest = anchor
        else:
            first = latest = None
            if direction=='conflict':
                before = None
    eligible = previous_end==end and latest is not None
    return dict(sequence=sequence, can_confirm=eligible,
                before=before, first_against=first, later_against=latest if eligible else None)


def advance_end_memory(previous, window, current):
    """Keep the last supported side and the current consecutive contrary run.

    Each observation is a full, arrived window. Missing or mixed observations
    interrupt the run but do not erase the event's last measured support.
    """
    a, b = window
    memory = {key: previous.get(key) for key in ('support', 'first_against', 'later_against')}
    if previous.get('through_s') != a:
        memory.update(first_against=None, later_against=None)
    directions = {row['direction'] for row in current}
    if directions == {'support'}:
        memory.update(support=current[0], first_against=None, later_against=None)
    elif directions == {'negative'} and memory['support']:
        if memory['first_against'] is None:
            memory['first_against'] = current[0]
        else:
            memory['later_against'] = current[0]
    else:
        memory.update(first_against=None, later_against=None)
    return dict(memory, through_s=b)
