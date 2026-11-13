"""Counterfactual measurement continuations and one-step incremental benefit."""

from functools import lru_cache
from itertools import combinations

import numpy as np

from .features import numeric
from .objectives import COST, COST_UNIT_MS, terminal
from .value import ABSTAIN, OUTPUT_SLOTS, STOP, output_index


def subsets(items):
    for count in range(len(items) + 1):
        yield from combinations(items, count)


def counterfactual(profile, budget, reference, preferences, decision_ms, *, advantage=False):
    initial = (profile.initial,) if profile.initial else ()
    order = {action: i for i, action in enumerate(profile.options)}

    def key(actions):
        return tuple(sorted(actions, key=order.get))

    @lru_cache(maxsize=None)
    def observation(actions):
        state = profile.state({a: profile.measurements[a] for a in actions}, {}, budget - len(actions))
        return state, state.summary()

    @lru_cache(maxsize=None)
    def local(actions):
        _, summary = observation(actions)
        rates = {ABSTAIN: terminal(None, reference)}
        if summary["rate_bpm"] is not None:
            rates[STOP] = terminal(summary["rate_bpm"], reference)
        return rates

    @lru_cache(maxsize=None)
    def values(actions):
        state, _ = observation(actions)
        results = {a: outcome.copy() for a, outcome in local(actions).items()}
        for action in state.options:
            if not state.remaining or action in actions:
                continue
            future = values(key((*actions, action)))
            chosen = min(future.values(), key=preferences.loss).copy()
            chosen[COST] += profile.measurements[action]["duration_ms"] / COST_UNIT_MS
            results[action] = chosen
        for outcome in results.values():
            outcome[COST] += decision_ms / COST_UNIT_MS
        return results

    def oracle(actions, legal):
        return min(legal, key=lambda action: preferences.loss(values(key(actions))[action]))

    def examples():
        for acquired in subsets(tuple(action for action in profile.options if action not in initial)):
            if len(acquired) + len(initial) > budget:
                continue
            actions = key((*initial, *acquired))
            state, summary = observation(actions)
            target = np.zeros((len(OUTPUT_SLOTS), 5), dtype=np.float32)
            mask = np.zeros_like(target)
            outcomes = values(actions)
            before = min(map(preferences.loss, (outcomes if advantage else local(actions)).values()))
            for action, outcome in outcomes.items():
                index = output_index(action, state.options)
                target[index, :4] = outcome
                mask[index, :4] = (outcome[2], 1., 1., 1.)
                if advantage or action in state.options:
                    after = (preferences.loss(outcome) if advantage else
                             min(map(preferences.loss, local(key((*actions, action))).values())))
                    target[index, 4] = before - after
                    mask[index, 4] = 1.
            remaining = tuple(action for action in state.options if action not in actions) if state.remaining else ()
            for seen in subsets(remaining):
                state.previews = {action: profile.previews[action] for action in seen}
                yield numeric(state, summary), target, mask, not seen

    return oracle, examples
