"""Separate outcome prediction from the predeclared decision preferences."""

from dataclasses import dataclass
from statistics import median

import numpy as np


ERROR, FAILURE, RETURN, COST, DELTA = range(5)
ERROR_CAP = 12.
COST_UNIT_MS = 100.


@dataclass(frozen=True)
class Objectives:
    error: float = 6.
    failure: float = .75
    rejection: float = 1.25
    cost: float = .08
    complementarity: float = .18

    def loss(self, value):
        return (self.error * value[ERROR] * value[RETURN] + self.failure * value[FAILURE]
                + self.rejection * (1 - value[RETURN]) + self.cost * value[COST])


def terminal(rate, reference):
    if rate is None:
        return np.zeros(4, dtype=np.float32)
    error = abs(rate - reference)
    return np.array([min(error, ERROR_CAP) / ERROR_CAP, float(error > 2), 1., 0.], dtype=np.float32)


def cost_table(profiles, decision_ms):
    groups = {kind: {} for kind in ("waveform", "qc")}
    for profile in profiles:
        for action, (_, _, slot) in profile.options.items():
            for kind, data in (("waveform", profile.measurements), ("qc", profile.previews)):
                groups[kind].setdefault(slot, []).append(data[action]["duration_ms"])
    return {kind: {slot: max(.01, median(values)) for slot, values in slots.items()}
            for kind, slots in groups.items()} | {"decision": decision_ms}
