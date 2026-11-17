"""Frozen lifecycle rules, independent of event storage and signal access."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class EventConfig:
    threshold: float = .5
    enter_windows: int = 2
    exit_windows: int = 2
    unknown_grace_s: float = 10.
    borderline_margin: float = .1
    recheck_windows: int = 3
    review_budget: int = 4

    def __post_init__(self):
        if not 0 < self.threshold < 1 or not math.isfinite(self.threshold):
            raise ValueError('Invalid threshold')
        if any(type(x) is not int or x < 1 for x in (self.enter_windows, self.exit_windows, self.recheck_windows)):
            raise ValueError('Window counts must be positive integers')
        if not math.isfinite(self.unknown_grace_s) or self.unknown_grace_s < 0:
            raise ValueError('Invalid unknown grace')
        if not 0 <= self.borderline_margin < 1 or not 1 <= self.review_budget <= 4:
            raise ValueError('Invalid review configuration')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EventRules:
    config: EventConfig
    def confirms(self, count): return count >= self.config.enter_windows
    def ends(self, count): return count >= self.config.exit_windows
    def unknown_expired(self, start, end): return end-start > self.config.unknown_grace_s
