"""Common active evidence state, action mask, stopping, and rule baselines."""

from collections.abc import Callable
from time import perf_counter

import numpy as np
from scipy.special import xlogy

from cardioagent.legacy.multimodal.costs import Costs
from cardioagent.common.recording import Recording
from cardioagent.legacy.multimodal.workflow import preferred_channel, rate_summary


STRATEGIES = ("ecg_only", "ppg_fallback", "alternate_lead_first", "quality_aware", "cost_aware_eig")
LEGACY_NAMES = {"ppg_first": "ppg_fallback", "lead_first": "alternate_lead_first"}
MEASUREMENT_SIGMA = {"ecg": 1.5, "ppg": 4.0, "pcg": 8.0}


class Acquisition:
    def __init__(self, record: Recording, budget: int, observe: Callable[[str, str], dict]):
        if budget < 1:
            raise ValueError("Action budget must be positive")
        self.record = record
        self.budget = budget
        self.observe = observe
        self.costs = Costs()
        self.evidence = []
        self.actions = []
        self.measured = set()
        self.channel_previews = {}
        self.options = {}
        if "ecg" in record.waves:
            channels = record.waves["ecg"].channels
            primary = preferred_channel("ecg", channels)
            self.options["ACQUIRE_ECG"] = ("ecg", primary)
            alternatives = [ch for ch in dict.fromkeys(("II", "V3", *channels)) if ch in channels and ch != primary]
            for ch in alternatives[:2]:
                self.options[f"VERIFY_ECG:{ch}"] = ("ecg", ch)
        for m in ("ppg", "pcg"):
            if m in record.waves:
                self.options[f"ACQUIRE_{m.upper()}"] = (m, preferred_channel(m, record.waves[m].channels))

    @property
    def remaining(self) -> int:
        return self.budget - self.costs.action_budget_units

    def legal(self) -> list[str]:
        available = [action for action in self.options if action != "ACQUIRE_ECG" and action not in self.measured]
        result = available if self.remaining else []
        if self.summary()["rate_bpm"] is not None:
            result.append("STOP")
        return [*result, "ABSTAIN"]

    def acquire(self, action: str) -> None:
        modality, channel = self.options[action]
        try:
            raw = self.observe(modality, channel)
        except (OSError, ValueError, RuntimeError) as exc:
            raw = {"status": "failed", "rate_bpm": None, "quality": "review",
                   "reason": f"{type(exc).__name__}: {exc}"}
        item = {"modality": modality, "channel": channel, "window_s": [0, self.record.shared_duration],
                "evidence_id": f"{modality}/{channel}", **raw}
        self.evidence.append(item)
        self.measured.add(action)
        self.costs.waveform(item, cached=item.get("cache_hit", False))
        self.actions.append({"action": action, "modality": modality, "channel": channel,
                             "evidence_id": item["evidence_id"], "result": item["status"],
                             "cache_hit": item.get("cache_hit", False), "remaining_budget": self.remaining})

    def summary(self) -> dict:
        return rate_summary(self.evidence, 0, self.record.shared_duration)

    def state(self, legal: list[str], *, structured: bool = False) -> dict:
        items = [
            {key: item[key] for key in ("modality", "channel", "status", "rate_bpm", "quality", "quality_reasons")
             if key in item}
            for item in self.evidence
        ]
        summary = self.summary()
        result = {"observations": items, "summary_status": summary["status"], "current_rate_bpm": summary["rate_bpm"],
                  "remaining_budget": self.remaining, "legal_actions": legal}
        if structured:
            from cardioagent.legacy.multimodal.policy.state import disagreement, preview_channel

            for action in legal:
                if action not in self.options or action in self.channel_previews:
                    continue
                modality, channel = self.options[action]
                wave = self.record.waves[modality]
                quality = preview_channel(wave.window(channel, 0, self.record.shared_duration), wave.fs, modality)
                self.channel_previews[action] = {"modality": modality, "channel": channel, **quality}
                self.costs.channel_qc_calls += 1
                self.costs.channel_qc_ms += quality["duration_ms"]
            for item, raw in zip(items, self.evidence):
                item.update({key: raw[key] for key in ("signal_quality", "event_quality", "event_score_median",
                                                       "pcg_candidate_periods_s") if key in raw})
            result.update(task="heart_rate", available_modalities=list(self.record.waves),
                          observed_rate_spread_bpm=disagreement(self.evidence),
                          candidate_quality={a: {k: v for k, v in self.channel_previews[a].items() if k != "duration_ms"}
                                             for a in legal if a in self.channel_previews})
        return result


def eig_order(state: Acquisition, available: list[str]) -> list[str]:
    """Rank actions by expected entropy reduction under a fixed Gaussian measurement model."""
    grid = np.arange(30., 221., 5.)
    posterior = np.ones(len(grid))
    for observation in state.evidence:
        if observation["rate_bpm"] is not None:
            sigma = MEASUREMENT_SIGMA[observation["modality"]] * (1 if observation["quality"] == "usable" else 8)
            posterior *= np.exp(-0.5 * ((grid - observation["rate_bpm"]) / sigma) ** 2) + 1e-12
    posterior /= posterior.sum()
    prior_entropy = -np.sum(xlogy(posterior, posterior))
    scores = {}
    for action in available:
        modality, _ = state.options[action]
        sigma = MEASUREMENT_SIGMA[modality]
        primary = next((row for row in state.evidence if row["modality"] == "ecg"), None)
        # A poor observed ECG lead weakens the prior for additional ECG measurements.
        if modality == "ecg" and primary and primary["quality"] != "usable":
            sigma *= 4
        likelihood = np.exp(-0.5 * ((grid[:, None] - grid) / sigma) ** 2)
        likelihood /= likelihood.sum(axis=1, keepdims=True)
        joint = posterior[:, None] * likelihood
        predictive = joint.sum(axis=0)
        posterior_entropy = -np.sum(xlogy(joint, joint / predictive))
        scores[action] = (prior_entropy - posterior_entropy) / 1  # All acquisitions cost one unit.
    return sorted(available, key=scores.get, reverse=True)


def choose_rule(strategy: str, state: Acquisition, legal: list[str]) -> str:
    if "STOP" in legal:
        return "STOP"
    if strategy == "ecg_only":
        return "ABSTAIN"
    alternatives = [action for action in legal if action.startswith("VERIFY_ECG:")]
    auxiliary = [action for action in ("ACQUIRE_PPG", "ACQUIRE_PCG") if action in legal]
    if strategy == "ppg_fallback":
        ordered = auxiliary
    elif strategy == "alternate_lead_first":
        ordered = alternatives + auxiliary
    elif strategy == "quality_aware":
        primary = state.evidence[0] if state.evidence else {}
        reasons = primary.get("quality_reasons", [])
        local_artifact = any(r in reasons for r in ("sustained flat samples", "baseline drift dominates", "high-frequency power"))
        ordered = alternatives + auxiliary if local_artifact else auxiliary + alternatives
    else:
        ordered = eig_order(state, alternatives + auxiliary)
    return ordered[0] if ordered else "ABSTAIN"


def run_acquisition(record: Recording, budget: int, observe, strategy: str, decide=None) -> dict:
    began = perf_counter()
    strategy = LEGACY_NAMES.get(strategy, strategy)
    if decide is None and strategy not in STRATEGIES:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    state = Acquisition(record, budget, observe)
    if "ACQUIRE_ECG" in state.options:
        state.acquire("ACQUIRE_ECG")
    while True:
        legal = state.legal()
        if legal == ["ABSTAIN"]:
            decision = "ABSTAIN"
        else:
            decision = decide(state, legal) if decide else choose_rule(strategy, state, legal)
        if decision not in legal:
            state.actions.append({"action": decision, "result": "invalid_action", "remaining_budget": state.remaining})
            decision = "ABSTAIN"
        if decision in ("STOP", "ABSTAIN"):
            break
        state.acquire(decision)
    state.actions.append({"action": decision, "result": "terminal", "remaining_budget": state.remaining})
    summary = state.summary()
    if decision == "ABSTAIN":
        summary = summary | {"status": "abstained", "rate_bpm": None, "evidence_ids": []}
    return summary | {
        "strategy": strategy, "decision": decision, "evidence": state.evidence, "actions": state.actions,
        "tool_calls": state.costs.action_budget_units, "cost": state.costs.report(perf_counter() - began),
    }
