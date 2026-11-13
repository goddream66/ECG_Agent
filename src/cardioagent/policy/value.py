"""Channel-level baselines and shared execution for measured or replayed evidence."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import xlogy

from ..costs import Costs
from .state import EvidenceState, SLOTS, actions_for, preview_channel


STOP = "STOP"
ABSTAIN = "ABSTAIN"
ABSTAIN_LOSS = 0.40
ACTION_PENALTY = 0.025
ERROR_LIMIT = 2.0
BASELINES = ("ecg_only", "ppg_fallback", "alternate_lead_first", "channel_quality",
             "fixed_gaussian_eig", "channel_eig", "cost_aware_channel_eig")
SIGMA = {"ecg": 1.5, "ppg": 4., "pcg": 8.}
OUTPUT_SLOTS = (*SLOTS, STOP, ABSTAIN)


def output_index(action: str, options: dict) -> int:
    return OUTPUT_SLOTS.index(options[action][2] if action in options else action)


def terminal_loss(rate: float | None, reference: float) -> float:
    return ABSTAIN_LOSS if rate is None else float(abs(rate - reference) > ERROR_LIMIT)


@dataclass
class Profile:
    task: str
    duration: float
    options: dict
    initial: str | None
    measurements: dict
    previews: dict

    def state(self, evidence: dict, previews: dict, remaining: int) -> EvidenceState:
        return EvidenceState(self.task, self.duration, self.options, evidence, previews, remaining)


def channel_observers(record):
    from ..rate_policy import observe_rate

    def measurement(modality, channel):
        began = perf_counter()
        item = observe_rate(record, modality, channel)
        return item | {"duration_ms": 1000 * (perf_counter() - began)}

    def quality(modality, channel):
        wave = record.waves[modality]
        return preview_channel(wave.window(channel, 0, record.shared_duration), wave.fs, modality)

    return measurement, quality


def capture_profile(record, task: str = "heart_rate", *, original=None, original_profile=None) -> Profile:
    options, initial = actions_for(record, task)
    measurement, quality = channel_observers(record)
    measurements, previews = {}, {}
    for action, (modality, channel, _) in options.items():
        unchanged = (original is not None and action in original_profile.options
                     and original_profile.options[action][:2] == (modality, channel)
                     and record.shared_duration == original.shared_duration
                     and (record.waves[modality] is original.waves[modality] or np.array_equal(
                         record.waves[modality].values[:, record.waves[modality].channels.index(channel)],
                         original.waves[modality].values[:, original.waves[modality].channels.index(channel)])))
        if unchanged:
            measurements[action], previews[action] = original_profile.measurements[action], original_profile.previews[action]
        else:
            measurements[action], previews[action] = measurement(modality, channel), quality(modality, channel)
    return Profile(task, record.shared_duration, options, initial, measurements, previews)


def estimate_costs(profiles: list[Profile]) -> dict:
    by_slot = {slot: [] for slot in SLOTS}
    for profile in profiles:
        for action, (_, _, slot) in profile.options.items():
            by_slot[slot].append(profile.measurements[action]["duration_ms"])
    available = [t for group in by_slot.values() for t in group]
    default = float(np.median(available)) if available else 1.
    return {slot: max(0.1, float(np.median(group)) if group else default) for slot, group in by_slot.items()}


def action_penalty(action: str, options: dict, costs: dict) -> float:
    return ACTION_PENALTY * costs[options[action][2]] / float(np.median(list(costs.values())))


def eig_scores(state: EvidenceState, candidates: list[str], costs: dict, *, channel_qc: bool, cost_aware: bool) -> dict:
    grid = np.arange(30., 221., 5.)
    prior = np.ones(len(grid))
    for row in state.evidence.values():
        if row["rate_bpm"] is not None:
            width = SIGMA[row["modality"]] * (1 if row["quality"] == "usable" else 8)
            prior *= np.exp(-0.5 * ((grid - row["rate_bpm"]) / width) ** 2) + 1e-12
    prior /= prior.sum()
    entropy = -np.sum(xlogy(prior, prior))
    scores = {}
    for action in candidates:
        modality, _, slot = state.options[action]
        width = SIGMA[modality]
        if channel_qc:
            q = state.previews[action]
            if not q["valid"]:
                scores[action] = -1.
                continue
            width *= 1 + 12 * (1 - q["score"]) ** 3
        elif modality == "ecg" and any(row["modality"] == "ecg" and row["quality"] != "usable"
                                     for row in state.evidence.values()):
            # Historical modality-level Gaussian baseline.
            width *= 4
        likelihood = np.exp(-0.5 * ((grid[:, None] - grid) / width) ** 2)
        likelihood /= likelihood.sum(axis=1, keepdims=True)
        joint = prior[:, None] * likelihood
        posterior_entropy = -np.sum(xlogy(joint, joint / joint.sum(axis=0)))
        scores[action] = float((entropy - posterior_entropy) / (costs[slot] if cost_aware else 1.))
    return scores


def rule_action(method: str, state: EvidenceState, legal: list[str], costs: dict) -> str:
    if STOP in legal:
        return STOP
    available = [a for a in legal if a in state.options]
    if method == "ecg_only" or not available:
        return ABSTAIN
    if method in ("ppg_fallback", "alternate_lead_first"):
        order = [a for a in available if method != "ppg_fallback" or not a.startswith("VERIFY_ECG:")]
    elif method == "channel_quality":
        order = sorted((a for a in available if state.previews[a]["valid"]),
                       key=lambda a: state.previews[a]["score"], reverse=True)
    else:
        scores = eig_scores(state, available, costs, channel_qc=method != "fixed_gaussian_eig",
                            cost_aware=method == "cost_aware_channel_eig")
        order = sorted((a for a in available if scores[a] >= 0), key=scores.get, reverse=True)
    return order[0] if order else ABSTAIN


def run_policy(profile: Profile, budget: int, method: str, costs: dict, *, model=None, calibrator=None,
               max_risk: float | None = None, oracle=None, live=None) -> dict:
    if budget < 1:
        raise ValueError("Budget must be positive")
    began = perf_counter()
    observed, previews, path, stop_trials = {}, {}, [], []
    ledger = Costs()
    waveform_ms = qc_ms = 0.
    live_measure, live_quality = live or (None, None)

    def acquire(action):
        nonlocal waveform_ms
        modality, channel, _ = profile.options[action]
        row = live_measure(modality, channel) if live else profile.measurements[action]
        observed[action] = row
        ledger.waveform(row, cached=live is None)
        waveform_ms += row["duration_ms"]
        path.append({"action": action, "channel": channel, "quality": row["quality"],
                     "remaining_budget": budget - len(observed), "duration_ms": row["duration_ms"]})

    def preview():
        nonlocal qc_ms
        for action, (modality, channel, _) in profile.options.items():
            if action not in previews:
                row = live_quality(modality, channel) if live else profile.previews[action]
                previews[action] = row
                qc_ms += row["duration_ms"]
                if live:
                    ledger.channel_qc_calls += 1
                    ledger.channel_qc_ms += row["duration_ms"]

    if profile.initial:
        acquire(profile.initial)
    while True:
        state = profile.state(observed, previews, budget - len(observed))
        summary = state.summary()
        legal = state.legal()
        if legal == [ABSTAIN]:
            action = ABSTAIN
        elif method == "oracle":
            action = oracle(tuple(observed), legal)
        elif method.startswith("learned_voi"):
            preview()
            scores = model(state.numeric(summary))
            if STOP in legal:
                raw_risk = float(scores[output_index(STOP, state.options)])
                risk = calibrator.predict(raw_risk) if calibrator else raw_risk
                stop_trials.append({"raw_risk": raw_risk, "risk": risk, "rate_bpm": summary["rate_bpm"]})
                if max_risk is not None and risk > max_risk:
                    legal.remove(STOP)
            def loss(a):
                if a == ABSTAIN:
                    return ABSTAIN_LOSS
                if a == STOP:
                    return raw_risk
                return float(scores[output_index(a, state.options)]) + action_penalty(a, state.options, costs)
            action = min(legal, key=loss)
        else:
            if method in ("channel_quality", "channel_eig", "cost_aware_channel_eig") and STOP not in legal:
                preview()
            action = rule_action(method, state, legal, costs)
        if action in (STOP, ABSTAIN):
            path.append({"action": action, "remaining_budget": budget - len(observed)})
            break
        acquire(action)
    return {
        "task": profile.task, "method": method, "rate_bpm": summary["rate_bpm"] if action == STOP else None,
        "decision": action, "candidate_rate_bpm": summary["rate_bpm"], "fusion_status": summary["status"],
        "quality_rank": state.quality_rank(summary), "stop_trials": stop_trials,
        "evidence": list(observed.values()), "candidate_quality": previews, "actions": path,
        "cost": ledger.report(perf_counter() - began) | {
            "qc_requests": len(previews), "qc_executions": len(previews) if live else 0,
            "qc_ms": qc_ms, "waveform_ms": waveform_ms, "analysis_ms": qc_ms + waveform_ms,
        },
    }
