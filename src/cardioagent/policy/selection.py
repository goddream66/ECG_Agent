"""Execute selective waveform/QC decisions online or against offline measurements."""

from collections import defaultdict
from time import perf_counter

import numpy as np

from .conditional_value import preferences
from .features import numeric
from .objectives import COST, COST_UNIT_MS, DELTA, FAILURE, RETURN, terminal
from .qc_prior import distribution
from .value import ABSTAIN, STOP, output_index


def fit_qc_priors(cases):
    groups = defaultdict(list)
    for row, profile in cases:
        if row["split"] != "train":
            raise ValueError("QC priors use training subjects only")
        for action, (_, _, slot) in profile.options.items():
            groups[f"{profile.task}/{slot}"].append({k: v for k, v in profile.previews[action].items() if k != "duration_ms"})
    return {key: [rows[round(q * (len(rows) - 1))] for q in (.12, .37, .63, .88)]
            for key, group in groups.items() for rows in [sorted(group, key=lambda item: item["score"])]}


def _scores(state, legal, predictions, objective, costs, calibrator, limit):
    result = {}
    for action in legal:
        vector = predictions[output_index(action, state.options)].copy()
        if action == ABSTAIN:
            vector[:4] = terminal(None, 0.)
            vector[COST] = costs["decision"] / COST_UNIT_MS
        elif action == STOP:
            if calibrator and calibrator.predict(float(vector[FAILURE])) > limit:
                continue
            vector[RETURN] = 1.
            vector[COST] = costs["decision"] / COST_UNIT_MS
        bonus = (objective.complementarity * float(np.clip(vector[DELTA], -1.25, 1.25))
                 if objective.complementarity and action in state.options else 0.)
        result[action] = objective.loss(vector) - bonus
    return result


def run(profile, budget, costs, priors, *, model=None, oracle=None, calibrator=None, limit=.2, live=None, qc_mode=None):
    if budget < 1 or (model is None) == (oracle is None):
        raise ValueError("Supply a positive budget and exactly one learned or oracle policy")
    observed, previews, qc_history, trace, trials = {}, {}, {}, [], []
    qc_levels = defaultdict(int)
    waveform_ms = qc_ms = decision_ms = 0.
    mode = qc_mode or (model.variant.qc if model else "none")
    objective = preferences(model.variant) if model else None

    def acquire(action):
        nonlocal waveform_ms
        modality, channel, _ = profile.options[action]
        row = live[0](modality, channel) if live else profile.measurements[action]
        observed[action] = row
        previews.pop(action, None)
        waveform_ms += row["duration_ms"]
        trace.append({"action": action, "channel": channel, "quality": row["quality"], "duration_ms": row["duration_ms"]})

    def preview(action):
        nonlocal qc_ms
        modality, channel, _ = profile.options[action]
        row = live[1](modality, channel) if live else profile.previews[action]
        previews[action] = qc_history[action] = row
        qc_ms += row["duration_ms"]
        trace.append({"action": f"QC:{action}", "channel": channel, "duration_ms": row["duration_ms"]})

    if profile.initial:
        acquire(profile.initial)
    if mode == "full" and len(observed) < budget:
        for action in profile.options:
            if action not in observed:
                preview(action)
    while True:
        started = perf_counter()
        state = profile.state(observed, previews, budget - len(observed))
        summary = state.summary()
        legal = [a for a in state.options if a not in observed] if state.remaining else []
        if summary["rate_bpm"] is not None:
            legal.append(STOP)
        legal.append(ABSTAIN)
        if oracle:
            chosen = oracle(tuple(observed), legal)
        elif legal == [ABSTAIN]:
            chosen = ABSTAIN
        else:
            qc_choices, batch = {}, [numeric(state, summary)]
            if mode == "selective":
                for action in legal:
                    if action not in state.options or action in previews:
                        continue
                    slot = profile.options[action][2]
                    samples, level = distribution(priors, profile, observed, slot)
                    if not samples:
                        continue
                    qc_levels[level] += 1
                    begin = len(batch)
                    for sample in samples:
                        hypothetical = profile.state(observed, previews | {action: sample}, state.remaining)
                        batch.append(numeric(hypothetical, summary))
                    qc_choices[action] = (slot, slice(begin, len(batch)))
            predictions = model.predict(batch, profile.task)
            if STOP in legal:
                raw_risk = float(predictions[0, output_index(STOP, state.options), FAILURE])
                risk = calibrator.predict(raw_risk) if calibrator else raw_risk
                trials.append({"rate_bpm": summary["rate_bpm"], "raw_risk": raw_risk, "risk": risk})
            scores = _scores(state, legal, predictions[0], objective, costs, calibrator, limit)
            chosen = min(scores, key=scores.get)
            best = scores[chosen]
            for action, (slot, indices) in qc_choices.items():
                expected = np.mean([min(_scores(state, legal, vector, objective, costs, calibrator, limit).values())
                                    for vector in predictions[indices]])
                qc_penalty = objective.cost * (costs["qc"][slot] + costs["decision"]) / COST_UNIT_MS
                if expected + qc_penalty < best - .002:
                    chosen, best = f"QC:{action}", expected + qc_penalty
        decision_ms += (perf_counter() - started) * 1000
        if chosen in (STOP, ABSTAIN):
            trace.append({"action": chosen})
            break
        if chosen.startswith("QC:"):
            preview(chosen.removeprefix("QC:"))
        else:
            acquire(chosen)
    return {"rate_bpm": summary["rate_bpm"] if chosen == STOP else None, "decision": chosen,
            "candidate_rate_bpm": summary["rate_bpm"], "quality_rank": state.quality_rank(summary),
            "stop_trials": trials, "actions": trace, "evidence": list(observed.values()), "previews": qc_history,
            "qc_distribution_queries": dict(qc_levels),
            "cost": {"waveform_requests": len(observed), "qc_requests": len(qc_history), "waveform_ms": waveform_ms,
                     "qc_ms": qc_ms, "decision_ms": decision_ms, "analysis_ms": waveform_ms + qc_ms + decision_ms}}
