"""Empirical conditional QC distributions from training observations only."""

from collections import defaultdict

from .targets import subsets


def context(profile, observed):
    anchor = observed.get(profile.initial)
    usable = [row for row in observed.values() if row.get("quality") == "usable" and row.get("rate_bpm") is not None]
    gaps = [abs(a["rate_bpm"] - b["rate_bpm"]) for a in usable for b in usable if a["modality"] != b["modality"]]
    primary = anchor["quality"] if anchor is not None else "no_anchor"
    cross = "none" if not gaps else ("conflict" if max(gaps) > 5 else "agreement")
    return f"{primary}/{cross}"


def quantile_samples(rows):
    ordered = sorted(rows, key=lambda item: item["score"])
    return [ordered[round(q * (len(ordered) - 1))] for q in (.125, .375, .625, .875)]


def fit_conditionals(cases, budget, min_subjects=5):
    buckets = defaultdict(list)
    support = defaultdict(set)
    for row, profile in cases:
        if row["split"] != "train":
            raise ValueError("QC distributions must be fitted on training subjects")
        initial = (profile.initial,) if profile.initial else ()
        contexts = set()
        for acquired in subsets(tuple(a for a in profile.options if a not in initial)):
            actions = initial + acquired
            if len(actions) >= budget:
                continue
            observed = {a: profile.measurements[a] for a in actions}
            signature = context(profile, observed)
            contexts.update((a, signature) for a in profile.options if a not in observed)
        contexts |= {(action, signature.split("/")[0]) for action, signature in contexts}
        for action, signature in sorted(contexts):
            slot = profile.options[action][2]
            sample = {k: v for k, v in profile.previews[action].items() if k != "duration_ms"}
            key = f"{profile.task}/{slot}/{signature}"
            buckets[key].append(sample)
            support[key].add(row["subject_id"])
    return {key: {"samples": quantile_samples(values), "records": len(values), "subjects": len(support[key])}
            for key, values in buckets.items() if len(support[key]) >= min_subjects}


def distribution(priors, profile, observed, slot):
    if "continuous" in priors:
        from .qc_continuous import distribution as continuous
        samples = continuous(priors["continuous"], profile, observed, slot)
        if samples is not None:
            return samples, "continuous"
    key = f"{profile.task}/{slot}"
    condition = context(profile, observed)
    for suffix, level in ((condition, "joint"), (condition.split("/")[0], "anchor")):
        found = priors.get("conditions", {}).get(f"{key}/{suffix}")
        if found is not None:
            return found["samples"], level
    return priors.get(key, ()), "marginal"


def validate_priors(cases, priors):
    """Empirical CRPS on held-out quality scores before the first extra measurement."""
    import numpy as np

    groups = defaultdict(list)
    for row, profile in cases:
        observed = {profile.initial: profile.measurements[profile.initial]} if profile.initial else {}
        for action, (_, _, slot) in profile.options.items():
            if action in observed:
                continue
            conditional, level = distribution(priors, profile, observed, slot)
            marginal = priors.get(f"{profile.task}/{slot}", ())
            if not conditional or not marginal:
                continue
            actual = profile.previews[action]["score"]
            def crps(samples):
                x = np.array([q["score"] for q in samples])
                return float(np.mean(abs(x - actual)) - .5 * np.mean(abs(x[:, None] - x)))
            groups[row["scenario"]].append((crps(conditional), crps(marginal), level != "marginal"))
    return {scenario: {"predictions": len(rows), "conditional_crps": float(np.mean([r[0] for r in rows])),
                       "marginal_crps": float(np.mean([r[1] for r in rows])),
                       "conditional_fraction": float(np.mean([r[2] for r in rows]))} for scenario, rows in groups.items()}
