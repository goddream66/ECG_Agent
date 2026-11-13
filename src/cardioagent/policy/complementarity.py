"""Observable relationships between candidates and the evidence acquired so far."""

from .state import SLOTS
from ..workflow import rates_agree


FEATURES_PER_SLOT = 7
WIDTH = len(SLOTS) * FEATURES_PER_SLOT


def context_features(state):
    options = {slot: (action, modality) for action, (modality, _, slot) in state.options.items()}
    observed = [row for row in state.evidence.values() if row.get("rate_bpm") is not None]
    features = []
    for slot in SLOTS:
        if slot not in options:
            features.extend([0.] * FEATURES_PER_SLOT)
            continue
        _, modality = options[slot]
        same = [row for row in observed if row["modality"] == modality]
        other = [row for row in observed if row["modality"] != modality]
        disagreements = [abs(a["rate_bpm"] - b["rate_bpm"]) / 60 for a in same for b in other]
        agreement = any(rates_agree(a["rate_bpm"], b["rate_bpm"]) for a in same for b in other)
        features.extend([len(same) / 3, len(other) / 4, any(row["quality"] == "usable" for row in same),
                         any(row["quality"] == "usable" for row in other), bool(disagreements),
                         min(2., max(disagreements, default=0.)), agreement])
    return features
