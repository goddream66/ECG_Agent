"""Small task-specific value heads and learned incremental measurement benefit."""

from dataclasses import asdict, dataclass, replace
from collections import Counter
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cardioagent.legacy.multimodal.policy.features import ablate, numeric
from cardioagent.legacy.multimodal.policy.objectives import COST, DELTA, ERROR, FAILURE, RETURN, Objectives
from cardioagent.legacy.multimodal.policy.state import TASKS
from cardioagent.legacy.multimodal.policy.targets import counterfactual
from cardioagent.legacy.multimodal.policy.value import OUTPUT_SLOTS


@dataclass(frozen=True)
class Variant:
    task: bool = True
    context: bool = True
    qc: str = "selective"
    cost: bool = True
    advantage: bool = False


VARIANTS = {
    "single_task_voi": Variant(context=False), "multitask_voi": Variant(context=False),
    "conditional_voi": Variant(), "no_task": Variant(task=False), "no_qc": Variant(qc="none"),
    "no_cost": Variant(cost=False), "full_qc": Variant(qc="full"),
}
TASK_INDEX = {task: i for i, task in enumerate(TASKS)}


class ConditionalNetwork(nn.Module):
    def __init__(self, width, variant):
        super().__init__()
        self.width = width
        self.variant = variant
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, (len(TASKS) if variant.task else 1) * len(OUTPUT_SLOTS) * 5)

    def forward(self, features, tasks):
        result = self.head(self.encoder(features)).reshape(len(features), -1, len(OUTPUT_SLOTS), 5)
        return result[torch.arange(len(features)), tasks] if self.variant.task else result[:, 0]

    @torch.inference_mode()
    def predict(self, features, task):
        features = ablate(np.asarray(features), task=self.variant.task, context=self.variant.context)
        raw = self(torch.from_numpy(features), torch.full((len(features),), TASK_INDEX[task]))
        raw[..., (FAILURE, RETURN)] = raw[..., (FAILURE, RETURN)].sigmoid()
        raw[..., (ERROR, COST)] = raw[..., (ERROR, COST)].clamp_min(0.)
        raw[..., ERROR] = raw[..., ERROR].clamp_max(1.)
        return raw.numpy()


def decision_estimate(profile, budget):
    state = profile.state({}, {}, budget)
    x = numeric(state, state.summary())
    net = ConditionalNetwork(len(x), Variant()).eval()
    net.predict([x], profile.task)
    timings = []
    for _ in range(24):
        began = perf_counter()
        net.predict([numeric(state, state.summary())], profile.task)
        timings.append((perf_counter() - began) * 1000)
    return float(np.median(timings))


def examples(cases, budget, decision_ms, preferences, *, advantage=False):
    if not cases or any(row["split"] != "train" for row, _ in cases):
        raise ValueError("V2.4 value targets require training cases only")
    subject_counts = Counter((p.task, row["subject_id"]) for row, p in cases)
    task_subjects = Counter(task for task, _ in subject_counts)
    columns = [[] for _ in range(6)]
    for row, profile in cases:
        _, generate = counterfactual(profile, budget, row["reference_bpm"], preferences, decision_ms, advantage=advantage)
        states = list(generate())
        weight = 1. / (len(states) * subject_counts[profile.task, row["subject_id"]] * task_subjects[profile.task])
        for x, y, mask, no_qc in states:
            for column, value in zip(columns, (x, y, mask, TASK_INDEX[profile.task], weight, no_qc)):
                column.append(value)
    return tuple(np.asarray(column, dtype=np.float32 if i == 4 else None) for i, column in enumerate(columns))


def train(data, variant, epochs, *, only_task=None):
    x, y, mask, tasks, weight, no_qc = data
    keep = np.ones(len(x), dtype=bool)
    if only_task:
        keep &= tasks == TASK_INDEX[only_task]
    if variant.qc == "none":
        keep &= no_qc
    x, y, mask, tasks, weight = [a[keep] for a in (x, y, mask, tasks, weight)]
    if not len(x):
        raise ValueError(f"No training examples for {only_task or variant}")
    x = ablate(x, task=variant.task, context=variant.context)
    x, y, mask, tasks, weight = map(torch.from_numpy, (x, y, mask, tasks, weight))
    weight /= weight.mean()
    torch.manual_seed(20261124)
    net = ConditionalNetwork(x.shape[1], variant)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
    scales = torch.tensor([6., 1., 1., .20, .6])
    history = []
    for _ in range(epochs):
        total = 0.
        for ids in torch.randperm(len(x)).split(512):
            raw, target, available = net(x[ids], tasks[ids]), y[ids], mask[ids]
            terms = (raw - target).square()
            terms[..., (FAILURE, RETURN)] = F.binary_cross_entropy_with_logits(
                raw[..., (FAILURE, RETURN)], target[..., (FAILURE, RETURN)], reduction="none")
            per_state = (terms * available * scales).sum(dim=(1, 2)) / available.sum(dim=(1, 2))
            loss = (per_state * weight[ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        history.append(total / len(x))
    return net.eval(), {"states": len(x), "epochs": epochs, "weighted_loss": history, "variant": asdict(variant)}


def preferences(variant):
    default = Objectives()
    return replace(default, cost=default.cost if variant.cost else 0.,
                   complementarity=default.complementarity if variant.context and not variant.advantage else 0.)
