"""Counterfactual supervision using actual offline tool outputs."""

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .network import ValueNetwork
from .value import ABSTAIN, ABSTAIN_LOSS, OUTPUT_SLOTS, STOP, action_penalty, output_index, terminal_loss


def counterfactual(profile, budget: int, reference: float, costs: dict):
    order = {action: i for i, action in enumerate(profile.options)}

    def key(actions):
        return tuple(sorted(actions, key=order.get))

    @lru_cache(maxsize=None)
    def state_at(actions):
        return profile.state({a: profile.measurements[a] for a in actions}, profile.previews, budget - len(actions))

    @lru_cache(maxsize=None)
    def action_values(actions):
        state = state_at(actions)
        summary = state.summary()
        result = {ABSTAIN: ABSTAIN_LOSS}
        if summary["rate_bpm"] is not None:
            result[STOP] = terminal_loss(summary["rate_bpm"], reference)
        if state.remaining:
            for a in state.options:
                if a not in actions:
                    future = action_values(key((*actions, a)))
                    result[a] = min(future.values()) + action_penalty(a, state.options, costs)
        return result

    initial = (profile.initial,) if profile.initial else ()
    action_values(initial)

    def pick(actions, legal):
        values = action_values(key(actions))
        return min(legal, key=values.get)

    def examples():
        stack = [initial]
        visited = set()
        while stack:
            actions = stack.pop()
            if actions in visited:
                continue
            visited.add(actions)
            state = state_at(actions)
            target = np.zeros(len(OUTPUT_SLOTS), dtype=np.float32)
            mask = np.zeros_like(target)
            for action, loss in action_values(actions).items():
                index = output_index(action, state.options)
                if action in state.options:
                    loss -= action_penalty(action, state.options, costs)
                    stack.append(key((*actions, action)))
                target[index] = np.clip(loss, 0, 1)
                mask[index] = 1
            yield state.numeric(state.summary()), target, mask

    return pick, examples


def fit_value_network(cases: list[tuple[dict, object]], budget: int, costs: dict,
                      epochs: int = 16, seed: int = 20261112) -> tuple[ValueNetwork, dict]:
    if any(row["split"] != "train" for row, _ in cases):
        raise ValueError("Only training-subject cases may fit the value network")
    tensors = []
    for row, profile in cases:
        _, examples = counterfactual(profile, budget, row["reference_bpm"], costs)
        tensors.extend(examples())
    if not tensors:
        raise ValueError("No training states")
    x, y, mask = (torch.from_numpy(np.stack(data)) for data in zip(*tensors))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    net = ValueNetwork(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(x, y, mask), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    losses = []
    for _ in range(epochs):
        error_sum = count = 0.
        for features, target, available in loader:
            prediction = net(features)
            error = ((prediction - target).square() * available).sum()
            n = available.sum()
            opt.zero_grad()
            (error / n).backward()
            opt.step()
            error_sum += error.item()
            count += n.item()
        losses.append(round(error_sum / count, 7))
    return net.eval(), {"states": len(tensors), "scenarios_times_records": len(cases),
                        "subjects": len({row["subject_id"] for row, _ in cases}), "epochs": epochs,
                        "seed": seed, "masked_mse": losses}
