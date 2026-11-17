import json
from itertools import combinations
import math
import time
from pathlib import Path
from statistics import median
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware, wrap_model_call
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from cardioagent.legacy.multimodal.costs import Costs
from cardioagent.legacy.multimodal.model_adapters import PretrainedModels
from cardioagent.common.recording import MODALITIES, Recording, sample_span
from cardioagent.common.signal_tools import measure_rate


TOOL_BUDGET = 20
SPECIALIST_BUDGET = 4
REVIEW_BUDGET = 2


class WorkflowState(TypedDict, total=False):
    mode: str
    report: dict


def rates_agree(a: float, b: float) -> bool:
    return abs(a - b) <= max(5, 0.05 * min(a, b))


def agreeing_groups(items: list[dict]) -> list[tuple[dict, ...]]:
    for size in range(len(items), 0, -1):
        groups = [group for group in combinations(items, size)
                  if all(rates_agree(a["rate_bpm"], b["rate_bpm"]) for a, b in combinations(group, 2))]
        if groups:
            return groups
    return []


def rate_summary(evidence: list[dict], start: float, end: float) -> dict:
    """Summarize channels within each modality before counting modality votes."""
    channels = {}
    for item in evidence:
        if item["window_s"] != [start, end] or item.get("operation", "measure") != "measure":
            continue
        key = (item["modality"], item["channel"])
        channels.setdefault(key, []).append(item)

    observed = {}
    hypotheses = {}
    for modality in MODALITIES:
        history = {ch: items for (m, ch), items in sorted(channels.items()) if m == modality}
        if not history:
            continue
        usable = []
        rate_items = []
        channel_details = {}
        for ch, attempts in history.items():
            valid = [item for item in attempts if item.get("status") == "ok" and item.get("rate_bpm") is not None]
            accepted = [item for item in valid if item.get("quality") == "usable"]
            chosen = (accepted or valid or attempts)[-1]
            if valid:
                rate_items.append(chosen)
            if accepted:
                usable.append(chosen)
            channel_details[ch] = {
                "rate_bpm": chosen.get("rate_bpm"), "quality": chosen.get("quality", "review"),
                "status": chosen.get("status"), "evidence_id": chosen["evidence_id"],
                "attempt_ids": [item["evidence_id"] for item in attempts],
            }
        candidates = agreeing_groups(usable)
        hypotheses[modality] = [
            {"rate_bpm": median(item["rate_bpm"] for item in group),
             "evidence_ids": [item["evidence_id"] for item in group]}
            for group in candidates
        ]
        observed[modality] = {
            "rate_bpm": rate_items[0]["rate_bpm"] if len(rate_items) == 1 else None,
            "quality": "review", "evidence_id": rate_items[0]["evidence_id"] if len(rate_items) == 1 else None,
            "channels": channel_details, "hypotheses": hypotheses[modality],
            "has_channel_conflict": bool(candidates and len(candidates[0]) < len(usable)),
            "resolution": "channel_conflict" if len(candidates) > 1 else "insufficient",
        }

    def accept(modality: str, candidate: dict, resolution: str) -> None:
        observed[modality].update(candidate, quality="usable", evidence_id=candidate["evidence_ids"][0],
                                  resolution=resolution)

    for modality, options in hypotheses.items():
        if len(options) == 1:
            accept(modality, options[0], "channel_consensus")
    independent = {m: item["rate_bpm"] for m, item in observed.items() if item["quality"] == "usable"}
    for modality, options in hypotheses.items():
        if len(options) < 2:
            continue
        # Only already resolved modalities may corroborate an ambiguous channel group.
        scores = [sum(rates_agree(option["rate_bpm"], rate) for m, rate in independent.items() if m != modality)
                  for option in options]
        if max(scores) > 0 and scores.count(max(scores)) == 1:
            accept(modality, options[scores.index(max(scores))], "cross_modality_support")

    usable = [item for item in observed.values() if item["quality"] == "usable"]
    groups = agreeing_groups(usable)
    best = groups[0] if len(groups) == 1 else ()
    unresolved = any(item["resolution"] == "channel_conflict" for item in observed.values())
    selected = ()
    if len(best) >= 2 and len(best) > len(usable) / 2:
        selected = best
        status = "agreement" if len(best) == len(usable) and not unresolved else "partial_agreement"
    elif len(usable) == 1 and not unresolved:
        status, selected = "single_source", best
    else:
        status = "disagreement" if usable or unresolved else "insufficient"
    return {
        "status": status,
        "rate_bpm": round(median([item["rate_bpm"] for item in selected]), 2) if selected else None,
        "evidence_ids": [evidence_id for item in selected for evidence_id in item["evidence_ids"]],
        "observed": observed,
    }


class Analysis:
    def __init__(
        self, record: Recording, start: float, end: float, models: PretrainedModels,
        run_dir: Path, chat_model=None, include_models: bool = False,
        *, action_budget: int | None = None, measure=measure_rate,
    ):
        if not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= record.shared_duration + 1e-9:
            raise ValueError(f"Window must be within 0–{record.shared_duration:g} seconds")
        if end - start < 5:
            raise ValueError("Analysis window must span at least 5 seconds")
        self.record = record
        self.start = start
        self.end = end
        self.models = models
        self.run_dir = run_dir
        self.chat_model = chat_model
        self.include_models = include_models
        self.evidence = []
        self.trace = []
        self.failures = []
        if action_budget is not None and action_budget < 1:
            raise ValueError("Action budget must be positive")
        self.action_budget = action_budget
        self.remaining = TOOL_BUDGET if action_budget is None else action_budget
        self.measure = measure
        self.costs = Costs()
        self.actions = []

    def observe(self, modality: str, operation: str, channel: str) -> dict:
        if self.remaining <= 0:
            self.actions.append({"action": operation, "modality": modality, "channel": channel,
                                 "result": "budget_exhausted", "remaining_budget": self.remaining})
            return {"status": "budget_exhausted"}
        self.remaining -= 1
        began = time.perf_counter()
        evidence_id = f"E{len(self.evidence) + 1:03d}"
        sample_start = self.start
        try:
            wave = self.record.waves[modality]
            values = wave.window(channel, self.start, self.end)
            first, _ = sample_span(wave.fs, len(wave.values), 0, self.start, self.end)
            sample_start = first / wave.fs
            if operation == "measure":
                result = self.measure(modality, values, wave.fs, sample_start)
            else:
                result = self.models.predict(modality, channel, values, wave.fs, self.run_dir / f"{evidence_id}.npy")
        except Exception as exc:
            result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        item = {
            "evidence_id": evidence_id,
            "record_id": self.record.record_id,
            "modality": modality,
            "channel": channel,
            "window_s": [self.start, self.end],
            "time_basis": "WFDB recording start",
            "sample_rate_hz": wave.fs if modality in self.record.waves else None,
            "operation": operation,
            "duration_ms": round(1000 * (time.perf_counter() - began)),
            **result,
        }
        if "input_duration_s" in result:
            item["model_input_window_s"] = [sample_start, sample_start + result["input_duration_s"]]
        self.evidence.append(item)
        self.costs.waveform(item)
        self.actions.append({"action": operation, "modality": modality, "channel": channel,
                             "evidence_id": evidence_id, "result": item["status"],
                             "remaining_budget": self.remaining})
        if self.action_budget is not None:
            return {key: item[key] for key in ("evidence_id", "modality", "channel", "status", "rate_bpm",
                                              "quality", "quality_reasons", "reason") if key in item}
        return {key: value for key, value in item.items() if key not in ("peak_times_s", "candidate_periods_s")}

    def tools_for(self, modality: str) -> list:
        @tool(f"measure_{modality}_rate")
        def measure(channel: str) -> dict:
            """Measure the beat or pulse rate of the named channel over the requested synchronized window."""
            return self.observe(modality, "measure", channel)

        @tool(f"run_{modality}_model")
        def predict(channel: str) -> dict:
            """Run the configured pretrained model on the named channel and return its verified output type."""
            return self.observe(modality, "model", channel)

        return [measure, predict] if self.include_models else [measure]

    def invoke_agent(self, role: str, instructions: str, payload: dict, tools: list, limit: int) -> None:
        if self.action_budget is not None:
            limit = min(limit, self.remaining)
            if limit <= 0:
                return
            instructions += " Keep responses brief. Stop calling tools once you have enough evidence."
        @wrap_model_call
        def track_model(request, handler):
            self.costs.llm_invocations += 1
            try:
                response = handler(request)
            except Exception:
                self.costs.llm_usage(None)
                raise
            usage = next((msg.usage_metadata for msg in response.result if isinstance(msg, AIMessage)), None)
            self.costs.llm_usage(usage)
            return response

        agent = create_agent(
            self.chat_model,
            tools=tools,
            system_prompt=instructions,
            middleware=[
                ToolCallLimitMiddleware(run_limit=limit,
                                        exit_behavior="continue" if self.action_budget is not None else "end"),
                ModelCallLimitMiddleware(run_limit=limit + 2, exit_behavior="end"),
                track_model,
            ],
        )
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
                config={"recursion_limit": 8 * (limit + 3), "max_concurrency": 1},
            )
            messages = result["messages"]
            self.trace.append({"role": role, "messages": [message.model_dump(mode="json") for message in messages]})
        except Exception as exc:
            self.failures.append({"role": role, "reason": f"{type(exc).__name__}: {exc}"})

    def specialist(self, modality: str, task: str) -> None:
        wave = self.record.waves[modality]
        previous = [
            {key: item[key] for key in ("evidence_id", "channel", "status", "rate_bpm", "quality") if key in item}
            for item in self.evidence if item["modality"] == modality and item["operation"] == "measure"
        ]
        default = preferred_channel(modality, wave.channels)
        model_instruction = (
            f"Call run_{modality}_model exactly once on {default!r} if not already checked in this turn. "
            "Use its documented output as supporting evidence. "
            if self.include_models else ""
        )
        self.invoke_agent(
            modality,
            f"You are the {modality.upper()} waveform specialist. First call measure_{modality}_rate "
            f"on {default!r} for an initial analysis. In a requested recheck, choose an available "
            "channel not already measured when appropriate. "
            f"{model_instruction}"
            "Inspect the actual results and try one other channel if quality is review or evidence "
            "conflicts. Finish with a brief explanation and the evidence IDs.",
            {
                "task": task, "channels": wave.channels, "window_s": [self.start, self.end],
                "previous_measurements": previous,
            },
            self.tools_for(modality),
            SPECIALIST_BUDGET,
        )

    def initial(self, state: WorkflowState) -> dict:
        mode = state["mode"]
        if mode == "single":
            model_instruction = (
                "Call the pretrained model tool once for each available modality when inputs are compatible. "
                "ECGFounder supports lead I in this setup. "
                if self.include_models else ""
            )
            budget_instruction = ""
            if self.action_budget is not None:
                defaults = ", ".join(f"{m}={preferred_channel(m, wave.channels)}"
                                     for m, wave in self.record.waves.items())
                budget_instruction = (
                    f"Start with only the preferred channel of each modality: {defaults}. "
                    "Do not call alternative channels until you receive the initial results. "
                )
            self.invoke_agent(
                "single",
                "Measure each available modality with its tool. Review low quality or conflicting rates "
                f"using another available channel when appropriate. {model_instruction}{budget_instruction}"
                "Refer to actual evidence IDs.",
                {"record": self.record.describe(), "window_s": [self.start, self.end]},
                [t for modality in self.record.waves for t in self.tools_for(modality)],
                TOOL_BUDGET,
            )
            return {}
        for modality, wave in self.record.waves.items():
            if mode == "fixed":
                channel = preferred_channel(modality, wave.channels)
                self.observe(modality, "measure", channel)
                if self.include_models:
                    self.observe(modality, "model", channel)
            else:
                self.specialist(modality, "Measure rate and assess the evidence for the requested window.")
        return {}

    def review(self, state: WorkflowState) -> dict:
        if state["mode"] != "multi" or self.remaining <= 0:
            return {}

        @tool
        def recheck(modality: str, reason: str) -> dict:
            """Ask one available modality specialist to resolve a specific evidence problem."""
            self.costs.coordinator_recheck_calls += 1
            step = {"action": "coordinator_recheck", "modality": modality, "reason": reason}
            self.actions.append(step)
            if self.action_budget is not None:
                if self.remaining <= 0:
                    step.update(result="budget_exhausted", remaining_budget=self.remaining)
                    return {"status": "budget_exhausted"}
                self.remaining -= 1
                self.costs.action_budget_units += 1
            step["remaining_budget"] = self.remaining
            if modality not in self.record.waves:
                step["result"] = "unavailable"
                return {"status": "unavailable", "modality": modality}
            before = len(self.evidence)
            self.specialist(modality, reason)
            step.update(result="delegated", evidence_ids=[item["evidence_id"] for item in self.evidence[before:]])
            return {"new_evidence_ids": [item["evidence_id"] for item in self.evidence[before:]],
                    "updated_summary": self.agent_summary()}

        self.invoke_agent(
            "coordinator",
            "Compare synchronized measurements and evidence quality. Request a specialist recheck when "
            "a plausible alternative channel or evidence can resolve a discrepancy. End if results agree "
            "or further evidence is unavailable. Treat missing modalities and unresolved discrepancies "
            "explicitly. The final numeric report is computed from tool evidence.",
            {"available": list(self.record.waves), "unavailable": self.record.unavailable,
             "summary": self.agent_summary()},
            [recheck], REVIEW_BUDGET,
        )
        return {}

    def agent_summary(self) -> dict:
        summary = rate_summary(self.evidence, self.start, self.end)
        for item in summary["observed"].values():
            item.pop("hypotheses")
        return summary

    def finish(self, state: WorkflowState) -> dict:
        return {"report": {
            "mode": state["mode"],
            "record": self.record.describe(),
            "window_s": [self.start, self.end],
            "rate_summary": rate_summary(self.evidence, self.start, self.end),
            "evidence": self.evidence,
            "tool_calls": (TOOL_BUDGET if self.action_budget is None else self.action_budget) - self.remaining,
            "agent_trace": self.trace,
            "agent_errors": self.failures,
            "token_usage": {"input_tokens": self.costs.input_tokens, "output_tokens": self.costs.output_tokens},
            "actions": self.actions,
        }}

    def run(self, mode: str) -> dict:
        if mode not in ("fixed", "single", "multi"):
            raise ValueError(f"Unknown mode: {mode}")
        if mode != "fixed" and self.chat_model is None:
            raise ValueError("An LLM endpoint is required for Agent modes")
        graph = StateGraph(WorkflowState)
        graph.add_node("initial_analysis", self.initial)
        graph.add_node("review", self.review)
        graph.add_node("report", self.finish)
        graph.add_edge(START, "initial_analysis")
        graph.add_edge("initial_analysis", "review")
        graph.add_edge("review", "report")
        graph.add_edge("report", END)
        began = time.perf_counter()
        result = graph.compile().invoke({"mode": mode})["report"]
        result["duration_s"] = round(time.perf_counter() - began, 3)
        result["cost"] = self.costs.report(time.perf_counter() - began)
        return result


def preferred_channel(modality: str, channels: tuple[str, ...]) -> str:
    preferred = {"ecg": "I", "ppg": "brachial_880nm"}.get(modality)
    return preferred if preferred in channels else channels[0]
