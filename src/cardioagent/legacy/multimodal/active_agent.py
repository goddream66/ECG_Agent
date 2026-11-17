"""LLM action selection over the same observations and action mask as rules."""

import json

from cardioagent.legacy.multimodal.acquisition import run_acquisition


def active_agent(record, budget: int, observe, client, mode: str, llm_budget: int, *, structured: bool = False) -> dict:
    trace = []
    errors = []

    def ask(state, legal, role, recommendation=None):
        payload = state.state(legal, structured=structured)
        if recommendation is not None:
            payload["advisory_action"] = recommendation
        instruction = (
            f"Role: {role}. Select one legal action for reliable cardiac rate under a waveform budget. "
            "STOP when observed evidence is usable and sufficient. Acquire new evidence if it can resolve "
            "low quality or disagreement. ABSTAIN when evidence is insufficient and no action helps. "
            "Output only the exact action string from legal_actions."
        )
        state.costs.llm_invocations += 1
        try:
            answer = client.invoke([("system", instruction), ("human", json.dumps(payload, separators=(",", ":")))])
        except Exception as exc:
            state.costs.llm_usage(None)
            errors.append({"role": role, "reason": f"{type(exc).__name__}: {exc}"})
            trace.append({"role": role, "state": payload, "result": "llm_failed"})
            return None
        state.costs.llm_usage(answer.usage_metadata)
        action = answer.content.strip() if isinstance(answer.content, str) else ""
        trace.append({"role": role, "state": payload, "response": answer.content,
                      "action": action if action in legal else None,
                      "result": "valid" if action in legal else "invalid_action"})
        return action if action in legal else None

    def decide(state, legal):
        remaining = llm_budget - state.costs.llm_invocations
        if remaining <= 0:
            trace.append({"role": mode, "result": "llm_budget_exhausted"})
            return "STOP" if "STOP" in legal else "ABSTAIN"
        if mode == "single_active":
            return ask(state, legal, "single policy") or "ABSTAIN"
        recommendation = None
        if remaining >= 2:
            observed = state.evidence[-1]["modality"].upper() if state.evidence else "physiological signals"
            recommendation = ask(state, legal, f"{observed} evidence specialist")
        return ask(state, legal, "coordinator", recommendation) or "ABSTAIN"

    result = run_acquisition(record, budget, observe, mode, decide=decide)
    return {
        "rate_summary": {k: v for k, v in result.items() if k not in ("evidence", "actions", "cost")},
        "evidence": result["evidence"], "actions": result["actions"], "cost": result["cost"],
        "tool_calls": result["tool_calls"], "agent_trace": trace, "agent_errors": errors,
    }
