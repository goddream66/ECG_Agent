"""Separate evidence requests, actual computation, and orchestration overhead."""

from dataclasses import asdict, dataclass


@dataclass
class Costs:
    waveform_requests: int = 0
    waveform_tool_calls: int = 0
    waveform_failures: int = 0
    waveform_insufficient: int = 0
    waveform_cache_hits: int = 0
    coordinator_recheck_calls: int = 0
    channel_qc_calls: int = 0
    channel_qc_ms: float = 0.
    llm_invocations: int = 0
    llm_usage_missing: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    action_budget_units: int = 0

    def waveform(self, result: dict, *, cached: bool = False) -> None:
        self.waveform_requests += 1
        self.waveform_cache_hits += int(cached)
        self.waveform_tool_calls += int(not cached)
        self.waveform_failures += int(not cached and result["status"] == "failed")
        self.waveform_insufficient += int(result["status"] == "insufficient")
        self.action_budget_units += 1

    def llm_usage(self, usage) -> None:
        if usage is None:
            self.llm_usage_missing += 1
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def report(self, seconds: float) -> dict:
        return asdict(self) | {"wall_time_s": round(seconds, 6)}
