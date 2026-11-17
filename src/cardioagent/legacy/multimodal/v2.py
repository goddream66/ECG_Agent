"""V2 M2 observation workflow, independent of V1 rules and LLM decisions."""

from cardioagent.common.recording import Recording
from cardioagent.signal.events import detect_events
from cardioagent.signal.quality import quality_features
from cardioagent.signal.synchronization import inspect_axes
from cardioagent.legacy.multimodal.workflow import preferred_channel


def inspect_recording(
    recording: Recording, *, documented_common_grid: bool, start: float | None = None,
    end: float | None = None, all_channels: bool = False, calibration: dict | None = None,
) -> dict:
    axes, sync = inspect_axes(recording, documented_common_grid, calibration)
    start = sync["common_window_s"][0] if start is None else start
    end = sync["common_window_s"][1] if end is None else end
    if not sync["common_window_s"][0] <= start < end <= sync["common_window_s"][1] + 1e-9 or end - start < 5:
        raise ValueError("Select at least 5 seconds within the common recording bounds")
    observations = []
    for modality, wave in recording.waves.items():
        axis = axes[modality]
        channels = wave.channels if all_channels else (preferred_channel(modality, wave.channels),)
        for channel in channels:
            values, first = axis.select(wave, channel, start, end)
            observations.append({
                "evidence_id": f"E{len(observations) + 1:03}",
                "modality": modality, "channel": channel, "window_s": [start, end],
                "signal_quality": quality_features(values, wave.fs, modality),
                "model_quality": None,
                **detect_events(modality, values, axis, first),
            })
    return {
        "schema": "cardioagent.v2.events.v2",
        "record": recording.describe(), "window_s": [start, end],
        "synchronization": sync, "observations": observations,
    }
