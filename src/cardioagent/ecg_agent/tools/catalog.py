"""Explicit tool allowlist; annotation readers never appear in this registry."""
from cardioagent.ecg_agent.schemas import ToolSpec, WindowArgs, BeatArgs, ProfileArgs, CachedScoreArgs, ShiftedScoreArgs
import cardioagent.ecg_agent.tools.measurements as m
from cardioagent.ecg_agent.tools.ecgfounder import score_ecgfounder, score_ecgfounder_shifted


def build_registry(*, include_cached_model=False):
    from cardioagent.ecg_agent.harness.runtime import ToolRegistry
    registry=ToolRegistry()
    for name,description,args,handler in (
        ('assess_quality','Measure ECG signal quality; heuristic scores, not probabilities.',WindowArgs,m.quality),
        ('detect_beats','Detect window-local R peaks and median-RR rate; no rhythm diagnosis.',BeatArgs,m.beats),
        ('get_rate_rhythm','Return raw RR and optional cleaning mask/statistics; no AF label.',WindowArgs,m.rhythm),
        ('get_feature_profile','Coarse RR variability bins to locate suspicious elapsed intervals.',ProfileArgs,m.profile),
    ):
        registry.register(ToolSpec(name,description,args,handler))
    for name,description,handler in (
        ('get_p_wave_proxy','Experimental P-candidate amplitude and P-peak to R-peak distance.',m.p_wave),
        ('get_qrs_proxy','Experimental suprathreshold occupancy, not QRS duration or PVC count.',m.qrs),
        ('get_st_proxy','Experimental R+120 minus R-40 voltage; does not identify J point.',m.st),
        ('get_t_wave_proxy','Experimental T-candidate amplitude in a fixed search window.',m.t_wave),
        ('get_qt_proxy','Experimental QT/Bazett proxies using assumed Q onset and threshold T end.',m.qt),
    ):
        registry.register(ToolSpec(name,description,WindowArgs,handler,experimental=True,
                                  source='Adapted from ECG-Scroll (MIT); local corrected proxy semantics'))
    from . import physio
    for name, description, handler in (
        ('detect_beats_featuredb', 'Experimental local R peaks and raw RR from physical ECG; not a diagnosis.', physio.detect_beats),
        ('extract_average_beat_featuredb', 'Experimental complete-beat average; reports coverage without zero padding.', physio.average_beat),
        ('assess_template_quality', 'Experimental beat-template correlation and separate prescreen flags; not a probability.', physio.template_quality),
    ):
        registry.register(ToolSpec(name, description, WindowArgs, handler, experimental=True,
                                  version=physio.QUALITY_VERSION if name=='assess_template_quality' else physio.VERSION, source='Adapted FeatureDB + ecg_signal_quality; see third_party/physio_tools/source.json'))
    if include_cached_model:
        registry.register(ToolSpec(
            'score_ecgfounder', 'Read uncalibrated AF and AFL scores for one arrived offline cache window.',
            CachedScoreArgs, score_ecgfounder, source='ECGFounder released single-channel checkpoint',
        ))
        registry.register(ToolSpec(
            'score_ecgfounder_shifted', 'Score an arrived ten-second window shifted by ±5 seconds. '
            'Use actual shifted start/end coordinates. Sensitivity evidence only.',
            ShiftedScoreArgs, score_ecgfounder_shifted, experimental=True,
            source='ECGFounder released single-channel checkpoint; window shift by CardioAgent',
        ))
    return registry
