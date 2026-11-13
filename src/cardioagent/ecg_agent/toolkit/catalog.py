"""Explicit tool allowlist; annotation readers never appear in this registry."""
from .contracts import ToolSpec, WindowArgs, BeatArgs, ProfileArgs
from . import measurements as m


def build_registry():
    from .runtime import ToolRegistry
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
    return registry
