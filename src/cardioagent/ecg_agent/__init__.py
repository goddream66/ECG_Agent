"""ECG event agent. Legacy measurement classes are loaded only when requested."""
__all__=['ECGEventAgent','EventConfig','EventTracker','Config','ECGAgent']
def __getattr__(name):
    from importlib import import_module
    if name == 'ECGEventAgent': return getattr(import_module('.agent', __name__), name)
    if name in ('EventConfig','EventTracker'): return getattr(import_module('.memory.events', __name__), name)
    if name in ('Config','ECGAgent'): return getattr(import_module('cardioagent.legacy.ecg_measurement.agent'), name)
    raise AttributeError(name)
