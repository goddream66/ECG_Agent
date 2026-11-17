"""Offline evaluation. Old beat-audit entry points remain lazy aliases."""
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.legacy.ecg_measurement.evaluation'), name)
