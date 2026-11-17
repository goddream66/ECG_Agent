"""Compatibility import. Canonical implementation: cardioagent.ecg_agent.data.preprocessing."""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module('cardioagent.ecg_agent.data.preprocessing')
