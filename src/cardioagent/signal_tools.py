"""Compatibility import. Canonical implementation: cardioagent.common.signal_tools."""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module('cardioagent.common.signal_tools')
