"""Compatibility import. Canonical implementation: cardioagent.legacy.multimodal.model_adapters."""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module('cardioagent.legacy.multimodal.model_adapters')
