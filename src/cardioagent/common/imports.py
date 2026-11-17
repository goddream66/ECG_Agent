from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

def upstream_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream implementation: {path}")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
