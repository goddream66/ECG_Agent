from cardioagent.common.paths import code_root
"""Regenerate the checked-in catalog without loading data or models."""
import json
from pathlib import Path
from cardioagent.ecg_agent.harness import build_registry

def main():
    destination=code_root()/'configs/ecg/tools.schema.json'
    destination.write_text(json.dumps({'schema_version':1,'tools':build_registry(include_cached_model=True).schemas(True)},indent=2)+'\n',encoding='utf-8')
    print(destination)


if __name__ == "__main__":
    main()
