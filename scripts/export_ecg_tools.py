"""Regenerate the checked-in catalog without loading data or models."""
import json
from pathlib import Path
from cardioagent.ecg_agent.toolkit import build_registry

if __name__=='__main__':
    destination=Path(__file__).resolve().parents[1]/'configs/ecg_tools_v1.json'
    destination.write_text(json.dumps({'schema_version':1,'tools':build_registry().schemas(True)},indent=2)+'\n',encoding='utf-8')
    print(destination)
