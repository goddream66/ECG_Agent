"""Lightweight router: ECG commands use the current package, old commands stay compatible."""
import sys
def main():
    if 'track-afdb' in sys.argv[1:]:
        from cardioagent.ecg_agent.experiments.cli import main as run
    else:
        from cardioagent.legacy.multimodal.cli import main as run
    run()
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.legacy.multimodal.cli'),name)
if __name__=='__main__':main()
