"""Compatible entry; implementation: cardioagent.ecg_agent.experiments.analysis."""
from cardioagent.ecg_agent.experiments.analysis import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.ecg_agent.experiments.analysis'),name)
if __name__=="__main__":main()
