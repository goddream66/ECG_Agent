"""Compatible entry; implementation: cardioagent.ecg_agent.experiments.audit."""
from cardioagent.ecg_agent.experiments.audit import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.ecg_agent.experiments.audit'),name)
if __name__=="__main__":main()
