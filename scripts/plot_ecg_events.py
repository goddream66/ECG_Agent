"""Compatible entry; implementation: cardioagent.ecg_agent.evaluation.plots."""
from cardioagent.ecg_agent.evaluation.plots import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.ecg_agent.evaluation.plots'),name)
if __name__=="__main__":main()
