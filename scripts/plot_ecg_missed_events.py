"""Compatible entry; implementation: cardioagent.ecg_agent.evaluation.missed_events."""
from cardioagent.ecg_agent.evaluation.missed_events import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.ecg_agent.evaluation.missed_events'),name)
if __name__=="__main__":main()
