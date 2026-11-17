"""Compatible entry; implementation: cardioagent.ecg_agent.tools.export."""
from cardioagent.ecg_agent.tools.export import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.ecg_agent.tools.export'),name)
if __name__=="__main__":main()
