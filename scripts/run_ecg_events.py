"""Compatible executable; implementation lives in the importable experiment package."""
from cardioagent.ecg_agent.experiments.runner import main
def __getattr__(name):
    from cardioagent.ecg_agent.experiments import runner
    return getattr(runner,name)
if __name__=='__main__': main()
