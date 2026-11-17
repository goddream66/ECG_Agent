"""Compatible entry; implementation: cardioagent.legacy.ecg_measurement.full_review_runner."""
from cardioagent.legacy.ecg_measurement.full_review_runner import main
def __getattr__(name):
    from importlib import import_module
    return getattr(import_module('cardioagent.legacy.ecg_measurement.full_review_runner'),name)
if __name__=="__main__":main()
