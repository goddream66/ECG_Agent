from pathlib import Path
import torch
from cardioagent.policy.audit_v24 import _trials
from cardioagent.policy.conditional_value import ConditionalNetwork, Variant
from cardioagent.policy.selection import run
from cardioagent.policy.replay import read_replay
from cardioagent.policy.domain_diagnostics import audit
from cardioagent.dataset import save_json

torch.set_num_threads(2)
d=torch.load('runs/20261112-182420-ec2a1dfa/models.pt',map_location='cpu',weights_only=True)
s=d['models']['conditional_voi']; model=ConditionalNetwork(s['width'],Variant(**s['variant'])); model.load_state_dict(s['state_dict']); model.eval()
cases=read_replay(Path('runs/v24-replay/bidmc-paired.jsonl'))[1]
trials=[t for r,p in cases if r['split']=='validation' and p.task=='heart_rate' for t in _trials(r,run(p,d['budget'],d['costs'],d['priors'],model=model))]
report=audit(trials)
save_json(Path('runs/v242_bidmc_calibration_diagnostic.json'),report)
print({k:v for k,v in report.items() if k!='loso'})
