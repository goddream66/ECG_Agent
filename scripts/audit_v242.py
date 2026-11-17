"""Frozen-policy V2.4.2 mechanism audit; no changes to V2.4.1 weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from cardioagent.common.io import save_json, digest
from cardioagent.legacy.multimodal.policy import shared_risk
from cardioagent.legacy.multimodal.policy.qc_continuous import fit as fit_qc
from cardioagent.legacy.multimodal.policy.qc_prior import validate_priors
from cardioagent.legacy.multimodal.policy.replay import read_replay
from cardioagent.legacy.multimodal.policy.conditional_value import ConditionalNetwork, Variant
from cardioagent.legacy.multimodal.policy.selection import run
from cardioagent.legacy.multimodal.policy.stopping import trial_weights
from cardioagent.legacy.multimodal.policy.value import run_policy


def vector(task, result):
    x = shared_risk.features(task, result['rate_bpm'], result['evidence'])
    for modality in ('ecg', 'ppg', 'pcg'):
        rows = [r for r in result['evidence'] if r['modality'] == modality]
        for key in ('interval_mad_ratio','plausible_interval_fraction','autocorrelation'):
            v = [r.get('event_quality', {}).get(key) for r in rows]
            v = [float(a) for a in v if a is not None and np.isfinite(a)]
            x.extend([np.mean(v) if v else 0., bool(v)])
        v = [r.get('event_score_median') for r in rows if r.get('event_score_median') is not None]
        x.extend([np.mean(v) if v else 0., bool(v)])
    previews = list(result.get('previews', {}).values())
    x.extend([len(previews), np.mean([r['score'] for r in previews]) if previews else 0.])
    return x


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = int(y.sum()), int((1-y).sum())
    auc = float((rankdata(p)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    ece = 0.
    for lo in np.arange(0, 1, .1):
        mask = (p >= lo) & (p < lo+.1 if lo < .9 else p <= 1)
        if mask.any(): ece += mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return {'n':len(y),'auc':auc,'brier':float(np.mean((p-y)**2)), 'ece_10_bins':float(ece)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--replays',type=Path,nargs='+',required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); torch.set_num_threads(3); torch.manual_seed(242)
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    stored=data['models']['conditional_voi']; model=ConditionalNetwork(stored['width'],Variant(**stored['variant'])); model.load_state_dict(stored['state_dict']); model.eval()
    cases=[c for path in args.replays for c in read_replay(path)[1]]
    subjects={s:{r['subject_id'] for r,p in cases if r['split']==s} for s in ('train','validation','test')}
    assert not (subjects['train']&subjects['validation'] or subjects['train']&subjects['test'] or subjects['validation']&subjects['test'])
    train=[(r,p) for r,p in cases if r['split']=='train' and not(r['dataset']=='bidmc' and p.task=='heart_rate')]
    priors=data['priors'] | {'continuous':fit_qc(train,data['budget'])}
    rows=[]
    for i,(row,profile) in enumerate(cases):
        if row['split']=='train' and row['dataset']=='bidmc' and profile.task=='heart_rate': continue
        for method in ('v241','v242','channel_quality','channel_eig'):
            if method.startswith('v'):
                result=run(profile,data['budget'],data['costs'],priors if method=='v242' else data['priors'],model=model)
            else:
                result=run_policy(profile,data['budget'],method,data['costs']['waveform'])
            rate=result['rate_bpm']; trials=result['stop_trials']
            rows.append({'split':row['split'],'subject_id':row['subject_id'],'case_id':row['case_id'],'group':f"{row['dataset']}/{profile.task}/{row['scenario']}",'method':method,'error':None if rate is None else abs(rate-row['reference_bpm']), 'x':None if rate is None else vector(profile.task,result),'native':trials[-1]['risk'] if trials else 1-result['quality_rank'], 'cost':result['cost']['analysis_ms']})
        if i%100==0: print(f'Evaluated {i}/{len(cases)}',flush=True)
    eligible=[r for r in rows if r['split']=='train' and r['x'] is not None]
    x=np.array([r['x'] for r in eligible]); center=x.mean(0); scale=np.maximum(x.std(0),.1)
    tx=torch.tensor((x-center)/scale,dtype=torch.float32); y=torch.tensor([r['error']>2 for r in eligible],dtype=torch.float32)
    w=trial_weights(eligible,'subject_record'); w=torch.tensor(w/w.sum(),dtype=torch.float32)
    scorers={}
    for name,net in [('linear',torch.nn.Linear(x.shape[1],1)),('mlp',torch.nn.Sequential(torch.nn.Linear(x.shape[1],16),torch.nn.Tanh(),torch.nn.Linear(16,1)))]:
        opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=.001)
        for _ in range(200):
            opt.zero_grad(); loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(tx).flatten(),y,reduction='none')*w).sum(); loss.backward(); opt.step()
        scorers[name]=net.eval()
    report={'experiment':'v242_frozen_policy_qc_refinement','checkpoint_sha256':digest(args.checkpoint),'replays':{str(p):digest(p) for p in args.replays},'qc_validation':validate_priors([(r,p) for r,p in cases if r['split']=='validation'],priors),'groups':{}}
    for split in ('validation','test'):
        for group in sorted({r['group'] for r in rows if r['split']==split}):
            for method in ('v241','v242','channel_quality','channel_eig'):
                allrows=[r for r in rows if r['split']==split and r['group']==group and r['method']==method]; valid=[r for r in allrows if r['x'] is not None]
                if not valid: continue
                xx=torch.tensor((np.array([r['x'] for r in valid])-center)/scale,dtype=torch.float32)
                yy=np.array([r['error']>2 for r in valid]); errors=np.array([r['error'] for r in valid])
                scores={k:torch.sigmoid(net(xx)).detach().numpy().ravel() for k,net in scorers.items()}; scores['native']=np.clip([r['native'] for r in valid],0,1); scores['oracle']=errors
                entry={'total':len(allrows),'returned':len(valid),'mae':float(errors.mean()),'mean_analysis_ms':float(np.mean([r['cost'] for r in allrows])),'risk_scores':{}}
                for name,p in scores.items():
                    points=[]
                    for fraction in (.25,.5,.75,1.):
                        n=max(1,int(np.ceil(fraction*len(valid)))); idx=np.argsort(p,kind='stable')[:n]
                        points.append({'returned':n,'coverage':n/len(allrows),'risk':float(yy[idx].mean()),'mean_acquisition_cost_ms':entry['mean_analysis_ms']})
                    entry['risk_scores'][name]={'metrics':metrics(yy,p) if name!='oracle' else None,'matched_count_curve':points}
                report['groups'][f'{split}/{group}/{method}']=entry
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'v242_audit.json',report)
    torch.save({'center':center.tolist(),'scale':scale.tolist(),'scorers':{k:v.state_dict() for k,v in scorers.items()},'priors':priors},args.output/'refinement.pt')
    print(args.output,flush=True)

if __name__=='__main__': main()
