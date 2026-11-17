"""Subject-disjoint target calibration diagnostics with fixed risk threshold."""
from hashlib import sha256
import numpy as np
from cardioagent.legacy.multimodal.policy.stopping import fit_calibrator, calibration_metrics, leave_one_subject_out


def audit(trials, limit=.2):
    subjects=sorted({r['subject_id'] for r in trials}, key=lambda s:sha256(str(s).encode()).digest())
    if len(subjects)<4:
        return {'status':'insufficient_subjects', 'subjects':len(subjects), 'required':4}
    fit_ids=set(subjects[::2]); calibration=[r for r in trials if r['subject_id'] in fit_ids]; validation=[r for r in trials if r['subject_id'] not in fit_ids]
    try: model=fit_calibrator(calibration,'subject_record')
    except ValueError as exc: return {'status':'unavailable','reason':str(exc)}
    returned=[r for r in validation if model.predict(r['raw_risk'])<=limit]
    return {'status':'diagnostic_only','fixed_threshold':limit,'calibration_subjects':len(fit_ids),'assessment_subjects':len(subjects)-len(fit_ids),'held_out_metrics':calibration_metrics(validation,model,'subject_record'),'coverage':len(returned)/len(validation),'selective_failure':float(np.mean([r['failed'] for r in returned])) if returned else None,'by_scenario':{scenario: calibration_metrics([r for r in validation if r['scenario']==scenario],model,'subject_record') for scenario in sorted({r['scenario'] for r in validation})},'loso':leave_one_subject_out(trials)}


def fit_ood(x):
    """Training-only standardized distance diagnostic, not a rejection rule."""
    x=np.asarray(x,dtype=float); center=np.median(x,axis=0); scale=np.maximum(np.median(abs(x-center),axis=0)*1.4826,.1)
    distance=np.mean(((x-center)/scale)**2,axis=1)
    return {'center':center.tolist(),'scale':scale.tolist(),'threshold':float(np.quantile(distance,.99))}


def ood_score(model,x):
    distance=float(np.mean(((np.asarray(x)-model['center'])/model['scale'])**2))
    return {'distance':distance,'out_of_distribution':distance>model['threshold']}
