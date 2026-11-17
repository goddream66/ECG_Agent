"""Reapply only the revised QC decision to immutable stored quality measurements."""
from copy import deepcopy
import hashlib,json
from cardioagent.ecg_agent.tools.physio import template_correlation_passes,QUALITY_VERSION
from cardioagent.ecg_agent.data.label_quality import summarize_quality

def reassess_window_quality(annotation,evidence):
    old_status=annotation['quality_status']
    if old_status not in ('no_quality_flag','review_required'):
        return old_status,list(annotation.get('quality_reasons',[])),deepcopy(evidence)
    revised=deepcopy(evidence);quality=revised.get('assess_template_quality')
    if quality is not None and quality['status']=='ok':
        original_id=quality['evidence_id']
        original_version=quality['tool_version']
        result=quality['result']
        passed=template_correlation_passes(result.get('template_correlation'))
        result.update(template_consistent=passed,quality_label='template_consistent' if passed else 'template_inconsistent',
                      heuristic_parameters={'correlation_threshold':.8,'group_count_gate_enabled':False})
        quality.update(evidence_id='qc-reassessment:'+hashlib.sha256((original_id+QUALITY_VERSION).encode()).hexdigest(),
                       tool_version=QUALITY_VERSION,source_evidence_id=original_id,source_tool_version=original_version,
                       evidence_kind='derived_rule_reassessment',duration_ms=0.,
                       derivation='Decision recomputed from stored correlation; waveform and detector not rerun')
        quality['warnings']=quality.get('warnings',[])+['Group count retained for audit only; no quality gating.']
    summary=summarize_quality([],list(revised.values()))
    return summary['status'],summary['reasons'],revised
