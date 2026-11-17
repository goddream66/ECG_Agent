from cardioagent.common.paths import code_root
"""Read-only verification of completed event checkpoints and evidence links."""
import argparse
import json
from pathlib import Path

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.memory.evidence import canonical_hash
from cardioagent.ecg_agent.evaluation.metrics import validate_intervals


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def audit(root):
    report={'phases':{},'complete':True}
    project=code_root()
    for phase in ('fixed','cached','live'):
        folder=root/phase
        if not (folder/'completion.json').exists():
            report['phases'][phase]={'status':'pending'}
            report['complete']=False
            continue
        manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['provenance']['inference_source'].items():
            if digest(project/name)!=expected:
                raise ValueError(f'Inference source changed: {name}')
        manifest_id=canonical_hash(manifest['provenance'])
        jobs=evidence_count=updates_count=0
        for path in sorted(folder.glob('*/records/*/job_manifest.json')):
            record=path.parent
            checkpoint=json.loads(path.read_text(encoding='utf-8'))
            job=checkpoint['job']
            if checkpoint['job_hash']!=canonical_hash(job) or job['manifest_id']!=manifest_id:
                raise ValueError(f'Checkpoint identity mismatch: {path}')
            for name,expected in checkpoint['artifacts'].items():
                if digest(record/name)!=expected:
                    raise ValueError(f'Artifact changed: {record/name}')
            evidence={}
            for item in rows(record/'evidence.jsonl'):
                if item['evidence_id'] in evidence:
                    raise ValueError('Repeated evidence ID within record')
                if item['record_id']!=job['record']['record_id'] or item['task_id']!=job['task']['task_id']:
                    raise ValueError('Evidence identity mismatch')
                if item['tool'] not in ('score_ecgfounder','assess_quality','get_rate_rhythm'):
                    raise ValueError('Unexpected tool in inference evidence')
                if item['window_s'] and item['window_s'][1]>item['available_at_s']:
                    raise ValueError('Future evidence')
                evidence[item['evidence_id']]=item['available_at_s']
                evidence_count+=1
            for name in ('assessments','event_updates'):
                for item in rows(record/f'{name}.jsonl'):
                    cursor=item['recorded_at_s']
                    if any(key not in evidence or evidence[key]>cursor for key in item['evidence_ids']):
                        raise ValueError('Missing/future cited evidence')
                    update=item.get('event_update')
                    if update and (update['recorded_at_s']!=cursor or update['interval'][1]>cursor):
                        raise ValueError('Retroactive or future event write')
                    updates_count+=name=='event_updates'
            result=json.loads((record/'result.json').read_text(encoding='utf-8'))
            validate_intervals(result['intervals'],job['task']['start_s'],job['task']['end_s'])
            if phase=='fixed' and not result['integrity']['identical']:
                raise ValueError('Frozen evidence is not identical')
            if phase=='live' and result['execution']['actual_model_calls']!=result['counts']['model_calls']:
                raise ValueError('Live execution/model-success counts differ; inspect errors')
            jobs+=1
        expected=json.loads((folder/'completion.json').read_text(encoding='utf-8'))['jobs']
        if jobs!=expected:
            raise ValueError('Missing record checkpoint')
        report['phases'][phase]={'status':'verified','jobs':jobs,'evidence':evidence_count,'event_updates':updates_count}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/ecg-events-v1-r1'))
    args=parser.parse_args()
    result=audit(args.runs)
    (args.runs/'analysis').mkdir(exist_ok=True)
    save_json(args.runs/'analysis/audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
