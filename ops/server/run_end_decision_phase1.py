"""Paired AFDB end-action pilot; one process/visible GPU and one record at a time."""
import argparse
import json
from pathlib import Path

from cardioagent.ecg_agent.experiments.runner import initialize, run_job


RUNS = Path('/data/1shared/gaojiajun/ECGagent/runs')
TARGET = RUNS / 'afdb-end-phase1-20261119'
SPANS = {'04126': (480.0, 700.0), '04936': (1600.0, 1850.0),
         '06453': (3000.0, 3130.0)}
SOURCE = RUNS / 'afdb-unified-extended-20261119/ecg1/records/04126/job_manifest.json'
RECORD_SOURCE = RUNS / 'ecg-scroll-afdb-t080-20261117T104112Z'


def make_job(arm, lead, record_id):
    job = json.loads(SOURCE.read_text(encoding='utf-8'))['job']
    if record_id != '04126':
        path = (RECORD_SOURCE / f'live-{lead}' / f'{lead}-event-adaptive' /
                'records' / record_id / 'job_manifest.json')
        reference = json.loads(path.read_text(encoding='utf-8'))['job']
        job['record'] = reference['record']
        job['task'] = reference['task']
    job['task']['start_s'], job['task']['end_s'] = SPANS[record_id]
    job['target'] = str(TARGET / arm / lead / 'records' / record_id)
    job['run_id'] = f'end-phase1-{arm}-{lead}-{record_id}'
    job['manifest_id'] = f'end-phase1-20261119-{record_id}'
    job['channel'] = lead.upper()
    job['device'] = 'cuda:0'
    job['threads'] = 4
    job['mode'] = 'live'
    job['policy'] = 'event-react'
    job['llm'].update(device='cuda:0', max_rounds=5, max_new_tokens=1024,
                      boundary_tools=True, scheduled_evidence=True,
                      event_control=True, unified_event_decision=True,
                      separate_final_decision=arm in ('B', 'C'),
                      rule_safety_fallback=arm == 'C')
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--lead', choices=('ecg1', 'ecg2'), required=True)
    parser.add_argument('--records', nargs='+', choices=tuple(SPANS), default=list(SPANS))
    args = parser.parse_args()
    initialize('live', '/data/1shared/gaojiajun/ECGagent/models', 4, 'cuda:0')
    for record_id in args.records:
        job = make_job(args.arm, args.lead, record_id)
        result, reused = run_job(job)
        print(json.dumps({'arm': args.arm, 'lead': args.lead, 'record_id': record_id,
                          'target': job['target'], 'intervals': result['intervals'],
                          'reused': reused}), flush=True)


if __name__ == '__main__':
    main()
