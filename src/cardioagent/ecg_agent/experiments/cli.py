"""Public event CLI; legacy commands are routed separately."""
import argparse
import os
from pathlib import Path
from cardioagent.common.paths import code_root, default_inputs
from cardioagent.ecg_agent.models.ecgfounder import normalize_ecg_device

def main():
    from . import runner
    defaults=default_inputs()
    parser=argparse.ArgumentParser(description='ECG event experiments')
    parser.add_argument('--models',type=Path,default=defaults['models'])
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('runs'))
    sub=parser.add_subparsers(dest='command',required=True).add_parser('track-afdb')
    sub.add_argument('--mode',choices=('live','cached','fixed'),required=True)
    for name in ('cache','tasks','scores','baseline'): sub.add_argument('--'+name,type=Path,default=defaults[name])
    sub.add_argument('--config',type=Path,default=code_root()/'configs/ecg/policy.json')
    sub.add_argument('--policies', nargs='+')
    sub.add_argument('--llm-config', type=Path)
    sub.add_argument('--run-dir',type=Path,required=True)
    sub.add_argument('--workers',type=int,default=4)
    sub.add_argument('--threads',type=int,default=2)
    sub.add_argument('--device',type=normalize_ecg_device,default=os.environ.get('CARDIOAGENT_ECG_DEVICE','cpu'))
    sub.add_argument('--limit-records',type=int,default=0)
    sub.add_argument('--duration-s',type=float,default=0.)
    sub.add_argument('--channels',choices=('ECG1','ECG2'),nargs='+',default=['ECG1','ECG2'])
    args=parser.parse_args()
    if args.workers<1 or args.threads<1 or args.limit_records<0 or args.duration_s<0: parser.error('Invalid resource or subset argument')
    args.output=args.run_dir
    before=Path.cwd()
    try:
        os.chdir(code_root());runner.run(args)
    finally: os.chdir(before)
