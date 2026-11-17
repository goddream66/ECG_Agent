"""Prepare/check the offline labeler; --run explicitly starts MC-MED labeling."""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from cardioagent.common.paths import code_root
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.data import build_mcmed as builder
from cardioagent.ecg_agent.data.ecgfounder_preprocessing import load_profile, preprocess_window
from cardioagent.ecg_agent.data.label_quality import POLICY
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_label_scorer, normalize_ecg_device
from cardioagent.ecg_agent.schemas import EvidenceContext


def load_config(path, **overrides):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {'schema_version', 'source_root', 'output_root', 'models', 'split', 'patients',
                'duration_s', 'max_patient_scan', 'seed', 'device', 'save_scores', 'quality_policy'}
    if set(config) != expected or config['schema_version'] != 1 or config['quality_policy'] != POLICY:
        raise ValueError('Unsupported or incomplete MC-MED labeling configuration')
    config.update({k: v for k, v in overrides.items() if v is not None})
    for name in ('patients', 'duration_s', 'max_patient_scan', 'seed'):
        if type(config[name]) is not int:
            raise ValueError(f'{name} must be an integer')
    if (config['patients'] < 1 or config['max_patient_scan'] < config['patients']
            or config['duration_s'] < 10 or config['duration_s'] % 10
            or config['split'] not in ('train', 'val', 'test')
            or type(config['save_scores']) is not bool):
        raise ValueError('Invalid patient, window, split or score-export configuration')
    config['device'] = normalize_ecg_device(config['device'])
    for name in ('source_root', 'output_root', 'models'):
        config[name] = Path(config[name]).resolve()
    if (config['output_root'].is_relative_to(config['source_root'])
            or config['source_root'].is_relative_to(config['output_root'])):
        raise ValueError('Source and output directories must be separate')
    config.pop('schema_version')
    config.pop('quality_policy')
    return argparse.Namespace(**config, mode='prelabel')


def configure_device(device):
    if device.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        if ',' in os.environ['CUDA_VISIBLE_DEVICES'] or not os.environ['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')


def check_ready(args):
    """Read headers/assets and run synthetic tools/model only; no patient inference."""
    checkpoint = args.models / '1_lead_ECGFounder.pth'
    for path in (args.source_root/'visits.csv',
                 args.source_root/f'split_random_{args.split}.csv', checkpoint):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty required asset: {path}')
    if not (args.source_root/'waveforms').is_dir():
        raise ValueError('Missing source waveform directory')
    with (args.source_root/'visits.csv').open(newline='') as stream:
        if not {'MRN', 'CSN'}.issubset(next(csv.reader(stream), [])):
            raise ValueError('visits.csv is missing patient/visit identity columns')
    t = np.arange(5000)/500
    raw = sum(np.exp(-((t-p)/.025)**2) for p in np.arange(.5, 10., 1.))
    raw += .03*np.sin(2*np.pi*1.3*t)
    session = ToolSession(ToolContext(Waveform(raw[:, None], 500, ('II',), ('mV',)), 0., 10.),
                          allow_experimental=True,
                          evidence_context=EvidenceContext('labeler-preflight', 'synthetic', 'synthetic'))
    responses = [session.call(name, {'lead': 'II', 'start_s': 0., 'end_s': 10.}) for name in builder.TOOLS]
    if any(r['status'] != 'ok' for r in responses):
        raise RuntimeError('Synthetic tool check failed: ' + json.dumps(responses))
    predict, labels, model_info = create_ecgfounder_label_scorer(args.models, device=args.device)
    scores = predict(preprocess_window(raw, 500))
    builder.automatic_labels(scores, labels)  # Validate full head and export contract.
    return {'status': 'ready', 'patient_inference': False, 'synthetic_model_calls': 1,
            'dataset_package_created': False, 'builder_version': builder.VERSION,
            'label_selection': 'ranked_candidates', 'quality_policy': POLICY,
            'preprocessing': load_profile(), 'model': model_info,
            'checkpoint_sha256': builder.sha(checkpoint),
            'labels_sha256': builder.sha(code_root()/'third_party/ecgfounder/tasks.txt'),
            'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'wfdb', 'neurokit2', 'torch')},
            'tools': [{'name': r['tool'], 'version': r['tool_version'], 'status': r['status']}
                      for r in responses]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=code_root()/'configs/ecg/mcmed_labeling.json')
    parser.add_argument('--run', action='store_true', help='Start labeling; default only checks readiness')
    parser.add_argument('--patients', type=int)
    parser.add_argument('--duration-s', type=int)
    parser.add_argument('--device')
    parser.add_argument('--report', type=Path, help='Optional readiness report path, outside source data')
    cli = parser.parse_args()
    try:
        args = load_config(cli.config, patients=cli.patients, duration_s=cli.duration_s, device=cli.device)
        configure_device(args.device)
        if cli.run:
            if cli.report:
                raise ValueError('--report is for readiness checks only')
            print('DATASET_OUTPUT=' + str(builder.build(args)), flush=True)
            return
        storage = Path(os.environ.get('ECG_AGENT_STORAGE', '/data/1shared/gaojiajun/ECGagent'))
        report = cli.report or storage/'runs/mcmed-labeling-preflight'/(
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]+'.json')
        report = report.resolve()
        if (report.is_relative_to(args.source_root) or report.is_relative_to(code_root())
                or report.is_relative_to(args.models) or report.is_relative_to(args.output_root)):
            raise ValueError('Readiness report must be separate from source, code, models and dataset packages')
        result = {'status': 'failed', 'dataset_package_created': False, 'patient_inference': False,
                  'config_sha256': builder.sha(cli.config),
                  'resolved_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        try:
            result.update(check_ready(args))
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print('READINESS_REPORT=' + str(report), flush=True)
        print('READY: source tables, tools, preprocessing and 150-label model checked; no patient labeling started.')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Labeler preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
