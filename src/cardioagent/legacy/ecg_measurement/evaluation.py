"""Independent MIT-BIH beat annotation audit; annotations never enter the agent."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import wfdb
from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.event_audit import MITDB_BEATS, MITDB_RECORDS, event_errors, aggregate_events
from cardioagent.common.recording import Waveform
from cardioagent.legacy.multimodal.provenance import save_provenance
from cardioagent.legacy.ecg_measurement.agent import Config, ECGAgent


def read_ecg(stem):
    record = wfdb.rdrecord(str(stem))
    known = {'I','II','III','AVR','AVL','AVF','MLI','MLII','MLIII','ECG','ECG1','ECG2',
             'V','V1','V2','V3','V4','V5','V6'}
    indices = [i for i, name in enumerate(record.sig_name) if name.strip().upper() in known]
    if not indices:
        raise ValueError('No recognized ECG lead names in WFDB header')
    return Waveform(record.p_signal[:, indices], float(record.fs),
                    tuple(record.sig_name[i] for i in indices), tuple(record.units[i] for i in indices))


def audit(source: Path, output: Path, config=Config(), records=MITDB_RECORDS):
    if not records or len(set(records)) != len(records):
        raise ValueError('Provide distinct local record IDs')
    if any(Path(r).name != r or r in ('.','..') for r in records):
        raise ValueError('Record IDs must be local filename stems')
    output.mkdir(parents=True,exist_ok=False)
    report={'experiment':'ecg_agent_v1_mitdb','reference':'expert-reviewed .atr beat annotations',
            'tolerance_ms':50,'evaluation_role':'development audit; these six records have been used in earlier audits',
            'methods':{},'inputs':{}}
    groups={p:[] for p in ('primary','quality','verify')}
    for record_id in records:
        stem=source/record_id
        ecg=read_ecg(stem)
        # No annotation path, object or symbol is supplied to the agent.
        runs={p:ECGAgent(replace(config,policy=p)).run(ecg) for p in groups}
        annotation=wfdb.rdann(str(stem),'atr')
        expected=[float(i/ecg.fs) for i,s in zip(annotation.sample,annotation.symbol) if s in MITDB_BEATS]
        for policy,result in runs.items():
            ledger=result['ledger']; detected=[p for e in ledger for p in e['r_peaks_s']]
            paired=[]
            for entry in ledger:
                start,end=entry['window_s']; ref=[t for t in expected if start<=t<end]
                if entry['rate_bpm'] is not None and len(ref)>=4:
                    reference=60/float(np.median(np.diff(ref)))
                    paired.append(abs(entry['rate_bpm']-reference))
            row={'record_id':record_id,**event_errors(expected,detected,.05),
                 'windows':len(ledger),'returned_windows':sum(e['rate_bpm'] is not None for e in ledger),
                 'paired_rate_windows':len(paired),'rate_absolute_errors_bpm':paired,
                 'measurements':sum(e['cost']['measurements'] for e in ledger),
                 'qc_calls':sum(e['cost']['qc_calls'] for e in ledger),
                 'analysis_ms':sum(e['cost']['analysis_ms'] for e in ledger),
                 'review_windows':sum(e['status']=='review_required' for e in ledger),
                 'beat_report_delay_s':[e['committed_at_s']-t for e in ledger for t in e['r_peaks_s']]}
            groups[policy].append(row)
            save_json(output/f'{record_id}_{policy}_trace.json',result)
        report['inputs'][record_id]={ext:digest(stem.with_suffix('.'+ext)) for ext in ('hea','dat','atr')}
        print(f'ECG-only independent annotation audit: {record_id}',flush=True)
    for policy,rows in groups.items():
        errors=[e for r in rows for e in r['rate_absolute_errors_bpm']]
        delays=[e for r in rows for e in r['beat_report_delay_s']]
        n=sum(r['windows'] for r in rows)
        report['methods'][policy]={'beat_metrics_all_windows':aggregate_events(rows),
            'window_coverage':sum(r['returned_windows'] for r in rows)/n,
            'paired_rate_windows':len(errors),'rate_mae_bpm':float(np.mean(errors)) if errors else None,
            'mean_measurements_per_window':sum(r['measurements'] for r in rows)/n,
            'mean_qc_calls_per_window':sum(r['qc_calls'] for r in rows)/n,
            'mean_analysis_ms_per_window':sum(r['analysis_ms'] for r in rows)/n,
            'mean_beat_report_delay_s':float(np.mean(delays)) if delays else None,'records':rows}
    report['limitations']=['Rate MAE is conditional on returned windows with >=4 reference beats; beat recall includes abstained windows.',
        'Beat report delay is chunk commit delay, not arrhythmia detection latency.',
        'Fixed heuristic policies; no trained controller, AF/PVC/ST classification or deployment risk guarantee.',
        'Record-level results are not a patient-independent held-out clinical validation.']
    report['provenance']=save_provenance(output,{'experiment':report['experiment'],'inputs':report['inputs'],'chunk_s':config.chunk_s,'measurement_budget':config.measurement_budget})
    save_json(output/'ecg_agent_audit.json',report)
    return report
