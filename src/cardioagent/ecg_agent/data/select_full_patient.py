"""Select every available II segment for one additional, deterministic patient."""
import hashlib
from pathlib import Path
import wfdb
from cardioagent.ecg_agent.data.build_mcmed import patient_index


def select_full_patient(root, split, excluded, seed, max_scan=1000):
    root=Path(root).resolve();patients=patient_index(root,split)
    order=sorted(patients,key=lambda p:hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    inspected=0
    for patient in order:
        if patient in excluded:continue
        inspected+=1
        if inspected>max_scan:break
        records=[];issues=[]
        for visit in sorted(patients[patient]):
            folder=root/'waveforms'/visit[-3:]/visit/'II'
            for header in sorted(folder.glob('*.hea')):
                try:
                    h=wfdb.rdheader(str(header.with_suffix('')))
                    if h.n_sig!=1 or h.sig_name!=['II'] or h.fs<=80 or not float(h.fs).is_integer():
                        raise ValueError('unsupported channel or sampling rate')
                    unit=h.units[0].strip().lower().replace('μ','u').replace('µ','u')
                    if unit not in ('mv','uv','v') or h.sig_len<=0:raise ValueError('invalid unit or length')
                    files=sorted({(header.parent/name).resolve() for name in h.file_name})
                    if not all(p.is_relative_to(root) and p.is_file() and p.stat().st_size for p in files):
                        raise ValueError('missing or invalid data file')
                    records.append({'patient_id':patient,'visit_id':visit,'segment_id':header.stem,
                        'header':str(header.resolve()),'data_files':list(map(str,files)),
                        'fs':float(h.fs),'signal_samples':int(h.sig_len),'unit':h.units[0],
                        'segment_start_datetime':str(h.base_datetime) if h.base_datetime else None,
                        'start_s':0.,'end_s':h.sig_len/h.fs,'split':split})
                except (OSError,ValueError,IndexError) as exc:
                    issues.append({'header':str(header),'reason':str(exc)})
        if records:
            records.sort(key=lambda r:(r['segment_start_datetime'] or '',r['header']))
            return records,{'scope':'all_available_II_segments_all_visits_for_one_patient',
                'patient_id':patient,'excluded_patient_ids':sorted(excluded),
                'source_visit_ids':sorted(patients[patient]),'header_issues':issues,
                'patients_scanned':inspected,'selected_patients':1,
                'retained_duration_s':sum(r['end_s'] for r in records),
                'gaps_preserved':True,'short_final_windows':'retained without model label'}
    raise ValueError('No additional patient with usable ECG headers found')
