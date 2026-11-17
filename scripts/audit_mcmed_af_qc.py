"""Read-only audit of 44 AF-score-high, QC-downgraded windows; no clinical relabeling."""
from pathlib import Path
from collections import Counter
import json,hashlib,math
import numpy as np
import wfdb
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

storage=Path('/data/1shared/gaojiajun/ECGagent')
source=storage/'datasets/mcmed-ecg-agent/v0.5.0/20261116T145225Z-full-d2be7f5f'
out=storage/'runs/mcmed-qc-af-review/20261116-44-windows'
out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir();(out/'sheets').mkdir()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((source/'manifest.json').read_text())
for name,h in manifest['artifacts'].items():assert sha(source/name)==h
allrows=rows(source/'task_annotations.jsonl')
records={r['segment_id']:r for r in json.loads((source/'records.json').read_text())}
evidence={e['evidence_id']:e for e in rows(source/'task_evidence.jsonl')}
targets=[r for r in allrows if r['af_score'] is not None and r['af_score']>=.6 and r['quality_status']=='review_required']
assert len(targets)==44
lookup={(r['segment_id'],r['start_s']):r for r in allrows}
def intervals(use_quality):
    result=[];active=None
    for r in allrows:
        pos=r['tasks']['af']['state']=='candidate_positive' if use_quality else r['tasks']['af']['proposed_state']=='candidate_positive'
        contiguous=active is not None and active['segment_id']==r['segment_id'] and abs(active['end_s']-r['start_s'])<1e-6
        if active and (not pos or not contiguous):result.append(active);active=None
        if pos:
            if active is None:active={'segment_id':r['segment_id'],'start_s':r['start_s'],'end_s':r['end_s']}
            active['end_s']=r['end_s']
    if active:result.append(active)
    for r in result:r['duration_s']=r['end_s']-r['start_s']
    return result
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
audit=[];central=[];subtypes=Counter()
for index,w in enumerate(targets,1):
    rec=records[w['segment_id']];fs=int(rec['fs']);start=max(0.,w['start_s']-30);end=min(rec['end_s'],w['end_s']+30)
    raw=wfdb.rdrecord(str(Path(rec['header']).with_suffix('')),sampfrom=round(start*fs),sampto=round(end*fs))
    factor={'mv':1.,'uv':.001,'v':1000.}[raw.units[0].lower()]
    x=raw.p_signal[:,0]*factor;t=start+np.arange(len(x))/fs
    a=round((w['start_s']-start)*fs);b=round((w['end_s']-start)*fs);x0=x[a:b];t0=np.arange(len(x0))/fs
    q=next(evidence[i] for i in w['evidence_ids'] if evidence[i]['tool']=='assess_template_quality')
    beat=next(evidence[i] for i in w['evidence_ids'] if evidence[i]['tool']=='detect_beats_featuredb')
    qv=q['result'];bv=beat['result'];corr=qv.get('template_correlation');groups=qv.get('coefficient_group_count')
    subtype='group_rule_only' if corr is not None and corr>.8 and groups>=3 else ('correlation_only' if groups<3 else 'correlation_and_group')
    subtypes[subtype]+=1
    neighbors=[lookup[(w['segment_id'],s)] for s in np.arange(start,end-9.99,10) if (w['segment_id'],s) in lookup]
    p=np.array(bv['r_peaks_samples'],dtype=int);rr=np.asarray(bv['rr_intervals_ms'])
    filtered=sosfiltfilt(butter(4,[.67,40],fs=fs,btype='bandpass',output='sos'),x0)
    context_high=sum(r['af_score'] is not None and r['af_score']>=.6 for r in neighbors)
    item={'id':index,'window_id':w['window_id'],'segment_id':w['segment_id'],'start_s':w['start_s'],'end_s':w['end_s'],
      'af_score':w['af_score'],'template_correlation':corr,'coefficient_group_count':groups,'failure_subtype':subtype,
      'prescreen_flags':qv['prescreen_flags'],'quality_reasons':w['quality_reasons'],
      'raw_finite':bool(np.isfinite(x0).all()),'raw_peak_to_peak_mv':float(np.ptp(x0)),
      'detected_beats':len(p),'rr_cv':float(np.std(rr)/np.mean(rr)) if len(rr)>1 else None,
      'context_start_s':start,'context_end_s':end,'high_score_context_windows':context_high,
      'neighbor_scores':[{'start_s':r['start_s'],'af_score':r['af_score'],'quality':r['quality_status']} for r in neighbors],
      'engineering_finding':'Only template heuristic triggered; not proof of artifact or AF.',
      'clinical_review':'pending','human_af':None,'human_quality':None,
      'case_figure':f'cases/case-{index:02d}.png'}
    audit.append(item);central.append((w,x0,p,corr,groups))
    fig,axes=plt.subplots(4,1,figsize=(17,12),gridspec_kw={'height_ratios':[1.2,1.6,1.,1.]})
    fig.subplots_adjust(left=.065,right=.97,top=.9,bottom=.07,hspace=.66)
    fig.suptitle(f"Case {index:02d} | {w['segment_id']} | {w['start_s']:g}-{w['end_s']:g} s | AF score {w['af_score']:.3f}",x=.065,ha='left',fontsize=19,fontweight='bold')
    fig.text(.065,.925,f"QC: correlation {corr:.3f}, coefficient groups {groups} | {subtype} | no prescreen flags | offline review",color='#9b580c',fontsize=11)
    axes[0].plot(t,x,lw=.65,color='#33465a');axes[0].axvspan(w['start_s'],w['end_s'],alpha=.2,color='#edaa4f')
    axes[0].set_title('Raw lead II context (up to 30 seconds before and after; no segment concatenation)',loc='left')
    axes[0].set_xlim(start,end);axes[0].set_ylabel('mV');axes[0].set_xlabel('Segment-relative seconds')
    axes[1].plot(t0,x0,lw=.85,color='#20374a',label='Raw');axes[1].plot(t0,filtered,lw=.7,alpha=.65,color='#14818b',label='Bandpass view')
    axes[1].scatter(p/fs,x0[p],s=17,color='#bf6413',zorder=4,label='Existing detected R peaks')
    axes[1].set_xlim(0,10);axes[1].set_ylabel('mV');axes[1].set_xlabel('Seconds within target window')
    axes[1].xaxis.set_major_locator(MultipleLocator(1));axes[1].xaxis.set_minor_locator(MultipleLocator(.2))
    axes[1].set_title('Target window: raw signal, display-only filtered signal and original tool peaks',loc='left');axes[1].legend(loc='upper right',ncol=3,fontsize=9)
    for r in neighbors:
        score=r['af_score'];quality=r['quality_status'];c='#c87819' if quality=='review_required' else '#167d8a'
        if score is not None:axes[2].bar(r['start_s'],score,width=9.6,align='edge',color=c,alpha=.8)
    axes[2].axhline(.6,ls='--',lw=1,color='#9d2744');axes[2].axhline(.4,ls=':',lw=1,color='#8796a1')
    axes[2].axvspan(w['start_s'],w['end_s'],facecolor='none',edgecolor='#9d2744',lw=2)
    axes[2].set_xlim(start,end);axes[2].set_ylim(0,1.05);axes[2].set_ylabel('AF score')
    axes[2].set_title('Existing scores: orange = QC review; teal = no QC flag; thresholds 0.4 / 0.6',loc='left')
    for r in neighbors:
        ev=next((evidence[i] for i in r['evidence_ids'] if evidence[i]['tool']=='detect_beats_featuredb'),None)
        if ev and ev['status']=='ok':
            axes[3].scatter(ev['result']['rr_start_times_s'],ev['result']['rr_intervals_ms'],s=14,color='#33465a')
    axes[3].axvspan(w['start_s'],w['end_s'],alpha=.2,color='#edaa4f');axes[3].set_xlim(start,end)
    axes[3].set_ylabel('Raw RR (ms)');axes[3].set_xlabel('Segment-relative seconds')
    axes[3].set_title('Original uncleaned RR evidence; irregularity can reflect rhythm, ectopy or peak errors',loc='left')
    for ax in axes:ax.grid(alpha=.18)
    fig.text(.065,.025,'AF score and QC correlation are not clinical confirmation. Future context is used for offline review only. Labels remain unchanged.',fontsize=10,color='#586b7c')
    fig.savefig(out/item['case_figure'],dpi=125,facecolor='white');plt.close(fig)
    print('CASE',index,flush=True)
for page in range(11):
    fig,axes=plt.subplots(4,1,figsize=(18,10))
    fig.subplots_adjust(left=.06,right=.97,top=.91,bottom=.11,hspace=.8)
    fig.suptitle(f'AF / QC audit: target windows {page*4+1:02d}-{page*4+4:02d} of 44',x=.06,ha='left',fontsize=21,fontweight='bold')
    for ax,(w,x,p,corr,groups),entry in zip(axes,central[page*4:page*4+4],audit[page*4:page*4+4]):
        fs=int(records[w['segment_id']]['fs']);t=np.arange(len(x))/fs
        ax.plot(t,x,lw=.8,color='#263f53');ax.scatter(p/fs,x[p],s=12,color='#b46b17',zorder=3)
        ax.set_title(f"{entry['id']:02d} | {w['segment_id']} | {w['start_s']:g}-{w['end_s']:g} s | AF={w['af_score']:.3f} | corr={corr:.3f}, groups={groups} | {entry['failure_subtype']}",loc='left',fontsize=11)
        ax.set_xlim(0,10);ax.set_ylabel('mV');ax.set_xlabel('Seconds within target');ax.xaxis.set_major_locator(MultipleLocator(1));ax.grid(alpha=.18)
    fig.text(.06,.025,'All 44 trigger template inconsistency only. Markers are tool detections, not adjudicated beats. Clinical AF/quality labels remain pending.',fontsize=11,color='#586b7c')
    fig.savefig(out/f'sheets/sheet-{page+1:02d}.png',dpi=120,facecolor='white');plt.close(fig)
before=intervals(False);after=intervals(True)
summary={'reviewed_windows':44,'failure_subtypes':dict(subtypes),
 'all_prescreen_flags_false':all(not any(r['prescreen_flags'].values()) for r in audit),
 'all_target_samples_finite':all(r['raw_finite'] for r in audit),
 'counterfactual_analysis_only':{'raw_score_only':{'windows':61,'events':len(before),'events_ge30s':sum(r['duration_s']>=30 for r in before),'max_duration_s':max(r['duration_s'] for r in before)},
 'current_quality_policy':{'windows':17,'events':len(after),'events_ge30s':sum(r['duration_s']>=30 for r in after),'max_duration_s':max(r['duration_s'] for r in after)}},
 'clinical_af_confirmed':False,'source_labels_changed':False}
(out/'audit_windows.json').write_text(json.dumps(audit,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'counterfactual_intervals.json').write_text(json.dumps({'score_only':before,'current_policy':after},indent=2)+'\n')
html='<meta charset="utf-8"><title>44-window AF / QC review</title><h1>44个AF高分、质量降级窗口</h1><p>每例含前后各30秒上下文。全部为离线复查材料，不改变标签，不代表房颤确诊。</p>'
for i in range(11):html+=f'<p><a href="sheets/sheet-{i+1:02d}.png">汇总图 {i+1}</a></p>'
for r in audit:html+=f'<p><a href="{r["case_figure"]}">案例{r["id"]:02d} {r["segment_id"]} {r["start_s"]:g}秒 AF={r["af_score"]:.3f} corr={r["template_correlation"]:.3f} groups={r["coefficient_group_count"]}</a></p>'
(out/'index.html').write_text(html,encoding='utf8')
report='# 44个AF高分、质量降级窗口复核\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'
report+='44例均仅被模板一致性规则降级。没有任何预筛查标志触发；这不等于没有噪声。分组规则单独失败者需要重点复核。对原分数取消模板质量降级的反事实结果仅衡量策略影响，不能当成真实性能改善或AF真值。全部临床AF与质量结论仍待人工。\n\n'
report+='|案例|片段|起点秒|AF分数|相关性|分组数|触发类型|\n|---|---|---|---|---|---|---|\n'
for r in audit:report+=f'|{r["id"]:02d}|{r["segment_id"]}|{r["start_s"]:g}|{r["af_score"]:.3f}|{r["template_correlation"]:.3f}|{r["coefficient_group_count"]}|{r["failure_subtype"]}|\n'
(out/'REPORT.md').write_text(report,encoding='utf8')
(out/'manifest.json').write_text(json.dumps({'source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),'script_sha256':sha(Path(__file__)),'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}},indent=2))
print('SUMMARY='+json.dumps(summary),flush=True);print('OUTPUT='+str(out),flush=True)
