"""Offline ECG1 FN inspection; preserves one-to-one matching and recorded predictions."""
import argparse
import csv
import json
from math import ceil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import wfdb

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.evaluation.metrics import greedy_pairs, intersection, union
from cardioagent.ecg_agent.evaluation.intervals import iou
from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import read_jsonl


COLORS = {'truth': '#058273', 'prediction': '#365c93', 'gap': '#c51b38',
          'score': '#687587', 'secondary': '#d38728', 'unknown': '#a6aeb8'}
TYPES = {'uncovered': '完全未覆盖', 'partial': '部分未覆盖',
         'covered_unmatched': '已覆盖，但事件匹配失败'}


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def uncovered(interval, predicted):
    a, b = interval
    cursor, gaps = a, []
    for start, end in intersection([interval], predicted):
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < b:
        gaps.append([cursor, b])
    return gaps


def collect(tasks, references, predictions):
    cases = []
    for key, ref in references.items():
        predicted = predictions[key]['intervals']
        matched = {j for _, j in greedy_pairs(predicted, ref['intervals'], ref['tau'])}
        for j, interval in enumerate(ref['intervals']):
            if j in matched:
                continue
            a, b = interval
            gaps = uncovered(interval, predicted)
            missed_s = union_seconds(gaps)
            covered_s = (b-a)-missed_s
            kind = 'uncovered' if covered_s < 1e-7 else 'covered_unmatched' if missed_s < 1e-7 else 'partial'
            focus = max(gaps, key=lambda p: p[1]-p[0]) if gaps else interval
            cases.append({'task_id': key, 'record_id': tasks[key]['record_id'], 'reference_index': j,
                          'start_s': a, 'end_s': b, 'duration_s': b-a, 'kind': kind,
                          'covered_percent': 100*covered_s/(b-a), 'uncovered_s': missed_s,
                          'uncovered_intervals': gaps, 'best_iou': max((iou(interval,p) for p in predicted), default=0.),
                          'focus_s': sum(focus)/2})
    return sorted(cases, key=lambda r: (r['record_id'], r['start_s']))


def load_record(args, name):
    folder = args.run / f'records/{name}'
    trace = list(lines(folder/'window_trace.jsonl'))
    secondary = [r for r in lines(folder/'evidence.jsonl')
                 if r['tool']=='score_ecgfounder' and r['lead']=='ECG2' and r['status']=='ok']
    cache_rows = {int(row['start_sample']): row for row in lines(args.cache/f'{name}.windows.jsonl')
                  if row['channel']=='ECG1'}
    waves = np.load(args.cache/f'{name}.npy', mmap_mode='r', allow_pickle=False)
    return trace, secondary, cache_rows, waves


def render_page(cases, page, total_pages, args, records, references, predictions):
    height = len(cases)*3.05+1.95
    fig, axes = plt.subplots(len(cases), 2, figsize=(20, height),
                             gridspec_kw={'width_ratios':[1.9,1]}, squeeze=False)
    fig.patch.set_facecolor('white')
    for row_number, case in enumerate(cases):
        left, right = axes[row_number]
        name, a, b = case['record_id'], case['start_s'], case['end_s']
        trace, secondary, cache_rows, waves = records[name]
        task = references[case['task_id']]
        lo, hi = max(task['start_s'], a-30), min(task['end_s'], b+30)
        scale = 60 if b-a > 180 else 1
        tx = lambda value: (value-a)/scale
        selected = [r for r in trace if r['end_s'] > lo and r['start_s'] < hi]
        for item in selected:
            score = max(item['af_score'],item['afl_score']) if item['af_score'] is not None else np.nan
            left.plot([tx(item['start_s']),tx(item['end_s'])],[score,score],
                      color=COLORS['score'], lw=1.)
            if item['decision'] in ('conflict','unavailable') or item['state'] in ('CANDIDATE','PENDING_END','UNKNOWN_HOLD'):
                left.axvspan(tx(max(lo,item['start_s'])),tx(min(hi,item['end_s'])),
                             ymin=0, ymax=.69, facecolor=COLORS['unknown'],alpha=.14,lw=0)
        for item in secondary:
            start,end=item['window_s']
            if end>lo and start<hi:
                score=max(item['result']['af_score'],item['result']['afl_score'])
                left.plot([tx(start),tx(end)],[score,score],color=COLORS['secondary'],lw=1.3,alpha=.8)
        for start,end in intersection(task['intervals'],[[lo,hi]]):
            left.broken_barh([(tx(start),(end-start)/scale)],(1.08,.08),facecolors=COLORS['truth'])
        for start,end in intersection(predictions[case['task_id']]['intervals'],[[lo,hi]]):
            left.broken_barh([(tx(start),(end-start)/scale)],(1.21,.08),facecolors=COLORS['prediction'])
        for start,end in case['uncovered_intervals']:
            left.broken_barh([(tx(start),(end-start)/scale)],(1.34,.08),facecolors=COLORS['gap'])
        left.axhline(.5,ls='--',lw=.8,color='#919aa7')
        left.axvline(0,lw=.7,color=COLORS['truth'],alpha=.55)
        left.axvline(tx(b),lw=.7,color=COLORS['truth'],alpha=.55)
        left.set(xlim=(tx(lo),tx(hi)),ylim=(-.03,1.48),yticks=[0,.5,1],ylabel='模型分数',
                 xlabel=f'相对该参考事件起点的时间（{"分钟" if scale==60 else "秒"}）')
        left.grid(alpha=.15)
        left.set_title(f"#{case['case_id']:02d}  记录 {name}  |  {a:.2f}–{b:.2f} 秒  |  {TYPES[case['kind']]}",
                       loc='left',fontsize=11,fontweight='bold',pad=8)
        left.text(.99,.98,f"参考时长 {b-a:.2f}s；覆盖 {case['covered_percent']:.1f}%；最大 IoU {case['best_iou']:.3f}",
                  ha='right',va='top',transform=left.transAxes,fontsize=8.5,
                  bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
        window_start = int(case['focus_s']//10)*2500
        source = cache_rows[window_start]
        wave_start, wave_end = source['start_sample']/250, source['end_sample']/250
        case['focus_window_start_s'],case['focus_window_end_s'] = wave_start,wave_end
        status='模型输入：500 Hz，归一化幅值'
        if source['waveform_row'] is not None:
            values=np.asarray(waves[source['waveform_row']])
            sample_time=wave_start+np.arange(len(values))/500
            unit='归一化幅值'
        else:
            raw=wfdb.rdrecord(str(args.raw_root/name),sampfrom=source['start_sample'],
                             sampto=source['end_sample'],channels=[0],physical=False)
            values=raw.d_signal[:,0].astype(float)
            sample_time=wave_start+np.arange(len(values))/250
            unit='原始 ADC'
            status='原始 ADC：该窗无可用模型输入'
        right.plot(sample_time-wave_start,values,color='#23394c',lw=.65)
        for start,end in intersection(task['intervals'],[[wave_start,wave_end]]):
            right.axvspan(start-wave_start,end-wave_start,color=COLORS['truth'],alpha=.13,lw=0)
        for start,end in intersection(case['uncovered_intervals'],[[wave_start,wave_end]]):
            right.axvspan(start-wave_start,end-wave_start,color=COLORS['gap'],alpha=.22,lw=0)
        observed=next(r for r in trace if r['start_s']==wave_start)
        case['focus_decision'],case['focus_reason']=observed['decision'],observed['reason']
        case['focus_state']=observed['state']
        score=max(observed['af_score'],observed['afl_score']) if observed['af_score'] is not None else None
        case['focus_primary_score']=score
        right.set(xlim=(0,wave_end-wave_start),xlabel=f'窗内时间（秒）；源位置 {wave_start:.0f}–{wave_end:.2f}s',ylabel=unit)
        right.set_title(f"局部 ECG1 波形 | {status}",fontsize=9,loc='left',pad=8)
        score_text = f"score={score:.3f}" if score is not None else 'score=不可用'
        right.text(.99,.98,f"{score_text} | {observed['state']}",
                   ha='right',va='top',transform=right.transAxes,fontsize=8,
                   bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
        right.grid(alpha=.15)
        # Both timelines and waves are retrospective inspection of already saved outputs.
        case['figure']=f'ecg1_missed_events_{page:02d}.png'
        case['figure_row']=row_number+1
    fig.suptitle(f'ECG1 新策略：事件级漏报核对  |  第 {page}/{total_pages} 页',
                 x=.055,ha='left',y=1-.15/height,fontsize=19,fontweight='bold')
    handles=[Patch(color=COLORS['truth'],label='参考 AF/AFL'),Patch(color=COLORS['prediction'],label='最终预测区间'),
             Patch(color=COLORS['gap'],label='实际未覆盖的 AF/AFL'),
             Line2D([],[],color=COLORS['score'],label='ECG1 分数'),
             Line2D([],[],color=COLORS['secondary'],label='实际调用的 ECG2 分数'),
             Patch(color=COLORS['unknown'],alpha=.3,label='待确认 / 未知')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.052,1-.52/height),ncol=6,frameon=False,fontsize=9)
    fig.text(.055,.32/height,'左侧按窗口范围展开分数，分数实际在窗口结束时取得；右侧仅展示选定局部窗口，不代表整个事件。',fontsize=9,color='#536174')
    fig.text(.055,.12/height,'事件 FN 使用原始一对一 IoU≥0.5 评分；已覆盖但未匹配不等于没有识别到 AF。此图仅作离线结果核对。',fontsize=9,color='#536174')
    fig.subplots_adjust(left=.055,right=.985,top=1-1.35/height,bottom=1./height,wspace=.15,hspace=.77)
    fig.savefig(args.output/f'ecg1_missed_events_{page:02d}.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/ecg-events-v1-r1/live/ecg1-event-adaptive'))
    parser.add_argument('--cache',type=Path,default=Path('runs/afdb-preprocessed-full-20261113'))
    parser.add_argument('--raw-root',type=Path,default=Path('D:/ECGagent/ECG/data/raw/afdb_local'))
    parser.add_argument('--tasks',type=Path,default=Path('runs/afdb-full-review/20261113-233212-64035fa8/agent_tasks.jsonl'))
    parser.add_argument('--references',type=Path,default=Path('runs/afdb-full-review/20261113-233212-64035fa8/evaluation_references.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('runs/ecg-events-v1-r1/analysis/ecg1-missed-events'))
    parser.add_argument('--rows',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.rows<=8: parser.error('--rows must be 1–8')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':9})
    tasks=read_jsonl(args.tasks,'task_id')
    refs=read_jsonl(args.references,'task_id')
    pred=read_jsonl(args.run/'predictions.jsonl','task_id')
    cases=collect(tasks,refs,pred)
    expected=sum(len(r['intervals'])-len(greedy_pairs(pred[key]['intervals'],r['intervals'],r['tau'])) for key,r in refs.items())
    assert len(cases)==expected
    args.output.mkdir(parents=True,exist_ok=True)
    records={name:load_record(args,name) for name in {c['record_id'] for c in cases}}
    for i,case in enumerate(cases,1): case['case_id']=i
    total=ceil(len(cases)/args.rows)
    for page,start in enumerate(range(0,len(cases),args.rows),1):
        render_page(cases[start:start+args.rows],page,total,args,records,refs,pred)
        print(f'page {page}/{total}',flush=True)
    counts={key:sum(c['kind']==key for c in cases) for key in TYPES}
    save_json(args.output/'index.json',{'policy':'live/ecg1-event-adaptive','event_FN':len(cases),
              'type_counts':counts,'pages':total,'rows_per_page':args.rows,
              'references_sha256':digest(args.references),'predictions_sha256':digest(args.run/'predictions.jsonl'),
              'waveform_selection':'center of largest truly uncovered span; if fully covered, reference midpoint',
              'cases':cases})
    fields=['case_id','record_id','start_s','end_s','duration_s','kind','covered_percent','uncovered_s','best_iou',
            'figure','figure_row','focus_window_start_s','focus_window_end_s','focus_primary_score','focus_decision','focus_reason','focus_state']
    with (args.output/'missed_events.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    readme=['# ECG1 事件级漏报可视化','',f'共 {len(cases)} 个事件 FN，分 {total} 张图，每张最多 {args.rows} 行。',
            '',f"完全未覆盖 {counts['uncovered']} 个；部分未覆盖 {counts['partial']} 个；已覆盖但未匹配 {counts['covered_unmatched']} 个。",'',
            '绿色为参考 AF/AFL，蓝色为最终预测，红色为真正未覆盖的参考时间。灰线为 ECG1 分数，橙线仅为实际取得的 ECG2 分数。',
            '右列显示局部模型输入；缺失时显示同一窗口原始 ADC，避免把未校准物理量标成 mV。',
            '这些是 61 个未匹配参考事件；成功匹配事件内部的时间级漏覆盖不包含在本图集中。','',
            '| 图 | FN 编号 |','|---|---|']
    for page,start in enumerate(range(0,len(cases),args.rows),1):
        name=f'ecg1_missed_events_{page:02d}.png'
        readme.append(f'| [{name}]({name}) | {start+1}–{min(start+args.rows,len(cases))} |')
    (args.output/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    archive=args.output/'ECG1_missed_events.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path != archive: z.write(path,path.name)
    print(json.dumps({'events':len(cases),'counts':counts,'pages':total,'output':str(args.output.resolve())},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
