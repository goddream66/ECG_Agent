"""Scientific figures for M0/M1; inputs are completed evaluation artifacts only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from cardioagent.ecg_agent.evaluation.metrics import greedy_pairs
from cardioagent.common.io import read_jsonl


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def plot_tradeoffs(report, output):
    from matplotlib.lines import Line2D
    fig, axes=plt.subplots(2,3,figsize=(16,9))
    markers={'baseline':'s','simple-baseline':'^','fixed':'D','cached':'o','live':'*'}
    colors={'baseline':'#7c8795','simple-baseline':'#dc8c30','fixed':'#8253a3','cached':'#14798e'}
    labels={'adaptive':'Old','hysteresis':'Hyst','fixed-evidence':'Fixed','event-adaptive':'New','dual':'Dual'}
    offsets={'Old':(5,-12),'Hyst':(-20,10),'Fixed':(5,-13),'New':(5,8),'Dual':(-22,-14)}
    for result in report['results']:
        if result['phase']=='live': continue
        m=result['metrics']
        channel=result['name'][:4]
        row=0 if channel=='ecg1' else 1
        color=colors[result['phase']]
        marker=markers[result['phase']]
        hours=sum(row['duration_s'] for row in result['rows'])/3600
        points=((m['model_calls']/hours,m['micro_episode_f1']),
                (m['non_af_alarm_minutes'],m['time_recall']),
                (m['unknown_coverage'],m['micro_episode_f1']))
        name=labels.get(result['name'][5:])
        for col,(x,y) in enumerate(points):
            if col==1 and y < .92: continue
            ax=axes[row,col]
            ax.scatter(x,y,c=color,marker=marker,s=45,alpha=.85,edgecolors='white',linewidths=.4)
            if name:
                ax.annotate(name,(x,y),xytext=offsets[name],textcoords='offset points',fontsize=8,
                            arrowprops={'arrowstyle':'-','color':color,'lw':.5})
    for row in range(2):
        axes[row,0].set(xlabel='Logical model calls / recording hour',ylabel=f'ECG{row+1}\nEvent F1',ylim=(-.025,.73))
        axes[row,1].set(xlabel='False-alarm time (minutes)',ylabel='Time recall',ylim=(.92,1.003))
        axes[row,2].set(xlabel='Unknown / undecided fraction',ylabel='Event F1',ylim=(-.025,.73))
    for ax,title in zip(axes[0],('Accuracy and acquisition cost','Time-level tradeoff (zoom)','Coverage and accuracy')):
        ax.set_title(title)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('AFDB: 23 full records | exploratory development | fixed primary configuration = New',fontsize=14)
    fig.legend(handles=[Line2D([],[],marker=markers[p],color=colors[p],linestyle='none',label=label)
                        for p,label in (('baseline','Original baselines'),('simple-baseline','Simple controls'),
                                        ('fixed','Fixed evidence'),('cached','Closed-loop rules'))],
               loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False)
    fig.text(.5,.012,'Time panels zoom recall to 0.92–1.00; zero-return budget settings remain in the other panels and tables.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,top=.91,bottom=.13,wspace=.27,hspace=.32)
    fig.savefig(output/'risk_cost_coverage.png',dpi=170)
    plt.close(fig)


def plot_examples(args,output):
    refs=read_jsonl(args.references,'task_id')
    old=read_jsonl(args.baseline/'ecg1-adaptive/predictions.jsonl','task_id')
    new_path=args.runs/'cached/ecg1-event-adaptive'
    if not (new_path/'predictions.jsonl').exists(): return
    new=read_jsonl(new_path/'predictions.jsonl','task_id')
    index=[]
    for record in ('06453','07859','08378'):
        key=next(k for k in refs if k.startswith('afdb-'+record+'-'))
        reference=refs[key]
        rows=list(lines(new_path/f'records/{record}/window_trace.jsonl'))
        fig,axes=plt.subplots(3,1,figsize=(16,8),gridspec_kw={'height_ratios':[1.4,1,1]},constrained_layout=True)
        x=np.array([r['start_s']/3600 for r in rows]+[rows[-1]['end_s']/3600])
        y=[max(r['af_score'],r['afl_score']) if r['af_score'] is not None else np.nan for r in rows]
        axes[0].stairs(y,x,color='#6b7785',linewidth=.6)
        axes[0].axhline(.5,linestyle='--',color='#929aaa',linewidth=.8)
        axes[0].set(ylabel='Primary AF/AFL score',ylim=(-.03,1.1))
        for ax,pred,title in ((axes[1],old[key]['intervals'],'Old adaptive'),(axes[2],new[key]['intervals'],'New event-adaptive')):
            pairs=greedy_pairs(pred,reference['intervals'])
            matched={i for i,_ in pairs}
            for a,b in reference['intervals']:
                ax.broken_barh([(a/3600,(b-a)/3600)],(.1,.25),facecolors='#058273')
            for i,(a,b) in enumerate(pred):
                ax.broken_barh([(a/3600,(b-a)/3600)],(.55,.25),facecolors='#365c93' if i in matched else '#c61b39')
            ax.set(yticks=[],ylim=(0,1),ylabel=title)
            ax.text(.01,.86,f'{len(pred)} predictions; {len(pred)-len(matched)} unmatched',transform=ax.transAxes,fontsize=9)
        for row in rows:
            if row['decision'] in ('conflict','unavailable'):
                axes[2].axvspan(row['start_s']/3600,row['end_s']/3600,color='#9ca7af',alpha=.18,linewidth=0)
        for ax in axes:
            ax.set_xlim(0,reference['end_s']/3600)
            ax.grid(axis='x',alpha=.15)
        axes[-1].set_xlabel('Time from recording start (hours)')
        fig.suptitle(f'AFDB {record}, ECG1 | frozen threshold 0.5 | one-to-one IoU >= 0.5')
        axes[0].legend(handles=[Patch(color='#058273',label='Reference AF/AFL'),
                               Patch(color='#365c93',label='Matched prediction'),
                               Patch(color='#c61b39',label='Unmatched prediction'),
                               Patch(color='#9ca7af',label='Unknown evidence')],loc='upper right',fontsize=8,ncol=4)
        path=output/f'event_comparison_{record}.png'
        fig.savefig(path,dpi=170)
        plt.close(fig)
        index.append({'record':record,'file':path.name,'selection':'predeclared failure cases from prior development'})
    (output/'examples.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')


def plot_live_cost(report, output):
    results=[r for r in report['results'] if r['phase']=='live']
    if not results: return
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=[r['name'].replace('event-adaptive','new').replace('legacy-dual','dual') for r in results]
    axes[0].bar(names,[r['metrics']['actual_model_calls'] for r in results],color=['#287e8e' if 'new' in n else '#989fa9' for n in names])
    axes[1].bar(names,[r['metrics']['neural_seconds']/60 for r in results],color=['#287e8e' if 'new' in n else '#989fa9' for n in names])
    axes[0].set_ylabel('Actually executed model calls')
    axes[1].set_ylabel('Summed neural execution time (minutes)')
    fig.suptitle('Live CPU inference; summed service times under concurrent workers, not isolated latency')
    fig.savefig(output/'live_cost.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/ecg-events-v1-r1'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/afdb-full-review/full-cpu'))
    parser.add_argument('--references',type=Path,default=Path('runs/afdb-full-review/20261113-233212-64035fa8/evaluation_references.jsonl'))
    args=parser.parse_args()
    output=args.runs/'analysis/figures'
    output.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.runs/'analysis/comparison.json').read_text(encoding='utf-8'))
    plot_tradeoffs(report,output)
    plot_examples(args,output)
    plot_live_cost(report,output)
    print(output.resolve())


if __name__=='__main__': main()
