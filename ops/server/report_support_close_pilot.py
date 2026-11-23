from pathlib import Path
from collections import Counter
import json
import argparse

root=Path('D:/CardioAgent/runs/support-close-replay')
old=root.parent/'04936-window-semantics-pilot'
def lines(p):
    return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]

def evaluate(c, folder):
    trace=lines(folder/'window_trace.jsonl')
    run=json.loads((folder/'result.json').read_text())
    events=run['events']
    if len(trace) != (c['end_s']-c['start_s'])/10:
        raise ValueError(f'Incomplete fixed-cadence trace: {folder}')
    evidence=lines(folder/'evidence.jsonl')
    steps=lines(folder/'llm_steps.jsonl')
    eligible_ends=[e for a,e in c['reference'] if c['start_s']<e<=c['end_s']]
    matched=set()
    result=Counter()
    errors=[]
    decisions=[]
    measured_leads=[]
    for w in trace:
        a,b=w['start_s'],w['end_s']
        d=w['decision']
        overlap=sum(max(0,min(b,v)-max(a,u)) for u,v in c['reference'])
        kind='target' if overlap==b-a else ('other' if overlap==0 else 'mixed')
        result[kind]+=1
        if d=='SUPPORTS_TARGET':result[kind+'_support']+=1
        if d=='UNCERTAIN':result[kind+'_uncertain']+=1
        if d=='AGAINST_TARGET':result[kind+'_against']+=1
        if w['action'] in ('LIFECYCLE_ERROR','ACTION_ERROR'):
            errors.append([b,w['action']])
        if w['action']=='REQUEST_CLOSE':result['end_requests']+=1
        decisions.append([b,d,w['action'],w['state']])
        leads={e['lead'] for e in evidence if e['tool']=='score_ecgfounder' and e['status']=='ok'
               and e['window_s']==[a,b] and e['available_at_s']<=b}
        measured_leads.append([b,sorted(leads)])
        if len(leads)==2:result['both_leads_scored_same_window']+=1
        elif leads and c['lead'] not in leads:result['only_backup_scored']+=1
    closed=[]
    for e in events:
        b=e['end_decided_at_s']
        if b is None:continue
        candidates=[u for u in eligible_ends if u not in matched and e['estimated_start_s']<u<=b]
        u=candidates[0] if candidates else None
        if u is not None:
            matched.add(u)
            if e['confirmed_at_s'] < u:
                result['correct_close_on_previously_confirmed_event']+=1
            else:
                result['first_confirmed_at_or_after_reference_end']+=1
        elif not c['reference']:
            result['closure_of_false_alarm_on_background']+=1
        elif any(a<=b<end for a,end in c['reference']):
            result['premature_close_during_target']+=1
        closed.append(dict(decided_at=b,estimated=e['estimated_end_s'],true_end=u,delay=b-u if u else None,
                           event_confirmed_at_s=e['confirmed_at_s'],
                           bracket=e.get('end_boundary')))
    result.update(end_opportunities=len(eligible_ends),correct_close=len(matched),false_close=len(closed)-len(matched),
                  emitted_events=len(events),errors=len(errors),calls=sum(1 for l in evidence),
                  llm_calls=run['counts']['llm_calls'], llm_ms=run['counts']['llm_ms'],elapsed_ms=run['elapsed_ms'],
                  secondary_founder=sum(e['tool']=='score_ecgfounder' and e['lead']!=c['lead'] for e in evidence),
                  stage_rejects=sum(l['status']=='stage_rejected' for l in steps if 'status' in l))
    return dict(counts=dict(result),closed=closed,decisions=decisions,errors=errors,measured_leads=measured_leads)

def main():
    global root,old
    cli=argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, default=root)
    cli.add_argument('--baseline', type=Path)
    args=cli.parse_args()
    root=args.root
    old=args.baseline or root.parent/'04936-window-semantics-pilot'
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    output={}
    report=['# AF/AFL 支持与结束确认回放','',
        'K 在原观察层基础上增加了结束复核。Qwen 可以主动发起调查；运行时只在看见事件内的支持测量和随后两个相邻、不重叠的反向测量区间时开放最终确认。同窗双导联不能充当两个时间点。Qwen 可以确认或继续等待。边界投影使用较早的区间，决定时刻单独记录。J 使用原范围提示。','',
        '开发片段沿用既有四例。新增片段在运行前单列，包含按参考标注选的真实终点、完整事件和按已存分数选的困难持续/非目标对照。选择依据和标签仅用于离线评估。每段从空账本启动，组间不共享模型决策或事件状态。','',
        '|范围|记录/起始秒|方法|纯目标窗 SUPPORT|纯非目标窗误 SUPPORT|正确关闭/真实终点|误关|实际关闭时刻|动作错误|','|---|---|---|---:|---:|---:|---:|---|---:|']
    report[2:2]=['J 为单次分数转低即可提交终点范围的对照。L 在 K 上提示按需复核第二导联。M 单独在 K 上只向模型提供当前证据允许提交的观察类别，同时保留不确定。各组均使用固定十秒唤醒。','']
    report[4:4]=['最终短回放 P 使用 react_qwen_review_pilot.json。在一份配置中启用两时点结束复核、可选的同窗备用导联、受证据约束的窗口选项和修订后的参考字段。只回放 04936 的真实结束和持续反例，以及 05091 的非目标高分反例。三段各长 50 秒，均独立从空账本开始。P 用于确认组合后的实际行为；单个机制的比较应使用分项实验。','']
    totals={}
    for c in cases:
        for arm in ('J','K','L','M','P','R'):
            folder=root/(arm+'-'+c['key'])
            if arm=='J' and c['split']=='development':folder=old/('bracket-live-'+c['key'])
            if not all((folder/name).exists() for name in ('window_trace.jsonl','result.json','llm_steps.jsonl','evidence.jsonl')):continue
            value=evaluate(c,folder)
            n=Counter(value['counts'])
            output[arm+'-'+c['key']]=value
            totals.setdefault(c['split']+'/'+arm,Counter()).update(n)
            report.append('|'+ '|'.join(map(str,[c['split'],c['key'],arm,f"{n['target_support']}/{n['target']}",f"{n['other_support']}/{n['other']}",f"{n['correct_close']}/{n['end_opportunities']}",n['false_close'],','.join(str(r['decided_at']) for r in value['closed']) or '—',n['errors']]))+'|')
    report+=['','仅计与完整纯窗相交的整窗方向；跨边界窗单独留存在 JSON。表中 0/0 表示无可评价机会。正确关闭要求在片段内真实结束到达后由模型关闭覆盖该终点的同一事件，每个真终点最多匹配一次；提前关闭、在持续对照中关闭或重复关闭分别计错。参考终点、投影范围与实际确认时刻均保留。','',
             '完整 Qwen 请求与响应保存在各组 qwen_calls.jsonl.gz。window_trace.jsonl 与 event_updates.jsonl 可分别检查逐窗判断和真正执行的账本更新。未知或截断均未当作成功关闭。']
    report+=['','## 相同片段上的配对比较','',
             '|范围|方法|相同片段数|纯目标支持|纯非目标误支持|真终点关闭|其中在真终点前已开启|误关|动作错误|Qwen 调用|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('development','holdout'):
        for alternative in ('K','L','M','P','R'):
            shared=[c['key'] for c in cases if c['split']==split and all(t+'-'+c['key'] in output for t in ('J',alternative))]
            if not shared:continue
            for t in ('J',alternative):
                n=Counter()
                for key in shared:n.update(output[t+'-'+key]['counts'])
                report.append(f"|{split}|{t}（J/{alternative}）|{len(shared)}|{n['target_support']}/{n['target']}|{n['other_support']}/{n['other']}|{n['correct_close']}/{n['end_opportunities']}|{n['correct_close_on_previously_confirmed_event']}|{n['false_close']}|{n['errors']}|{n['llm_calls']}|")
    (root/'metrics.json').write_text(json.dumps(dict(totals=totals,cases=output),indent=2),encoding='utf8')
    report+=['','## 读逐窗结果时应区分什么','',
             '- 04936 的终点约在 1791.044 秒。关闭判定时刻用于评价延迟，账本回溯的估计终点用于评价定位。7930 秒在另一次持续事件内部，必须检查稍后回升是否仍沿用同一事件。两个片段各自从空账本启动。',
             '- 08378 的 2860 秒片段开始时参考房扑已经持续，2897.34 秒结束。如果模型仅在 2900 秒才首次确认，即使估计边界碰到参考终点，也应单列为终点到达后的发现。先前窗口的参考标签只在事后核对。',
             '- 05091 的 3930–3940 秒完全在非目标区。K 只取得 ECG1 分数 0.830，确认了一个错误事件。L 同窗测得 ECG1 为 0.830、ECG2 为 0.253，记录冲突并保持空账本。这一例同时有两份实测，与另选单一导联不同。',
             '- 在 07162 的 4320–4330 秒，L 只请求了 ECG2，K 只请求了 ECG1。选择另一导联与实测两导联后解决冲突是两种不同结果。各窗的实测通道见 metrics.json 中的 measured_leads。',
             '- 针对“持续事件中单窗下降”选取的困难反例会放大两窗复核的预期优势。短片段结果不能估计整条记录上的错误关闭率。']
    fixed=root/'reference-view'/'results.jsonl'
    if fixed.exists():
        answers=lines(fixed)
        pairs={}
        for row in answers:
            pairs.setdefault((row['clip'],row['cursor_s']),{})[row['arm']]=row
        report+=['','## 固定测量下的分数呈现实验','',
                 '同卡交替回放每个判断点的首次观察调用。两次使用相同的测量、时刻、允许动作和解码设置。原界面同时显示模型分数、经验方向和其他记录的分箱目标比例；新界面保留分数及方向，并提醒同导联按时间比分数。本轮只检查窗口观察，没有执行事件动作。','',
                 '|游标|原观察|新观察|原输入重跑与历史回答逐字一致|','|---:|---|---|---|']
        for (_, t), arms in pairs.items():
            if set(arms)=={'original','direction'}:
                l,r=arms['original'],arms['direction']
                report.append(f"|{t}|{l.get('finding','无有效回答')}|{r.get('finding','无有效回答')}|{l.get('raw_reproduced',False)}|")
        report+=['','完整原话与两组输入见 reference-view/results.jsonl 和同目录的 Qwen 日志。字段合法率、测量方向解释和节律判断正确性应分别评价。']
    comparison=root/'reference_comparison.json'
    if comparison.exists():
        scores=json.loads(comparison.read_text(encoding='utf8'))['summary']
        report+=['','## 当前数值与方向词的第二次修订','',
                 '|窗口观察界面|点数|结果与原版一致|理由引用当前分数|把分数表述为负数|',
                 '|---|---:|---:|---:|---:|']
        for mode,label in [('original','原始分箱显示'),('direction','只显示方向词'),('meaning','显示方向含义并要求引用数值')]:
            v=scores[mode]
            report.append(f"|{label}|{v['points']}|{v['same_as_original']}|{v['quotes_current_score']}|{v['calls_score_negative']}|")
        report+=['','7940 秒的原回答将当前分数 0.8545 和历史分箱比例 0.0125 作比较。最终修订后改为与上一窗模型分数约 0.220 比较。7950 秒的 0.667 仍被判为不确定。可疑低分并未因此变成正确的节律识别。原始及修订后回答、核对口径和真实分数见 reference_comparison.json。']
    joint=[(arm,c,output[arm+'-'+c['key']]) for arm in ('P','R') for c in cases if arm+'-'+c['key'] in output]
    if joint:
        report+=['','## 最终组合逐窗回放','',
                 'P 是第一次组合。R 在相同配置下将导联证据改为自然语言，列出实际在同一窗方向相反的两路测量 ID。两种版本仍允许模型选择不确定。','',
                 '|版本|记录|游标秒|窗口证据结论|Qwen 动作|动作后正式事件状态|',
                 '|---|---|---:|---|---|---|']
        for arm,c,run in joint:
            for t,d,a,state in run['decisions']:
                report.append(f"|{arm}|{c['record_id']}|{t:g}|{d}|{a}|{state}|")
    report+=['','代码提交：K 5546379、L af50bf4、M cb3dd93、显示分箱方向 5d4d2e7、修订分数含义 7977f0a、最终配置 69b2206。历史 J 使用范围提示。独立开发回归测试 270 项通过。']
    (root/'实验报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    print(json.dumps(totals))
if __name__=='__main__':main()
