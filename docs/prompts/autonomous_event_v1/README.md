# 流式 ECG 自主事件复核：Qwen 提示词与从零操作手册

适用设计为 AFIB/AFL 合并事件的自主 OPEN/CLOSE 与预约复核。模型为文本 Qwen，输入来自工具、记忆和运行时快照。这里区分三个对象：英文提示词是模型指令，快照是本次可见事实，例题是离线教学材料。英文和中文是同一版本的两种表述；一次运行选一种语言。

[机器可读的中英文提示词](qwen_event_prompts_v1.json) · [按账本状态选用的最终工具定义](final_tools_v1.json) · [研究方案](../../ECG_AGENT_WINDOW_EVENT_DECISION_PLAN.md)

## 1. 当班的第一分钟怎么做

你会分批收到一条长 ECG 记录。假设第一批是记录的前十秒。你要编出一份可追溯的事件时间表，包括哪段支持房颤或房扑、什么时候真的发出了报警、哪些区间仍有疑问。你看不到未来、人工节律标注或整条记录的最终答案。ECG1、ECG2 是通道名，按运行时给出的名称使用，不推断其解剖导联名称。

1. **看当前任务。** 读 `assessment_window_s`（本次要评价的区间）、`cursor_s`（信号已到达的最晚时刻）、`retained_window_s`（现在仍能取原波形的范围），以及有没有开放事件。时间使用从记录开始计算的秒，`[a,b)` 包括 a，不包括 b。第一次没有历史，分数也只有在已测量时才存在。
2. **挑一项能解决疑问的工具。** 比如想知道当前窗是否像两个目标亚型，可在 `[0,10)` 的一个可用导联上请求 ECGFounder。看清实际返回的区间、导联、成功状态及 AF、AFL 两个分数。若问题在于刚刚有没有发生变化，可查以前的支持证据，再测变化附近。选几项由你判断。
3. **写出本窗观察。** 可用证据支持目标、反对目标，或目前不确定。“低质量”“没有测”“两路可用结果相反”“证据可用但不足”要分别写清。缺失不填零，不将反对目标扩大为“整张心电图正常”。明确判断应引用与本窗相交、能解释当前节律的成功测量；单靠旧事件或质量提示无法完成这一步。
4. **决定要不要动事件。** 即使首窗就有充分证据，也可以正式开事件。已有事件时，当前偏阴性与“找到了结束边界”是两项判断。没有边界依据可以挂起疑问。账本只记录已执行的正式变更；每一项观察都带着本来的记录时间。
5. **决定何时回来。** 下一次常规巡视也要安排时间。有具体疑问待追办时，写明为了什么等待、届时哪项新信息能减少疑问。等待期间不调用工具；下一次醒来可以依法回看仍保留的信号。后续得到的信息只能用于此时提交的判断和修订。

## 2. 四个动作与三个时刻

| 动作 | 可以在何时用 | 它实际改变什么 |
|---|---|---|
| `OPEN` | 当前无开放事件，并有足够的当前正向证据 | 正式登记事件，系统生成身份。可直接在首窗执行，不要求连续两窗。若第一眼就看到目标节律，只能知道首次观察时已存在 |
| `RECORD_ONLY` | 空闲或有开放事件，本次只需记观察，无待追办事件问题 | 记录支持、反向或不确定的本窗观察；事件身份保持原状。已有待办时说明如何解决或为何撤回 |
| `DEFER_DECISION` | 空闲或有开放事件，有一个具体问题值得稍后处理 | 建立或延续待办，安排复核。可以已经知道本窗偏阴性，同时仍拿不准终点。也可以先查记忆当场解决疑问，避免无用等待 |
| `CLOSE` | 当前存在开放事件，有证据支持终止且边界可定位或夹定 | 完成该事件的结束登记，保存边界及依据。事件身份由运行时绑定。右截断、末次检出心搏、数据中断都不是独立的终止观察 |

**三个时刻举例。** 82 秒可以是模型从局部测量估计出的节律转变，90 秒才是获得材料并实际提交判断的游标，105 秒可以是模型预约的下次巡视。若数据每十秒交付，请求 105 秒可能在 110 秒实际唤醒。这三个数字回答不同问题。时间请求不应依赖未来将出现的分数。持续开放但未测的空档保留为未知。

## 3. 模型怎样取得证据

以下名称来自基线四组实验的复核入口。实际运行只调用当前阶段下发的工具，并以其参数 schema 和返回结果为准。首版没有预设必须先跑两个导联、必须测 RR 或必须用完预算的顺序。

| 想核对什么 | 可调用工具 | 怎样读结果 |
|---|---|---|
| 这个导联的十秒波形是否像目标亚型 | `score_ecgfounder` | 一次请求一个导联的完整十秒；AF 和 AFL 分数分别读。任一亚型可提供正向证据；分数不是临床概率。检查已返回的测量再决定是否补查另一导联 |
| 可用性、掉线或噪声是否影响这次解释 | `assess_quality` | 检查状态及原因，作为复核线索。必要时看另一导联。质量提示不会自动将成功的测量变成阴性或使整段作废 |
| 检出的心搏与节律变异是否可信 | `detect_beats`、`get_rate_rhythm`、`get_feature_profile` | 核对峰数、原始 RR、清洗信息、局部趋势。RR-CV 为 RR 的标准差除以均值。越低表示检出的间隔越规则；房扑仍是可能解释。与检峰共享来源的指标不能累计成多次独立确认 |
| 检峰或边界附近的形态是否需要复核 | `detect_beats_xqrs_proxy`、`get_beat_timeline_proxy`、`get_p_wave_proxy`、`get_qrs_proxy` | 分别提供补充检峰、对齐的心搏时间线、P 波候选、QRS 代理量。可检查转变两侧；代理返回了一个峰，不能证明该峰就是节律起止。无法定位时保留范围或待办 |
| 以前有没有实测过；这个事件有没有真正执行过关闭 | `retrieve_memory` | 查测量用 `scope="evidence"`，按绝对时间、导联或工具筛选；查事件用 `scope="events"`，必要时用已知事件 ID。空结果表示没有查到可见记录。两个导联、同窗重复调用或相邻重叠窗均需考虑证据相关性 |
| 哪份细节值得下次专门回看 | `write_memory` | 使用本轮成功且可写的测量返回的准确 `evidence_id`。原始账本自动保存，写记忆标记供后续复核的细节。每次检索或写入占一个交互轮次 |
| 取证够了或再测无益 | `finish_measurements` | 结束取证，转入单独的提交阶段；它不会开启或关闭任何事件。最终只提供拟议新版的 `submit_event_decision` |

**三类 ID。** 测量工具返回的 `ecg:…` 是持久证据 ID，写记忆时复制实际值。最终清单的 `E1` 等只是本次复核的测量别名，可写入本次决策的 `evidence_ids`。下一窗的 `E1` 可能指另一项测量，历史引用用本次真实展示且准许引用的持久 ID。事件 ID 属于账本；可以按已知事件 ID 检索事件，提交事件动作时由运行时绑定，模型无需填写。动作报错并未产生新的测量事实。

## 4. 怎样把提示发给 Qwen

机器文件的每个片段同时含 `en`、`zh` 两种文字，拼接时使用空行。每次实际调用仅组合以下适用片段，完整手册和教学例题不进入模型上下文。

| 运行条件 | 本轮 system 消息组成 |
|---|---|
| 第一次测量，空账本，自主调度 | `core + no_open + acquire + autonomous_time` |
| 同一次的最终提交 | `core + no_open + final + autonomous_time` |
| 有开放事件且因待办醒来，开始取证 | `core + open + resume + acquire + autonomous_time`；若仍空闲则改用 `no_open` |
| 固定时间对照 | 仅将 `autonomous_time` 换成 `fixed_time`，使用调度器提供的下一时刻 |
| 已到真实流末 | 仅用 `stream_end` 替换时间片段；窗内是否发生转变仍需测量支持 |
| 最终提交校验失败 | 按最新账本状态重新渲染当前 final 组合，附加 `retry` 和实际字段错误；仅给一次纠错机会 |

运行时另传一份结构化快照，其中包括本窗、游标、保留范围、可用通道、真实账本状态、最近执行变更、可见历史、待办和剩余预算。自主组提供合法预约范围，固定组提供下一预约时刻，流末显式标记。取证阶段附当轮真实工具定义；最终阶段提供本轮可引用证据目录及必要实测细节。应替换当前阶段的 system 消息，避免把同一套规则逐轮叠加。临床参考标注、未来波形、其它分组结果一律留在评价侧。

**运行版本。** 固定复核与自主预约配置已接入 v1.3。程序按事件状态和待办状态生成最终工具定义。有待办时增加 `pending_resolution`。运行时发送英文片段，中译用于核对。每次调用实际看到的工具、消息和原始回复均可在运行日志中查看。[运行配置和结果文件](../../ECG_AGENT_REACT_EVENT_MECHANISM.md)

## 5. 全部提示词片段（中英文对应）

### `core` · 共同任务与证据

**发给 Qwen 的英文**

```text
Your job is to track atrial fibrillation (AF or AFIB) and atrial flutter (AFL) on a streaming ECG. Either rhythm counts. If AF changes directly to AFL or vice versa, keep tracking the same episode.

At each review, answer three questions in order.
1. What does this segment show? Use SUPPORTS_TARGET when a measurement supports AF or AFL, AGAINST_TARGET when a measurement points away from both, and UNCERTAIN when you cannot tell. Missing data tells you nothing about the rhythm.
2. Does the event ledger need an update? Choose an action allowed in its current state. To report that a rhythm is continuing, cite what still supports it.
3. When should you look again? If you need to wait, name the question that remains open.

Use only measurements available by cursor_s. All times are seconds from the start of the recording. Check the lead and interval before citing a result. Leave unobserved time unknown.
```

**中文等义稿**

你的任务是追踪心电图中的房颤（AF 或 AFIB）和房扑（AFL）。出现其中任何一种，都属于目标心律。房颤直接转成房扑，或房扑直接转成房颤，继续追踪同一次事件。

每次复核，按顺序回答三个问题。
1. 这一段心电图说明什么？有测量支持房颤或房扑，选 SUPPORTS_TARGET；有测量提示两者都不符合，选 AGAINST_TARGET；看不清或证据不够，选 UNCERTAIN。没有数据就不能判断心律。
2. 事件账本现在要更新吗？先看账本当前的状态，再从允许的动作里选。若认为事件仍在持续，要说出这一段还有什么证据支持它。
3. 什么时候再看？如果决定等一等，说明还有哪个问题没弄清楚。

只能使用截至当前游标（cursor_s）已经取得的测量。所有时间都从记录开头算，以秒为单位。引用结果前，核对导联和测量区间。没观察到的时段保留为未知。

### `no_open` · 账本中暂无开放事件

**发给 Qwen 的英文**

```text
There is no open event. Ask whether the current interval contains evidence to start tracking an episode. OPEN formally registers it, even on the first review. For a measured onset give a localized point or an evidence-bounded range. If already present at first observation, report the first supported interval and mark onset unobserved. RECORD_ONLY logs an observation without an outstanding question. DEFER_DECISION leaves a specific onset or rhythm question pending. After a previous closure, assess this interval afresh.
```

**中文等义稿**

当前没有开放事件。判断本区间有无依据开始登记一次事件。OPEN 正式登记事件，首窗即可执行。实测到起点时给出有依据的定位点或范围。若首次观察时目标已存在，报告最早有支持的区间，标记起点未观察到。RECORD_ONLY 只记录观察，没有要追办的问题。DEFER_DECISION 挂起一个具体的起点或节律疑问。此前事件关闭后，要重新评估当前区间。

### `open` · 账本中已有开放事件

**发给 Qwen 的英文**

```text
One event is open. RECORD_ONLY records current evidence and leaves it open. DEFER_DECISION leaves a specific unresolved question for later. Use CLOSE when evidence supports an end and locates or brackets the change. Identify what changed, on which lead and over what interval. A last detected beat, window edge or score drop alone does not locate an end. Keeping the event open does not label gaps positive.
```

**中文等义稿**

当前有一个开放事件。RECORD_ONLY 记下本次观察并保持事件开放。DEFER_DECISION 留下一个需要以后处理的具体疑问。当前证据支持事件结束且能定位或夹定转变范围时，可选 CLOSE。说明是什么变化、出现在哪个导联和时段。仅有末次检出心搏、窗边缘或分数下降无法定位终点。保持事件身份不等于把间隔记成阳性。

### `acquire` · 取证阶段

**发给 Qwen 的英文**

```text
Call one offered function per turn. Pick the check that could change your decision.
- For AF/AFL scores, call score_ecgfounder on one available lead over ten retained seconds. A high AFL score can support the target even with a low AF score, and vice versa. Scores are uncalibrated; no cutoff dictates an action.
- If a result looks unreliable, consider assess_quality or the other lead. A quality warning is a reason to check. For RR timing, use get_rate_rhythm: lower RR-CV means more regular intervals. Check beat counts and detector quality. Regular RR can occur in AFL.
- For a possible change, compare measurements on either side using arrived signal. get_beat_timeline_proxy gives beat times and morphology clues. Measurements sharing peaks or samples count as related evidence. A proxy cannot diagnose the rhythm on its own.

Use retrieve_memory for a previous measurement or event. Past opinions can be wrong; interpret past measurements at their own times. Use write_memory with the exact evidence_id of a useful successful measurement if its detail will matter later. Both memory tools use a turn without spending a measurement call. Use finish_measurements when ready or when another check would not help. In cached mode score_ecgfounder is available only for the displayed current window. Other ECG tools use their own interval constraints. Use only the offered tools.
```

**中文等义稿**

每轮调用一个可用工具。先想清楚哪项检查可能改变你的判断。
- 想看房颤和房扑分数，就用 score_ecgfounder 测一个可用导联已保留的完整十秒。房扑分数高、房颤分数低，仍可支持目标；反过来也一样。分数未经校准，没有必须触发某个动作的阈值。
- 怀疑结果不可靠，可用 assess_quality 查质量，或补查另一导联。质量告警提示复核。想看心搏间隔，用 get_rate_rhythm。RR-CV 越低，间隔越规则；先核对检出的心搏数和检峰质量。房扑也可能有规则的 RR。
- 怀疑节律发生变化，比较变化前后已经到达的信号。get_beat_timeline_proxy 提供心搏时刻和形态线索。共用心搏或波形采样点的结果相互关联；代理特征不能单独确诊心律。

要查早先的测量或事件，用 retrieve_memory。过去的判断可能有误，历史测量只解释它自己的时段。有一项成功测量值得以后再看时，可用 write_memory 标记，填写工具实际返回的 evidence_id。两种记忆操作各占一轮交互，不占测量次数。证据够了，或再查也解决不了问题，就用 finish_measurements 结束取证。 缓存模式下 ECGFounder 只能测当前原窗，其余工具按各自区间约束工作。只调用当前提供的工具。

### `final` · 提交阶段

**发给 Qwen 的英文**

```text
Call submit_event_decision. Judge assessment_window_s and choose an allowed event action. For each evidence_ids entry, supply one evidence_attributions entry with its ID and role. SUPPORTS_TARGET means the measurement favors AF or AFL. AGAINST_TARGET means it weighs against both. Use CONTEXT_ONLY for quality, failed checks, or measurements with no clear direction. Low AF alone cannot rule out AFL; regular RR can occur in AFL. Shared samples do not provide independent votes.

Ground a directional assessment in at least one matching measurement from this interval. If opposing roles overlap within this interval, use UNCERTAIN with cause conflicting and identify the disagreement. Different directions at successive times can describe a change. For other uncertainty, give the cause. Subtype or boundary uncertainty alone need not obscure the rhythm.

Use this review's E-numbers or historical IDs visible here. If nothing was measured, submit both lists empty. Give a brief reason. For a localized boundary cite a measured change within a local measurement; for a bracket cite a local measurement or evidence on both sides. Leave midpoint projection to the evaluator. For DEFER_DECISION state the open question and needed evidence. Set next_review_s according to the scheduling instructions.
```

**中文等义稿**

调用 submit_event_decision。判断 assessment_window_s 内的心律，选择当前允许的事件动作。evidence_ids 每引用一项测量，都在 evidence_attributions 中填写它的 ID 和一个角色。测量支持房颤或房扑，填 SUPPORTS_TARGET；提示两者均不符合，填 AGAINST_TARGET；质量检查、失败测量、方向不清的测量，填 CONTEXT_ONLY。单凭 AF 低分无法排除房扑，房扑也可能有规则 RR。共用同一段波形的测量不能重复投票。

给整窗下方向性结论，至少要有一项覆盖本窗且方向一致的测量。同一段时间内正反证据重叠，选 UNCERTAIN，原因填 conflicting，指出哪两项相互矛盾。若前一段支持、后一段反对，则可能观察到了变化。其他不确定情况写明原因。亚型或边界难以确定时，仍可报告有依据的心律方向。

本轮测量用清单中的 E 编号，历史测量用本轮可见的 ID。什么都没测到时，两张引用列表都填空。简要说明动作理由。报告精确边界须引用局部测量内的实际变化；报告范围须引用覆盖该范围的局部测量或分别约束两侧的证据。评价器负责取中点。选择 DEFER_DECISION 时，留下待解决的问题及需要的证据；按调度要求填写 next_review_s。

### `resume` · 处理上次的待办

**发给 Qwen 的英文**

```text
You are resuming the supplied pending question. Read the last executed ledger change, original question and newly available interval. Retrieve or measure what can discriminate the alternatives. Say whether the question is resolved, withdrawn or still pending. If waiting again, link to the same question and say what is still missing. Set pending_resolution to resolved, withdrawn, or still_pending. Use still_pending when deferring the same question again.
```

**中文等义稿**

你正在处理上次挂起的问题。先读账本最后一次已执行变更、最初疑问和新到达区间，再检索或测量有助于区分几种解释的证据。说明问题已解决、已撤回或仍待处理。若再次延迟，沿用原待办并指出还缺什么。 用 pending_resolution 填写 resolved、withdrawn 或 still_pending。再次延迟同一个问题时填写 still_pending。

### `autonomous_time` · Qwen 自主预约

**发给 Qwen 的英文**

```text
Choose the next review in absolute seconds after cursor_s, within the scheduling bounds shown. Choose a time that could add useful evidence while accounting for a short event you might miss and data that may leave the buffer. Until the next wake, signal is buffered without measurements. A request between arrivals wakes at the first batch covering it.
```

**中文等义稿**

自主选择晚于 cursor_s、且在本次允许范围内的绝对记录秒数作为下次复核时间。权衡届时能否获得新信息、是否可能错过短事件，以及所需信号是否会移出缓冲区。等待期间只缓冲信号，不做测量。请求时刻位于两次交付之间时，在首个覆盖该时刻的批次唤醒。

### `fixed_time` · 固定唤醒对照

**发给 Qwen 的英文**

```text
The control arm wakes at the next scheduled batch. Use the supplied scheduled_review_s as next_review_s. You may defer a specific question or act now using the same evidence rules.
```

**中文等义稿**

固定时间对照组在下一个预定批次唤醒。将给出的 scheduled_review_s 填入 next_review_s。你仍可挂起具体问题或现在作决定，证据要求相同。

### `stream_end` · 已确认流末

**发给 Qwen 的英文**

```text
The runtime reports the stream has ended. Set next_review_s to null. Report any unresolved question as unresolved at stream end. An open event without measured termination is right-censored; the stop of recording supplies no rhythm transition.
```

**中文等义稿**

运行时已报告数据流结束。next_review_s 填 null，尚未解决的问题标为流末未决。开放事件缺乏实际终止证据时记为右截断，录制停止本身不提供节律转换证据。

### `retry` · 动作被拒绝后的单次纠错

**发给 Qwen 的英文**

```text
The last submission was rejected and no event change or appointment was executed. Read the field error, current ledger state, allowed actions and citation list. Correct only what needs correction and call submit_event_decision once more. A format correction supplies no new ECG evidence.
```

**中文等义稿**

上一份提交被拒绝，未执行事件变更或预约。读取报错字段、账本当前状态、合法动作和可用引用，仅修正必要内容，再调用一次 submit_event_decision。修正格式不会产生新的 ECG 证据。

## 6. 最终工具参数约定与输入示范

最终阶段根据有无开放事件和待办，从 final_tools_v1.json 的四份定义中选一份。取证阶段使用实际工具注册表。决策时刻和实际唤醒时刻由运行时记录。

函数 schema 检查字段、值类型、允许的动作与边界表示。运行时还需按下表校验字段间关系、范围顺序、可见证据、边界是否有依据、预约是否合法，并再次核对事件版本。只通过 JSON 结构校验不能改变账本。固定证据组禁用补测时要向模型直接提供所有允许使用的证据。

| 字段 | 拟议值与条件 |
|---|---|
| `assessment` | `SUPPORTS_TARGET`、`AGAINST_TARGET` 或 `UNCERTAIN`，对应快照中的 assessment_window_s。历史边界可能位于别的区间，应单独给出引用 |
| `uncertainty_cause` | UNCERTAIN 时填 `missing`（未测/缺失）、`unusable`（测量不可靠）、`conflicting`（可用观测指向相反）、`insufficient`（已有信息区分力不足）中主要一项；其余填 null。可在一句理由里补充从属原因 |
| `event_action` | 只使用账本当前状态允许的动作。OPEN 要求当前正向依据。CLOSE 需要反对持续的节律依据，以及局部变化可定位或可夹定的证据。仅见局部分歧时可以暂缓 |
| `boundary` | OPEN/CLOSE 可用 `{"kind":"localized","time_s":t}` 或 `{"kind":"bracketed","range_s":[lo,hi]}`。点估计须在成功的局部测量内且对应实测变化。范围的两端表示不确定区间的下、上界，需有两侧证据支持，均不得晚于当前游标。只有 OPEN 还可用 `{"kind":"onset_unobserved","first_supported_interval_s":[a,b]}`；其余动作填 null |
| `evidence_ids`、`reason` | 无重复的可见证据引用；理由用一两句指明哪段、哪个观测、为何与动作相符。引用了历史也要分得清时间。方向性判断不能只引用质量检查或历史动作 |
| `evidence_attributions` | 为每个可见的引用分配一个角色。账本按真实测量时段累计，质量检查只提供背景。 |
| `pending` | DEFER_DECISION 时填 `{"question":"…","expected_evidence":"…"}`；其余填 null。延续同一问题时由运行时关联当前待办。醒来时，在 reason 中说明解决、撤回或仍未解决；流末不能预约时保留未决状态 |
| `next_review_s` | 自主组为允许范围内、严格晚于游标的绝对秒数；固定组使用提供的预约；已确认流末填 null。不把“实际醒来时间”“期望看多少秒”“估计边界”混入这个字段 |

示范输入是一个**空账本、空历史、可自主巡视的首窗**。下例的参数名是新快照设计，时间范围是示例配置。运行时还需附真正可用的工具 schema。最终消息里才给当次证据目录与实际结果。

```json
{
  "phase": "acquire",
  "assessment_window_s": [0, 10],
  "cursor_s": 10,
  "retained_window_s": [0, 10],
  "available_leads": ["ECG1", "ECG2"],
  "open_event": null,
  "last_executed_change": null,
  "pending_question": null,
  "recent_measurements": [],
  "remaining_measurements": 4,
  "scheduling_mode": "autonomous",
  "max_next_review_s": 50,
  "stream_ended": false
}
```

这时一种合理的第一步是调用 `score_ecgfounder`，参数为 `{"lead":"ECG1","start_s":0,"end_s":10}`。如果对备用导联或质量有疑问，先读这次真正取得的结果，再决定是否继续。以后想核对最初的实测分数，例如在 50 秒游标下，可取证时调用 `retrieve_memory`，参数为 `{"scope":"evidence","start_s":0,"end_s":10,"lead":"ECG1","tool":"score_ecgfounder","limit":1}`。返回的是早先已保存的测量，不是对当前波形的重测。

## 7. 教学例题与逐步回放

这些是合成教学题，数字是不同证据的作用示范，均非实验实测。每题只展示一种合法选择；实际提交需要依据当时真实返回的测量。最终函数名均为 `submit_event_decision`，代码块仅为其 arguments。勿将这些示例的时间或标签放入正式模型提示。

### 1. 第一窗就可以正式开事件，起点不可见

游标 10 秒，当前评估区间 [0, 10)。窗内 ECG1 的 AFL 方向性支持明显，AF 分数低（E1）；ECG2 的 AFL 方向相同（E2）；质量检查未提示影响这次解释的明显问题（E3）。开头没有更早的波形。这里示范当模型认为已有充分正向依据时怎样登记。

```json
{
  "assessment": "SUPPORTS_TARGET",
  "uncertainty_cause": null,
  "event_action": "OPEN",
  "boundary": {
    "kind": "onset_unobserved",
    "first_supported_interval_s": [
      0,
      10
    ]
  },
  "evidence_ids": [
    "E1",
    "E2",
    "E3"
  ],
  "evidence_attributions": [{"evidence_id": "E1", "role": "SUPPORTS_TARGET"}, {"evidence_id": "E2", "role": "SUPPORTS_TARGET"}, {"evidence_id": "E3", "role": "CONTEXT_ONLY"}],
  "reason": "Both leads support AFL in this interval. The first observed interval already contains target evidence; onset was not observed.",
  "pending": null,
  "next_review_s": 30
}
```

### 2. 本窗偏反向，但暂缓判断已有事件何时结束

游标 30 秒，当前评估区间 [20, 30)。醒来后，两路当前分数均较此前的支持区间下降，测量成功（本轮 E1、E2）；本窗局部实测也显示心搏模式发生变化（E3），但是否为稳定转变仍不清楚。此前预约期间若未测过某段，就保留那个空档。

```json
{
  "assessment": "AGAINST_TARGET",
  "uncertainty_cause": null,
  "event_action": "DEFER_DECISION",
  "boundary": null,
  "evidence_ids": [
    "E1",
    "E2",
    "E3"
  ],
  "evidence_attributions": [{"evidence_id": "E1", "role": "AGAINST_TARGET"}, {"evidence_id": "E2", "role": "AGAINST_TARGET"}, {"evidence_id": "E3", "role": "CONTEXT_ONLY"}],
  "reason": "Current measurements weigh against continuation. They do not yet distinguish a transient dip from a sustained change.",
  "pending": {
    "question": "Has the previously tracked rhythm ended?",
    "expected_evidence": "Compare the next arrived segment with this change and inspect the elapsed gap if retained."
  },
  "next_review_s": 50
}
```

### 3. 按约定回来，撤回结束疑问

游标 50 秒，当前评估区间 [40, 50)。40 秒到达的批次只缓存，没有在 40 秒生成诊断。50 秒唤醒时原事件仍开放。新窗实测再次支持目标（本轮 E1、E2）。若需要早先细节，取证阶段先检索以前的记录。当前支持恢复，本例选择撤回结束疑问，中间未判定区间继续留在账本。

```json
{
  "assessment": "SUPPORTS_TARGET",
  "uncertainty_cause": null,
  "event_action": "RECORD_ONLY",
  "boundary": null,
  "evidence_ids": [
    "E1",
    "E2"
  ],
  "evidence_attributions": [{"evidence_id": "E1", "role": "SUPPORTS_TARGET"}, {"evidence_id": "E2", "role": "SUPPORTS_TARGET"}],
  "reason": "Target evidence is present again in the new measurements. Withdraw the pending end question for now; the unassessed gap remains unknown.",
  "pending": null,
  "next_review_s": 70,
  "pending_resolution": "withdrawn"
}
```

### 4. 可以夹定真实测到的变化；不制造精确终点

游标 90 秒，当前评估区间 [80, 90)。另一条独立的虚构场景。已开放事件在 82 秒之前仍有可靠支持（E1，测量覆盖 78–82 秒）；86–90 秒的可用节律测量倾向相反状态（E2）。跨越两侧的局部时间线（E3）支持发生了稳定变化，但不能把转变精确到某个心搏。此例假定这些观测足以确认结束。

```json
{
  "assessment": "AGAINST_TARGET",
  "uncertainty_cause": null,
  "event_action": "CLOSE",
  "boundary": {
    "kind": "bracketed",
    "range_s": [
      82,
      86
    ]
  },
  "evidence_ids": [
    "E1",
    "E2",
    "E3"
  ],
  "evidence_attributions": [{"evidence_id": "E1", "role": "SUPPORTS_TARGET"}, {"evidence_id": "E2", "role": "AGAINST_TARGET"}, {"evidence_id": "E3", "role": "CONTEXT_ONLY"}],
  "reason": "Target evidence persists to the earlier interval and is opposed in the later interval. The local timeline supports a transition between them, without a single localized beat.",
  "pending": null,
  "next_review_s": 110
}
```

### 5. 没有任何成功测量时，明确写不确定

游标 10 秒，当前评估区间 [0, 10)。独立虚构场景。首窗暂时无法测量，没有任何可引用的测量返回。若确实得到过失败记录且该 ID 在允许清单内，可引用它解释失败。不能给这个窗口填正常或直接开事件。

```json
{
  "assessment": "UNCERTAIN",
  "uncertainty_cause": "missing",
  "event_action": "DEFER_DECISION",
  "boundary": null,
  "evidence_ids": [],
  "evidence_attributions": [],
  "reason": "No rhythm measurement is available for this interval.",
  "pending": {
    "question": "Is the target rhythm present?",
    "expected_evidence": "An arrived, measurable ten-second interval on an available lead."
  },
  "next_review_s": 20
}
```

### 6. 已到流末，证据不确定，也没有可执行的后续复核

独立示例。评估区间为 [90,100)，账本仍有开放事件。最后一批两路均有严重干扰，E1、E2 是本轮两路的质量结果，无法提供节律依据。此前没有待办。本轮只登记未知观察，运行时在流末对仍开放的事件作右截断，无须要求模型替它生成终点。

```json
{
  "assessment": "UNCERTAIN",
  "uncertainty_cause": "unusable",
  "event_action": "RECORD_ONLY",
  "boundary": null,
  "evidence_ids": ["E1", "E2"],
  "evidence_attributions": [{"evidence_id": "E1", "role": "CONTEXT_ONLY"}, {"evidence_id": "E2", "role": "CONTEXT_ONLY"}],
  "reason": "Neither lead is usable here; no rhythm endpoint was measured before the recording stopped.",
  "pending": null,
  "next_review_s": null
}
```

## 8. 接入前用这些问题自检

| 给操作者或集成程序的题目 | 能独立完成任务的表现 |
|---|---|
| 空账本，AF 低而 AFL 明显支持，RR 比之前规则 | 知道房扑属于目标。说明何种当窗测量支持或反对合并目标，不以两个分数不同单独宣告冲突 |
| 上一窗刚刚 CLOSE，新窗出现明确正向证据 | 先读最新账本。可以提出新的 OPEN 或保留疑问；不会重新关闭旧对象或填入旧 event_id |
| 已开放，导联冲突，但另一份已到达证据现在就能取 | 能解释为何查哪个已到达区间、哪一导联或哪条历史。查后可当场作决定；无需先强制延迟 |
| 没有一个需要追办的疑问，想降低普通巡视频率 | 使用 RECORD_ONLY 与合适的下次巡视时刻。不能把所有间隔都计作 DEFER 或正确阴性 |
| 等了两次仍缺证据 | 继续原来的待办或说明为何撤回；写出本次新增观测和仍缺的信息，不能编造稳定转变 |
| 窗口右端点恰好有时间戳，最后一个 R 峰靠近它 | 两者只能提示在哪里复查。能指出转变两侧的实际变化才可定位，否则夹定区间或留待再查 |
| 收到“引用非法”的拒绝消息 | 知道变更未执行。读取最新状态和允许引用的清单，纠错一次；保留失败原因。ACTION_ERROR 不是一次自主延迟 |
| 流结束但有开放事件或待办 | next_review_s 为 null，保留右截断和未决项。不强行 CLOSE，也不预约已结束记录的未来时间 |

交付验收分两部分。结构检查验证片段拼接、状态许可、预约时间、示例字段和引用一致性。模型验证须在新版运行时实际执行，另报正确开关、暂缓后解决率、短事件漏检、延迟、证据引用质量和成本；结构合法并不能替代这些指标。
