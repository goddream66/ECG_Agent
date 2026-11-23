"""One-step counterfactual on the last rejected v1.4 submission, with its actual prior state."""
import gzip
import json
from pathlib import Path
from cardioagent.ecg_agent.harness.autonomous_decision import validate
from cardioagent.ecg_agent.harness.react import CLOSURE_EVIDENCE_TOOLS
from cardioagent.ecg_agent.llm.providers import parse_qwen_tool_reply


def replay(source):
    def rows(name):
        return [json.loads(s) for s in (source/name).read_bytes().splitlines() if s]
    bad={r['cursor_s'] for r in rows('window_trace.jsonl') if r['decision']=='ACTION_ERROR'}
    ledger={r['evidence_id']:r for r in rows('evidence.jsonl')}
    result={}
    with gzip.open(source/'qwen_calls.jsonl.gz','rt') as stream:
        for line in stream:
            request=json.loads(line)
            t=request['cursor_s']
            if t not in bad or [v['function']['name'] for v in request['tools']]!=['submit_event_decision']:
                continue
            messages=request['messages']
            state=json.loads(messages[1]['content'])
            # The final table and tool conversation contain this model's actual visible IDs.
            tables=[json.loads(m['content']) for m in messages if m['role']=='user' and 'measurement_ids' in m['content'] and m['content'].lstrip().startswith('{')]
            aliases={i['id']:i['evidence_id'] for i in tables[-1]['measurement_ids']}
            try:
                fn=parse_qwen_tool_reply(request['raw'])['tool_calls'][0]['function']
                if fn['name']!='submit_event_decision':
                    raise ValueError('Wrong tool')
                args=json.loads(fn['arguments'])
                args['next_review_s']=None if state['stream_end'] else t+10
                visible=set(aliases.values())|{key for key in args['evidence_ids'] if key in ledger and key in messages[1]['content']}
                for m in messages:
                    if m['role']=='tool':
                        visible.update(k for k in args['evidence_ids'] if k in ledger and k in m['content'])
                validate(args,aliases,visible,ledger,state,CLOSURE_EVIDENCE_TOOLS)
            except (ValueError,KeyError,TypeError) as exc:
                result[t]=dict(event_valid=False,error=str(exc)[:300])
            else:
                result[t]=dict(event_valid=True,action=args['event_action'])
    if set(result)!=bad:
        raise ValueError('Incomplete rejected-window coverage')
    return result


if __name__=='__main__':
    import sys
    old=Path(sys.argv[1])
    output=Path(sys.argv[2])
    result={c:replay(old/f'new-autonomous-{c}/{c}-event-react/records/04936') for c in ('ecg1','ecg2')}
    output.write_text(json.dumps(result,indent=2))
