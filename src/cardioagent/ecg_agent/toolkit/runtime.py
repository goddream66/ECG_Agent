"""One runtime for rules, LLM tool calls and replay; no annotation or network access."""
import json
import math
from time import perf_counter
from pydantic import ValidationError
from .contracts import ToolContext, ToolResult
from ...recording import Waveform


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, spec):
        if spec.name in self._tools:
            raise ValueError(f'Duplicate tool: {spec.name}')
        self._tools[spec.name] = spec

    def schemas(self, include_experimental=False):
        return [s.schema() for s in self._tools.values() if include_experimental or not s.experimental]

    def function_schemas(self, include_experimental=False):
        return [{'type':'function','function':{k:s[k] for k in ('name','description','parameters')}}
                for s in self.schemas(include_experimental)]


class ToolSession:
    def __init__(self, context: ToolContext, registry=None, *, max_calls=32, allow_experimental=False):
        from .catalog import build_registry
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        w = context.waveform
        if (not math.isfinite(w.fs) or w.fs <= 60 or not math.isfinite(context.origin_s)
                or not math.isfinite(context.cursor_s) or context.origin_s < 0
                or not context.origin_s < context.cursor_s <= context.origin_s+w.duration+1e-9
                or w.values.ndim != 2 or w.values.shape[1] != len(w.channels)
                or len(w.units) != len(w.channels) or len(set(w.channels)) != len(w.channels)):
            raise ValueError('Invalid ECG block context')
        count = min(len(w.values), math.ceil((context.cursor_s-context.origin_s)*w.fs-1e-8))
        snapshot = Waveform(w.values[:count].copy(), w.fs, w.channels, w.units)
        self.context = ToolContext(snapshot, context.origin_s, context.cursor_s)
        self.registry = registry or build_registry()
        self.max_calls, self.allow_experimental = max_calls, allow_experimental
        self.calls, self.cache, self.internal_cache = [], {}, {}

    def window(self, args):
        c = self.context
        return c.waveform.window(args.lead, args.start_s-c.origin_s, args.end_s-c.origin_s)

    def call(self, name, arguments):
        began=perf_counter(); c=self.context
        seq=len(self.calls)+1
        common={'evidence_id':f'ecg:{c.origin_s:.8f}:{c.cursor_s:.8f}:{seq}',
                'tool':str(name),'tool_version':'unknown','window_s':None,
                'available_at_s':c.cursor_s,'lead':None,'result':{},'warnings':[],
                'error':None,'status':'error'}
        code='invalid_arguments'
        try:
            if len(self.calls)>=self.max_calls:
                code='budget_exhausted'; raise ValueError('Tool call budget exhausted')
            if not isinstance(name,str) or name not in self.registry._tools:
                code='unknown_tool'; raise ValueError('Unknown tool')
            spec=self.registry._tools[name]; common['tool_version']=spec.version
            if spec.experimental and not self.allow_experimental:
                code='experimental_disabled'; raise ValueError('Enable experimental tools explicitly')
            args=spec.arguments.model_validate(arguments)
            common.update(lead=args.lead, window_s=[args.start_s,args.end_s])
            if args.lead not in c.waveform.channels:
                code='unknown_lead'; raise ValueError('Lead not present in ECG context')
            if args.start_s < c.origin_s-1e-9 or args.end_s > c.cursor_s+1e-9:
                code='unavailable_window'; raise ValueError('Requested signal is outside elapsed retained block')
            if args.end_s-args.start_s < 5:
                common.update(status='insufficient',warnings=['At least 5 seconds required'])
            else:
                key=(name,args.model_dump_json())
                if key in self.cache:
                    # Return a fresh result object: model edits cannot poison cached evidence.
                    previous=self.cache[key]
                    common.update(json.loads(json.dumps(previous)))
                    common['cached']=True
                else:
                    code='execution_error'
                    payload,status,warnings=spec.handler(self,args)
                    json.dumps(payload,allow_nan=False)
                    common.update(result=payload,status=status,warnings=warnings)
                    if status == 'error':
                        common['error'] = {'code':'tool_failed','message':str(payload.get('detector_error','Tool failed'))}
                    if status != 'error':
                        self.cache[key]={k:common[k] for k in ('result','status','warnings')}
        except (ValueError,TypeError,IndexError,ArithmeticError,RuntimeError,OSError,ValidationError) as exc:
            common.update(status='error',error={'code':code,'message':str(exc)})
        result=ToolResult(**common,duration_ms=(perf_counter()-began)*1000).model_dump()
        # Deep-copy caller-visible result to keep the audit transcript immutable.
        self.calls.append(json.loads(json.dumps(result,allow_nan=False)))
        return result
