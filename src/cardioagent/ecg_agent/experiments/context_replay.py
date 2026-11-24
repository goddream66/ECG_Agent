"""Measure lossless history compression against archived Qwen inputs with the same tokenizer."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from cardioagent.ecg_agent.harness.compact_history import TABLE_GUIDE, table_view


def compare(path, tokenizer):
    counts = defaultdict(Counter)
    for file in sorted(path.glob('*/output/*/records/*/qwen_calls.jsonl.gz')):
        with gzip.open(file, 'rt', encoding='utf8') as stream:
            for line in stream:
                call = json.loads(line)
                name = call['tools'][0]['function']['name']
                messages = call['messages']
                original = json.loads(messages[1]['content'])
                view = table_view(original)
                changed = [key for key in original if original[key] is not view[key]]
                for key in changed:
                    table = view[key]
                    decoded = [dict(table['shared'], **dict(zip(table['columns'], row, strict=True)))
                               for row in table['rows']]
                    if decoded != original[key]:
                        raise ValueError(f'Changed evidence in {file}, {name}, {key}')
                def count_tokens(text):
                    rendered = tokenizer.apply_chat_template(text, tools=call['tools'], tokenize=False,
                                                             add_generation_prompt=True, enable_thinking=False)
                    return len(tokenizer.encode(rendered, add_special_tokens=False))
                before = count_tokens(messages)
                if 'prompt_tokens' in call.get('usage', {}) and before != call['usage']['prompt_tokens']:
                    raise ValueError(f'Archived prompt length disagrees with tokenizer in {file}')
                after = before
                if changed:
                    modified = [dict(item) for item in messages]
                    modified[0]['content'] += TABLE_GUIDE
                    modified[1]['content'] = json.dumps(view, allow_nan=False)
                    after = count_tokens(modified)
                counts[name].update(calls=1, changed=int(bool(changed)), original_tokens=before, compact_tokens=after)
    if not counts:
        raise ValueError('No completed Qwen call logs at this path')
    return dict(by_stage=counts, total=sum(counts.values(), Counter()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    report = compare(args.runs, tokenizer)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
