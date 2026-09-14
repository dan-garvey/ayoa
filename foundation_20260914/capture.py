"""Capture public finals and public read metadata only; never expose reasoning."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = Path('/home/dan/ayoa-worktrees/covenant-single-llm')
agent = '/root/' + sys.argv[1].removeprefix('/root/')
match = re.fullmatch(r'/root/foundation_(cov|bw)_(\d\d)_(author|editor)', agent)
assert match, agent
story = {'cov': 'covenant', 'bw': 'breakwater'}[match[1]]
turn, stage = int(match[2]), match[3]
log_path = None
for path in Path('/home/dan/.codex/sessions/2026/09/14').glob('*.jsonl'):
    with path.open() as handle:
        first = json.loads(handle.readline())
    if first.get('payload', {}).get('agent_path') == agent:
        log_path = path
        metadata = first['payload']
        break
assert log_path, agent

finals, calls, output_texts, configurations = [], [], [], []


def texts(value):
    if isinstance(value, list):
        for item in value:
            yield from texts(item)
    elif isinstance(value, dict):
        for key in ('text', 'output', 'content'):
            if key in value:
                yield from texts(value[key])
    elif isinstance(value, str):
        yield value
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(parsed, (dict, list)):
            yield from texts(parsed)


for line in log_path.open():
    event = json.loads(line)
    payload = event.get('payload', {})
    if event.get('type') == 'turn_context':
        configurations.append({'model': payload.get('model'), 'effort': payload.get('effort')})
    if event.get('type') != 'response_item':
        continue
    kind = payload.get('type')
    if kind == 'message' and payload.get('role') == 'assistant' and payload.get('phase') == 'final_answer':
        finals.append({'id': payload.get('id'), 'text': ''.join(part.get('text', '') for part in payload.get('content', []) if part.get('type') == 'output_text')})
    elif kind in {'function_call', 'custom_tool_call'}:
        calls.append({'call_id': payload.get('call_id'), 'name': payload.get('name'), 'source': payload.get('input', payload.get('arguments'))})
    elif kind in {'function_call_output', 'custom_tool_call_output'}:
        output_texts.extend(texts(payload.get('output')))

assert finals, 'Final response is not available yet'
assert all(c == {'model': 'gpt-5.6-terra', 'effort': 'max'} for c in configurations)
final = finals[0]['text']
assert final.strip()
written = ROOT / story / f'{turn:02d}.{stage}.txt'
written_text = written.read_text()
session = ACTIVE / 'sessions/foundation_20260914' / story
attempts = []
for path in (session / 'attempts').iterdir():
    meta = json.loads((path / 'meta.json').read_text())
    if meta['turn'] == turn and meta['stage'] == stage:
        attempts.append(path)
assert len(attempts) == 1, 'Record technical retries explicitly before capture'
request_path = attempts[0] / 'request.txt'
request = request_path.read_text()
joined = '\n'.join(output_texts)
position = 0
missing = []
for line in (line for line in request.splitlines() if line.strip()):
    found = joined.find(line, position)
    if found == -1:
        missing.append(line)
    else:
        position = found + len(line)
capture = {
    'agent': agent, 'session_id': metadata['id'], 'log_path': str(log_path),
    'configurations': configurations, 'first_public_final_id': finals[0]['id'],
    'public_final_count': len(finals),
    'first_public_final_sha256': hashlib.sha256(final.encode()).hexdigest(),
    'written_response_sha256': hashlib.sha256(written.read_bytes()).hexdigest(),
    'written_vs_final': ('identical' if written_text == final else 'trailing_newline_only' if written_text.rstrip('\n') == final.rstrip('\n') else 'different'),
    'request_id': attempts[0].name,
    'request_sha256': hashlib.sha256(request.encode()).hexdigest(),
    'request_seen_verbatim_in_public_tool_output': request in joined,
    'request_lines_seen_in_order': not missing,
    'missing_request_lines': missing,
    'public_tool_calls': calls,
    'scope': 'Public final, model/effort and public tool I/O metadata only. No private reasoning inspected. Request delivery is a file-based coding-agent proxy, not direct API equivalence.',
}
final_path = ROOT / story / f'{turn:02d}.{stage}.final.txt'
if final_path.exists():
    assert final_path.read_text() == final
else:
    final_path.write_text(final)
(ROOT / story / f'{turn:02d}.{stage}.capture.json').write_text(json.dumps(capture, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: capture[k] for k in ['agent','written_vs_final','request_lines_seen_in_order','request_seen_verbatim_in_public_tool_output']}))
