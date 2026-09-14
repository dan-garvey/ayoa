"""Accept a captured first proxy final through the foundation's public interface."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = Path('/home/dan/ayoa-worktrees/covenant-single-llm')
sys.path.insert(0, str(ACTIVE))
from narrative import core

task = sys.argv[1]
match = re.fullmatch(r'foundation_(cov|bw)_(\d\d)_(author|editor)', task)
assert match
story = {'cov': 'covenant', 'bw': 'breakwater'}[match[1]]
turn, stage = int(match[2]), match[3]
subprocess.run([sys.executable, str(ROOT / 'capture.py'), task], check=True)
capture = core.read_json(ROOT / story / f'{turn:02d}.{stage}.capture.json')
assert capture['request_lines_seen_in_order'], 'Incomplete source exposure; do not advance'
assert capture['public_final_count'] == 1
file = ROOT / story / f'{turn:02d}.{stage}.final.txt'
session = ACTIVE / 'sessions/foundation_20260914' / story
state = core.load_session(session)[1]
assert len(state['turns']) == turn
assert state['pending']['request_id'] == capture['request_id']
log = core.read_json(ROOT / 'launches.json')
assert not any(entry['task'] == task for entry in log)
result = core.accept(session, capture['request_id'], file.read_text())
log.append({
    'task': task, 'model': 'gpt-5.6-terra', 'reasoning_effort': 'max',
    'story': story, 'turn': turn, 'stage': stage,
    'request_id': capture['request_id'], 'response_file': str(file.relative_to(ROOT)),
    'recorded_at': core.now(),
})
core.write_json(ROOT / 'launches.json', log)
if result['status'] == 'published':
    result = {'status': 'published', 'turn': result['turn'], 'words': len(result['output'].split())}
print(json.dumps(result))
