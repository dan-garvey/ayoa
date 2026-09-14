"""Audit and archive completed sessions through the same request builder."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = Path('/home/dan/ayoa-worktrees/covenant-single-llm')
sys.path.insert(0, str(ACTIVE))
from narrative import core

launches = core.read_json(ROOT / 'launches.json')
assert len(launches) == 16
assert len({item['task'] for item in launches}) == 16
source_lock = core.read_json(ROOT / 'source_lock.json')
delivery = core.read_json(ROOT / 'delivery_cleanup.json')['changes']
delivered_changes = {}
for name, expected in source_lock.items():
    current = (ACTIVE / name).read_bytes()
    if hashlib.sha256(current).hexdigest() == expected:
        continue
    # The tested canon remains byte-for-byte frozen; delivery removes only
    # three trailing blanks after the run. Do not relax other source checks.
    assert name == 'stories/covenant/canon.md'
    change = delivery[name]
    frozen = (ROOT / change['frozen_original']).read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == expected == change['before_sha256']
    assert hashlib.sha256(current).hexdigest() == change['after_sha256']
    assert current == b'\n'.join(line.rstrip(b' \t') for line in frozen.split(b'\n'))
    assert [i for i, (a, b) in enumerate(zip(frozen.splitlines(), current.splitlines()), 1) if a != b] == change['changed_lines']
    delivered_changes[name] = change
assert delivered_changes == delivery
results = {}
for story in ['covenant', 'breakwater']:
    session = ACTIVE / 'sessions/foundation_20260914' / story
    manifest, state = core.load_session(session)
    assert state['pending'] is None and len(state['turns']) == 4
    pairs = []
    for turn in state['turns']:
        base = {'turns': state['turns'][:turn['turn']], 'pending': {'input': turn['input'], 'author': None, 'request_id': turn['author']}}
        pair = {'turn': turn['turn'], 'input': turn['input']}
        for stage in ['author', 'editor']:
            request_id = turn[stage]
            path = session / 'attempts' / request_id
            if stage == 'editor':
                base['pending']['author'] = turn['author']
                base['pending']['request_id'] = request_id
            request = core.read_json(path / 'request.json')
            assert request == core.make_request(session, manifest, base)
            assert (path / 'request.txt').read_text() == core.render_request(request)
            response = core.response_text(core.read_json(path / 'response.json'))
            entry = next(item for item in launches if item['request_id'] == request_id)
            artifact = ROOT / entry['response_file']
            assert artifact.read_text() == response
            capture = core.read_json(ROOT / story / f"{turn['turn']:02d}.{stage}.capture.json")
            assert capture['request_id'] == request_id and capture['request_lines_seen_in_order']
            assert capture['public_final_count'] == 1
            pair[stage + '_words'] = len(response.split())
            pair[stage + '_sha256'] = hashlib.sha256(response.encode()).hexdigest()
            if stage == 'editor':
                assert response == turn['output']
        pairs.append(pair)
    summary = core.export(session)
    assert summary['requests_prepared'] == summary['responses_received'] == 8
    assert summary['api_calls_started'] == 0
    assert summary['reported_usage'] is None
    target = ROOT / story / 'session'
    if not target.exists():
        shutil.copytree(session, target, ignore=shutil.ignore_patterns('.lock'))
    else:
        for source in session.rglob('*'):
            if source.is_file() and source.name != '.lock':
                assert (target / source.relative_to(session)).read_bytes() == source.read_bytes()
    # The archived copy is independently readable after leaving the active checkout.
    copied_manifest, copied_state = core.load_session(target)
    assert copied_manifest == manifest and copied_state == state
    results[story] = {'summary': summary, 'pairs': pairs}

report = {
    'passed': True, 'stories': results, 'fresh_model_responses': 16,
    'published_passages': 8, 'player_turns_after_openings': 6,
    'direct_api_calls': 0, 'rerolls': 0, 'frozen_source_locks_match': True,
    'delivered_source_changes': delivered_changes,
    'all_requests_reconstructed_from_frozen_sources_and_published_history': True,
    'all_public_source_reads_complete': True,
    'source_and_response_notes': [
        'The first two author inputs were accepted from their agent-written files, which differ from the captured public finals only by one trailing newline. Both byte sequences are preserved.',
        'Subsequent stages accept the captured first public final verbatim. Agent-written response artifacts remain separately preserved.',
        'This is a short implementation smoke playtest, not the broader literary selection protocol. No private reasoning or direct API usage was inspected or inferred.',
    ],
}
core.write_json(ROOT / 'validation.json', report)
print(json.dumps(report, indent=2))
