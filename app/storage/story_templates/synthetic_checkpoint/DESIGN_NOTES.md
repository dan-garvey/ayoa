# Synthetic Story Checkpoint Template

This directory is an authoring aid, not a playable story. The runtime story
list reads only `app/storage/stories/<story_id>/ckpt_0000.json`, so this
template will not appear in `/story list`.

Use `app/storage/stories/the_unblessed_summon/ckpt_0000.json` as the current
high-density reference for a hand-authored synthetic story. It shows the
expected depth for world lore, hidden lore, public/private character
descriptions, player-safe visual loadouts, playable-slot handling, and optional
D&D mechanics. Do not copy its D&D assumptions into rules-neutral stories.

## Authoring Flow

1. Copy `ckpt_0000.json` to `app/storage/stories/<story_id>/ckpt_0000.json`.
2. Replace both `session.session_id` and `session.story_id` with `<story_id>`.
3. Keep `turn_index` at `0`, and keep all conversation, transcript,
   visibility, render, Cat II, combat, loot, commitment, and canonical-event
   collections empty.
4. Fill public world fields first: `world_state.setting`, `world_state.facts`,
   `world_state.physics_ruleset`, `world_state.lore`, and
   `config.narrative_rules`.
5. Fill private world fields second: `world_state.hidden_lore` and
   `world_state.hidden_facts`. These are router/agent context, not player
   briefing text.
6. Fill `player_primer` with a short spoiler-free onboarding brief. This is
   shown by `/story start`; it is not opening prose. The opening scene is
   generated through the normal `(begin)` router and narrator path.
7. Add `characters` with stable ids and complete public/private fields. Every
   important NPC should have `backstory`, `personality`, `known_context`,
   `descriptions.public`, `descriptions.private`, and
   `visuals.default_loadout`.
8. Mark only actual player slots with `is_playable: true`. A blank player slot
   may intentionally have empty `backstory`, `personality`, `known_context`,
   and `mechanics` when those will be supplied by the player.
9. For D&D stories, set both `session.config.settings.ruleset_id` and
   `config.settings.ruleset_id` to `dnd5e_basic`. Leave rules-neutral stories
   at `narrative`.
10. Validate before handoff:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from app.schemas.checkpoint import CheckpointFile

path = Path("app/storage/stories/<story_id>/ckpt_0000.json")
CheckpointFile.model_validate_json(path.read_text())
print(f"valid: {path}")
PY
```

## Field Notes

`world_state.facts` should be concise public truths the router can rely on.
`world_state.lore` can be long-form public setting context. `hidden_lore` and
`hidden_facts` may contain spoilers, conspiracies, and secrets, but never put
player-facing onboarding text there.

`CharacterRecord.descriptions.public` is player-safe identity context. It must
not include secrets, hidden allegiances, author-only labels, motives, or private
body details. Put those in `descriptions.private`, `private_state.secrets`,
`backstory`, or `known_context` as appropriate.

`private_state.goals` are existential drives. `private_state.current_objectives`
are what the character is trying to do now. `intentions_enabled` should be true
for characters whose goals should keep moving when the player is elsewhere.

`known_context` is per-character knowledge, not omniscient lore. Write it in a
shape useful to that character's future agent calls: what they know, what they
believe, what they misunderstand, and what they are trying not to reveal.

`mechanics` stays `{}` for rules-neutral stories. For D&D stories, player
characters can still intentionally start with `{}` so `/attach` can import the
real sheet later; NPCs that roll in play should carry enough D&D fields for the
mechanics helpers to compute modifiers, armor class, hit points, resources, and
conditions.

Do not add retired importer fields (`importer_version`, `import_analysis`) or
old runtime fields (`world_state.locations`, `turns_since_last_tick`,
`surfaced_world_facts`, `pending_router_state_changes`,
`opening_narrative`). If a current schema field feels unclear, inspect
`app/schemas/checkpoint.py`, `app/schemas/state.py`, and
`app/schemas/characters.py` rather than adding a parallel authoring field.
