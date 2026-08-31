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
3. Add the new story directory and `ckpt_0000.json` to the explicit
   `.gitignore` allowlist. Runtime session checkpoints and local story drafts
   stay ignored; shipped story seeds must be intentional.
4. Keep `turn_index` at `0`, and keep all conversation, visibility, render,
   Cat II, combat, loot, commitment, and canonical-event
   collections empty.
5. Fill public world fields first: `world_state.setting`, `world_state.facts`,
   `world_state.physics_ruleset`, `world_state.lore`, and
   `config.narrative_rules`.
6. Fill private world fields second: `world_state.hidden_lore` and
   `world_state.hidden_facts`. These are router/agent context, not player
   briefing text.
7. Fill `player_primer` with a short spoiler-free onboarding brief. This is
   shown by `/story start`; it is not opening prose. The opening scene is
   generated through the normal `(begin)` router and narrator path.
8. Add `characters` with stable ids. Put player-safe identity in
   `public_sheet` and visible first-look material in `visuals.default_loadout`.
   Give a character an `actor` record only when authored private facts or
   offstage agency are warranted; zero facts and `actor: null` are valid.
9. Mark only actual player slots with `is_playable: true`. A blank player slot
   may intentionally have `actor: null` and empty `mechanics` when identity
   will be supplied by the player.
10. For D&D stories, set `session.config.settings.ruleset_id` to
   `dnd5e_basic`. Leave rules-neutral stories at `narrative`.
11. Validate before handoff:

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

`CharacterRecord.public_sheet.public_context` is player-safe identity context.
It must not include secrets, hidden allegiances, author-only labels, motives,
or private body details.

`CharacterRecord.actor.facts` is a sparse list of concrete second-person facts
owned by that character. Each fact records `origin` as `lived`, `witnessed`,
`told`, or `inferred`; uncertainty belongs in the text. Do not pad characters
to a common fact count or force biography, motive, secret, voice, and objective
categories. Set `actor.may_act_offstage` only for people who can sustain
independent pressure away from the current scene.

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
