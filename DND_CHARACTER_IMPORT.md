# D&D Character Import

This document defines the first-class character-sheet import path for D&D
sessions. The primary source is a user-exported D&D Beyond character snapshot.
Foundry, Avrae, and hand-authored JSON should be compatibility paths, not
requirements.

The canonical machine-readable schema lives at
`app/schemas/dnd_character_snapshot.schema.json`.

## Goals

* Let users bring their own D&D Beyond character data without Ayoa handling DDB
  credentials or scraping DDB server-side.
* Preserve enough structured character data for D&D play: checks, saves,
  attacks, spellcasting, resources, conditions, inventory, features, and
  provenance.
* Keep prompt context small. Full imported sheets and raw DDB payloads are
  checkpoint data, not normal router/narrator context.
* Keep Ayoa's core rules-neutral. D&D sheets are optional mechanics adapters
  attached to characters, not global assumptions in the narrative engine.
* Make future reimport/debug/rewind possible by preserving source metadata,
  import warnings, and the raw source payload when the user chooses to keep it.

## Non-Goals

* Ayoa should not ship protected D&D content.
* Ayoa should not require a Foundry license or account.
* Ayoa should not depend on Avrae's GPL bot internals.
* Ayoa should not require live D&D Beyond synchronization for initial support.
* Ayoa should not put full character features, spell descriptions, inventory
  dumps, or raw source JSON into ordinary LLM history.

## Source Flow

The intended path is:

```text
D&D Beyond character page
        |
        | user-run browser export helper
        v
DDB browser export JSON
        |
        | ayoa import command
        v
Ayoa D&D Character Snapshot
        |
        | compact projection
        v
CharacterRecord.mechanics
```

The export helper runs in the user's browser session while they are viewing a
character they can already access. It writes a local JSON file. Ayoa imports
that file. Ayoa does not receive DDB credentials, cookies, auth tokens, or
server-side DDB session state.

The importer should accept versioned export envelopes because the exact browser
shape may change over time:

```json
{
  "exporter": {
    "name": "ayoa-ddb-export",
    "version": "0.1.0"
  },
  "source": {
    "type": "dndbeyond_browser_export",
    "character_id": "123456",
    "url": "https://www.dndbeyond.com/characters/123456"
  },
  "raw": {}
}
```

The importer then normalizes the raw data into the canonical snapshot schema.
If DDB changes its page/runtime shape, only the browser exporter and DDB mapper
should change. Runtime mechanics should continue reading the canonical
snapshot.

Current CLI:

```bash
./.venv/bin/python scripts/import_dndbeyond_character.py \
  /path/to/dndbeyond_character.json \
  --out /tmp/character_snapshot.json \
  --mechanics-out /tmp/character_mechanics.json
```

Use `--no-raw-source` when you want a smaller snapshot that omits the full DDB
payload while still preserving normalized actions, spells, features, inventory,
resources, and source references.

## Storage Contract

Until the mechanics model gets a dedicated typed field, store the full snapshot
under `CharacterRecord.mechanics.dnd5e_sheet` and keep the existing compact
fields at the top level for current Cat II compatibility.

```json
{
  "mechanics": {
    "ruleset_id": "dnd5e_basic",
    "ability_scores": {
      "str": 16,
      "dex": 12,
      "con": 14,
      "int": 10,
      "wis": 10,
      "cha": 10
    },
    "proficiency_bonus": 2,
    "skill_proficiencies": ["athletics"],
    "saving_throw_proficiencies": ["str", "con"],
    "armor_class": 16,
    "hit_points": {
      "current": 12,
      "max": 12,
      "temporary": 0
    },
    "conditions": [],
    "resources": {},
    "raw": {},
    "dnd5e_sheet": {
      "schema_version": 1,
      "ruleset_id": "dnd5e_2024",
      "source": {
        "type": "dndbeyond_browser_export",
        "exported_at": "2026-05-05T00:00:00Z"
      },
      "identity": {
        "name": "Alice"
      },
      "statblock": {
        "ability_scores": {
          "str": {"score": 16},
          "dex": {"score": 12},
          "con": {"score": 14},
          "int": {"score": 10},
          "wis": {"score": 10},
          "cha": {"score": 10}
        },
        "proficiency_bonus": 2,
        "skills": {
          "athletics": {
            "ability": "str",
            "value": 5,
            "proficiency_multiplier": 1
          }
        },
        "saves": {
          "str": {
            "ability": "str",
            "value": 5,
            "proficiency_multiplier": 1
          },
          "con": {
            "ability": "con",
            "value": 4,
            "proficiency_multiplier": 1
          }
        },
        "defenses": {
          "armor_class": {"value": 16},
          "hit_points": {
            "current": 12,
            "max": 12,
            "temporary": 0
          }
        }
      }
    }
  }
}
```

Important: do not put the full DDB payload in top-level `mechanics.raw` while
`mechanics_summary()` still exposes `raw` to the D&D Cat II prompt. Full raw
source belongs inside `dnd5e_sheet.raw_source`, which is ignored by the current
prompt projection.

The compact projection is a convenience view. It should not be the long-term
source of truth for D&D math because it cannot represent expertise, half
proficiency, item bonuses, conditional advantage, class features, spell slots,
or consumable resources. Future mechanics helpers should prefer
`dnd5e_sheet.statblock` when present and fall back to the compact projection
only for old checkpoints and hand-authored test fixtures.

## Snapshot Shape

The schema is intentionally Avrae-like without depending on Avrae code:

* `identity`: name, species, background, class levels, portrait, appearance.
* `statblock.ability_scores`: six abilities with scores, modifiers, and
  provenance.
* `statblock.skills` and `statblock.saves`: computed total modifiers plus
  proficiency multiplier, bonuses, advantage state, and source refs.
* `statblock.defenses`: AC, HP, hit dice, death saves, movement, senses,
  resistances, immunities, vulnerabilities, and current conditions.
* `statblock.resources`: class features, item charges, spell slots, hit dice,
  limited-use traits, and other counters with reset semantics.
* `statblock.actions`: attacks, spells, features, item activations, reactions,
  bonus actions, and custom actions.
* `statblock.spellcasting`: casting profiles, slots, pact slots, and spell
  records.
* `statblock.inventory`: equipment, weapons, armor, consumables, currency, and
  item-granted actions/resources.
* `statblock.features`: class, species, background, feat, item, boon, and
  custom features.
* `statblock.effects`: passive and active effects that affect rolls, defenses,
  resources, or actions.
* `raw_source`: optional local-only source payload for reimport/debug.

Most imported objects include `source_refs`, `description`, `automation`, and
`raw` fields. That is deliberate:

* `source_refs` let us trace a computed bonus back to a DDB entity, source
  book, item, class feature, or homebrew object.
* `description` can preserve user-owned content locally without making it part
  of prompt context by default.
* `automation` lets us later support Avrae-style or Ayoa-native effect trees
  for actions, spells, and features.
* `raw` gives each importer a narrow escape hatch without forcing schema churn
  for every DDB edge case.

## Runtime Projections

The engine should derive narrow views from the full snapshot instead of passing
the snapshot wholesale to an LLM.

Current Cat II roll planning needs:

```json
{
  "ruleset_id": "dnd5e_basic",
  "ability_scores": {},
  "proficiency_bonus": 2,
  "skill_proficiencies": [],
  "saving_throw_proficiencies": [],
  "armor_class": 16,
  "hit_points": {},
  "conditions": [],
  "resources": {}
}
```

Near-term D&D state application needs:

```json
{
  "defenses": {
    "armor_class": {},
    "hit_points": {},
    "conditions": []
  },
  "resources": [],
  "effects": []
}
```

Action/spell execution needs a selected action view:

```json
{
  "actor_id": "alice",
  "action": {
    "id": "action_shortsword",
    "name": "Shortsword",
    "activation": {"type": "action"},
    "attack": {"bonus": 5},
    "damage": [{"formula": "1d6+3", "damage_type": "piercing"}],
    "consumes": []
  },
  "relevant_resources": [],
  "relevant_effects": []
}
```

Narration needs only canonical outcome facts plus, occasionally, a tiny
human-readable equipment or condition label. It should not receive raw spell
descriptions, full feature text, source JSON, roll ledgers, or import metadata.

## Rewind And Reimport

The checkpoint is the authority for current in-session mechanics state. That
means HP, resources, conditions, temporary effects, and inventory mutations
change in the snapshot stored in the checkpoint as play proceeds. Rewinding to
an old checkpoint should restore the exact mechanics state from that turn.

DDB reimport is a later explicit operation, not a live sync. It should:

* compare `source.source_hash` and per-object ids where possible
* preserve in-session mutations unless the user explicitly chooses to reset
  from DDB
* report merge conflicts for HP, resources, prepared spells, inventory, and
  level changes
* keep the previous snapshot reachable for debug until the checkpoint history
  naturally rolls forward

## Implementation Plan

Current implementation status:

* `app.engine.dnd_character_import` imports the Ayoa browser-export envelope,
  normalizes DDB `character-service` payloads into the canonical snapshot
  shape, and derives the compact `CharacterRecord.mechanics` projection.
* `scripts/import_dndbeyond_character.py` provides a local CLI converter.
* Discord `/join` still exposes the existing character picker: pre-authored
  playable slots plus `Create your own character`. After a player selects or
  creates a character, the bot privately points them at `/attach` because
  Discord cannot accept file uploads from select menus or modals.
* Discord `/attach attachment:<json> [character_id] [name_override]` attaches a
  D&D Beyond JSON export to the invoker's currently bound character. It preserves
  the AYOA story identity by default; `name_override` is the explicit opt-in to
  rename the `CharacterRecord`. A successful attach also enables D&D session
  mode by setting `ruleset_id=dnd5e_basic`, while preserving the current
  `player_roll_mode`.
* Discord `/sheet [page]` renders compact Avrae-style pages from
  `mechanics.dnd5e_sheet` without reading or displaying `raw_source`.
* `app.engine.mechanics.roll_modifier()` now prefers detailed
  `mechanics.dnd5e_sheet.statblock.skills` and `saves` values when present,
  falling back to the older compact proficiency-list math for old checkpoints.
* The implementation has been validated against a real DDB browser export
  without committing that private export to the repo.

Remaining work:

1. Add Pydantic models matching
   `app/schemas/dnd_character_snapshot.schema.json`, or generate validators
   from the JSON Schema if we choose that route.
2. Add a redacted synthetic fixture for broader committed tests as more DDB
   shapes are encountered.
3. Keep full sheet data out of prompts by adding tests around
   `mechanics_summary()` and the D&D Cat II packet renderer.
4. Improve attack bonus derivation for weapon/spell actions. The importer
   preserves action definitions now, but exact attack automation should be a
   dedicated mechanics-engine pass.
5. Add Foundry Actor JSON and native Ayoa JSON importers as compatibility
   adapters that emit the same canonical snapshot.

## Open Questions

* Do we normalize 2014 and 2024 D&D into one `dnd5e` runtime, or keep separate
  ruleset ids when a rule materially differs?
* Do we store full protected spell/feature text by default when the user exports
  it, or require an explicit `--include-source-text` import flag?
* How much of the DDB browser exporter do we want in-repo versus documented as a
  user-run snippet?
