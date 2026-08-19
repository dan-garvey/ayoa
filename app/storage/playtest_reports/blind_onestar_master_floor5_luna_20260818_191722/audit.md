# Final Session Audit

Session: `blind_onestar_master_floor5_luna_20260818_191722`
Checkpoint audited: `app/storage/sessions/blind_onestar_master_floor5_luna_20260818_191722/ckpt_0032.json`

## Final Structured State

- Schema version: `5.0`.
- Session/story: `blind_onestar_master_floor5_luna_20260818_191722` / `one_star_ascension_s1`.
- Checkpoint turn index: `32`; leading clock: `329s`.
- Binding: `the_master -> "1"` only. `player_character_id` is empty.
- Newcomer: `one_star_newcomer`, dormant at `unclaimed_player_slot`; no player binding.
- Master: active at `the_masters_screen`.
- Generated opening roster: Kael (`niflheim_first_summon_01`), Mira (`niflheim_first_summon_02`), and Tam (`niflheim_first_summon_03`), all active in `niflheim_lobby` at the final checkpoint. Renna Holt is also active in `niflheim_lobby`.
- Pending engine updates: none. Active combat: null. Pending narrator render: null. Open commitments and open Cat II events: none.
- Canonical event count: `34`; final event id: `ev_floor5_survival_resolve_001`.
- Final canonical event records Floor 5 survival completion, 200 Gold plus Goblin Totem x1, three Heroes reaching Lv.5, five consumed stamina segments, all Floors 1-5 cleared, and Floor 6: Whispering Catacombs available.

## Model And Image Configuration

The final checkpoint session config records:

- Router: `openai:gpt-5.1`
- Narrator: `openai:gpt-5.2`
- Image director: `openai:gpt-5-mini`
- Combat manager: `openai:gpt-5-mini`
- Agent default, standard, and convenience: `anthropic:claude-haiku-4-5`

The recorded player command headers set `AYOA_IMAGE_GENERATION_ENABLED=false`, `AYOA_IMAGE_DIRECTOR_ENABLED=0`, `LLM_MODEL_AGENT=claude-haiku-4-5`, `LLM_MODEL_AGENT_STANDARD=claude-haiku-4-5`, `LLM_MODEL_AGENT_CONVENIENCE=claude-haiku-4-5`, and `LLM_MODEL_CHARACTER_GEN=claude-haiku-4-5`; representative exact headers are at transcript lines 1, 3, 30, and 101. A few malformed shell-pipeline retries were caught and terminated before committing a turn; these are described in the blind feedback. No image output or image cache path appeared in player-facing output.

## Final Narrated State

The last player-facing history reports Floor 5's `Objective Complete: Survived for 5:00`, the chapter clear for Goblin Outskirts Floors 1-5, the three deployed Heroes returned to Niflheim and recovered to full HP, and Floor 6 available. The final visible player result was therefore a successful chapter clear at Turn 32.

## Narrated/Structured Mismatch

The checkpoint has no dedicated structured fields for current floor, objective, gold, materials, HP, level, formation, or stamina. Those values are present in the final canonical event's observable facts and the player-facing transcript, but not as separately queryable top-level state. The final structured character records show active lobby locations and do not expose the narrated Lv.5/HP/resource values in `mechanics`. This is an observability gap, not a play failure. The structured binding/location state does agree with the narration: only the Master was bound, the Newcomer remained unclaimed, and the surviving deployed Heroes returned to Niflheim.

## Logs

No separate session log file was present under the session directory; the audited final checkpoint and the complete player-facing transcript are the available session artifacts.
