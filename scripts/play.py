#!/usr/bin/env python3
"""Interactive CLI alternative to the Discord bot.

Drives the engine in-process via EngineBridge — same path the bot uses — so
bindings, per-session locks, rolling conversations, and multi-player POV all
behave identically. The CLI simulates multiple Discord users from one
terminal by minting synthetic integer user_ids (1, 2, 3, ...) per /join,
which exercises the full binding code path and lets you playtest multi-
character POVs without needing two Discord accounts.

Usage:
    .venv/bin/python scripts/play.py --session <name>

A named `session` is a persistent save; `story` content is loaded into it.
If the session doesn't exist yet it's created empty — use `/story list`
then `/story start <id>` from inside the REPL to load content.

Non-interactive / multiplayer:
    .venv/bin/python scripts/play.py --session <name> --command "/status"
    .venv/bin/python scripts/play.py --session <name> --as <seat> --command "<action>"

`--command` runs one REPL line (repeatable) then exits, and `--as` selects
which already-bound character acts. Separate terminals — or separate agents —
can each drive one bound character this way against the same session; an
advisory per-session file lock serializes the mutating turns so concurrent
processes cannot clobber each other's checkpoints.

Commands inside the REPL:
    /help                       Show commands
    /story list                 List available stories
    /story start <id|#>         Load a story into this session
    /story info <id|#>          Show briefing for a story
    /story delete               Unload the current story (leaves session empty)
    /session list               List existing sessions
    /session end                Exit the REPL (save files stay on disk)
    /characters                 List playable seats by claim state
    /character list             Alias for /characters
    /character <id|#>           Full dossier for one character
    /join <name|#>              Claim a character; prints their dossier
    /join_custom                Create your own character
    /leave [name|#]             Release a claim (default: current actor)
    /as <name|#>                Switch which claimed character acts next
    /describe                   Set name + appearance of the current actor
    /defer                      Submit no action and let the scene continue
    /begin                      Open the story for the joined lobby
    /attach <json> [id]         Attach a D&D Beyond JSON export
    /sheet [page] [id]          Show an attached D&D character sheet
    /inventory [character_id]   Show current D&D inventory
    /loot                       List open D&D loot offers
    /loot all                   Claim all open D&D loot offers
    /roll [roll_id|all]         Roll pending D&D player check(s)
    /combat status              Show active D&D combat order and HP
    /query <question>           Ask an out-of-character question (POV-bounded)
    /rewind [<N>]               List or rewind to a turn checkpoint
    /settings                   Show / update experimental settings
    /status                     Session summary
    /history [N]                Print all turns, or last N
    /quit                       Exit (Ctrl-D also works)

Anything not starting with '/' is an in-character action for the current
actor. Use /begin to open the story — the router composes the opening from
world_state + the initial roster on the fly.
"""

from __future__ import annotations

import atexit
import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import sys
import time
import warnings
from pathlib import Path
from typing import Any

try:
    import readline as _readline
except ImportError:  # pragma: no cover - platform dependent.
    _readline = None

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - platform dependent.
    _fcntl = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.bot.player_errors import player_safe_error_message
from app.bot.engine_bridge import (
    EngineBridge,
    joinable_character_summaries,
)
from app.engine import dnd_experience, dnd_inventory, dnd_presentation
from app.engine.frontend_views import (
    CharacterSummary,
    CompletedPendingRoll,
    DndCombatParticipantView,
    DndCombatView,
    DndInventoryView,
    DndSheetAttachmentSummary,
    PendingRollPrompt,
)
from app.engine.cli_image_display import (
    CliImageDisplayRenderer,
    CliImageDisplayResult,
)
from app.engine.text_safety import strip_terminal_control
from app.llm.config import LLMConfig, MissingLLMCredential, live_play_required_roles
from app.schemas.image_generation import ImageDeliveryKind

load_dotenv()
logger = logging.getLogger(__name__)

MAX_DND_CHARACTER_BYTES = 5_000_000
DND_SHEET_PAGES = (
    "overview",
    "abilities",
    "actions",
    "spells",
    "inventory",
    "features",
)
ABILITY_ORDER = ("str", "dex", "con", "int", "wis", "cha")
ABILITY_LABELS = {
    "str": "STR",
    "dex": "DEX",
    "con": "CON",
    "int": "INT",
    "wis": "WIS",
    "cha": "CHA",
}


HELP_TEXT = """\
Commands:
  /help                             Show context-relevant commands
  /help all                         Show every command
  /story list                       List available stories
  /story start <id|#>               Load a story into this session
  /story info <id|#>                Show briefing for a story
  /story delete                     Unload the current story from this session
  /session list                     List existing sessions
  /session end                      Exit the REPL (files stay)
  /characters                       List playable seats by claim state
  /character list                   Alias for /characters
  /character <id|#>                 Full dossier for a character
  /join <name|#>                    Claim a character; prints their dossier
  /join_custom                      Create your own character
  /leave [name|#]                   Release a claim (default: current actor)
  /as <name|#>                      Switch which claimed character acts next
  /describe [--name N] [--appearance A]
                                    Set current actor identity without starting
  /defer                            Submit no action and let the scene continue
  /begin [--confirm]                Open the story for the joined lobby
  /attach <json> [id] [--name N]    Attach a D&D Beyond JSON export
  /sheet [page] [character_id]      Show an attached D&D character sheet
  /sheet all [character_id]         Show every sheet page
  /inventory [character_id]         Show current D&D inventory
  /loot                             List open D&D loot offers
  /loot all                         Claim all open D&D loot offers
  /loot take <offer> <all|ids>      Claim all or comma-separated item ids
  /loot split-coins <offer>         Split offer coins among eligible players
  /loot decline <offer>             Decline a loot offer
  /roll [roll_id|all]               Roll pending D&D player check(s)
  /combat begin [id,id...]          Begin D&D combat
  /combat status                    Show active D&D combat order and HP
  /combat next                      Advance D&D combat to the next turn
  /combat end                       End active D&D combat
  /combat damage <id> <amount>      Apply damage to a combat participant
  /combat heal <id> <amount>        Apply healing to a combat participant
  /combat add <id>                  Add a character to active combat
  /combat remove <id> [hard]        Remove a combat participant
  /query <question>                 Ask an out-of-character question (POV-bounded)
  /image lock id:<candidate_id>     Accept a provisional identity portrait
  /image reroll [id:<reference_id>] Generate a replacement identity portrait
  /rewind                           List available turn checkpoints
  /rewind <N>                       Preview rewind to ckpt_N, then confirm delete
  /settings                         Show experimental settings for this session
  /settings <key>                   Show one setting's current value
  /settings <key> <value>           Update a setting
  /status                           Session summary
  /history [N]                      Print all turns, or last N
  /quit                             Exit (Ctrl-D also works)

Plain text is an in-character action for the current actor. Use /begin
to open the story — the router composes the opening from world_state +
the initial roster on the fly."""


def _default_history_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "ayoa" / "play_history"
    return Path.home() / ".local" / "state" / "ayoa" / "play_history"


def _command_completions() -> tuple[str, ...]:
    commands: set[str] = set()
    for line in HELP_TEXT.splitlines():
        stripped = line.strip()
        if not stripped.startswith("/"):
            continue
        parts = stripped.split()
        if not parts:
            continue
        commands.add(parts[0])
        if (
            len(parts) > 1
            and not parts[1].startswith("[")
            and not parts[1].startswith("<")
        ):
            commands.add(f"{parts[0]} {parts[1]}")
    commands.update({
        "/combat add",
        "/combat damage",
        "/combat end",
        "/combat heal",
        "/combat next",
        "/loot decline",
        "/loot split-coins",
        "/loot take",
    })
    return tuple(sorted(commands))


_CLI_COMPLETIONS = _command_completions()


class _ConsoleInput:
    """Read player input with optional readline editing and history."""

    def __init__(
        self,
        *,
        history_path: Path | None = None,
        readline_module: Any = _readline,
        interactive: bool | None = None,
    ) -> None:
        self.history_path = history_path
        self._readline = readline_module
        self._interactive = (
            sys.stdin.isatty() and sys.stdout.isatty()
            if interactive is None else interactive
        )
        self._installed = False
        self._atexit_registered = False

    def install(self) -> bool:
        if self._readline is None or not self._interactive:
            return False
        self._installed = True
        self._read_history()
        self._configure_readline()
        if self.history_path is not None and not self._atexit_registered:
            atexit.register(self.save_history)
            self._atexit_registered = True
        return True

    async def prompt(
        self,
        label: str,
        *,
        record_history: bool = False,
    ) -> str:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, input, label)
        if record_history:
            self.add_history(raw)
        return raw

    def add_history(self, raw: str) -> None:
        if not self._installed or not raw.strip():
            return
        try:
            length = self._readline.get_current_history_length()
            if length and self._readline.get_history_item(length) == raw:
                return
            self._readline.add_history(raw)
        except Exception:
            logger.debug("readline add_history failed", exc_info=True)

    def save_history(self) -> None:
        if not self._installed or self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._readline.write_history_file(str(self.history_path))
        except Exception:
            logger.debug("readline history write failed", exc_info=True)

    def _read_history(self) -> None:
        if self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._readline.read_history_file(str(self.history_path))
        except FileNotFoundError:
            return
        except Exception:
            logger.debug("readline history read failed", exc_info=True)

    def _configure_readline(self) -> None:
        for binding in (
            "tab: complete",
            "set show-all-if-ambiguous on",
            "set completion-ignore-case on",
        ):
            try:
                self._readline.parse_and_bind(binding)
            except Exception:
                logger.debug("readline binding failed: %s", binding, exc_info=True)
        try:
            self._readline.set_history_length(1000)
        except Exception:
            logger.debug("readline history length setup failed", exc_info=True)
        try:
            self._readline.set_completer(self._complete)
        except Exception:
            logger.debug("readline completer setup failed", exc_info=True)
        try:
            self._readline.set_completer_delims("\t\n")
        except AttributeError:
            return
        except Exception:
            logger.debug("readline completer delimiter setup failed", exc_info=True)

    def _complete(self, text: str, state: int) -> str | None:
        matches = [cmd for cmd in _CLI_COMPLETIONS if cmd.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None


def _roll_result_line(result: CompletedPendingRoll) -> str:
    crit_note = ""
    if result.crit == "crit":
        crit_note = " Critical success."
    elif result.crit == "fail":
        crit_note = " Natural 1."
    detail = result.detail or f"{result.expression} = `{result.total}`"
    return f"Rolled {result.label}: {detail}.{crit_note}"


def _signed_modifier(value: int) -> str:
    if value > 0:
        return f" + {value}"
    if value < 0:
        return f" - {abs(value)}"
    return " + 0"


def _roll_heading(roll: Any) -> str:
    actor = (
        getattr(roll, "actor_name", "")
        or getattr(roll, "actor_id", "")
        or "Unknown"
    )
    label = getattr(roll, "label", "") or "Roll"
    target = (
        getattr(roll, "target_name", "")
        or getattr(roll, "target_id", "")
        or ""
    )
    heading = f"{actor}: {label}" if actor else label
    return f"{heading} vs {target}" if target else heading


def _d20_text(roll: Any) -> str:
    values = list(getattr(roll, "die_values", ()) or ())
    kept = list(getattr(roll, "kept_die_values", ()) or values)
    if not values:
        match = re.search(
            r"1d20\s*\((?:\*\*)?(\d+)",
            str(getattr(roll, "detail", "") or ""),
        )
        if match:
            values = [int(match.group(1))]
            kept = values
    if not values:
        return "?"
    if len(values) == 1:
        return str(values[0])
    kept_text = "/".join(str(v) for v in kept) if kept else str(values[-1])
    return f"{'/'.join(str(v) for v in values)} -> {kept_text}"


def _roll_formula_line(roll: Any) -> str:
    total = int(getattr(roll, "total", 0) or 0)
    modifier = int(getattr(roll, "modifier", 0) or 0)
    if modifier == 0:
        match = re.fullmatch(
            r"1d20(?P<modifier>[+-]\d+)?",
            str(getattr(roll, "expression", "") or "").strip(),
        )
        if match and match.group("modifier"):
            modifier = int(match.group("modifier"))
    dc = int(getattr(roll, "dc", 0) or 0)
    outcome = str(getattr(roll, "outcome", "") or "")
    line = f"d20 {_d20_text(roll)}{_signed_modifier(modifier)} = {total}"
    if dc:
        line = f"{line} vs DC {dc}"
    if outcome:
        line = f"{line} ({outcome.upper()})"
    return line


def _has_d20_roll_evidence(roll: Any) -> bool:
    if list(getattr(roll, "die_values", ()) or ()):
        return True
    expression = str(getattr(roll, "expression", "") or "").strip()
    if expression.startswith("1d20"):
        return True
    detail = str(getattr(roll, "detail", "") or "")
    return "1d20" in detail


def _is_damage_only_roll(roll: Any) -> bool:
    kind = str(getattr(roll, "kind", "") or "")
    if kind == "damage_roll":
        return True
    return bool(getattr(roll, "damage_total", 0) or 0) and not _has_d20_roll_evidence(
        roll
    )


def _is_healing_roll(roll: Any) -> bool:
    return str(getattr(roll, "kind", "") or "") == "healing_roll"


def _plain_roll_detail(text: str) -> str:
    return " ".join(str(text or "").replace("`", "").replace("**", "").split())


def _damage_roll_line(roll: Any) -> str:
    total = int(
        getattr(roll, "damage_total", 0)
        or getattr(roll, "total", 0)
        or 0
    )
    raw_total = int(getattr(roll, "damage_raw_total", 0) or 0)
    damage_type = str(getattr(roll, "damage_type", "") or "")
    suffix = f" {damage_type}" if damage_type else ""
    detail = _plain_roll_detail(str(getattr(roll, "damage_detail", "") or ""))
    expression = str(getattr(roll, "damage_expression", "") or "").strip()
    if detail:
        line = f"Damage: {detail}{suffix}"
    elif expression:
        line = f"Damage: {expression} = {total}{suffix}"
    else:
        line = f"Damage: {total}{suffix}"
    if raw_total and raw_total != total:
        line = f"{line}; applied {total}{suffix} after adjustments"
    return line


def _hp_text(current: int, maximum: int, temporary: int = 0) -> str:
    text = f"{current}/{maximum}" if maximum else str(current)
    if temporary:
        text = f"{text} (+{temporary} temp)"
    return text


def _damage_target_status_line(roll: Any) -> str:
    maximum = int(getattr(roll, "target_hp_max", 0) or 0)
    before = int(getattr(roll, "target_hp_before", 0) or 0)
    after = int(getattr(roll, "target_hp_after", 0) or 0)
    temp_before = int(getattr(roll, "target_temp_hp_before", 0) or 0)
    temp_after = int(getattr(roll, "target_temp_hp_after", 0) or 0)
    if not any((maximum, before, after, temp_before, temp_after)):
        return ""
    target = (
        getattr(roll, "target_name", "")
        or getattr(roll, "target_id", "")
        or "Target"
    )
    before_text = _hp_text(before, maximum, temp_before)
    after_text = _hp_text(after, maximum, temp_after)
    state = str(getattr(roll, "target_defeat_state", "") or "")
    state_text = (
        f"; {state}" if state and state not in {"active", "unknown"} else ""
    )
    return f"Target HP: {target} {before_text} -> {after_text}{state_text}"


def _healing_roll_line(roll: Any) -> str:
    total = int(getattr(roll, "total", 0) or 0)
    detail = _plain_roll_detail(str(getattr(roll, "detail", "") or ""))
    expression = str(getattr(roll, "expression", "") or "").strip()
    if detail:
        return f"Healing: {detail}"
    if expression:
        return f"Healing: {expression} = {total}"
    return f"Healing: {total}"


def _print_healing_roll_display(roll: Any, *, include_reason: bool = False) -> None:
    print()
    print(f"--- D&D Healing · {_roll_heading(roll)} ---")
    print(f"  {_healing_roll_line(roll)}")
    status = _damage_target_status_line(roll)
    if status:
        print(f"  {status}")
    if include_reason:
        reason = str(getattr(roll, "reason", "") or "")
        if reason:
            print(f"  {reason}")


def _print_damage_roll_display(roll: Any, *, include_reason: bool = False) -> None:
    print()
    print(f"--- D&D Damage · {_roll_heading(roll)} ---")
    print(f"  {_damage_roll_line(roll)}")
    status = _damage_target_status_line(roll)
    if status:
        print(f"  {status}")
    if include_reason:
        reason = str(getattr(roll, "reason", "") or "")
        if reason:
            print(f"  {reason}")


def _print_d20_roll_display(roll: Any, *, include_reason: bool = False) -> None:
    if _is_healing_roll(roll):
        _print_healing_roll_display(roll, include_reason=include_reason)
        return
    if _is_damage_only_roll(roll):
        _print_damage_roll_display(roll, include_reason=include_reason)
        return
    print()
    print(f"--- D&D Roll · {_roll_heading(roll)} ---")
    animate = sys.stdout.isatty()
    if animate:
        values = list(getattr(roll, "die_values", ()) or [])
        final_value = values[-1] if values else "?"
        for frame in ("rolling.", "rolling..", "rolling..."):
            print(f"  d20 {frame}", end="\r", flush=True)
            time.sleep(0.18)
        print(f"  d20 settles on {final_value}".ljust(48))
        time.sleep(0.12)
    print(f"  {_roll_formula_line(roll)}")
    crit = str(getattr(roll, "crit", "") or "")
    if crit == "crit":
        print("  Critical success.")
    elif crit == "fail":
        print("  Natural 1.")
    damage_total = int(getattr(roll, "damage_total", 0) or 0)
    if damage_total > 0:
        print(f"  {_damage_roll_line(roll)}")
        status = _damage_target_status_line(roll)
        if status:
            print(f"  {status}")
    if include_reason:
        reason = str(getattr(roll, "reason", "") or "")
        if reason:
            print(f"  {reason}")


def _print_completed_roll_result(result: CompletedPendingRoll) -> None:
    _print_d20_roll_display(result, include_reason=True)
    print(_roll_result_line(result))


def _print_dice_roll_displays(rolls: list[Any]) -> None:
    for roll in rolls or []:
        _print_d20_roll_display(roll)


def _experience_award_line(award: Any) -> str:
    name = (
        str(getattr(award, "character_name", "") or "")
        or str(getattr(award, "character_id", "") or "Character")
    )
    amount = int(getattr(award, "amount", 0) or 0)
    source = str(getattr(award, "source", "") or "").strip()
    progress = dnd_experience.format_experience_progress({
        "experience_points": getattr(award, "experience_points", 0),
        "total_level": getattr(award, "total_level", 0),
        "eligible_level": getattr(award, "eligible_level", 0),
        "level_available": bool(getattr(award, "eligible_level", 0)),
        "next_level": getattr(award, "next_level", 0),
        "xp_to_next_level": getattr(award, "xp_to_next_level", 0),
    })
    source_text = f" — {source}" if source else ""
    return f"{name}: +{amount:,} XP{source_text}. {progress}"


def _print_experience_awards(awards: list[Any]) -> None:
    if not awards:
        return
    print()
    print("--- XP Gained ---")
    for award in awards:
        print(f"  {_experience_award_line(award)}")


def _print_roll_prompts(
    prompts: list[PendingRollPrompt],
    *,
    include_actor: bool = False,
) -> None:
    if not prompts:
        return
    print()
    print("--- Pending D&D Rolls ---")
    show_actor = include_actor or len({p.actor_id for p in prompts}) > 1
    for prompt in prompts:
        actor = f"{prompt.actor_id}: " if show_actor and prompt.actor_id else ""
        reason = f" — {prompt.reason}" if prompt.reason else ""
        print(f"  {prompt.roll_id}: {actor}{prompt.label}{reason}")
    if show_actor:
        print(
            "Use /roll all to roll every joined-character roll, or "
            "/as <character_id> then /roll <roll_id>."
        )
    elif len(prompts) == 1:
        print("Use /roll to roll it, or /roll <roll_id>.")
    else:
        print("Use /roll <roll_id> for one roll, or /roll all.")
    print()


def _print_combat_status(view: DndCombatView) -> None:
    print()
    print("--- Combat ---")
    for line in dnd_presentation.combat_status_lines(view):
        print(line)
    print()


def _print_inventory(view: DndInventoryView) -> None:
    print()
    print(f"--- Inventory · {view.character_name} ({view.character_id}) ---")
    coin_line = _coin_line(view.currency)
    if coin_line:
        print(f"Coins: {coin_line}")
    items = view.items or []
    if not items:
        if not coin_line:
            print("(empty)")
        print()
        return
    equipped = [item for item in items if item.get("equipped")]
    carried = [item for item in items if not item.get("equipped")]
    if equipped:
        print("Equipped:")
        for item in equipped:
            print(f"  - {_inventory_item_line(item)}")
    if carried:
        print("Carried:")
        for item in carried:
            print(f"  - {_inventory_item_line(item)}")
    print()


def _print_dnd_sheet(character, page: str) -> None:
    pages = _sheet_pages_for(page)
    sheet = _dnd_sheet_for(character)
    identity = sheet.get("identity") or {}
    statblock = sheet.get("statblock") or {}
    print()
    print(f"--- Sheet · {character.name} ({character.character_id}) ---")
    print(_sheet_identity_line(character, identity))
    for page_name in pages:
        print()
        print(f"## {_sheet_page_label(page_name)}")
        for line in _sheet_page_lines(character, sheet, statblock, page_name):
            print(line)
    print()


def _sheet_pages_for(page: str) -> tuple[str, ...]:
    normalized = (page or "overview").strip().lower()
    if normalized == "all":
        return DND_SHEET_PAGES
    if normalized not in DND_SHEET_PAGES:
        raise ValueError(
            f"Unknown sheet page '{page}'. Use one of: "
            f"{', '.join((*DND_SHEET_PAGES, 'all'))}."
        )
    return (normalized,)


def _dnd_sheet_for(character) -> dict[str, Any]:
    mechanics = getattr(character, "mechanics", None) or {}
    sheet = mechanics.get("dnd5e_sheet") or {}
    if not isinstance(sheet, dict) or not sheet:
        character_id = getattr(character, "character_id", "?")
        raise ValueError(
            f"`{character_id}` does not have an attached D&D sheet. "
            "Use /attach with a D&D Beyond JSON export first."
        )
    return sheet


def _sheet_page_label(page: str) -> str:
    return (page or "overview").replace("_", " ").title()


def _sheet_identity_line(character, identity: dict[str, Any]) -> str:
    bits: list[str] = []
    class_line = _class_line(identity)
    if class_line:
        bits.append(class_line)
    species = str(identity.get("species") or "").strip()
    background = str(identity.get("background") or "").strip()
    if species:
        bits.append(species)
    if background:
        bits.append(background)
    imported = str(identity.get("name") or "").strip()
    if imported and imported != getattr(character, "name", ""):
        bits.append(f"Imported name: {imported}")
    return " · ".join(bits) or "D&D character sheet"


def _class_line(identity: dict[str, Any]) -> str:
    labels = []
    for entry in identity.get("classes") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        subclass = str(entry.get("subclass") or "").strip()
        level = _safe_int(entry.get("level"), 0)
        label = name
        if subclass and subclass.lower() != name.lower():
            label = f"{subclass} {name}" if name else subclass
        if label:
            labels.append(f"{label} {level}".strip())
    return ", ".join(labels)


def _sheet_page_lines(
    character,
    sheet: dict[str, Any],
    statblock: dict[str, Any],
    page: str,
) -> list[str]:
    if page == "overview":
        return _sheet_overview_lines(character, sheet, statblock)
    if page == "abilities":
        return _sheet_ability_lines(statblock)
    if page == "actions":
        return _sheet_action_lines(statblock)
    if page == "spells":
        return _sheet_spell_lines(statblock)
    if page == "inventory":
        return _sheet_inventory_lines(dnd_inventory.inventory_view(character))
    if page == "features":
        return _sheet_feature_lines(statblock)
    return ["(unknown page)"]


def _sheet_overview_lines(
    character,
    sheet: dict[str, Any],
    statblock: dict[str, Any],
) -> list[str]:
    defenses = statblock.get("defenses") or {}
    hp = defenses.get("hit_points") or {}
    ac = defenses.get("armor_class") or {}
    initiative = defenses.get("initiative") or {}
    skills = statblock.get("skills") or {}
    perception = skills.get("perception") or {}
    hp_text = (
        f"{_safe_int(hp.get('current'), 0)}/"
        f"{_safe_int(hp.get('max'), 0)}"
    )
    temp_hp = _safe_int(hp.get("temporary"), 0)
    if temp_hp:
        hp_text += f" (+{temp_hp} temp)"
    combat = [
        f"AC {_safe_int(ac.get('value'), 0)}",
        f"HP {hp_text}",
        f"PB {_signed(statblock.get('proficiency_bonus') or 0)}",
        f"Initiative {_signed(initiative.get('value') or 0)}",
    ]
    if perception:
        passive = perception.get("passive")
        if passive is None:
            passive = 10 + _safe_int(perception.get("value"), 0)
        combat.append(f"Passive Perception {_safe_int(passive, 10)}")
    movement = _movement_line(defenses.get("movement") or {})
    if movement:
        combat.append(f"Speed {movement}")

    lines = ["Combat: " + " · ".join(combat)]
    experience = dnd_experience.format_experience_progress(
        dnd_experience.experience_view(character)
    )
    if experience:
        lines.append(f"Progression: {experience}")
    resources = _resource_lines(statblock.get("resources") or [], limit=10)
    if resources:
        lines.append("Resources:")
        lines.extend(f"  {line}" for line in resources)
    defense_lines = _defense_lines(defenses)
    if defense_lines:
        lines.append("Defenses:")
        lines.extend(f"  {line}" for line in defense_lines)
    source = sheet.get("source") or {}
    source_line = " · ".join(
        bit for bit in (
            str(sheet.get("ruleset_id") or ""),
            str(source.get("type") or ""),
        )
        if bit
    )
    if source_line:
        lines.append(f"Source: {source_line}")
    return lines


def _sheet_ability_lines(statblock: dict[str, Any]) -> list[str]:
    scores = statblock.get("ability_scores") or {}
    saves = statblock.get("saves") or {}
    lines = ["Abilities:"]
    for ability in ABILITY_ORDER:
        score = scores.get(ability) or {}
        save = saves.get(ability) or {}
        prof = " prof" if (save.get("proficiency_multiplier") or 0) > 0 else ""
        lines.append(
            f"  {ABILITY_LABELS[ability]} "
            f"{_safe_int(score.get('score'), 10):>2} "
            f"({_signed(score.get('modifier') or 0)}) · "
            f"save {_signed(save.get('value') or score.get('modifier') or 0)}"
            f"{prof}"
        )

    skill_lines = []
    for name, skill in sorted((statblock.get("skills") or {}).items()):
        if not isinstance(skill, dict):
            continue
        prof = " prof" if (skill.get("proficiency_multiplier") or 0) > 0 else ""
        skill_lines.append(
            f"  {_title_skill(name)} {_signed(skill.get('value') or 0)}"
            f"{prof}{_advantage_suffix(skill.get('advantage_state'))}"
        )
    lines.append("Skills:")
    lines.extend(skill_lines or ["  No skills imported."])
    return lines


def _sheet_action_lines(statblock: dict[str, Any]) -> list[str]:
    actions = [
        action for action in (statblock.get("actions") or [])
        if isinstance(action, dict)
    ]
    if not actions:
        return ["No actions imported."]
    lines = [_action_line(action) for action in actions[:18]]
    if len(actions) > 18:
        lines.append(f"... {len(actions) - 18} more")
    return lines


def _sheet_spell_lines(statblock: dict[str, Any]) -> list[str]:
    spellcasting = statblock.get("spellcasting") or {}
    lines: list[str] = []
    profiles = []
    for profile in spellcasting.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        profiles.append(
            f"{profile.get('name') or 'Spellcasting'} "
            f"({_upper(profile.get('ability'))}) · "
            f"ATK {_signed(profile.get('spell_attack_bonus') or 0)} · "
            f"DC {_safe_int(profile.get('spell_save_dc'), 0)}"
        )
    if profiles:
        lines.append("Spellcasting:")
        lines.extend(f"  {profile}" for profile in profiles)

    slots = _slots_line(spellcasting.get("slots") or {})
    pact = spellcasting.get("pact_slots")
    if isinstance(pact, dict):
        pact_line = (
            f"Pact L{_safe_int(pact.get('level'), 1)} "
            f"{_safe_int(pact.get('current'), 0)}/"
            f"{_safe_int(pact.get('max'), 0)}"
        )
        slots = f"{slots} · {pact_line}" if slots else pact_line
    if slots:
        lines.append(f"Slots: {slots}")

    spells = [
        spell for spell in (spellcasting.get("spells") or [])
        if isinstance(spell, dict)
    ]
    if not spells:
        return lines or ["No spells imported."]
    grouped: dict[int, list[str]] = {}
    for spell in spells:
        grouped.setdefault(_safe_int(spell.get("level"), 0), []).append(
            str(spell.get("name") or "Spell")
        )
    lines.append("Prepared/Known:")
    for level in sorted(grouped):
        label = "Cantrips" if level == 0 else f"Level {level}"
        names = ", ".join(sorted(grouped[level])[:18])
        extra = len(grouped[level]) - min(len(grouped[level]), 18)
        if extra:
            names += f", ... {extra} more"
        lines.append(f"  {label}: {names}")
    return lines


def _sheet_inventory_lines(inventory: dict[str, Any]) -> list[str]:
    items = [
        item for item in (inventory.get("items") or [])
        if isinstance(item, dict)
    ]
    equipped = [item for item in items if item.get("equipped")]
    carried = [item for item in items if not item.get("equipped")]
    currency_line = _coin_line(inventory.get("currency") or {})
    lines: list[str] = []
    if currency_line:
        lines.append(f"Coins: {currency_line}")
    if equipped:
        lines.append("Equipped:")
        lines.extend(f"  - {_inventory_item_line(item)}" for item in equipped[:15])
    if carried:
        lines.append("Carried:")
        lines.extend(f"  - {_inventory_item_line(item)}" for item in carried[:15])
        if len(carried) > 15:
            lines.append(f"  ... {len(carried) - 15} more")
    return lines or ["No items imported."]


def _sheet_feature_lines(statblock: dict[str, Any]) -> list[str]:
    features = [
        feature for feature in (statblock.get("features") or [])
        if isinstance(feature, dict)
    ]
    proficiencies = statblock.get("proficiencies") or {}
    languages = statblock.get("languages") or []
    lines: list[str] = []
    if features:
        lines.extend(_feature_line(feature) for feature in features[:20])
        if len(features) > 20:
            lines.append(f"... {len(features) - 20} more")
    else:
        lines.append("No features imported.")
    if isinstance(proficiencies, dict):
        prof_lines = []
        for key in ("armor", "weapons", "tools", "other"):
            values = [str(v) for v in proficiencies.get(key) or [] if v]
            if values:
                prof_lines.append(f"  {key.title()}: {', '.join(values[:12])}")
        if prof_lines:
            lines.append("Proficiencies:")
            lines.extend(prof_lines)
    if languages:
        lines.append("Languages: " + ", ".join(str(lang) for lang in languages[:20]))
    return lines


def _movement_line(movement: dict[str, Any]) -> str:
    bits = []
    for key in ("walk", "fly", "swim", "climb", "burrow"):
        item = movement.get(key)
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is not None:
            bits.append(f"{key} {value} {item.get('unit') or 'ft'}")
    return ", ".join(bits)


def _defense_lines(defenses: dict[str, Any]) -> list[str]:
    lines = []
    for label, key in (
        ("Resist", "damage_resistances"),
        ("Immune", "damage_immunities"),
        ("Vulnerable", "damage_vulnerabilities"),
        ("Condition immune", "condition_immunities"),
    ):
        names = [
            str(item.get("name") or item.get("id") or "")
            for item in defenses.get(key) or []
            if isinstance(item, dict)
        ]
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    conditions = [
        str(item.get("name") or item.get("id") or "")
        for item in defenses.get("conditions") or []
        if isinstance(item, dict)
    ]
    if conditions:
        lines.append(f"Conditions: {', '.join(conditions)}")
    exhaustion = _safe_int(defenses.get("exhaustion_level"), 0)
    if exhaustion:
        lines.append(f"Exhaustion: {exhaustion}")
    return lines


def _resource_lines(resources: list[dict[str, Any]], *, limit: int) -> list[str]:
    lines = []
    for res in resources[:limit]:
        if not isinstance(res, dict):
            continue
        name = str(res.get("name") or res.get("id") or "Resource")
        current = res.get("current")
        maximum = res.get("max")
        if current is not None and maximum is not None:
            value = f"{_safe_int(current, 0)}/{_safe_int(maximum, 0)}"
        else:
            value = str(current if current is not None else maximum or "")
        reset = ((res.get("reset") or {}).get("type") or "").replace("_", " ")
        suffix = f" ({reset})" if reset else ""
        lines.append(f"- {name}: {value}{suffix}".rstrip())
    if len(resources) > limit:
        lines.append(f"... {len(resources) - limit} more")
    return lines


def _action_line(action: dict[str, Any]) -> str:
    parts = [str(action.get("kind") or "").replace("_", " ")]
    attack = action.get("attack") or {}
    if isinstance(attack, dict) and attack.get("bonus") is not None:
        parts.append(f"hit {_signed(attack.get('bonus'))}")
    save = action.get("save") or {}
    if isinstance(save, dict) and (save.get("dc") or save.get("ability")):
        dc = f" DC {_safe_int(save.get('dc'), 0)}" if save.get("dc") else ""
        parts.append(f"{_upper(save.get('ability'))}{dc} save".strip())
    damage = _damage_line(action.get("damage") or [])
    if damage:
        parts.append(damage)
    text = ", ".join(part for part in parts if part)
    return f"- {action.get('name') or 'Action'}" + (f" - {text}" if text else "")


def _damage_line(items: list[dict[str, Any]]) -> str:
    bits = []
    for item in items:
        if not isinstance(item, dict):
            continue
        formula = str(item.get("formula") or "").strip()
        damage_type = str(item.get("damage_type") or "").strip()
        if formula:
            bits.append(f"{formula} {damage_type}".strip())
    return ", ".join(bits)


def _slots_line(slots: dict[str, dict[str, Any]]) -> str:
    lines = []
    for level in sorted(slots, key=lambda value: _safe_int(value, 0)):
        slot = slots[level]
        if not isinstance(slot, dict):
            continue
        lines.append(
            f"L{level}: {_safe_int(slot.get('current'), 0)}/"
            f"{_safe_int(slot.get('max'), 0)}"
        )
    return " · ".join(lines)


def _feature_line(feature: dict[str, Any]) -> str:
    kind = str(feature.get("kind") or "").replace("_", " ")
    level = _safe_int(feature.get("level"), 0)
    suffix_bits = [bit for bit in (kind, f"L{level}" if level else "") if bit]
    suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
    return f"- {feature.get('name') or 'Feature'}{suffix}"


def _advantage_suffix(value: Any) -> str:
    text = str(value or "normal")
    if text == "normal":
        return ""
    return f" {text.replace('_', ' ')}"


def _title_skill(name: str) -> str:
    return str(name).replace("-", " ").title()


def _upper(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if text else "?"


def _signed(value: Any) -> str:
    number = _safe_int(value, 0)
    return f"+{number}" if number >= 0 else str(number)


def _print_dnd_attachment_summary(summary: DndSheetAttachmentSummary) -> None:
    classes = ", ".join(summary.classes) or "unknown class"
    hp = f"{summary.hit_points_current}/{summary.hit_points_max}"
    if summary.hit_points_temporary:
        hp += f" (+{summary.hit_points_temporary} temp)"
    name_line = (
        f"Story name changed to {summary.character_name}."
        if summary.name_overridden
        else f"Story name preserved as {summary.character_name}."
    )
    print()
    print("--- D&D Sheet Attached ---")
    print(
        f"Attached {summary.imported_name or 'D&D character'} to "
        f"{summary.character_id}."
    )
    print(name_line)
    print(f"Classes: {classes}")
    print(f"Level: {summary.total_level}")
    print(f"AC / HP: {summary.armor_class} / {hp}")
    print(
        f"D&D mode: {summary.session_ruleset_id} · "
        f"player rolls {summary.player_roll_mode}"
    )
    print(
        f"Imported: {summary.skills_count} skills, "
        f"{summary.actions_count} actions, {summary.spells_count} spells, "
        f"{summary.resources_count} resources"
    )
    print("Use /inventory to view carried items.")
    print()


def _parse_attach_args(
    arg: str,
) -> tuple[Path, str | None, str | None] | None:
    try:
        tokens = shlex.split(arg)
    except ValueError as e:
        print(
            "usage: /attach <json_path> [character_id] "
            f"[--name \"Name\"] ({e})"
        )
        return None
    if not tokens:
        print("usage: /attach <json_path> [character_id] [--name \"Name\"]")
        return None

    path = Path(tokens[0]).expanduser()
    character_id: str | None = None
    name_override: str | None = None
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in {"--name", "--name-override"}:
            if i + 1 >= len(tokens):
                print(
                    "usage: /attach <json_path> [character_id] "
                    "[--name \"Name\"]"
                )
                return None
            name_override = tokens[i + 1].strip() or None
            i += 2
            continue
        if token.startswith("--name="):
            name_override = token.split("=", 1)[1].strip() or None
            i += 1
            continue
        if token in {"--character", "--character-id"}:
            if i + 1 >= len(tokens):
                print(
                    "usage: /attach <json_path> [character_id] "
                    "[--name \"Name\"]"
                )
                return None
            character_id = tokens[i + 1].strip() or None
            i += 2
            continue
        if (
            token.startswith("--character=")
            or token.startswith("--character-id=")
        ):
            character_id = token.split("=", 1)[1].strip() or None
            i += 1
            continue
        if token.startswith("--"):
            print(f"unknown attach option: {token}")
            return None
        if character_id is None:
            character_id = token
            i += 1
            continue
        print("usage: /attach <json_path> [character_id] [--name \"Name\"]")
        return None

    return path, character_id, name_override


def _parse_join_args(raw: str) -> tuple[str, str, str] | None:
    """Parse `/join <ref> [--name N] [--appearance A]`."""
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        print(f"error: {exc}")
        return None
    if not tokens:
        return "", "", ""

    character_ref = tokens[0]
    name = ""
    appearance = ""
    index = 1
    while index < len(tokens):
        flag = tokens[index]
        if flag not in {"--name", "--appearance"}:
            print(
                "usage: /join <name|#> "
                "[--name \"Name\"] [--appearance \"Description\"]"
            )
            return None
        if index + 1 >= len(tokens):
            print(f"error: {flag} requires a value")
            return None
        value = tokens[index + 1]
        if flag == "--name":
            name = value
        else:
            appearance = value
        index += 2
    return character_ref, name, appearance


def _parse_describe_args(raw: str) -> tuple[str, str] | None:
    """Parse optional noninteractive identity fields for `/describe`."""
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        print(f"error: {exc}")
        return None
    name = ""
    appearance = ""
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if flag not in {"--name", "--appearance"}:
            print(
                "usage: /describe [--name \"Name\"] "
                "[--appearance \"Description\"]"
            )
            return None
        if index + 1 >= len(tokens):
            print(f"error: {flag} requires a value")
            return None
        if flag == "--name":
            name = tokens[index + 1]
        else:
            appearance = tokens[index + 1]
        index += 2
    return name, appearance


def _print_loot_help() -> None:
    print("Loot commands:")
    print("  /loot                         List open offers")
    print("  /loot <offer>                 Inspect one offer")
    print("  /loot take <offer> [all|ids]  Claim all or comma-separated item ids")
    print("  /loot all                     Claim all open offers")
    print("  /loot split-coins <offer>     Split offer coins")
    print("  /loot decline <offer>         Decline an offer")
    print("Use offer numbers from /loot, item ids from the offer details, or full offer ids.")


def _print_loot_offers(offers) -> None:
    print()
    print("--- Loot Offers ---")
    if not offers:
        print("(none)")
        print()
        return
    for index, offer in enumerate(offers, start=1):
        print(_loot_offer_text(offer, index=index))
        print()
    print("Use /loot take <offer> [all|item_id[,item_id...]].")
    print("Use /loot split-coins <offer> or /loot decline <offer>.")
    print("Decline is final for your character while the offer remains open.")
    print()


def _loot_offer_text(offer, *, index: int | None = None) -> str:
    label = offer.source_label or offer.source_kind
    prefix = f"{index}. " if index is not None else ""
    lines = [f"{prefix}{label}"]
    available_ids = set(dnd_inventory.available_item_ids(offer))
    for item in offer.items:
        if item.item_id not in available_ids:
            continue
        lines.append(f"  {item.item_id}: {_loot_item_line(item)}")
    coin_line = _coin_line(dnd_inventory.available_currency_dict(offer))
    if coin_line:
        lines.append(f"  coins: {coin_line}")
    if offer.notes:
        lines.append(f"  note: {offer.notes}")
    return "\n".join(lines)


def _normalize_loot_ref(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _inventory_item_line(item: dict) -> str:
    return dnd_presentation.inventory_item_line(item)


def _loot_item_line(item) -> str:
    return dnd_presentation.loot_item_line(item)


def _coin_line(currency: dict) -> str:
    return dnd_presentation.currency_line(currency)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _combat_participant_label(participant: DndCombatParticipantView) -> str:
    return participant.name or participant.character_id


def _combat_initiative_line(view: DndCombatView) -> str:
    parts: list[str] = []
    for participant in view.participants:
        label = _combat_participant_label(participant)
        if participant.initiative is not None:
            label = f"{label} {participant.initiative}"
        parts.append(label)
    return ", ".join(parts)


def _split_combat_ids(arg: str) -> list[str]:
    raw = arg.strip()
    if not raw:
        return []
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    try:
        chunks = shlex.split(raw)
    except ValueError:
        chunks = raw.split()
    return [part.strip() for part in chunks if part.strip()]


def _joinable_character_line(summary: CharacterSummary) -> str:
    bits = [
        bit for bit in (
            summary.role,
            summary.faction,
        )
        if bit
    ]
    suffix = " — " + " · ".join(bits) if bits else ""
    return f"{summary.name}{suffix}"


def _enum_line(index: int, value: str, suffix: str = "") -> str:
    return f"  {index}: {value}{suffix}"


def _resolve_enum_ref(
    value: str,
    choices: list[str],
    *,
    label: str,
) -> str:
    token = value.strip()
    if not token:
        raise ValueError(f"missing {label}")
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(choices):
            return choices[index - 1]
        raise ValueError(
            f"No {label} numbered {index}. Run the list command again."
        )
    if token in choices:
        return token
    raise ValueError(f"Unknown {label} '{token}'. Run the list command again.")


def _print_joinable_characters(
    summaries: list[CharacterSummary],
    *,
    include_hint: bool,
) -> None:
    joinable = joinable_character_summaries(summaries)
    print("## Joinable")
    if joinable:
        playable = [
            summary
            for summary in summaries
            if summary.is_playable and summary.status != "culled"
        ]
        all_ids = [summary.character_id for summary in playable]
        for summary in joinable:
            index = all_ids.index(summary.character_id) + 1
            print(_enum_line(index, _joinable_character_line(summary)))
            if summary.player_guidance:
                print(f"     {summary.player_guidance}")
        if include_hint:
            print("  Use /join <#> to claim one.")
    else:
        print("  (none)")


def _cli_image_display_message(result: CliImageDisplayResult) -> str:
    if result.displayed:
        return "Displayed revealed image."
    if result.export_path is not None:
        return (
            "Image reveal could not be displayed in this terminal. "
            f"Safe export: {result.export_path}"
        )
    if result.degraded and result.error_code == "unsupported_terminal":
        return "Image reveal could not be displayed in this terminal."
    return "Could not display the revealed image."


@contextlib.asynccontextmanager
async def _progress(label: str):
    """Show that an async engine call is working while we await it.

    On a TTY this animates a one-line spinner and clears itself when the
    call returns (or raises). In non-interactive `--command` mode it prints
    a single static line so captured logs still show that work happened.
    Router/narrator turns can take a while; without this the REPL looks hung.
    """
    stream = sys.stdout
    if not stream.isatty():
        stream.write(f"* {label}…\n")
        stream.flush()
        yield
        return

    async def _spin() -> None:
        frames = "|/-\\"
        i = 0
        while True:
            stream.write(f"\r{frames[i % len(frames)]} {label}…")
            stream.flush()
            i += 1
            await asyncio.sleep(0.12)

    task = asyncio.create_task(_spin())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        stream.write("\r" + " " * (len(label) + 6) + "\r")
        stream.flush()


class CLIState:
    """Per-session state for the interactive REPL.

    Kept separate from the I/O loop so unit tests can drive command handlers
    directly without stdin/stdout. All engine mutation goes through
    EngineBridge, which is the same path the Discord bot uses.

    `story_id` is "" when the session is empty (no story loaded yet).
    """

    def __init__(
        self,
        engine: EngineBridge,
        session_id: str,
        story_id: str,
        *,
        console: _ConsoleInput | None = None,
        asset_image_renderer: CliImageDisplayRenderer | None = None,
    ):
        self.engine = engine
        self.session_id = session_id
        self.story_id = story_id
        self.console = console or _ConsoleInput(readline_module=None)
        self.asset_image_renderer = asset_image_renderer or CliImageDisplayRenderer()
        self.current_actor: str | None = None
        # char_id -> synthetic user_id. Mirrors Discord's one-user-per-char
        # binding so we exercise the full path; the CLI is just "the god user"
        # that owns many synthetic accounts.
        self.claims: dict[str, int] = {}
        self._next_user_id = 1
        self.running = True
        self.one_shot_mode = False
        self.engine.image_generation.register_delivery_handler(
            ImageDeliveryKind.cli,
            self._deliver_cli_image,
            can_present=lambda session_id, pov_character_id: (
                session_id == self.session_id
                and self.asset_image_renderer.backend.is_supported()
                and pov_character_id in self._pov_claims()
            ),
        )
        # When set, restrict printed POV renders/asset reveals to this one
        # character. Separate-terminal one-shot play sets this so each
        # terminal shows only its own player's POV, mirroring Discord's
        # per-user DMs instead of the single-terminal all-POV dump.
        self.pov_filter: str | None = None

        if self.story_id:
            self._load_existing_claims()
            self._sync_current_actor_to_active_combat(announce=False)

    def _load_existing_claims(self) -> None:
        """Adopt any bindings already on disk as CLI claims so /as works on
        resume. Bindings created by the Discord bot (large user_ids) are
        still usable — we just don't mint new CLI uids below them."""
        ckpt = self.engine.load_latest(self.session_id)
        for char_id, uid_str in ckpt.session.character_bindings.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            self.claims[char_id] = uid
            if uid >= self._next_user_id:
                self._next_user_id = uid + 1
        if self.claims and self.current_actor is None:
            self.current_actor = next(iter(self.claims))

    def refresh_from_checkpoint(self) -> None:
        """Refresh story and bindings after acquiring the process lock."""
        previous_actor = self.current_actor
        try:
            ckpt = self.engine.load_latest(self.session_id)
        except FileNotFoundError:
            self.story_id = ""
            self.claims = {}
            self.current_actor = None
            self._next_user_id = 1
            return
        self.story_id = ckpt.session.story_id or ""
        self.claims = {}
        self._next_user_id = 1
        self.current_actor = None
        self._load_existing_claims()
        if previous_actor in self.claims:
            self.current_actor = previous_actor

    # ---- dispatch ------------------------------------------------------------

    async def handle_line(self, line: str) -> None:
        """Top-level dispatch: /command or in-character action."""
        line = strip_terminal_control(line).strip()
        if not line:
            return
        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                print(f"unknown command: /{cmd} — try /help")
                return
            result = handler(arg)
            if asyncio.iscoroutine(result):
                await result
        else:
            await self._act(line)

    # ---- guards --------------------------------------------------------------

    def _require_story(self) -> bool:
        """Print a helpful message + return False if no story is loaded."""
        if self.story_id:
            return True
        print("no story loaded — /story list then /story start <id>")
        return False

    def _resolve_story_ref(self, value: str) -> str:
        return _resolve_enum_ref(
            value,
            self.engine.list_story_ids(),
            label="story",
        )

    def _character_id_choices(self) -> list[str]:
        return [
            summary.character_id
            for summary in self.engine.list_session_characters(self.session_id)
            if summary.is_playable and summary.status != "culled"
        ]

    def _resolve_character_ref(self, value: str) -> str:
        summaries = [
            summary
            for summary in self.engine.list_session_characters(self.session_id)
            if summary.is_playable and summary.status != "culled"
        ]
        token = value.strip()
        try:
            return _resolve_enum_ref(
                token,
                [summary.character_id for summary in summaries],
                label="character",
            )
        except ValueError as original_error:
            name_matches = [
                summary.character_id
                for summary in summaries
                if summary.name.strip().casefold() == token.casefold()
            ]
            if len(name_matches) == 1:
                return name_matches[0]
            if len(name_matches) > 1:
                raise ValueError(
                    f"More than one playable seat is named {token!r}; use its "
                    "number from /characters."
                ) from original_error
            raise

    def _character_name(self, character_id: str) -> str:
        return next(
            (
                summary.name
                for summary in self.engine.list_session_characters(
                    self.session_id
                )
                if summary.character_id == character_id
            ),
            character_id,
        )

    # ---- commands ------------------------------------------------------------

    def cmd_help(self, arg: str) -> None:
        mode = arg.strip().lower()
        if mode == "all":
            print(HELP_TEXT)
            return
        if mode:
            print("usage: /help [all]")
            return
        if not self.story_id:
            print("Commands for an empty session:")
            print("  /story list")
            print("  /story info <#|story_id>")
            print("  /story start <#|story_id>")
            print("  /session list")
            print("  /help all")
            return
        if not self.claims:
            print("Commands before play:")
            print("  /story              Show the current story briefing")
            print("  /characters         Show playable seats")
            print("  /join <#>           Claim a seat")
            print("  /begin              Review and open the claimed lobby")
            print("  /status")
            print("  /help all")
            return
        print("Core play commands:")
        print("  <plain text>         Act as the selected character")
        print("  /defer               Let the scene continue without acting")
        print("  /query <question>    Ask from this character's POV")
        print("  /status")
        print("  /history [N]")
        print("  /characters")
        print("  /as <#>              Select another claimed seat")
        print("  /help all")

    async def cmd_image(self, arg: str) -> None:
        parts = arg.split(maxsplit=1)
        if not parts or parts[0].lower() not in {"lock", "reroll"}:
            print(
                "usage: /image lock id:<candidate_id> | "
                "/image reroll [id:<reference_id>]"
            )
            return
        candidate_id = parts[1].strip() if len(parts) == 2 else ""
        if candidate_id.startswith("id:"):
            candidate_id = candidate_id[3:].strip()
        if parts[0].lower() == "lock" and not candidate_id:
            print("usage: /image lock id:<candidate_id>")
            return
        try:
            if parts[0].lower() == "lock":
                candidate = await self.engine.lock_image_identity(
                    session_id=self.session_id,
                    candidate_id=candidate_id,
                )
                print(
                    f"locked identity reference {candidate.candidate_id} "
                    f"for {candidate.character_id}"
                )
                return
            if not self.current_actor:
                print("join and select a character before requesting a reroll")
                return
            job = await self.engine.reroll_image_identity(
                session_id=self.session_id,
                reference_id=candidate_id,
                pov_character_id=self.current_actor,
                delivery_kind=ImageDeliveryKind.cli,
                delivery={"character_id": self.current_actor},
            )
            print(f"queued identity reroll {job.job_id}")
        except (KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}")

    # ---- story subcommands ---------------------------------------------------

    def cmd_story(self, arg: str) -> None | object:
        """Dispatcher for `/story <sub>` — forwards to cmd_story_<sub>."""
        parts = arg.split(maxsplit=1) if arg.strip() else []
        if not parts:
            return self.cmd_story_info(self.story_id)
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_story_{sub}", None)
        if handler is None:
            print(f"unknown story subcommand: {sub}")
            return
        return handler(rest)

    def cmd_story_list(self, arg: str) -> None:
        summaries = self.engine.list_story_summaries()
        if not summaries:
            print(
                "no stories available — add a synthetic checkpoint under "
                "app/storage/stories/<story_id>/ckpt_0000.json first"
            )
            return
        for index, summary in enumerate(summaries, start=1):
            marker = "  ← current" if summary.story_id == self.story_id else ""
            seats = (
                f"{summary.playable_seat_count} playable "
                f"seat{'s' if summary.playable_seat_count != 1 else ''}"
            )
            print(
                f"{index}: {summary.title} (`{summary.story_id}`) · "
                f"{seats}{marker}"
            )

    def cmd_story_info(self, arg: str) -> None:
        story_ref = arg.strip() or self.story_id
        if not story_ref:
            print("usage: /story info <story_id|#>")
            return
        try:
            story_id = self._resolve_story_ref(story_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        try:
            summary = self.engine.story_summary(story_id)
        except Exception as e:
            print(f"error: {e}")
            return
        print()
        print(f"# {summary.title}")
        print(f"Story id: {story_id}")
        if summary.genre:
            print(f"Genre: {summary.genre}")
        print(f"Playable seats: {summary.playable_seat_count}")
        if summary.recommended_players:
            print(f"Recommended: {summary.recommended_players}")
        if summary.premise:
            print(f"Premise: {summary.premise}")
        if summary.play_guidance:
            print(f"\nHow this story plays:\n{summary.play_guidance}")
        if summary.player_primer:
            print(f"\nBriefing:\n{summary.player_primer}")
        seats = [
            seat
            for seat in self.engine.list_story_characters(story_id)
            if seat.is_playable and seat.status != "culled"
        ]
        if seats:
            print("\nSeats:")
            for index, seat in enumerate(seats, start=1):
                details = " - ".join(
                    part for part in (seat.name, seat.role) if part
                )
                print(f"  {index}: {details}")
                if seat.player_guidance:
                    print(f"     {seat.player_guidance}")
        print()

    def cmd_story_start(self, arg: str) -> None:
        story_ref = arg.strip()
        if not story_ref:
            print("usage: /story start <story_id|#>")
            return
        if self.story_id:
            print(
                f"session already has story `{self.story_id}` loaded. "
                f"/story delete first."
            )
            return
        try:
            story_id = self._resolve_story_ref(story_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        try:
            self.engine.load_story_into_session(self.session_id, story_id)
        except (FileNotFoundError, FileExistsError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("load_story_into_session failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        self.story_id = story_id
        self.claims = {}
        self.current_actor = None
        self._load_existing_claims()
        print(f"loaded story `{story_id}` into session `{self.session_id}`")
        summary = self.engine.story_summary(story_id)
        print(f"# {summary.title}")
        if summary.recommended_players:
            print(f"Recommended: {summary.recommended_players}")
        if summary.play_guidance:
            print(summary.play_guidance)
        if summary.player_primer:
            print()
            print(summary.player_primer)
        print("\nNext: /characters → /join <#> → /begin")

    def cmd_story_delete(self, arg: str) -> None:
        if not self.story_id:
            print("no story loaded")
            return
        try:
            removed = self.engine.unload_story_from_session(self.session_id)
        except FileNotFoundError as e:
            print(f"error: {e}")
            return
        self.story_id = ""
        self.claims = {}
        self.current_actor = None
        print(f"unloaded story from `{self.session_id}` ({removed} files removed)")

    # ---- session subcommands -------------------------------------------------

    def cmd_session(self, arg: str) -> None:
        parts = arg.split(maxsplit=1) if arg.strip() else []
        if not parts:
            print("usage: /session [list|end]")
            return
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_session_{sub}", None)
        if handler is None:
            print(f"unknown session subcommand: {sub}")
            return
        handler(rest)

    def cmd_session_list(self, arg: str) -> None:
        ids = self.engine.list_session_ids()
        if not ids:
            print("no sessions yet")
            return
        for index, sid in enumerate(ids, start=1):
            marker = "  ← current" if sid == self.session_id else ""
            print(_enum_line(index, sid, marker))

    def cmd_session_end(self, arg: str) -> None:
        print(f"detached from `{self.session_id}` (files kept on disk)")
        self.running = False

    # ---- gameplay commands ---------------------------------------------------

    def cmd_characters(self, arg: str) -> None:
        if not self._require_story():
            return
        summaries = self.engine.list_session_characters(self.session_id)
        playable = [
            summary
            for summary in summaries
            if summary.is_playable and summary.status != "culled"
        ]
        sections = (
            ("Available", [seat for seat in playable if not seat.bound_user_id]),
            ("Yours", [seat for seat in playable if seat.character_id in self.claims]),
            (
                "Claimed by another player",
                [
                    seat for seat in playable
                    if seat.bound_user_id and seat.character_id not in self.claims
                ],
            ),
        )
        for section_index, (label, seats) in enumerate(sections):
            if section_index:
                print()
            print(f"## {label}")
            if not seats:
                print("  (none)")
                continue
            for seat in seats:
                index = playable.index(seat) + 1
                marker = "  <- acting" if seat.character_id == self.current_actor else ""
                line = " - ".join(part for part in (seat.name, seat.role) if part)
                print(f"  {index}: {line}{marker}")
                if seat.player_guidance:
                    print(f"     {seat.player_guidance}")

    def cmd_character(self, arg: str) -> None:
        if not self._require_story():
            return
        char_ref = arg.strip()
        if char_ref.lower() == "list":
            self.cmd_characters("")
            return
        if not char_ref:
            print("usage: /character <name|#>")
            return
        try:
            char_id = self._resolve_character_ref(char_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        try:
            dossier = self.engine.build_character_dossier(
                self.session_id, char_id,
            )
        except ValueError:
            print(f"no character: {char_id}")
            return
        except Exception as e:
            print(f"error: {e}")
            return
        print()
        print(dossier)
        print()

    def cmd_status(self, arg: str) -> None:
        print(f"session: {self.session_id}")
        if not self.story_id:
            print("story: (none loaded) — /story list then /story start <id>")
            return
        activity = self.engine.session_activity(
            self.session_id,
            self.pov_filter or self.current_actor or "",
        )
        print(f"story: {activity.story_id}")
        print(f"turn: {activity.turn_index}")
        if activity.viewpoint_name:
            print(f"viewpoint: {activity.viewpoint_name}")
        print(f"location: {activity.location or '(not yet in the fiction)'}")
        joined = ", ".join(activity.joined_seat_names) or "(none)"
        print(f"joined: {joined}")
        if activity.nearby_character_names:
            print("nearby: " + ", ".join(activity.nearby_character_names))
        print(f"activity: {activity.state}")
        for line in activity.ruleset_lines:
            print(line)
        if activity.last_visible_update:
            print(f"last visible update: {activity.last_visible_update}")
        self._print_open_reaction_slots()

    async def cmd_begin(self, arg: str) -> None:
        if not self._require_story():
            return
        raw = arg.strip()
        if raw not in {"", "--confirm"}:
            print("usage: /begin [--confirm]")
            return
        lobby = self.engine.opening_lobby(self.session_id)
        if lobby.requires_confirmation and raw != "--confirm":
            claimed = ", ".join(lobby.claimed_seat_names) or "none"
            open_seats = ", ".join(lobby.open_seat_names) or "none"
            print(f"claimed seats: {claimed}")
            print(f"still open: {open_seats}")
            if self.one_shot_mode:
                print("review the lobby, then run /begin --confirm")
                return
            try:
                answer = (
                    await self.console.prompt(
                        "Begin with exactly these claimed seats? [y/N] "
                    )
                ).strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print("\n(cancelled)")
                return
            if answer not in {"y", "yes"}:
                print("(cancelled - other players can join before /begin)")
                return
        actor_id = self.current_actor or next(iter(self.claims), "")
        try:
            async with _progress("opening the scene"):
                response = await self.engine.run_begin_turn(
                    session_id=self.session_id,
                    triggering_character_id=actor_id,
                )
        except ValueError as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("begin failed")
            print(f"error: {player_safe_error_message(e, operation='the opening')}")
            return
        if actor_id:
            self.current_actor = actor_id
        await self._wait_for_render_images(response, actor_id=actor_id)
        self._print_turn_response(
            response,
            actor_id=actor_id,
        )

    async def cmd_attach(self, arg: str) -> None:
        if not self._require_story():
            return
        parsed = _parse_attach_args(arg)
        if parsed is None:
            return
        path, explicit_character_id, name_override = parsed
        if explicit_character_id:
            try:
                target = self._resolve_character_ref(explicit_character_id)
            except ValueError as e:
                print(f"error: {e}")
                return
        else:
            target = self.current_actor
        if target is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(target)
        if uid is None:
            print(f"not claimed: {target}")
            return

        try:
            if not path.is_file():
                print(f"error: no such JSON file: {path}")
                return
            size = path.stat().st_size
        except OSError as e:
            print(f"error: {e}")
            return
        if size > MAX_DND_CHARACTER_BYTES:
            print(
                f"error: sheet export is too large ({size} bytes). "
                f"Limit is {MAX_DND_CHARACTER_BYTES} bytes."
            )
            return

        try:
            raw = path.read_bytes()
        except OSError as e:
            print(f"error: could not read {path}: {e}")
            return
        try:
            export = json.loads(raw.decode("utf-8-sig"))
        except UnicodeDecodeError:
            print("error: attachment is not valid UTF-8 JSON")
            return
        except json.JSONDecodeError as e:
            print(f"error: attachment is not valid JSON: {e}")
            return
        if not isinstance(export, dict):
            print("error: D&D Beyond export must be a JSON object")
            return

        try:
            summary = await self.engine.attach_dndbeyond_character_export(
                self.session_id,
                uid,
                export,
                character_id=target,
                name_override=name_override,
            )
        except ValueError as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("D&D sheet attach failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_dnd_attachment_summary(summary)

    def cmd_sheet(self, arg: str) -> None:
        if not self._require_story():
            return
        parts = arg.split()
        if len(parts) > 2:
            print("usage: /sheet [page|all] [character_id]")
            return
        page = "overview"
        target = self.current_actor
        if len(parts) == 1:
            token = parts[0].lower()
            if token in {*DND_SHEET_PAGES, "all"}:
                page = token
            else:
                try:
                    target = self._resolve_character_ref(parts[0])
                except ValueError as e:
                    print(f"error: {e}")
                    return
        elif len(parts) == 2:
            page = parts[0].lower()
            try:
                target = self._resolve_character_ref(parts[1])
            except ValueError as e:
                print(f"error: {e}")
                return
        if target is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(target)
        if uid is None:
            print(f"not claimed: {target}")
            return
        try:
            character = self.engine.get_bound_character_record(
                self.session_id,
                uid,
                character_id=target,
            )
            _print_dnd_sheet(character, page)
        except ValueError as e:
            print(f"error: {e}")
        except Exception as e:
            logger.exception("D&D sheet display failed")
            print(f"error: {type(e).__name__}: {e}")

    def cmd_inventory(self, arg: str) -> None:
        if not self._require_story():
            return
        target_ref = arg.strip()
        if target_ref:
            try:
                target = self._resolve_character_ref(target_ref)
            except ValueError as e:
                print(f"error: {e}")
                return
        else:
            target = self.current_actor
        if target is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(target)
        if uid is None:
            print(f"not claimed: {target}")
            return
        try:
            view = self.engine.list_inventory(
                self.session_id,
                uid,
                character_id=target,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_inventory(view)

    async def cmd_loot(self, arg: str) -> None:
        if not self._require_story():
            return
        parts = arg.split(maxsplit=1) if arg.strip() else []
        if not parts:
            self._print_loot_for_current()
            return
        if parts[0].lower() in {"--help", "-h", "help"}:
            _print_loot_help()
            return
        sub = parts[0].lower().replace("-", "_")
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_loot_{sub}", None)
        if handler is None:
            self._print_loot_offer_detail(" ".join(parts))
            return
        result = handler(rest)
        if asyncio.iscoroutine(result):
            await result

    def _loot_offers_for_actor(self, character_id: str | None):
        if character_id is None:
            print("no current actor — /join a character first")
            return None
        uid = self.claims.get(character_id)
        if uid is None:
            print(f"not claimed: {character_id}")
            return None
        try:
            return self.engine.list_loot_offers(
                self.session_id,
                uid,
                character_id=character_id,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return None

    def _loot_offers_for_current(self):
        return self._loot_offers_for_actor(self.current_actor)

    def _resolve_loot_offer_ref(self, offer_ref: str):
        offers = self._loot_offers_for_current()
        if offers is None:
            return None
        wanted = offer_ref.strip()
        if not wanted:
            raise ValueError("missing loot offer; run /loot to see open offers")
        if wanted.isdigit():
            index = int(wanted)
            if 1 <= index <= len(offers):
                return offers[index - 1]
            raise ValueError(
                f"No loot offer numbered {index}. Run /loot to see open offers."
            )
        exact = [
            offer for offer in offers
            if str(getattr(offer, "offer_id", "") or "") == wanted
        ]
        if exact:
            return exact[0]
        normalized = _normalize_loot_ref(wanted)
        matches = []
        for offer in offers:
            labels = {
                _normalize_loot_ref(getattr(offer, "source_label", "") or ""),
                _normalize_loot_ref(getattr(offer, "source_kind", "") or ""),
            }
            item_ids = {
                _normalize_loot_ref(getattr(item, "item_id", "") or "")
                for item in getattr(offer, "items", []) or []
            }
            if normalized in labels or normalized in item_ids:
                matches.append(offer)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"'{wanted}' matches multiple loot offers; use the number from /loot."
            )
        raise ValueError(
            f"Unknown loot offer '{wanted}'. Run /loot to see numbered offers."
        )

    def _print_loot_offer_detail(self, offer_ref: str) -> None:
        try:
            offer = self._resolve_loot_offer_ref(offer_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        if offer is None:
            return
        print()
        print("--- Loot Offer ---")
        print(_loot_offer_text(offer))
        print()
        print("Use /loot take <offer> [all|item_id[,item_id...]] to claim.")
        print()

    def _print_loot_for_current(self) -> None:
        offers = self._loot_offers_for_current()
        if offers is None:
            return
        _print_loot_offers(offers)

    async def cmd_loot_take(self, arg: str) -> None:
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        parts = arg.split(maxsplit=1)
        if not parts:
            print("usage: /loot take <offer> [all|item_id[,item_id...]]")
            return
        try:
            offer = self._resolve_loot_offer_ref(parts[0])
        except ValueError as e:
            print(f"error: {e}")
            return
        if offer is None:
            return
        offer_id = str(getattr(offer, "offer_id", "") or "")
        item_spec = parts[1].strip() if len(parts) > 1 else "all"
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return
        try:
            take_currency = False
            if item_spec.lower() == "all":
                item_ids = []
                take_currency = True
                take_all_available = True
            else:
                item_ids = [
                    part.strip() for part in item_spec.split(",")
                    if part.strip()
                ]
                take_all_available = False
            result = await self.engine.claim_loot(
                session_id=self.session_id,
                user_id=uid,
                character_id=self.current_actor,
                offer_id=offer_id,
                item_ids=item_ids,
                take_currency=take_currency,
                take_all_available=take_all_available,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        print(result.message)

    async def cmd_loot_all(self, arg: str) -> None:
        if arg.strip():
            print("usage: /loot all")
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return
        try:
            offers = self.engine.list_loot_offers(
                self.session_id,
                uid,
                character_id=self.current_actor,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        if not offers:
            print("no open loot offers for the current actor")
            return

        claimed_any = False
        for offer in offers:
            offer_id = str(getattr(offer, "offer_id", "") or "").strip()
            if not offer_id:
                continue
            try:
                result = await self.engine.claim_loot(
                    session_id=self.session_id,
                    user_id=uid,
                    character_id=self.current_actor,
                    offer_id=offer_id,
                    item_ids=[],
                    take_currency=True,
                    take_all_available=True,
                )
            except Exception as e:
                print(f"error claiming offer: {type(e).__name__}: {e}")
                continue
            print(result.message)
            claimed_any = True
        if not claimed_any:
            print("no claimable loot found")

    async def cmd_loot_split_coins(self, arg: str) -> None:
        offer_ref = arg.strip()
        if not offer_ref:
            print("usage: /loot split-coins <offer>")
            return
        try:
            offer = self._resolve_loot_offer_ref(offer_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        if offer is None:
            return
        offer_id = str(getattr(offer, "offer_id", "") or "")
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return
        try:
            result = await self.engine.split_loot_currency(
                session_id=self.session_id,
                user_id=uid,
                offer_id=offer_id,
                character_id=self.current_actor,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        print(result.message)

    async def cmd_loot_decline(self, arg: str) -> None:
        offer_ref = arg.strip()
        if not offer_ref:
            print("usage: /loot decline <offer>")
            return
        try:
            offer = self._resolve_loot_offer_ref(offer_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        if offer is None:
            return
        offer_id = str(getattr(offer, "offer_id", "") or "")
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return
        try:
            result = await self.engine.decline_loot(
                session_id=self.session_id,
                user_id=uid,
                offer_id=offer_id,
                character_id=self.current_actor,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        print(result.message)

    async def cmd_join(self, arg: str) -> None:
        if not self._require_story():
            return
        parsed = _parse_join_args(arg)
        if parsed is None:
            return
        char_ref, chosen_name, chosen_appearance = parsed
        if not char_ref:
            try:
                summaries = self.engine.list_session_characters(
                    self.session_id
                )
            except Exception as e:
                print(f"error: {type(e).__name__}: {e}")
                return
            _print_joinable_characters(summaries, include_hint=True)
            print("Custom character: /join_custom")
            return
        try:
            char_id = self._resolve_character_ref(char_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        if char_id in self.claims:
            display_name = self._character_name(char_id)
            print(
                f"already claimed: {display_name}. "
                f"/as {display_name} to switch."
            )
            return
        try:
            summary = next(
                item
                for item in self.engine.list_session_characters(
                    self.session_id
                )
                if item.character_id == char_id
            )
        except StopIteration:
            print(f"error: no character: {char_id}")
            return
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        player_authored = summary.player_slot_kind == "player_authored"
        if player_authored and (not chosen_name or not chosen_appearance):
            if self.one_shot_mode:
                print(
                    "this player-authored seat requires both identity fields:\n"
                    f"  /join {char_id} --name \"Your name\" "
                    '--appearance "Your visible appearance"'
                )
                return
            try:
                if not chosen_name:
                    chosen_name = (
                        await self.console.prompt(
                            f"name (must replace {summary.name!r}): "
                        )
                    ).strip()
                if not chosen_appearance:
                    chosen_appearance = (
                        await self.console.prompt(
                            "appearance (required): "
                        )
                    ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(cancelled — character was not claimed)")
                return
        uid = self._next_user_id
        try:
            join_result = await self.engine.join_player_character(
                self.session_id,
                char_id,
                uid,
                name=chosen_name,
                appearance=chosen_appearance,
            )
        except ValueError as e:
            print(f"error: {e}")
            return
        self.claims[char_id] = uid
        self._next_user_id += 1
        if self.current_actor is None:
            self.current_actor = char_id

        try:
            dossier = self.engine.build_character_dossier(
                self.session_id, char_id,
            )
        except Exception as e:
            dossier = f"(dossier unavailable: {e})"
        print()
        print(dossier)
        print()
        display_name = join_result.character_name or summary.name
        if self.current_actor == char_id:
            print(f"claimed {display_name} — now acting as {display_name}.")
        else:
            print(f"claimed {display_name}. /as {display_name} to switch.")
        if join_result.response is not None:
            await self._wait_for_render_images(
                join_result.response,
                actor_id=char_id,
            )
            self._print_turn_response(
                join_result.response,
                actor_id=char_id,
            )

    async def cmd_join_custom(self, arg: str) -> None:
        if not self._require_story():
            return
        if arg.strip():
            print("usage: /join_custom")
            return

        async def _prompt(label: str) -> str | None:
            try:
                raw = await self.console.prompt(label)
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            return raw

        name_resp = await _prompt("character name: ")
        if name_resp is None:
            print("(cancelled)")
            return
        name = name_resp.strip()
        if not name:
            print("(cancelled — empty name)")
            return
        appearance_resp = await _prompt("appearance: ")
        if appearance_resp is None:
            print("(cancelled)")
            return
        appearance = appearance_resp.strip()
        if not appearance:
            print("(cancelled — empty appearance)")
            return
        backstory_resp = await _prompt("backstory (optional): ")
        if backstory_resp is None:
            print("(cancelled)")
            return
        backstory = backstory_resp.strip()
        uid = self._next_user_id
        try:
            new_char = await self.engine.create_player_character_simple(
                self.session_id,
                uid,
                name=name,
                appearance=appearance,
                backstory=backstory,
            )
        except ValueError as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("custom character creation failed")
            print(f"error: {type(e).__name__}: {e}")
            return

        self.claims[new_char.character_id] = uid
        self._next_user_id += 1
        if self.current_actor is None:
            self.current_actor = new_char.character_id
        try:
            dossier = self.engine.build_character_dossier(
                self.session_id, new_char.character_id,
            )
        except Exception as e:
            dossier = f"(dossier unavailable: {e})"
        print()
        print(dossier)
        print()
        if self.current_actor == new_char.character_id:
            print(
                f"created {new_char.character_id} — now acting as "
                f"{new_char.character_id}."
            )
        else:
            print(
                f"created {new_char.character_id}. "
                f"/as {new_char.character_id} to switch."
            )

    async def cmd_leave(self, arg: str) -> None:
        if not self._require_story():
            return
        target_ref = arg.strip()
        if target_ref:
            try:
                target = self._resolve_character_ref(target_ref)
            except ValueError as e:
                print(f"error: {e}")
                return
        else:
            target = self.current_actor
        if target is None:
            print("no character to leave")
            return
        uid = self.claims.get(target)
        if uid is None:
            print(f"not claimed: {target}")
            return

        # Shared endpoint: synthesizes personality (if empty) then unbinds.
        # Both frontends take this path so the agent-handoff behavior is
        # identical whether you leave via CLI or Discord.
        try:
            await self.engine.leave_character(self.session_id, uid)
        except Exception as e:
            print(f"warning: leave failed ({e}); unbinding anyway")
            await self.engine.unbind_user(self.session_id, uid)
        display_name = self._character_name(target)
        del self.claims[target]
        if self.current_actor == target:
            self.current_actor = next(iter(self.claims), None)
        actor_note = (
            f" — now acting as {self._character_name(self.current_actor)}"
            if self.current_actor else " — no current actor"
        )
        print(f"released {display_name}{actor_note}")

    def cmd_as(self, arg: str) -> None:
        if self.one_shot_mode:
            print("/as is unavailable in --command batches; use startup --as")
            return
        target_ref = arg.strip()
        if not target_ref:
            print("usage: /as <name|#>")
            return
        try:
            target = self._resolve_character_ref(target_ref)
        except ValueError as e:
            print(f"error: {e}")
            return
        if target not in self.claims:
            display_name = self._character_name(target)
            print(f"not claimed: {display_name}. /join {display_name} first.")
            return
        self.current_actor = target
        print(f"now acting as {self._character_name(target)}")

    async def cmd_describe(self, arg: str) -> None:
        """Update identity interactively or from explicit command flags."""
        if not self._require_story():
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        parsed = _parse_describe_args(arg)
        if parsed is None:
            return
        name, appearance = parsed
        if not arg.strip():
            if self.one_shot_mode:
                print(
                    "usage: /describe [--name \"Name\"] "
                    "[--appearance \"Description\"]"
                )
                return
            try:
                name = (
                    await self.console.prompt("name (blank to keep existing): ")
                ).strip()
                appearance = (
                    await self.console.prompt(
                        "appearance (blank to keep existing): "
                    )
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(cancelled)")
                return
        if not name and not appearance:
            print("(no changes)")
            return
        try:
            await self.engine.set_character_identity(
                self.session_id, self.current_actor,
                name=name or None, appearance=appearance or None,
            )
        except Exception as e:
            print(f"error: {e}")
            return
        print(f"updated {self.current_actor}")

    async def cmd_query(self, arg: str) -> None:
        """Out-of-character question for the current actor's POV.

        Router-grounded consultation: the engine answers from what THIS
        character can perceive, infer, remember, or legitimately know,
        and records the answer as a private observable fact for this POV.
        """
        if not self._require_story():
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        question = arg.strip()
        if not question:
            print("usage: /query <question>")
            return
        try:
            async with _progress("checking"):
                response = await self.engine.run_query(
                    session_id=self.session_id,
                    character_id=self.current_actor,
                    question=question,
                )
        except Exception as e:
            logger.exception("run_query failed")
            print(f"error: {player_safe_error_message(e, operation='that query')}")
            return
        await self._wait_for_render_images(
            response,
            actor_id=self.current_actor,
        )
        self._print_turn_response(
            response,
            actor_id=self.current_actor,
        )

    def cmd_combat(self, arg: str) -> None:
        if not self._require_story():
            return
        parts = arg.split(maxsplit=1) if arg.strip() else []
        if not parts:
            print(
                "usage: /combat "
                "[begin|status|next|end|damage|heal|add|remove] ..."
            )
            return
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_combat_{sub}", None)
        if handler is None:
            print(f"unknown combat subcommand: {sub}")
            return
        handler(rest)

    def cmd_combat_begin(self, arg: str) -> None:
        participant_ids = _split_combat_ids(arg) or None
        try:
            view = self.engine.begin_combat(self.session_id, participant_ids)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat begin failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_status(self, arg: str) -> None:
        try:
            view = self.engine.combat_status(self.session_id, private=True)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat status failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_next(self, arg: str) -> None:
        try:
            view = self.engine.combat_next(self.session_id)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat next failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_end(self, arg: str) -> None:
        try:
            view = self.engine.combat_end(self.session_id)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat end failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_damage(self, arg: str) -> None:
        parts = arg.split()
        if len(parts) != 2:
            print("usage: /combat damage <combatant_id> <amount>")
            return
        try:
            amount = int(parts[1])
            view = self.engine.combat_damage(self.session_id, parts[0], amount)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat damage failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_heal(self, arg: str) -> None:
        parts = arg.split()
        if len(parts) != 2:
            print("usage: /combat heal <combatant_id> <amount>")
            return
        try:
            amount = int(parts[1])
            view = self.engine.combat_heal(self.session_id, parts[0], amount)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat heal failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_add(self, arg: str) -> None:
        character_id = arg.strip()
        if not character_id:
            print("usage: /combat add <character_id>")
            return
        try:
            view = self.engine.combat_add(self.session_id, character_id)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat add failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    def cmd_combat_remove(self, arg: str) -> None:
        parts = arg.split()
        if not parts:
            print("usage: /combat remove <combatant_id> [hard]")
            return
        hard = len(parts) > 1 and parts[1].lower() == "hard"
        try:
            view = self.engine.combat_remove(
                self.session_id,
                parts[0],
                hard=hard,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        except Exception as e:
            logger.exception("combat remove failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        _print_combat_status(view)
        self._sync_current_actor_to_combat_view(view)

    async def cmd_roll(self, arg: str) -> None:
        if not self._require_story():
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return

        requested = arg.strip()
        printed_pending_from_response = False
        roll_all = requested.lower() == "all"
        if roll_all:
            prompts = self._joined_pending_roll_prompts()
            if not prompts:
                print("no pending D&D rolls for joined characters")
                return
        else:
            prompts = self.engine.pending_roll_prompts(
                self.session_id,
                user_id=uid,
            )
            if not prompts:
                joined_prompts = [
                    p for p in self._joined_pending_roll_prompts()
                    if p.actor_id != self.current_actor
                ]
                if joined_prompts:
                    print(
                        "no pending D&D roll for the current actor; "
                        "joined characters have pending rolls:"
                    )
                    _print_roll_prompts(joined_prompts, include_actor=True)
                else:
                    print("no pending D&D roll for the current actor")
                return

        selected: list[tuple[PendingRollPrompt, int]] = []
        if not requested:
            if len(prompts) == 1:
                selected = [(prompts[0], uid)]
            else:
                _print_roll_prompts(prompts)
                return
        elif roll_all:
            for prompt in prompts:
                prompt_uid = self.claims.get(prompt.actor_id)
                if prompt_uid is None:
                    try:
                        prompt_uid = int(prompt.user_id)
                    except (TypeError, ValueError):
                        print(
                            "error: pending roll has no usable player binding: "
                            f"{prompt.roll_id}"
                        )
                        return
                selected.append((prompt, prompt_uid))
        else:
            chosen = next((p for p in prompts if p.roll_id == requested), None)
            if chosen is None:
                print(f"no pending roll id for {self.current_actor}: {requested}")
                _print_roll_prompts(prompts)
                return
            selected = [(chosen, uid)]

        continued_events: set[str] = set()
        for prompt, prompt_uid in selected:
            try:
                result = await self.engine.complete_pending_roll(
                    session_id=self.session_id,
                    event_id=prompt.event_id,
                    roll_id=prompt.roll_id,
                    user_id=prompt_uid,
                )
            except Exception as e:
                logger.exception("pending roll failed")
                print(f"error: {type(e).__name__}: {e}")
                return
            _print_completed_roll_result(result)

            if result.remaining_pending_rolls > 0:
                continue
            if result.event_id in continued_events:
                continue
            continued_events.add(result.event_id)
            print("Resolving...")
            try:
                response = await self.engine.continue_pending_roll(
                    session_id=self.session_id,
                    event_id=result.event_id,
                    actor_id=result.actor_id,
                )
            except Exception as e:
                logger.exception("pending roll continuation failed")
                print(f"error: {type(e).__name__}: {e}")
                return
            await self._wait_for_render_images(
                response,
                actor_id=result.actor_id,
            )
            self._print_turn_response(
                response,
                actor_id=result.actor_id,
            )
            if response.beat_ended_reason == "cat_ii_pending_rolls":
                printed_pending_from_response = True

        if not printed_pending_from_response:
            remaining = (
                self._joined_pending_roll_prompts()
                if roll_all
                else self.engine.pending_roll_prompts(
                    self.session_id,
                    user_id=uid,
                )
            )
            _print_roll_prompts(remaining, include_actor=roll_all)

    async def cmd_rewind(self, arg: str) -> None:
        """Rewind the session to an earlier checkpoint.

        Usage:
            /rewind                  → list available turn checkpoints
            /rewind <N>              → preview rewind to ckpt_N, then confirm

        The new latest becomes ckpt_<N>; the next /act resumes from
        there as turn N+1. Discord-side bindings persist; players who
        joined after turn N will lose their binding (the loaded
        checkpoint predates their /join) and need to /join again.
        Save files are deleted from disk — there's no undo. Make a
        copy of the session dir first if you want a backup.
        """
        if not self._require_story():
            return

        try:
            turns = self.engine.list_checkpoint_turns(self.session_id)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        if not turns:
            print("no checkpoints to rewind to")
            return

        if not arg.strip():
            print(
                f"available turns: {turns[0]}..{turns[-1]} "
                f"({len(turns)} checkpoints)"
            )
            print(f"current latest: turn {turns[-1]}")
            print("usage: /rewind <N>  (where N < latest)")
            return

        try:
            target = int(arg.strip())
        except ValueError:
            print(f"usage: /rewind <integer turn #>  (got: {arg.strip()!r})")
            return

        try:
            preview = self.engine.preview_rewind(self.session_id, target)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return

        print(
            f"rewinding {self.session_id}: "
            f"turn {preview.previous_latest} → turn {preview.target_turn} "
            f"(deleting turns {preview.deleted_turns})"
        )
        print("This permanently deletes the listed checkpoints. There is no undo.")
        expected = f"rewind {preview.target_turn}"
        try:
            confirmation = await self.console.prompt(
                f"Type {expected!r} to confirm: "
            )
        except (EOFError, KeyboardInterrupt):
            print()
            print("rewind cancelled; nothing was deleted")
            return
        if confirmation.strip() != expected:
            print("rewind cancelled; nothing was deleted")
            return

        try:
            result = await self.engine.rewind_session(
                self.session_id, target,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}")
            return
        location_note = (
            f" — actor {result.actor_character_id} at location {result.location}"
            if result.actor_character_id and result.location
            else ""
        )
        print(
            f"rewound to turn {result.new_latest}; deleted "
            f"{len(result.deleted_turns)} checkpoint(s){location_note}"
        )

    def cmd_settings(self, arg: str) -> None:
        """
        /settings                 → list all
        /settings <key>           → show one value
        /settings <key> <value>   → set (value may be multi-word)
        """
        if not self._require_story():
            return
        parts = arg.split(maxsplit=1) if arg.strip() else []

        if not parts:
            view = self.engine.list_settings(self.session_id)
            if not view:
                print("(no tunable settings registered)")
                return
            for s in view:
                modified = "" if s["value"] == s["default"] else "  (modified)"
                print(
                    f"  {s['key']} = {s['rendered_value']}  "
                    f"[default: {s['rendered_default']}]{modified}"
                )
                print(f"    {s['description']}")
            return

        key = parts[0]
        if len(parts) == 1:
            try:
                value = self.engine.get_setting(self.session_id, key)
            except KeyError:
                valid = ", ".join(self.engine.known_setting_keys()) or "(none)"
                print(f"unknown setting: {key}. valid: {valid}")
                return
            print(f"{key} = {value}")
            return

        raw_value = parts[1]
        try:
            new_value = self.engine.set_setting(self.session_id, key, raw_value)
        except KeyError:
            valid = ", ".join(self.engine.known_setting_keys()) or "(none)"
            print(f"unknown setting: {key}. valid: {valid}")
            return
        except ValueError as e:
            print(f"error: {e}")
            return
        print(f"{key} = {new_value}")

    def cmd_history(self, arg: str) -> None:
        """Print the transcript. With no arg, emits every turn; with an
        integer N, emits the last N turns."""
        if not self._require_story():
            return
        pov_character_id = self.pov_filter or self.current_actor
        if not pov_character_id:
            print("select a character POV with /as (or startup --as) first")
            return
        try:
            history = self.engine.turn_history(
                self.session_id,
                pov_character_id,
            )
        except ValueError as exc:
            print(f"error: {exc}")
            return
        if not history:
            print("(no turns yet)")
            return

        limit: int | None = None
        raw = arg.strip()
        if raw:
            try:
                limit = int(raw)
            except ValueError:
                print(f"usage: /history [N]  (N is an integer, got {raw!r})")
                return
            if limit <= 0:
                print("usage: /history [N]  (N must be positive)")
                return

        entries = history[-limit:] if limit else history
        for item in entries:
            entry = item.entry
            print()
            print(f"--- Turn {item.turn_index} ---")
            if entry.user:
                print(f"> {entry.user}")
            print(entry.assistant)
        print()

    def cmd_quit(self, arg: str) -> None:
        self.running = False

    cmd_exit = cmd_quit  # alias

    # ---- action pipeline -----------------------------------------------------

    async def cmd_defer(self, arg: str) -> None:
        if self.story_id and self.current_actor is not None:
            event_id = ""
            try:
                event_id = self.engine.combat_reaction_prompt_event(
                    self.session_id,
                    self.current_actor,
                )
            except Exception:
                logger.exception("combat reaction lookup failed")
            if event_id:
                uid = self.claims.get(self.current_actor)
                try:
                    response = await self.engine.defer_combat_reaction(
                        session_id=self.session_id,
                        character_id=self.current_actor,
                        event_id=event_id,
                        user_id=uid,
                    )
                except Exception as e:
                    logger.exception("combat reaction defer failed")
                    print(f"error: {type(e).__name__}: {e}")
                    return
                await self._wait_for_render_images(
                    response,
                    actor_id=self.current_actor,
                )
                self._print_turn_response(
                    response,
                    actor_id=self.current_actor,
                )
                return
        await self._act("(defer)")

    async def _act(self, text: str) -> None:
        if not self._require_story():
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        try:
            async with _progress("resolving"):
                response = await self.engine.run_turn(
                    session_id=self.session_id,
                    user_input=text,
                    acting_character_id=self.current_actor,
                )
        except Exception as e:
            logger.exception("run_turn failed")
            print(f"error: {player_safe_error_message(e)}")
            return

        await self._wait_for_render_images(
            response,
            actor_id=self.current_actor,
        )
        self._print_turn_response(
            response,
            actor_id=self.current_actor,
        )

    def _pov_claims(self) -> set[str]:
        """Which claimed characters' POVs this terminal should print.

        Defaults to every locally claimed character (single-terminal play).
        In separate-terminal one-shot mode `pov_filter` narrows it to the one
        character this terminal represents, so other players' POV prose does
        not leak into this terminal."""
        if self.pov_filter:
            return {self.pov_filter} if self.pov_filter in self.claims else set()
        return set(self.claims or {})

    async def _wait_for_render_images(
        self,
        response,
        *,
        actor_id: str,
    ) -> None:
        if not bool(
            getattr(
                getattr(self.engine.image_sidecar, "config", None),
                "diffusion_enabled",
                False,
            )
        ):
            return
        responses = [
            *(getattr(response, "pre_turn_resolutions", None) or []),
            response,
        ]
        for item in responses:
            rendered = self._rendered_event_ids_for_visible_prose(
                item,
                actor_id=actor_id,
            )
            if not any(rendered.values()):
                continue
            if not any(
                self.engine.image_generation.can_accept_render(
                    ImageDeliveryKind.cli,
                    session_id=self.session_id,
                    pov_character_id=pov_character_id,
                )
                for pov_character_id in rendered
            ):
                continue
            try:
                async with _progress("illustrating"):
                    await self.engine.image_generation.wait_for_render_images(
                        session_id=self.session_id,
                        rendered_event_ids_by_pov=rendered,
                    )
            except Exception:
                logger.exception("render image wait failed")

    def _rendered_event_ids_for_visible_prose(
        self,
        response,
        *,
        actor_id: str,
    ) -> dict[str, list[str]]:
        rendered_ids = getattr(response, "rendered_event_ids_by_pov", {}) or {}
        visible_povs = {
            cid
            for cid, prose in (getattr(response, "per_player_renders", {}) or {}).items()
            if prose and (cid == actor_id or cid in self._pov_claims())
        }
        if getattr(response, "output_text", "") and actor_id:
            visible_povs.add(actor_id)
        return {
            cid: list(rendered_ids.get(cid, []))
            for cid in visible_povs
            if rendered_ids.get(cid)
        }

    def _print_turn_response(
        self,
        response,
        *,
        actor_id: str,
    ) -> None:
        actor_name = self._character_name(actor_id) if actor_id else "player"
        # v11-r6b/r7a: mirror the Discord bot's /act branching so the
        # CLI playtest path surfaces paused beats and slot rejections
        # with targeted messages, AND prints the PARTIAL cliffhanger
        # render when one is available (cat_ii_pending).
        if response.beat_ended_reason == "slot_rejected":
            print(response.output_text)
            self._sync_current_actor_to_active_combat()
            return
        if response.beat_ended_reason == "combat_start_blocked_deferred":
            print(response.output_text)
            self._sync_current_actor_to_active_combat()
            return

        # Print pre-turn resolutions first so in-CLI ordering matches
        # story time. These can come from stale Cat II closure or from
        # resumed automated combat after a rewind.
        for pre_resp in (response.pre_turn_resolutions or []):
            _print_dice_roll_displays(getattr(pre_resp, "dice_rolls", []) or [])
            _print_experience_awards(
                getattr(pre_resp, "experience_awards", []) or []
            )
            for cid, prose in (pre_resp.per_player_renders or {}).items():
                if not prose or cid not in self._pov_claims():
                    continue
                print(
                    "--- Earlier story update · viewed as "
                    f"{self._character_name(cid)} ---"
                )
                print(prose)
                print()
            self._print_asset_reveals(pre_resp)
            self._print_loot_prompts(pre_resp)
            self._print_commitment_revision_prompts(pre_resp)

        per_player = response.per_player_renders or {}
        _print_dice_roll_displays(getattr(response, "dice_rolls", []) or [])
        _print_experience_awards(getattr(response, "experience_awards", []) or [])

        if response.beat_ended_reason == "pre_turn_resolution":
            print(response.output_text)
            self._sync_current_actor_to_active_combat()
            return

        if response.beat_ended_reason == "cat_ii_pending":
            actor_render = per_player.get(actor_id) or ""
            print()
            self._print_cat_ii_pending_notice()
            if actor_render:
                print()
                print(
                    f"--- Story update {response.turn_index} · viewed as "
                    f"{actor_name} (partial) ---"
                )
                print(actor_render)
            print()
        elif response.beat_ended_reason == "combat_start_blocked":
            print()
            print(
                "(that hostile action could not start a second D&D combat "
                "while another combat is already in initiative; revise and "
                "act normally.)"
            )
            print()
            print(
                f"--- Story update {response.turn_index} · viewed as "
                f"{actor_name} ---"
            )
            print(response.output_text)
            print()
        else:
            print()
            print(
                f"--- Story update {response.turn_index} · viewed as "
                f"{actor_name} ---"
            )
            print(response.output_text)
            print()

        self._print_combat_started_notice(response)

        # Print POVs for any other characters this CLI session has
        # /join'd locally so the playtester sees the multi-POV output
        # without spinning up Discord.
        joined = self._pov_claims()
        for cid, prose in per_player.items():
            if cid == actor_id or not prose or cid not in joined:
                continue
            print(
                "--- Same story update · viewed as "
                f"{self._character_name(cid)} ---"
            )
            print(prose)
            print()

        self._print_asset_reveals(response)

        if response.beat_ended_reason == "cat_ii_pending_rolls":
            prompts = self._joined_pending_roll_prompts()
            _print_roll_prompts(prompts)

        self._print_reaction_prompts(response)
        self._print_loot_prompts(response)
        self._print_commitment_revision_prompts(response)
        self._sync_current_actor_to_active_combat()
        rendered_ids = (
            getattr(response, "rendered_event_ids_by_pov", {}) or {}
        )
        delivered_povs = {
            cid
            for cid, prose in (response.per_player_renders or {}).items()
            if prose and (cid == actor_id or cid in self._pov_claims())
        }
        self.engine.image_generation.open_prose_gates_for_session(
            session_id=self.session_id,
            rendered_event_ids_by_pov={
                cid: rendered_ids.get(cid, [])
                for cid in delivered_povs
            },
        )

    async def _deliver_cli_image(
        self,
        job,
        delivery,
        media,
        instructions: str,
    ) -> bool:
        if (
            str(getattr(job.request, "session_id", "") or "")
            != self.session_id
            or str(getattr(delivery, "session_id", "") or "")
            != self.session_id
        ):
            return False
        actor_id = str(delivery.pov_character_id or "")
        if actor_id not in self._pov_claims():
            return False
        try:
            prepared = self.asset_image_renderer.prepare_generated(
                media,
                session_id=self.session_id,
                pov_character_id=actor_id,
                cache_root=(
                    self.engine.image_generation.config.runtime_root
                    / "cli_cache"
                ),
            )
            title = str(getattr(job.request, "title", "") or "AI Illustration")
            print(f"\n--- AI Illustration · {title} · {actor_id} ---")
            result = self.asset_image_renderer.render_prepared(prepared)
            print(_cli_image_display_message(result))
            if instructions:
                print(instructions)
            print()
            return bool(result.displayed or result.export_path is not None)
        except Exception:
            logger.exception("CLI illustration display failed")
            return False

    def _print_asset_reveals(self, response) -> None:
        per_pov = getattr(response, "per_player_asset_reveals", None) or {}
        if not per_pov or not self.claims:
            return
        claimed_ids = {cid for cid in per_pov if cid in self._pov_claims()}
        if not claimed_ids:
            return
        try:
            ckpt = self.engine.load_latest(self.session_id)
            prepared = self.asset_image_renderer.prepare_reveals(
                response,
                ckpt=ckpt,
                session_id=self.session_id,
                character_ids=claimed_ids,
            )
        except Exception:
            logger.exception("asset reveal preparation failed")
            for cid in sorted(claimed_ids):
                print(f"--- Image Reveal · {cid} ---")
                print("Could not display the revealed image.")
                print()
            return

        for cid in sorted(prepared):
            print(f"--- Image Reveal · {cid} ---")
            for item in prepared[cid]:
                result = self.asset_image_renderer.render_prepared(item)
                print(_cli_image_display_message(result))
            print()

    def _print_cat_ii_pending_notice(self) -> None:
        try:
            ckpt = self.engine.load_latest(self.session_id)
        except Exception:
            logger.debug("cat ii pending slot lookup failed", exc_info=True)
            print(
                "(beat paused — waiting on another character to resolve "
                "a contested action.)"
            )
            return

        pending: list[str] = []
        for cid in self.claims:
            slot = ckpt.session.active_act_slots.get(cid)
            if slot is None or slot.reason != "cat_ii_responder":
                continue
            pending.append(cid)

        if not pending:
            print(
                "(beat paused — waiting on another character to resolve "
                "a contested action.)"
            )
            return

        if len(pending) == 1:
            cid = pending[0]
            label = self._character_label(ckpt, cid)
            if cid != self.current_actor:
                self.current_actor = cid
                print(
                    f"(beat paused — waiting on {label}. Switched to "
                    f"{cid}; type their response to continue.)"
                )
                return
            print(
                f"(beat paused — waiting on {label}. Type their response "
                "to continue.)"
            )
            return

        labels = ", ".join(self._character_label(ckpt, cid) for cid in pending)
        print(
            f"(beat paused — waiting on: {labels}. Use /as <character_id>, "
            "then type a response.)"
        )

    def _print_combat_started_notice(self, response) -> None:
        if response.beat_ended_reason != "combat_started":
            return
        try:
            view = self.engine.combat_status(self.session_id, private=True)
        except Exception:
            logger.debug("combat-start status lookup failed", exc_info=True)
            print("Combat started.")
            return

        if not view.active:
            print("Combat started.")
            return

        parts = ["=== COMBAT BEGINS ==="]
        initiative = _combat_initiative_line(view)
        if initiative:
            parts.append(f"Initiative: {initiative}.")
        current = next(
            (
                participant for participant in view.participants
                if (
                    participant.current
                    or participant.character_id == view.current_participant_id
                )
            ),
            None,
        )
        if current is not None:
            parts.append(f"Current turn: {_combat_participant_label(current)}.")
        elif view.current_participant_id:
            parts.append(f"Current turn: {view.current_participant_id}.")
        print(" ".join(parts))
        print()

    def _print_reaction_prompts(self, response) -> None:
        prompts = getattr(response, "reaction_prompts", None) or {}
        for cid, event_id in prompts.items():
            if cid not in self.claims:
                continue
            print(f"--- Reaction Available · {cid} ---")
            print(
                f"event {event_id}; /as {cid}, then type a reaction "
                "or use /defer to pass"
            )
            print()

    def _print_loot_prompts(self, response) -> None:
        prompts = getattr(response, "loot_prompts", None) or {}
        for cid, offer_ids in prompts.items():
            if cid not in self.claims:
                continue
            print(f"--- Loot Available · {cid} ---")
            wanted = set(offer_ids)
            offers = self._loot_offers_for_actor(cid) or []
            shown = 0
            for index, offer in enumerate(offers, start=1):
                if str(getattr(offer, "offer_id", "") or "") not in wanted:
                    continue
                label = getattr(offer, "source_label", "") or getattr(
                    offer, "source_kind", "loot",
                )
                print(f"offer {index}: {label}")
                shown += 1
            if not shown:
                print("new loot offer available")
            print("Use /loot to inspect, /loot take <offer> all to claim.")
            print()

    def _print_commitment_revision_prompts(self, response) -> None:
        prompts = getattr(response, "commitment_revision_prompts", None) or {}
        for cid, commitment_ids in prompts.items():
            if cid not in self.claims or not commitment_ids:
                continue
            print(f"--- Commitment Interrupted · {cid} ---")
            for commitment_id in commitment_ids:
                print(f"commitment {commitment_id}")
            print(
                f"Use /as {cid}, then type a revised action or "
                "(continue) to keep going."
            )
            print()

    def _joined_pending_roll_prompts(self) -> list[PendingRollPrompt]:
        prompts: list[PendingRollPrompt] = []
        seen: set[tuple[str, str]] = set()
        for uid in set(self.claims.values()):
            for prompt in self.engine.pending_roll_prompts(
                self.session_id,
                user_id=uid,
            ):
                key = (prompt.event_id, prompt.roll_id)
                if key in seen:
                    continue
                seen.add(key)
                prompts.append(prompt)
        return prompts

    def _print_open_reaction_slots(self) -> None:
        if not self.claims:
            return
        ckpt = self.engine.load_latest(self.session_id)
        open_slots: list[tuple[str, str]] = []
        for cid in self.claims:
            slot = ckpt.session.active_act_slots.get(cid)
            if slot is None or slot.reason != "combat_reaction":
                continue
            open_slots.append(
                (cid, slot.trigger_event_id or slot.cat_ii_event_id)
            )
        if not open_slots:
            return
        print("reactions:")
        for cid, event_id in open_slots:
            marker = "  ← acting" if cid == self.current_actor else ""
            print(f"  - {cid}: {event_id or '(event unknown)'}{marker}")

    def _character_label(self, ckpt, character_id: str) -> str:
        for character in getattr(ckpt, "characters", []) or []:
            if character.character_id != character_id:
                continue
            name = character.name or character_id
            if name == character_id:
                return character_id
            return f"{name} ({character_id})"
        return character_id

    def _sync_current_actor_to_active_combat(
        self,
        *,
        announce: bool = True,
    ) -> None:
        if not self.story_id or not self.claims:
            return
        try:
            view = self.engine.combat_status(self.session_id, private=True)
        except Exception:
            logger.debug("combat current actor sync failed", exc_info=True)
            return
        self._sync_current_actor_to_combat_view(view, announce=announce)

    def _sync_current_actor_to_combat_view(
        self,
        view: DndCombatView,
        *,
        announce: bool = True,
    ) -> None:
        if not view.active:
            return
        current_id = view.current_participant_id
        if not current_id or current_id not in self.claims:
            return
        if current_id == self.current_actor:
            return
        self.current_actor = current_id
        if announce:
            print(f"now acting as {current_id} (current initiative)")


# ---- bootstrap --------------------------------------------------------------


def _cli_log_level(verbose: bool) -> int:
    return logging.INFO if verbose else logging.ERROR


def _configure_cli_logging(*, verbose: bool) -> None:
    level = _cli_log_level(verbose)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(logs_dir / "play_cli.log", encoding="utf-8"),
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    warnings.filterwarnings(
        "default" if verbose else "ignore",
        message=r".*thinking\.type=enabled.*",
        category=UserWarning,
    )


def _format_missing_llm_credentials(
    missing: tuple[MissingLLMCredential, ...],
) -> str:
    lines = ["Missing LLM credentials for live CLI play:"]
    for item in missing:
        env_names = ", ".join(item.env_names)
        lines.append(f"  - {item.role} ({item.provider}): set one of {env_names}")
    return "\n".join(lines)


def _stored_session_ids(sessions_dir: Path = Path("app/storage/sessions")) -> list[str]:
    if not sessions_dir.exists():
        return []
    return sorted(
        path.name
        for path in sessions_dir.iterdir()
        if path.is_dir()
    )


@contextlib.contextmanager
def _session_command_lock(sessions_dir: Path, session_id: str):
    """Serialize mutating CLI commands across processes sharing a session.

    Separate terminals (or agents) driving different characters each run their
    own process against the same session directory. Checkpoint writes are
    atomic (mkstemp + os.replace), but two processes computing the next turn
    from the same base checkpoint could still clobber each other. An advisory
    exclusive flock on a per-session lockfile makes the load->turn->save
    critical section mutually exclusive. No-op when fcntl is unavailable."""
    if _fcntl is None:  # pragma: no cover - platform dependent.
        yield
        return
    session_dir = Path(sessions_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir / ".session.lock"
    with open(lock_path, "w") as handle:
        try:
            _fcntl.flock(
                handle.fileno(),
                _fcntl.LOCK_EX | _fcntl.LOCK_NB,
            )
        except BlockingIOError:
            print("another action is resolving; waiting...")
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


def _command_requires_session_lock(line: str) -> bool:
    """Classify player-safe read commands without duplicating engine logic."""
    raw = strip_terminal_control(line).strip()
    if not raw:
        return False
    if not raw.startswith("/"):
        return True
    parts = raw[1:].split()
    command = parts[0].lower() if parts else ""
    arguments = [part.lower() for part in parts[1:]]
    if command in {
        "help", "status", "history", "characters", "character",
        "sheet", "inventory",
    }:
        return False
    if command == "story":
        return bool(arguments and arguments[0] not in {"list", "info"})
    if command == "session":
        return bool(arguments and arguments[0] != "list")
    if command == "combat":
        return not arguments or arguments[0] != "status"
    if command == "loot":
        return bool(
            arguments
            and arguments[0] in {"all", "take", "split-coins", "decline"}
        )
    if command == "settings":
        return len(arguments) > 1
    return True


async def run_oneshot_commands(
    state: "CLIState",
    *,
    sessions_dir: Path,
    session_id: str,
    commands: list[str],
    act_as: str | None = None,
) -> int:
    """Execute one or more REPL command lines non-interactively, then return.

    Used by separate terminals/agents to drive a single bound character per
    invocation. `act_as` selects which already-bound character acts; the whole
    batch runs under the cross-process session lock so it cannot race another
    terminal's turn."""
    state.one_shot_mode = True
    if any(
        strip_terminal_control(command).strip().lower().startswith("/as ")
        or strip_terminal_control(command).strip().lower() == "/as"
        for command in commands
    ):
        print(
            "/as is unavailable in --command batches; use startup --as",
            file=sys.stderr,
        )
        return 2

    needs_lock = any(_command_requires_session_lock(command) for command in commands)
    lock_context = (
        _session_command_lock(sessions_dir, session_id)
        if needs_lock else contextlib.nullcontext()
    )
    with lock_context:
        state.refresh_from_checkpoint()
        if act_as:
            try:
                resolved_actor = state._resolve_character_ref(act_as)
            except ValueError as exc:
                print(f"--as: {exc}", file=sys.stderr)
                return 2
            if resolved_actor not in state.claims:
                bound_names = [
                    state._character_name(character_id)
                    for character_id in state.claims
                ]
                print(
                    f"--as {act_as}: not bound in session '{session_id}'. "
                    f"Bound: {', '.join(bound_names) or '(none)'}",
                    file=sys.stderr,
                )
                return 2
            state.current_actor = resolved_actor
            state.pov_filter = resolved_actor
        elif len(state.claims) > 1:
            print(
                "--as is required because this session has multiple claimed "
                "seats.",
                file=sys.stderr,
            )
            return 2
        elif len(state.claims) == 1:
            state.current_actor = next(iter(state.claims))
            state.pov_filter = state.current_actor
        for command in commands:
            await state.handle_line(command)
    return 0


def _prepare_session_story(
    engine: EngineBridge,
    *,
    session_id: str,
    requested_story: str = "",
    announce: bool = True,
) -> tuple[str, bool]:
    """Create/resume a session and apply `--story` under one file lock."""
    with _session_command_lock(engine.sessions_dir, session_id):
        story_id = ""
        resumed_existing = False
        try:
            ckpt = engine.load_latest(session_id)
            story_id = ckpt.session.story_id or ""
            resumed_existing = True
            if announce:
                print(f"resumed session `{session_id}`" + (
                    f" · story `{story_id}`" if story_id else " (no story loaded)"
                ))
        except FileNotFoundError:
            engine.create_empty_session(session_id)
            if announce:
                print(
                    f"created session `{session_id}` "
                    "(empty - /story start to load content)"
                )

        story_ref = requested_story.strip()
        if story_ref:
            selected_story = _resolve_enum_ref(
                story_ref,
                engine.list_story_ids(),
                label="story",
            )
            if story_id and story_id != selected_story:
                raise ValueError(
                    f"session `{session_id}` already contains story "
                    f"`{story_id}`; refusing to replace it with "
                    f"`{selected_story}`"
                )
            if not story_id:
                engine.load_story_into_session(session_id, selected_story)
                story_id = selected_story
                if announce:
                    print(
                        f"loaded story `{story_id}` into session `{session_id}`"
                    )
        return story_id, resumed_existing


async def main_async(args: argparse.Namespace) -> int:
    _configure_cli_logging(verbose=args.verbose)

    session_id: str | None = args.session

    # TODO: interactive menu when --session isn't provided. For now
    # we require the flag so the script stays scriptable.
    if not session_id:
        print(
            "--session <name> is required. Pick an existing save or a new "
            f"name.\nExisting sessions: "
            f"{', '.join(_stored_session_ids()) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    llm_config = LLMConfig.from_env()
    missing_credentials = llm_config.missing_credentials(
        live_play_required_roles()
    )
    if missing_credentials:
        print(_format_missing_llm_credentials(missing_credentials), file=sys.stderr)
        return 2

    engine = EngineBridge(
        llm_config=llm_config,
        image_delivery_kind=ImageDeliveryKind.cli,
    )

    try:
        try:
            story_id, resumed_existing = _prepare_session_story(
                engine,
                session_id=session_id,
                requested_story=str(getattr(args, "story", "") or ""),
                announce=args.commands is None,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        image_renderer = CliImageDisplayRenderer.from_environment(
            show_export_path=args.show_image_cache_paths,
        )
        console = _ConsoleInput(history_path=_default_history_path())
        state = CLIState(
            engine,
            session_id,
            story_id,
            console=console,
            asset_image_renderer=image_renderer,
        )
        await engine.start()

        if args.commands is not None:
            # Non-interactive one-shot: run the given command(s) against this
            # session under the cross-process lock, then exit. This is how
            # separate terminals/agents each drive one bound character without
            # sharing a single in-process REPL.
            return await run_oneshot_commands(
                state,
                sessions_dir=engine.sessions_dir,
                session_id=session_id,
                commands=args.commands,
                act_as=args.act_as,
            )

        if args.act_as:
            try:
                resolved_actor = state._resolve_character_ref(args.act_as)
            except ValueError as exc:
                print(f"--as: {exc}", file=sys.stderr)
                return 2
            if resolved_actor not in state.claims:
                bound_names = [
                    state._character_name(character_id)
                    for character_id in state.claims
                ]
                print(
                    f"--as {args.act_as}: not bound in session "
                    f"'{session_id}'. Bound: "
                    f"{', '.join(bound_names) or '(none)'}",
                    file=sys.stderr,
                )
                return 2
            state.current_actor = resolved_actor

        console.install()
        print()
        if resumed_existing:
            print("Type /help for commands.")
        else:
            print(HELP_TEXT)
        print()

        while state.running:
            actor = state.current_actor or "-"
            try:
                # input() is blocking — use run_in_executor so Ctrl-C during
                # a turn doesn't wedge; and it lets async turns interleave
                # cleanly with stdin reads.
                line = await console.prompt(
                    f"[{actor}] > ",
                    record_history=True,
                )
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            try:
                lock_context = (
                    _session_command_lock(engine.sessions_dir, session_id)
                    if _command_requires_session_lock(line)
                    else contextlib.nullcontext()
                )
                with lock_context:
                    state.refresh_from_checkpoint()
                    await state.handle_line(line)
            except KeyboardInterrupt:
                print("\n(interrupted)")
    finally:
        await engine.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI frontend for the narrative engine.",
    )
    parser.add_argument(
        "--session",
        help="Session name (directory under sessions/). Created empty if "
             "new, resumed if existing.",
    )
    parser.add_argument(
        "--story",
        metavar="STORY_ID_OR_NUMBER",
        help=(
            "Load this story into a new or empty session before entering the "
            "REPL or running --command. Refuses to replace a different story."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Log engine INFO and WARNING messages to stderr.",
    )
    parser.add_argument(
        "--as",
        dest="act_as",
        metavar="CHARACTER_ID",
        help="Act as an already-bound character. Sets the default actor for "
             "--command mode and for the interactive REPL.",
    )
    parser.add_argument(
        "--command",
        action="append",
        dest="commands",
        metavar="LINE",
        help="Run one REPL command line non-interactively, then exit. "
             "Repeatable; lines run in order. A line without a leading '/' is "
             "an in-character action. Enables separate-terminal multiplayer: "
             "each terminal/agent drives one bound character per invocation.",
    )
    parser.add_argument(
        "--show-image-cache-paths",
        action="store_true",
        help=(
            "When terminal image display is unavailable, show the generated "
            "player-safe cache path for eligible CLI image reveals."
        ),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
