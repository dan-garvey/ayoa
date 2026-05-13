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

Commands inside the REPL:
    /help                       Show commands
    /story list                 List available stories
    /story start <id>           Load a story into this session
    /story info <id>            Show briefing for a story
    /story delete               Unload the current story (leaves session empty)
    /session list               List existing sessions
    /session end                Exit the REPL (save files stay on disk)
    /characters                 List roster grouped by location/status
    /character <id>             Full dossier for one character
    /join <character_id>        Claim a character; prints their dossier
    /join_custom [mode]         Play a custom character (describe/replace)
    /leave [character_id]       Release a claim (default: current actor)
    /as <character_id>          Switch which claimed character acts next
    /describe                   Set name + appearance of the current actor
    /defer                      Submit no action and let the scene continue
    /inventory [character_id]   Show current D&D inventory
    /loot                       List open D&D loot offers
    /roll [roll_id|all]         Roll pending D&D player check(s)
    /combat status              Show active D&D combat order and HP
    /query <question>           Ask an out-of-character question (POV-bounded)
    /rewind [<N>]               List or rewind to a turn checkpoint
    /settings                   Show / update experimental settings
    /status                     Session summary
    /history [N]                Print all turns, or last N
    /quit                       Exit (Ctrl-D also works)

Anything not starting with '/' is an in-character action for the current
actor. Use "(begin)" to open the story — the router composes the opening
from world_state + the initial roster on the fly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.bot.engine_bridge import (
    CompletedPendingRoll,
    DndCombatParticipantView,
    DndCombatView,
    DndInventoryView,
    EngineBridge,
    PendingRollPrompt,
)
from app.engine import dnd_inventory

load_dotenv()
logger = logging.getLogger(__name__)


HELP_TEXT = """\
Commands:
  /help                             Show this help
  /story list                       List available stories
  /story start <id>                 Load a story into this session
  /story info <id>                  Show briefing for a story
  /story delete                     Unload the current story from this session
  /session list                     List existing sessions
  /session end                      Exit the REPL (files stay)
  /characters                       List roster grouped by location/status
  /character <id>                   Full dossier for a character
  /join <character_id>              Claim a character; prints their dossier
  /join_custom [describe|replace]   Play a character of your own description
  /leave [character_id]             Release a claim (default: current actor)
  /as <character_id>                Switch which claimed character acts next
  /describe                         Set name + appearance of the current actor
  /defer                            Submit no action and let the scene continue
  /inventory [character_id]         Show current D&D inventory
  /loot                             List open D&D loot offers
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
  /rewind                           List available turn checkpoints
  /rewind <N>                       Rewind to ckpt_N (deletes ckpt_>N — no undo)
  /settings                         Show experimental settings for this session
  /settings <key>                   Show one setting's current value
  /settings <key> <value>           Update a setting
  /status                           Session summary
  /history [N]                      Print all turns, or last N
  /quit                             Exit (Ctrl-D also works)

Plain text is an in-character action for the current actor. Use "(begin)"
to open the story — the router composes the opening from world_state +
the initial roster on the fly."""


def _roll_result_line(result: CompletedPendingRoll) -> str:
    crit_note = ""
    if result.crit == "crit":
        crit_note = " Critical success."
    elif result.crit == "fail":
        crit_note = " Natural 1."
    detail = result.detail or f"{result.expression} = `{result.total}`"
    return f"Rolled {result.label}: {detail}.{crit_note}"


def _print_roll_prompts(prompts: list[PendingRollPrompt]) -> None:
    if not prompts:
        return
    print()
    print("--- Pending D&D Rolls ---")
    for prompt in prompts:
        reason = f" — {prompt.reason}" if prompt.reason else ""
        print(f"  {prompt.roll_id}: {prompt.label}{reason}")
    if len(prompts) == 1:
        print("Use /roll to roll it, or /roll <roll_id>.")
    else:
        print("Use /roll <roll_id> for one roll, or /roll all.")
    print()


def _print_combat_status(view: DndCombatView) -> None:
    print()
    print("--- Combat ---")
    if not view.active:
        print(view.message or "No active combat.")
        print()
        return

    if view.round_number:
        header = f"Round {view.round_number}"
        if view.turn_number:
            header += f" · Turn {view.turn_number}"
        print(header)
    if view.message:
        print(view.message)

    current_id = view.current_participant_id
    if not view.participants:
        print("(no participants)")
        print()
        return

    for participant in view.participants:
        marker = ">" if (
            participant.current or participant.character_id == current_id
        ) else "-"
        bits: list[str] = []
        if participant.hp_current is not None:
            hp_text = str(participant.hp_current)
            if participant.hp_max is not None:
                hp_text += f"/{participant.hp_max}"
            if participant.hp_temporary:
                hp_text += f" (+{participant.hp_temporary})"
            bits.append(f"HP {hp_text}")
        if participant.armor_class is not None:
            bits.append(f"AC {participant.armor_class}")
        if participant.initiative is not None:
            bits.append(f"Init {participant.initiative}")
        if participant.defeat_state != "active":
            state = participant.defeat_state
            if state == "down":
                state += (
                    f" ({participant.death_save_successes}S/"
                    f"{participant.death_save_failures}F)"
                )
            bits.append(state)
        if participant.conditions:
            bits.append(", ".join(participant.conditions))
        if participant.active_effects:
            bits.append(f"Effects: {', '.join(participant.active_effects)}")
        if participant.current and participant.pending_initiating_action:
            bits.append(f"Declared: {participant.pending_initiating_action}")
        suffix = f" - {'; '.join(bits)}" if bits else ""
        print(
            f"{marker} {participant.name} "
            f"({participant.character_id}){suffix}"
        )
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


def _print_loot_offers(offers) -> None:
    print()
    print("--- Loot Offers ---")
    if not offers:
        print("(none)")
        print()
        return
    for offer in offers:
        print(_loot_offer_text(offer))
        print()
    print("Use /loot take <offer_id> <all|item_id[,item_id...]>.")
    print("Use /loot split-coins <offer_id> or /loot decline <offer_id>.")
    print()


def _loot_offer_text(offer) -> str:
    label = offer.source_label or offer.source_kind
    lines = [f"{label} [{offer.offer_id}]"]
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


def _inventory_item_line(item: dict) -> str:
    qty = _safe_int(item.get("quantity"), 1)
    prefix = f"{qty}x " if qty != 1 else ""
    kind = str(item.get("kind") or "").replace("_", " ")
    item_id = str(item.get("id") or item.get("item_id") or "")
    suffix_bits = [kind, item_id]
    suffix = " (" + ", ".join(bit for bit in suffix_bits if bit) + ")"
    return f"{prefix}{item.get('name') or 'Item'}{suffix if suffix != ' ()' else ''}"


def _loot_item_line(item) -> str:
    qty = _safe_int(getattr(item, "quantity", 1), 1)
    prefix = f"{qty}x " if qty != 1 else ""
    kind = str(getattr(item, "kind", "") or "").replace("_", " ")
    suffix = f" ({kind})" if kind else ""
    notes = str(getattr(item, "notes", "") or "").strip()
    if notes:
        suffix += f" - {notes}"
    return f"{prefix}{getattr(item, 'name', 'Item')}{suffix}"


def _coin_line(currency: dict) -> str:
    parts = []
    for key in ("pp", "gp", "ep", "sp", "cp"):
        value = _safe_int(currency.get(key), 0)
        if value:
            parts.append(f"{value} {key}")
    return ", ".join(parts)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _combat_participant_label(participant: DndCombatParticipantView) -> str:
    if participant.name and participant.name != participant.character_id:
        return f"{participant.name} ({participant.character_id})"
    return participant.character_id


def _combat_initiative_line(view: DndCombatView) -> str:
    parts: list[str] = []
    for participant in view.participants:
        label = _combat_participant_label(participant)
        if participant.initiative is not None:
            label = f"{label} {participant.initiative}"
        parts.append(label)
    return ", ".join(parts)


def _split_combat_ids(arg: str) -> list[str]:
    return [
        part.strip()
        for chunk in arg.split()
        for part in chunk.split(",")
        if part.strip()
    ]


class CLIState:
    """Per-session state for the interactive REPL.

    Kept separate from the I/O loop so unit tests can drive command handlers
    directly without stdin/stdout. All engine mutation goes through
    EngineBridge, which is the same path the Discord bot uses.

    `story_id` is "" when the session is empty (no story loaded yet).
    """

    def __init__(self, engine: EngineBridge, session_id: str, story_id: str):
        self.engine = engine
        self.session_id = session_id
        self.story_id = story_id
        self.current_actor: str | None = None
        # char_id -> synthetic user_id. Mirrors Discord's one-user-per-char
        # binding so we exercise the full path; the CLI is just "the god user"
        # that owns many synthetic accounts.
        self.claims: dict[str, int] = {}
        self._next_user_id = 1
        self.running = True

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

    # ---- dispatch ------------------------------------------------------------

    async def handle_line(self, line: str) -> None:
        """Top-level dispatch: /command or in-character action."""
        line = line.strip()
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

    # ---- commands ------------------------------------------------------------

    def cmd_help(self, arg: str) -> None:
        print(HELP_TEXT)

    # ---- story subcommands ---------------------------------------------------

    def cmd_story(self, arg: str) -> None | object:
        """Dispatcher for `/story <sub>` — forwards to cmd_story_<sub>."""
        parts = arg.split(maxsplit=1) if arg.strip() else []
        if not parts:
            print("usage: /story [list|start|info|delete] ...")
            return
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_story_{sub}", None)
        if handler is None:
            print(f"unknown story subcommand: {sub}")
            return
        return handler(rest)

    def cmd_story_list(self, arg: str) -> None:
        ids = self.engine.list_story_ids()
        if not ids:
            print("no stories imported — run scripts/import_story.py first")
            return
        for sid in ids:
            print(f"  {sid}")

    def cmd_story_info(self, arg: str) -> None:
        story_id = arg.strip()
        if not story_id:
            print("usage: /story info <story_id>")
            return
        if story_id not in self.engine.list_story_ids():
            print(f"unknown story: {story_id}")
            return
        try:
            ckpt = self.engine.load_story_ckpt(story_id)
        except Exception as e:
            print(f"error: {e}")
            return
        setting = ckpt.world_state.setting
        print()
        print(f"# {story_id}")
        print(f"Genre: {setting.genre}")
        print(f"Tone: {setting.tone}")
        print(f"Premise: {setting.premise}")
        print(f"Characters: {len(ckpt.characters)}")
        print("Scenes: (router-managed perception)")
        print()

    def cmd_story_start(self, arg: str) -> None:
        story_id = arg.strip()
        if not story_id:
            print("usage: /story start <story_id>")
            return
        if self.story_id:
            print(
                f"session already has story `{self.story_id}` loaded. "
                f"/story delete first."
            )
            return
        if story_id not in self.engine.list_story_ids():
            print(f"unknown story: {story_id}")
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
        print("run /characters then /join <character_id> to claim one")

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
        for sid in ids:
            marker = "  ← current" if sid == self.session_id else ""
            print(f"  {sid}{marker}")

    def cmd_session_end(self, arg: str) -> None:
        print(f"detached from `{self.session_id}` (files kept on disk)")
        self.running = False

    # ---- gameplay commands ---------------------------------------------------

    def cmd_characters(self, arg: str) -> None:
        if not self._require_story():
            return
        ckpt = self.engine.load_latest(self.session_id)
        from app.engine.context_builder import resolve_location_for_character
        location = resolve_location_for_character(
            ckpt,
            self.current_actor or next(iter(self.claims), None),
        )
        claimed_ids = set(self.claims)

        here: list[str] = []
        elsewhere: list[str] = []
        dormant: list[str] = []
        culled: list[str] = []
        for c in ckpt.characters:
            # Claimed characters are shown exclusively in "Claimed by you" so
            # they don't appear twice in the output.
            if c.character_id in claimed_ids:
                continue
            status = c.status.value
            if status == "dormant":
                dormant.append(c.character_id)
            elif status == "culled":
                culled.append(c.character_id)
            else:
                if c.location == location:
                    here.append(c.character_id)
                else:
                    elsewhere.append(c.character_id)

        def _suffix(cid: str) -> str:
            return ""

        print(f"## Here ({location or 'no active location'})")
        for cid in here:
            print(f"  {cid}{_suffix(cid)}")
        if not here:
            print("  (none)")

        print()
        print("## Claimed by you")
        if self.claims:
            for cid, uid in self.claims.items():
                marker = "  ← acting" if cid == self.current_actor else ""
                print(f"  {cid}  (uid {uid}){marker}")
        else:
            print("  (none)")

        print()
        print("## Active (elsewhere)")
        if elsewhere:
            for cid in elsewhere:
                print(f"  {cid}{_suffix(cid)}")
        else:
            print("  (none)")

        print()
        print("## Dormant")
        if dormant:
            for cid in dormant:
                print(f"  {cid}{_suffix(cid)}")
        else:
            print("  (none)")

        print()
        print("## Culled")
        if culled:
            for cid in culled:
                print(f"  {cid}{_suffix(cid)}")
        else:
            print("  (none)")

    def cmd_character(self, arg: str) -> None:
        if not self._require_story():
            return
        char_id = arg.strip()
        if not char_id:
            print("usage: /character <id>")
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
        ckpt = self.engine.load_latest(self.session_id)
        from app.engine.context_builder import resolve_location_for_character
        location = resolve_location_for_character(
            ckpt,
            self.current_actor or next(iter(self.claims), None),
        )
        location_name = location or "(no active location — /act to begin)"
        print(f"story: {self.story_id}")
        print(f"turn: {ckpt.session.turn_index}")
        print(f"location: {location_name}")
        if not self.claims:
            print("claims: (none)")
            return
        print("claims:")
        for char_id, uid in self.claims.items():
            marker = "  ← acting" if char_id == self.current_actor else ""
            print(f"  - {char_id} (uid {uid}){marker}")
        self._print_open_reaction_slots()

    def cmd_inventory(self, arg: str) -> None:
        if not self._require_story():
            return
        target = arg.strip() or self.current_actor
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
        sub = parts[0].lower().replace("-", "_")
        rest = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_loot_{sub}", None)
        if handler is None:
            print(f"unknown loot subcommand: {parts[0]}")
            return
        result = handler(rest)
        if asyncio.iscoroutine(result):
            await result

    def _print_loot_for_current(self) -> None:
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
        _print_loot_offers(offers)

    async def cmd_loot_take(self, arg: str) -> None:
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            print("usage: /loot take <offer_id> <all|item_id[,item_id...]>")
            return
        offer_id, item_spec = parts[0], parts[1].strip()
        uid = self.claims.get(self.current_actor)
        if uid is None:
            print(f"not claimed: {self.current_actor}")
            return
        try:
            take_currency = False
            if item_spec.lower() == "all":
                offers = self.engine.list_loot_offers(
                    self.session_id,
                    uid,
                    character_id=self.current_actor,
                )
                offer = next((o for o in offers if o.offer_id == offer_id), None)
                if offer is None:
                    raise ValueError("That loot offer is already closed.")
                item_ids = dnd_inventory.available_item_ids(offer)
                take_currency = offer.has_available_currency()
            else:
                item_ids = [
                    part.strip() for part in item_spec.split(",")
                    if part.strip()
                ]
            result = await self.engine.claim_loot(
                session_id=self.session_id,
                user_id=uid,
                character_id=self.current_actor,
                offer_id=offer_id,
                item_ids=item_ids,
                take_currency=take_currency,
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            return
        print(result.message)

    async def cmd_loot_split_coins(self, arg: str) -> None:
        offer_id = arg.strip()
        if not offer_id:
            print("usage: /loot split-coins <offer_id>")
            return
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
        offer_id = arg.strip()
        if not offer_id:
            print("usage: /loot decline <offer_id>")
            return
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

    def cmd_join(self, arg: str) -> None:
        if not self._require_story():
            return
        char_id = arg.strip()
        if not char_id:
            print("usage: /join <character_id>")
            return
        if char_id in self.claims:
            print(f"already claimed: {char_id}. /as {char_id} to switch.")
            return
        uid = self._next_user_id
        try:
            self.engine.takeover(self.session_id, char_id, uid)
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
        if self.current_actor == char_id:
            print(f"claimed {char_id} — now acting as {char_id}.")
        else:
            print(f"claimed {char_id}. /as {char_id} to switch.")

    async def cmd_join_custom(self, arg: str) -> None:
        if not self._require_story():
            return
        loop = asyncio.get_event_loop()

        async def _prompt(label: str) -> str | None:
            try:
                raw = await loop.run_in_executor(None, input, label)
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            return raw

        mode = arg.strip().lower()
        if not mode:
            resp = await _prompt("describe or replace? > ")
            if resp is None:
                print("(cancelled)")
                return
            mode = resp.strip().lower()
        if mode not in {"describe", "replace"}:
            print("usage: /join_custom [describe|replace]")
            return

        desc_resp = await _prompt("describe the character you want to play: ")
        if desc_resp is None:
            print("(cancelled)")
            return
        description = desc_resp.strip()
        if not description:
            print("(cancelled — empty description)")
            return

        uid = self._next_user_id

        if mode == "describe":
            try:
                new_char = await self.engine.create_custom_character(
                    self.session_id, uid, description,
                )
            except ValueError as e:
                print(f"error: {e}")
                return
            except Exception as e:
                print(f"error: {e}")
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
            return

        # replace mode
        try:
            suggestion = await self.engine.suggest_replacement_targets(
                self.session_id, description,
            )
        except Exception as e:
            print(f"error: {e}")
            return

        preamble = suggestion.get("preamble", "") or ""
        candidates = suggestion.get("candidates", []) or []
        if preamble.strip():
            print()
            print(preamble.strip())
        if not candidates:
            print("(no replacement candidates suggested)")
            return

        print()
        for i, cand in enumerate(candidates, start=1):
            cid = cand.get("character_id", "")
            name = cand.get("name", "")
            rationale = (cand.get("fit_rationale") or "").strip().replace("\n", " ")
            print(f"  [{i}] {cid} — {name} — {rationale}")
        print()

        pick_resp = await _prompt("pick a number (or blank to cancel): ")
        if pick_resp is None:
            print("(cancelled)")
            return
        pick_raw = pick_resp.strip()
        if not pick_raw:
            print("(cancelled)")
            return
        try:
            idx = int(pick_raw)
        except ValueError:
            print(f"(cancelled — not a number: {pick_raw!r})")
            return
        if idx < 1 or idx > len(candidates):
            print(f"(cancelled — out of range: {idx})")
            return
        target_id = candidates[idx - 1].get("character_id", "")
        if not target_id:
            print("(cancelled — candidate missing character_id)")
            return

        try:
            mutated = await self.engine.replace_with_custom(
                self.session_id, uid, target_id, description,
            )
        except ValueError as e:
            print(f"error: {e}")
            return
        except Exception as e:
            print(f"error: {e}")
            return

        self.claims[mutated.character_id] = uid
        self._next_user_id += 1
        if self.current_actor is None:
            self.current_actor = mutated.character_id
        try:
            dossier = self.engine.build_character_dossier(
                self.session_id, mutated.character_id,
            )
        except Exception as e:
            dossier = f"(dossier unavailable: {e})"
        print()
        print(dossier)
        print()
        if self.current_actor == mutated.character_id:
            print(
                f"replaced {mutated.character_id} — now acting as "
                f"{mutated.character_id}."
            )
        else:
            print(
                f"replaced {mutated.character_id}. "
                f"/as {mutated.character_id} to switch."
            )

    async def cmd_leave(self, arg: str) -> None:
        if not self._require_story():
            return
        target = arg.strip() or self.current_actor
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
            self.engine.unbind_user(self.session_id, uid)
        del self.claims[target]
        if self.current_actor == target:
            self.current_actor = next(iter(self.claims), None)
        actor_note = (
            f" — now acting as {self.current_actor}"
            if self.current_actor else " — no current actor"
        )
        print(f"released {target}{actor_note}")

    def cmd_as(self, arg: str) -> None:
        target = arg.strip()
        if not target:
            print("usage: /as <character_id>")
            return
        if target not in self.claims:
            print(f"not claimed: {target}. /join {target} first.")
            return
        self.current_actor = target
        print(f"now acting as {target}")

    async def cmd_describe(self, arg: str) -> None:
        """Prompt for the minimum info needed to play: name + appearance.
        Both are optional per-prompt — leave blank to keep the existing
        value. Trailing arg is ignored; this command is always
        interactive."""
        if not self._require_story():
            return
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        loop = asyncio.get_event_loop()
        try:
            name = (await loop.run_in_executor(
                None, input, "name (blank to keep existing): ",
            )).strip()
            appearance = (await loop.run_in_executor(
                None, input, "appearance (blank to keep existing): ",
            )).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(cancelled)")
            return
        if not name and not appearance:
            print("(no changes)")
            return
        try:
            ckpt = self.engine.set_character_identity(
                self.session_id, self.current_actor,
                name=name or None, appearance=appearance or None,
            )
        except Exception as e:
            print(f"error: {e}")
            return
        print(f"updated {self.current_actor}")

        # Mirror the Discord bot: if no narrator turns yet, fire (begin) so
        # the opening lands with the description in hand.
        #
        # v11-r6c note: `(begin)` rides through the normal /act path and
        # gets wrapped as "{name} attempts: (begin)" by the dispatcher.
        # The event_router prompt's Author-directive OOC rule fires on
        # the parenthesized-input shape itself, independent of the
        # "attempts:" framing, so OOC routing is correct without a
        # dedicated CLI code path. Same reasoning as /describe in the
        # Discord frontend.
        if not any(ckpt.narrator_conversations.values()):
            print("opening story…")
            await self._act("(begin)")

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
            result = await self.engine.run_query(
                session_id=self.session_id,
                character_id=self.current_actor,
                question=question,
            )
        except Exception as e:
            logger.exception("run_query failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        gate_tag = (
            f"  [gated: {result.gate_reason or '?'}]"
            if result.knowledge_gated else ""
        )
        print()
        print(f"--- /query · {self.current_actor}{gate_tag} ---")
        print(result.answer or "(no answer)")
        print()

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

        prompts = self.engine.pending_roll_prompts(
            self.session_id,
            user_id=uid,
        )
        if not prompts:
            print("no pending D&D roll for the current actor")
            return

        requested = arg.strip()
        printed_pending_from_response = False
        if not requested:
            if len(prompts) == 1:
                selected = [prompts[0]]
            else:
                _print_roll_prompts(prompts)
                return
        elif requested.lower() == "all":
            selected = prompts
        else:
            chosen = next((p for p in prompts if p.roll_id == requested), None)
            if chosen is None:
                print(f"no pending roll id for {self.current_actor}: {requested}")
                _print_roll_prompts(prompts)
                return
            selected = [chosen]

        continued_events: set[str] = set()
        for prompt in selected:
            try:
                result = await self.engine.complete_pending_roll(
                    session_id=self.session_id,
                    event_id=prompt.event_id,
                    roll_id=prompt.roll_id,
                    user_id=uid,
                )
            except Exception as e:
                logger.exception("pending roll failed")
                print(f"error: {type(e).__name__}: {e}")
                return
            print(_roll_result_line(result))

            if result.remaining_pending_rolls > 0:
                continue
            if result.event_id in continued_events:
                continue
            continued_events.add(result.event_id)
            print("Interpreting the outcome...")
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
            self._print_turn_response(response, actor_id=result.actor_id)
            if response.beat_ended_reason == "cat_ii_pending_rolls":
                printed_pending_from_response = True

        if not printed_pending_from_response:
            remaining = self.engine.pending_roll_prompts(
                self.session_id,
                user_id=uid,
            )
            _print_roll_prompts(remaining)

    async def cmd_rewind(self, arg: str) -> None:
        """Rewind the session to an earlier checkpoint.

        Usage:
            /rewind                  → list available turn checkpoints
            /rewind <N>              → rewind to ckpt_N (deletes ckpt_>N)

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

        # CLI path: a single playtester driving things deliberately. We
        # show the preview, then commit immediately. Discord wraps this
        # same primitive in a confirm button because public sessions
        # have multiple stakeholders.
        print(
            f"rewinding {self.session_id}: "
            f"turn {preview.previous_latest} → turn {preview.target_turn} "
            f"(deleting turns {preview.deleted_turns})"
        )
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
        ckpt = self.engine.load_latest(self.session_id)
        transcript = ckpt.transcript
        if not transcript:
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

        entries = transcript[-limit:] if limit else transcript
        start = len(transcript) - len(entries) + 1
        for i, entry in enumerate(entries, start=start):
            print()
            print(f"--- Turn {i} ---")
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
            response = await self.engine.run_turn(
                session_id=self.session_id,
                user_input=text,
                acting_character_id=self.current_actor,
            )
        except Exception as e:
            logger.exception("run_turn failed")
            print(f"error: {type(e).__name__}: {e}")
            return

        self._print_turn_response(response, actor_id=self.current_actor)

    def _print_turn_response(
        self,
        response,
        *,
        actor_id: str,
    ) -> None:
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

        # Print pre-turn AFK-sweep resolutions first so in-CLI ordering
        # matches story time.
        for pre_resp in (response.pre_turn_resolutions or []):
            for cid, prose in (pre_resp.per_player_renders or {}).items():
                if not prose or cid not in self.claims:
                    continue
                print(f"--- AFK auto-resolution · POV {cid} ---")
                print(prose)
                print()
            self._print_loot_prompts(pre_resp)

        per_player = response.per_player_renders or {}

        if response.beat_ended_reason == "cat_ii_pending":
            actor_render = per_player.get(actor_id) or ""
            print()
            self._print_cat_ii_pending_notice()
            if actor_render:
                print()
                print(f"--- Turn {response.turn_index} · "
                      f"{actor_id} (partial) ---")
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
            print(f"--- Turn {response.turn_index} · {actor_id} ---")
            print(response.output_text)
            print()
        else:
            print()
            print(f"--- Turn {response.turn_index} · {actor_id} ---")
            print(response.output_text)
            print()

        self._print_combat_started_notice(response)

        # Print POVs for any other characters this CLI session has
        # /join'd locally so the playtester sees the multi-POV output
        # without spinning up Discord.
        joined = set(self.claims or {})
        for cid, prose in per_player.items():
            if cid == actor_id or not prose or cid not in joined:
                continue
            print(f"--- POV · {cid} ---")
            print(prose)
            print()

        if response.beat_ended_reason == "cat_ii_pending_rolls":
            prompts = self._joined_pending_roll_prompts()
            _print_roll_prompts(prompts)

        self._print_reaction_prompts(response)
        self._print_loot_prompts(response)
        self._sync_current_actor_to_active_combat()

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
            print(
                "Combat started. The initiating action has not resolved "
                "before initiative."
            )
            return

        if not view.active:
            print(
                "Combat started. The initiating action has not resolved "
                "before initiative."
            )
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
        parts.append("The initiating action has not resolved before initiative.")
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
            for offer_id in offer_ids:
                print(f"offer {offer_id}")
            print("Use /loot to inspect, /loot take <offer_id> all to claim.")
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


async def main_async(args: argparse.Namespace) -> int:
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    engine = EngineBridge()

    session_id: str | None = args.session

    # TODO: interactive menu when --session isn't provided. For now
    # we require the flag so the script stays scriptable.
    if not session_id:
        print(
            "--session <name> is required. Pick an existing save or a new "
            f"name.\nExisting sessions: "
            f"{', '.join(engine.list_session_ids()) or '(none)'}",
            file=sys.stderr,
        )
        await engine.close()
        return 2

    try:
        # Resolve story_id from the latest checkpoint if the session already
        # has one loaded; otherwise create an empty session to write into.
        story_id = ""
        try:
            ckpt = engine.load_latest(session_id)
            story_id = ckpt.session.story_id or ""
            print(f"resumed session `{session_id}`" + (
                f" · story `{story_id}`" if story_id else " (no story loaded)"
            ))
        except FileNotFoundError:
            try:
                engine.create_empty_session(session_id)
            except FileExistsError as e:
                print(f"error: {e}", file=sys.stderr)
                await engine.close()
                return 2
            print(f"created session `{session_id}` (empty — /story start to load content)")

        state = CLIState(engine, session_id, story_id)

        print()
        print(HELP_TEXT)
        print()

        loop = asyncio.get_event_loop()
        while state.running:
            actor = state.current_actor or "-"
            try:
                # input() is blocking — use run_in_executor so Ctrl-C during
                # a turn doesn't wedge; and it lets async turns interleave
                # cleanly with stdin reads.
                line = await loop.run_in_executor(
                    None, input, f"[{actor}] > ",
                )
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            try:
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
        "-v", "--verbose", action="store_true",
        help="Log engine INFO messages to stderr.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
