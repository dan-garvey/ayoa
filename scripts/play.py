#!/usr/bin/env python3
"""Interactive CLI alternative to the Discord bot.

Drives the engine in-process via EngineBridge — same path the bot uses — so
bindings, per-session locks, rolling conversations, and multi-player POV all
behave identically. The CLI simulates multiple Discord users from one
terminal by minting synthetic integer user_ids (1, 2, 3, ...) per /join,
which exercises the full binding code path and lets you playtest multi-
character scenes without needing two Discord accounts.

Usage:
    .venv/bin/python scripts/play.py --story <story_id>
    .venv/bin/python scripts/play.py --story <story_id> --session <session_id>
    .venv/bin/python scripts/play.py --story <story_id> --name "Aldric"
    .venv/bin/python scripts/play.py --session <session_id>    # resume only

Commands inside the REPL:
    /help                       Show commands
    /characters                 List roster with status and bindings
    /join <character_id>        Claim a character; prints their dossier
    /leave [character_id]       Release a claim (default: current actor)
    /as <character_id>          Switch which claimed character acts next
    /describe <traits>          Set appearance of the current actor
    /status                     Session summary
    /quit                       Exit (Ctrl-D also works)

Anything else is an in-character action for the current actor. Use "(begin)"
to open the scene per the author's opening directive.
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

from app.bot.engine_bridge import EngineBridge

load_dotenv()
logger = logging.getLogger(__name__)


HELP_TEXT = """\
Commands:
  /help                             Show this help
  /characters                       List roster grouped by location/status
  /character <id>                   Full dossier for a character
  /join <character_id>              Claim a character; prints their dossier
  /join_custom [describe|replace]   Play a character of your own description
  /leave [character_id]             Release a claim (default: current actor)
  /as <character_id>                Switch which claimed character acts next
  /describe <traits>                Set appearance of the current actor
  /settings                         Show experimental settings for this session
  /settings <key>                   Show one setting's current value
  /settings <key> <value>           Update a setting
  /status                           Session summary
  /history [N]                      Print all turns, or last N
  /quit                             Exit (Ctrl-D also works)

Plain text is an in-character action for the current actor. Use "(begin)"
to open the scene from the author's opening directive."""


class CLIState:
    """Per-session state for the interactive REPL.

    Kept separate from the I/O loop so unit tests can drive command handlers
    directly without stdin/stdout. All engine mutation goes through
    EngineBridge, which is the same path the Discord bot uses.
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

        self._load_existing_claims()

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

    # ---- commands ------------------------------------------------------------

    def cmd_help(self, arg: str) -> None:
        print(HELP_TEXT)

    def cmd_characters(self, arg: str) -> None:
        ckpt = self.engine.load_latest(self.session_id)
        scene_id = ckpt.world_state.locations.current_scene_id
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
                if c.location == scene_id:
                    here.append(c.character_id)
                else:
                    elsewhere.append(c.character_id)

        def _suffix(cid: str) -> str:
            return ""

        print(f"## Here ({scene_id})")
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
        ckpt = self.engine.load_latest(self.session_id)
        scene_id = ckpt.world_state.locations.current_scene_id
        scene = ckpt.world_state.locations.scene_graph.get(scene_id, {})
        scene_name = scene.get("name", scene_id) if isinstance(scene, dict) else scene_id
        print(f"story: {self.story_id}")
        print(f"session: {self.session_id}")
        print(f"turn: {ckpt.session.turn_index}")
        print(f"scene: {scene_name} ({scene_id})")
        if not self.claims:
            print("claims: (none)")
            return
        print("claims:")
        for char_id, uid in self.claims.items():
            marker = "  ← acting" if char_id == self.current_actor else ""
            print(f"  - {char_id} (uid {uid}){marker}")

    def cmd_join(self, arg: str) -> None:
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

    def cmd_leave(self, arg: str) -> None:
        target = arg.strip() or self.current_actor
        if target is None:
            print("no character to leave")
            return
        uid = self.claims.get(target)
        if uid is None:
            print(f"not claimed: {target}")
            return
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
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        traits = arg.strip()
        if not traits:
            print("usage: /describe <traits>")
            return
        try:
            ckpt = self.engine.set_character_appearance(
                self.session_id, self.current_actor, traits,
            )
        except Exception as e:
            print(f"error: {e}")
            return
        print(f"updated appearance for {self.current_actor}")

        # Mirror the Discord bot: if no narrator turns yet, fire (begin) so
        # the scene opens with the description in hand. This is pre-play
        # auto-open behavior; subsequent /describe just updates the sheet.
        if not ckpt.narrator_conversation:
            print("opening scene…")
            await self._act("(begin)")

    def cmd_settings(self, arg: str) -> None:
        """
        /settings                 → list all
        /settings <key>           → show one value
        /settings <key> <value>   → set (value may be multi-word)
        """
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

    async def _act(self, text: str) -> None:
        if self.current_actor is None:
            print("no current actor — /join a character first")
            return
        try:
            response = await self.engine.run_turn(
                session_id=self.session_id,
                user_input=text,
                acting_character_id=self.current_actor,
                debug=False,
            )
        except Exception as e:
            logger.exception("run_turn failed")
            print(f"error: {type(e).__name__}: {e}")
            return
        print()
        print(f"--- Turn {response.turn_index} · {self.current_actor} ---")
        print(response.output_text)
        print()


# ---- bootstrap --------------------------------------------------------------


async def _ensure_session(
    engine: EngineBridge,
    story_id: str,
    session_id: str,
    player_name: str,
) -> None:
    """Resume an existing CLI session if present, otherwise create a new one
    with the caller as creator (synthetic uid 1)."""
    try:
        engine.load_latest(session_id)
        print(f"resumed session `{session_id}`")
        return
    except FileNotFoundError:
        pass
    print(f"creating new session `{session_id}` for story `{story_id}`…")
    await engine.create_session(
        story_id=story_id,
        session_id=session_id,
        player_display_name=player_name,
        creator_user_id=1,
    )
    print("session ready — /characters to see the roster")


async def main_async(args: argparse.Namespace) -> int:
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    engine = EngineBridge()

    session_id: str | None = args.session
    story_id: str | None = args.story

    # Resume path: if the named session is already on disk, derive story_id
    # from its checkpoint so the caller can omit --story on resume.
    resumed = False
    if session_id:
        try:
            ckpt = engine.load_latest(session_id)
            resumed = True
            existing = ckpt.session.story_id
            if story_id and existing and existing != story_id:
                print(
                    f"session `{session_id}` belongs to story `{existing}`, "
                    f"not `{story_id}`",
                    file=sys.stderr,
                )
                await engine.close()
                return 2
            story_id = story_id or existing
        except FileNotFoundError:
            pass

    if not resumed:
        if not story_id:
            print(
                "--story is required when starting a new session "
                "(pass --session <existing> to resume)",
                file=sys.stderr,
            )
            await engine.close()
            return 2
        if story_id not in engine.list_story_ids():
            print(
                f"unknown story `{story_id}`. Available: "
                f"{', '.join(engine.list_story_ids()) or '(none)'}",
                file=sys.stderr,
            )
            await engine.close()
            return 2

    session_id = session_id or f"cli_{story_id}"
    player_name = args.name or "Player"

    try:
        await _ensure_session(engine, story_id, session_id, player_name)
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
        "--story",
        help="Story ID (directory under app/storage/saves). "
             "Required for new sessions; omit when resuming via --session.",
    )
    parser.add_argument(
        "--session",
        help="Session ID. Defaults to cli_<story_id>. "
             "If the session already exists on disk, --story can be omitted.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Player display name substituted for PLAYER_NAME on a new session.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Log engine INFO messages to stderr.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
