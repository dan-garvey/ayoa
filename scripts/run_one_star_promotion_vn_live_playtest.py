#!/usr/bin/env python3
"""Run the One-Star promotion fixture through real CLI/runtime turns.

The scripted inputs use ``CLIState`` and the shared ``EngineBridge`` used by
Discord. Router, character-agent, narrator, image-director, checkpoint, sprite
resolution, and VN compositor behavior are all production paths. The harness
records every model request/output and every delivered deck. It never authors
narrator pages or sends image bytes to an LLM.

The isolated image runtime starts empty. The live story must reach Mara's
two-star prewarm boundary, generate her candidate pack through the production
worker path, and then resolve one of those generated variants at the three-star
reveal. No earlier story output or sprite-job database is reused.
"""

from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path.

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.engine.one_star_adapter import load_one_star_account, load_one_star_hero
from app.engine.one_star_visuals import (
    generated_sprite_pack_id,
    sprite_set_id_for_viewer,
)
from app.llm.config import LLMConfig, live_play_required_roles
from app.schemas.checkpoint import CheckpointFile
from app.schemas.responses import TurnResponse
from scripts.export_vn_playtest_slideshow import export_vn_playtest_slideshow
from scripts.play import CLIState


STORY_ID = "one_star_ascension_s1_promotion_playtest"
MASTER_ID = "the_master"
RENNA_ID = "renna_holt"
MARA_ID = "promotion_playtest_faceless"
CASTOR_ID = "castor_valebrand"
SOURCE_STORY_DIR = REPO_ROOT / "app/storage/stories" / STORY_ID
SOURCE_SESSION_ID = "one-star-promotion-sprite-20260827t173922z"
DEFAULT_REPORT_ROOT = REPO_ROOT / "app/storage/playtest_reports"
DEFAULT_WINDOWS_ROOT = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarPromotionLivePlaytest_20260827"
)


class RecordingCLIState(CLIState):
    """CLI delivery without an interactive terminal image pager."""

    async def _play_visual_novel_deck(
        self,
        deck,
        *,
        character_id: str,
    ) -> None:
        del character_id
        for card in deck.cards:
            print(card.accessible_text)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_playtest_inputs(run_dir: Path) -> tuple[Path, Path]:
    if not (SOURCE_STORY_DIR / "ckpt_0000.json").is_file():
        raise RuntimeError("promotion playtest story seed is missing")

    story_dir = run_dir / "stories" / STORY_ID
    shutil.copytree(SOURCE_STORY_DIR, story_dir)

    runtime_root = run_dir / "runtime" / "image_generation"
    runtime_root.mkdir(parents=True)
    (runtime_root / "artifacts").mkdir()
    (runtime_root / "tmp").mkdir()
    return story_dir, runtime_root


def _character(checkpoint: CheckpointFile, character_id: str):
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == character_id
    )


def _hero_summary(
    checkpoint: CheckpointFile,
    character_id: str,
) -> dict[str, object]:
    character = _character(checkpoint, character_id)
    hero = load_one_star_hero(character)
    if hero is None:
        raise RuntimeError(f"missing One-Star state for {character_id}")
    return {
        "character_id": character_id,
        "name": character.name,
        "status": character.status.value,
        "location": character.location,
        "stars": hero.current_stars,
        "level": hero.level,
        "experience_points": hero.experience_points,
        "generated_for_summon": hero.generated_for_summon,
        "current_variant_key": (
            character.visuals.visual_novel_presentation.current_variant_key
        ),
        "sprite_set_id": sprite_set_id_for_viewer(
            checkpoint,
            viewer_character_id=MASTER_ID,
            character=character,
        ),
    }


def _state_summary(checkpoint: CheckpointFile) -> dict[str, object]:
    _owner, account = load_one_star_account(checkpoint)
    pending = account.state.pending_operation
    return {
        "turn_index": checkpoint.session.turn_index,
        "checkpoint_id": f"ckpt_{checkpoint.session.turn_index:04d}",
        "pending_operation": (
            pending.model_dump(mode="json") if pending is not None else None
        ),
        "remaining_promotion_stones": account.state.resources.materials.get(
            "lesser_promotion_stone",
            0,
        ),
        "renna": _hero_summary(checkpoint, RENNA_ID),
        "mara": _hero_summary(checkpoint, MARA_ID),
        "castor": _hero_summary(checkpoint, CASTOR_ID),
    }


def _message_contains_image_bytes(messages: Any) -> bool:
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or "").casefold()
            if kind in {"image", "image_url", "input_image"}:
                return True
            source = item.get("source")
            if isinstance(source, dict) and source.get("data"):
                return True
    return False


def _flatten_render_pages(render_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for segment in render_payload.get("segments") or ():
        pages.extend(segment.get("pages") or ())
    return pages


def _deck_has_sprite_transition(
    deck_record: dict[str, Any],
    *,
    character_name: str,
    identity_handle: str,
) -> bool:
    pages = _flatten_render_pages(deck_record["render"])
    sections = deck_record["manifest"]["identity"]["sections"]
    if len(sections) < len(pages):
        return False
    for page, section in zip(pages, sections[:len(pages)], strict=True):
        cue_names = {str(label) for label in page.get("sprites") or ()}
        handles = {
            str(sprite.get("identity_handle") or "")
            for sprite in section.get("sprites") or ()
        }
        if character_name in cue_names and identity_handle in handles:
            return True
    return False


def _deck_has_committed_identity_reveal(
    deck_record: dict[str, Any],
    *,
    character_name: str,
    before_identity_handle: str,
    after_identity_handle: str,
    expected_stage_sha256: str,
) -> bool:
    """Verify old authored pages, then flash, then first new-identity card."""

    pages = _flatten_render_pages(deck_record["render"])
    sections = deck_record["manifest"]["identity"]["sections"]
    if len(sections) != len(pages) + 2:
        return False
    authored_sections = sections[:len(pages)]
    depicted_before = False
    for page, section in zip(pages, authored_sections, strict=True):
        cue_names = {str(label) for label in page.get("sprites") or ()}
        handles = {
            str(sprite.get("identity_handle") or "")
            for sprite in section.get("sprites") or ()
        }
        if character_name not in cue_names:
            continue
        depicted_before = True
        if before_identity_handle not in handles or after_identity_handle in handles:
            return False
    flash, reveal = sections[-2:]
    flash_handles = {
        str(sprite.get("identity_handle") or "")
        for sprite in flash.get("sprites") or ()
    }
    reveal_handles = {
        str(sprite.get("identity_handle") or "")
        for sprite in reveal.get("sprites") or ()
    }
    return (
        depicted_before
        and flash.get("card_style") == "identity_flash"
        and reveal.get("card_style") == "identity_reveal"
        and flash_handles == {before_identity_handle}
        and reveal_handles == {after_identity_handle}
        # Earlier event-aligned pages may still occur in the lobby before the
        # character enters.  The last authored page and the appended flash /
        # reveal must share the fixed chamber so the transition itself cannot
        # jump locations.
        and authored_sections[-1].get("stage_sha256")
        == expected_stage_sha256
        and flash.get("stage_sha256") == expected_stage_sha256
        and reveal.get("stage_sha256") == expected_stage_sha256
    )


def _deck_uses_only_stage(
    deck_record: dict[str, Any],
    *,
    stage_sha256: str,
) -> bool:
    sections = deck_record["manifest"]["identity"]["sections"]
    return bool(sections) and all(
        section.get("stage_sha256") == stage_sha256
        for section in sections
    )


def _configure_luna_agents() -> LLMConfig:
    env_path = Path("/home/dan/ayoa/.env")
    load_dotenv(env_path if env_path.is_file() else None)
    os.environ["LLM_MODEL_AGENT"] = "gpt-5.6-luna"
    os.environ["LLM_MODEL_AGENT_STANDARD"] = "gpt-5.6-luna"
    os.environ["LLM_MODEL_AGENT_CONVENIENCE"] = "gpt-5.6-luna"
    os.environ["LLM_MODEL_CHARACTER_GEN"] = "gpt-5.6-luna"
    config = LLMConfig.from_env()
    missing = config.missing_credentials(live_play_required_roles())
    if missing:
        detail = ", ".join(f"{item.role} ({item.provider})" for item in missing)
        raise RuntimeError(f"missing live LLM credentials: {detail}")
    return config


async def _run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"one_star_promotion_vn_live_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("llm_calls", "responses", "snapshots"):
        (run_dir / directory).mkdir()
    story_dir, runtime_root = _copy_playtest_inputs(run_dir)

    logging.basicConfig(
        filename=run_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    config = _configure_luna_agents()
    engine = EngineBridge(
        stories_dir=str(run_dir / "stories"),
        sessions_dir=str(run_dir / "sessions"),
        prompts_dir=str(REPO_ROOT / "app/prompts"),
        llm_config=config,
    )
    if engine.image_generation.config.runtime_root != runtime_root:
        raise RuntimeError("isolated image runtime path did not match")

    current_action = {"label": "startup", "input": ""}
    llm_call_counter = 0
    llm_calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    decks: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    response_origins: dict[str, dict[str, str]] = {}
    render_origins: dict[str, list[dict[str, str]]] = {}

    real_complete = engine.client.complete

    async def recording_complete(*call_args, **kwargs):
        nonlocal llm_call_counter
        role = str(kwargs.get("role") or (call_args[0] if call_args else ""))
        messages = kwargs.get("messages") or []
        llm_call_counter += 1
        call_index = llm_call_counter
        record: dict[str, Any] = {
            "index": call_index,
            "action_label": current_action["label"],
            "player_input": current_action["input"],
            "role": role,
            "response_model": getattr(
                kwargs.get("response_model"),
                "__name__",
                "",
            ),
            "messages": _jsonable(messages),
            "contains_image_bytes": _message_contains_image_bytes(messages),
        }
        try:
            result = await real_complete(*call_args, **kwargs)
        except Exception:
            record["error"] = traceback.format_exc()
            llm_calls.append(record)
            _write_json(
                run_dir / "llm_calls" / f"{call_index:03d}_{role}.json",
                record,
            )
            raise
        record.update(
            {
                "model": result.model,
                "usage": dict(result.usage or {}),
                "raw_output": result.content,
                "parsed_output": _jsonable(result.parsed),
                "reasoning_summaries": list(result.reasoning_summaries or []),
            }
        )
        llm_calls.append(record)
        _write_json(
            run_dir / "llm_calls" / f"{call_index:03d}_{role}.json",
            record,
        )
        return result

    engine.client.complete = recording_complete  # type: ignore[method-assign]

    def record_response(kind: str, response: TurnResponse) -> None:
        origin = {
            "action_label": current_action["label"],
            "player_input": current_action["input"],
        }
        response_origins[response.checkpoint_id] = origin
        for pov_character_id, render in (
            response.per_player_visual_novel_renders or {}
        ).items():
            render_key = hashlib.sha256(
                (
                    response.checkpoint_id
                    + "\0"
                    + pov_character_id
                    + "\0"
                    + render.model_dump_json()
                ).encode("utf-8")
            ).hexdigest()
            render_origins.setdefault(render_key, []).append(origin)
        record = {
            "index": len(responses) + 1,
            "kind": kind,
            "action_label": current_action["label"],
            "player_input": current_action["input"],
            "response": response.model_dump(mode="json"),
        }
        responses.append(record)
        _write_json(
            run_dir
            / "responses"
            / f"{record['index']:03d}_{current_action['label']}.json",
            record,
        )

    real_run_turn = engine.run_turn

    async def recording_run_turn(*args_, **kwargs_):
        response = await real_run_turn(*args_, **kwargs_)
        record_response("run_turn", response)
        return response

    engine.run_turn = recording_run_turn  # type: ignore[method-assign]

    real_begin = engine.run_begin_turn

    async def recording_begin(*args_, **kwargs_):
        response = await real_begin(*args_, **kwargs_)
        record_response("run_begin_turn", response)
        return response

    engine.run_begin_turn = recording_begin  # type: ignore[method-assign]

    real_synthesis = engine.run_one_star_synthesis_command

    async def recording_synthesis(*args_, **kwargs_):
        response = await real_synthesis(*args_, **kwargs_)
        record_response("run_one_star_synthesis_command", response)
        return response

    engine.run_one_star_synthesis_command = (  # type: ignore[method-assign]
        recording_synthesis
    )

    real_prepare_deck = engine.prepare_visual_novel_deck

    async def recording_prepare_deck(*args_, **kwargs_):
        deck = await real_prepare_deck(*args_, **kwargs_)
        render = kwargs_.get("render")
        checkpoint_id = str(kwargs_.get("checkpoint_id") or "")
        pov_character_id = str(kwargs_.get("pov_character_id") or "")
        render_key = hashlib.sha256(
            (
                checkpoint_id
                + "\0"
                + pov_character_id
                + "\0"
                + (render.model_dump_json() if render else "")
            ).encode("utf-8")
        ).hexdigest()
        queued_origins = render_origins.get(render_key, [])
        origin = (
            queued_origins.pop(0)
            if queued_origins
            else response_origins.get(checkpoint_id, current_action)
        )
        manifest = json.loads(deck.manifest_path.read_text(encoding="utf-8"))
        record = {
            "index": len(decks) + 1,
            "action_label": origin["action_label"],
            "player_input": origin["player_input"],
            "checkpoint_id": checkpoint_id,
            "pov_character_id": kwargs_.get("pov_character_id", ""),
            "deck_id": deck.deck_id,
            "manifest_path": str(deck.manifest_path.relative_to(run_dir)),
            "render": render.model_dump(mode="json") if render else {},
            "manifest": manifest,
            "cards": [
                {
                    "index": card.index,
                    "speaker": card.speaker,
                    "text": card.text,
                    "accessible_text": card.accessible_text,
                    "image_path": str(card.image_path.relative_to(run_dir)),
                    "image_sha256": hashlib.sha256(card.image_bytes).hexdigest(),
                }
                for card in deck.cards
            ],
        }
        decks.append(record)
        return deck

    engine.prepare_visual_novel_deck = (  # type: ignore[method-assign]
        recording_prepare_deck
    )

    state: RecordingCLIState | None = None
    error = ""
    candidate_wait: dict[str, Any] = {
        "minimum_variants": 2,
        "timeout_seconds": args.sprite_wait_seconds,
        "elapsed_seconds": 0.0,
        "variant_keys": [],
    }

    async def execute(label: str, line: str) -> dict[str, Any]:
        current_action.update(label=label, input=line)
        before = None
        try:
            before = _state_summary(engine.load_latest(SOURCE_SESSION_ID))
        except FileNotFoundError:
            pass
        output = io.StringIO()
        with redirect_stdout(output):
            assert state is not None
            await state.handle_line(line)
        after_checkpoint = engine.load_latest(SOURCE_SESSION_ID)
        after = _state_summary(after_checkpoint)
        source_checkpoint = max(
            (run_dir / "sessions" / SOURCE_SESSION_ID).glob("ckpt_*.json")
        )
        snapshot_path = (
            run_dir
            / "snapshots"
            / f"{len(transcript) + 1:02d}_{label}__{source_checkpoint.name}"
        )
        shutil.copy2(source_checkpoint, snapshot_path)
        item = {
            "index": len(transcript) + 1,
            "label": label,
            "input": line,
            "output": output.getvalue(),
            "state_before": before,
            "state_after": after,
            "snapshot": str(snapshot_path.relative_to(run_dir)),
        }
        transcript.append(item)
        _write_json(run_dir / "transcript.partial.json", transcript)
        print(
            f"[{label}] turn={after['turn_index']} "
            f"renna={after['renna']['stars']}★ "
            f"mara={after['mara']['stars']}★/L{after['mara']['level']} "
            f"castor={after['castor']['status']}",
            flush=True,
        )
        return after

    async def advance_until(
        label: str,
        predicate: Callable[[dict[str, Any]], bool],
        state_after_selection: dict[str, Any],
        followup_inputs: tuple[str, ...],
    ) -> dict[str, Any]:
        current = state_after_selection
        for attempt, line in enumerate(followup_inputs, start=1):
            if predicate(current):
                return current
            current = await execute(f"{label}_followup_{attempt}", line)
        if not predicate(current):
            raise RuntimeError(
                f"{label} did not reach its required state after "
                f"{len(followup_inputs)} explicit follow-up turns"
            )
        return current

    async def wait_for_generated_candidate_pack() -> tuple[str, ...]:
        """Observe the real async prewarm rather than fabricating a candidate."""

        started = time.monotonic()
        deadline = started + args.sprite_wait_seconds
        while True:
            checkpoint = engine.load_latest(SOURCE_SESSION_ID)
            character = _character(checkpoint, MARA_ID)
            sprite_pack_id = generated_sprite_pack_id(checkpoint, character)
            variant_keys = engine.image_generation.store.sprite_variant_keys(
                session_id=SOURCE_SESSION_ID,
                character_id=MARA_ID,
                sprite_pack_id=sprite_pack_id,
            )
            candidate_wait.update(
                elapsed_seconds=round(time.monotonic() - started, 3),
                sprite_pack_id=sprite_pack_id,
                variant_keys=list(variant_keys),
            )
            if "neutral" in variant_keys and len(variant_keys) >= 2:
                print(
                    "[mara_sprite_prewarm] generated variants="
                    + ",".join(variant_keys),
                    flush=True,
                )
                return variant_keys
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Mara's production sprite prewarm did not produce neutral "
                    f"plus one alternate within {args.sprite_wait_seconds}s; "
                    f"available={list(variant_keys)}"
                )
            await asyncio.sleep(2)

    await engine.start()
    try:
        engine.create_empty_session(SOURCE_SESSION_ID)
        state = RecordingCLIState(engine, SOURCE_SESSION_ID, "")
        state.one_shot_mode = True

        await execute("setup_story", f"/story start {STORY_ID}")
        await execute("setup_join", f"/join {MASTER_ID}")
        await execute("opening", "/begin")
        renna_before = await execute(
            "renna_before",
            (
                "I observe Renna Holt and Mara Venn together in Niflheim's "
                "lobby before choosing either of them for promotion."
            ),
        )
        if renna_before["renna"]["stars"] != 1:
            raise RuntimeError("Renna was not one-star in the before scene")

        selected = await execute(
            "renna_select_promotion",
            (
                "I select Renna Holt for promotion and turn the Promotion "
                "Chamber's pale light on her."
            ),
        )
        await advance_until(
            "renna_promotion",
            lambda value: value["renna"]["stars"] == 2,
            selected,
            (
                (
                    "I inspect the authorized requirements and effects of "
                    "Renna Holt's pending promotion without spending the "
                    "stone."
                ),
                (
                    "I confirm Renna Holt's pending promotion and authorize "
                    "the chamber to complete it when she enters."
                ),
                (
                    "I keep the active feed on the Promotion Chamber and wait "
                    "for Renna Holt's decision without adding another order."
                ),
            ),
        )
        await execute(
            "renna_after",
            (
                "I keep the feed on Renna Holt and inspect her after the "
                "Promotion Chamber releases her."
            ),
        )

        mara_before = await execute(
            "mara_before",
            (
                "I move the feed to Mara Venn and observe her before choosing "
                "her for promotion."
            ),
        )
        if mara_before["mara"]["stars"] != 1:
            raise RuntimeError("Mara was not one-star in the before scene")

        selected = await execute(
            "mara_select_two_star",
            (
                "I select Mara Venn for promotion and turn the Promotion "
                "Chamber's pale light on her."
            ),
        )
        await advance_until(
            "mara_two_star_promotion",
            lambda value: value["mara"]["stars"] == 2,
            selected,
            (
                (
                    "I inspect the authorized requirements and effects of "
                    "Mara Venn's pending promotion without spending the stone."
                ),
                (
                    "I confirm Mara Venn's pending promotion and authorize "
                    "the chamber to complete it when she enters."
                ),
                (
                    "I keep the active feed on the Promotion Chamber and wait "
                    "for Mara Venn's decision without adding another order."
                ),
                "/defer",
            ),
        )
        await wait_for_generated_candidate_pack()

        selected = await execute(
            "mara_castor_synthesis",
            "/master synthesis Mara Venn from Castor",
        )
        await advance_until(
            "mara_synthesis",
            lambda value: (
                value["mara"]["level"] >= 20 and value["castor"]["status"] == "culled"
            ),
            selected,
            (
                (
                    "I confirm the pending synthesis of Castor into Mara Venn "
                    "and keep the feed on the Synthesis Chamber."
                ),
                (
                    "I keep watching the confirmed Synthesis Chamber operation "
                    "without speaking for Castor, Mara Venn, or Iselle."
                ),
            ),
        )

        released = await execute(
            "mara_synthesis_release",
            (
                "I keep the active feed on Mara Venn as the Synthesis Chamber "
                "opens, and wait for her to return to Niflheim's lobby before "
                "making another selection."
            ),
        )
        await advance_until(
            "mara_return_to_lobby",
            lambda value: value["mara"]["location"] == "niflheim_lobby",
            released,
            (
                (
                    "I continue watching Mara Venn without issuing another "
                    "command while the open route back to the lobby remains "
                    "available."
                ),
                "/defer",
            ),
        )

        selected = await execute(
            "mara_select_three_star",
            (
                "I select Mara Venn for promotion again and turn the Promotion "
                "Chamber's pale light on her in the lobby."
            ),
        )
        await advance_until(
            "mara_three_star_promotion",
            lambda value: (
                value["mara"]["stars"] == 3 and value["mara"]["level"] == 30
            ),
            selected,
            (
                (
                    "I inspect the authorized requirements and effects of "
                    "Mara Venn's pending three-star promotion without spending "
                    "another stone."
                ),
                (
                    "I confirm Mara Venn's pending promotion and authorize "
                    "the chamber to complete it when she enters."
                ),
                (
                    "I keep the active feed on the Promotion Chamber and wait "
                    "for Mara Venn's decision without adding another order."
                ),
                "/defer",
            ),
        )
        await execute(
            "mara_after",
            (
                "I keep the feed on Mara Venn and inspect her after the "
                "Promotion Chamber releases her at three stars."
            ),
        )
        await execute("mara_after_followup", "/defer")
    except Exception:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
    finally:
        await engine.close()

    final_checkpoint = engine.load_latest(SOURCE_SESSION_ID)
    final_state = _state_summary(final_checkpoint)
    generated_pack_id = str(final_state["mara"]["sprite_set_id"])
    generated_variant_keys = (
        engine.image_generation.store.sprite_variant_keys(
            session_id=SOURCE_SESSION_ID,
            character_id=MARA_ID,
            sprite_pack_id=generated_pack_id,
        )
    )
    expected = {
        "renna_before": ("Renna Holt", "osa_vnset_veiled_feminine_v1"),
        "renna_after": ("Renna Holt", "osa_vnset_renna_holt_v1"),
        "mara_before": ("Mara Venn", "osa_vnset_veiled_feminine_v1"),
        "mara_after": ("Mara Venn", generated_pack_id),
    }
    transition_checks: dict[str, bool] = {}
    for label, (character_name, identity_handle) in expected.items():
        transition_checks[label] = any(
            (
                deck["action_label"] == label
                or deck["action_label"].startswith(label + "_followup")
            )
            and _deck_has_sprite_transition(
                deck,
                character_name=character_name,
                identity_handle=identity_handle,
            )
            for deck in decks
        )
    promotion_stage_sha256 = "c73621400a8d9c960a38391816c3fe16f57d5e04fb00e4bc2968d7fdeb07512a"
    identity_reveal_checks = {
        "renna_promotion_old_flash_new": any(
            (
                deck["action_label"] == "renna_select_promotion"
                or deck["action_label"].startswith("renna_promotion_followup_")
            )
            and _deck_has_committed_identity_reveal(
                deck,
                character_name="Renna Holt",
                before_identity_handle="osa_vnset_veiled_feminine_v1",
                after_identity_handle="osa_vnset_renna_holt_v1",
                expected_stage_sha256=promotion_stage_sha256,
            )
            for deck in decks
        ),
        "mara_promotion_old_flash_new": any(
            (
                deck["action_label"] == "mara_select_three_star"
                or deck["action_label"].startswith("mara_three_star_promotion_followup_")
            )
            and _deck_has_committed_identity_reveal(
                deck,
                character_name="Mara Venn",
                before_identity_handle="osa_vnset_veiled_feminine_v1",
                after_identity_handle=generated_pack_id,
                expected_stage_sha256=promotion_stage_sha256,
            )
            for deck in decks
        ),
    }
    synthesis_stage_sha256 = "668e6fc4c1e7ac9b79601e6024aaf97f3565901ba9cbacc0aeac9e50d1d4afc1"
    fixed_stage_checks = {
        "synthesis_deck_uses_fixed_chamber": any(
            (
                deck["action_label"] == "mara_castor_synthesis"
                or deck["action_label"].startswith("mara_synthesis_followup_")
            )
            and _deck_uses_only_stage(
                deck,
                stage_sha256=synthesis_stage_sha256,
            )
            for deck in decks
        ),
    }

    manifest_paths = [run_dir / deck["manifest_path"] for deck in decks]
    slideshow_dir = None
    slideshow_cards: tuple[Path, ...] = ()
    if manifest_paths:
        slideshow_dir, slideshow_cards = export_vn_playtest_slideshow(
            run_dir,
            deck_manifest_paths=manifest_paths,
        )

    agent_calls = [
        call
        for call in llm_calls
        if call["role"] in {"agent", "agent_standard", "agent_convenience"}
    ]
    narrator_calls = [call for call in llm_calls if call["role"].startswith("narrator")]
    router_calls = [call for call in llm_calls if call["role"] == "event_router"]
    checks = {
        "live_run_completed_without_exception": not error,
        "real_router_outputs_recorded": bool(router_calls),
        "real_narrator_outputs_recorded": bool(narrator_calls),
        "character_agents_used_luna": bool(agent_calls)
        and all(
            "gpt-5.6-luna" in str(call.get("model") or "").casefold()
            for call in agent_calls
        ),
        "no_llm_received_image_bytes": all(
            not call["contains_image_bytes"] for call in llm_calls
        ),
        "renna_reached_two_stars": final_state["renna"]["stars"] == 2,
        "mara_reached_three_stars_level_30": (
            final_state["mara"]["stars"] == 3 and final_state["mara"]["level"] == 30
        ),
        "castor_was_consumed_by_live_synthesis": (
            final_state["castor"]["status"] == "culled"
        ),
        "renna_before_narrator_deck_uses_veil": transition_checks["renna_before"],
        "renna_after_narrator_deck_uses_authored_sprite": transition_checks[
            "renna_after"
        ],
        "mara_before_narrator_deck_uses_veil": transition_checks["mara_before"],
        "mara_after_narrator_deck_uses_generated_sprite": transition_checks[
            "mara_after"
        ],
        "mara_candidate_pack_generated_in_this_run": (
            "neutral" in generated_variant_keys
            and len(generated_variant_keys) >= 2
        ),
        "character_agent_presentation_footer_recorded": any(
            "<presentation>" in str(call.get("raw_output") or "")
            for call in agent_calls
        ),
        **identity_reveal_checks,
        **fixed_stage_checks,
        "chronological_slideshow_exported": bool(slideshow_cards),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "end_to_end_cli_enginebridge_playtest",
        "session_id": SOURCE_SESSION_ID,
        "story_id": STORY_ID,
        "story_seed": {
            "path": str(story_dir.relative_to(run_dir)),
            "checkpoint_sha256": _sha256(story_dir / "ckpt_0000.json"),
        },
        "candidate_generation": {
            "source": "fresh isolated production image queue",
            "generated_variant_keys": list(generated_variant_keys),
            "prewarm_wait": candidate_wait,
            "reason": (
                "exercise the two-star prewarm and three-star generated reveal "
                "without reusing an earlier sprite-job database"
            ),
        },
        "runtime_llm_vision_calls": 0,
        "model_roles": {
            role: config.model_for_role(role)
            for role in (
                "event_router",
                "narrator",
                "agent",
                "agent_standard",
                "agent_convenience",
                "image_director",
            )
        },
        "transcript": transcript,
        "llm_call_files": [
            f"llm_calls/{call['index']:03d}_{call['role']}.json" for call in llm_calls
        ],
        "response_files": [
            f"responses/{response['index']:03d}_{response['action_label']}.json"
            for response in responses
        ],
        "decks": decks,
        "transition_checks": transition_checks,
        "identity_reveal_checks": identity_reveal_checks,
        "fixed_stage_checks": fixed_stage_checks,
        "final_state": final_state,
        "checks": checks,
        "slideshow": {
            "directory": (
                str(slideshow_dir.relative_to(run_dir)) if slideshow_dir else ""
            ),
            "card_count": len(slideshow_cards),
        },
        "error": error,
    }
    _write_json(run_dir / "report.json", report)
    (run_dir / "transcript.txt").write_text(
        "\n\n".join(
            f"$ {item['input']}\n{item['output'].strip()}" for item in transcript
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        default=DEFAULT_WINDOWS_ROOT,
    )
    parser.add_argument(
        "--sprite-wait-seconds",
        type=float,
        default=900.0,
        help=(
            "Maximum time to observe the production prewarm until neutral "
            "and one alternate generated sprite are durable."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir, report = asyncio.run(_run(args))
    windows_copy = ""
    slideshow_relative = report["slideshow"]["directory"]
    if slideshow_relative:
        source = run_dir / slideshow_relative
        args.windows_root.mkdir(parents=True, exist_ok=True)
        destination = args.windows_root / run_dir.name
        shutil.copytree(source, destination)
        windows_copy = str(destination)
    passed = all(report["checks"].values())
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "windows_slideshow": windows_copy,
                "all_checks_passed": passed,
                "checks": report["checks"],
                "error": report["error"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
