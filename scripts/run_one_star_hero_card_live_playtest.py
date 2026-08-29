#!/usr/bin/env python3
"""Live CLI/EngineBridge proof for One-Star summon and formation cards.

The run is isolated from production sessions.  It copies the reviewed story,
claims the Master through the normal CLI contract, opens the authored summon,
and confirms the opening party through the normal router/agent/narrator loop.
Every delivered visual-novel page and every model call is retained for review;
image bytes are never included in a model request.
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
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.bot.engine_bridge import EngineBridge
from app.engine.one_star_adapter import load_one_star_account, load_one_star_hero
from app.engine.one_star_hero_cards import (
    committed_one_star_hero_card_event,
    new_one_star_hero_card_events,
)
from app.llm.config import LLMConfig, live_play_required_roles
from app.schemas.checkpoint import CheckpointFile
from app.schemas.responses import TurnResponse
from scripts.export_vn_playtest_slideshow import export_vn_playtest_slideshow
from scripts.play import CLIState


STORY_ID = "one_star_ascension_s1"
SESSION_ID = "one-star-hero-cards-live-20260829"
MASTER_ID = "the_master"
SOURCE_STORY_DIR = REPO_ROOT / "app/storage/stories" / STORY_ID
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "app/storage/playtest_reports/one_star_hero_cards_20260829"
)
DEFAULT_WINDOWS_ROOT = Path(
    "/mnt/c/Users/danim/Pictures/Ayoa/OneStarHeroCardLivePlaytest_20260829"
)


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


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_playtest_story(run_dir: Path) -> Path:
    if not (SOURCE_STORY_DIR / "ckpt_0000.json").is_file():
        raise RuntimeError("One-Star story seed is missing")
    story_dir = run_dir / "stories" / STORY_ID
    shutil.copytree(SOURCE_STORY_DIR, story_dir)
    return story_dir


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


def _configure_live_models() -> LLMConfig:
    env_path = REPO_ROOT / ".env"
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


def _hero_state(checkpoint: CheckpointFile) -> list[dict[str, object]]:
    heroes: list[dict[str, object]] = []
    for character in checkpoint.characters:
        hero = load_one_star_hero(character)
        if hero is None or not hero.acquisition_event_id:
            continue
        heroes.append({
            "character_id": character.character_id,
            "name": character.name,
            "status": character.status.value,
            "location": character.location,
            "current_stars": hero.current_stars,
            "acquisition_event_id": hero.acquisition_event_id,
        })
    return heroes


def _state_summary(checkpoint: CheckpointFile) -> dict[str, object]:
    _owner, account = load_one_star_account(checkpoint)
    pending = account.state.pending_operation
    mission = account.state.active_mission
    return {
        "checkpoint_id": f"ckpt_{checkpoint.session.turn_index:04d}",
        "turn_index": checkpoint.session.turn_index,
        "heroes": _hero_state(checkpoint),
        "pending_operation": (
            pending.model_dump(mode="json") if pending is not None else None
        ),
        "active_mission": (
            mission.model_dump(mode="json") if mission is not None else None
        ),
        "committed_event_ids": list(account.state.applied_event_fingerprints),
    }


def _latest_card_event(checkpoint: CheckpointFile, kind: str):
    for canonical_event in reversed(checkpoint.canonical_events):
        event = committed_one_star_hero_card_event(
            checkpoint,
            canonical_event.event_id,
        )
        if event is not None and event.kind == kind:
            return event
    return None


def _system_panel_card_indexes(manifest: dict[str, Any]) -> list[int]:
    card_index = 0
    result: list[int] = []
    for section in manifest["identity"]["sections"]:
        page_count = len(section["pages"])
        if section["card_style"] == "system_panel":
            result.extend(range(card_index, card_index + page_count))
        card_index += page_count
    return result


class RecordingCLIState(CLIState):
    """CLI delivery that records decks and skips the interactive image pager."""

    def __init__(
        self,
        *args: Any,
        run_dir: Path,
        current_action: dict[str, str],
        deliveries: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._run_dir = run_dir
        self._current_action = current_action
        self._deliveries = deliveries

    async def _play_visual_novel_deck(
        self,
        deck,
        *,
        character_id: str,
    ) -> None:
        record = {
            "index": len(self._deliveries) + 1,
            "action_label": self._current_action["label"],
            "character_id": character_id,
            "deck_id": deck.deck_id,
            "manifest_path": str(deck.manifest_path.relative_to(self._run_dir)),
            "cards": [
                {
                    "index": card.index,
                    "speaker": card.speaker,
                    "kind": card.kind,
                    "accessible_text": card.accessible_text,
                    "image_path": str(card.image_path.relative_to(self._run_dir)),
                    "image_sha256": hashlib.sha256(card.image_bytes).hexdigest(),
                }
                for card in deck.cards
            ],
        }
        self._deliveries.append(record)
        for card in deck.cards:
            print(card.accessible_text)


async def _run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"one_star_hero_cards_live_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("llm_calls", "responses", "snapshots"):
        (run_dir / directory).mkdir()
    story_dir = _copy_playtest_story(run_dir)
    prompt_sha256_before = _tree_sha256(REPO_ROOT / "app/prompts")

    logging.basicConfig(
        filename=run_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    config = _configure_live_models()
    engine = EngineBridge(
        stories_dir=str(run_dir / "stories"),
        sessions_dir=str(run_dir / "sessions"),
        prompts_dir=str(REPO_ROOT / "app/prompts"),
        llm_config=config,
    )

    current_action = {"label": "startup", "input": ""}
    llm_calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    prepared_decks: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    error = ""

    real_complete = engine.client.complete

    async def recording_complete(*call_args, **kwargs):
        role = str(kwargs.get("role") or (call_args[0] if call_args else ""))
        messages = kwargs.get("messages") or []
        index = len(llm_calls) + 1
        record: dict[str, Any] = {
            "index": index,
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
            _write_json(run_dir / "llm_calls" / f"{index:03d}_{role}.json", record)
            raise
        record.update({
            "model": result.model,
            "usage": dict(result.usage or {}),
            "raw_output": result.content,
            "parsed_output": _jsonable(result.parsed),
            "reasoning_summaries": list(result.reasoning_summaries or []),
        })
        llm_calls.append(record)
        _write_json(run_dir / "llm_calls" / f"{index:03d}_{role}.json", record)
        return result

    engine.client.complete = recording_complete  # type: ignore[method-assign]

    def record_response(kind: str, response: TurnResponse) -> None:
        record = {
            "index": len(responses) + 1,
            "kind": kind,
            "action_label": current_action["label"],
            "player_input": current_action["input"],
            "response": response.model_dump(mode="json"),
        }
        responses.append(record)
        _write_json(
            run_dir / "responses" / f"{record['index']:03d}_{current_action['label']}.json",
            record,
        )

    real_begin = engine.run_begin_turn

    async def recording_begin(*call_args, **kwargs):
        response = await real_begin(*call_args, **kwargs)
        record_response("run_begin_turn", response)
        return response

    engine.run_begin_turn = recording_begin  # type: ignore[method-assign]
    real_run_turn = engine.run_turn

    async def recording_run_turn(*call_args, **kwargs):
        response = await real_run_turn(*call_args, **kwargs)
        record_response("run_turn", response)
        return response

    engine.run_turn = recording_run_turn  # type: ignore[method-assign]
    real_prepare_deck = engine.prepare_visual_novel_deck

    async def recording_prepare_deck(*call_args, **kwargs):
        deck = await real_prepare_deck(*call_args, **kwargs)
        checkpoint_id = str(kwargs.get("checkpoint_id") or "")
        pov_character_id = str(kwargs.get("pov_character_id") or "")
        render = kwargs.get("render")
        checkpoint = engine.load_checkpoint(SESSION_ID, checkpoint_id)
        previous = engine._previous_visual_novel_checkpoint(
            session_id=SESSION_ID,
            checkpoint_id=checkpoint_id,
        )
        card_events = new_one_star_hero_card_events(checkpoint, previous)
        event_ids = [event.event_id for event in card_events]
        event_kinds = [event.kind for event in card_events]
        manifest = json.loads(deck.manifest_path.read_text(encoding="utf-8"))
        styles = [
            section["card_style"]
            for section in manifest["identity"]["sections"]
        ]
        expected_prefix: list[str] = []
        inserted: set[str] = set()
        event_page_counts = {
            event.event_id: (len(event.characters) + 4) // 5
            for event in card_events
        }
        for segment in render.segments if render is not None else ():
            expected_prefix.extend("adv" for _page in segment.pages)
            if segment.rendered_event_id in event_page_counts and (
                segment.rendered_event_id not in inserted
            ):
                expected_prefix.extend(
                    ["system_panel"]
                    * event_page_counts[segment.rendered_event_id]
                )
                inserted.add(segment.rendered_event_id)
        restarted = engine.visual_novel_renderer.load_deck(deck.deck_id)
        record = {
            "index": len(prepared_decks) + 1,
            "action_label": current_action["label"],
            "checkpoint_id": checkpoint_id,
            "pov_character_id": pov_character_id,
            "deck_id": deck.deck_id,
            "manifest_path": str(deck.manifest_path.relative_to(run_dir)),
            "render": render.model_dump(mode="json") if render is not None else {},
            "new_card_event_ids": event_ids,
            "new_card_event_kinds": event_kinds,
            "section_styles": styles,
            "expected_section_prefix": expected_prefix,
            "immediate_card_order": styles[: len(expected_prefix)] == expected_prefix,
            "restart_valid": restarted is not None,
            "restart_bytes_identical": (
                restarted is not None
                and [card.image_bytes for card in restarted.cards]
                == [card.image_bytes for card in deck.cards]
            ),
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
        prepared_decks.append(record)
        return deck

    engine.prepare_visual_novel_deck = (  # type: ignore[method-assign]
        recording_prepare_deck
    )

    state: RecordingCLIState | None = None

    async def execute(label: str, line: str) -> dict[str, Any]:
        current_action.update(label=label, input=line)
        before = None
        try:
            before = _state_summary(engine.load_latest(SESSION_ID))
        except FileNotFoundError:
            pass
        output = io.StringIO()
        with redirect_stdout(output):
            assert state is not None
            await state.handle_line(line)
        checkpoint = engine.load_latest(SESSION_ID)
        after = _state_summary(checkpoint)
        source_path = max(
            (run_dir / "sessions" / SESSION_ID).glob("ckpt_*.json")
        )
        snapshot_path = (
            run_dir
            / "snapshots"
            / f"{len(transcript) + 1:02d}_{label}__{source_path.name}"
        )
        shutil.copy2(source_path, snapshot_path)
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
        mission = after["active_mission"]
        print(
            f"[{label}] turn={after['turn_index']} "
            f"heroes={len(after['heroes'])} mission={bool(mission)}",
            flush=True,
        )
        return after

    await engine.start()
    try:
        engine.create_empty_session(SESSION_ID)
        state = RecordingCLIState(
            engine,
            SESSION_ID,
            "",
            run_dir=run_dir,
            current_action=current_action,
            deliveries=deliveries,
        )
        state.one_shot_mode = True

        await execute("setup_story", f"/story start {STORY_ID}")
        await execute("setup_join", f"/join {MASTER_ID}")
        await execute("opening_summon", "/begin")

        opening_checkpoint = engine.load_latest(SESSION_ID)
        summon_event = _latest_card_event(opening_checkpoint, "summon")
        if summon_event is None:
            raise RuntimeError("opening did not commit a summon card event")
        hero_names = [character.name for character in summon_event.characters]
        if not hero_names:
            raise RuntimeError("opening summon produced an empty roster")
        formation_labels = ("front", "middle", "rear", "reserve", "reserve")
        formation = ", ".join(
            f"{name} in {formation_labels[index]}"
            for index, name in enumerate(hero_names)
        )
        line = (
            "I select Floor 1 and the newly summoned Heroes in their displayed "
            f"order, set {formation}, and confirm the formation through the gate."
        )
        current = await execute("select_and_confirm_formation", line)
        for attempt in range(1, args.max_followups + 1):
            if current["active_mission"] is not None:
                break
            current = await execute(f"formation_followup_{attempt}", "/defer")
        if current["active_mission"] is None:
            raise RuntimeError(
                "formation did not commit a mission after the allowed follow-ups"
            )
    except Exception:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
    finally:
        await engine.close()

    final_checkpoint = engine.load_latest(SESSION_ID)
    final_state = _state_summary(final_checkpoint)
    summon_event = _latest_card_event(final_checkpoint, "summon")
    mission_event = _latest_card_event(final_checkpoint, "mission_start")
    system_panels: list[dict[str, Any]] = []
    for deck in prepared_decks:
        indexes = _system_panel_card_indexes(deck["manifest"])
        for card_index in indexes:
            card = deck["cards"][card_index]
            image_path = run_dir / card["image_path"]
            with Image.open(image_path) as opened:
                image_size = list(opened.size)
                image_info = dict(opened.info)
            system_panels.append({
                "action_label": deck["action_label"],
                "checkpoint_id": deck["checkpoint_id"],
                "pov_character_id": deck["pov_character_id"],
                "deck_id": deck["deck_id"],
                "new_card_event_ids": deck["new_card_event_ids"],
                "new_card_event_kinds": deck["new_card_event_kinds"],
                "immediate_card_order": deck["immediate_card_order"],
                "restart_valid": deck["restart_valid"],
                "restart_bytes_identical": deck["restart_bytes_identical"],
                "accessible_text": card["accessible_text"],
                "image_path": card["image_path"],
                "image_sha256": card["image_sha256"],
                "image_size": image_size,
                "image_info": image_info,
            })

    summon_panels = [
        panel
        for panel in system_panels
        if panel["accessible_text"].startswith(
            "System panel — Heroes acquired"
        )
    ]
    mission_panels = [
        panel
        for panel in system_panels
        if panel["accessible_text"].startswith(
            "System panel — Formation confirmed"
        )
    ]
    frame_id = "osa_hero_card_frame_obsidian_orrery_v1"
    source_tokens = [frame_id, "osa_hero_card_portrait_", "one_star_hero_cards"]
    panel_payloads = [
        (run_dir / panel["image_path"]).read_bytes()
        + panel["accessible_text"].encode("utf-8")
        for panel in system_panels
    ]
    source_scan_payloads = [
        *panel_payloads,
        *(
            json.dumps(deck["manifest"], ensure_ascii=False).encode("utf-8")
            for deck in prepared_decks
            if "system_panel" in deck["section_styles"]
        ),
        "\n".join(item["output"] for item in transcript).encode("utf-8"),
    ]
    mission_party_ids = (
        list(final_state["active_mission"]["party_ids"])
        if final_state["active_mission"] is not None
        else []
    )
    summon_ids = (
        [character.character_id for character in summon_event.characters]
        if summon_event is not None
        else []
    )
    mission_ids = (
        [character.character_id for character in mission_event.characters]
        if mission_event is not None
        else []
    )

    manifest_paths = [
        run_dir / delivery["manifest_path"] for delivery in deliveries
    ]
    slideshow_dir = None
    slideshow_cards: tuple[Path, ...] = ()
    if manifest_paths:
        slideshow_dir, slideshow_cards = export_vn_playtest_slideshow(
            run_dir,
            deck_manifest_paths=manifest_paths,
        )

    prompt_sha256_after = _tree_sha256(REPO_ROOT / "app/prompts")
    checks = {
        "live_run_completed_without_exception": not error,
        "real_router_outputs_recorded": any(
            call["role"] == "event_router" for call in llm_calls
        ),
        "real_narrator_outputs_recorded": any(
            call["role"].startswith("narrator") for call in llm_calls
        ),
        "no_llm_received_image_bytes": all(
            not call["contains_image_bytes"] for call in llm_calls
        ),
        "opening_summon_committed": summon_event is not None,
        "mission_start_committed": mission_event is not None,
        "formation_order_matches_active_mission": (
            bool(mission_ids) and mission_ids == mission_party_ids
        ),
        "formation_preserves_summon_order": (
            bool(summon_ids) and mission_ids == summon_ids
        ),
        "one_summon_board": len(summon_panels) == 1,
        "one_formation_board": len(mission_panels) == 1,
        "boards_are_master_only": bool(system_panels)
        and all(panel["pov_character_id"] == MASTER_ID for panel in system_panels),
        "boards_follow_exact_render_segment": bool(system_panels)
        and all(panel["immediate_card_order"] for panel in system_panels),
        "boards_survive_restart_with_immutable_bytes": bool(system_panels)
        and all(
            panel["restart_valid"] and panel["restart_bytes_identical"]
            for panel in system_panels
        ),
        "boards_are_exact_1024x576_unannotated_pngs": bool(system_panels)
        and all(
            panel["image_size"] == [1024, 576] and not panel["image_info"]
            for panel in system_panels
        ),
        "source_ids_absent_from_panels_and_accessibility": bool(system_panels)
        and all(
            token.encode("utf-8") not in payload
            for token in source_tokens
            for payload in source_scan_payloads
        ),
        "prompts_unchanged_by_playtest": prompt_sha256_before == prompt_sha256_after,
        "chronological_slideshow_contains_every_delivery_page": (
            len(slideshow_cards)
            == sum(len(delivery["cards"]) for delivery in deliveries)
        ),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "live_cli_enginebridge_summon_and_formation",
        "session_id": SESSION_ID,
        "story_id": STORY_ID,
        "story_seed": {
            "path": str(story_dir.relative_to(run_dir)),
            "checkpoint_sha256": _sha256(story_dir / "ckpt_0000.json"),
        },
        "runtime_llm_vision_calls": sum(
            int(call["contains_image_bytes"]) for call in llm_calls
        ),
        "model_roles": {
            role: config.model_for_role(role)
            for role in (
                "event_router",
                "narrator",
                "agent",
                "agent_standard",
                "agent_convenience",
                "character_gen",
                "image_director",
            )
        },
        "transcript": transcript,
        "deliveries": deliveries,
        "prepared_decks": prepared_decks,
        "system_panels": system_panels,
        "summon_event_id": summon_event.event_id if summon_event is not None else "",
        "mission_event_id": mission_event.event_id if mission_event is not None else "",
        "summon_party_ids": summon_ids,
        "mission_party_ids": mission_ids,
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--windows-root", type=Path, default=DEFAULT_WINDOWS_ROOT)
    parser.add_argument(
        "--max-followups",
        type=int,
        default=6,
        help="Maximum live /defer turns allowed for the guide-owned crossing.",
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
    print(json.dumps({
        "run_dir": str(run_dir),
        "windows_slideshow": windows_copy,
        "all_checks_passed": passed,
        "checks": report["checks"],
        "error": report["error"],
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
