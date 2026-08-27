#!/usr/bin/env python3
"""Exercise the generated One-Star VN sprite reveal against a live worker.

This is a coding-time playtest. It advances the focused promotion fixture with
the real One-Star transaction adapter, runs the production sprite prewarm
coordinator, and renders the generated sprite into a deterministic VN card.
No image bytes are sent to a runtime LLM.
"""

from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path.

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.engine.image_generation import (
    ImageGenerationConfig,
    ImageGenerationCoordinator,
)
from app.engine.one_star_adapter import (
    load_one_star_account,
    load_one_star_hero,
    prepare_one_star_transaction,
)
from app.engine.one_star_visuals import (
    generated_sprite_pack_id,
    sprite_set_id_for_viewer,
)
from app.engine.visual_novel_presentation import (
    VisualNovelCardRenderer,
    VisualNovelDeckSection,
)
from app.engine.visual_novel_sprites import (
    resolve_visual_novel_sprite_placements,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.image_generation import ImageGenerationStatus
from app.schemas.narrator import VisualNovelPage, VisualNovelSpriteCue
from app.schemas.one_star import OneStarTransaction
from app.schemas.visual_references import VISUAL_NOVEL_SPRITE_EXPRESSIONS
from scripts.export_vn_playtest_slideshow import (
    export_vn_playtest_slideshow,
)


PLAYTEST_STORY = (
    REPO_ROOT
    / "app/storage/stories/one_star_ascension_s1_promotion_playtest"
)
FACELESS_ID = "promotion_playtest_faceless"
MASTER_ID = "the_master"


def _transaction(*operations: dict[str, object]) -> OneStarTransaction:
    return OneStarTransaction.model_validate({
        "present": True,
        "operations": list(operations),
    })


def _promote(
    checkpoint: CheckpointFile,
    *,
    character_id: str,
    operation_id: str,
) -> CheckpointFile:
    opened = prepare_one_star_transaction(
        checkpoint,
        event_id=f"{operation_id}_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": {
                "operation_id": operation_id,
                "kind": "promotion",
                "participant_ids": [character_id],
                "target_id": character_id,
                "destination": "niflheim_promotion_chamber",
                "opened_at_s": 0,
            },
        }),
        canonical_at_s=0,
        initiating_actor_id=MASTER_ID,
    )
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id=f"{operation_id}_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": operation_id,
        }),
        location_updates={character_id: "niflheim_promotion_chamber"},
        canonical_at_s=0,
        initiating_actor_id=MASTER_ID,
    )
    return resolved.after_checkpoint


def _synthesize_castor_into_mara(
    checkpoint: CheckpointFile,
) -> tuple[CheckpointFile, dict[str, int]]:
    opened = prepare_one_star_transaction(
        checkpoint,
        event_id="mara_castor_synthesis_open",
        transaction=_transaction({
            "operation": "pending_open",
            "pending": {
                "operation_id": "mara_castor_synthesis",
                "kind": "synthesis",
                "participant_ids": ["castor_valebrand"],
                "target_id": FACELESS_ID,
                "destination": "niflheim_synthesis_chamber",
                "opened_at_s": 0,
            },
        }),
        canonical_at_s=0,
        initiating_actor_id=MASTER_ID,
    )
    _owner, account = load_one_star_account(opened.after_checkpoint)
    pending = account.state.pending_operation
    if pending is None or pending.synthesis_preview is None:
        raise RuntimeError("synthesis preview was not authored")
    preview = {
        "offered_xp": pending.synthesis_preview.offered_xp,
        "applied_xp": pending.synthesis_preview.applied_xp,
        "wasted_xp": pending.synthesis_preview.wasted_xp,
    }
    resolved = prepare_one_star_transaction(
        opened.after_checkpoint,
        event_id="mara_castor_synthesis_resolve",
        transaction=_transaction({
            "operation": "pending_resolve",
            "operation_id": "mara_castor_synthesis",
        }),
        location_updates={
            FACELESS_ID: "niflheim_synthesis_chamber",
            "castor_valebrand": "niflheim_synthesis_chamber",
        },
        canonical_at_s=0,
        initiating_actor_id=MASTER_ID,
    )
    return resolved.after_checkpoint, preview


def _character(checkpoint: CheckpointFile, character_id: str):
    return next(
        character
        for character in checkpoint.characters
        if character.character_id == character_id
    )


def _hero_state(
    checkpoint: CheckpointFile,
    character_id: str,
) -> dict[str, object]:
    character = _character(checkpoint, character_id)
    hero = load_one_star_hero(character)
    if hero is None:
        raise RuntimeError(f"missing One-Star state for {character_id}")
    return {
        "character_id": character.character_id,
        "name": character.name,
        "status": character.status.value,
        "stars": hero.current_stars,
        "level": hero.level,
        "experience_points": hero.experience_points,
        "generated_for_summon": hero.generated_for_summon,
        "master_sprite_set_id": sprite_set_id_for_viewer(
            checkpoint,
            viewer_character_id=MASTER_ID,
            character=character,
        ),
    }


def _write_checkpoint(
    checkpoint: CheckpointFile,
    destination: Path,
) -> None:
    destination.write_text(
        checkpoint.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        ) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gateway_health() -> dict[str, Any]:
    url = "http://127.0.0.1:8199/health"
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _contact_sheet(
    variants: list[tuple[str, bytes]],
    destination: Path,
) -> None:
    cell_width = 420
    cell_height = 580
    columns = 4
    rows = 2
    sheet = Image.new(
        "RGB",
        (cell_width * columns, cell_height * rows),
        (24, 28, 36),
    )
    regular = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        22,
    )
    bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        24,
    )
    for index, (expression, data) in enumerate(variants):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * cell_height
        panel = Image.new("RGB", (cell_width, cell_height), (34, 40, 52))
        ImageDraw.Draw(panel).rectangle(
            (cell_width // 2, 0, cell_width, cell_height - 54),
            fill=(226, 229, 232),
        )
        with Image.open(BytesIO(data)) as opened:
            sprite = opened.convert("RGBA")
        sprite.thumbnail((390, 500), Image.Resampling.LANCZOS)
        panel.paste(
            sprite,
            ((cell_width - sprite.width) // 2, 8),
            sprite,
        )
        draw = ImageDraw.Draw(panel)
        draw.rectangle(
            (0, cell_height - 54, cell_width, cell_height),
            fill=(8, 12, 18),
        )
        label = f"Mara Venn — {expression}"
        label_width = draw.textbbox((0, 0), label, font=bold)[2]
        draw.text(
            ((cell_width - label_width) // 2, cell_height - 42),
            label,
            font=bold,
            fill=(248, 248, 248),
        )
        sheet.paste(panel, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 8),
        "Live runtime-generated sweep — dark/light matte review",
        font=regular,
        fill=(255, 255, 255),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=False)


async def _wait_for_jobs(
    coordinator: ImageGenerationCoordinator,
    job_ids: tuple[str, ...],
    *,
    timeout: float,
) -> None:
    if not job_ids:
        return
    await asyncio.gather(*(
        coordinator.wait_for_terminal(job_id, timeout=timeout)
        for job_id in job_ids
    ))


async def _run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    source_path = PLAYTEST_STORY / "ckpt_0000.json"
    if args.resume_dir is not None:
        run_dir = args.resume_dir.resolve()
        if not (run_dir / "image-runtime/jobs.sqlite").is_file():
            raise RuntimeError("resume directory has no durable image store")
        prior_checkpoint = CheckpointFile.model_validate_json(
            (run_dir / "checkpoints/02-mara-2star.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = CheckpointFile.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )
        checkpoint.session.session_id = prior_checkpoint.session.session_id
        checkpoint.session.turn_index = 1
        _write_checkpoint(checkpoint, run_dir / "checkpoints/00-start.json")
        progression: list[dict[str, object]] = [{
            "step": "start",
            "renna": _hero_state(checkpoint, "renna_holt"),
            "mara": _hero_state(checkpoint, FACELESS_ID),
            "castor": _hero_state(checkpoint, "castor_valebrand"),
        }]
        checkpoint = _promote(
            checkpoint,
            character_id="renna_holt",
            operation_id="renna_two_star",
        )
        progression.append({
            "step": "renna_promoted_to_two_star",
            "renna": _hero_state(checkpoint, "renna_holt"),
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/01-renna-2star.json",
        )
        checkpoint = _promote(
            checkpoint,
            character_id=FACELESS_ID,
            operation_id="mara_two_star",
        )
        progression.append({
            "step": "mara_promoted_to_two_star_prewarm_boundary",
            "mara": _hero_state(checkpoint, FACELESS_ID),
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/02-mara-2star.json",
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root / f"one_star_promotion_sprite_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        (run_dir / "prompts").mkdir()
        (run_dir / "raw").mkdir()
        (run_dir / "sprites").mkdir()

        checkpoint = CheckpointFile.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )
        checkpoint.session.session_id = (
            f"one-star-promotion-sprite-{timestamp.lower()}"
        )
        checkpoint.session.turn_index = 1
        _write_checkpoint(checkpoint, run_dir / "checkpoints/00-start.json")

        progression = [{
            "step": "start",
            "renna": _hero_state(checkpoint, "renna_holt"),
            "mara": _hero_state(checkpoint, FACELESS_ID),
            "castor": _hero_state(checkpoint, "castor_valebrand"),
        }]

        checkpoint = _promote(
            checkpoint,
            character_id="renna_holt",
            operation_id="renna_two_star",
        )
        progression.append({
            "step": "renna_promoted_to_two_star",
            "renna": _hero_state(checkpoint, "renna_holt"),
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/01-renna-2star.json",
        )

        checkpoint = _promote(
            checkpoint,
            character_id=FACELESS_ID,
            operation_id="mara_two_star",
        )
        progression.append({
            "step": "mara_promoted_to_two_star_prewarm_boundary",
            "mara": _hero_state(checkpoint, FACELESS_ID),
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/02-mara-2star.json",
        )

    for directory in ("prompts", "raw", "sprites"):
        (run_dir / directory).mkdir(exist_ok=True)

    runtime_root = run_dir / "image-runtime"
    coordinator = ImageGenerationCoordinator(
        sessions_dir=run_dir / "sessions",
        config=ImageGenerationConfig.from_environment(
            runtime_root=runtime_root,
        ),
        repo_root=REPO_ROOT,
    )
    await coordinator.start()
    if not coordinator.can_generate_render():
        await coordinator.close()
        raise RuntimeError("live image worker did not pass preflight")

    try:
        mara = _character(checkpoint, FACELESS_ID)
        pack_id = generated_sprite_pack_id(checkpoint, mara)
        existing_jobs = {
            job.request.sprite_expression: job
            for job in coordinator.store.all_jobs()
            if job.request.sprite_pack_id == pack_id
        }
        if "neutral" not in existing_jobs:
            print("admitting generated neutral candidate", flush=True)
            neutral_ids = await coordinator.ensure_visual_novel_sprite_prewarm(
                checkpoint
            )
            if len(neutral_ids) != 1:
                raise RuntimeError(
                    f"expected one neutral admission, got {len(neutral_ids)}"
                )
        else:
            print("resuming durable generated neutral candidate", flush=True)
            neutral_ids = (existing_jobs["neutral"].job_id,)
        await _wait_for_jobs(
            coordinator,
            neutral_ids,
            timeout=args.job_timeout,
        )
        neutral = coordinator.store.get(neutral_ids[0])
        if neutral is None or neutral.status != ImageGenerationStatus.succeeded:
            error = "missing" if neutral is None else neutral.error_code
            raise RuntimeError(f"neutral generation failed: {error}")

        print("neutral succeeded; waiting for seven referenced variants", flush=True)
        # Neutral completion expands the seven reference-guided jobs before it
        # signals terminal waiters. The public prewarm call therefore normally
        # returns no new ids here: the durable store is the source of truth.
        await coordinator.ensure_visual_novel_sprite_prewarm(checkpoint)
        generated_jobs = {
            job.request.sprite_expression: job
            for job in coordinator.store.all_jobs()
            if job.request.sprite_pack_id == pack_id
        }
        sweep_ids = tuple(
            generated_jobs[expression].job_id
            for expression in VISUAL_NOVEL_SPRITE_EXPRESSIONS[1:]
            if expression in generated_jobs
        )
        if len(sweep_ids) != 7:
            raise RuntimeError(
                f"expected seven durable sweep jobs, got {len(sweep_ids)}"
            )
        await _wait_for_jobs(
            coordinator,
            sweep_ids,
            timeout=args.job_timeout,
        )
        # Re-entering the ordinary prewarm boundary materializes every terminal
        # successful job into immutable normalized PNG variants.
        await coordinator.ensure_visual_novel_sprite_prewarm(checkpoint)

        jobs = {
            job.request.sprite_expression: job
            for job in coordinator.store.all_jobs()
            if job.request.sprite_pack_id == pack_id
        }
        ordered_jobs = [
            jobs[expression]
            for expression in VISUAL_NOVEL_SPRITE_EXPRESSIONS
            if expression in jobs
        ]
        for job in ordered_jobs:
            (run_dir / "prompts" / f"{job.request.sprite_expression}.txt").write_text(
                job.request.prompt + "\n",
                encoding="utf-8",
            )
            if job.status == ImageGenerationStatus.succeeded:
                raw = coordinator.resolve_job_media(job)
                (run_dir / "raw" / f"{job.request.sprite_expression}.webp").write_bytes(
                    raw.data
                )

        resolved_variants: list[tuple[str, bytes]] = []
        variant_records: list[dict[str, object]] = []
        for expression in VISUAL_NOVEL_SPRITE_EXPRESSIONS:
            resolved = coordinator.resolve_visual_novel_sprite_variant(
                session_id=checkpoint.session.session_id,
                character_id=FACELESS_ID,
                sprite_pack_id=pack_id,
                expression=expression,
            )
            if resolved is None:
                variant_records.append({
                    "expression": expression,
                    "resolved": False,
                    "job_status": jobs[expression].status.value,
                    "error_code": jobs[expression].error_code,
                })
                continue
            handle, media, source_facing = resolved
            sprite_path = run_dir / "sprites" / f"{expression}.png"
            sprite_path.write_bytes(media.data)
            resolved_variants.append((expression, media.data))
            variant_records.append({
                "expression": expression,
                "resolved": True,
                "job_status": jobs[expression].status.value,
                "variant_handle": handle,
                "source_facing": source_facing,
                "sha256": media.sha256,
                "width": media.width,
                "height": media.height,
                "relative_path": str(sprite_path.relative_to(run_dir)),
            })
        if not resolved_variants or resolved_variants[0][0] != "neutral":
            raise RuntimeError("generated neutral was not materialized")

        sheet_path = run_dir / "mara-venn-generated-sweep.png"
        _contact_sheet(resolved_variants, sheet_path)

        checkpoint, preview = _synthesize_castor_into_mara(checkpoint)
        progression.append({
            "step": "castor_synthesized_into_mara",
            "preview": preview,
            "mara": _hero_state(checkpoint, FACELESS_ID),
            "castor": _hero_state(checkpoint, "castor_valebrand"),
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/03-mara-level20-after-synthesis.json",
        )

        checkpoint = _promote(
            checkpoint,
            character_id=FACELESS_ID,
            operation_id="mara_three_star",
        )
        final_mara = _hero_state(checkpoint, FACELESS_ID)
        progression.append({
            "step": "mara_promoted_to_three_star_generated_reveal",
            "mara": final_mara,
        })
        _write_checkpoint(
            checkpoint,
            run_dir / "checkpoints/04-mara-3star-level30.json",
        )
        if final_mara["master_sprite_set_id"] != pack_id:
            raise RuntimeError("three-star Mara did not select generated pack")

        page = VisualNovelPage(
            kind="dialogue",
            speaker="Mara Venn",
            text=(
                "I remember enough to know this face is mine. Whatever the "
                "System calls me now, it does not get to take that away."
            ),
            sprites=[VisualNovelSpriteCue(
                character="Mara Venn",
                expression="neutral",
            )],
        )
        placements = resolve_visual_novel_sprite_placements(
            checkpoint=checkpoint,
            viewer_character_id=MASTER_ID,
            page=page,
            generation=coordinator,
        )
        if len(placements) != 1 or placements[0].identity_handle != pack_id:
            raise RuntimeError("VN placement did not resolve generated Mara")

        stage_path = (
            PLAYTEST_STORY
            / "visual-references/locations/niflheim/"
            "lobby_1f_open_air_courtyard_v1.png"
        )
        renderer = VisualNovelCardRenderer(run_dir / "presentation")
        deck = renderer.render_deck([
            VisualNovelDeckSection(
                pages=(page,),
                stage_path=stage_path,
                sprite_placements=placements,
            )
        ])
        card = deck.cards[0]
        manifest = json.loads(deck.manifest_path.read_text(encoding="utf-8"))
        manifest_sprite = manifest["identity"]["sections"][0]["sprites"][0]
        checks = {
            "eight_jobs_admitted": len(ordered_jobs) == 8,
            "neutral_succeeded": jobs["neutral"].status
            == ImageGenerationStatus.succeeded,
            "generated_variants_resolved": len(resolved_variants),
            "renna_uses_authored_sprite_at_two_star": (
                progression[1]["renna"]["master_sprite_set_id"]
                == "osa_vnset_renna_holt_v1"
            ),
            "mara_remains_veiled_at_two_star": (
                progression[2]["mara"]["master_sprite_set_id"]
                == "osa_vnset_veiled_feminine_v1"
            ),
            "mara_reaches_three_star_level30": (
                final_mara["stars"] == 3 and final_mara["level"] == 30
            ),
            "mara_uses_generated_pack_at_three_star": (
                final_mara["master_sprite_set_id"] == pack_id
            ),
            "card_speaker_is_exact_roster_name": card.speaker == "Mara Venn",
            "accessible_text_uses_exact_roster_name": card.accessible_text.startswith(
                "Mara Venn:"
            ),
            "transcript_uses_exact_roster_name": deck.transcript.startswith(
                "Mara Venn:"
            ),
            "manifest_binds_generated_pack": (
                manifest_sprite["identity_handle"] == pack_id
            ),
        }
        if not all(
            value if isinstance(value, bool) else value >= 1
            for value in checks.values()
        ):
            raise RuntimeError(f"playtest checks failed: {checks}")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": _sha256(source_path),
            "session_id": checkpoint.session.session_id,
            "gateway_health": _gateway_health(),
            "runtime_llm_vision_calls": 0,
            "generation_seconds": round(
                max(job.completed_at or job.updated_at for job in ordered_jobs)
                - min(job.created_at for job in ordered_jobs),
                3,
            ),
            "sprite_pack_id": pack_id,
            "progression": progression,
            "variants": variant_records,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "expression": job.request.sprite_expression,
                    "status": job.status.value,
                    "error_code": job.error_code,
                    "attempts": job.attempts,
                    "prompt_sha256": job.request.prompt_sha256,
                    "reference_count": len(job.request.reference_inputs),
                    "model_id": job.request.model_id,
                    "model_revision": job.request.model_revision,
                }
                for job in ordered_jobs
            ],
            "checks": checks,
            "card": {
                "deck_id": deck.deck_id,
                "speaker": card.speaker,
                "accessible_text": card.accessible_text,
                "transcript": deck.transcript,
                "image_path": str(card.image_path.relative_to(run_dir)),
                "image_sha256": hashlib.sha256(card.image_bytes).hexdigest(),
                "manifest_path": str(deck.manifest_path.relative_to(run_dir)),
            },
            "contact_sheet": {
                "path": str(sheet_path.relative_to(run_dir)),
                "sha256": _sha256(sheet_path),
            },
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        await coordinator.close()

    return run_dir, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "app/storage/playtest_reports",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Resume one preserved run after interruption or harness failure.",
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        help="Optional parent folder receiving one hash-identical review copy.",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=1800.0,
        help="Maximum seconds to wait for each admitted generation job.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()
    run_dir, report = asyncio.run(_run(args))
    slideshow_directory, slideshow_cards = export_vn_playtest_slideshow(run_dir)
    report["slideshow"] = {
        "directory": str(slideshow_directory.relative_to(run_dir)),
        "index": str(
            (slideshow_directory / "index.json").relative_to(run_dir)
        ),
        "card_count": len(slideshow_cards),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    windows_copy = None
    if args.windows_root is not None:
        args.windows_root.mkdir(parents=True, exist_ok=True)
        windows_copy = args.windows_root / run_dir.name
        shutil.copytree(run_dir, windows_copy)
    print(json.dumps({
        "run_dir": str(run_dir),
        "windows_copy": str(windows_copy) if windows_copy else "",
        "all_checks_passed": all(
            value if isinstance(value, bool) else value >= 1
            for value in report["checks"].values()
        ),
        "resolved_variants": report["checks"]["generated_variants_resolved"],
        "generation_seconds": report["generation_seconds"],
        "deck_id": report["card"]["deck_id"],
        "slideshow": str(slideshow_directory),
        "slideshow_card_count": len(slideshow_cards),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
