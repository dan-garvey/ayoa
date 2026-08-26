from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.engine.image_director import (
    ImageDirector,
    PublicCharacterVisual,
    SelectableVisualReference,
    VisibleEventProjection,
    build_projection_groups,
    build_render_batch_projection_groups,
    image_loadout_for_character,
    projection_checkpoint_snapshot,
    source_event_fingerprint,
)
from app.engine.event_image_sidecar import EventImageSidecar
from app.engine.image_generation import (
    ImageDeliveryTarget,
    ImageGenerationConfig,
    ImageGenerationCoordinator,
    _authored_identity_reroll_input,
    build_diffusion_prompt,
)
from app.engine.image_worker_client import ImageWorkerError
from app.engine.image_job_store import ImageJobStore
from app.engine.prompt_manager import PromptManager
from app.schemas.characters import (
    CharacterRecord,
    CharacterVisuals,
    PlayerSlotKind,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import REDACTED_IMPORT_SENTINEL
from app.schemas.events import ObservableFact
from app.schemas.image_director import ImageDirection, ImageDirectorOutput
from app.schemas.image_generation import (
    FrozenReferenceInput,
    GeneratedImageArtifact,
    IdentityReferenceStatus,
    ImageDeliveryKind,
    ImageGenerationStatus,
    ImageWorkerResult,
)
from app.schemas.state import (
    RenderBufferEntry,
    SessionState,
    StorySetting,
    WorldState,
)
from tests.support.factories import router_output


ONE_STAR_RULESET_ID = "one_star_ascension"


class FakeImageWorker:
    def __init__(self, *, wait: bool = False, error_code: str = "") -> None:
        self.available = True
        self.config = SimpleNamespace(
            model_id="fake/flux-klein",
            model_revision="test-revision",
        )
        self.wait = wait
        self.error_code = error_code
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted = False
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate(self, request, *, output_path):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.wait:
                await self.release.wait()
            if self.aborted:
                raise ImageWorkerError("worker_cancelled")
            if self.error_code:
                raise ImageWorkerError(self.error_code)
            data = _fake_webp(request.width, request.height)
            Path(output_path).write_bytes(data)
            return ImageWorkerResult(
                ok=True,
                sha256=hashlib.sha256(data).hexdigest(),
                mime_type="image/webp",
                width=request.width,
                height=request.height,
                byte_count=len(data),
                generation_seconds=0.01,
            )
        finally:
            self.active -= 1

    async def abort_current(self):
        self.aborted = True
        self.release.set()

    async def close(self):
        self.release.set()


class FailedPreflightWorker(FakeImageWorker):
    async def preflight(self) -> bool:
        return False


class FakeDirectorClient:
    def __init__(
        self,
        output: ImageDirectorOutput | list[ImageDirectorOutput],
    ) -> None:
        self.outputs = output if isinstance(output, list) else [output]
        self.messages = []
        self.calls = 0

    async def complete(self, **kwargs):
        self.messages = kwargs["messages"]
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return SimpleNamespace(parsed=output)


def _checkpoint(session_id: str = "image_test") -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(
            session_id=session_id,
            turn_index=1,
            character_bindings={"alice": "11", "bob": "22"},
        ),
        world_state=WorldState(
            setting=StorySetting(
                genre="rain-washed campus romance",
                era="contemporary",
                tone="earnest",
                premise="Two friends face an uncertain graduation.",
                visual_style="soft anime-inspired cinematic illustration",
            ),
            hidden_lore="PRIVATE WORLD SECRET",
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                public_sheet=PublicSheet(
                    role="student photographer",
                    appearance="short dark hair and a yellow raincoat",
                ),
                visuals=CharacterVisuals(
                    default_loadout="canvas satchel and rain-spotted shoes",
                ),
                private_state=PrivateState(
                    secrets=["PRIVATE ALICE SECRET"],
                    intentions_enabled=True,
                ),
                is_playable=True,
            ),
            CharacterRecord(
                character_id="bob",
                name="Bob",
                public_sheet=PublicSheet(appearance="silver hair"),
                visuals=CharacterVisuals(default_loadout="black formal coat"),
                private_state=PrivateState(secrets=["PRIVATE BOB SECRET"]),
            ),
        ],
    )
    ckpt.session.config.narrative_rules = "PRIVATE NARRATIVE RULE"
    return ckpt


def _projection(
    *,
    transaction_id: str = "tx_1",
    event_id: str = "evt_1",
    event_sequence: int = 0,
    viewers: tuple[str, ...] = ("alice", "bob"),
    source_turn_index: int = 1,
    presentation_mode: str = "prose",
) -> VisibleEventProjection:
    return VisibleEventProjection(
        session_id="image_test",
        transaction_id=transaction_id,
        source_turn_index=source_turn_index,
        event_id=event_id,
        event_sequence=event_sequence,
        event_fingerprint=hashlib.sha256(event_id.encode()).hexdigest(),
        viewer_character_ids=viewers,
        perception_level="direct",
        effective_at_s=12,
        duration_s=3,
        visible_facts=(("Alice steps into the rain.", 0, 3),),
        characters=(
            PublicCharacterVisual(
                character_id="alice",
                name="Alice",
                appearance="short dark hair and a yellow raincoat",
                default_loadout="canvas satchel",
                depiction_policy="normal",
                is_new_character=False,
                has_identity_reference=False,
            ),
            PublicCharacterVisual(
                character_id="bob",
                name="Bob",
                appearance="silver hair",
                default_loadout="black formal coat",
                depiction_policy="normal",
                is_new_character=False,
                has_identity_reference=False,
            ),
        ),
        story_genre="romance",
        story_era="contemporary",
        story_tone="earnest",
        story_premise="Two friends face an uncertain graduation.",
        canonical_event_count=3,
        active_roster_count=2,
        total_roster_count=2,
        engine_visual_style="soft cinematic illustration",
        presentation_mode=presentation_mode,
    )


def _config(tmp_path: Path, **updates) -> ImageGenerationConfig:
    values = {
        "runtime_root": tmp_path / "runtime",
        "queue_limit": 4,
        "per_session_queue_limit": 4,
    }
    values.update(updates)
    return ImageGenerationConfig(**values)


def _target(character_id: str) -> ImageDeliveryTarget:
    return ImageDeliveryTarget(
        pov_character_id=character_id,
        delivery_kind=ImageDeliveryKind.cli,
        delivery={"character_id": character_id},
    )


def _begin(
    coordinator: ImageGenerationCoordinator,
    transaction_id: str,
) -> None:
    coordinator.begin_transaction(
        transaction_id=transaction_id,
        session_id="image_test",
        source_turn_index=1,
        source_checkpoint_sha256="a" * 64,
    )


def _director_attempt(
    coordinator: ImageGenerationCoordinator,
    run_id: str,
) -> int:
    with coordinator.store._connect() as db:
        row = db.execute(
            "SELECT attempts FROM image_director_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise AssertionError(f"director run is unavailable: {run_id}")
    return int(row["attempts"])


def _complete_director(
    coordinator: ImageGenerationCoordinator,
    run_id: str,
    output: ImageDirectorOutput,
):
    return coordinator.store.complete_director_run(
        run_id,
        output,
        attempt=_director_attempt(coordinator, run_id),
    )


def _finalize_director(
    coordinator: ImageGenerationCoordinator,
    run_id: str,
    *,
    projection: VisibleEventProjection,
    admitted_job_ids: list[str],
):
    return coordinator.store.finalize_director_materialization(
        run_id,
        attempt=_director_attempt(coordinator, run_id),
        projection=projection,
        admitted_job_ids=admitted_job_ids,
    )


def _heartbeat_director(
    coordinator: ImageGenerationCoordinator,
    run_id: str,
) -> bool:
    return coordinator.store.heartbeat_director_run(
        run_id,
        attempt=_director_attempt(coordinator, run_id),
    )


def _fail_director(
    coordinator: ImageGenerationCoordinator,
    run_id: str,
    error_code: str,
):
    return coordinator.store.fail_director_run(
        run_id,
        error_code,
        attempt=_director_attempt(coordinator, run_id),
    )


async def _persist_finalized_visual_novel_job(
    coordinator: ImageGenerationCoordinator,
):
    _begin(coordinator, "tx_restart")
    projection = _projection(
        transaction_id="tx_restart",
        event_id="evt_restart",
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    direction = ImageDirection(
        kind="establishing",
        title="Restart Courtyard",
        subject_character_ids=[],
        scene_prompt="A rain-dark courtyard beneath stone arcades.",
    )
    run = coordinator.store.enqueue_director_run(projection)
    assert coordinator.store.claim_next_director_run() is not None
    output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    _complete_director(coordinator, run.run_id, output)
    job = await coordinator.enqueue_direction(
        projection=projection,
        direction=direction,
        request_ordinal=0,
        visual_style="soft cinematic illustration",
        delivery_targets=[],
        director_run_id=run.run_id,
        director_attempt=_director_attempt(coordinator, run.run_id),
    )
    assert job is not None
    _finalize_director(
        coordinator,
        run.run_id,
        projection=projection,
        admitted_job_ids=[job.job_id],
    )
    assert coordinator.store.commit_transaction(
        "tx_restart",
        target_checkpoint_sha256="b" * 64,
    )
    return projection, run, job


async def _wait_until(predicate, *, timeout: float = 2) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError
        await asyncio.sleep(0.02)


def _fake_webp(width: int, height: int) -> bytes:
    encoded_width = width - 1
    encoded_height = height - 1
    payload = b"\x2f" + bytes(
        (
            encoded_width & 0xFF,
            ((encoded_width >> 8) & 0x3F)
            | ((encoded_height & 0x03) << 6),
            (encoded_height >> 2) & 0xFF,
            (encoded_height >> 10) & 0x0F,
        )
    )
    chunk = b"VP8L" + len(payload).to_bytes(4, "little") + payload + b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_projection_groups_equivalent_viewers_and_respects_fact_visibility():
    ckpt = _checkpoint()
    shared = router_output(
        event_id="evt_shared",
        observer_ids=["alice", "bob"],
        facts=[ObservableFact.all("Rain sweeps across the empty courtyard.")],
    )
    grouped = build_projection_groups(
        checkpoint=ckpt,
        event=shared,
        event_sequence=0,
        transaction_id="tx_shared",
        source_turn_index=1,
    )
    assert len(grouped) == 1
    assert grouped[0].viewer_character_ids == ("alice", "bob")

    split = router_output(
        event_id="evt_split",
        observer_ids=["alice", "bob"],
        facts=[
            ObservableFact.all("Rain sweeps across the courtyard."),
            ObservableFact.only("Alice spots a hidden key.", ["alice"]),
        ],
    )
    projections = build_projection_groups(
        checkpoint=ckpt,
        event=split,
        event_sequence=1,
        transaction_id="tx_split",
        source_turn_index=1,
    )
    assert len(projections) == 2
    by_viewer = {projection.viewer_character_ids[0]: projection for projection in projections}
    assert "hidden key" in str(by_viewer["alice"].visible_facts)
    assert "hidden key" not in str(by_viewer["bob"].visible_facts)
    assert "PRIVATE" not in str(projections)


def test_render_batch_keeps_private_povs_separate_and_anchors_latest_event():
    ckpt = _checkpoint()
    shared = router_output(
        event_id="evt_shared",
        observer_ids=["alice", "bob"],
        facts=[ObservableFact.all("Rain sweeps across the courtyard.")],
    )
    private = router_output(
        event_id="evt_private",
        observer_ids=["alice"],
        facts=[ObservableFact.only("Alice spots a hidden key.", ["alice"])],
    )
    ckpt.canonical_events.extend((shared, private))

    projections = build_render_batch_projection_groups(
        checkpoint=ckpt,
        buffered_events_by_pov={
            "alice": [
                RenderBufferEntry(
                    event_id="evt_shared",
                    visible_at_s=0,
                    event_sequence=17,
                ),
                RenderBufferEntry(
                    event_id="evt_private",
                    visible_at_s=1,
                    event_sequence=18,
                ),
            ],
            "bob": [RenderBufferEntry(
                event_id="evt_shared",
                visible_at_s=0,
                event_sequence=17,
            )],
        },
        eligible_viewer_ids={"alice", "bob"},
        transaction_id="tx_batch",
        source_turn_index=1,
    )

    assert len(projections) == 2
    by_viewer = {
        projection.viewer_character_ids[0]: projection
        for projection in projections
    }
    assert by_viewer["alice"].event_id == "evt_private"
    assert by_viewer["alice"].event_sequence == 18
    assert by_viewer["alice"].canonical_event_count == 19
    assert "hidden key" in str(by_viewer["alice"].visible_facts)
    assert by_viewer["bob"].event_id == "evt_shared"
    assert by_viewer["bob"].event_sequence == 17
    assert by_viewer["bob"].canonical_event_count == 18
    assert "hidden key" not in str(by_viewer["bob"].visible_facts)


def test_projection_snapshot_contains_no_private_story_or_character_state():
    source = _checkpoint()
    kept = router_output(event_id="evt_kept", observer_ids=["alice"])
    omitted = router_output(event_id="evt_omitted", observer_ids=["alice"])
    source.canonical_events.extend((kept, omitted))
    source.characters[0].visuals.identity_reference_id = (
        "imgref_PRIVATE_FILE_ID"
    )
    snapshot = projection_checkpoint_snapshot(source, event_ids={"evt_kept"})
    serialized = snapshot.model_dump_json()
    assert "PRIVATE WORLD SECRET" not in serialized
    assert "PRIVATE ALICE SECRET" not in serialized
    assert "PRIVATE NARRATIVE RULE" not in serialized
    assert "imgref_PRIVATE_FILE_ID" not in serialized
    assert "yellow raincoat" in serialized
    assert [event.event_id for event in snapshot.canonical_events] == [
        "evt_kept"
    ]


@pytest.mark.asyncio
async def test_one_star_image_projection_uses_only_current_visible_equipment():
    ckpt = _checkpoint()
    ckpt.session.config.settings.ruleset_id = ONE_STAR_RULESET_ID
    ckpt.characters[0].visuals.default_loadout = (
        "stale authored loadout with retired hidden armor"
    )
    ckpt.characters[0].mechanics = {
        "one_star_hero": {
            "birth_stars": 1,
            "current_stars": 1,
            "stats": {"power": 3, "agility": 2, "resilience": 1},
            "terminal_event_id": "",
            "progression_seed": "alice_progression_seed",
            "strong_stat_id": "power",
            "weak_stat_id": "resilience",
            "potential_grade": 1,
            "equipment": [
                {
                    "item_id": "live_blade",
                    "name": "Live Blade",
                    "slot": "hand",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": True,
                },
                {
                    "item_id": "secret_armor",
                    "name": "Secret Armor",
                    "slot": "armor",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": False,
                },
            ],
        },
    }

    assert image_loadout_for_character(ckpt, ckpt.characters[0]) == (
        "Live Blade worn or carried in the hand slot"
    )
    event = router_output(
        event_id="evt_live_equipment",
        observer_ids=["alice"],
        facts=[ObservableFact.all("Alice steps into the rain.")],
    )
    projection = build_projection_groups(
        checkpoint=ckpt,
        event=event,
        event_sequence=0,
        transaction_id="tx_live_equipment",
        source_turn_index=1,
        actor_id="alice",
    )[0]
    assert projection.characters[0].default_loadout == (
        "Live Blade worn or carried in the hand slot"
    )
    client = FakeDirectorClient(ImageDirectorOutput(requests=[]))
    await ImageDirector(
        client,
        PromptManager("app/prompts"),
    ).decide(projection)
    prompt_text = "\n".join(
        str(message.get("content", "")) for message in client.messages
    )
    assert "Live Blade worn or carried in the hand slot" in prompt_text
    assert "Secret Armor" not in prompt_text
    assert "stale authored loadout" not in prompt_text
    rendered_snapshot = projection_checkpoint_snapshot(ckpt)
    assert rendered_snapshot.session.config.settings.ruleset_id == (
        ONE_STAR_RULESET_ID
    )
    assert rendered_snapshot.characters[0].visuals.default_loadout == (
        "Live Blade worn or carried in the hand slot"
    )
    assert rendered_snapshot.characters[0].mechanics == {}
    assert image_loadout_for_character(
        rendered_snapshot,
        rendered_snapshot.characters[0],
    ) == "Live Blade worn or carried in the hand slot"
    snapshot_text = rendered_snapshot.model_dump_json()
    assert "Secret Armor" not in snapshot_text


def test_non_one_star_image_projection_keeps_authored_loadout():
    ckpt = _checkpoint()
    ckpt.characters[0].mechanics = {
        "one_star_hero": {
            "birth_stars": 1,
            "current_stars": 1,
            "stats": {"power": 3, "agility": 2, "resilience": 1},
            "terminal_event_id": "",
            "progression_seed": "alice_progression_seed",
            "strong_stat_id": "power",
            "weak_stat_id": "resilience",
            "potential_grade": 1,
            "equipment": [{
                "item_id": "live_blade",
                "name": "Live Blade",
                "slot": "hand",
                "quantity": 1,
                "durability_current": 0,
                "durability_max": 0,
                "tags": [],
                "visible": True,
            }],
        },
    }
    assert image_loadout_for_character(ckpt, ckpt.characters[0]) == (
        "canvas satchel and rain-spotted shoes"
    )


def test_one_star_manual_identity_reroll_uses_current_visible_equipment():
    ckpt = _checkpoint()
    ckpt.session.config.settings.ruleset_id = ONE_STAR_RULESET_ID
    ckpt.characters[0].visuals.default_loadout = "stale retired loadout"
    ckpt.characters[0].mechanics = {
        "one_star_hero": {
            "birth_stars": 1,
            "current_stars": 1,
            "stats": {"power": 3, "agility": 2, "resilience": 1},
            "terminal_event_id": "",
            "progression_seed": "alice_progression_seed",
            "strong_stat_id": "power",
            "weak_stat_id": "resilience",
            "potential_grade": 1,
            "equipment": [
                {
                    "item_id": "live_blade",
                    "name": "Live Blade",
                    "slot": "hand",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": True,
                },
                {
                    "item_id": "secret_armor",
                    "name": "Secret Armor",
                    "slot": "armor",
                    "quantity": 1,
                    "durability_current": 0,
                    "durability_max": 0,
                    "tags": [],
                    "visible": False,
                },
            ],
        },
    }
    projection, direction, _style = _authored_identity_reroll_input(
        checkpoint=ckpt,
        character_id="alice",
        transaction_id="imgtx_manual_equipment",
        max_scene_prompt_chars=2_000,
    )
    assert direction.scene_prompt.endswith(
        "Live Blade worn or carried in the hand slot"
    )
    assert "Secret Armor" not in direction.scene_prompt
    assert "stale retired loadout" not in direction.scene_prompt
    assert projection.characters[0].default_loadout == (
        "Live Blade worn or carried in the hand slot"
    )


def test_projection_snapshot_omits_unclaimed_player_authored_slot():
    source = _checkpoint()
    source.characters.append(CharacterRecord(
        character_id="blank_arrival",
        name="the Newcomer",
        is_playable=True,
        player_slot_kind=PlayerSlotKind.player_authored,
        public_sheet=PublicSheet(appearance="UNCLAIMED APPEARANCE"),
    ))

    snapshot = projection_checkpoint_snapshot(source)
    serialized = snapshot.model_dump_json()

    assert "blank_arrival" not in serialized
    assert "UNCLAIMED APPEARANCE" not in serialized


def test_projection_does_not_disclose_engine_known_actor_to_other_viewers():
    ckpt = _checkpoint()
    event = router_output(
        event_id="evt_anonymous_actor",
        observer_ids=["alice", "bob"],
        facts=[ObservableFact.all("Footsteps sound behind the closed door.")],
    )

    projections = build_projection_groups(
        checkpoint=ckpt,
        event=event,
        event_sequence=2,
        transaction_id="tx_anonymous_actor",
        source_turn_index=2,
        actor_id="bob",
    )

    by_viewer = {
        viewer_id: projection
        for projection in projections
        for viewer_id in projection.viewer_character_ids
    }
    assert all(
        character.character_id != "bob"
        for character in by_viewer["alice"].characters
    )
    assert any(
        character.character_id == "bob"
        for character in by_viewer["bob"].characters
    )


@pytest.mark.asyncio
async def test_director_receives_only_text_projection_and_can_return_zero():
    ckpt = _checkpoint()
    event = router_output(
        event_id="evt_director",
        observer_ids=["alice"],
        facts=[ObservableFact.all("Alice waits beneath the station awning.")],
    )
    projection = build_projection_groups(
        checkpoint=ckpt,
        event=event,
        event_sequence=0,
        transaction_id="tx_director",
        source_turn_index=1,
    )[0]
    client = FakeDirectorClient(ImageDirectorOutput(requests=[]))
    director = ImageDirector(
        client,
        PromptManager("app/prompts"),
    )

    output = await director.decide(projection)

    assert output.requests == []
    rendered = "\n".join(
        str(message.get("content", "")) for message in client.messages
    )
    assert "station awning" in rendered
    assert "yellow raincoat" in rendered
    assert "role=student photographer" in rendered
    assert "player_controlled=yes" in rendered
    assert "recurring_actor=yes" in rendered
    assert "canonical events so far: 1" in rendered
    assert "active roster count: 2" in rendered
    assert "PRIVATE WORLD SECRET" not in rendered
    assert "PRIVATE ALICE SECRET" not in rendered
    assert not any(isinstance(message.get("content"), bytes) for message in client.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("presentation_mode", ["prose", "visual_novel"])
async def test_director_redacts_private_paths_and_metadata_before_provider_input(
    presentation_mode: str,
):
    base = _projection(presentation_mode=presentation_mode)
    projection = replace(
        base,
        story_premise=(
            "SAFE PREMISE SENTINEL. actor.hidden "
            "app/storage/stories/private/outline.txt /secret.env "
            "https://example.com/public/story-guide.png"
        ),
        visible_facts=((
            "SAFE EVENT SENTINEL. Alice enters from "
            r"C:\Users\dan\ayoa\private\plate.png /secret.pem",
            0,
            3,
        ),),
        characters=(replace(
            base.characters[0],
            appearance=(
                "SAFE APPEARANCE SENTINEL. "
                "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            ),
            default_loadout=(
                "SAFE LOADOUT SENTINEL. tests/fixtures/private/alice.png "
                "/secret.key"
            ),
        ),),
        reference_options=(SelectableVisualReference(
            reference_id="authored.alice.v1",
            scope="character",
            scope_id="alice",
            selection_hint=(
                "SAFE REFERENCE SENTINEL. "
                r"\\authoring-host\private-share\alice.png /secret.p12"
            ),
        ),),
    )
    output = ImageDirectorOutput(
        stage_action=("reuse" if presentation_mode == "visual_novel" else "independent"),
        requests=[],
    )
    client = FakeDirectorClient(output)

    await ImageDirector(
        client,
        PromptManager("app/prompts"),
    ).decide(
        projection,
        stage_context=(
            "SAFE STAGE SENTINEL. scripts/private/stage-builder.py "
            "/secret.kdbx\x1b[31m",
        ),
    )

    rendered = "\n".join(
        str(message.get("content", "")) for message in client.messages
    )
    for safe_text in (
        "SAFE PREMISE SENTINEL",
        "SAFE EVENT SENTINEL",
        "SAFE APPEARANCE SENTINEL",
        "SAFE LOADOUT SENTINEL",
        "SAFE REFERENCE SENTINEL",
        "SAFE STAGE SENTINEL",
    ):
        assert safe_text in rendered
    assert "https://example.com/public/story-guide.png" in rendered
    for private_text in (
        "actor.hidden",
        "app/storage/stories",
        r"C:\Users\dan\ayoa",
        "sha256:",
        "0123456789abcdef",
        "tests/fixtures",
        r"\\authoring-host\private-share",
        "scripts/private",
        "/secret.env",
        "/secret.pem",
        "/secret.key",
        "/secret.p12",
        "/secret.kdbx",
        "\x1b",
    ):
        assert private_text not in rendered
    assert REDACTED_IMPORT_SENTINEL in rendered


def test_projection_includes_creator_player_without_binding():
    ckpt = _checkpoint()
    ckpt.session.character_bindings = {"bob": "22"}
    ckpt.session.player_character_id = "alice"
    event = router_output(
        event_id="evt_creator",
        observer_ids=["alice"],
        facts=[ObservableFact.all("Alice raises her camera.")],
    )

    projections = build_projection_groups(
        checkpoint=ckpt,
        event=event,
        event_sequence=0,
        transaction_id="tx_creator",
        source_turn_index=1,
        actor_id="alice",
    )

    assert len(projections) == 1
    assert projections[0].viewer_character_ids == ("alice",)
    assert projections[0].characters[0].is_playable is True
    assert projections[0].characters[0].recurring_actor is True


@pytest.mark.asyncio
async def test_director_retries_one_semantically_invalid_output():
    projection = _projection(viewers=("alice",))
    client = FakeDirectorClient([
        ImageDirectorOutput(requests=[
            ImageDirection(
                kind="portrait",
                title="Unavailable Portrait",
                subject_character_ids=["unknown"],
                scene_prompt="An unavailable character poses.",
            )
        ]),
        ImageDirectorOutput(requests=[]),
    ])
    director = ImageDirector(
        client,
        PromptManager("app/prompts"),
    )

    output = await director.decide(projection)

    assert output.requests == []
    assert client.calls == 2
    assert "violated this visual contract" in str(
        client.messages[-1]["content"]
    )


@pytest.mark.asyncio
async def test_director_retries_scene_prompt_that_requests_visible_ui_text():
    projection = _projection(viewers=("alice",))
    client = FakeDirectorClient([
        ImageDirectorOutput(requests=[
            ImageDirection(
                kind="detail",
                title="System Window",
                subject_character_ids=[],
                scene_prompt=(
                    "A System window showing a readable text line and stars."
                ),
            )
        ]),
        ImageDirectorOutput(requests=[]),
    ])
    director = ImageDirector(
        client,
        PromptManager("app/prompts"),
    )

    output = await director.decide(projection)

    assert output.requests == []
    assert client.calls == 2


@pytest.mark.asyncio
async def test_director_retries_multi_panel_or_promotional_composition():
    projection = _projection(viewers=("alice",))
    client = FakeDirectorClient([
        ImageDirectorOutput(requests=[
            ImageDirection(
                kind="portrait",
                title="Alice Collage",
                subject_character_ids=["alice"],
                scene_prompt=(
                    "A split panel character-sheet collage of Alice."
                ),
            )
        ]),
        ImageDirectorOutput(requests=[]),
    ])
    director = ImageDirector(
        client,
        PromptManager("app/prompts"),
    )

    output = await director.decide(projection)

    assert output.requests == []
    assert client.calls == 2


def test_director_rejects_omitted_or_unknown_subjects():
    projection = _projection()
    omitted = PublicCharacterVisual(
        character_id="master",
        name="Master",
        appearance="",
        default_loadout="unseen",
        depiction_policy="omit",
        is_new_character=False,
        has_identity_reference=False,
    )
    projection = VisibleEventProjection(
        **{**projection.__dict__, "characters": (*projection.characters, omitted)}
    )
    director = ImageDirector(
        FakeDirectorClient(ImageDirectorOutput(requests=[])),
        PromptManager("app/prompts"),
    )
    with pytest.raises(ValueError, match="non-depictable"):
        director.validate_output(
            projection,
            ImageDirectorOutput(requests=[
                ImageDirection(
                    kind="portrait",
                    title="Unseen Ruler",
                    subject_character_ids=["master"],
                    scene_prompt="An unseen ruler.",
                )
            ]),
        )
    with pytest.raises(ValueError, match="anonymous or omitted"):
        director.validate_output(
            projection,
            ImageDirectorOutput(requests=[
                ImageDirection(
                    kind="detail",
                    title="Hidden Interface",
                    subject_character_ids=[],
                    scene_prompt="The Master watches through the interface.",
                )
            ]),
        )


def test_director_schema_supports_zero_or_multiple_requests():
    assert ImageDirectorOutput(requests=[]).requests == []
    output = ImageDirectorOutput(requests=[
        ImageDirection(
            kind="portrait",
            title="Alice Portrait",
            subject_character_ids=["alice"],
            scene_prompt="Alice beneath the awning.",
        ),
        ImageDirection(
            kind="detail",
            title="Brass Key",
            subject_character_ids=[],
            scene_prompt="Rain collecting on a brass key.",
        ),
    ])
    assert [request.kind for request in output.requests] == [
        "portrait",
        "detail",
    ]


def test_director_validates_generation_mode_and_reference_selection():
    projection = _projection(viewers=("alice",))
    projection = VisibleEventProjection(
        **{
            **projection.__dict__,
            "reference_options": (
                SelectableVisualReference(
                    reference_id="authored.alice.face",
                    scope="character",
                    scope_id="alice",
                    selection_hint="Close face identity reference.",
                ),
            ),
        }
    )
    director = ImageDirector(
        FakeDirectorClient(ImageDirectorOutput(requests=[])),
        PromptManager("app/prompts"),
        generation_modes=("compose", "edit"),
    )
    valid = ImageDirectorOutput(
        requests=[
            ImageDirection(
                kind="portrait",
                title="Alice Close",
                subject_character_ids=["alice"],
                generation_mode="edit",
                reference_ids=["authored.alice.face"],
                scene_prompt="Alice turns toward the rain.",
            )
        ]
    )
    director.validate_output(projection, valid)

    valid.requests[0].reference_ids = []
    with pytest.raises(ValueError, match="requires a selected reference"):
        director.validate_output(projection, valid)


def test_visual_novel_replace_requires_anchored_listed_named_subjects():
    projection = replace(
        _projection(presentation_mode="visual_novel"),
        characters=tuple(
            replace(character, has_identity_reference=True)
            for character in _projection().characters
        ),
    )
    director = ImageDirector(
        FakeDirectorClient(ImageDirectorOutput(requests=[])),
        PromptManager("app/prompts"),
    )

    with pytest.raises(ValueError, match="not listed subjects.*bob"):
        director.validate_output(
            projection,
            ImageDirectorOutput(
                stage_action="replace",
                requests=[
                    ImageDirection(
                        kind="action",
                        title="Alice In Rain",
                        subject_character_ids=["alice"],
                        scene_prompt="Alice and Bob face each other in the rain.",
                    )
                ],
            ),
        )

    unanchored = replace(
        projection,
        characters=tuple(
            replace(
                character,
                has_identity_reference=(character.character_id == "alice"),
            )
            for character in projection.characters
        ),
    )
    with pytest.raises(ValueError, match="active identity references.*bob"):
        director.validate_output(
            unanchored,
            ImageDirectorOutput(
                stage_action="replace",
                requests=[
                    ImageDirection(
                        kind="action",
                        title="Bob In Rain",
                        subject_character_ids=["bob"],
                        scene_prompt="Bob pauses alone in the rain.",
                    )
                ],
            ),
        )


def test_visual_novel_reference_budget_includes_unselected_subject_identities():
    options = (
        SelectableVisualReference(
            reference_id="authored.station.front",
            scope="location",
            scope_id="station",
            selection_hint="Front courtyard framing.",
        ),
        SelectableVisualReference(
            reference_id="authored.station.light",
            scope="location",
            scope_id="station",
            selection_hint="Courtyard lighting.",
        ),
    )
    projection = replace(
        _projection(presentation_mode="visual_novel"),
        characters=tuple(
            replace(character, has_identity_reference=True)
            for character in _projection().characters
        ),
        reference_options=options,
    )
    director = ImageDirector(
        FakeDirectorClient(ImageDirectorOutput(requests=[])),
        PromptManager("app/prompts"),
        max_references=4,
        generation_modes=("compose", "edit"),
    )

    with pytest.raises(ValueError, match="3-reference limit"):
        director.validate_output(
            projection,
            ImageDirectorOutput(
                stage_action="replace",
                requests=[
                    ImageDirection(
                        kind="group_portrait",
                        title="Rainy Meeting",
                        subject_character_ids=["alice", "bob"],
                        generation_mode="edit",
                        reference_ids=[
                            "authored.station.front",
                            "authored.station.light",
                        ],
                        scene_prompt="Alice and Bob meet in the rainy courtyard.",
                    )
                ],
            ),
        )


@pytest.mark.asyncio
async def test_visual_novel_director_corrects_invalid_named_subject_once():
    projection = replace(
        _projection(presentation_mode="visual_novel"),
        characters=tuple(
            replace(character, has_identity_reference=True)
            for character in _projection().characters
        ),
    )
    invalid = ImageDirectorOutput(
        stage_action="replace",
        requests=[
            ImageDirection(
                kind="action",
                title="Alice In Rain",
                subject_character_ids=["alice"],
                scene_prompt="Alice and Bob stand in the rain.",
            )
        ],
    )
    client = FakeDirectorClient(
        [invalid, ImageDirectorOutput(stage_action="clear", requests=[])]
    )

    output = await ImageDirector(
        client,
        PromptManager("app/prompts"),
    ).decide(projection)

    assert output.stage_action == "clear"
    assert client.calls == 2

    failing_client = FakeDirectorClient([invalid, invalid])
    with pytest.raises(ValueError, match="not listed subjects"):
        await ImageDirector(
            failing_client,
            PromptManager("app/prompts"),
        ).decide(projection)
    assert failing_client.calls == 2


def test_director_requires_first_portraits_for_new_named_characters():
    projection = _projection()
    projection = VisibleEventProjection(
        **{
            **projection.__dict__,
            "characters": (
                PublicCharacterVisual(
                    character_id="davan",
                    name="Davan",
                    appearance="broad-shouldered late-twenties trainee",
                    default_loadout="rough gray tunic and short iron sword",
                    depiction_policy="normal",
                    is_new_character=True,
                    has_identity_reference=False,
                ),
                PublicCharacterVisual(
                    character_id="tuck",
                    name="Tuck",
                    appearance="wiry late-twenties archer",
                    default_loadout="rough gray tunic and unfinished bow",
                    depiction_policy="normal",
                    is_new_character=True,
                    has_identity_reference=False,
                ),
            ),
        }
    )
    director = ImageDirector(
        FakeDirectorClient(ImageDirectorOutput(requests=[])),
        PromptManager("app/prompts"),
    )

    with pytest.raises(ValueError, match="individual first portraits"):
        director.validate_output(
            projection,
            ImageDirectorOutput(requests=[
                ImageDirection(
                    kind="group_portrait",
                    title="New Summons",
                    subject_character_ids=["davan", "tuck"],
                    scene_prompt="Davan and Tuck stand in the summoning hall.",
                )
            ]),
        )

    director.validate_output(
        projection,
        ImageDirectorOutput(requests=[
            ImageDirection(
                kind="portrait",
                title="Davan Portrait",
                subject_character_ids=["davan"],
                scene_prompt="Davan stands in rough gray starter gear.",
            ),
            ImageDirection(
                kind="portrait",
                title="Tuck Portrait",
                subject_character_ids=["tuck"],
                scene_prompt="Tuck stands with an unfinished bow.",
            ),
            ImageDirection(
                kind="group_portrait",
                title="New Summons",
                subject_character_ids=["davan", "tuck"],
                scene_prompt="Davan and Tuck stand in the summoning hall.",
            ),
        ]),
    )


def test_diffusion_prompt_uses_engine_style_and_public_subject_metadata_only():
    prompt = build_diffusion_prompt(
        projection=_projection(),
        direction=ImageDirection(
            kind="portrait",
            title="Alice Portrait",
            subject_character_ids=["alice"],
            scene_prompt="Alice pauses beneath the station awning.",
        ),
        visual_style="soft cinematic illustration",
        max_scene_prompt_chars=2_000,
        max_style_chars=800,
        style_trigger="ayoapmu2",
    )
    assert prompt.startswith("Alice pauses beneath the station awning.")
    assert "soft cinematic" in prompt
    assert "yellow raincoat" in prompt
    assert "silver hair" not in prompt
    assert "PRIVATE" not in prompt
    assert prompt.index("Alice pauses") < prompt.index("Identity:")
    assert prompt.index("Identity:") < prompt.index("Style and composition:")
    assert "head and boots visible" in prompt
    assert "no readable text" in prompt
    assert "abstract nonlinguistic geometry" in prompt


def test_diffusion_prompt_labels_reference_images_by_subject():
    prompt = build_diffusion_prompt(
        projection=_projection(),
        direction=ImageDirection(
            kind="group_portrait",
            title="Alice And Bob",
            subject_character_ids=["alice", "bob"],
            scene_prompt="Alice and Bob stand in the station hall.",
        ),
        visual_style="soft cinematic illustration",
        max_scene_prompt_chars=2_000,
        max_style_chars=800,
        reference_inputs=[
            FrozenReferenceInput(
                reference_id="alice_ref",
                sha256="a" * 64,
                mime_type="image/webp",
                width=768,
                height=1024,
                byte_count=10,
                relative_path="artifacts/a.webp",
                allowed_root="artifacts",
            ),
            FrozenReferenceInput(
                reference_id="bob_ref",
                sha256="b" * 64,
                mime_type="image/webp",
                width=768,
                height=1024,
                byte_count=10,
                relative_path="artifacts/b.webp",
                allowed_root="artifacts",
            ),
        ],
        identity_reference_owners={
            "alice_ref": "alice",
            "bob_ref": "bob",
        },
    )

    assert "Reference image 1 is Alice (alice)." in prompt
    assert "Reference image 2 is Bob (bob)." in prompt
    assert "reference identity, proportions, face, hair" in prompt
    assert "override general style text" in prompt
    assert "Do not transfer visual traits" in prompt
    assert "no readable text" in prompt
    for unsupported_negative in (
        "never a collage",
        "Render no words",
        "Do not change a subject's stated age",
    ):
        assert unsupported_negative not in prompt


def test_generation_config_defaults_to_base_model_sampling(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("AYOA_IMAGE_STEPS", raising=False)
    monkeypatch.delenv("AYOA_IMAGE_GUIDANCE", raising=False)
    monkeypatch.delenv("AYOA_IMAGE_WORKER_BACKEND", raising=False)

    config = ImageGenerationConfig.from_environment(
        runtime_root=tmp_path / "runtime",
    )

    assert config.steps == 50
    assert config.guidance == 4.0


def test_generation_config_uses_twenty_steps_for_remote_dev(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("AYOA_IMAGE_STEPS", raising=False)
    monkeypatch.setenv("AYOA_IMAGE_WORKER_BACKEND", "remote")

    config = ImageGenerationConfig.from_environment(
        runtime_root=tmp_path / "runtime",
    )

    assert config.steps == 20
    assert config.guidance == 4.0


def test_one_star_story_has_reviewed_style_and_omits_unseen_master():
    path = Path(
        "app/storage/stories/one_star_ascension_s1/ckpt_0000.json"
    )
    ckpt = CheckpointFile.model_validate_json(path.read_text())
    master = next(
        character
        for character in ckpt.characters
        if character.character_id == "the_master"
    )
    assert ckpt.world_state.setting.visual_style
    assert "artwork filling every edge" in (
        ckpt.world_state.setting.visual_style
    )
    assert "locked identity reference establishes mascot proportions" in (
        ckpt.world_state.setting.visual_style
    )
    assert len(ckpt.world_state.setting.visual_style) <= 800
    assert master.visuals.depiction_policy == "omit"


def test_v1_actor_cadence_store_is_retired_in_direct_v10_migration(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE image_store_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        db.execute(
            "INSERT INTO image_store_meta VALUES ('schema_version', '1')"
        )
        db.execute(
            "CREATE TABLE image_eligible_beats (session_id TEXT)"
        )

    store = ImageJobStore(db_path)

    with store._connect() as db:
        version = db.execute(
            """
            SELECT value FROM image_store_meta WHERE key = 'schema_version'
            """
        ).fetchone()["value"]
        old_table = db.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'image_eligible_beats'
            """
        ).fetchone()
    assert version == "10"
    assert old_table is None


def test_v7_prose_gate_store_is_retired_before_foreign_key_parent(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute(
            "CREATE TABLE image_store_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        db.execute(
            "INSERT INTO image_store_meta VALUES ('schema_version', '7')"
        )
        db.execute(
            "CREATE TABLE image_transactions (transaction_id TEXT PRIMARY KEY)"
        )
        db.execute(
            """
            CREATE TABLE image_prose_gates (
                transaction_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                pov_character_id TEXT NOT NULL,
                FOREIGN KEY(transaction_id)
                    REFERENCES image_transactions(transaction_id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE image_prose_receipts (
                session_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                pov_character_id TEXT NOT NULL
            )
            """
        )
        db.execute("INSERT INTO image_transactions VALUES ('tx_old')")
        db.execute(
            "INSERT INTO image_prose_gates VALUES ('tx_old', 'evt_old', 'alice')"
        )
        db.execute(
            "INSERT INTO image_prose_receipts VALUES ('old', 'evt_old', 'alice')"
        )

    store = ImageJobStore(db_path)

    with store._connect() as db:
        version = db.execute(
            "SELECT value FROM image_store_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        retired = {
            row["name"]
            for row in db.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name LIKE 'image_prose_%'
                """
            ).fetchall()
        }
        transaction_count = db.execute(
            "SELECT COUNT(*) AS count FROM image_transactions"
        ).fetchone()["count"]
    assert version == "10"
    assert retired == set()
    assert transaction_count == 0


def test_same_version_intermediate_store_is_retired_by_layout(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE image_store_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        db.execute(
            "INSERT INTO image_store_meta VALUES ('schema_version', '5')"
        )
        db.execute(
            """
            CREATE TABLE image_jobs (
                job_id TEXT PRIMARY KEY,
                next_delivery_at REAL NOT NULL DEFAULT 0
            )
            """
        )

    store = ImageJobStore(db_path)

    with store._connect() as db:
        job_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(image_jobs)").fetchall()
        }
        delivery_columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(image_deliveries)"
            ).fetchall()
        }
    assert "request_json" in job_columns
    assert "next_delivery_at" not in job_columns
    assert "next_attempt_at" in delivery_columns


@pytest.mark.asyncio
async def test_identity_delivery_waits_for_transaction_commit(
    tmp_path,
):
    worker = FakeImageWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    delivered: list[str] = []

    async def deliver(job, delivery, media, instructions):
        delivered.append(delivery.pov_character_id)
        return True

    coordinator.register_delivery_handler(ImageDeliveryKind.cli, deliver)
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        job = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="action",
                title="Rain Run",
                subject_character_ids=["alice"],
                scene_prompt="Alice runs into the rain.",
            ),
            request_ordinal=0,
            visual_style="soft cinematic illustration",
            delivery_targets=[_target("alice"), _target("bob")],
        )
        assert job is not None
        assert 0 <= job.request.seed <= (1 << 63) - 1
        completed = await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert completed is not None
        assert completed.status == ImageGenerationStatus.succeeded
        assert worker.calls == 1
        assert delivered == []

        await coordinator.commit_transaction(
            "tx_1",
            target_checkpoint_sha256="b" * 64,
        )
        await _wait_until(lambda: sorted(delivered) == ["alice", "bob"])
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_render_wait_tracks_all_requested_event_images(tmp_path):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    projection = _projection(viewers=("alice",))
    direction_one = ImageDirection(
        kind="action",
        title="Rain Run",
        subject_character_ids=["alice"],
        scene_prompt="Alice runs into the rain.",
    )
    direction_two = ImageDirection(
        kind="detail",
        title="Brass Key",
        subject_character_ids=[],
        scene_prompt="Rain beads on a brass key.",
    )
    try:
        queued = coordinator.store.enqueue_director_run(projection)
        claimed = coordinator.store.claim_next_director_run()
        assert claimed is not None
        assert claimed.run_id == queued.run_id
        _complete_director(coordinator,
            claimed.run_id,
            ImageDirectorOutput(requests=[direction_one, direction_two]),
        )

        first = await coordinator.enqueue_direction(
            projection=projection,
            direction=direction_one,
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
            director_run_id=claimed.run_id,
            director_attempt=claimed.attempts,
        )
        second = await coordinator.enqueue_direction(
            projection=projection,
            direction=direction_two,
            request_ordinal=1,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
            director_run_id=claimed.run_id,
            director_attempt=claimed.attempts,
        )
        assert first is not None
        assert second is not None
        assert first.request.title == "Rain Run"
        assert second.request.title == "Brass Key"
        _finalize_director(coordinator,
            claimed.run_id,
            projection=projection,
            admitted_job_ids=[first.job_id, second.job_id],
        )

        await asyncio.wait_for(worker.started.wait(), timeout=1)
        assert await coordinator.wait_for_render_images(
            session_id="image_test",
            rendered_event_ids_by_pov={"alice": ["evt_1"]},
            timeout=0.1,
            discovery_grace_seconds=0,
        ) is False

        worker.release.set()
        await coordinator.wait_for_terminal(first.job_id, timeout=2)
        await coordinator.wait_for_terminal(second.job_id, timeout=2)
        assert await coordinator.wait_for_render_images(
            session_id="image_test",
            rendered_event_ids_by_pov={"alice": ["evt_1"]},
            timeout=0.1,
            discovery_grace_seconds=0,
        ) is True
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_render_wait_finishes_when_no_director_requests_are_admitted(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_1")
    projection = _projection(viewers=("alice",))
    queued = coordinator.store.enqueue_director_run(projection)
    claimed = coordinator.store.claim_next_director_run()
    assert claimed is not None
    direction = ImageDirection(
        kind="action",
        title="Rejected Rain Run",
        subject_character_ids=["alice"],
        scene_prompt="Alice runs into the rain.",
    )
    _complete_director(coordinator,
        queued.run_id,
        ImageDirectorOutput(requests=[direction]),
    )
    with coordinator.store._connect() as db:
        materialized_at = db.execute(
            "SELECT materialized_at FROM image_director_runs WHERE run_id = ?",
            (queued.run_id,),
        ).fetchone()["materialized_at"]
    assert materialized_at is None

    assert await coordinator.wait_for_render_images(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
        timeout=0.05,
        discovery_grace_seconds=0,
    ) is False

    _finalize_director(coordinator,
        queued.run_id,
        projection=projection,
        admitted_job_ids=[],
    )
    with coordinator.store._connect() as db:
        materialized_at = db.execute(
            "SELECT materialized_at FROM image_director_runs WHERE run_id = ?",
            (queued.run_id,),
        ).fetchone()["materialized_at"]
    assert materialized_at is not None
    assert await coordinator.wait_for_render_images(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
        timeout=0.05,
        discovery_grace_seconds=0,
    ) is True


def test_director_claim_order_is_durable_across_store_instances(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_order")
    earlier = coordinator.store.enqueue_director_run(_projection(
        transaction_id="tx_order",
        event_id="evt_order_a",
        event_sequence=10,
        source_turn_index=3,
        presentation_mode="visual_novel",
    ))
    sibling = coordinator.store.enqueue_director_run(_projection(
        transaction_id="tx_order",
        event_id="evt_order_b",
        event_sequence=10,
        source_turn_index=3,
        presentation_mode="visual_novel",
    ))
    later = coordinator.store.enqueue_director_run(_projection(
        transaction_id="tx_order",
        event_id="evt_order_later",
        event_sequence=11,
        source_turn_index=3,
        presentation_mode="visual_novel",
    ))
    second_process_store = ImageJobStore(coordinator.store.db_path)

    first_claim = coordinator.store.claim_next_director_run()
    second_claim = second_process_store.claim_next_director_run()
    assert first_claim is not None
    assert second_claim is not None
    assert {first_claim.run_id, second_claim.run_id} == {
        earlier.run_id,
        sibling.run_id,
    }
    assert second_process_store.claim_next_director_run() is None

    for claim in (first_claim, second_claim):
        completed = coordinator.store.complete_director_run(
            claim.run_id,
            ImageDirectorOutput(stage_action="clear", requests=[]),
            attempt=claim.attempts,
        )
        assert completed is not None
        coordinator.store.finalize_director_materialization(
            claim.run_id,
            attempt=claim.attempts,
            projection=claim.projection,
            admitted_job_ids=[],
        )

    later_claim = second_process_store.claim_next_director_run()
    assert later_claim is not None
    assert later_claim.run_id == later.run_id


@pytest.mark.asyncio
async def test_sidecar_start_recovers_expired_unmaterialized_director_run(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_1")
    projection = _projection(
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    direction = ImageDirection(
        kind="establishing",
        title="Interrupted Courtyard",
        subject_character_ids=[],
        scene_prompt="A rain-dark courtyard beneath stone arcades.",
    )
    queued = coordinator.store.enqueue_director_run(projection)
    claimed = coordinator.store.claim_next_director_run()
    assert claimed is not None and claimed.run_id == queued.run_id
    materializing = _complete_director(coordinator,
        queued.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=[direction],
        ),
    )
    assert materializing is not None
    assert materializing.status == "materializing"
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
    ) == (True, False)

    with coordinator.store._connect() as db:
        db.execute(
            "UPDATE image_director_runs SET updated_at = 0 WHERE run_id = ?",
            (queued.run_id,),
        )
    assert _heartbeat_director(coordinator, queued.run_id)
    assert coordinator.store.recover_expired_director_runs(
        lease_seconds=1
    ) == 0
    with coordinator.store._connect() as db:
        db.execute(
            "UPDATE image_director_runs SET updated_at = 0 WHERE run_id = ?",
            (queued.run_id,),
        )
    assert coordinator.store.commit_transaction(
        "tx_1",
        target_checkpoint_sha256="b" * 64,
    )

    class ClearDirector:
        def __init__(self) -> None:
            self.called = asyncio.Event()

        async def decide(self, _projection, *, stage_context=()):
            del stage_context
            self.called.set()
            return ImageDirectorOutput(stage_action="clear", requests=[])

    director = ClearDirector()
    sidecar = EventImageSidecar(
        director=director,  # type: ignore[arg-type]
        generation=coordinator,
        spawn_authoring=SimpleNamespace(),  # type: ignore[arg-type]
    )
    await sidecar.start()
    try:
        await asyncio.wait_for(director.called.wait(), timeout=1)
        await _wait_until(lambda: coordinator.store.rendered_event_image_status(
            session_id="image_test",
            rendered_event_ids_by_pov={"alice": ["evt_1"]},
        ) == (True, True))
        with coordinator.store._connect() as db:
            recovered = db.execute(
                """
                SELECT status, output_json, materialized_at, attempts
                FROM image_director_runs WHERE run_id = ?
                """,
                (queued.run_id,),
            ).fetchone()
        assert recovered is not None
        assert recovered["status"] == "succeeded"
        assert recovered["materialized_at"] is not None
        assert recovered["attempts"] == 2
        assert ImageDirectorOutput.model_validate_json(
            recovered["output_json"]
        ).stage_action == "clear"
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_stale_director_attempt_cannot_mutate_reclaimed_attempt(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_fenced")
    projection = _projection(
        transaction_id="tx_fenced",
        event_id="evt_fenced",
        event_sequence=2,
        presentation_mode="visual_novel",
    )
    first_direction = ImageDirection(
        kind="establishing",
        title="First Attempt",
        subject_character_ids=[],
        scene_prompt="A rain-dark courtyard before the interruption.",
    )
    second_direction = ImageDirection(
        kind="establishing",
        title="Reclaimed Attempt",
        subject_character_ids=[],
        scene_prompt="A bright courtyard after the interruption.",
    )
    run = coordinator.store.enqueue_director_run(projection)
    first_claim = coordinator.store.claim_next_director_run()
    assert first_claim is not None and first_claim.run_id == run.run_id
    first_output = ImageDirectorOutput(
        stage_action="replace",
        requests=[first_direction],
    )
    assert coordinator.store.complete_director_run(
        run.run_id,
        first_output,
        attempt=first_claim.attempts,
    ) is not None
    first_job = await coordinator.enqueue_direction(
        projection=projection,
        direction=first_direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=run.run_id,
        director_attempt=first_claim.attempts,
    )
    assert first_job is not None

    with coordinator.store._connect() as db:
        db.execute(
            "UPDATE image_director_runs SET updated_at = 0 WHERE run_id = ?",
            (run.run_id,),
        )
    assert coordinator.store.recover_expired_director_runs(
        lease_seconds=1
    ) == 1
    recovered_first_job = coordinator.store.get(first_job.job_id)
    assert recovered_first_job is not None
    assert recovered_first_job.status == ImageGenerationStatus.cancelled

    second_claim = coordinator.store.claim_next_director_run()
    assert second_claim is not None
    assert second_claim.run_id == run.run_id
    assert second_claim.attempts == first_claim.attempts + 1
    assert coordinator.store.complete_director_run(
        run.run_id,
        first_output,
        attempt=first_claim.attempts,
    ) is None
    assert not coordinator.store.heartbeat_director_run(
        run.run_id,
        attempt=first_claim.attempts,
    )

    second_output = ImageDirectorOutput(
        stage_action="replace",
        requests=[second_direction],
    )
    assert coordinator.store.complete_director_run(
        run.run_id,
        second_output,
        attempt=second_claim.attempts,
    ) is not None
    second_job = await coordinator.enqueue_direction(
        projection=projection,
        direction=second_direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=run.run_id,
        director_attempt=second_claim.attempts,
    )
    assert second_job is not None

    assert coordinator.store.fail_director_run(
        run.run_id,
        "stale_failure",
        attempt=first_claim.attempts,
    ) is None
    with pytest.raises(RuntimeError, match="attempt is stale"):
        coordinator.store.finalize_director_materialization(
            run.run_id,
            attempt=first_claim.attempts,
            projection=projection,
            admitted_job_ids=[first_job.job_id],
        )
    current_second_job = coordinator.store.get(second_job.job_id)
    assert current_second_job is not None
    assert current_second_job.status == ImageGenerationStatus.queued
    with coordinator.store._connect() as db:
        links = db.execute(
            """
            SELECT attempt, job_id, finalized
            FROM image_director_run_jobs WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchall()
    assert [tuple(link) for link in links] == [
        (second_claim.attempts, second_job.job_id, 0)
    ]

    finalized = coordinator.store.finalize_director_materialization(
        run.run_id,
        attempt=second_claim.attempts,
        projection=projection,
        admitted_job_ids=[second_job.job_id],
    )
    assert finalized.status == "succeeded"
    assert finalized.output == second_output


@pytest.mark.asyncio
async def test_cleanup_aborts_only_last_linked_running_image_attempt(tmp_path):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_shared_attempt")
    shared = _projection(
        transaction_id="tx_shared_attempt",
        event_id="evt_shared_attempt",
        event_sequence=4,
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    sibling_projection = replace(
        shared,
        viewer_character_ids=("bob",),
        visible_facts=(("Bob sees the same courtyard privately.", 0, 3),),
    )
    direction = ImageDirection(
        kind="establishing",
        title="Shared Courtyard",
        subject_character_ids=[],
        scene_prompt="A single courtyard shared across private viewpoints.",
    )
    output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    first_run = coordinator.store.enqueue_director_run(shared)
    second_run = coordinator.store.enqueue_director_run(sibling_projection)
    first_claim = coordinator.store.claim_next_director_run()
    second_claim = coordinator.store.claim_next_director_run()
    assert first_claim is not None
    assert second_claim is not None
    assert {first_claim.run_id, second_claim.run_id} == {
        first_run.run_id,
        second_run.run_id,
    }
    claims = {
        first_claim.run_id: first_claim,
        second_claim.run_id: second_claim,
    }
    for claim in claims.values():
        assert coordinator.store.complete_director_run(
            claim.run_id,
            output,
            attempt=claim.attempts,
        ) is not None

    try:
        first_job = await coordinator.enqueue_direction(
            projection=shared,
            direction=direction,
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[],
            diffusion_prompt_override=direction.scene_prompt,
            director_run_id=first_run.run_id,
            director_attempt=claims[first_run.run_id].attempts,
        )
        second_job = await coordinator.enqueue_direction(
            projection=sibling_projection,
            direction=direction,
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[],
            diffusion_prompt_override=direction.scene_prompt,
            director_run_id=second_run.run_id,
            director_attempt=claims[second_run.run_id].attempts,
        )
        assert first_job is not None
        assert second_job is not None
        assert first_job.job_id == second_job.job_id
        await asyncio.wait_for(worker.started.wait(), timeout=1)

        failed, cancelled_attempts = (
            coordinator.store.fail_director_run_with_cleanup(
                second_run.run_id,
                "sibling_failed",
                attempt=claims[second_run.run_id].attempts,
            )
        )
        assert failed is not None
        assert cancelled_attempts == ()
        assert not await coordinator.abort_cancelled_attempts(
            cancelled_attempts
        )
        retained = coordinator.store.get(first_job.job_id)
        assert retained is not None
        assert retained.status == ImageGenerationStatus.running
        assert worker.aborted is False

        with coordinator.store._connect() as db:
            db.execute(
                """
                UPDATE image_director_runs SET updated_at = 0
                WHERE run_id = ?
                """,
                (first_run.run_id,),
            )
        recovered, cancelled_attempts = (
            coordinator.store.recover_expired_director_runs_with_cleanup(
                lease_seconds=1
            )
        )
        assert recovered == 1
        assert cancelled_attempts == ((first_job.job_id, 1),)
        assert await coordinator.abort_cancelled_attempts(
            cancelled_attempts
        )
        assert worker.aborted is True
        cancelled = coordinator.store.get(first_job.job_id)
        assert cancelled is not None
        assert cancelled.status == ImageGenerationStatus.cancelled
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_reclaimed_attempt_admits_identical_orphan_before_capacity(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path, queue_limit=1, per_session_queue_limit=1),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_orphan")
    projection = _projection(
        transaction_id="tx_orphan",
        event_id="evt_orphan",
        event_sequence=3,
        presentation_mode="visual_novel",
    )
    direction = ImageDirection(
        kind="establishing",
        title="Recovered Courtyard",
        subject_character_ids=[],
        scene_prompt="A courtyard whose queued render survived a restart.",
    )
    output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    run = coordinator.store.enqueue_director_run(projection)
    first_claim = coordinator.store.claim_next_director_run()
    assert first_claim is not None
    assert coordinator.store.complete_director_run(
        run.run_id,
        output,
        attempt=first_claim.attempts,
    ) is not None

    # Simulate the historical crash window: the exact queued request survived,
    # but the director-attempt association did not.
    orphan = await coordinator.enqueue_direction(
        projection=projection,
        direction=direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
    )
    assert orphan is not None
    assert coordinator.store.active_count() == 1
    with coordinator.store._connect() as db:
        db.execute(
            "UPDATE image_director_runs SET updated_at = 0 WHERE run_id = ?",
            (run.run_id,),
        )
    assert coordinator.store.recover_expired_director_runs(
        lease_seconds=1
    ) == 1
    recovered_orphan = coordinator.store.get(orphan.job_id)
    assert recovered_orphan is not None
    assert recovered_orphan.status == ImageGenerationStatus.queued

    second_claim = coordinator.store.claim_next_director_run()
    assert second_claim is not None
    assert coordinator.store.complete_director_run(
        run.run_id,
        output,
        attempt=second_claim.attempts,
    ) is not None
    admitted = await coordinator.enqueue_direction(
        projection=projection,
        direction=direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=run.run_id,
        director_attempt=second_claim.attempts,
    )
    assert admitted is not None
    assert admitted.job_id == orphan.job_id
    assert coordinator.store.active_count() == 1
    coordinator.store.finalize_director_materialization(
        run.run_id,
        attempt=second_claim.attempts,
        projection=projection,
        admitted_job_ids=[admitted.job_id],
    )
    with coordinator.store._connect() as db:
        association = db.execute(
            """
            SELECT attempt, job_id, finalized
            FROM image_director_run_jobs WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()
    assert association is not None
    assert tuple(association) == (second_claim.attempts, orphan.job_id, 1)
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_orphan"]},
    ) == (True, False)
    claimed_job = coordinator.store.claim_next()
    assert claimed_job is not None and claimed_job.job_id == orphan.job_id
    coordinator.store.mark_failed(claimed_job.job_id, "test_terminal")
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_orphan"]},
    ) == (True, True)


@pytest.mark.asyncio
async def test_director_recovery_runs_between_continuously_queued_work(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_cadence_1")
    _begin(coordinator, "tx_cadence_2")
    first = coordinator.store.enqueue_director_run(_projection(
        transaction_id="tx_cadence_1",
        event_id="evt_cadence_1",
        event_sequence=20,
        source_turn_index=5,
        presentation_mode="visual_novel",
    ))
    second = coordinator.store.enqueue_director_run(_projection(
        transaction_id="tx_cadence_2",
        event_id="evt_cadence_2",
        event_sequence=21,
        source_turn_index=5,
        presentation_mode="visual_novel",
    ))

    recover = coordinator.store.recover_expired_director_runs_with_cleanup
    recovery_calls = 0

    def track_recovery(
        *,
        lease_seconds: float = 300,
    ) -> tuple[int, tuple[tuple[str, int], ...]]:
        nonlocal recovery_calls
        recovery_calls += 1
        return recover(lease_seconds=lease_seconds)

    coordinator.store.recover_expired_director_runs_with_cleanup = (  # type: ignore[method-assign]
        track_recovery
    )
    observations: list[tuple[str, int]] = []
    second_decision = asyncio.Event()

    class ClearDirector:
        async def decide(self, projection, *, stage_context=()):
            del stage_context
            observations.append((projection.event_id, recovery_calls))
            if projection.event_id == second.projection.event_id:
                second_decision.set()
            return ImageDirectorOutput(stage_action="clear", requests=[])

    sidecar = EventImageSidecar(
        director=ClearDirector(),  # type: ignore[arg-type]
        generation=coordinator,
        spawn_authoring=SimpleNamespace(),  # type: ignore[arg-type]
    )

    def both_finalized() -> bool:
        with coordinator.store._connect() as db:
            count = db.execute(
                """
                SELECT COUNT(*) FROM image_director_runs
                WHERE run_id IN (?, ?) AND status = 'succeeded'
                  AND materialized_at IS NOT NULL
                """,
                (first.run_id, second.run_id),
            ).fetchone()[0]
        return int(count) == 2

    await sidecar.start()
    try:
        await asyncio.wait_for(second_decision.wait(), timeout=1)
        await _wait_until(both_finalized)
    finally:
        await sidecar.close()

    assert observations == [
        (first.projection.event_id, 1),
        (second.projection.event_id, 2),
    ]


@pytest.mark.asyncio
async def test_materialization_failure_fails_run_and_cancels_partial_jobs(
    tmp_path,
):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    projection = _projection(
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    directions = [
        ImageDirection(
            kind="establishing",
            title="Admitted Courtyard",
            subject_character_ids=[],
            scene_prompt="A rain-dark courtyard beneath stone arcades.",
        ),
        ImageDirection(
            kind="detail",
            title="Broken Admission",
            subject_character_ids=[],
            scene_prompt="A brass key at the edge of a puddle.",
        ),
    ]
    run = coordinator.store.enqueue_director_run(projection)
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        run.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=directions,
        ),
    )
    enqueue_direction = coordinator.enqueue_direction
    calls = 0

    async def fail_second_admission(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.wait_for(worker.started.wait(), timeout=1)
            raise RuntimeError("admission failed")
        return await enqueue_direction(**kwargs)

    coordinator.enqueue_direction = fail_second_admission  # type: ignore[method-assign]
    sidecar = EventImageSidecar(
        director=SimpleNamespace(),  # type: ignore[arg-type]
        generation=coordinator,
        spawn_authoring=SimpleNamespace(),  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(RuntimeError, match="admission failed"):
            await sidecar._materialize_requests(
                run.run_id,
                _director_attempt(coordinator, run.run_id),
                projection,
                ImageDirectorOutput(
                    stage_action="replace",
                    requests=directions,
                ),
            )
    finally:
        worker.release.set()
        await coordinator.close()

    jobs = coordinator.store.all_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == ImageGenerationStatus.cancelled
    assert worker.aborted is True
    with coordinator.store._connect() as db:
        failed_run = db.execute(
            """
            SELECT status, error_code, materialized_at
            FROM image_director_runs WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()
        association_count = db.execute(
            "SELECT COUNT(*) FROM image_director_run_jobs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
    assert failed_run is not None
    assert failed_run["status"] == "failed"
    assert failed_run["error_code"] == "materialization_RuntimeError"
    assert failed_run["materialized_at"] is None
    assert association_count == 0
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
    ) == (True, True)


@pytest.mark.asyncio
async def test_failed_preflight_finalizes_persisted_run_with_zero_jobs(tmp_path):
    worker = FailedPreflightWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    _begin(coordinator, "tx_1")
    projection = _projection(
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    direction = ImageDirection(
        kind="establishing",
        title="Unavailable Courtyard",
        subject_character_ids=[],
        scene_prompt="A rain-dark courtyard beneath stone arcades.",
    )
    run = coordinator.store.enqueue_director_run(projection)
    assert coordinator.store.claim_next_director_run() is not None
    output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    _complete_director(coordinator, run.run_id, output)
    await coordinator.start()
    assert worker.available is True
    assert coordinator.can_generate_render() is False
    sidecar = EventImageSidecar(
        director=SimpleNamespace(),  # type: ignore[arg-type]
        generation=coordinator,
        spawn_authoring=SimpleNamespace(),  # type: ignore[arg-type]
    )

    try:
        await sidecar._materialize_requests(
            run.run_id,
            _director_attempt(coordinator, run.run_id),
            projection,
            output,
        )
    finally:
        await coordinator.close()

    with coordinator.store._connect() as db:
        finalized = db.execute(
            """
            SELECT status, materialized_at FROM image_director_runs
            WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()
        association_count = db.execute(
            "SELECT COUNT(*) FROM image_director_run_jobs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
    assert finalized is not None
    assert finalized["status"] == "succeeded"
    assert finalized["materialized_at"] is not None
    assert association_count == 0
    assert coordinator.store.all_jobs() == []
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
    ) == (True, True)


@pytest.mark.asyncio
async def test_failed_preflight_restart_settles_finalized_job_without_owner(
    tmp_path,
):
    config = _config(tmp_path)
    original = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FakeImageWorker(),
    )
    projection, run, job = await _persist_finalized_visual_novel_job(original)
    await original.close()

    restarted = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FailedPreflightWorker(),
    )
    try:
        await restarted.start()

        settled = restarted.store.get(job.job_id)
        assert settled is not None
        assert settled.status == ImageGenerationStatus.failed
        assert settled.error_code == "worker_unavailable"
        assert restarted.store.rendered_event_image_status(
            session_id=projection.session_id,
            rendered_event_ids_by_pov={"alice": [projection.event_id]},
        ) == (True, True)
        stage, media = restarted.resolve_visual_novel_stage(
            session_id=projection.session_id,
            pov_character_id="alice",
            rendered_event_ids=[projection.event_id],
        )
        assert stage.source_run_id == run.run_id
        assert stage.artifact is None
        assert stage.fallback_reason == "replacement_failed"
        assert media is None
        assert await restarted.wait_for_render_images(
            session_id=projection.session_id,
            rendered_event_ids_by_pov={"alice": [projection.event_id]},
            timeout=0.1,
            discovery_grace_seconds=0,
        )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_failed_preflight_restart_preserves_job_with_live_owner(tmp_path):
    config = _config(tmp_path)
    original = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FakeImageWorker(),
    )
    projection, _, job = await _persist_finalized_visual_novel_job(original)
    await original.close()

    capable_owner = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FakeImageWorker(),
    )
    capable_owner._queue_owner = capable_owner._acquire_queue_owner()
    assert capable_owner._queue_owner is True
    assert capable_owner.can_generate_render() is True
    restarted = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FailedPreflightWorker(),
    )
    try:
        await restarted.start()

        preserved = restarted.store.get(job.job_id)
        assert preserved is not None
        assert preserved.status == ImageGenerationStatus.queued
        assert restarted.store.rendered_event_image_status(
            session_id=projection.session_id,
            rendered_event_ids_by_pov={"alice": [projection.event_id]},
        ) == (True, False)
        assert not await restarted.wait_for_render_images(
            session_id=projection.session_id,
            rendered_event_ids_by_pov={"alice": [projection.event_id]},
            timeout=0.05,
            discovery_grace_seconds=0,
        )
    finally:
        await restarted.close()
        await capable_owner.close()

    ownerless_restart = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FailedPreflightWorker(),
    )
    try:
        await ownerless_restart.start()
        settled = ownerless_restart.store.get(job.job_id)
        assert settled is not None
        assert settled.status == ImageGenerationStatus.failed
        assert ownerless_restart.store.rendered_event_image_status(
            session_id=projection.session_id,
            rendered_event_ids_by_pov={"alice": [projection.event_id]},
        ) == (True, True)
    finally:
        await ownerless_restart.close()


@pytest.mark.asyncio
async def test_split_private_pov_stages_resolve_only_their_linked_artifact(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_split")
    shared = _projection(
        transaction_id="tx_split",
        event_id="evt_split",
        event_sequence=1,
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    alice_projection = replace(
        shared,
        visible_facts=(("Alice sees the brass key.", 0, 3),),
    )
    bob_projection = replace(
        shared,
        viewer_character_ids=("bob",),
        visible_facts=(("Bob sees the sealed letter.", 0, 3),),
    )
    alice_direction = ImageDirection(
        kind="detail",
        title="Alice Brass Key",
        subject_character_ids=[],
        scene_prompt="A rain-bright brass key in Alice's sight.",
    )
    bob_direction = ImageDirection(
        kind="detail",
        title="Bob Sealed Letter",
        subject_character_ids=[],
        scene_prompt="A sealed letter held within Bob's sight.",
    )

    alice_run = coordinator.store.enqueue_director_run(alice_projection)
    claimed = coordinator.store.claim_next_director_run()
    assert claimed is not None and claimed.run_id == alice_run.run_id
    _complete_director(coordinator,
        alice_run.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=[alice_direction],
        ),
    )
    bob_run = coordinator.store.enqueue_director_run(bob_projection)
    claimed = coordinator.store.claim_next_director_run()
    assert claimed is not None and claimed.run_id == bob_run.run_id
    _complete_director(coordinator,
        bob_run.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=[bob_direction],
        ),
    )
    assert alice_run.run_id != bob_run.run_id

    alice_job = await coordinator.enqueue_direction(
        projection=alice_projection,
        direction=alice_direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=alice_run.run_id,
        director_attempt=_director_attempt(coordinator, alice_run.run_id),
    )
    bob_job = await coordinator.enqueue_direction(
        projection=bob_projection,
        direction=bob_direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=bob_run.run_id,
        director_attempt=_director_attempt(coordinator, bob_run.run_id),
    )
    assert alice_job is not None and bob_job is not None
    _finalize_director(coordinator,
        alice_run.run_id,
        projection=alice_projection,
        admitted_job_ids=[alice_job.job_id],
    )
    _finalize_director(coordinator,
        bob_run.run_id,
        projection=bob_projection,
        admitted_job_ids=[bob_job.job_id],
    )

    artifacts = {
        "Alice Brass Key": GeneratedImageArtifact(
            sha256="a" * 64,
            relative_path="artifacts/aa/alice-stage.webp",
            width=1024,
            height=576,
            byte_count=101,
        ),
        "Bob Sealed Letter": GeneratedImageArtifact(
            sha256="b" * 64,
            relative_path="artifacts/bb/bob-stage.webp",
            width=1024,
            height=576,
            byte_count=102,
        ),
    }
    for _ in range(2):
        job = coordinator.store.claim_next()
        assert job is not None
        coordinator.store.mark_succeeded(
            job.job_id,
            artifacts[job.request.title],
        )
    assert coordinator.store.commit_transaction(
        "tx_split",
        target_checkpoint_sha256="c" * 64,
    )

    alice_stage = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="alice",
        rendered_event_ids=["evt_split"],
    )
    bob_stage = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="bob",
        rendered_event_ids=["evt_split"],
    )
    assert alice_stage.artifact == artifacts["Alice Brass Key"]
    assert bob_stage.artifact == artifacts["Bob Sealed Letter"]

    _begin(coordinator, "tx_split_next")
    alice_next = coordinator.store.enqueue_director_run(replace(
        alice_projection,
        transaction_id="tx_split_next",
        event_id="evt_split_next",
        event_sequence=2,
    ))
    bob_next = coordinator.store.enqueue_director_run(replace(
        bob_projection,
        transaction_id="tx_split_next",
        event_id="evt_split_next",
        event_sequence=2,
    ))
    alice_context = coordinator.store.visual_novel_stage_context_before_run(
        alice_next.run_id
    )
    bob_context = coordinator.store.visual_novel_stage_context_before_run(
        bob_next.run_id
    )
    assert "title=Alice Brass Key" in "\n".join(alice_context)
    assert "Bob Sealed Letter" not in "\n".join(alice_context)
    assert "title=Bob Sealed Letter" in "\n".join(bob_context)
    assert "Alice Brass Key" not in "\n".join(bob_context)


@pytest.mark.asyncio
async def test_split_private_pov_readiness_waits_for_its_own_finalization(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_split")
    shared = _projection(
        transaction_id="tx_split",
        event_id="evt_split",
        event_sequence=1,
        viewers=("alice",),
        presentation_mode="visual_novel",
    )
    alice_projection = replace(
        shared,
        visible_facts=(("Alice sees the brass key.", 0, 3),),
    )
    bob_projection = replace(
        shared,
        viewer_character_ids=("bob",),
        visible_facts=(("Bob sees the sealed letter.", 0, 3),),
    )
    alice_direction = ImageDirection(
        kind="detail",
        title="Alice Brass Key",
        subject_character_ids=[],
        scene_prompt="A rain-bright brass key in Alice's sight.",
    )
    bob_direction = ImageDirection(
        kind="detail",
        title="Bob Sealed Letter",
        subject_character_ids=[],
        scene_prompt="A sealed letter held within Bob's sight.",
    )

    alice_run = coordinator.store.enqueue_director_run(alice_projection)
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        alice_run.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=[alice_direction],
        ),
    )
    bob_run = coordinator.store.enqueue_director_run(bob_projection)
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        bob_run.run_id,
        ImageDirectorOutput(
            stage_action="replace",
            requests=[bob_direction],
        ),
    )
    alice_job = await coordinator.enqueue_direction(
        projection=alice_projection,
        direction=alice_direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=alice_run.run_id,
        director_attempt=_director_attempt(coordinator, alice_run.run_id),
    )
    assert alice_job is not None
    _finalize_director(coordinator,
        alice_run.run_id,
        projection=alice_projection,
        admitted_job_ids=[alice_job.job_id],
    )
    claimed_job = coordinator.store.claim_next()
    assert claimed_job is not None
    coordinator.store.mark_succeeded(
        claimed_job.job_id,
        GeneratedImageArtifact(
            sha256="a" * 64,
            relative_path="artifacts/aa/alice-stage.webp",
            width=1024,
            height=576,
            byte_count=101,
        ),
    )

    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_split"]},
    ) == (True, True)
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"bob": ["evt_split"]},
    ) == (True, False)
    with pytest.raises(ValueError, match="direction mismatch"):
        await coordinator.enqueue_direction(
            projection=alice_projection,
            direction=alice_direction,
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[],
            director_run_id=bob_run.run_id,
            director_attempt=_director_attempt(coordinator, bob_run.run_id),
        )
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"bob": ["evt_split"]},
    ) == (True, False)

    finalized = _finalize_director(coordinator,
        bob_run.run_id,
        projection=bob_projection,
        admitted_job_ids=[],
    )
    assert finalized.materialized_at is not None
    assert finalized.output is not None
    assert finalized.output.requests == [bob_direction]
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"bob": ["evt_split"]},
    ) == (True, True)
    assert coordinator.store.commit_transaction(
        "tx_split",
        target_checkpoint_sha256="c" * 64,
    )
    bob_stage = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="bob",
        rendered_event_ids=["evt_split"],
    )
    assert bob_stage.artifact is None
    assert bob_stage.fallback_reason == "replacement_failed"


@pytest.mark.asyncio
async def test_visual_novel_stage_reuse_is_explicit_and_failed_replace_is_safe(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    direction = ImageDirection(
        kind="establishing",
        title="Rainy Courtyard",
        subject_character_ids=[],
        scene_prompt=(
            "A rainy courtyard with sheltered stone arcades in a stable "
            "medium-wide scene."
        ),
    )

    _begin(coordinator, "tx_stage_1")
    first_projection = _projection(
        transaction_id="tx_stage_1",
        event_id="evt_stage_1",
        event_sequence=1,
        viewers=("alice",),
        source_turn_index=1,
        presentation_mode="visual_novel",
    )
    first_run = coordinator.store.enqueue_director_run(first_projection)
    assert coordinator.store.claim_next_director_run() is not None
    first_output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    _complete_director(coordinator, first_run.run_id, first_output)
    first_job = await coordinator.enqueue_direction(
        projection=first_projection,
        direction=direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=first_run.run_id,
        director_attempt=_director_attempt(coordinator, first_run.run_id),
    )
    assert first_job is not None
    assert first_job.request.width == 1024
    assert first_job.request.height == 576
    claimed = coordinator.store.claim_next()
    assert claimed is not None
    artifact = GeneratedImageArtifact(
        sha256="b" * 64,
        relative_path="artifacts/bb/stage.webp",
        width=1024,
        height=576,
        byte_count=123,
    )
    coordinator.store.mark_succeeded(claimed.job_id, artifact)
    _finalize_director(coordinator,
        first_run.run_id,
        projection=first_projection,
        admitted_job_ids=[first_job.job_id],
    )
    assert coordinator.store.commit_transaction(
        "tx_stage_1",
        target_checkpoint_sha256="c" * 64,
    )

    _begin(coordinator, "tx_stage_2")
    reuse_projection = _projection(
        transaction_id="tx_stage_2",
        event_id="evt_stage_2",
        event_sequence=2,
        viewers=("alice",),
        source_turn_index=2,
        presentation_mode="visual_novel",
    )
    reuse_run = coordinator.store.enqueue_director_run(reuse_projection)
    assert "current_stage=active" in "\n".join(
        coordinator.store.visual_novel_stage_context_before_run(
            reuse_run.run_id
        )
    )
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        reuse_run.run_id,
        ImageDirectorOutput(stage_action="reuse", requests=[]),
    )
    _finalize_director(coordinator,
        reuse_run.run_id,
        projection=reuse_projection,
        admitted_job_ids=[],
    )
    assert coordinator.store.commit_transaction(
        "tx_stage_2",
        target_checkpoint_sha256="d" * 64,
    )

    reused = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="alice",
        rendered_event_ids=["evt_stage_2"],
    )
    assert reused.action == "reuse"
    assert reused.artifact == artifact
    assert reused.fallback_reason == ""

    _begin(coordinator, "tx_stage_3")
    failed_projection = _projection(
        transaction_id="tx_stage_3",
        event_id="evt_stage_3",
        event_sequence=3,
        viewers=("alice",),
        source_turn_index=3,
        presentation_mode="visual_novel",
    )
    failed_run = coordinator.store.enqueue_director_run(failed_projection)
    assert "current_stage=reused" in "\n".join(
        coordinator.store.visual_novel_stage_context_before_run(
            failed_run.run_id
        )
    )
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        failed_run.run_id,
        ImageDirectorOutput(stage_action="replace", requests=[direction]),
    )
    failed_job = await coordinator.enqueue_direction(
        projection=failed_projection,
        direction=direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=failed_run.run_id,
        director_attempt=_director_attempt(coordinator, failed_run.run_id),
    )
    assert failed_job is not None
    _finalize_director(coordinator,
        failed_run.run_id,
        projection=failed_projection,
        admitted_job_ids=[failed_job.job_id],
    )
    claimed_failed = coordinator.store.claim_next()
    assert claimed_failed is not None
    coordinator.store.mark_failed(claimed_failed.job_id, "worker_failed")
    assert coordinator.store.commit_transaction(
        "tx_stage_3",
        target_checkpoint_sha256="e" * 64,
    )

    failed = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="alice",
        rendered_event_ids=["evt_stage_3"],
    )
    assert failed.action == "replace"
    assert failed.artifact is None
    assert failed.fallback_reason == "replacement_failed"

    _begin(coordinator, "tx_stage_4")
    failed_direction_projection = _projection(
        transaction_id="tx_stage_4",
        event_id="evt_stage_4",
        event_sequence=4,
        viewers=("alice",),
        source_turn_index=4,
        presentation_mode="visual_novel",
    )
    failed_direction_run = coordinator.store.enqueue_director_run(
        failed_direction_projection
    )
    assert coordinator.store.visual_novel_stage_context_before_run(
        failed_direction_run.run_id
    ) == ["current_stage=neutral; reason=no shared compatible stage"]
    assert coordinator.store.claim_next_director_run() is not None
    _fail_director(coordinator,
        failed_direction_run.run_id,
        "director_failed",
    )
    assert coordinator.store.commit_transaction(
        "tx_stage_4",
        target_checkpoint_sha256="f" * 64,
    )

    _begin(coordinator, "tx_stage_5")
    unsafe_reuse_projection = _projection(
        transaction_id="tx_stage_5",
        event_id="evt_stage_5",
        event_sequence=5,
        viewers=("alice",),
        source_turn_index=5,
        presentation_mode="visual_novel",
    )
    unsafe_reuse_run = coordinator.store.enqueue_director_run(
        unsafe_reuse_projection
    )
    assert coordinator.store.visual_novel_stage_context_before_run(
        unsafe_reuse_run.run_id
    ) == ["current_stage=neutral; reason=no shared compatible stage"]
    assert coordinator.store.claim_next_director_run() is not None
    _complete_director(coordinator,
        unsafe_reuse_run.run_id,
        ImageDirectorOutput(stage_action="reuse", requests=[]),
    )
    _finalize_director(coordinator,
        unsafe_reuse_run.run_id,
        projection=unsafe_reuse_projection,
        admitted_job_ids=[],
    )
    assert coordinator.store.commit_transaction(
        "tx_stage_5",
        target_checkpoint_sha256="0" * 64,
    )

    unsafe_reuse = coordinator.store.resolve_visual_novel_stage(
        session_id="image_test",
        pov_character_id="alice",
        rendered_event_ids=["evt_stage_5"],
    )
    assert unsafe_reuse.action == "reuse"
    assert unsafe_reuse.artifact is None
    assert unsafe_reuse.fallback_reason == "reused_stage_transition_failed"


async def _visual_novel_stage_context_for_schedule(
    runtime_root,
    *,
    commit_current_before_claim,
) -> list[str]:
    coordinator = ImageGenerationCoordinator(
        sessions_dir=runtime_root / "sessions",
        config=_config(runtime_root),
        worker=FakeImageWorker(),
    )
    direction = ImageDirection(
        kind="establishing",
        title="Established Courtyard",
        subject_character_ids=[],
        scene_prompt="An established courtyard beneath a clear evening sky.",
    )

    _begin(coordinator, "tx_prior")
    prior_projection = _projection(
        transaction_id="tx_prior",
        event_id="evt_prior",
        event_sequence=1,
        viewers=("alice",),
        source_turn_index=1,
        presentation_mode="visual_novel",
    )
    prior_run = coordinator.store.enqueue_director_run(prior_projection)
    assert coordinator.store.claim_next_director_run() is not None
    prior_output = ImageDirectorOutput(
        stage_action="replace",
        requests=[direction],
    )
    _complete_director(coordinator, prior_run.run_id, prior_output)
    prior_job = await coordinator.enqueue_direction(
        projection=prior_projection,
        direction=direction,
        request_ordinal=0,
        visual_style="cinematic",
        delivery_targets=[],
        director_run_id=prior_run.run_id,
        director_attempt=_director_attempt(coordinator, prior_run.run_id),
    )
    assert prior_job is not None
    claimed_job = coordinator.store.claim_next()
    assert claimed_job is not None
    coordinator.store.mark_succeeded(
        claimed_job.job_id,
        GeneratedImageArtifact(
            sha256="a" * 64,
            relative_path="artifacts/aa/prior-stage.webp",
            width=1024,
            height=576,
            byte_count=123,
        ),
    )
    _finalize_director(coordinator,
        prior_run.run_id,
        projection=prior_projection,
        admitted_job_ids=[prior_job.job_id],
    )
    assert coordinator.store.commit_transaction(
        "tx_prior",
        target_checkpoint_sha256="b" * 64,
    )

    _begin(coordinator, "tx_current")
    current_projection = _projection(
        transaction_id="tx_current",
        event_id="evt_current",
        event_sequence=2,
        viewers=("alice",),
        source_turn_index=2,
        presentation_mode="visual_novel",
    )
    current_run = coordinator.store.enqueue_director_run(current_projection)
    if commit_current_before_claim:
        assert coordinator.store.commit_transaction(
            "tx_current",
            target_checkpoint_sha256="c" * 64,
        )
    claimed_current = coordinator.store.claim_next_director_run()
    assert claimed_current is not None
    assert claimed_current.run_id == current_run.run_id
    assert claimed_current.status == "running"

    sibling_projection = _projection(
        transaction_id="tx_current",
        event_id="evt_same_position",
        event_sequence=2,
        viewers=("alice",),
        source_turn_index=2,
        presentation_mode="visual_novel",
    )
    sibling_run = coordinator.store.enqueue_director_run(sibling_projection)
    assert sibling_run.status == "queued"

    _begin(coordinator, "tx_future")
    future_projection = _projection(
        transaction_id="tx_future",
        event_id="evt_future",
        event_sequence=3,
        viewers=("alice",),
        source_turn_index=3,
        presentation_mode="visual_novel",
    )
    future_run = coordinator.store.enqueue_director_run(future_projection)
    assert future_run.status == "queued"
    assert coordinator.store.commit_transaction(
        "tx_future",
        target_checkpoint_sha256="d" * 64,
    )

    context = coordinator.store.visual_novel_stage_context_before_run(
        current_run.run_id
    )

    if not commit_current_before_claim:
        assert coordinator.store.commit_transaction(
            "tx_current",
            target_checkpoint_sha256="c" * 64,
        )
    return context


@pytest.mark.asyncio
async def test_visual_novel_stage_context_is_relative_to_claimed_run(tmp_path):
    committed_before_claim = await _visual_novel_stage_context_for_schedule(
        tmp_path / "committed-before-claim",
        commit_current_before_claim=True,
    )
    committed_after_context = await _visual_novel_stage_context_for_schedule(
        tmp_path / "committed-after-context",
        commit_current_before_claim=False,
    )

    assert committed_before_claim == committed_after_context
    assert len(committed_before_claim) == 1
    assert committed_before_claim[0].startswith("current_stage=active;")


def test_visual_novel_stage_context_requires_stored_target_run(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )

    with pytest.raises(RuntimeError, match="director run is unavailable"):
        coordinator.store.visual_novel_stage_context_before_run("missing")


@pytest.mark.asyncio
async def test_cancelled_materializing_director_run_cannot_wedge_render_wait(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_1")
    projection = _projection(viewers=("alice",))
    queued = coordinator.store.enqueue_director_run(projection)
    claimed = coordinator.store.claim_next_director_run()
    assert claimed is not None
    assert claimed.run_id == queued.run_id
    _complete_director(coordinator,
        queued.run_id,
        ImageDirectorOutput(requests=[ImageDirection(
            kind="action",
            title="Discarded Rain Run",
            subject_character_ids=["alice"],
            scene_prompt="Alice runs into the rain.",
        )]),
    )

    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
    ) == (True, False)

    await coordinator.cancel_transaction(
        "tx_1",
        reason="narrator_continued",
    )

    with coordinator.store._connect() as db:
        row = db.execute(
            "SELECT status FROM image_director_runs WHERE run_id = ?",
            (queued.run_id,),
        ).fetchone()
    assert row is not None
    assert row["status"] == "cancelled"
    assert coordinator.store.rendered_event_image_status(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
    ) == (False, True)
    assert await coordinator.wait_for_render_images(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
        timeout=0.05,
        discovery_grace_seconds=0,
    ) is True


@pytest.mark.asyncio
async def test_capacity_rejects_new_event_without_fallback(tmp_path):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path, queue_limit=1, per_session_queue_limit=1),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    _begin(coordinator, "tx_2")
    try:
        first = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="establishing",
                title="Rain Station",
                subject_character_ids=[],
                scene_prompt="A rain-washed station.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert first is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        second = await coordinator.enqueue_direction(
            projection=_projection(
                transaction_id="tx_2",
                event_id="evt_2",
                event_sequence=1,
            ),
            direction=ImageDirection(
                kind="detail",
                title="Wet Key",
                subject_character_ids=[],
                scene_prompt="A key on wet concrete.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert second is None
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_committed_raw_identity_delivery_needs_no_story_prose_receipt(
    tmp_path,
):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    delivered = asyncio.Event()

    async def deliver(job, delivery, media, instructions):
        delivered.set()
        return True

    coordinator.register_delivery_handler(ImageDeliveryKind.cli, deliver)
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        job = await coordinator.enqueue_direction(
            projection=_projection(viewers=("alice",)),
            direction=ImageDirection(
                kind="detail",
                title="Brass Key",
                subject_character_ids=[],
                scene_prompt="Rain beads on a brass key.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert job is not None
        await coordinator.commit_transaction(
            "tx_1",
            target_checkpoint_sha256="b" * 64,
        )
        await asyncio.wait_for(delivered.wait(), timeout=2)
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_first_portrait_establishes_reference_and_reroll_swaps_on_success(
    tmp_path,
):
    worker = FakeImageWorker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    delivered_instructions: list[str] = []

    async def deliver(job, delivery, media, instructions):
        delivered_instructions.append(instructions)
        return True

    coordinator.register_delivery_handler(ImageDeliveryKind.cli, deliver)
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        portrait = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="portrait",
                title="Alice Portrait",
                subject_character_ids=["alice"],
                scene_prompt="Alice in her yellow raincoat.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice"), _target("bob")],
        )
        assert portrait is not None
        await coordinator.wait_for_terminal(portrait.job_id, timeout=2)
        original = coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        )
        assert original is not None
        assert original.status == IdentityReferenceStatus.provisional
        await coordinator.commit_transaction(
            "tx_1",
            target_checkpoint_sha256="b" * 64,
        )
        await _wait_until(lambda: len(delivered_instructions) == 2)
        assert all(
            original.candidate_id in instructions
            and "/image lock" in instructions
            and "/image reroll" in instructions
            for instructions in delivered_instructions
        )

        action = await coordinator.enqueue_direction(
            projection=_projection(event_id="evt_2", event_sequence=1),
            direction=ImageDirection(
                kind="action",
                title="Rain Run",
                subject_character_ids=["alice"],
                scene_prompt="Alice runs through the rain.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert action is not None
        assert [item.reference_id for item in action.request.reference_inputs] == [
            original.candidate_id
        ]
        await coordinator.wait_for_terminal(action.job_id, timeout=2)
        group = await coordinator.enqueue_direction(
            projection=_projection(event_id="evt_3", event_sequence=2),
            direction=ImageDirection(
                kind="group_portrait",
                title="Under The Awning",
                subject_character_ids=["alice", "bob"],
                scene_prompt="Alice and Bob wait beneath the awning.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert group is not None
        await coordinator.wait_for_terminal(group.job_id, timeout=2)
        assert coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="bob",
        ) is None

        worker.wait = True
        worker.started = asyncio.Event()
        worker.release = asyncio.Event()
        reroll = await coordinator.reroll_identity_reference(
            session_id="image_test",
            reference_id=original.candidate_id,
            delivery_targets=[_target("alice")],
        )
        assert reroll.request.reference_inputs == []
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        while_running = coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        )
        assert while_running is not None
        assert while_running.candidate_id == original.candidate_id

        worker.release.set()
        await coordinator.wait_for_terminal(reroll.job_id, timeout=2)
        replacement = coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        )
        assert replacement is not None
        assert replacement.candidate_id != original.candidate_id
        assert replacement.reroll_of_reference_id == original.candidate_id
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_retirement_blocks_late_portrait_from_establishing_identity(
    tmp_path,
):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        portrait = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="portrait",
                title="Alice Portrait",
                subject_character_ids=["alice"],
                scene_prompt="Alice beneath the station awning.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert portrait is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)

        coordinator.retire_character_identity(
            session_id="image_test",
            character_id="alice",
            source_turn_index=1,
        )
        worker.release.set()
        await coordinator.wait_for_terminal(portrait.job_id, timeout=2)

        assert coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        ) is None
    finally:
        worker.release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_rewind_restores_identity_retired_by_removed_cull(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    await coordinator.start()
    _begin(coordinator, "tx_identity_before_cull")
    try:
        portrait = await coordinator.enqueue_direction(
            projection=_projection(
                transaction_id="tx_identity_before_cull",
            ),
            direction=ImageDirection(
                kind="portrait",
                title="Alice Portrait",
                subject_character_ids=["alice"],
                scene_prompt="Alice beneath the station awning.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert portrait is not None
        await coordinator.wait_for_terminal(portrait.job_id, timeout=2)
        original = coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        )
        assert original is not None

        coordinator.retire_character_identity(
            session_id="image_test",
            character_id="alice",
            source_turn_index=2,
        )
        assert coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        ) is None

        coordinator.store.cancel_after("image_test", 1)

        restored = coordinator.active_identity_candidate(
            session_id="image_test",
            character_id="alice",
        )
        assert restored is not None
        assert restored.candidate_id == original.candidate_id
        assert restored.status == IdentityReferenceStatus.provisional
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_transaction_cancel_aborts_running_generation_and_deliveries(
    tmp_path,
):
    worker = FakeImageWorker(wait=True)
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        job = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="action",
                title="Rain Run",
                subject_character_ids=["alice"],
                scene_prompt="Alice runs into the rain.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert job is not None
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        assert await coordinator.cancel_transaction("tx_1") == 1
        cancelled = await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert cancelled is not None
        assert cancelled.status == ImageGenerationStatus.cancelled
        assert worker.aborted is True
    finally:
        await coordinator.close()


def test_rewind_cancels_transaction_before_late_director_enqueue(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    _begin(coordinator, "tx_late_projection")

    coordinator.store.cancel_after("image_test", 0)

    with pytest.raises(RuntimeError, match="active image transaction"):
        coordinator.store.enqueue_director_run(
            _projection(transaction_id="tx_late_projection")
        )


@pytest.mark.asyncio
async def test_lineage_reconciliation_cancels_successful_stale_artifact(tmp_path):
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=FakeImageWorker(),
    )
    await coordinator.start()
    _begin(coordinator, "tx_1")
    try:
        job = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="detail",
                title="Brass Key",
                subject_character_ids=[],
                scene_prompt="Rain beads on a brass key.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert job is not None
        await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert coordinator.reconcile_lineage(
            session_id="image_test",
            canonical_event_fingerprints={},
        ) == 1
        stale = coordinator.store.get(job.job_id)
        assert stale is not None
        assert stale.status == ImageGenerationStatus.cancelled
        assert stale.error_code == "event_not_in_lineage"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_restart_commits_speculative_work_when_target_checkpoint_exists(
    tmp_path,
):
    config = _config(tmp_path)
    first = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FakeImageWorker(),
    )
    event = router_output(event_id="evt_recovered", observer_ids=["alice"])
    projection = VisibleEventProjection(
        **{
            **_projection(
                transaction_id="tx_recovered",
                event_id=event.event_id,
            ).__dict__,
            "event_fingerprint": source_event_fingerprint(event),
        }
    )
    first.begin_transaction(
        transaction_id="tx_recovered",
        session_id="image_test",
        source_turn_index=1,
        source_checkpoint_sha256="a" * 64,
    )
    first.store.enqueue_director_run(projection)
    ckpt = _checkpoint()
    ckpt.session.turn_index = 1
    ckpt.canonical_events.append(event)
    session_dir = tmp_path / "sessions" / "image_test"
    session_dir.mkdir(parents=True)
    (session_dir / "ckpt_0001.json").write_text(
        ckpt.model_dump_json(indent=2)
    )

    recovered = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=config,
        worker=FakeImageWorker(),
    )
    try:
        await recovered.start()
        with recovered.store._connect() as db:
            row = db.execute(
                """
                SELECT status, target_checkpoint_sha256
                FROM image_transactions WHERE transaction_id = ?
                """,
                ("tx_recovered",),
            ).fetchone()
        assert row is not None
        assert row["status"] == "committed"
        assert len(row["target_checkpoint_sha256"]) == 64
    finally:
        await recovered.close()
