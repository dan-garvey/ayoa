from __future__ import annotations

import asyncio
import hashlib
import sqlite3
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
    projection_checkpoint_snapshot,
    source_event_fingerprint,
)
from app.engine.image_generation import (
    ImageDeliveryTarget,
    ImageGenerationConfig,
    ImageGenerationCoordinator,
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
from app.schemas.events import ObservableFact
from app.schemas.image_director import ImageDirection, ImageDirectorOutput
from app.schemas.image_generation import (
    FrozenReferenceInput,
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
) -> VisibleEventProjection:
    return VisibleEventProjection(
        session_id="image_test",
        transaction_id=transaction_id,
        source_turn_index=1,
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
        delivery_kind="cli",
        viewer_delivery_bindings=tuple((viewer, "") for viewer in viewers),
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
        delivery_kind="cli",
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
        delivery_kind="cli",
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
        delivery_kind="cli",
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
        delivery_kind="cli",
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
        delivery_kind="cli",
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


def test_v1_actor_cadence_store_is_retired_in_direct_v7_migration(tmp_path):
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
    assert version == "7"
    assert old_table is None


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
async def test_generation_starts_speculatively_but_delivery_waits_for_commit_and_prose(
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
        await asyncio.sleep(0.6)
        assert delivered == []

        coordinator.open_prose_gates(
            transaction_id="tx_1",
            rendered_event_ids_by_pov={"alice": ["evt_1"]},
        )
        await _wait_until(lambda: delivered == ["alice"])
        coordinator.open_prose_gates_for_session(
            session_id="image_test",
            rendered_event_ids_by_pov={"bob": ["evt_1"]},
        )
        await _wait_until(lambda: delivered == ["alice", "bob"])
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
        coordinator.store.complete_director_run(
            claimed.run_id,
            ImageDirectorOutput(requests=[direction_one, direction_two]),
        )

        first = await coordinator.enqueue_direction(
            projection=projection,
            direction=direction_one,
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        second = await coordinator.enqueue_direction(
            projection=projection,
            direction=direction_two,
            request_ordinal=1,
            visual_style="cinematic",
            delivery_targets=[_target("alice")],
        )
        assert first is not None
        assert second is not None
        assert first.request.title == "Rain Run"
        assert second.request.title == "Brass Key"

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
    coordinator.store.complete_director_run(
        queued.run_id,
        ImageDirectorOutput(requests=[direction]),
    )

    assert await coordinator.wait_for_render_images(
        session_id="image_test",
        rendered_event_ids_by_pov={"alice": ["evt_1"]},
        timeout=0.05,
        discovery_grace_seconds=0,
    ) is False

    coordinator.store.finalize_director_materialization(
        queued.run_id,
        ImageDirectorOutput(requests=[]),
    )
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
async def test_prose_receipt_survives_when_it_arrives_before_director_job(
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
        assert coordinator.open_prose_gates_for_session(
            session_id="image_test",
            rendered_event_ids_by_pov={"alice": ["evt_1"]},
        ) == 0
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
        coordinator.open_prose_gates(
            transaction_id="tx_1",
            rendered_event_ids_by_pov={
                "alice": ["evt_1"],
                "bob": ["evt_1"],
            },
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
        assert "runs through the rain" in "\n".join(
            coordinator.store.recent_illustrations(
                "image_test",
                viewer_character_ids=["alice"],
            )
        )
        assert "runs through the rain" not in "\n".join(
            coordinator.store.recent_illustrations(
                "image_test",
                viewer_character_ids=["bob"],
            )
        )
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
