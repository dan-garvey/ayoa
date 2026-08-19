from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from app.bot.engine_bridge import EngineBridge
from app.engine.image_director import (
    ImageDirector,
    PublicCharacterVisual,
    VisibleEventProjection,
    build_projection_groups,
    projection_checkpoint_snapshot,
)
from app.engine.image_generation import (
    ImageDeliveryTarget,
    ImageGenerationConfig,
    ImageGenerationCoordinator,
)
from app.engine.prompt_manager import PromptManager
from app.engine.reviewed_visual_references import (
    ReviewedVisualReferenceError,
    freeze_story_visual_references,
    load_frozen_visual_references,
    validate_story_visual_references,
)
from app.schemas.characters import (
    CharacterRecord,
    CharacterVisuals,
    PrivateState,
    PublicSheet,
)
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.events import ObservableFact
from app.schemas.image_director import ImageDirection, ImageDirectorOutput
from app.schemas.image_generation import (
    IdentityReferenceStatus,
    ImageDeliveryKind,
    ImageGenerationStatus,
    ImageWorkerResult,
)
from app.schemas.state import SessionState, StorySetting, WorldState
from app.schemas.visual_references import ReviewedVisualReference
from tests.support.factories import router_output


class _Worker:
    supported_generation_modes = ("compose", "edit")

    def __init__(self, *, error_code: str = "") -> None:
        self.available = True
        self.error_code = error_code
        self.requests = []
        self.config = SimpleNamespace(
            model_id="offline/test-image-model",
            model_revision="test",
        )

    async def generate(self, request, *, output_path):
        self.requests.append(request)
        if self.error_code:
            from app.engine.image_worker_client import ImageWorkerError

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

    async def abort_current(self):
        return None

    async def close(self):
        return None


class _DirectorClient:
    def __init__(self) -> None:
        self.messages = []

    async def complete(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(parsed=ImageDirectorOutput(requests=[]))


def _write_png(path: Path, color: tuple[int, int, int]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path, format="PNG")
    return path.read_bytes()


def _metadata(
    path: Path,
    *,
    reference_id: str,
    purpose: str,
    scope: str,
) -> ReviewedVisualReference:
    data = path.read_bytes()
    return ReviewedVisualReference(
        reference_id=reference_id,
        storage_ref=path.name,
        mime_type="image/png",
        width=12,
        height=8,
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        purpose=purpose,
        scope=scope,
        scope_id="alice" if scope == "character" else "station",
        selection_hint=(
            "Front identity reference for portraits."
            if scope == "character"
            else "Platform environment reference for location framing."
        ),
        diffusion_authorized=True,
    )


def _checkpoint(
    *,
    session_id: str = "reviewed_refs",
    references: list[ReviewedVisualReference] | None = None,
    identity_reference_id: str = "",
    location_reference_ids: list[str] | None = None,
) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(
            session_id=session_id,
            story_id="test_story",
            turn_index=1,
            player_character_id="alice",
            character_bindings={"alice": "11"},
        ),
        world_state=WorldState(
            setting=StorySetting(
                genre="mystery",
                era="contemporary",
                tone="quiet",
                premise="A witness waits through the rain.",
                visual_style="restrained cinematic illustration",
            )
        ),
        characters=[
            CharacterRecord(
                character_id="alice",
                name="Alice",
                location="station",
                public_sheet=PublicSheet(
                    role="witness",
                    appearance="short dark hair and a yellow raincoat",
                ),
                visuals=CharacterVisuals(
                    default_loadout="canvas satchel",
                    identity_reference_id=identity_reference_id,
                ),
                private_state=PrivateState(intentions_enabled=True),
                is_playable=True,
            )
        ],
        reviewed_visual_references=references or [],
        location_visual_reference_ids=(
            {"station": location_reference_ids} if location_reference_ids else {}
        ),
    )


def _reviewed_story(
    tmp_path: Path,
) -> tuple[
    Path,
    CheckpointFile,
    ReviewedVisualReference,
    ReviewedVisualReference,
]:
    story_dir = tmp_path / "story"
    visual_dir = story_dir / "visual-references"
    identity_path = visual_dir / "alice.png"
    location_path = visual_dir / "station.png"
    _write_png(identity_path, (200, 180, 30))
    _write_png(location_path, (20, 60, 100))
    identity = _metadata(
        identity_path,
        reference_id="authored.alice.v1",
        purpose="identity",
        scope="character",
    )
    location = _metadata(
        location_path,
        reference_id="authored.station.v1",
        purpose="environment",
        scope="location",
    )
    checkpoint = _checkpoint(
        references=[identity, location],
        identity_reference_id=identity.reference_id,
        location_reference_ids=[location.reference_id],
    )
    return story_dir, checkpoint, identity, location


def _projection(
    *,
    has_location_reference: bool = True,
    location_label: str = "station",
) -> VisibleEventProjection:
    return VisibleEventProjection(
        session_id="reviewed_refs",
        transaction_id="tx_reviewed",
        source_turn_index=1,
        event_id="evt_reviewed",
        event_sequence=0,
        event_fingerprint=hashlib.sha256(b"evt_reviewed").hexdigest(),
        viewer_character_ids=("alice",),
        perception_level="direct",
        effective_at_s=2,
        duration_s=1,
        visible_facts=(("Rain passes across the platform.", 0, 1),),
        characters=(
            PublicCharacterVisual(
                character_id="alice",
                name="Alice",
                appearance="short dark hair and a yellow raincoat",
                default_loadout="canvas satchel",
                depiction_policy="normal",
                is_new_character=False,
                has_identity_reference=True,
            ),
        ),
        story_genre="mystery",
        story_era="contemporary",
        story_tone="quiet",
        story_premise="A witness waits through the rain.",
        canonical_event_count=1,
        active_roster_count=1,
        total_roster_count=1,
        engine_visual_style="restrained cinematic illustration",
        delivery_kind="cli",
        viewer_delivery_bindings=(("alice", ""),),
        engine_location_label=(location_label if has_location_reference else ""),
        has_location_reference=has_location_reference,
    )


def _target() -> ImageDeliveryTarget:
    return ImageDeliveryTarget(
        pov_character_id="alice",
        delivery_kind=ImageDeliveryKind.cli,
        delivery={"character_id": "alice"},
    )


def _config(tmp_path: Path, **updates) -> ImageGenerationConfig:
    values = {
        "runtime_root": tmp_path / "runtime",
        "queue_limit": 8,
        "per_session_queue_limit": 8,
        "max_references": 4,
    }
    values.update(updates)
    return ImageGenerationConfig(**values)


def _register(
    coordinator: ImageGenerationCoordinator,
    checkpoint: CheckpointFile,
    frozen,
) -> None:
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=frozen,
    )
    coordinator.begin_transaction(
        transaction_id="tx_reviewed",
        session_id=checkpoint.session.session_id,
        source_turn_index=1,
        source_checkpoint_sha256="a" * 64,
    )


def _fake_webp(width: int, height: int) -> bytes:
    encoded_width = width - 1
    encoded_height = height - 1
    payload = b"\x2f" + bytes(
        (
            encoded_width & 0xFF,
            ((encoded_width >> 8) & 0x3F) | ((encoded_height & 0x03) << 6),
            (encoded_height >> 2) & 0xFF,
            (encoded_height >> 10) & 0x0F,
        )
    )
    chunk = b"VP8L" + len(payload).to_bytes(4, "little") + payload + b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_reviewed_references_validate_freeze_and_detect_tampering(tmp_path):
    story_dir, checkpoint, identity, _location = _reviewed_story(tmp_path)
    runtime_root = tmp_path / "runtime"

    validate_story_visual_references(checkpoint, story_dir=story_dir)
    frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=runtime_root,
    )

    assert list(frozen) == [
        "authored.alice.v1",
        "authored.station.v1",
    ]
    identity_frozen = frozen[identity.reference_id]
    assert identity_frozen.relative_path.startswith(
        f"artifacts/references/{identity.sha256[:2]}/"
    )
    assert (runtime_root / identity_frozen.relative_path).read_bytes() == (
        story_dir / "visual-references" / identity.storage_ref
    ).read_bytes()
    assert (runtime_root / identity_frozen.relative_path).stat().st_mode & 0o077 == 0
    assert (
        load_frozen_visual_references(
            checkpoint,
            runtime_root=runtime_root,
        )
        == frozen
    )

    source_path = story_dir / "visual-references" / identity.storage_ref
    _write_png(source_path, (1, 2, 3))
    with pytest.raises(
        ReviewedVisualReferenceError,
        match="reference_(?:byte_count|hash)_mismatch",
    ):
        validate_story_visual_references(checkpoint, story_dir=story_dir)

    frozen_path = runtime_root / identity_frozen.relative_path
    frozen_path.write_bytes(b"changed")
    with pytest.raises(
        ReviewedVisualReferenceError,
        match="reference_byte_count_mismatch",
    ):
        load_frozen_visual_references(
            checkpoint,
            runtime_root=runtime_root,
        )


def test_reviewed_reference_rejects_unsafe_relative_path(tmp_path):
    story_dir, checkpoint, identity, location = _reviewed_story(tmp_path)
    payload = identity.model_dump()
    payload["storage_ref"] = "../alice.png"
    with pytest.raises(ValidationError, match="stay inside"):
        ReviewedVisualReference.model_validate(payload)

    outside = tmp_path / "outside.png"
    _write_png(outside, (1, 2, 3))
    source = story_dir / "visual-references" / location.storage_ref
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(
        ReviewedVisualReferenceError,
        match="reference_path_unavailable_or_unsafe",
    ):
        validate_story_visual_references(checkpoint, story_dir=story_dir)


@pytest.mark.asyncio
async def test_llm_projection_exposes_only_authored_selection_metadata(tmp_path):
    story_dir, checkpoint, identity, location = _reviewed_story(tmp_path)
    default_json = checkpoint.model_dump_json()
    private_json = checkpoint.model_dump_json(
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True}
    )
    public_snapshot = projection_checkpoint_snapshot(checkpoint)

    for secret in (
        identity.reference_id,
        identity.storage_ref,
        identity.sha256,
        location.reference_id,
        location.storage_ref,
        location.sha256,
    ):
        assert secret not in default_json
        assert secret not in public_snapshot.model_dump_json()
        assert secret in private_json

    event = router_output(
        event_id="evt_location",
        observer_ids=["alice"],
        facts=[ObservableFact.all("Rain passes across the platform.")],
    )
    projection = build_projection_groups(
        checkpoint=public_snapshot,
        event=event,
        event_sequence=0,
        transaction_id="tx_location",
        source_turn_index=1,
        actor_id="alice",
        active_identity_character_ids={"alice"},
        active_location_labels={"station"},
    )[0]
    assert projection.has_location_reference is True
    assert projection.engine_location_label == "station"

    checkpoint.characters.append(
        CharacterRecord(
            character_id="bob",
            name="Bob",
            location="hidden-laboratory",
        )
    )
    mediated = router_output(
        event_id="evt_mediated",
        observer_ids=["alice"],
        facts=[ObservableFact.all("A distant impact echoes through a speaker.")],
    )
    mediated.observers[0].observation_level = "i"
    mediated_projection = build_projection_groups(
        checkpoint=checkpoint,
        event=mediated,
        event_sequence=1,
        transaction_id="tx_mediated",
        source_turn_index=1,
        actor_id="bob",
        active_location_labels={"station", "hidden-laboratory"},
    )[0]
    assert mediated_projection.engine_location_label == "station"

    client = _DirectorClient()
    director = ImageDirector(
        client,
        PromptManager(prompts_dir="app/prompts"),
    )
    await director.decide(projection)
    rendered = "\n".join(message["content"] for message in client.messages)
    assert "has_identity_reference=yes" in rendered
    assert "has_location_reference=yes" in rendered
    assert "authored.alice.v1" in rendered
    assert "authored.station.v1" in rendered
    assert "Front identity reference for portraits." in rendered
    assert "Platform environment reference for location framing." in rendered
    assert identity.storage_ref not in rendered
    assert identity.sha256 not in rendered
    assert location.storage_ref not in rendered
    assert location.sha256 not in rendered
    # The opaque location binding is private runtime routing, not model input.
    assert "applies_to=station" not in rendered.lower()
    assert str(story_dir) not in rendered


@pytest.mark.asyncio
async def test_subject_then_location_references_forward_to_worker(tmp_path):
    story_dir, checkpoint, identity, location = _reviewed_story(tmp_path)
    style_payload = location.model_dump()
    style_payload.update(
        reference_id="authored.station.style.v1",
        purpose="style",
    )
    style = ReviewedVisualReference.model_validate(style_payload)
    checkpoint.reviewed_visual_references.append(style)
    checkpoint.location_visual_reference_ids["station"].append(style.reference_id)
    frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=tmp_path / "runtime",
    )
    worker = _Worker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    _register(coordinator, checkpoint, frozen)
    await coordinator.start()
    try:
        job = await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="action",
                title="Platform Crossing",
                subject_character_ids=["alice"],
                scene_prompt="Alice crosses the rain-swept platform.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target()],
        )
        assert job is not None
        assert [item.reference_id for item in job.request.reference_inputs] == [
            identity.reference_id,
            location.reference_id,
            style.reference_id,
        ]
        await coordinator.wait_for_terminal(job.job_id, timeout=2)
        assert worker.requests[0].reference_inputs == (job.request.reference_inputs)
        coordinator.begin_transaction(
            session_id=checkpoint.session.session_id,
            transaction_id="tx_edit",
            source_turn_index=1,
            source_checkpoint_sha256="d" * 64,
        )
        edited = await coordinator.enqueue_direction(
            projection=build_projection_groups(
                checkpoint=projection_checkpoint_snapshot(checkpoint),
                event=router_output(
                    event_id="evt_edit",
                    observer_ids=["alice"],
                    facts=[ObservableFact.all("Alice turns into the rain.")],
                ),
                event_sequence=1,
                transaction_id="tx_edit",
                source_turn_index=1,
                actor_id="alice",
                active_identity_character_ids={"alice"},
                active_location_labels={"station"},
            )[0],
            direction=ImageDirection(
                kind="action",
                title="Turning in Rain",
                subject_character_ids=["alice"],
                generation_mode="edit",
                reference_ids=[identity.reference_id],
                scene_prompt="Preserve Alice while turning her into the rain.",
            ),
            request_ordinal=2,
            visual_style="cinematic",
            delivery_targets=[_target()],
        )
        assert edited is not None
        assert edited.request.generation_mode == "edit"
        assert [
            item.reference_id for item in edited.request.reference_inputs
        ] == [identity.reference_id]
        portrait = await coordinator.enqueue_direction(
            projection=_projection(
                has_location_reference=False,
            ),
            direction=ImageDirection(
                kind="portrait",
                title="Alice Portrait",
                subject_character_ids=["alice"],
                scene_prompt="Individual portrait of Alice in her raincoat.",
            ),
            request_ordinal=1,
            visual_style="cinematic",
            delivery_targets=[_target()],
        )
        assert portrait is not None
        await coordinator.wait_for_terminal(portrait.job_id, timeout=2)
        # A reviewed identity is already locked; an ordinary portrait must not
        # replace it with a generated provisional candidate or lock reminder.
        assert (
            coordinator.active_identity_candidate(
                session_id=checkpoint.session.session_id,
                character_id="alice",
            )
            is None
        )
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_required_identity_and_location_set_fails_when_limit_cannot_fit(
    tmp_path,
):
    story_dir, checkpoint, _identity, _location = _reviewed_story(tmp_path)
    frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=tmp_path / "runtime",
    )
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path, max_references=1),
        worker=_Worker(),
    )
    _register(coordinator, checkpoint, frozen)
    with pytest.raises(
        ValueError,
        match="required authored reference set",
    ):
        await coordinator.enqueue_direction(
            projection=_projection(),
            direction=ImageDirection(
                kind="action",
                title="Platform Crossing",
                subject_character_ids=["alice"],
                scene_prompt="Alice crosses the platform.",
            ),
            request_ordinal=0,
            visual_style="cinematic",
            delivery_targets=[_target()],
        )


@pytest.mark.asyncio
async def test_authored_reroll_keeps_fallback_until_generated_success(
    tmp_path,
):
    story_dir, checkpoint, identity, _location = _reviewed_story(tmp_path)
    frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=tmp_path / "runtime",
    )
    worker = _Worker()
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=worker,
    )
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=frozen,
    )
    await coordinator.start()
    try:
        reroll = await coordinator.reroll_identity_reference(
            session_id=checkpoint.session.session_id,
            reference_id=identity.reference_id,
            delivery_targets=[_target()],
            checkpoint=checkpoint,
            source_checkpoint_sha256="a" * 64,
        )
        assert [item.reference_id for item in reroll.request.reference_inputs] == [
            identity.reference_id
        ]
        assert (
            coordinator.store.reviewed_identity_reference(
                session_id=checkpoint.session.session_id,
                character_id="alice",
            )
            is not None
        )
        await coordinator.wait_for_terminal(reroll.job_id, timeout=2)
        replacement = coordinator.active_identity_candidate(
            session_id=checkpoint.session.session_id,
            character_id="alice",
        )
        assert replacement is not None
        assert replacement.status == IdentityReferenceStatus.provisional
        assert replacement.reroll_of_reference_id == identity.reference_id
        assert (
            coordinator.store.reviewed_identity_reference(
                session_id=checkpoint.session.session_id,
                character_id="alice",
            )
            is not None
        )
    finally:
        await coordinator.close()

    failed_config = ImageGenerationConfig(
        runtime_root=tmp_path / "failed-runtime",
        queue_limit=8,
        per_session_queue_limit=8,
        max_references=4,
    )
    failed_frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=failed_config.runtime_root,
    )
    failed = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "failed-sessions",
        config=failed_config,
        worker=_Worker(error_code="offline_failure"),
    )
    failed.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=failed_frozen,
    )
    await failed.start()
    try:
        job = await failed.reroll_identity_reference(
            session_id=checkpoint.session.session_id,
            reference_id=identity.reference_id,
            delivery_targets=[_target()],
            checkpoint=checkpoint,
            source_checkpoint_sha256="a" * 64,
        )
        terminal = await failed.wait_for_terminal(job.job_id, timeout=2)
        assert terminal.status == ImageGenerationStatus.failed
        assert (
            failed.active_identity_candidate(
                session_id=checkpoint.session.session_id,
                character_id="alice",
            )
            is None
        )
        assert (
            failed.store.reviewed_identity_reference(
                session_id=checkpoint.session.session_id,
                character_id="alice",
            )
            is not None
        )
    finally:
        await failed.close()


@pytest.mark.asyncio
async def test_cull_suppresses_authored_binding_and_rewind_restores_it(
    tmp_path,
):
    story_dir, checkpoint, identity, _location = _reviewed_story(tmp_path)
    frozen = freeze_story_visual_references(
        checkpoint,
        story_dir=story_dir,
        runtime_root=tmp_path / "runtime",
    )
    coordinator = ImageGenerationCoordinator(
        sessions_dir=tmp_path / "sessions",
        config=_config(tmp_path),
        worker=_Worker(),
    )
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=frozen,
    )
    frozen_path = (
        coordinator.config.runtime_root / frozen[identity.reference_id].relative_path
    )
    assert "alice" in coordinator.active_identity_character_ids(
        checkpoint.session.session_id
    )

    coordinator.retire_character_identity(
        session_id=checkpoint.session.session_id,
        character_id="alice",
        source_turn_index=5,
    )
    checkpoint.characters[0].visuals.identity_reference_id = ""
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=load_frozen_visual_references(
            checkpoint,
            runtime_root=coordinator.config.runtime_root,
        ),
    )
    assert "alice" not in coordinator.active_identity_character_ids(
        checkpoint.session.session_id
    )
    assert frozen_path.exists()

    await coordinator.cancel_after(checkpoint.session.session_id, 4)
    checkpoint.characters[0].visuals.identity_reference_id = identity.reference_id
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=load_frozen_visual_references(
            checkpoint,
            runtime_root=coordinator.config.runtime_root,
        ),
    )
    assert "alice" in coordinator.active_identity_character_ids(
        checkpoint.session.session_id
    )
    assert (
        coordinator.store.reviewed_identity_reference(
            session_id=checkpoint.session.session_id,
            character_id="alice",
        )
        is not None
    )
    assert frozen_path.exists()

    coordinator.suppress_reviewed_identity_binding(
        session_id=checkpoint.session.session_id,
        character_id="alice",
    )
    assert "alice" not in coordinator.active_identity_character_ids(
        checkpoint.session.session_id
    )
    assert frozen_path.exists()
    coordinator.register_reviewed_visual_references(
        checkpoint=checkpoint,
        frozen_references=load_frozen_visual_references(
            checkpoint,
            runtime_root=coordinator.config.runtime_root,
        ),
    )
    assert "alice" in coordinator.active_identity_character_ids(
        checkpoint.session.session_id
    )


def test_story_without_reviewed_references_needs_no_asset_directory(tmp_path):
    checkpoint = _checkpoint()
    story_dir = tmp_path / "plain-story"
    story_dir.mkdir()

    validate_story_visual_references(checkpoint, story_dir=story_dir)
    assert (
        freeze_story_visual_references(
            checkpoint,
            story_dir=story_dir,
            runtime_root=tmp_path / "runtime",
        )
        == {}
    )
    assert (
        load_frozen_visual_references(
            checkpoint,
            runtime_root=tmp_path / "runtime",
        )
        == {}
    )

    with pytest.raises(
        ValidationError,
        match="unknown authored identity reference",
    ):
        _checkpoint(identity_reference_id="authored.missing.v1")


@pytest.mark.asyncio
async def test_story_and_session_loading_freeze_then_revalidate_references(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-offline-test")
    story_dir, checkpoint, identity, _location = _reviewed_story(tmp_path / "source")
    checkpoint.session.session_id = "seed"
    checkpoint.session.turn_index = 0
    (story_dir / "ckpt_0000.json").write_text(
        checkpoint.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )
    )
    bridge = EngineBridge(
        stories_dir=str(story_dir.parent),
        sessions_dir=str(tmp_path / "sessions"),
        prompts_dir="app/prompts",
    )
    try:
        bridge.create_empty_session("live")
        loaded = bridge.load_story_into_session("live", story_dir.name)
        assert loaded.session.story_id == story_dir.name
        assert (
            loaded.characters[0].visuals.identity_reference_id == identity.reference_id
        )
        frozen = load_frozen_visual_references(
            loaded,
            runtime_root=bridge.image_generation.config.runtime_root,
        )
        frozen_path = (
            bridge.image_generation.config.runtime_root
            / frozen[identity.reference_id].relative_path
        )
        assert frozen_path.exists()

        culled = loaded.model_copy(deep=True)
        culled.session.turn_index = 2
        culled.characters[0].visuals.identity_reference_id = ""
        bridge.checkpoint_mgr.save(culled)
        bridge.load_latest("live")
        assert "alice" not in bridge.image_generation.active_identity_character_ids(
            "live"
        )
        bridge.preview_rewind("live", 1)
        # Historical preview must not mutate live runtime bindings before the
        # rewind is confirmed.
        assert "alice" not in bridge.image_generation.active_identity_character_ids(
            "live"
        )
        original_frozen_bytes = frozen_path.read_bytes()
        frozen_path.write_bytes(b"tampered")
        with pytest.raises(
            ReviewedVisualReferenceError,
            match="reference_byte_count_mismatch",
        ):
            await bridge.rewind_session("live", 1)
        assert bridge.list_checkpoint_turns("live")[-1] == 2
        frozen_path.write_bytes(original_frozen_bytes)

        await bridge.rewind_session("live", 1)
        assert "alice" in bridge.image_generation.active_identity_character_ids("live")

        reroll = await bridge.reroll_image_identity(
            session_id="live",
            reference_id="",
            pov_character_id="alice",
            delivery_kind=ImageDeliveryKind.cli,
            delivery={"character_id": "alice"},
        )
        assert reroll.request.reroll_of_reference_id == identity.reference_id
        assert [item.reference_id for item in reroll.request.reference_inputs] == [
            identity.reference_id
        ]

        # Live checkpoints resolve the private immutable copy, not mutable
        # story source paths.
        for path in (story_dir / "visual-references").iterdir():
            path.unlink()
        assert bridge.load_latest("live").session.story_id == story_dir.name

        frozen_path.write_bytes(b"tampered")
        resumed = EngineBridge(
            stories_dir=str(story_dir.parent),
            sessions_dir=str(tmp_path / "sessions"),
            prompts_dir="app/prompts",
        )
        try:
            with pytest.raises(
                ReviewedVisualReferenceError,
                match="reference_byte_count_mismatch",
            ):
                resumed.load_latest("live")
        finally:
            await resumed.close()
    finally:
        await bridge.close()
