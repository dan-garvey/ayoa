from __future__ import annotations

import json

from app.bot.engine_bridge import EngineBridge
from app.engine.checkpoint_manager import CheckpointManager
from app.llm.config import LLMConfig
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import (
    ContentFrontState,
    ContentKnowledgeEntityState,
    ContentPackState,
    ContentVillainState,
    IntroducedContentRef,
    PendingContentSignal,
)
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from app.schemas.state import SessionState
from tests.support.factories import checkpoint


def test_content_state_defaults_are_empty():
    state = SessionState(session_id="s")
    pack = ContentPackState()

    assert state.content_state == {}
    assert pack.introduced_refs == {}
    assert pack.pending_signals == {}
    assert pack.fronts == {}
    assert pack.villains == {}
    assert pack.knowledge_map == {}
    assert IntroducedContentRef().dedupe_key() == "::::"
    assert PendingContentSignal().content_key() == "::::"


def test_content_state_round_trips_through_checkpoint_dump():
    ref = IntroducedContentRef(
        pack_id="pack",
        ref_id="location/chapel",
        content_hash="sha256:abc",
        label="Old Chapel",
        kind="Location",
        source_event_id="evt_1",
        introduced_at_s=30,
    )
    signal = PendingContentSignal(
        signal_id="sig_1",
        pack_id="pack",
        ref_id="villain/warden",
        content_hash="sha256:def",
        reason="The warden was named but not loaded.",
        source_event_id="evt_2",
        priority=2,
        created_at_s=45,
        requested_fields=["goals", "goals", "location"],
        metadata={"scope": "front"},
    )
    front = ContentFrontState(
        front_id="front_prison",
        label="Prison Unrest",
        status="Active",
        clock=2,
        max_clock=6,
        villain_ids=["warden"],
        introduced_ref_keys=[ref.dedupe_key()],
    )
    villain = ContentVillainState(
        villain_id="warden",
        label="The Warden",
        status="Hidden",
        front_ids=["front_prison"],
        goals=["Preserve control"],
        introduced_ref_keys=[signal.content_key()],
    )
    knowledge = ContentKnowledgeEntityState(
        entity_id="warden",
        known_refs=["pack:villain/warden@sha256:def"],
        suspected_refs=[
            "pack:villain/warden@sha256:def",
            "pack:room/escape@sha256:ghi",
        ],
        notes="Tracks the party's prison movements.",
        last_source_fact_ids=["f01", "f01", "bad value"],
    )
    pack = ContentPackState(
        pack_id="pack",
        introduced_refs={ref.dedupe_key(): ref},
        pending_signals={signal.signal_id: signal},
        fronts={front.front_id: front},
        villains={villain.villain_id: villain},
        knowledge_map={"stale_key": knowledge},
        metadata={"source": "fixture"},
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": pack}

    payload = ckpt.model_dump(mode="json")
    rebuilt = CheckpointFile.model_validate(payload)

    assert rebuilt.model_dump(mode="json") == payload
    rebuilt_pack = rebuilt.session.content_state["pack"]
    assert rebuilt_pack.introduced_refs[ref.dedupe_key()].kind == "location"
    assert rebuilt_pack.pending_signals["sig_1"].requested_fields == [
        "goals",
        "location",
    ]
    assert rebuilt_pack.fronts["front_prison"].status == "active"
    assert rebuilt_pack.villains["warden"].status == "hidden"
    assert sorted(rebuilt_pack.knowledge_map) == ["warden"]
    assert rebuilt_pack.knowledge_map["warden"].suspected_refs == [
        "pack:room/escape@sha256:ghi"
    ]


def test_default_checkpoint_dump_omits_imported_asset_source_sentinels():
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            pending_signals={
                "sig": PendingContentSignal(
                    signal_id="sig",
                    pack_id="pack",
                    ref_id="room/redacted",
                    metadata={
                        "source_ref": "raw-row",
                        "delivery_ref": "asset://synthetic/hidden-map",
                    },
                )
            },
            metadata={
                "db_path": "/private/table/compiled.sqlite",
                "raw_ocr": "PROTECTED_SOURCE_EXCERPT",
                "asset": {
                    "delivery_ref": "asset://synthetic/hidden-map",
                    "safe_status": "reviewed",
                },
            },
        )
    }

    dumped = json.dumps(ckpt.model_dump(mode="json"), sort_keys=True)

    for sentinel in (
        "/private/table/compiled.sqlite",
        "PROTECTED_SOURCE_EXCERPT",
        "asset://synthetic/hidden-map",
        "source_ref",
        "delivery_ref",
        "raw_ocr",
    ):
        assert sentinel not in dumped
    assert "safe_status" in dumped


def test_checkpoint_manager_save_preserves_private_runtime_metadata(tmp_path):
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            metadata={
                "db_path": str(tmp_path / "compiled.sqlite"),
                "asset_media_root": str(tmp_path / "media"),
            },
        )
    }
    manager = CheckpointManager(str(tmp_path / "sessions"))

    manager.save(ckpt)
    loaded = manager.load_latest("s")

    assert loaded.session.content_state["pack"].metadata["db_path"] == str(
        tmp_path / "compiled.sqlite"
    )
    assert loaded.session.content_state["pack"].metadata["asset_media_root"] == str(
        tmp_path / "media"
    )


def test_story_start_preserves_private_runtime_content_pack_metadata(tmp_path):
    stories_dir = tmp_path / "stories"
    sessions_dir = tmp_path / "sessions"
    story_id = "content_seed"
    story_dir = stories_dir / story_id
    story_dir.mkdir(parents=True)
    bridge = EngineBridge(
        stories_dir=str(stories_dir),
        sessions_dir=str(sessions_dir),
        llm_config=LLMConfig(api_key="anthropic-key", openai_api_key="openai-key"),
    )
    ckpt = checkpoint(session_id=story_id)
    ckpt.session.story_id = story_id
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            metadata={
                "db_path": str(tmp_path / "compiled.sqlite"),
                "asset_media_root": str(tmp_path / "media"),
                "pack_version": "1.0.0",
            },
        )
    }
    (story_dir / "ckpt_0000.json").write_text(
        ckpt.model_dump_json(
            indent=2,
            context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
        )
    )

    bridge.create_empty_session("session_1")
    loaded = bridge.load_story_into_session("session_1", story_id)

    assert loaded.session.content_state["pack"].metadata["db_path"] == str(
        tmp_path / "compiled.sqlite"
    )
    assert loaded.session.content_state["pack"].metadata["asset_media_root"] == str(
        tmp_path / "media"
    )


def test_introduced_refs_are_dedup_friendly_by_pack_ref_and_hash():
    first = IntroducedContentRef(
        pack_id="pack",
        ref_id="room/1",
        content_hash="hash-a",
        label="First label",
    )
    duplicate = IntroducedContentRef(
        pack_id=" pack ",
        ref_id=" room/1 ",
        content_hash=" hash-a ",
        label="Second label",
    )
    changed = IntroducedContentRef(
        pack_id="pack",
        ref_id="room/1",
        content_hash="hash-b",
    )

    introduced = {
        first.dedupe_key(): first,
        duplicate.dedupe_key(): duplicate,
        changed.dedupe_key(): changed,
    }

    assert first.dedupe_key() == "pack::room/1::hash-a"
    assert duplicate.dedupe_key() == first.dedupe_key()
    assert changed.dedupe_key() == "pack::room/1::hash-b"
    assert sorted(introduced) == [
        "pack::room/1::hash-a",
        "pack::room/1::hash-b",
    ]
    assert introduced[first.dedupe_key()].label == "Second label"


def test_front_state_persists_in_session_content_state():
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            fronts={
                "front_a": ContentFrontState(
                    front_id="front_a",
                    label="Storm Front",
                    status="active",
                    clock=3,
                    max_clock=4,
                    villain_ids=["villain_a"],
                    notes="Pressure is rising.",
                )
            },
            villains={
                "villain_a": ContentVillainState(
                    villain_id="villain_a",
                    label="The Rival",
                    status="revealed",
                    front_ids=["front_a"],
                )
            },
        )
    }

    rebuilt = CheckpointFile.model_validate(ckpt.model_dump(mode="json"))
    front = rebuilt.session.content_state["pack"].fronts["front_a"]

    assert front.label == "Storm Front"
    assert front.clock == 3
    assert front.max_clock == 4
    assert front.villain_ids == ["villain_a"]
    assert (
        rebuilt.session.content_state["pack"].villains["villain_a"].front_ids
        == ["front_a"]
    )


def test_pristine_checkpoint_still_validates_without_content_state():
    payload = CheckpointFile(
        session=SessionState(session_id="minimal")
    ).model_dump(mode="json")

    rebuilt = CheckpointFile.model_validate(payload)

    assert rebuilt.session.content_state == {}
