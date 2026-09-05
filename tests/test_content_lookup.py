from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.content_pack_compiler import (
    CompiledContentPackMismatchError,
    CompiledContentPackWriter,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.engine.content_lookup import (
    EventRouterContentLookupOutput,
    MissingContentError,
    append_router_content_lookup_records,
    append_router_content_lookup_records_with_llm,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.conversation import ConversationMessage
from app.schemas.content import ContentPackState, IntroducedContentRef
from tests.support.factories import canonical_event, checkpoint, llm_response


PACK_VERSION = "1.0.0"
SOURCE_FINGERPRINT = "sha256:test-source"


def _pack_db(
    tmp_path,
    rows: list[tuple[str, str, str, str, str, str]],
    *,
    aliases: list[tuple[str, str]] | None = None,
    source_fingerprint: str = SOURCE_FINGERPRINT,
):
    db_path = tmp_path / "pack.sqlite"
    writer = CompiledContentPackWriter(
        db_path,
        pack_id="pack",
        pack_version=PACK_VERSION,
        source_fingerprint=source_fingerprint,
    )
    writer.write_pack(
        pages=[],
        cards=[
            {
                "pack_id": pack_id,
                "ref": ref,
                "content_hash": content_hash,
                "card_kind": kind,
                "visibility": visibility,
                "summary": summary,
                "review_status": "approved",
                "confidence": 1.0,
            }
            for pack_id, ref, content_hash, kind, visibility, summary in rows
        ],
        aliases=[
            {
                "alias": alias,
                "ref": ref,
                "review_status": "approved",
                "confidence": 1.0,
            }
            for alias, ref in aliases or []
        ],
    )
    return db_path


def _pack_state(
    db_path,
    *,
    introduced_refs=None,
    pending_signals=None,
    **metadata_overrides,
):
    metadata = {
        "db_path": str(db_path),
        "pack_version": PACK_VERSION,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "schema_version": CONTENT_PACK_SCHEMA_VERSION,
    }
    metadata.update(metadata_overrides)
    return ContentPackState(
        pack_id="pack",
        introduced_refs=introduced_refs or {},
        pending_signals=pending_signals or {},
        metadata=metadata,
    )


def test_lookup_preflight_fetches_alias_match_once_as_assistant_history(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-entry",
                "location_card",
                "hidden",
                "Entry chamber context.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            aliases={
                "threshold": {
                    "ref": "room/entry",
                    "reason": "player inspects threshold",
                }
            },
        )
    }

    first = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold.",
    )
    second = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold again.",
    )

    assert first == [
        (
            "location_card ref=room/entry visibility=hidden "
            'pack=pack summary="Entry chamber context."'
        )
    ]
    assert second == []
    assert [message.role for message in ckpt.session_conversation] == ["assistant"]
    assert ckpt.session_conversation[0].content == first[0]
    assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
        "pack::room/entry::hash-entry"
    ]


def test_lookup_preflight_uses_sqlite_alias_index(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "front/vampire",
                "hash-front",
                "front_signal",
                "hidden",
                "The vampire notices public trouble.",
            )
        ],
        aliases=[("vampire", "front/vampire")],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(db_path)
    }

    records = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I ask whether the vampire has heard about this.",
    )

    assert records == [
        (
            "front_signal ref=front/vampire visibility=hidden "
            "pack=pack "
            'summary="The vampire notices public trouble."'
        )
    ]
    assert [message.content for message in ckpt.session_conversation] == records
    assert ckpt.canonical_events == []
    assert not hasattr(ckpt, "transcript")
    assert ckpt.session.narrator_render_jobs == []
    assert ckpt.session.delivery_outbox == []


def test_lookup_preflight_raises_auditable_missing_content_error(tmp_path):
    db_path = _pack_db(tmp_path, [])
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(db_path, aliases={"secret door": "room/secret"})
    }

    with pytest.raises(MissingContentError) as exc:
        append_router_content_lookup_records(
            ckpt,
            actor_id="alice",
            current_input="I search for the secret door.",
        )

    assert "pack=pack ref=room/secret alias=secret door" in str(exc.value)
    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


def test_lookup_preflight_rejects_missing_source_fingerprint(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-entry",
                "location_card",
                "hidden",
                "Entry chamber context.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            source_fingerprint="",
            aliases={"threshold": "room/entry"},
        )
    }

    with pytest.raises(CompiledContentPackMismatchError, match="source_fingerprint"):
        append_router_content_lookup_records(
            ckpt,
            actor_id="alice",
            current_input="I inspect the threshold.",
        )

    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


def test_lookup_preflight_rejects_stale_source_fingerprint(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-entry",
                "location_card",
                "hidden",
                "Entry chamber context.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            source_fingerprint="sha256:stale",
            aliases={"threshold": "room/entry"},
        )
    }

    with pytest.raises(CompiledContentPackMismatchError, match="source_fingerprint"):
        append_router_content_lookup_records(
            ckpt,
            actor_id="alice",
            current_input="I inspect the threshold.",
        )

    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


@pytest.mark.parametrize(
    ("alias", "ref"),
    [
        ("blocked room", "room/blocked"),
        ("unreviewed room", "room/unreviewed"),
        ("low confidence room", "room/low-confidence"),
        ("raw source room", "room/raw-source"),
        ("hashless room", "room/hashless"),
    ],
)
def test_lookup_preflight_refuses_unsafe_runtime_rows_without_appending(
    tmp_path,
    alias,
    ref,
):
    db_path = _pack_db(
        tmp_path,
        [
            ("pack", "room/blocked", "hash-blocked", "location_card", "hidden", "Blocked."),
            (
                "pack",
                "room/unreviewed",
                "hash-unreviewed",
                "location_card",
                "hidden",
                "Unreviewed.",
            ),
            (
                "pack",
                "room/low-confidence",
                "hash-low-confidence",
                "location_card",
                "hidden",
                "Low confidence.",
            ),
            (
                "pack",
                "room/raw-source",
                "hash-raw-source",
                "location_card",
                "hidden",
                "Raw source.",
            ),
            ("pack", "room/hashless", "hashless", "location_card", "hidden", "Hashless."),
        ],
        aliases=[
            ("blocked room", "room/blocked"),
            ("unreviewed room", "room/unreviewed"),
            ("low confidence room", "room/low-confidence"),
            ("raw source room", "room/raw-source"),
            ("hashless room", "room/hashless"),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE content_cards SET gate_status = 'blocked' WHERE ref = 'room/blocked'"
        )
        conn.execute(
            """
            UPDATE content_cards
            SET review_status = 'needs_review'
            WHERE ref = 'room/unreviewed'
            """
        )
        conn.execute(
            """
            UPDATE content_cards
            SET gate_status = 'blocked', gate_reasons_json = '["low_confidence"]'
            WHERE ref = 'room/low-confidence'
            """
        )
        conn.execute(
            """
            UPDATE content_cards
            SET metadata_json = '{"raw_source_path":"/private/module.pdf"}'
            WHERE ref = 'room/raw-source'
            """
        )
        conn.execute(
            "UPDATE content_cards SET content_hash = '' WHERE ref = 'room/hashless'"
        )

    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}

    with pytest.raises(MissingContentError) as exc:
        append_router_content_lookup_records(
            ckpt,
            actor_id="alice",
            current_input=f"I inspect the {alias}.",
        )

    assert f"ref={ref}" in str(exc.value)
    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


def test_lookup_preflight_is_atomic_when_one_requested_record_fails_gate(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/ready",
                "hash-ready",
                "location_card",
                "hidden",
                "Ready room.",
            ),
            (
                "pack",
                "room/raw-source",
                "hash-raw-source",
                "location_card",
                "hidden",
                "Raw source.",
            ),
        ],
        aliases=[
            ("ready room", "room/ready"),
            ("raw source room", "room/raw-source"),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE content_cards
            SET metadata_json = '{"raw_source_path":"/private/module.pdf"}'
            WHERE ref = 'room/raw-source'
            """
        )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}

    with pytest.raises(MissingContentError):
        append_router_content_lookup_records(
            ckpt,
            actor_id="alice",
            current_input="I inspect the ready room and the raw source room.",
        )

    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


def test_lookup_preflight_reintroduces_known_ref_when_hash_changes(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-new",
                "location_card",
                "hidden",
                "Updated entry chamber context.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            introduced_refs={
                "pack::room/entry::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-old",
                )
            },
            aliases={"threshold": "room/entry"},
        )
    }

    records = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold.",
    )

    assert records == [
        (
            "location_card ref=room/entry visibility=hidden "
            'pack=pack summary="Updated entry chamber context."'
        )
    ]
    assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
        "pack::room/entry::hash-new"
    ]


def test_lookup_preflight_missing_known_ref_is_noop(tmp_path):
    db_path = _pack_db(tmp_path, [])
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            introduced_refs={
                "pack::room/entry::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-old",
                )
            },
            aliases={"threshold": "room/entry"},
        )
    }

    records = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold.",
    )

    assert records == []
    assert ckpt.session_conversation == []
    assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
        "pack::room/entry::hash-old"
    ]


def test_lookup_preflight_known_ref_without_db_path_is_noop():
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            introduced_refs={
                "pack::room/entry::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-old",
                )
            },
            metadata={"aliases": {"threshold": "room/entry"}},
        )
    }

    records = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold.",
    )

    assert records == []
    assert ckpt.session_conversation == []


def test_lookup_preflight_noops_without_content_pack():
    ckpt = checkpoint()

    records = append_router_content_lookup_records(
        ckpt,
        actor_id="alice",
        current_input="I inspect the threshold.",
    )

    assert records == []
    assert ckpt.session.content_state == {}
    assert ckpt.session_conversation == []


def test_llm_lookup_preflight_fetches_reviewed_hidden_ref(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/secret",
                "hash-secret",
                "location_card",
                "hidden",
                "A reviewed secret door latch record.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=llm_response(
        EventRouterContentLookupOutput(
            requests=[
                {
                    "pack_id": "pack",
                    "ref": "room/secret",
                    "reason": "The wall inspection may need the secret latch record.",
                }
            ]
        )
    ))

    records = asyncio_run(
        append_router_content_lookup_records_with_llm(
            ckpt,
            actor_id="alice",
            current_input="I study the odd draft coming from the wall.",
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )

    assert records == [
        (
            "location_card ref=room/secret visibility=hidden "
            'pack=pack summary="A reviewed secret door latch record."'
        )
    ]
    assert [message.content for message in ckpt.session_conversation] == records
    assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
        "pack::room/secret::hash-secret"
    ]
    lookup_messages = client.complete.await_args.kwargs["messages"]
    lookup_text = "\n".join(message["content"] for message in lookup_messages)
    assert "room/secret" in lookup_text
    assert str(db_path) not in lookup_text
    assert "source_fingerprint" not in lookup_text


def test_llm_lookup_preflight_rebuilds_legacy_hashed_router_history(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-entry",
                "location_card",
                "hidden",
                "Entry chamber context.",
            )
        ],
    )
    ckpt = checkpoint()
    event = canonical_event()
    event.event_id = "evt_deadbeefcafe"
    event.causal_lane_id = "lane_0123456789abcdef"
    ckpt.canonical_events = [event]
    ckpt.session_conversation = [ConversationMessage(
        role="assistant",
        content=(
            "prior_event evt_deadbeefcafe lane=lane_0123456789abcdef "
            "submissions=submission_cafebabefeed"
        ),
    )]
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=llm_response(
        EventRouterContentLookupOutput(
            requests=[],
            no_lookup_reason="No reviewed content is needed.",
        )
    ))

    asyncio_run(
        append_router_content_lookup_records_with_llm(
            ckpt,
            actor_id="alice",
            current_input="I wait.",
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )

    lookup_messages = client.complete.await_args.kwargs["messages"]
    lookup_text = "\n".join(message["content"] for message in lookup_messages)
    assert "prior_event sequence=0 causal_group=0" in lookup_text
    assert "evt_deadbeefcafe" not in lookup_text
    assert "lane_0123456789abcdef" not in lookup_text
    assert "submission_cafebabefeed" not in lookup_text


def test_llm_lookup_preflight_fails_loudly_on_unresolved_required_ref(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/entry",
                "hash-entry",
                "location_card",
                "hidden",
                "A reviewed entry room record.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=llm_response(
        EventRouterContentLookupOutput(
            requests=[
                {
                    "pack_id": "pack",
                    "ref": "room/missing",
                    "reason": "The model requested an unauthored room.",
                }
            ]
        )
    ))

    with pytest.raises(MissingContentError) as exc:
        asyncio_run(
            append_router_content_lookup_records_with_llm(
                ckpt,
                actor_id="alice",
                current_input="I inspect the blank wall.",
                client=client,
                prompt_mgr=PromptManager("app/prompts"),
            )
        )

    assert "ref=room/missing" in str(exc.value)
    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].introduced_refs == {}


def test_llm_lookup_preflight_retry_is_bounded_and_receives_missing_refs(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "room/secret",
                "hash-secret",
                "location_card",
                "hidden",
                "A reviewed secret door latch record.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(side_effect=[
        llm_response(
            EventRouterContentLookupOutput(
                requests=[{"pack_id": "pack", "ref": "room/missing"}]
            )
        ),
        llm_response(
            EventRouterContentLookupOutput(
                requests=[{"pack_id": "pack", "ref": "room/secret"}]
            )
        ),
    ])

    records = asyncio_run(
        append_router_content_lookup_records_with_llm(
            ckpt,
            actor_id="alice",
            current_input="I inspect the blank wall.",
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
            max_lookup_passes=2,
        )
    )

    assert client.complete.await_count == 2
    second_lookup_text = "\n".join(
        message["content"]
        for message in client.complete.await_args_list[1].kwargs["messages"]
    )
    assert "room/missing" in second_lookup_text
    assert records[0].startswith("location_card ref=room/secret")


def test_llm_lookup_preflight_noops_without_catalog():
    ckpt = checkpoint()
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock()

    records = asyncio_run(
        append_router_content_lookup_records_with_llm(
            ckpt,
            actor_id="alice",
            current_input="I inspect the threshold.",
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )

    assert records == []
    client.complete.assert_not_awaited()


def asyncio_run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
