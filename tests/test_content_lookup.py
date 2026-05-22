from __future__ import annotations

import sqlite3

import pytest

from app.engine.content_pack_compiler import (
    CompiledContentPackMismatchError,
    CompiledContentPackWriter,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.engine.content_lookup import (
    MissingContentError,
    append_router_content_lookup_records,
    plan_llm_router_content_lookup_requests,
)
from app.schemas.content import ContentPackState, IntroducedContentRef
from tests.support.factories import checkpoint


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
            "location_card ref=room/entry visibility=hidden hash=hash-entry "
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
            "hash=hash-front pack=pack "
            'summary="The vampire notices public trouble."'
        )
    ]
    assert [message.content for message in ckpt.session_conversation] == records
    assert ckpt.canonical_events == []
    assert ckpt.transcript == []
    assert ckpt.session.render_buffers == {}


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
            "location_card ref=room/entry visibility=hidden hash=hash-new "
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


def test_llm_lookup_extension_point_is_currently_disabled():
    assert plan_llm_router_content_lookup_requests(
        checkpoint(),
        actor_id="alice",
        current_input="I inspect the threshold.",
        known_refs=set(),
        deterministic_requests=[],
    ) == []
