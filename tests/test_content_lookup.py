from __future__ import annotations

import sqlite3

import pytest

from app.engine.content_lookup import (
    MissingContentError,
    append_router_content_lookup_records,
    plan_llm_router_content_lookup_requests,
)
from app.schemas.content import ContentPackState, IntroducedContentRef
from tests.support.factories import checkpoint


def _pack_db(tmp_path, rows: list[tuple[str, str, str, str, str, str]]):
    db_path = tmp_path / "pack.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE content_cards (
                pack_id TEXT,
                ref TEXT,
                content_hash TEXT,
                kind TEXT,
                visibility TEXT,
                summary TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO content_cards VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return db_path


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
        "pack": ContentPackState(
            pack_id="pack",
            metadata={
                "db_path": str(db_path),
                "aliases": {
                    "threshold": {
                        "ref": "room/entry",
                        "reason": "player inspects threshold",
                    }
                },
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
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE content_aliases (
                pack_id TEXT,
                alias TEXT,
                ref TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO content_aliases VALUES ('pack', 'vampire', 'front/vampire')"
        )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            metadata={"db_path": str(db_path)},
        )
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


def test_lookup_preflight_raises_auditable_missing_content_error(tmp_path):
    db_path = _pack_db(tmp_path, [])
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            metadata={
                "db_path": str(db_path),
                "aliases": {"secret door": "room/secret"},
            },
        )
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
        "pack": ContentPackState(
            pack_id="pack",
            introduced_refs={
                "pack::room/entry::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-old",
                )
            },
            metadata={
                "db_path": str(db_path),
                "aliases": {"threshold": "room/entry"},
            },
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
        "pack": ContentPackState(
            pack_id="pack",
            introduced_refs={
                "pack::room/entry::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-old",
                )
            },
            metadata={
                "db_path": str(db_path),
                "aliases": {"threshold": "room/entry"},
            },
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
