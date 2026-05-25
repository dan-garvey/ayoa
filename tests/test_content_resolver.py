from __future__ import annotations

from types import SimpleNamespace

from app.engine.content_resolver import (
    append_pending_router_content_records,
    content_ref_needs_introduction,
    drain_pending_content_signals,
    format_compact_record,
    format_front_signal_record,
    load_content_cards,
)
from app.schemas.content import ContentPackState, PendingContentSignal
from tests.support.factories import checkpoint


def test_pending_content_signals_drain_once_and_only_changed_refs_repeat():
    state = SimpleNamespace(
        pack_id="curse-test",
        introduced_refs=[],
        pending_signals=[
            {
                "ref": "area.entry",
                "content_hash": "hash-1",
                "status": "pending",
                "kind": "location_card",
                "visibility": "hidden",
                "summary": "Entry chamber.",
                "exits": ["north door"],
                "hazards": ["loose floor"],
                "clues": ["old crest"],
                "narrator_text": "leak to narrator",
                "agent_prompt": "leak to agent",
            }
        ],
    )

    records = drain_pending_content_signals(state)

    assert records == [
        (
            'location_card ref=area.entry exits=["north door"] '
            'hazards=["loose floor"] clues=["old crest"] visibility=hidden '
            "hash=hash-1 pack=curse-test summary=\"Entry chamber.\""
        )
    ]
    assert state.pending_signals == []
    assert content_ref_needs_introduction(
        state,
        {"ref": "area.entry", "content_hash": "hash-1"},
    ) is False
    assert "narrator" not in records[0]
    assert "agent" not in records[0]

    state.pending_signals.append(
        {
            "ref": "area.entry",
            "content_hash": "hash-1",
            "status": "pending",
            "kind": "location_card",
            "summary": "Entry chamber.",
            "exits": ["north door"],
        }
    )
    assert drain_pending_content_signals(state) == []
    assert state.pending_signals == []

    state.pending_signals.append(
        {
            "ref": "area.entry",
            "content_hash": "hash-2",
            "status": "pending",
            "kind": "location_card",
            "summary": "Entry chamber after the door opens.",
            "exits": ["north door", "stairs"],
        }
    )
    changed = drain_pending_content_signals(state)

    assert len(changed) == 1
    assert "hash=hash-2" in changed[0]
    assert "Entry chamber after the door opens." in changed[0]
    assert content_ref_needs_introduction(
        state,
        {"ref": "area.entry", "content_hash": "hash-2"},
    ) is False


def test_front_signal_record_uses_allowlisted_router_fields_only():
    record = format_front_signal_record(
        {
            "pack_id": "curse-test",
            "ref": "front.villain",
            "content_hash": "front-hash-1",
            "kind": "front_signal",
            "visibility": "hidden",
            "actor": "strahd",
            "knows": "the party made a public enemy",
            "pressure": "send spies",
            "summary": "A public consequence may reach the front.",
            "narrator_text": "do not show this to players",
            "agent_inbox": "do not send this to NPC agents",
            "source_path": "/private/module.pdf",
        }
    )

    assert record == (
        "front_signal ref=front.villain actor=strahd "
        'knows="the party made a public enemy" pressure="send spies" '
        "visibility=hidden hash=front-hash-1 pack=curse-test "
        'summary="A public consequence may reach the front."'
    )
    assert "narrator" not in record
    assert "agent" not in record
    assert "source_path" not in record
    assert "/private" not in record


def test_dict_backed_content_state_uses_ref_id_keys_and_clears_pending_map():
    state = {
        "pack_id": "pack",
        "introduced_refs": {},
        "pending_signals": {
            "sig-1": {
                "ref_id": "room/1",
                "content_hash": "hash-a",
                "status": "pending",
                "kind": "content_known",
                "visibility": "hidden",
                "summary": "Private room key.",
            }
        },
    }

    records = drain_pending_content_signals(state)

    assert records == [
        (
            "content_known ref=room/1 scope=router visibility=hidden "
            'hash=hash-a kind=content_known pack=pack summary="Private room key."'
        )
    ]
    assert state["pending_signals"] == {}
    assert sorted(state["introduced_refs"]) == ["pack::room/1::hash-a"]
    assert content_ref_needs_introduction(
        state,
        {"ref_id": "room/1", "content_hash": "hash-a"},
    ) is False


def test_drain_prunes_non_pending_signals_without_introducing_refs():
    state = {
        "pack_id": "pack",
        "introduced_refs": {},
        "pending_signals": {
            "sig-pending": {
                "ref_id": "room/pending",
                "content_hash": "hash-pending",
                "status": "pending",
                "kind": "content_known",
                "visibility": "hidden",
                "summary": "Pending room key.",
            },
            "sig-resolved": {
                "ref_id": "room/resolved",
                "content_hash": "hash-resolved",
                "status": "resolved",
                "kind": "content_known",
                "visibility": "hidden",
                "summary": "Resolved stale key.",
            },
            "sig-dismissed": {
                "ref_id": "room/dismissed",
                "content_hash": "hash-dismissed",
                "status": "dismissed",
                "kind": "content_known",
                "visibility": "hidden",
                "summary": "Dismissed stale key.",
            },
        },
    }

    records = drain_pending_content_signals(state)

    assert records == [
        (
            "content_known ref=room/pending scope=router visibility=hidden "
            "hash=hash-pending kind=content_known pack=pack "
            'summary="Pending room key."'
        )
    ]
    assert state["pending_signals"] == {}
    assert sorted(state["introduced_refs"]) == [
        "pack::room/pending::hash-pending"
    ]
    assert "room/resolved" not in records[0]
    assert "room/dismissed" not in records[0]


def test_load_content_cards_rejects_simple_sqlite_rows_by_default(tmp_path):
    import sqlite3

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
                summary TEXT,
                title TEXT,
                body TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO content_cards VALUES (
                'pack', 'room/1', 'hash-a', 'location', 'hidden',
                'Private room key.', 'Room One', 'Only router should see this.',
                '{"tier":"dm"}'
            )
            """
        )

    assert load_content_cards(db_path, refs=["room/1"], pack_id="pack") == []

    cards = load_content_cards(
        db_path,
        refs=["room/1"],
        pack_id="pack",
        runtime_only=False,
    )

    assert len(cards) == 1
    assert cards[0].ref == "room/1"
    assert cards[0].content_hash == "hash-a"
    assert cards[0].metadata == {"tier": "dm"}


def test_load_content_cards_allows_reviewed_router_metadata(tmp_path):
    import sqlite3

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
                summary TEXT,
                title TEXT,
                body TEXT,
                metadata_json TEXT,
                review_status TEXT,
                gate_status TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO content_cards VALUES (
                'pack', 'front.villain', 'hash-front', 'front_signal',
                'router_hidden', 'A front can react.', 'Front', 'Router packet.',
                '{"actions":["apply pressure"],"goals":["recover maps"],"minions":["actor.guard"]}',
                'approved', 'runtime_ready'
            )
            """
        )

    cards = load_content_cards(
        db_path,
        refs=["front.villain"],
        pack_id="pack",
        runtime_only=True,
    )

    assert len(cards) == 1
    assert cards[0].metadata["actions"] == ["apply pressure"]
    assert cards[0].metadata["goals"] == ["recover maps"]
    record = format_compact_record(cards[0], pack_id="pack")
    assert record.startswith("front_signal ")
    assert 'goals=["recover maps"]' in record
    assert 'actions=["apply pressure"]' in record

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE content_cards
            SET metadata_json = '{"raw_source_path":"/private/module.pdf"}'
            WHERE ref = 'front.villain'
            """
        )

    assert (
        load_content_cards(
            db_path,
            refs=["front.villain"],
            pack_id="pack",
            runtime_only=True,
        )
        == []
    )


def test_append_pending_router_content_records_noops_for_non_content_story():
    ckpt = checkpoint()

    records = append_pending_router_content_records(ckpt)

    assert records == []
    assert ckpt.session.content_state == {}
    assert ckpt.session_conversation == []


def test_append_pending_router_content_records_adds_assistant_history_once():
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            pending_signals={
                "sig-1": PendingContentSignal(
                    signal_id="sig-1",
                    pack_id="pack",
                    ref_id="front/villain",
                    content_hash="hash-front",
                    status="pending",
                    metadata={
                        "kind": "front_signal",
                        "visibility": "hidden",
                        "actor": "villain",
                        "knows": "the party caused public trouble",
                        "pressure": "send spies",
                        "summary": "The front may now react.",
                        "source_path": "/private/module.pdf",
                    },
                )
            },
        )
    }

    first = append_pending_router_content_records(ckpt)
    second = append_pending_router_content_records(ckpt)

    assert first == [
        (
            "front_signal ref=front/villain actor=villain "
            'knows="the party caused public trouble" pressure="send spies" '
            "visibility=hidden hash=hash-front pack=pack "
            'summary="The front may now react."'
        )
    ]
    assert second == []
    assert [message.role for message in ckpt.session_conversation] == ["assistant"]
    assert ckpt.session_conversation[0].content == first[0]
    assert "source_path" not in first[0]
    assert ckpt.session.content_state["pack"].pending_signals == {}
    assert sorted(ckpt.session.content_state["pack"].introduced_refs) == [
        "pack::front/villain::hash-front"
    ]


def test_append_pending_router_content_records_ignores_terminal_signals():
    ckpt = checkpoint()
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            pending_signals={
                "sig-dismissed": PendingContentSignal(
                    signal_id="sig-dismissed",
                    pack_id="pack",
                    ref_id="front/stale",
                    content_hash="hash-stale",
                    status="dismissed",
                    metadata={
                        "kind": "front_signal",
                        "visibility": "hidden",
                        "summary": "This stale front was dismissed.",
                    },
                ),
                "sig-resolved": PendingContentSignal(
                    signal_id="sig-resolved",
                    pack_id="pack",
                    ref_id="front/resolved",
                    content_hash="hash-resolved",
                    status="resolved",
                    metadata={
                        "kind": "front_signal",
                        "visibility": "hidden",
                        "summary": "This front already resolved.",
                    },
                ),
            },
        )
    }

    records = append_pending_router_content_records(ckpt)

    assert records == []
    assert ckpt.session_conversation == []
    assert ckpt.session.content_state["pack"].pending_signals == {}
    assert ckpt.session.content_state["pack"].introduced_refs == {}
