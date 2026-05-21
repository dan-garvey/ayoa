from __future__ import annotations

from types import SimpleNamespace

from app.engine.content_resolver import (
    content_ref_needs_introduction,
    drain_pending_content_signals,
    format_front_signal_record,
    load_content_cards,
)


def test_pending_content_signals_drain_once_and_only_changed_refs_repeat():
    state = SimpleNamespace(
        pack_id="curse-test",
        introduced_refs=[],
        pending_signals=[
            {
                "ref": "area.entry",
                "content_hash": "hash-1",
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


def test_load_content_cards_reads_simple_sqlite_rows(tmp_path):
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

    cards = load_content_cards(db_path, refs=["room/1"], pack_id="pack")

    assert len(cards) == 1
    assert cards[0].ref == "room/1"
    assert cards[0].content_hash == "hash-a"
    assert cards[0].metadata == {"tier": "dm"}
