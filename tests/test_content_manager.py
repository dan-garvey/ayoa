from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.content_manager import (
    ContentManagerValidationError,
    build_content_manager_messages,
    build_entity_knowledge_block,
    build_recent_canonical_facts_block,
    format_content_manager_update_records,
    plan_content_manager_updates,
    validate_content_manager_output,
)
from app.engine.content_pack_compiler import (
    CompiledContentPackWriter,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.content import ContentPackState
from app.schemas.content_manager import ContentManagerOutput
from app.schemas.events import ObservableFact
from tests.support.factories import checkpoint, llm_response, router_output


PACK_VERSION = "1.0.0"
SOURCE_FINGERPRINT = "sha256:test-source"


def _pack_db(tmp_path, rows: list[tuple[str, str, str, str, str, str]]):
    db_path = tmp_path / "pack.sqlite"
    writer = CompiledContentPackWriter(
        db_path,
        pack_id="pack",
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
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
    )
    return db_path


def _pack_state(db_path):
    return ContentPackState(
        pack_id="pack",
        metadata={
            "db_path": str(db_path),
            "pack_version": PACK_VERSION,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "schema_version": CONTENT_PACK_SCHEMA_VERSION,
        },
    )


def test_content_manager_prompt_uses_recent_facts_and_compressed_entity_knowledge():
    ckpt = checkpoint()
    ckpt.canonical_events = [
        router_output(
            event_id=f"evt_{index}",
            facts=[ObservableFact.all(f"public fact {index}")],
        )
        for index in range(14)
    ]
    ckpt.canonical_events.append(
        router_output(facts=[ObservableFact.all("unsafe /private/module.pdf fact")])
    )
    entity_knowledge = {
        "strahd": {
            "known_refs": ["pack:front/strahd@hash-old"],
            "notes": "watching the party",
            "source": "/private/module.pdf",
        }
    }

    facts_block = build_recent_canonical_facts_block(ckpt, limit=12)
    entity_block = build_entity_knowledge_block(entity_knowledge)
    messages = build_content_manager_messages(
        ckpt,
        entity_knowledge=entity_knowledge,
        prompt_mgr=PromptManager("app/prompts"),
        catalog_block=(
            'pack=pack ref=front/strahd kind=front_signal '
            'summary="The antagonist tracks public trouble."'
        ),
        entity_knowledge_block=entity_block,
        max_recent_facts=12,
    )

    assert "public fact 0" not in facts_block
    assert not any(
        line.endswith('text="public fact 1"')
        for line in facts_block.splitlines()
    )
    assert "public fact 2" in facts_block
    assert "public fact 13" in facts_block
    assert "/private" not in facts_block
    assert ".pdf" not in facts_block
    assert "known_refs=pack:front/strahd@hash-old" in entity_block
    assert "watching the party" in entity_block
    assert "/private" not in entity_block
    assert ".pdf" not in entity_block

    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "public fact 13" not in system
    assert "strahd" not in system
    assert "public fact 13" in user
    assert "strahd" in user
    assert "/private" not in user
    assert ".pdf" not in user


def test_plan_content_manager_updates_uses_role_and_validates_refs(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "front/strahd",
                "hash-front",
                "front_signal",
                "hidden",
                "The antagonist tracks public trouble.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    ckpt.canonical_events = [
        router_output(facts=[ObservableFact.all("The party lights the beacon.")])
    ]
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=llm_response(
        ContentManagerOutput(
            updates=[
                {
                    "entity_id": "strahd",
                    "pack_id": "pack",
                    "ref": "front/strahd",
                    "content_hash": "",
                    "knowledge_state": "known",
                    "reason": "f01 makes the front relevant",
                    "source_fact_ids": ["f01"],
                }
            ]
        )
    ))

    output = asyncio.run(
        plan_content_manager_updates(
            ckpt,
            entity_knowledge={"strahd": {"known_refs": []}},
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )

    assert client.complete.await_args.kwargs["role"] == "content_manager"
    assert client.complete.await_args.kwargs["response_model"] is ContentManagerOutput
    assert output.updates[0].content_hash == "hash-front"
    assert format_content_manager_update_records(output) == [
        (
            "content_update entity=strahd state=known pack=pack "
            "ref=front/strahd hash=hash-front facts=f01 "
            'reason="f01 makes the front relevant"'
        )
    ]


def test_content_manager_validation_rejects_missing_ref(tmp_path):
    db_path = _pack_db(tmp_path, [])
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    output = ContentManagerOutput(
        updates=[
            {
                "entity_id": "strahd",
                "pack_id": "pack",
                "ref": "front/missing",
                "content_hash": "",
                "knowledge_state": "known",
                "reason": "",
                "source_fact_ids": ["f01"],
            }
        ]
    )

    with pytest.raises(ContentManagerValidationError, match="ref=front/missing"):
        validate_content_manager_output(
            ckpt,
            output,
            entity_knowledge={"strahd": {"known_refs": []}},
        )


def test_content_manager_validation_rejects_unknown_entity(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "front/strahd",
                "hash-front",
                "front_signal",
                "hidden",
                "The antagonist tracks public trouble.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    output = ContentManagerOutput(
        updates=[
            {
                "entity_id": "rahadin",
                "pack_id": "pack",
                "ref": "front/strahd",
                "content_hash": "",
                "knowledge_state": "known",
                "reason": "",
                "source_fact_ids": [],
            }
        ]
    )

    with pytest.raises(ContentManagerValidationError, match="unknown entity_id=rahadin"):
        validate_content_manager_output(
            ckpt,
            output,
            entity_knowledge={"strahd": {"known_refs": []}},
        )


def test_content_manager_validation_rejects_hash_mismatch(tmp_path):
    db_path = _pack_db(
        tmp_path,
        [
            (
                "pack",
                "front/strahd",
                "hash-front",
                "front_signal",
                "hidden",
                "The antagonist tracks public trouble.",
            )
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    output = ContentManagerOutput(
        updates=[
            {
                "entity_id": "strahd",
                "pack_id": "pack",
                "ref": "front/strahd",
                "content_hash": "hash-stale",
                "knowledge_state": "known",
                "reason": "source /private/module.pdf",
                "source_fact_ids": [],
            }
        ]
    )

    assert output.updates[0].reason == ""
    with pytest.raises(ContentManagerValidationError, match="content hash mismatch"):
        validate_content_manager_output(
            ckpt,
            output,
            entity_knowledge={"strahd": {"known_refs": []}},
        )
