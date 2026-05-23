from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.content_manager import (
    ContentManagerValidationError,
    build_candidate_turn_entities_block,
    build_content_manager_messages,
    build_known_router_refs_block,
    build_recent_canonical_facts_block,
    format_content_manager_router_records,
    plan_content_manager_updates,
    validate_content_manager_output,
)
from app.engine.content_pack_compiler import (
    CompiledContentPackWriter,
    SCHEMA_VERSION as CONTENT_PACK_SCHEMA_VERSION,
)
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.content import ContentPackState, IntroducedContentRef
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


def _pack_state(db_path, introduced_refs=None):
    return ContentPackState(
        pack_id="pack",
        introduced_refs=introduced_refs or {},
        metadata={
            "db_path": str(db_path),
            "pack_version": PACK_VERSION,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "schema_version": CONTENT_PACK_SCHEMA_VERSION,
        },
    )


def test_content_manager_prompt_uses_deltas_not_entity_knowledge_dump(tmp_path):
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
    ckpt.session.content_state = {
        "pack": _pack_state(
            db_path,
            introduced_refs={
                "pack::front/old::hash-old": IntroducedContentRef(
                    pack_id="pack",
                    ref_id="front/old",
                    content_hash="hash-old",
                )
            },
        )
    }
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
    candidates = {
        "strahd": {
            "role": "antagonist",
            "location": "castle",
            "known_refs": ["pack:front/everything_he_knows"],
            "notes": "full private knowledge should not be forwarded",
            "source": "/private/module.pdf",
        }
    }

    facts_block = build_recent_canonical_facts_block(ckpt, limit=12)
    candidates_block = build_candidate_turn_entities_block(candidates)
    known_refs_block = build_known_router_refs_block(ckpt)
    messages = build_content_manager_messages(
        ckpt,
        candidate_entities=candidates,
        prompt_mgr=PromptManager("app/prompts"),
        catalog_block=(
            'pack=pack ref=front/strahd kind=front_signal '
            'summary="The antagonist tracks public trouble."'
        ),
        candidate_entities_block=candidates_block,
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

    assert "character=strahd" in candidates_block
    assert "role=antagonist" in candidates_block
    assert "location=castle" in candidates_block
    assert "known_refs" not in candidates_block
    assert "everything_he_knows" not in candidates_block
    assert "full private knowledge" not in candidates_block
    assert "/private" not in candidates_block
    assert ".pdf" not in candidates_block

    assert "pack=pack ref=front/old" in known_refs_block

    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "public fact 13" not in system
    assert "strahd" not in system
    assert "public fact 13" in user
    assert "character=strahd" in user
    assert "everything_he_knows" not in user
    assert "/private" not in user
    assert ".pdf" not in user


def test_plan_content_manager_updates_uses_role_and_validates_router_packet(tmp_path):
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
            content_updates=[
                {
                    "pack_id": "pack",
                    "ref": "front/strahd",
                    "content_hash": "",
                    "update_kind": "introduce",
                    "reason": "f01 makes the front relevant",
                    "source_fact_ids": ["f01"],
                }
            ],
            turn_hints=[
                {
                    "character_id": "strahd",
                    "priority": "high",
                    "reason": "The front may want attention.",
                    "source_fact_ids": ["f01"],
                    "related_content_refs": ["pack:front/strahd"],
                }
            ],
        )
    ))

    output = asyncio.run(
        plan_content_manager_updates(
            ckpt,
            candidate_entities={"strahd": {"role": "antagonist"}},
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
        )
    )

    assert client.complete.await_args.kwargs["role"] == "content_manager"
    assert client.complete.await_args.kwargs["response_model"] is ContentManagerOutput
    assert output.content_updates[0].content_hash == "hash-front"
    assert format_content_manager_router_records(output) == [
        (
            "content_update kind=introduce pack=pack ref=front/strahd "
            "hash=hash-front facts=f01 reason=\"f01 makes the front relevant\""
        ),
        (
            "turn_hint character=strahd priority=high refs=pack:front/strahd "
            "facts=f01 reason=\"The front may want attention.\""
        ),
    ]


def test_content_manager_validation_rejects_missing_ref(tmp_path):
    db_path = _pack_db(tmp_path, [])
    ckpt = checkpoint()
    ckpt.session.content_state = {"pack": _pack_state(db_path)}
    output = ContentManagerOutput(
        content_updates=[
            {
                "pack_id": "pack",
                "ref": "front/missing",
                "content_hash": "",
                "update_kind": "introduce",
                "reason": "",
                "source_fact_ids": ["f01"],
            }
        ]
    )

    with pytest.raises(ContentManagerValidationError, match="ref=front/missing"):
        validate_content_manager_output(
            ckpt,
            output,
            candidate_entities={"strahd": {"role": "antagonist"}},
        )


def test_content_manager_validation_rejects_unknown_turn_hint_character(tmp_path):
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
        turn_hints=[
            {
                "character_id": "rahadin",
                "priority": "high",
                "reason": "",
                "source_fact_ids": [],
                "related_content_refs": ["pack:front/strahd"],
            }
        ]
    )

    with pytest.raises(ContentManagerValidationError, match="unknown character_id"):
        validate_content_manager_output(
            ckpt,
            output,
            candidate_entities={"strahd": {"role": "antagonist"}},
        )


def test_content_manager_validation_rejects_hash_mismatch_and_bad_hint_ref(tmp_path):
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
        content_updates=[
            {
                "pack_id": "pack",
                "ref": "front/strahd",
                "content_hash": "hash-stale",
                "update_kind": "introduce",
                "reason": "source /private/module.pdf",
                "source_fact_ids": [],
            }
        ],
        turn_hints=[
            {
                "character_id": "strahd",
                "priority": "high",
                "reason": "",
                "source_fact_ids": [],
                "related_content_refs": ["pack:front/missing"],
            }
        ],
    )

    assert output.content_updates[0].reason == ""
    with pytest.raises(ContentManagerValidationError) as exc:
        validate_content_manager_output(
            ckpt,
            output,
            candidate_entities={"strahd": {"role": "antagonist"}},
        )

    assert "content hash mismatch" in str(exc.value)
    assert "invalid hint refs" in str(exc.value)
