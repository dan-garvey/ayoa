from __future__ import annotations

import json

import pytest

from app.engine.content_lookup import build_router_content_lookup_catalog_block
from app.engine.content_pack_compiler import (
    CompiledContentPackWriter,
)
from app.engine.content_pack_projections import (
    ContentProjectionBuildError,
    build_content_pack_projection_artifact,
    content_pack_state_from_projection,
)
from app.schemas.content_pack import (
    AgentContextSliceRecord,
    ActorDossierRecord,
    CompiledContentCard,
    ContentPackDomainCatalog,
    ContentSectionRecord,
    FrontDossierRecord,
    KnowledgeGraphEdgeRecord,
    LocationRecord,
)
from app.schemas.content_privacy import PRIVATE_RUNTIME_METADATA_CONTEXT
from tests.support.factories import checkpoint


PACK_ID = "pack"
PACK_VERSION = "1.0.0"
SOURCE_FINGERPRINT = "sha256:test-source"


def test_projection_builder_emits_import_owned_runtime_slices():
    catalog = _catalog()
    projection = build_content_pack_projection_artifact(
        catalog,
        runtime_cards=_cards(catalog),
        initial_router_lookup_refs=[
            "sec.startup",
            "loc.entry",
            "actor.villain",
        ],
        field_start_router_lookup_refs=["loc.route"],
        active_front_refs=["front.clock"],
        active_character_ids=["villain"],
        intentions_enabled_character_ids=["villain"],
        character_overrides={"villain": {"name": "The Villain"}},
        checkpoint={
            "player_primer": "Open at the patron's table.",
            "world_facts": ["The expedition has not left yet."],
            "narrative_rules": "Start in negotiation.",
            "world_lore": "The route becomes concrete when discovered.",
        },
        field_start={
            "router_lookup_refs": ["loc.route"],
            "location_ref": "loc.route",
            "active_character_ids": ["pc", "villain"],
            "knowledge_grants": [
                {
                    "entity_id": "villain",
                    "known_refs": ["loc.route"],
                    "notes": "Field-start grant.",
                }
            ],
        },
    )

    assert projection.projection_hash.startswith("sha256:")
    assert [ref.ref for ref in projection.router.initial_lookup_refs] == [
        "sec.startup",
        "loc.entry",
        "actor.villain",
    ]
    assert len(projection.router.lookup_catalog) == len(catalog._domain_records())
    assert [key.key for key in projection.router.knowledge_keys] == [
        "pack.startup",
        "pack.field_start",
    ]

    character = projection.characters[0]
    assert character.name == "The Villain"
    assert character.status == "active"
    assert character.intentions_enabled is True
    assert "pack:actor.villain@hash-actor-villain" in character.known_refs
    assert "pack:actor.scout@hash-actor-scout" in character.known_refs
    assert "pack:loc.route@hash-loc-route" in character.known_refs
    assert "pack:kg.villain.route@hash-kg-route" in character.known_refs
    assert "commands: Scout supplies current reports." in character.known_context
    assert "Useful assets include: informants." in character.known_context
    assert "Route" not in character.known_context
    assert "secret route" in character.secrets[1]
    assert "agent_context" not in character.known_context
    assert "actor.scout" not in character.known_context
    assert "hash-" not in character.known_context

    state = content_pack_state_from_projection(
        projection,
        db_path="private_extractions/compiled/pack.sqlite",
        start_mode="field",
    )[PACK_ID]
    assert "field_01" in state.pending_signals
    assert state.pending_signals["field_01"].ref_id == "loc.route"
    assert "pack:loc.route@hash-loc-route" in state.knowledge_map["villain"].known_refs

    private_dump = state.model_dump(
        mode="json",
        context={PRIVATE_RUNTIME_METADATA_CONTEXT: True},
    )
    assert "domain_catalog" in private_dump["metadata"]
    assert "router_lookup_catalog" in private_dump["metadata"]
    assert "router_knowledge_index" in private_dump["metadata"]
    assert "router_knowledge_packets" in private_dump["metadata"]
    assert "content_hash" not in json.dumps(
        private_dump["metadata"]["router_lookup_catalog"],
        sort_keys=True,
    )
    assert "hash-loc-route" in json.dumps(
        private_dump["metadata"]["router_knowledge_packets"],
        sort_keys=True,
    )
    assert "content_hash" not in json.dumps(
        private_dump["metadata"]["router_knowledge_index"],
        sort_keys=True,
    )

    public_dump = json.dumps(state.model_dump(mode="json"), sort_keys=True)
    assert "domain_catalog" not in public_dump
    assert "router_lookup_catalog" not in public_dump
    assert "router_knowledge_index" not in public_dump
    assert "router_knowledge_packets" not in public_dump
    assert "private_extractions" not in public_dump


def test_projection_builder_rejects_missing_runtime_refs():
    catalog = _catalog()

    with pytest.raises(ContentProjectionBuildError, match="missing runtime card"):
        build_content_pack_projection_artifact(
            catalog,
            runtime_cards=_cards(catalog),
            initial_router_lookup_refs=["loc.missing"],
        )


def test_projection_builder_rejects_blocked_or_unreviewed_records():
    blocked_catalog = _catalog(
        actor_updates={"gate_status": "blocked"},
    )
    with pytest.raises(ContentProjectionBuildError, match="not runtime_ready"):
        build_content_pack_projection_artifact(
            blocked_catalog,
            runtime_cards=_cards(blocked_catalog),
            initial_router_lookup_refs=["actor.villain"],
        )

    unreviewed_catalog = _catalog(
        actor_updates={"review_status": "needs_review", "gate_status": "flagged"},
    )
    with pytest.raises(ContentProjectionBuildError, match="not reviewed"):
        build_content_pack_projection_artifact(
            unreviewed_catalog,
            runtime_cards=_cards(unreviewed_catalog),
            initial_router_lookup_refs=["actor.villain"],
        )


def test_projection_builder_accepts_authored_router_knowledge_keys():
    catalog = _catalog()
    projection = build_content_pack_projection_artifact(
        catalog,
        runtime_cards=_cards(catalog),
        initial_router_lookup_refs=["loc.entry"],
        router_knowledge_keys=[
            {
                "key": "pack.route.entry",
                "kind": "route",
                "label": "Entry route context",
                "summary": "A compact selector for entry-route knowledge.",
                "scope_facets": {
                    "location_refs": ["loc.entry"],
                    "actor_refs": ["actor.villain"],
                    "character_ids": ["villain"],
                },
                "activation_hints": ["entry", "route"],
                "priority": 25,
                "packet_refs": ["loc.entry", "front.clock"],
            }
        ],
    )

    key = projection.router.knowledge_keys[0]
    assert key.key == "pack.route.entry"
    assert [ref.ref for ref in key.packet_refs] == ["loc.entry", "front.clock"]
    assert key.prompt_index_entry()["packet_count"] == 2
    assert "content_hash" not in json.dumps(key.prompt_index_entry())
    assert "hash-loc-entry" in json.dumps(key.packet_metadata_entry())


def test_router_lookup_catalog_prefers_import_projection_packet(tmp_path):
    db_path = tmp_path / "pack.sqlite"
    CompiledContentPackWriter(
        db_path,
        pack_id=PACK_ID,
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
    ).write_pack(
        pages=[],
        cards=[
            _card("loc.entry", "hash-loc-entry", summary="DB entry summary."),
            _card("loc.db_only", "hash-db-only", summary="DB-only summary."),
        ],
    )
    ckpt = checkpoint()
    ckpt.session.content_state = {
        PACK_ID: content_pack_state_from_projection(
            build_content_pack_projection_artifact(
                _catalog(),
                runtime_cards=_cards(_catalog()),
                initial_router_lookup_refs=["loc.entry"],
            ),
            db_path=str(db_path),
        )[PACK_ID]
    }
    state = ckpt.session.content_state[PACK_ID]
    state.metadata["router_lookup_catalog"] = [
        {
            "ref": "loc.entry",
            "kind": "location_card",
            "visibility": "router_hidden",
            "summary": "Projected entry summary.",
            "content_hash": "hash-should-not-render",
        }
    ]

    block = build_router_content_lookup_catalog_block(ckpt)

    assert "Projected entry summary." in block
    assert "DB-only summary." not in block
    assert "hash-should-not-render" not in block


def _catalog(
    *,
    actor_updates: dict[str, object] | None = None,
) -> ContentPackDomainCatalog:
    actor_data = {
        "ref": "actor.villain",
        "content_hash": "hash-actor-villain",
        "title": "Villain",
        "summary": "A reviewed antagonist.",
        "review_status": "approved",
        "gate_status": "runtime_ready",
        "spoiler_class": "none",
        "actor_kind": "villain",
        "character_id_hint": "villain",
        "front_refs": ["front.clock"],
        "home_location_refs": ["loc.entry"],
        "agent_context_slice_ref": "agent_context.villain.startup",
        "goals": ["Recover the map"],
        "constraints": ["Do not overrun the startup scene"],
        "resources": ["informants"],
        "knowledge_channel_refs": ["kg.villain.route"],
        "relationship_edges": [
            {
                "target_ref": "actor.scout",
                "stance": "commands",
                "summary": "Scout supplies current reports.",
                "public": True,
            },
            {
                "target_ref": "loc.route",
                "stance": "secret route",
                "summary": "Knows the route cache matters more than admitted.",
                "public": False,
            },
        ],
    }
    actor_data.update(actor_updates or {})
    return ContentPackDomainCatalog(
        pack_id=PACK_ID,
        pack_version=PACK_VERSION,
        source_fingerprint=SOURCE_FINGERPRINT,
        build_hash="sha256:catalog-build",
        sections=[
            _section("sec.startup", "hash-sec-startup", "Startup", "Startup context."),
        ],
        locations=[
            _location("loc.entry", "hash-loc-entry", "Entry", "Entry context."),
            _location("loc.route", "hash-loc-route", "Route", "Route context."),
        ],
        front_dossiers=[
            _front("front.clock", "hash-front-clock", "Clock", "Front context."),
        ],
        actor_dossiers=[
            ActorDossierRecord(**actor_data),
            ActorDossierRecord(
                ref="actor.scout",
                content_hash="hash-actor-scout",
                title="Scout",
                summary="A reviewed reporting scout.",
                review_status="approved",
                gate_status="runtime_ready",
                spoiler_class="none",
                actor_kind="npc",
                character_id_hint="scout",
            ),
        ],
        agent_context_slices=[
            AgentContextSliceRecord(
                ref="agent_context.villain.startup",
                content_hash="hash-agent-context",
                title="Villain Startup Context",
                summary="Reviewed character seed.",
                review_status="approved",
                gate_status="runtime_ready",
                spoiler_class="none",
                actor_ref="actor.villain",
                known_context="Knows the map matters.",
                private_state="Wants reports before direct action.",
                current_agenda=["Watch the entry"],
                beliefs=["The route can still be recovered"],
                uncertainties=["Who has the map"],
                hard_boundaries=["Do not act beyond reviewed locations"],
                local_context_refs=["loc.entry"],
                graph_edge_refs=["kg.villain.route"],
            )
        ],
        knowledge_graph_edges=[
            KnowledgeGraphEdgeRecord(
                ref="kg.villain.route",
                content_hash="hash-kg-route",
                title="Villain Route Knowledge",
                summary="The villain can learn route disturbances.",
                review_status="approved",
                gate_status="runtime_ready",
                spoiler_class="none",
                from_ref="actor.villain",
                relation="can_dispatch",
                to_ref="loc.route",
                object_kind="location",
            )
        ],
    )


def _section(ref: str, content_hash: str, title: str, summary: str):
    return ContentSectionRecord(
        ref=ref,
        content_hash=content_hash,
        title=title,
        summary=summary,
        review_status="approved",
        gate_status="runtime_ready",
        spoiler_class="none",
    )


def _location(ref: str, content_hash: str, title: str, summary: str):
    return LocationRecord(
        ref=ref,
        content_hash=content_hash,
        title=title,
        summary=summary,
        review_status="approved",
        gate_status="runtime_ready",
        spoiler_class="none",
    )


def _front(ref: str, content_hash: str, title: str, summary: str):
    return FrontDossierRecord(
        ref=ref,
        content_hash=content_hash,
        title=title,
        summary=summary,
        review_status="approved",
        gate_status="runtime_ready",
        spoiler_class="none",
    )


def _cards(catalog: ContentPackDomainCatalog) -> list[CompiledContentCard]:
    return [
        _card(
            record.ref,
            record.content_hash,
            kind=(
                "location_card"
                if record.record_kind == "location"
                else "front_signal"
                if record.record_kind == "front_dossier"
                else record.record_kind
            ),
            title=record.title,
            summary=record.summary,
        )
        for record in catalog._domain_records()
    ]


def _card(
    ref: str,
    content_hash: str,
    *,
    kind: str = "content",
    title: str = "",
    summary: str = "",
) -> CompiledContentCard:
    return CompiledContentCard(
        pack_id=PACK_ID,
        ref=ref,
        content_hash=content_hash,
        card_kind=kind,
        visibility="router_hidden",
        title=title or ref,
        summary=summary or f"Summary for {ref}.",
        review_status="approved",
        gate_status="runtime_ready",
        confidence=1.0,
    )
