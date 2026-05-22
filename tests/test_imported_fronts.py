from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.content_resolver import append_pending_router_content_records
from app.engine.imported_fronts import (
    ImportedFrontDossierCatalog,
    ImportedFrontValidationError,
    front_dossier_router_payload,
    queue_front_dossier_signal,
)
from app.engine.narrator import compose_pov_render
from app.engine.prompt_manager import PromptManager
from app.llm.client import LLMClient
from app.schemas.content import ContentPackState
from app.schemas.conversation import ConversationMessage
from app.schemas.content_pack import FrontDossierRecord
from app.schemas.events import ObservableFact
from app.schemas.state import RenderBufferEntry
from tests.support.factories import (
    character_record,
    checkpoint,
    narrator_llm_response,
    router_output,
)


def test_front_dossier_projects_to_router_only_action_palette_signal() -> None:
    record = _front()

    payload = front_dossier_router_payload(record, pack_id="curse")

    assert payload["kind"] == "front_signal"
    assert payload["ref"] == "front.strahd"
    assert payload["actor"] == "npc.strahd"
    assert payload["villains"] == ["npc.strahd"]
    assert payload["knows"] == ["The party insulted a public servant"]
    assert payload["goals"] == ["Isolate Ireena"]
    assert payload["resources"] == ["spy network"]
    assert payload["minions"] == ["npc.rahadin"]
    assert payload["actions"] == [
        {
            "id": "send_spies",
            "kind": "spy",
            "priority": "5",
            "trigger": "A public insult reaches the castle.",
            "cooldown": "once per night",
            "target_scope": "party location",
            "summary": "Send spies to observe the party.",
            "resources": ["spy network"],
            "minions": ["npc.rahadin"],
            "restraints": ["avoid direct violence before dinner"],
            "consequences": ["consequence.spies"],
            "encounters": ["enc.spy_probe"],
            "statblocks": ["stat.spy"],
        }
    ]
    assert "provenance" not in payload
    assert "field_provenance" not in payload


def test_front_dossier_signal_drains_to_router_history_without_default_leakage() -> None:
    ckpt = checkpoint()
    pack_state = ContentPackState(pack_id="curse")
    ckpt.session.content_state = {"curse": pack_state}

    signal = queue_front_dossier_signal(
        pack_state,
        _front(),
        pack_id="curse",
        reason="Reviewed front dossier is in scope.",
        priority=7,
    )
    records = append_pending_router_content_records(ckpt)

    assert signal.priority == 7
    assert len(records) == 1
    record = records[0]
    assert record.startswith(
        "front_signal ref=front.strahd actor=npc.strahd villains=npc.strahd"
    )
    assert 'knows=["The party insulted a public servant"]' in record
    assert "visibility=hidden" in record
    assert "hash=hash-front-strahd" in record
    assert "pack=curse" in record
    assert 'goals=["Isolate Ireena"]' in record
    assert 'constraints=["must preserve plausible deniability"]' in record
    assert 'knowledge_channels=["spies in taverns"]' in record
    assert 'resources=["spy network"]' in record
    assert "minions=npc.rahadin" in record
    assert 'escalation_thresholds=["party publicly shelters Ireena"]' in record
    assert 'cooldowns=["one major pressure per night"]' in record
    assert 'restraints=["avoid direct violence before dinner"]' in record
    assert '"id":"send_spies"' in record
    assert '"resources":["spy network"]' in record
    assert '"minions":["npc.rahadin"]' in record
    assert '"encounters":["enc.spy_probe"]' in record
    default_dump = json.dumps(signal.model_dump(mode="json"), sort_keys=True)
    assert "spy network" not in default_dump
    assert "send_spies" not in default_dump
    assert "avoid direct violence before dinner" not in default_dump


def test_front_dossier_pending_signal_default_dump_excludes_private_dossier_text() -> None:
    private_dossier_text = "PRIVATE DOSSIER: Strahd intends the dinner ambush."
    hidden_plan = "HIDDEN PLAN: abduct Ireena before dawn."
    ckpt = checkpoint()
    pack_state = ContentPackState(pack_id="curse")
    ckpt.session.content_state = {"curse": pack_state}
    front = _front()
    front.summary = private_dossier_text
    front.goals = [hidden_plan]

    signal = queue_front_dossier_signal(
        pack_state,
        front,
        pack_id="curse",
    )

    private_dump = json.dumps(
        signal.model_dump(
            mode="json",
            context={"include_private_runtime_metadata": True},
        ),
        sort_keys=True,
    )
    assert private_dossier_text in front.model_dump_json()
    assert private_dossier_text not in private_dump
    assert hidden_plan in private_dump

    default_dump = json.dumps(ckpt.model_dump(mode="json"), sort_keys=True)
    assert "Reviewed front dossier is ready for router use." in default_dump
    assert private_dossier_text not in default_dump
    assert hidden_plan not in default_dump
    assert "send_spies" not in default_dump
    assert "spy network" not in default_dump

    records = append_pending_router_content_records(ckpt)

    assert len(records) == 1
    router_record = records[0]
    assert router_record.startswith(
        "front_signal ref=front.strahd actor=npc.strahd"
    )
    assert hidden_plan in router_record
    assert '"summary":"Send spies to observe the party."' in router_record
    assert private_dossier_text not in router_record


def test_front_signal_router_history_does_not_reach_narrator_prompt() -> None:
    hidden_plan = "HIDDEN PLAN: send Rahadin through the servant stairs."
    ckpt = checkpoint(
        bindings={"alice": "1"},
        player_character_id="alice",
        characters=[
            character_record(
                "alice",
                name="Alice",
                role="player",
                is_playable=True,
            )
        ],
    )
    router_record = (
        'front_signal ref=front.strahd actor=npc.strahd '
        f'pressure="{hidden_plan}" visibility=hidden hash=hash-front pack=curse'
    )
    ckpt.session_conversation.append(
        ConversationMessage(role="assistant", content=router_record)
    )
    ckpt.canonical_events.append(
        router_output(
            event_id="evt_public_wolves",
            observer_ids=["alice"],
            facts=[
                ObservableFact.all(
                    "Wolves are seen pacing the ridge above the road."
                )
            ],
        )
    )
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=narrator_llm_response("RENDERED"))

    asyncio.run(
        compose_pov_render(
            client=client,
            prompt_mgr=PromptManager("app/prompts"),
            ckpt=ckpt,
            pov_character_id="alice",
            buffered_events=[
                RenderBufferEntry(
                    event_id="evt_public_wolves",
                    observation_level="direct",
                )
            ],
            partial_mode=False,
        )
    )

    messages = client.complete.call_args.kwargs["messages"]
    flat = "\n".join(
        message["content"]
        for message in messages
        if isinstance(message.get("content"), str)
    )
    assert "Wolves are seen pacing the ridge above the road." in flat
    assert "front_signal ref=front.strahd" not in flat
    assert hidden_plan not in flat


def test_front_catalog_rejects_unreviewed_or_blocked_dossiers() -> None:
    with pytest.raises(ImportedFrontValidationError, match="not reviewed"):
        ImportedFrontDossierCatalog(
            [
                {
                    **_front().model_dump(mode="json"),
                    "review_status": "needs_review",
                    "gate_status": "flagged",
                }
            ]
        )

    with pytest.raises(ImportedFrontValidationError, match="not runtime-ready"):
        ImportedFrontDossierCatalog(
            [
                {
                    **_front().model_dump(mode="json"),
                    "gate_status": "blocked",
                    "gate_reasons": ["blocked_section"],
                }
            ]
        )


def _front() -> FrontDossierRecord:
    return FrontDossierRecord(
        ref="front.strahd",
        content_hash="hash-front-strahd",
        title="Strahd Front",
        summary="Reviewed Strahd pressure dossier.",
        review_status="approved",
        gate_status="runtime_ready",
        villain_refs=["npc.strahd"],
        goals=["Isolate Ireena"],
        constraints=["must preserve plausible deniability"],
        knowledge_channels=["spies in taverns"],
        resources=["spy network"],
        minion_refs=["npc.rahadin"],
        initial_knowledge=["The party insulted a public servant"],
        escalation_thresholds=["party publicly shelters Ireena"],
        cooldowns=["one major pressure per night"],
        restraints=["avoid direct violence before dinner"],
        action_palette=[
            {
                "action_id": "send_spies",
                "action_kind": "spy",
                "priority": 5,
                "trigger": "A public insult reaches the castle.",
                "cooldown": "once per night",
                "target_scope": "party location",
                "summary": "Send spies to observe the party.",
                "resource_refs": ["spy network"],
                "minion_refs": ["npc.rahadin"],
                "restraints": ["avoid direct violence before dinner"],
                "consequence_refs": ["consequence.spies"],
                "encounter_template_refs": ["enc.spy_probe"],
                "statblock_refs": ["stat.spy"],
            }
        ],
    )
