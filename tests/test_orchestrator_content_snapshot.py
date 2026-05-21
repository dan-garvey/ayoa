from __future__ import annotations

from app.engine.orchestrator import (
    _automated_turn_snapshot,
    _rollback_automated_turn_snapshot,
)
from app.schemas.content import (
    ContentPackState,
    IntroducedContentRef,
    PendingContentSignal,
)
from app.schemas.conversation import ConversationMessage
from tests.support.factories import checkpoint


def test_automated_turn_rollback_restores_content_state_and_engine_updates():
    ckpt = checkpoint()
    ckpt.session.pending_engine_state_updates = ["before"]
    ckpt.session.content_state = {
        "pack": ContentPackState(
            pack_id="pack",
            pending_signals={
                "sig-1": PendingContentSignal(
                    signal_id="sig-1",
                    pack_id="pack",
                    ref_id="room/entry",
                    content_hash="hash-1",
                )
            },
        )
    }

    snapshot = _automated_turn_snapshot(ckpt)

    ckpt.session.pending_engine_state_updates.append("after")
    ckpt.session.content_state["pack"].pending_signals.clear()
    ckpt.session.content_state["pack"].introduced_refs[
        "pack::room/entry::hash-1"
    ] = IntroducedContentRef(
        pack_id="pack",
        ref_id="room/entry",
        content_hash="hash-1",
    )
    ckpt.session_conversation.append(
        ConversationMessage(role="assistant", content="content_known ref=room/entry")
    )

    _rollback_automated_turn_snapshot(ckpt, snapshot)

    assert ckpt.session.pending_engine_state_updates == ["before"]
    restored = ckpt.session.content_state["pack"]
    assert sorted(restored.pending_signals) == ["sig-1"]
    assert restored.introduced_refs == {}
    assert ckpt.session_conversation == []
