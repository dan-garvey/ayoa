"""Narrator composition (phase 2) — composes polished prose from events + agent outputs.

The narrator carries a session-wide rolling conversation on the checkpoint
(`checkpoint.narrator_conversation`). Every NP2 call sees every prior turn's
prose it has written, so voice and continuity hold across the whole session.
"""

from __future__ import annotations

import json
import logging
import time

from app.engine.prompt_manager import PromptManager
from app.engine.context_builder import append_turn_to_conversation
from app.engine.turn_loop_contracts import PARTIAL_MODE_MARKER
from app.llm.client import LLMClient
from app.schemas.agents import CharacterAgentOutput
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import EventRouterOutput
from app.schemas.events import CanonicalEvent
from app.schemas.narrator import NarratorFinalOutput
from app.schemas.state import RenderBufferEntry

logger = logging.getLogger(__name__)


class Narrator:
    """Narrator phase 2: composes final prose over a rolling session conversation."""

    def __init__(self, client: LLMClient, prompt_manager: PromptManager):
        self.client = client
        self.prompt_manager = prompt_manager
        # Usage from the most recent compose() call.
        self.last_usage: dict[str, int] = {}

    async def compose(
        self,
        user_input: str,
        event: CanonicalEvent,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
        routed: EventRouterOutput | None = None,
        acting_character_id: str = "",
    ) -> NarratorFinalOutput:
        """Compose final narrative prose and append the turn to narrator_conversation."""
        from app.engine.context_builder import (
            build_player_characters_block,
            resolve_acting_character,
        )

        setting_summary = self._build_setting_summary(checkpoint)
        narrative_rules = checkpoint.config.narrative_rules or "No specific narrative rules."
        scene_context = self._build_scene_context(checkpoint)
        canonical_event = json.dumps(event.model_dump(), indent=2, sort_keys=True)
        formatted_agents = self._format_agent_outputs(agent_outputs, checkpoint, routed)

        acting_id, _, acting_name = resolve_acting_character(
            checkpoint, acting_character_id,
        )
        player_characters_block = build_player_characters_block(checkpoint, acting_id)

        # Opening directive fires only on the very first turn of the story
        # (when the narrator has no prior turns to draw from).
        opening_directive = ""
        if not checkpoint.narrator_conversation and checkpoint.opening_narrative:
            opening_directive = (
                "## Opening Scene Directive\n"
                "This is the OPENING of the story. Use the author's opening "
                "guidance below for TONE, sensory grounding, environmental "
                "framing, and the felt weight of arrival. If the guidance "
                "describes a journey, approach, or movement INTO the starting "
                "scene, render that approach — not in media res with the "
                "player characters already settled.\n"
                "\n"
                "**Do NOT transcribe dialogue from the opening prose.** Any "
                "speech on this turn comes from a character agent in the "
                "Character Responses section, exactly like every other turn. "
                "If the opening prose has an NPC speak but you see no agent "
                "output for them here, they do not speak this turn — the "
                "router places characters and their agents produce dialogue; "
                "your job is to render the result. If the author's opening "
                "described someone arriving or speaking and the router did "
                "not place or call them, treat the scene as quieter than the "
                "author's prose suggests and end at a decision point sooner.\n"
                "\n"
                "Weave in each listed player character's physical presence "
                "(from the system prompt's Player Characters block) so "
                "observers naturally see them as described.\n"
                "\n"
                "End at a natural first decision point for the acting character.\n"
                "\n"
                "## Author's Opening Guidance (tone and framing reference — "
                "NOT a script to transcribe)\n"
                f"{checkpoint.opening_narrative}\n"
            )

        render_t0 = time.monotonic()
        messages = self.prompt_manager.render_conversation(
            "narrator_phase2",
            history=checkpoint.narrator_conversation,
            setting_summary=setting_summary,
            narrative_rules=narrative_rules,
            canonical_event=canonical_event,
            agent_outputs=formatted_agents,
            scene_context=scene_context,
            user_input=user_input,
            acting_character_name=acting_name,
            player_characters_block=player_characters_block,
            opening_directive=opening_directive,
        )
        render_ms = (time.monotonic() - render_t0) * 1000

        user_content = messages[-1]["content"]

        logger.info(
            "Narrator: composing with %d agent outputs (history=%d msgs)",
            len(agent_outputs),
            len(checkpoint.narrator_conversation),
        )

        response = await self.client.complete(
            role="narrator",
            messages=messages,
            response_model=NarratorFinalOutput,
            temperature=0.5,
            max_tokens=4000,
            cache=True,
            compact=True,
        )
        result: NarratorFinalOutput = response.parsed
        self.last_usage = {**response.usage, "prompt_render_ms": render_ms}

        append_turn_to_conversation(
            checkpoint.narrator_conversation, user_content, response,
        )

        logger.info("Narrator result: %d chars", len(result.final_text))

        return result

    def _format_agent_outputs(
        self,
        agent_outputs: list[CharacterAgentOutput],
        checkpoint: CheckpointFile,
        routed: EventRouterOutput | None = None,
    ) -> str:
        """Format agent outputs for the NP2 prompt."""
        if not agent_outputs:
            return "No characters responded to this event."

        # Router emits single-char codes (d/i/f); expand to human-readable
        # words for the narrator's rule-5 language.
        _level_names = {"d": "direct", "i": "indirect", "f": "inferred"}
        obs_levels: dict[str, str] = {}
        if routed:
            for obs in routed.observers:
                obs_levels[obs.character_id] = _level_names.get(
                    obs.observation_level, obs.observation_level
                )

        from app.engine.context_builder import iter_agent_beats

        known = set(checkpoint.world_state.known_characters)
        sections = []
        for output, char in iter_agent_beats(agent_outputs, checkpoint):
            # Use name only if player has met this character; otherwise describe them
            if output.character_id in known:
                label = char.name if char else output.character_id
            elif char and char.public_sheet.appearance:
                label = char.public_sheet.appearance
            elif char and char.public_sheet.role:
                label = char.public_sheet.role
            else:
                label = output.character_id

            parts = [f"### {label}"]
            obs_level = obs_levels.get(output.character_id, "direct")
            parts.append(f"[Observation: {obs_level}]")

            if output.public_response.actions:
                parts.append("Actions:")
                for action in output.public_response.actions:
                    parts.append(f"  - {action}")

            if output.public_response.dialogue:
                parts.append("Dialogue:")
                for line in output.public_response.dialogue:
                    parts.append(f'  - "{line}"')

            if output.public_response.expression:
                parts.append(f"Expression: {output.public_response.expression}")

            sections.append("\n".join(parts))

        return "\n\n".join(sections)

    def _build_setting_summary(self, checkpoint: CheckpointFile) -> str:
        setting = checkpoint.world_state.setting
        parts = []
        if setting.genre:
            parts.append(f"Genre: {setting.genre}")
        if setting.era:
            parts.append(f"Era: {setting.era}")
        if setting.tone:
            parts.append(f"Tone: {setting.tone}")
        if setting.premise:
            parts.append(f"Premise: {setting.premise}")
        return "\n".join(parts) if parts else "No setting information available."

    def _build_scene_context(self, checkpoint: CheckpointFile) -> str:
        locations = checkpoint.world_state.locations
        scene_id = locations.current_scene_id
        if not scene_id:
            return "No scene information available."

        scene = locations.scene_graph.get(scene_id, {})
        if not scene:
            return f"Current location: {scene_id}"

        name = scene.get("name", scene_id)
        desc = scene.get("description", "")
        connected = scene.get("connected_to", [])

        parts = [f"Location: {name}"]
        if desc:
            parts.append(f"Description: {desc}")
        if connected:
            connections = []
            for conn_id in connected:
                conn_scene = locations.scene_graph.get(conn_id, {})
                conn_name = conn_scene.get("name", conn_id)
                connections.append(conn_name)
            parts.append(f"Connected to: {', '.join(connections)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# v11 per-POV narrator entry point
# ---------------------------------------------------------------------------
#
# `compose_pov_render` is the v11 replacement for the session-wide
# `Narrator.compose` flow above. The legacy `compose` is kept as a
# fallback during the transition; new callers (the turn_loop dispatcher)
# should use `compose_pov_render`.


def _build_setting_summary(checkpoint: CheckpointFile) -> str:
    # TODO(v11): duplicated from Narrator._build_setting_summary above —
    # factor into context_builder once the legacy compose is removed.
    setting = checkpoint.world_state.setting
    parts = []
    if setting.genre:
        parts.append(f"Genre: {setting.genre}")
    if setting.era:
        parts.append(f"Era: {setting.era}")
    if setting.tone:
        parts.append(f"Tone: {setting.tone}")
    if setting.premise:
        parts.append(f"Premise: {setting.premise}")
    return "\n".join(parts) if parts else "No setting information available."


def _build_scene_context(checkpoint: CheckpointFile) -> str:
    # TODO(v11): duplicated from Narrator._build_scene_context above —
    # factor into context_builder once the legacy compose is removed.
    locations = checkpoint.world_state.locations
    scene_id = locations.current_scene_id
    if not scene_id:
        return "No scene information available."
    scene = locations.scene_graph.get(scene_id, {})
    if not scene:
        return f"Current location: {scene_id}"
    name = scene.get("name", scene_id)
    desc = scene.get("description", "")
    connected = scene.get("connected_to", [])
    parts = [f"Location: {name}"]
    if desc:
        parts.append(f"Description: {desc}")
    if connected:
        connections = []
        for conn_id in connected:
            conn_scene = locations.scene_graph.get(conn_id, {})
            conn_name = conn_scene.get("name", conn_id)
            connections.append(conn_name)
        parts.append(f"Connected to: {', '.join(connections)}")
    return "\n".join(parts)


def _resolve_buffered_events(
    ckpt: CheckpointFile,
    buffered_events: list[RenderBufferEntry],
) -> list[tuple[RenderBufferEntry, EventRouterOutput]]:
    """Walk the render buffer and resolve each entry against
    `ckpt.canonical_events`. Missing event_ids are warned and skipped —
    a stale buffer entry must not abort the render.
    """
    by_id: dict[str, EventRouterOutput] = {
        ev.event_id: ev for ev in ckpt.canonical_events
    }
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]] = []
    for entry in buffered_events:
        ev = by_id.get(entry.event_id)
        if ev is None:
            logger.warning(
                "compose_pov_render: buffered event_id %r not found in "
                "canonical_events; skipping",
                entry.event_id,
            )
            continue
        resolved.append((entry, ev))
    return resolved


_OBS_LEVEL_NAMES = {
    "d": "direct",
    "i": "indirect",
    "f": "inferred",
    "direct": "direct",
    "indirect": "indirect",
    "inferred": "inferred",
}


def _format_canonical_events_block(
    resolved: list[tuple[RenderBufferEntry, EventRouterOutput]],
) -> str:
    """Serialize the resolved events into a prose block the narrator
    can read. One section per event with its observation level tag."""
    if not resolved:
        return "No canonical events to render."
    sections: list[str] = []
    for idx, (entry, ev) in enumerate(resolved, start=1):
        obs = _OBS_LEVEL_NAMES.get(entry.observation_level, entry.observation_level)
        ca = ev.canonical_event
        lines = [
            f"## Event {idx}: {ev.event_id} [Observation: {obs}]",
            f"attempted_action: {ca.world_adjudication.attempted_action}",
            f"resolved_outcome: {ca.world_adjudication.resolved_outcome}",
        ]
        if ca.observable_facts:
            lines.append("observable_facts:")
            for fact in ca.observable_facts:
                lines.append(f"- {fact}")
        else:
            lines.append("observable_facts: (none)")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


async def compose_pov_render(
    client: LLMClient,
    prompt_mgr: PromptManager,
    ckpt: CheckpointFile,
    pov_character_id: str,
    buffered_events: list[RenderBufferEntry],
    partial_mode: bool,
) -> str:
    """v11 per-POV narrator entry point.

    Renders the beat from `pov_character_id`'s point of view in
    second-person present tense, using their per-character rolling
    conversation history (`ckpt.narrator_conversations[pov_character_id]`).

    Events are resolved by `event_id` against `ckpt.canonical_events`;
    observation levels on the buffer entries tag how each event is
    framed (direct / indirect / inferred).

    When `partial_mode=True`, the PARTIAL_MODE_MARKER is prepended to
    the user message so the narrator prompt's rule-15 PARTIAL mode
    fires; the rendered passage then ends mid-attempt to prompt the
    pinned responder's /act.

    Returns the final prose string. Appends the exchange into
    `ckpt.narrator_conversations[pov_character_id]` in-place — the
    caller is responsible for saving the checkpoint.
    """
    from app.engine.context_builder import build_player_characters_block

    resolved = _resolve_buffered_events(ckpt, buffered_events)

    # POV character identity. Fall back to the raw id if the roster
    # doesn't know them (pristine tests, legacy checkpoints).
    pov_char = next(
        (c for c in ckpt.characters if c.character_id == pov_character_id),
        None,
    )
    acting_name = pov_char.name if pov_char else pov_character_id

    setting_summary = _build_setting_summary(ckpt)
    narrative_rules = ckpt.config.narrative_rules or "No specific narrative rules."
    scene_context = _build_scene_context(ckpt)
    player_characters_block = build_player_characters_block(ckpt, pov_character_id)
    canonical_event_block = _format_canonical_events_block(resolved)

    # v11 per-POV callers don't accumulate agent outputs the way the
    # legacy compose did — the canonical events themselves carry the
    # resolved outcome. The template still requires the variable, so
    # render a neutral placeholder.
    agent_outputs_block = (
        "Character responses are folded into each event's "
        "`resolved_outcome` above."
    )

    # Opening directive only fires on the very first render for this
    # POV character (no prior per-character narrator history).
    opening_directive = ""
    pov_history = ckpt.narrator_conversations.setdefault(pov_character_id, [])
    if not pov_history and ckpt.opening_narrative:
        first_is_begin = bool(resolved) and (
            resolved[0][1].canonical_event.world_adjudication.attempted_action.strip()
            .lower() == "(begin)"
        )
        if first_is_begin:
            opening_directive = (
                "## Opening Scene Directive\n"
                f"{ckpt.opening_narrative}\n\n"
            )

    # v11 triggering user-input for this POV render. We don't have the
    # initiator's full intention here; leave empty so the template
    # simply shows "{name} — " with no tail.
    user_input = ""

    render_t0 = time.monotonic()
    messages = prompt_mgr.render_conversation(
        "narrator_phase2",
        history=pov_history,
        setting_summary=setting_summary,
        narrative_rules=narrative_rules,
        canonical_event=canonical_event_block,
        agent_outputs=agent_outputs_block,
        scene_context=scene_context,
        user_input=user_input,
        acting_character_name=acting_name,
        player_characters_block=player_characters_block,
        opening_directive=opening_directive,
    )
    render_ms = (time.monotonic() - render_t0) * 1000

    # Prepend the PARTIAL marker to the per-turn user message body so
    # rule-15 PARTIAL mode fires in the prompt.
    user_content = messages[-1]["content"]
    if partial_mode:
        user_content = f"{PARTIAL_MODE_MARKER}\n\n{user_content}"
        messages[-1] = {"role": "user", "content": user_content}

    logger.info(
        "compose_pov_render: pov=%s events=%d partial=%s history=%d msgs "
        "(prompt_render_ms=%.1f)",
        pov_character_id, len(resolved), partial_mode, len(pov_history),
        render_ms,
    )

    response = await client.complete(
        role="narrator",
        messages=messages,
        response_model=NarratorFinalOutput,
        temperature=0.5,
        max_tokens=4000,
        cache=True,
        compact=True,
    )
    result: NarratorFinalOutput = response.parsed

    # Append user + assistant to the POV's rolling history. We can't
    # reuse `append_turn_to_conversation` here because the user content
    # we want stored is the marker-prepended string (when partial);
    # the helper would pick up `user_content` from the messages list
    # anyway, so just use it directly for clarity.
    append_turn_to_conversation(pov_history, user_content, response)

    final_text = result.final_text if result is not None else ""
    logger.info(
        "compose_pov_render: pov=%s rendered %d chars",
        pov_character_id, len(final_text),
    )
    return final_text
