"""Tests for router-created scenes (F3.2).

Exercises the pure-state mutation in orchestrator._apply_scene_creations
— no LLM in the loop."""

from __future__ import annotations

import logging

import pytest

from app.engine.orchestrator import _apply_scene_creations
from app.schemas.checkpoint import CheckpointFile
from app.schemas.event_router import SceneCreation
from app.schemas.state import LocationState, SessionState, WorldState


def _ckpt(graph: dict | None = None) -> CheckpointFile:
    return CheckpointFile(
        session=SessionState(session_id="test"),
        world_state=WorldState(
            locations=LocationState(
                scene_graph=graph if graph is not None else {
                    "hall": {
                        "name": "Great Hall",
                        "description": "A long stone hall.",
                        "connected_to": [],
                        "properties": {},
                    },
                },
            ),
        ),
    )


class TestSceneCreationHappyPath:
    def test_adds_new_scene_to_graph(self):
        ckpt = _ckpt()
        _apply_scene_creations(ckpt, [
            SceneCreation(
                scene_id="garden",
                name="Garden",
                description="Roses in winter bloom.",
                connected_to=["hall"],
            ),
        ])
        g = ckpt.world_state.locations.scene_graph
        assert "garden" in g
        assert g["garden"]["name"] == "Garden"
        assert g["garden"]["description"] == "Roses in winter bloom."
        assert g["garden"]["connected_to"] == ["hall"]

    def test_auto_adds_reverse_edge(self):
        ckpt = _ckpt()
        _apply_scene_creations(ckpt, [
            SceneCreation(scene_id="garden", name="Garden", description="", connected_to=["hall"]),
        ])
        # The hall should now list garden in its connected_to.
        assert ckpt.world_state.locations.scene_graph["hall"]["connected_to"] == ["garden"]

    def test_batch_internal_connections_work_both_orders(self):
        """If A and B are both created in the same batch with A→B, the
        reverse edge B→A must also land regardless of declaration order."""
        ckpt = _ckpt()
        _apply_scene_creations(ckpt, [
            SceneCreation(scene_id="a", name="A", description="", connected_to=["b"]),
            SceneCreation(scene_id="b", name="B", description="", connected_to=[]),
        ])
        g = ckpt.world_state.locations.scene_graph
        assert "a" in g and "b" in g
        # A declared the edge, B didn't — reverse edge still added to B.
        assert "b" in g["a"]["connected_to"]
        assert "a" in g["b"]["connected_to"]

    def test_no_duplicate_reverse_edge(self):
        """If the graph already lists the new scene on a neighbor, don't
        append a duplicate."""
        ckpt = _ckpt()
        # Seed hall with garden already in its connected_to.
        ckpt.world_state.locations.scene_graph["hall"]["connected_to"] = ["garden"]
        _apply_scene_creations(ckpt, [
            SceneCreation(scene_id="garden", name="Garden", description="", connected_to=["hall"]),
        ])
        assert ckpt.world_state.locations.scene_graph["hall"]["connected_to"] == ["garden"]


class TestSceneCreationValidation:
    def test_rejects_duplicate_id(self, caplog):
        ckpt = _ckpt()
        with caplog.at_level(logging.WARNING):
            _apply_scene_creations(ckpt, [
                SceneCreation(scene_id="hall", name="Another Hall", description="", connected_to=[]),
            ])
        # Existing entry unchanged.
        assert ckpt.world_state.locations.scene_graph["hall"]["name"] == "Great Hall"
        assert any("already exists" in r.message for r in caplog.records)

    def test_rejects_empty_id(self, caplog):
        ckpt = _ckpt()
        with caplog.at_level(logging.WARNING):
            _apply_scene_creations(ckpt, [
                SceneCreation(scene_id="", name="Nameless", description="", connected_to=[]),
            ])
        assert "" not in ckpt.world_state.locations.scene_graph
        assert any("empty scene_id" in r.message for r in caplog.records)

    def test_drops_dangling_connected_to_refs(self, caplog):
        ckpt = _ckpt()
        with caplog.at_level(logging.WARNING):
            _apply_scene_creations(ckpt, [
                SceneCreation(
                    scene_id="garden",
                    name="Garden",
                    description="",
                    connected_to=["hall", "phantom_scene"],
                ),
            ])
        g = ckpt.world_state.locations.scene_graph
        # phantom_scene was dropped; hall stayed.
        assert g["garden"]["connected_to"] == ["hall"]
        assert any("unknown scene" in r.message for r in caplog.records)

    def test_self_reference_dropped_silently(self):
        ckpt = _ckpt()
        _apply_scene_creations(ckpt, [
            SceneCreation(
                scene_id="garden",
                name="Garden",
                description="",
                connected_to=["garden", "hall"],
            ),
        ])
        assert ckpt.world_state.locations.scene_graph["garden"]["connected_to"] == ["hall"]

    def test_dedupes_connected_to(self):
        ckpt = _ckpt()
        _apply_scene_creations(ckpt, [
            SceneCreation(
                scene_id="garden",
                name="Garden",
                description="",
                connected_to=["hall", "hall"],
            ),
        ])
        assert ckpt.world_state.locations.scene_graph["garden"]["connected_to"] == ["hall"]


class TestEmptyBatch:
    def test_no_creations_is_noop(self):
        ckpt = _ckpt()
        graph_before = dict(ckpt.world_state.locations.scene_graph)
        _apply_scene_creations(ckpt, [])
        assert ckpt.world_state.locations.scene_graph == graph_before
