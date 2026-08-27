#!/usr/bin/env python3
"""Render the exact built-in image-generation prompts for this frozen experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def render_prompt(character: dict[str, object], variant_id: str, pose: str) -> str:
    name = str(character["display_name"])
    return (
        "\n".join(
            [
                "Use case: stylized-concept",
                "Asset type: experimental visual-novel character sprite intermediate",
                (
                    f"Primary request: Generate one new illustration of {name} expressing "
                    f"{variant_id} through both face and body language. This is one isolated "
                    "character asset, not a scene or reference sheet."
                ),
                f"Input images: {character['reference_roles']}",
                f"Subject: {character['subject']}",
                f"Expression and pose: {pose} {character['orientation']}",
                (
                    "Style/medium: Match the references' polished clean anime/manhwa character "
                    "illustration, crisp dark linework, restrained cel shading, and realistic "
                    "adult proportions. Preserve the canonical face, hair, anatomy, clothing "
                    "construction, colors, and weapon design; do not redesign the character."
                ),
                (
                    "Composition/framing: Portrait canvas, one centered full-body figure in a "
                    "three-quarter view, from the complete head through the complete boots. Keep a "
                    "modest top margin and a consistent foot baseline. Keep the "
                    "face, both hands, and emotion-defining gesture above the bottom 25 percent "
                    f"dialogue-safe zone. Keep these edges complete: {character['edge_details']}."
                ),
                (
                    "Scene/backdrop: One perfectly flat, uniform, fully opaque pure magenta chroma "
                    "background (#FF00FF), edge to edge. No floor, scenery, gradient, texture, "
                    "border, cast shadow, or checkerboard."
                ),
                (
                    f"Constraints: {character['constraints']} The magenta must not overlap any part "
                    "of the character or owned weapon. No text, logo, watermark, caption, labels, "
                    "frame, or UI."
                ),
                f"Avoid: {character['avoid']}",
                "Output intent: Clean chroma-key intermediate for deterministic authoring-time alpha extraction.",
            ]
        )
        + "\n"
    )


def main() -> None:
    for character_id, character in PROTOCOL["characters"].items():
        prompt_dir = ROOT / "prompts" / character_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for variant_id, pose in character["variants"].items():
            path = prompt_dir / f"{variant_id}_v1.txt"
            # Preserve the hand-authored pilot prompt and its iteration trail.
            if variant_id == "angry" and character_id == "mirelle_voss":
                continue
            if variant_id == "skeptical" and character_id == "rowan_kest":
                continue
            path.write_text(
                render_prompt(character, variant_id, pose), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
