from __future__ import annotations

import pytest

from app.schemas.content_privacy import (
    REDACTED_IMPORT_SENTINEL,
    redact_imported_asset_text,
    redact_imported_content_metadata_text,
)


@pytest.mark.parametrize(
    "redactor",
    (redact_imported_asset_text, redact_imported_content_metadata_text),
)
@pytest.mark.parametrize(
    "private_path",
    (
        "/home/dan/ayoa/app/storage/stories/secret/hero.png",
        "/secret.png",
        '"/secret.png"',
        r"C:\Users\dan\ayoa\private\hero.png",
        "D:/ayoa/private/hero.png",
        r"\\authoring-host\private-share\ayoa\hero.png",
        "app/storage/stories/secret/hero.png",
        r"tests\fixtures\private\hero.png",
        "/home/dan/ayoa/private hero.png",
        r"C:\Program Files\Ayoa\private hero.png",
        "app/storage/stories/private hero.png",
        '"C:\\Program Files\\Ayoa\\private hero.png"',
        '"app/storage/stories/private hero.png"',
    ),
)
def test_content_metadata_redactor_removes_private_path_forms(
    private_path: str,
    redactor,
) -> None:
    cleaned = redactor(
        f"A scarlet coat remains visible beside {private_path} after the rain."
    )

    assert "A scarlet coat remains visible" in cleaned
    assert REDACTED_IMPORT_SENTINEL in cleaned
    assert private_path not in cleaned
    assert "hero.png" not in cleaned
    assert "Ayoa\\private" not in cleaned
    assert "app/storage" not in cleaned


def test_content_metadata_redactor_removes_source_markers() -> None:
    cleaned = redact_imported_content_metadata_text(
        "Scarlet coat. actor.hidden "
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )

    assert cleaned.startswith("Scarlet coat.")
    assert "actor.hidden" not in cleaned
    assert "sha256:" not in cleaned
    assert REDACTED_IMPORT_SENTINEL in cleaned


def test_content_metadata_redactor_preserves_ordinary_prose() -> None:
    prose = (
        "The app hums softly as windows open onto the courtyard. "
        "A scout takes the north/east fork, and C: marks the third alcove."
    )

    assert redact_imported_content_metadata_text(prose) == prose


@pytest.mark.parametrize(
    "redactor",
    (redact_imported_asset_text, redact_imported_content_metadata_text),
)
@pytest.mark.parametrize(
    "prose",
    (
        "Use /combat status to see the current state.",
        "Type /help for the command list.",
        'The button says "/roll".',
        "Try /inventory next.",
    ),
)
def test_content_redactors_preserve_single_token_slash_commands(
    redactor,
    prose: str,
) -> None:
    assert redactor(prose) == prose
