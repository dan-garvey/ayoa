"""Deterministic classic-ADV card composition.

The narrator owns ordered page semantics. This module owns pixels, wrapping,
fallbacks, caching, and the transport-neutral deck manifest. It never sends
image data to an LLM and never changes the supplied scene plate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont

from app.engine.player_media import PlayerMediaBytes
from app.schemas.narrator import (
    VisualNovelPage,
    visual_novel_pages_contain_source_identifiers,
    visual_novel_text_contains_source_identifiers,
)


CARD_WIDTH = 1024
CARD_HEIGHT = 576
_RENDERER_VERSION = "classic-adv-v7-full-speaker-nameplate"
_SPEAKER_NAME_MAX_WIDTH = 900
_MANIFEST_VERSION = 2
_SHA256_LENGTH = 64
_MAX_BODY_LINES = 4
_MAX_SPRITES_PER_SECTION = 2
_MAX_SPRITE_BYTES = 20_000_000
_MAX_SPRITE_EDGE = 8_192
_MAX_SPRITE_PIXELS = 40_000_000
_MIN_SPRITE_SCALE_PERCENT = 25
_MAX_SPRITE_SCALE_PERCENT = 150
_MIN_SPRITE_BASELINE_Y = 396
_OPAQUE_SPRITE_HANDLE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?…]+(?:[\"”’')\]]+)?(?=\s+|$)"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:—–])\s+")
_COMMON_ABBREVIATIONS = {
    "dr.",
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "st.",
    "vs.",
}
_CONTINUATION_FINAL_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "because",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "in",
    "into",
    "is",
    "its",
    "of",
    "on",
    "or",
    "our",
    "over",
    "than",
    "that",
    "the",
    "their",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "was",
    "were",
    "with",
    "your",
}
_DEFAULT_REGULAR_FONT = Path(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)
_DEFAULT_BOLD_FONT = Path(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)


@dataclass(frozen=True)
class VisualNovelCard:
    """One physical card with immutable bytes as transport truth.

    ``image_path`` locates the persisted artifact for diagnostics and cache
    maintenance. Frontends must use ``image_bytes`` so a later filesystem
    change cannot alter bytes already accepted by the manifest validator.
    """

    index: int
    count: int
    kind: str
    speaker: str
    text: str
    image_path: Path
    image_bytes: bytes = field(repr=False)

    @property
    def accessible_text(self) -> str:
        """Return this physical card's transport-neutral text projection."""

        text = " ".join(str(self.text or "").split())
        if self.kind != "dialogue":
            return text
        speaker = " ".join(str(self.speaker or "").split())
        return f"{speaker}: {text}" if speaker else text


@dataclass(frozen=True)
class VisualNovelDeck:
    deck_id: str
    cards: tuple[VisualNovelCard, ...]
    transcript: str
    manifest_path: Path
    used_neutral_stage: bool


class VisualNovelSpriteError(ValueError):
    """A resolved sprite violated the deterministic compositor contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"visual-novel sprite validation failed ({code})")


@dataclass(frozen=True)
class VisualNovelSpritePlacement:
    """One resolved transparent cutout and its deterministic card transform.

    The source canvas is anchored by its bottom-center point at ``anchor``.
    ``scale_percent`` normalizes the source canvas height against the card
    height before placement. ``source_facing`` describes the reviewed source;
    the compositor mirrors it only when ``facing`` differs. Handles are opaque
    provenance and are never player text or LLM input.
    """

    identity_handle: str
    variant_handle: str
    media: PlayerMediaBytes
    slot: Literal["left", "center", "right"]
    source_facing: Literal["left", "right"]
    facing: Literal["left", "right"]
    anchor: tuple[int, int]
    scale_percent: int = 100


@dataclass(frozen=True)
class VisualNovelDeckSection:
    """Ordered semantic pages resolved against one immutable stage plate.

    Every physical page produced from this section inherits the same ordered
    sprite placements. A caller changes a pose or expression on a later page
    by starting another section against the same stage with different resolved
    variant provenance.
    """

    pages: tuple[VisualNovelPage, ...]
    stage_path: str | Path | None = None
    stage_media: PlayerMediaBytes | None = None
    sprite_placements: tuple[VisualNovelSpritePlacement, ...] = ()


@dataclass(frozen=True)
class _ResolvedSpritePlacement:
    image: Image.Image = field(repr=False)
    identity: dict[str, object]


@dataclass(frozen=True)
class _ResolvedDeckSection:
    composed_stage: Image.Image = field(repr=False)
    stage_sha256: str
    used_neutral_stage: bool
    sprites: tuple[_ResolvedSpritePlacement, ...]
    pages: tuple[VisualNovelPage, ...]


class VisualNovelCardRenderer:
    """Build and load content-addressed 1024x576 PNG card decks."""

    def __init__(
        self,
        runtime_root: str | Path,
        *,
        regular_font_path: str | Path = _DEFAULT_REGULAR_FONT,
        bold_font_path: str | Path = _DEFAULT_BOLD_FONT,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.deck_root = self.runtime_root / "decks"
        self._absolute_runtime_root = Path(os.path.abspath(self.runtime_root))
        try:
            runtime_fd = _open_directory_path(
                self._absolute_runtime_root,
                create=True,
            )
            try:
                try:
                    os.mkdir("decks", dir_fd=runtime_fd)
                except FileExistsError:
                    pass
                deck_fd = _open_child_directory(runtime_fd, "decks")
                try:
                    self._runtime_root_identity = _fd_identity(runtime_fd)
                    self._deck_root_identity = _fd_identity(deck_fd)
                finally:
                    os.close(deck_fd)
            finally:
                os.close(runtime_fd)
        except OSError as exc:
            raise RuntimeError(
                "visual-novel deck root is not a safe directory"
            ) from exc
        self.regular_font_path = Path(regular_font_path)
        self.bold_font_path = Path(bold_font_path)

    def render_deck(
        self,
        sections: Sequence[VisualNovelDeckSection],
    ) -> VisualNovelDeck:
        self._verify_pinned_deck_root()
        if not sections:
            raise ValueError("visual-novel decks require at least one section")
        if any(
            visual_novel_pages_contain_source_identifiers(section.pages)
            for section in sections
        ):
            raise ValueError(
                "visual-novel deck pages cannot expose source-shaped ids"
            )
        fonts = self._fonts()
        resolved_sections: list[_ResolvedDeckSection] = []
        for section in sections:
            if not section.pages:
                raise ValueError(
                    "visual-novel deck sections require at least one page"
                )
            if section.stage_path is not None and section.stage_media is not None:
                raise ValueError(
                    "visual-novel deck sections accept one stage source"
                )
            stage, stage_sha256, used_neutral = _load_stage(
                section.stage_path,
                stage_media=section.stage_media,
            )
            sprites = _resolve_sprite_placements(section.sprite_placements)
            resolved_sections.append(_ResolvedDeckSection(
                composed_stage=_compose_sprite_stage(stage, sprites),
                stage_sha256=stage_sha256,
                used_neutral_stage=used_neutral,
                sprites=sprites,
                pages=tuple(_paginate_pages(section.pages, fonts.body)),
            ))
        identity = {
            "renderer": _RENDERER_VERSION,
            "card_size": [CARD_WIDTH, CARD_HEIGHT],
            "fonts": {
                "regular_sha256": _file_sha256(self.regular_font_path),
                "bold_sha256": _file_sha256(self.bold_font_path),
            },
            "sections": [
                {
                    "stage_sha256": section.stage_sha256,
                    "used_neutral_stage": section.used_neutral_stage,
                    "sprites": [
                        sprite.identity for sprite in section.sprites
                    ],
                    "pages": [
                        page.model_dump(mode="json", exclude={"sprites"})
                        for page in section.pages
                    ],
                }
                for section in resolved_sections
            ],
        }
        count = sum(
            len(section.pages) for section in resolved_sections
        )
        rendered_cards: list[tuple[VisualNovelPage, bytes, str]] = []
        physical_pages: list[VisualNovelPage] = []
        index = 0
        for section in resolved_sections:
            for page in section.pages:
                index += 1
                physical_pages.append(page)
                card_image = _compose_card(
                    section.composed_stage,
                    page,
                    index=index,
                    count=count,
                    fonts=fonts,
                )
                encoded = BytesIO()
                card_image.save(encoded, format="PNG", optimize=False)
                image_bytes = encoded.getvalue()
                rendered_cards.append((
                    page,
                    image_bytes,
                    hashlib.sha256(image_bytes).hexdigest(),
                ))

        transcript = _transcript(physical_pages)
        used_neutral = any(
            section.used_neutral_stage for section in resolved_sections
        )
        card_sha256s = [sha256 for _page, _data, sha256 in rendered_cards]
        deck_id = _deck_content_id(identity, card_sha256s)
        deck_dir = self.deck_root / deck_id
        manifest_path = deck_dir / "manifest.json"

        with self._pinned_deck_root_fd() as deck_root_fd:
            deck_fd = self._open_render_deck_directory(
                deck_root_fd,
                deck_id,
            )
            try:
                cached = self._load_validated_deck_from_fd(
                    clean_id=deck_id,
                    deck_fd=deck_fd,
                )
                if cached is not None:
                    return cached

                _require_safe_regular_target(
                    deck_fd,
                    "manifest.json",
                    label="manifest",
                )
                for card_index in range(1, count + 1):
                    _require_safe_regular_target(
                        deck_fd,
                        f"page-{card_index:03d}.png",
                        label="card",
                    )

                cards: list[VisualNovelCard] = []
                raw_card_manifest: list[dict[str, object]] = []
                for card_index, (page, image_bytes, sha256) in enumerate(
                    rendered_cards,
                    start=1,
                ):
                    filename = f"page-{card_index:03d}.png"
                    self._verify_pinned_deck_root()
                    _atomic_write_regular_file(
                        deck_fd,
                        filename,
                        image_bytes,
                    )
                    cards.append(VisualNovelCard(
                        index=card_index,
                        count=count,
                        kind=page.kind,
                        speaker=page.speaker,
                        text=page.text,
                        image_path=deck_dir / filename,
                        image_bytes=image_bytes,
                    ))
                    raw_card_manifest.append({
                        "index": card_index,
                        "count": count,
                        "kind": page.kind,
                        "speaker": page.speaker,
                        "text": page.text,
                        "filename": filename,
                        "sha256": sha256,
                    })

                manifest = {
                    "version": _MANIFEST_VERSION,
                    "deck_id": deck_id,
                    "identity": identity,
                    "used_neutral_stage": used_neutral,
                    "transcript": transcript,
                    "cards": raw_card_manifest,
                }
                manifest_bytes = (
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n"
                ).encode("utf-8")
                self._verify_pinned_deck_root()
                _atomic_write_regular_file(
                    deck_fd,
                    "manifest.json",
                    manifest_bytes,
                )
            finally:
                os.close(deck_fd)

            return VisualNovelDeck(
                deck_id=deck_id,
                cards=tuple(cards),
                transcript=transcript,
                manifest_path=manifest_path,
                used_neutral_stage=used_neutral,
            )

    def load_deck(self, deck_id: str) -> VisualNovelDeck | None:
        """Load one fail-closed persisted deck.

        Only version 2 is accepted. It requires this renderer contract,
        binds stage and ordered sprite provenance plus the ordered card digests
        into the deck id, verifies and snapshots every card's exact bytes, and
        rejects source-shaped identifiers in player-visible fields. Its font
        digests identify the historical render inputs; they need not match
        fonts installed after a restart. Older versions and unknown renderers
        are not migrated or guessed.
        """

        clean_id = str(deck_id or "").strip().lower()
        if (
            len(clean_id) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in clean_id)
        ):
            return None
        try:
            with self._pinned_deck_root_fd() as deck_root_fd:
                deck_fd = self._open_load_deck_directory(
                    deck_root_fd,
                    clean_id,
                )
                try:
                    return self._load_validated_deck_from_fd(
                        clean_id=clean_id,
                        deck_fd=deck_fd,
                    )
                finally:
                    os.close(deck_fd)
        except Exception:
            # Persisted files are a restart boundary. Malformed JSON, odd path
            # types, Pillow decoder failures, and unexpected legacy values all
            # fail closed instead of escaping through a Discord callback.
            return None

    def _load_validated_deck_from_fd(
        self,
        *,
        clean_id: str,
        deck_fd: int,
    ) -> VisualNovelDeck | None:
        deck_dir = self.deck_root / clean_id
        manifest_path = deck_dir / "manifest.json"
        self._verify_pinned_deck_root()
        manifest_bytes = _read_regular_file(deck_fd, "manifest.json")
        if manifest_bytes is None:
            return None
        payload = json.loads(manifest_bytes.decode("utf-8"))
        if type(payload) is not dict:
            return None

        version = payload.get("version")
        if type(version) is not int or version != _MANIFEST_VERSION:
            return None
        expected_manifest_keys = {
            "version",
            "deck_id",
            "identity",
            "used_neutral_stage",
            "transcript",
            "cards",
        }
        if set(payload) != expected_manifest_keys:
            return None
        if type(payload["deck_id"]) is not str or payload["deck_id"] != clean_id:
            return None
        if type(payload["used_neutral_stage"]) is not bool:
            return None
        if type(payload["transcript"]) is not str:
            return None

        raw_cards = payload["cards"]
        if type(raw_cards) is not list or not raw_cards:
            return None
        expected_card_keys = {
            "index",
            "count",
            "kind",
            "speaker",
            "text",
            "filename",
            "sha256",
        }

        cards: list[VisualNovelCard] = []
        pages: list[VisualNovelPage] = []
        card_sha256s: list[str] = []
        count = len(raw_cards)
        for expected_index, raw in enumerate(raw_cards, start=1):
            if type(raw) is not dict or set(raw) != expected_card_keys:
                return None
            if type(raw["index"]) is not int or raw["index"] != expected_index:
                return None
            if type(raw["count"]) is not int or raw["count"] != count:
                return None
            if not _valid_page_fields(
                kind=raw["kind"],
                speaker=raw["speaker"],
                text=raw["text"],
            ):
                return None
            filename = raw["filename"]
            canonical_filename = f"page-{expected_index:03d}.png"
            if type(filename) is not str or filename != canonical_filename:
                return None
            expected_sha256 = raw["sha256"]
            if not _is_sha256(expected_sha256):
                return None
            self._verify_pinned_deck_root()
            image_bytes = _read_regular_file(deck_fd, filename)
            if image_bytes is None or not _valid_card_png_bytes(
                image_bytes,
                expected_sha256=expected_sha256,
            ):
                return None
            card_sha256s.append(expected_sha256)
            cards.append(VisualNovelCard(
                index=raw["index"],
                count=raw["count"],
                kind=raw["kind"],
                speaker=raw["speaker"],
                text=raw["text"],
                image_path=deck_dir / filename,
                image_bytes=image_bytes,
            ))
            pages.append(VisualNovelPage(
                kind=raw["kind"],
                speaker=raw["speaker"],
                text=raw["text"],
            ))

        if payload["transcript"] != _transcript(pages):
            return None
        if not _valid_v2_identity(
            payload["identity"],
            deck_id=clean_id,
            pages=pages,
            used_neutral_stage=payload["used_neutral_stage"],
            card_sha256s=card_sha256s,
        ):
            return None
        return VisualNovelDeck(
            deck_id=clean_id,
            cards=tuple(cards),
            transcript=payload["transcript"],
            manifest_path=manifest_path,
            used_neutral_stage=payload["used_neutral_stage"],
        )

    def _fonts(self) -> "_CardFonts":
        if not self.regular_font_path.is_file():
            raise RuntimeError(
                f"visual-novel font is unavailable: {self.regular_font_path}"
            )
        if not self.bold_font_path.is_file():
            raise RuntimeError(
                f"visual-novel font is unavailable: {self.bold_font_path}"
            )
        return _CardFonts(
            body=ImageFont.truetype(str(self.regular_font_path), 24),
            speaker=ImageFont.truetype(str(self.bold_font_path), 25),
            counter=ImageFont.truetype(str(self.regular_font_path), 16),
        )

    def _open_pinned_root_fds(self) -> tuple[int, int]:
        try:
            runtime_fd = _open_directory_path(
                self._absolute_runtime_root,
                create=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "visual-novel deck root changed after renderer construction"
            ) from exc
        try:
            if _fd_identity(runtime_fd) != self._runtime_root_identity:
                raise RuntimeError(
                    "visual-novel deck root changed after renderer construction"
                )
            try:
                deck_fd = _open_child_directory(runtime_fd, "decks")
            except OSError as exc:
                raise RuntimeError(
                    "visual-novel deck root changed after renderer construction"
                ) from exc
            if _fd_identity(deck_fd) != self._deck_root_identity:
                os.close(deck_fd)
                raise RuntimeError(
                    "visual-novel deck root changed after renderer construction"
                )
        except Exception:
            os.close(runtime_fd)
            raise
        return runtime_fd, deck_fd

    def _verify_pinned_deck_root(self) -> None:
        runtime_fd, deck_fd = self._open_pinned_root_fds()
        os.close(deck_fd)
        os.close(runtime_fd)

    @contextmanager
    def _pinned_deck_root_fd(self) -> Iterator[int]:
        """Anchor I/O to pinned inodes and recheck their named path on exit."""

        runtime_fd, deck_fd = self._open_pinned_root_fds()
        completed = False
        try:
            yield deck_fd
            completed = True
        finally:
            os.close(deck_fd)
            os.close(runtime_fd)
            if completed:
                self._verify_pinned_deck_root()

    def _open_render_deck_directory(
        self,
        deck_root_fd: int,
        deck_id: str,
    ) -> int:
        self._verify_pinned_deck_root()
        try:
            return _open_child_directory(deck_root_fd, deck_id)
        except FileNotFoundError:
            self._verify_pinned_deck_root()
            try:
                os.mkdir(deck_id, dir_fd=deck_root_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    "visual-novel deck path is not a safe directory"
                ) from exc
            try:
                return _open_child_directory(deck_root_fd, deck_id)
            except OSError as exc:
                raise RuntimeError(
                    "visual-novel deck path is not a safe directory"
                ) from exc
        except OSError as exc:
            raise RuntimeError(
                "visual-novel deck path is not a safe directory"
            ) from exc

    def _open_load_deck_directory(
        self,
        deck_root_fd: int,
        deck_id: str,
    ) -> int:
        self._verify_pinned_deck_root()
        return _open_child_directory(deck_root_fd, deck_id)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )


def _open_child_directory(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("directory name must be one canonical component")
    return os.open(
        name,
        _directory_open_flags(),
        dir_fd=parent_fd,
    )


def _open_directory_path(path: Path, *, create: bool) -> int:
    """Open a directory by components without following symlink ancestors."""

    if not path.is_absolute():
        raise ValueError("directory path must be absolute")
    current_fd = os.open("/", _directory_open_flags())
    try:
        for component in path.parts[1:]:
            if create:
                try:
                    os.mkdir(component, dir_fd=current_fd)
                except FileExistsError:
                    pass
            next_fd = _open_child_directory(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _fd_identity(fd: int) -> tuple[int, int]:
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError("file descriptor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _read_regular_file(directory_fd: int, filename: str) -> bytes | None:
    try:
        fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    except OSError:
        return None
    finally:
        os.close(fd)


def _require_safe_regular_target(
    directory_fd: int,
    filename: str,
    *,
    label: str,
) -> None:
    try:
        metadata = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"visual-novel {label} path is not a safe file"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(
            f"visual-novel {label} path is not a safe file"
        )


def _atomic_write_regular_file(
    directory_fd: int,
    filename: str,
    data: bytes,
) -> None:
    label = "manifest" if filename == "manifest.json" else "card"
    _require_safe_regular_target(
        directory_fd,
        filename,
        label=label,
    )
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except Exception:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _deck_content_id(
    identity: object,
    card_sha256s: Sequence[str],
) -> str:
    return _canonical_json_sha256({
        "identity": identity,
        "card_sha256s": list(card_sha256s),
    })


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_page_fields(*, kind: object, speaker: object, text: object) -> bool:
    if type(kind) is not str or kind not in {"narration", "dialogue"}:
        return False
    if type(speaker) is not str or len(speaker) > 80:
        return False
    if type(text) is not str or not text.strip() or len(text) > 4_000:
        return False
    if speaker != speaker.strip() or text != text.strip():
        return False
    if any(
        visual_novel_text_contains_source_identifiers(value)
        for value in (speaker, text)
    ):
        return False
    if kind == "narration":
        return speaker == ""
    return bool(speaker.strip())


def _valid_card_png_bytes(
    data: bytes,
    *,
    expected_sha256: str | None,
) -> bool:
    try:
        if (
            expected_sha256 is not None
            and hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            return False
        with Image.open(BytesIO(data)) as opened:
            if not _valid_static_card_image(opened):
                return False
            opened.verify()
        # ``verify`` checks container integrity without decoding pixel data.
        # Reopen and load so truncated IDAT streams also fail closed.
        with Image.open(BytesIO(data)) as opened:
            if not _valid_static_card_image(opened):
                return False
            opened.load()
    except Exception:
        return False
    return True


def _valid_static_card_image(image: Image.Image) -> bool:
    return (
        image.format == "PNG"
        and image.mode == "RGB"
        and image.size == (CARD_WIDTH, CARD_HEIGHT)
        and not bool(getattr(image, "is_animated", False))
        and int(getattr(image, "n_frames", 1)) == 1
    )


def _valid_sprite_manifest_identity(value: object) -> bool:
    expected_keys = {
        "identity_handle",
        "variant_handle",
        "source_sha256",
        "source_mime_type",
        "source_byte_count",
        "source_size",
        "slot",
        "source_facing",
        "facing",
        "anchor",
        "scale_percent",
    }
    if type(value) is not dict or set(value) != expected_keys:
        return False
    if not _valid_opaque_sprite_handle(value["identity_handle"]):
        return False
    if not _valid_opaque_sprite_handle(value["variant_handle"]):
        return False
    if not _is_sha256(value["source_sha256"]):
        return False
    if value["source_mime_type"] != "image/png":
        return False
    if (
        type(value["source_byte_count"]) is not int
        or not 0 < value["source_byte_count"] <= _MAX_SPRITE_BYTES
    ):
        return False
    source_size = value["source_size"]
    if (
        type(source_size) is not list
        or len(source_size) != 2
        or not _valid_sprite_dimensions(source_size[0], source_size[1])
    ):
        return False
    anchor = value["anchor"]
    if type(anchor) is not list:
        return False
    if not _valid_sprite_transform(
        slot=value["slot"],
        source_facing=value["source_facing"],
        facing=value["facing"],
        anchor=anchor,
        scale_percent=value["scale_percent"],
    ):
        return False
    target_width, _target_height = _scaled_sprite_size(
        source_size[0],
        source_size[1],
        value["scale_percent"],
    )
    return target_width <= CARD_WIDTH * 2


def _valid_v2_identity(
    identity: object,
    *,
    deck_id: str,
    pages: Sequence[VisualNovelPage],
    used_neutral_stage: bool,
    card_sha256s: Sequence[str],
) -> bool:
    if type(identity) is not dict or set(identity) != {
        "renderer",
        "card_size",
        "fonts",
        "sections",
    }:
        return False
    if identity["renderer"] != _RENDERER_VERSION:
        return False
    card_size = identity["card_size"]
    if (
        type(card_size) is not list
        or len(card_size) != 2
        or any(type(dimension) is not int for dimension in card_size)
        or card_size != [CARD_WIDTH, CARD_HEIGHT]
    ):
        return False
    fonts = identity["fonts"]
    if type(fonts) is not dict or set(fonts) != {
        "regular_sha256",
        "bold_sha256",
    }:
        return False
    if not all(_is_sha256(value) for value in fonts.values()):
        return False

    sections = identity["sections"]
    if type(sections) is not list or not sections:
        return False
    identity_pages: list[VisualNovelPage] = []
    neutral_sections: list[bool] = []
    for section in sections:
        if type(section) is not dict or set(section) != {
            "stage_sha256",
            "used_neutral_stage",
            "sprites",
            "pages",
        }:
            return False
        if not _is_sha256(section["stage_sha256"]):
            return False
        if type(section["used_neutral_stage"]) is not bool:
            return False
        raw_sprites = section["sprites"]
        if (
            type(raw_sprites) is not list
            or len(raw_sprites) > _MAX_SPRITES_PER_SECTION
            or not all(
                _valid_sprite_manifest_identity(sprite)
                for sprite in raw_sprites
            )
        ):
            return False
        slots = [sprite["slot"] for sprite in raw_sprites]
        identity_handles = [
            sprite["identity_handle"] for sprite in raw_sprites
        ]
        if len(slots) != len(set(slots)):
            return False
        if len(identity_handles) != len(set(identity_handles)):
            return False
        if len(raw_sprites) == 2 and set(slots) != {"left", "right"}:
            return False
        raw_pages = section["pages"]
        if type(raw_pages) is not list or not raw_pages:
            return False
        neutral_sections.append(section["used_neutral_stage"])
        for raw_page in raw_pages:
            if type(raw_page) is not dict or set(raw_page) != {
                "kind",
                "speaker",
                "text",
            }:
                return False
            if not _valid_page_fields(
                kind=raw_page["kind"],
                speaker=raw_page["speaker"],
                text=raw_page["text"],
            ):
                return False
            identity_pages.append(VisualNovelPage(**raw_page))

    if _deck_content_id(identity, card_sha256s) != deck_id:
        return False
    if any(neutral_sections) != used_neutral_stage:
        return False
    return [
        page.model_dump(mode="json", exclude={"sprites"})
        for page in identity_pages
    ] == [
        page.model_dump(mode="json", exclude={"sprites"}) for page in pages
    ]


@dataclass(frozen=True)
class _CardFonts:
    body: ImageFont.FreeTypeFont
    speaker: ImageFont.FreeTypeFont
    counter: ImageFont.FreeTypeFont


def _resolve_sprite_placements(
    placements: tuple[VisualNovelSpritePlacement, ...],
) -> tuple[_ResolvedSpritePlacement, ...]:
    if type(placements) is not tuple:
        raise VisualNovelSpriteError("placements_not_tuple")
    if len(placements) > _MAX_SPRITES_PER_SECTION:
        raise VisualNovelSpriteError("too_many_placements")

    resolved: list[_ResolvedSpritePlacement] = []
    slots: set[str] = set()
    identity_handles: set[str] = set()
    for placement in placements:
        if type(placement) is not VisualNovelSpritePlacement:
            raise VisualNovelSpriteError("invalid_placement")
        resolved_placement = _resolve_sprite_placement(placement)
        if placement.slot in slots:
            raise VisualNovelSpriteError("duplicate_slot")
        if placement.identity_handle in identity_handles:
            raise VisualNovelSpriteError("duplicate_identity")
        slots.add(placement.slot)
        identity_handles.add(placement.identity_handle)
        resolved.append(resolved_placement)

    if len(resolved) == 2 and slots != {"left", "right"}:
        raise VisualNovelSpriteError("two_sprite_slots_must_be_left_right")
    return tuple(resolved)


def _resolve_sprite_placement(
    placement: VisualNovelSpritePlacement,
) -> _ResolvedSpritePlacement:
    if not _valid_opaque_sprite_handle(placement.identity_handle):
        raise VisualNovelSpriteError("invalid_identity_handle")
    if not _valid_opaque_sprite_handle(placement.variant_handle):
        raise VisualNovelSpriteError("invalid_variant_handle")
    if not _valid_sprite_transform(
        slot=placement.slot,
        source_facing=placement.source_facing,
        facing=placement.facing,
        anchor=placement.anchor,
        scale_percent=placement.scale_percent,
    ):
        raise VisualNovelSpriteError("invalid_transform")

    media = placement.media
    if not isinstance(media, PlayerMediaBytes):
        # Deliberately accept bytes only. Paths, including symlinks, must be
        # resolved and validated by the private asset owner before this layer.
        raise VisualNovelSpriteError("unresolved_media")
    if (
        type(media.filename) is not str
        or not media.filename
        or len(media.filename) > 255
        or "/" in media.filename
        or "\\" in media.filename
        or not media.filename.lower().endswith(".png")
    ):
        raise VisualNovelSpriteError("unsafe_media_filename")
    if type(media.mime_type) is not str or media.mime_type != "image/png":
        raise VisualNovelSpriteError("invalid_media_type")
    if type(media.data) is not bytes or not media.data:
        raise VisualNovelSpriteError("invalid_media_bytes")
    if len(media.data) > _MAX_SPRITE_BYTES:
        raise VisualNovelSpriteError("media_too_large")
    if not _is_sha256(media.sha256):
        raise VisualNovelSpriteError("invalid_media_hash")
    if hashlib.sha256(media.data).hexdigest() != media.sha256:
        raise VisualNovelSpriteError("media_hash_mismatch")
    if (
        type(media.byte_count) is not int
        or media.byte_count != len(media.data)
    ):
        raise VisualNovelSpriteError("media_byte_count_mismatch")
    if not _valid_sprite_dimensions(media.width, media.height):
        raise VisualNovelSpriteError("invalid_media_dimensions")
    target_width, _target_height = _scaled_sprite_size(
        media.width,
        media.height,
        placement.scale_percent,
    )
    if target_width > CARD_WIDTH * 2:
        raise VisualNovelSpriteError("rendered_sprite_too_wide")

    image = _decode_sprite_png(
        media.data,
        expected_width=media.width,
        expected_height=media.height,
    )
    return _ResolvedSpritePlacement(
        image=image,
        identity={
            "identity_handle": placement.identity_handle,
            "variant_handle": placement.variant_handle,
            "source_sha256": media.sha256,
            "source_mime_type": media.mime_type,
            "source_byte_count": media.byte_count,
            "source_size": [media.width, media.height],
            "slot": placement.slot,
            "source_facing": placement.source_facing,
            "facing": placement.facing,
            "anchor": list(placement.anchor),
            "scale_percent": placement.scale_percent,
        },
    )


def _decode_sprite_png(
    data: bytes,
    *,
    expected_width: int,
    expected_height: int,
) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as opened:
            if not _valid_static_sprite_image(
                opened,
                expected_width=expected_width,
                expected_height=expected_height,
            ):
                raise VisualNovelSpriteError("invalid_png")
            opened.verify()
        with Image.open(BytesIO(data)) as opened:
            if not _valid_static_sprite_image(
                opened,
                expected_width=expected_width,
                expected_height=expected_height,
            ):
                raise VisualNovelSpriteError("invalid_png")
            image = opened.convert("RGBA")
            image.load()
    except VisualNovelSpriteError:
        raise
    except Exception as exc:
        raise VisualNovelSpriteError("invalid_png") from exc

    alpha_minimum, alpha_maximum = image.getchannel("A").getextrema()
    if alpha_minimum == 255:
        raise VisualNovelSpriteError("opaque_png")
    if alpha_maximum == 0:
        raise VisualNovelSpriteError("empty_png")
    return image


def _valid_static_sprite_image(
    image: Image.Image,
    *,
    expected_width: int,
    expected_height: int,
) -> bool:
    has_alpha = (
        "A" in image.getbands()
        or "transparency" in image.info
    )
    return (
        image.format == "PNG"
        and image.size == (expected_width, expected_height)
        and has_alpha
        and not bool(getattr(image, "is_animated", False))
        and int(getattr(image, "n_frames", 1)) == 1
    )


def _valid_sprite_dimensions(width: object, height: object) -> bool:
    return (
        type(width) is int
        and type(height) is int
        and 0 < width <= _MAX_SPRITE_EDGE
        and 0 < height <= _MAX_SPRITE_EDGE
        and width * height <= _MAX_SPRITE_PIXELS
    )


def _scaled_sprite_size(
    source_width: int,
    source_height: int,
    scale_percent: int,
) -> tuple[int, int]:
    target_height = max(
        1,
        (CARD_HEIGHT * scale_percent + 50) // 100,
    )
    target_width = max(
        1,
        (source_width * target_height + source_height // 2)
        // source_height,
    )
    return target_width, target_height


def _valid_opaque_sprite_handle(value: object) -> bool:
    return (
        type(value) is str
        and _OPAQUE_SPRITE_HANDLE_RE.fullmatch(value) is not None
    )


def _valid_sprite_transform(
    *,
    slot: object,
    source_facing: object,
    facing: object,
    anchor: object,
    scale_percent: object,
) -> bool:
    if type(slot) is not str or slot not in {"left", "center", "right"}:
        return False
    if type(source_facing) is not str or source_facing not in {"left", "right"}:
        return False
    if type(facing) is not str or facing not in {"left", "right"}:
        return False
    if (
        type(anchor) not in {tuple, list}
        or len(anchor) != 2
        or any(type(value) is not int for value in anchor)
    ):
        return False
    anchor_x, anchor_y = anchor
    if not 0 <= anchor_x <= CARD_WIDTH:
        return False
    if not _MIN_SPRITE_BASELINE_Y <= anchor_y <= CARD_HEIGHT:
        return False
    if slot == "left" and anchor_x > CARD_WIDTH // 2:
        return False
    if slot == "right" and anchor_x < CARD_WIDTH // 2:
        return False
    if slot == "center" and not CARD_WIDTH // 4 <= anchor_x <= 3 * CARD_WIDTH // 4:
        return False
    return (
        type(scale_percent) is int
        and _MIN_SPRITE_SCALE_PERCENT
        <= scale_percent
        <= _MAX_SPRITE_SCALE_PERCENT
    )


def _compose_sprite_stage(
    stage: Image.Image,
    sprites: Sequence[_ResolvedSpritePlacement],
) -> Image.Image:
    composed = stage.copy().convert("RGBA")
    for sprite in sprites:
        source_width, source_height = sprite.identity["source_size"]
        scale_percent = sprite.identity["scale_percent"]
        target_width, target_height = _scaled_sprite_size(
            source_width,
            source_height,
            scale_percent,
        )
        transformed = sprite.image
        if transformed.size != (target_width, target_height):
            # Resample premultiplied channels so transparent canvas pixels do
            # not introduce a dark fringe around antialiased cutout edges.
            transformed = (
                transformed.convert("RGBa")
                .resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS,
                )
                .convert("RGBA")
            )
        if sprite.identity["source_facing"] != sprite.identity["facing"]:
            transformed = transformed.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
        anchor_x, anchor_y = sprite.identity["anchor"]
        composed.alpha_composite(
            transformed,
            dest=(
                anchor_x - target_width // 2,
                anchor_y - target_height,
            ),
        )
    return composed.convert("RGB")


def _load_stage(
    stage_path: str | Path | None,
    *,
    stage_media: PlayerMediaBytes | None = None,
) -> tuple[Image.Image, str, bool]:
    if stage_media is not None:
        try:
            with Image.open(BytesIO(stage_media.data)) as opened:
                stage = opened.convert("RGB")
                stage.load()
            return (
                _cover(stage, CARD_WIDTH, CARD_HEIGHT),
                stage_media.sha256,
                False,
            )
        except (OSError, ValueError):
            pass
    if stage_path is not None:
        path = Path(stage_path)
        try:
            data = path.read_bytes()
            with Image.open(path) as opened:
                stage = opened.convert("RGB")
                stage.load()
            return (
                _cover(stage, CARD_WIDTH, CARD_HEIGHT),
                hashlib.sha256(data).hexdigest(),
                False,
            )
        except (OSError, ValueError):
            pass
    neutral = _neutral_stage()
    pixels = neutral.tobytes()
    return neutral, hashlib.sha256(pixels).hexdigest(), True


def _cover(source: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (
            max(width, round(source.width * scale)),
            max(height, round(source.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _neutral_stage() -> Image.Image:
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT))
    pixels = image.load()
    for y in range(CARD_HEIGHT):
        ratio = y / max(1, CARD_HEIGHT - 1)
        for x in range(CARD_WIDTH):
            horizontal = abs((x / max(1, CARD_WIDTH - 1)) - 0.5) * 2
            shade = int(13 * ratio + 8 * horizontal)
            pixels[x, y] = (
                max(8, 30 - shade),
                max(14, 50 - shade),
                max(24, 72 - shade),
            )
    return image


def _paginate_pages(
    pages: Sequence[VisualNovelPage],
    body_font: ImageFont.FreeTypeFont,
) -> list[VisualNovelPage]:
    result: list[VisualNovelPage] = []
    for page in _coalesce_incomplete_pages(pages):
        for physical_text in _paginate_page_text(page.text, body_font):
            result.append(VisualNovelPage(
                kind=page.kind,
                speaker=page.speaker,
                text=physical_text,
            ))
    return result


def _coalesce_incomplete_pages(
    pages: Sequence[VisualNovelPage],
) -> list[VisualNovelPage]:
    """Repair adjacent model pages that split one speaker's sentence."""

    coalesced: list[VisualNovelPage] = []
    for source_page in pages:
        page = source_page.model_copy(deep=True)
        if not coalesced:
            coalesced.append(page)
            continue
        previous = coalesced[-1]
        same_channel = (
            previous.kind == page.kind
            and previous.speaker == page.speaker
        )
        if not same_channel or not _continues_prior_page(
            previous.text,
            page.text,
        ):
            coalesced.append(page)
            continue
        separator = "" if previous.text.rstrip().endswith(("-", "—")) else " "
        previous.text = (
            previous.text.rstrip() + separator + page.text.lstrip()
        )
    return coalesced


def _ends_complete_sentence(text: str) -> bool:
    value = str(text or "").rstrip()
    if not value:
        return False
    return any(
        match.end() == len(value)
        for match in _SENTENCE_BOUNDARY_RE.finditer(value)
    )


def _continues_prior_page(previous_text: str, next_text: str) -> bool:
    """Recognize a likely model-authored mid-sentence page boundary."""

    previous = str(previous_text or "").rstrip()
    following = str(next_text or "").lstrip()
    if not previous or not following or _ends_complete_sentence(previous):
        return False
    if previous.endswith((",", ";", ":", "-", "—", "(", "[")):
        return True
    if _looks_like_abbreviation(previous):
        return True
    first_letter = re.search(r"[A-Za-z]", following)
    if first_letter is not None and first_letter.group(0).islower():
        return True
    final_word = re.search(r"([A-Za-z]+)[^A-Za-z]*$", previous)
    return bool(
        final_word is not None
        and final_word.group(1).casefold() in _CONTINUATION_FINAL_WORDS
    )


def _looks_like_abbreviation(sentence_prefix: str) -> bool:
    value = sentence_prefix.rstrip("\"”’')]").rstrip()
    token_match = re.search(r"(?:^|\s)([^\s]+)$", value)
    if token_match is None:
        return False
    token = token_match.group(1)
    if token.casefold() in _COMMON_ABBREVIATIONS:
        return True
    return bool(
        re.fullmatch(r"(?:[A-Z]\.){1,4}", token)
        or re.fullmatch(r"[A-Z]\.", token)
    )


def _sentence_units(text: str) -> list[str]:
    """Split normalized page prose at complete sentence/utterance ends."""

    value = " ".join(str(text or "").split())
    if not value:
        return [""]
    units: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(value):
        end = match.end()
        candidate = value[start:end].strip()
        if not candidate:
            continue
        if match.group(0).rstrip("\"”’')]") == "." and (
            _looks_like_abbreviation(candidate)
        ):
            continue
        units.append(candidate)
        start = end
    tail = value[start:].strip()
    if tail:
        units.append(tail)
    return units or [value]


def _wrapped_page_text(
    text: str,
    body_font: ImageFont.FreeTypeFont,
) -> str:
    return "\n".join(_wrap_text(text, body_font, max_width=924))


def _oversized_sentence_pages(
    sentence: str,
    body_font: ImageFont.FreeTypeFont,
) -> list[str]:
    """Prefer clause boundaries before a last-resort measured line split."""

    clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY_RE.split(sentence)
        if clause.strip()
    ]
    if len(clauses) > 1:
        result: list[str] = []
        pending = ""
        for clause in clauses:
            candidate = f"{pending} {clause}".strip()
            candidate_lines = _wrap_text(
                candidate,
                body_font,
                max_width=924,
            )
            if len(candidate_lines) <= _MAX_BODY_LINES:
                pending = candidate
                continue
            if pending:
                result.append(_wrapped_page_text(pending, body_font))
                pending = ""
            clause_lines = _wrap_text(clause, body_font, max_width=924)
            if len(clause_lines) <= _MAX_BODY_LINES:
                pending = clause
                continue
            result.extend(
                "\n".join(chunk)
                for chunk in _chunks(clause_lines, _MAX_BODY_LINES)
            )
        if pending:
            result.append(_wrapped_page_text(pending, body_font))
        return result

    lines = _wrap_text(sentence, body_font, max_width=924)
    return [
        "\n".join(chunk)
        for chunk in _chunks(lines, _MAX_BODY_LINES)
    ]


def _paginate_page_text(
    text: str,
    body_font: ImageFont.FreeTypeFont,
) -> list[str]:
    """Fit a semantic page while preserving ordinary sentence boundaries."""

    result: list[str] = []
    pending = ""
    for sentence in _sentence_units(text):
        candidate = f"{pending} {sentence}".strip()
        candidate_lines = _wrap_text(candidate, body_font, max_width=924)
        if len(candidate_lines) <= _MAX_BODY_LINES:
            pending = candidate
            continue
        if pending:
            result.append(_wrapped_page_text(pending, body_font))
            pending = ""
        sentence_lines = _wrap_text(sentence, body_font, max_width=924)
        if len(sentence_lines) <= _MAX_BODY_LINES:
            pending = sentence
            continue
        result.extend(_oversized_sentence_pages(sentence, body_font))
    if pending:
        result.append(_wrapped_page_text(pending, body_font))
    return result or [""]


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    paragraphs = str(text or "").splitlines() or [""]
    lines: list[str] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        if not words:
            if lines and paragraph_index < len(paragraphs) - 1:
                lines.append("")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if _text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            pieces = _split_long_word(draw, word, font, max_width)
            if pieces:
                lines.extend(pieces[:-1])
                current = pieces[-1]
        if current:
            lines.append(current)
    return lines or [""]


def _split_long_word(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    pieces: list[str] = []
    current = ""
    for character in word:
        candidate = current + character
        if current and _text_width(draw, candidate, font) > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start:start + size])


def _compose_card(
    stage: Image.Image,
    page: VisualNovelPage,
    *,
    index: int,
    count: int,
    fonts: _CardFonts,
) -> Image.Image:
    card = stage.copy().convert("RGBA")
    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cyan = (157, 231, 245, 255)
    panel = (12, 22, 43, 224)
    panel_box = (20, 396, 1004, 558)
    draw.rounded_rectangle(
        panel_box,
        radius=4,
        fill=panel,
        outline=cyan,
        width=2,
    )

    body_y = 423
    if page.kind == "dialogue":
        display_speaker = _ellipsize(
            draw,
            page.speaker,
            fonts.speaker,
            max_width=_SPEAKER_NAME_MAX_WIDTH,
        )
        speaker_width = min(
            950,
            max(188, _text_width(draw, display_speaker, fonts.speaker) + 48),
        )
        name_box = (34, 354, 34 + speaker_width, 408)
        draw.rounded_rectangle(
            name_box,
            radius=4,
            fill=(11, 25, 48, 238),
            outline=cyan,
            width=2,
        )
        draw.text(
            (55, 366),
            display_speaker,
            font=fonts.speaker,
            fill=cyan,
            stroke_width=1,
            stroke_fill=(4, 12, 26, 255),
        )

    draw.multiline_text(
        (50, body_y),
        page.text,
        font=fonts.body,
        fill=(246, 249, 255, 255),
        spacing=5,
        stroke_width=1,
        stroke_fill=(3, 8, 18, 245),
    )
    counter = f"{index} / {count}"
    counter_width = _text_width(draw, counter, fonts.counter)
    draw.text(
        (958 - counter_width, 532),
        counter,
        font=fonts.counter,
        fill=(196, 220, 231, 255),
    )
    draw.polygon(
        ((973, 532), (989, 532), (981, 542)),
        fill=cyan,
    )
    return Image.alpha_composite(card, overlay).convert("RGB")


def _ellipsize(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "…"
    candidate = text
    while candidate and _text_width(draw, candidate + suffix, font) > max_width:
        candidate = candidate[:-1]
    return candidate.rstrip() + suffix


def _transcript(pages: Sequence[VisualNovelPage]) -> str:
    paragraphs: list[str] = []
    for page in pages:
        if page.kind == "dialogue":
            paragraphs.append(f"{page.speaker}: {page.text.replace(chr(10), ' ')}")
        else:
            paragraphs.append(page.text.replace("\n", " "))
    return "\n\n".join(paragraphs)
