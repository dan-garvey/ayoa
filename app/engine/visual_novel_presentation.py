"""Deterministic classic-ADV card composition.

The narrator owns ordered page semantics. This module owns pixels, wrapping,
fallbacks, caching, and the transport-neutral deck manifest. It never sends
image data to an LLM and never changes the supplied scene plate.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from app.engine.player_media import PlayerMediaBytes
from app.schemas.narrator import VisualNovelPage


CARD_WIDTH = 1024
CARD_HEIGHT = 576
_RENDERER_VERSION = "classic-adv-v2-segmented-stages"
_DEFAULT_REGULAR_FONT = Path(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)
_DEFAULT_BOLD_FONT = Path(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)


@dataclass(frozen=True)
class VisualNovelCard:
    index: int
    count: int
    kind: str
    speaker: str
    text: str
    image_path: Path

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


@dataclass(frozen=True)
class VisualNovelDeckSection:
    """Ordered semantic pages resolved against one immutable stage plate."""

    pages: tuple[VisualNovelPage, ...]
    stage_path: str | Path | None = None
    stage_media: PlayerMediaBytes | None = None


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
        self.deck_root.mkdir(parents=True, exist_ok=True)
        self.regular_font_path = Path(regular_font_path)
        self.bold_font_path = Path(bold_font_path)

    def render_deck(
        self,
        sections: Sequence[VisualNovelDeckSection],
    ) -> VisualNovelDeck:
        if not sections:
            raise ValueError("visual-novel decks require at least one section")
        fonts = self._fonts()
        resolved_sections: list[
            tuple[Image.Image, str, bool, list[VisualNovelPage]]
        ] = []
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
            resolved_sections.append((
                stage,
                stage_sha256,
                used_neutral,
                _paginate_pages(section.pages, fonts.body),
            ))
        identity = {
            "renderer": _RENDERER_VERSION,
            "sections": [
                {
                    "stage_sha256": stage_sha256,
                    "pages": [
                        page.model_dump(mode="json") for page in physical_pages
                    ],
                }
                for _stage, stage_sha256, _used_neutral, physical_pages
                in resolved_sections
            ],
        }
        deck_id = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        deck_dir = self.deck_root / deck_id
        manifest_path = deck_dir / "manifest.json"
        if manifest_path.is_file():
            cached = self.load_deck(deck_id)
            if cached is not None:
                return cached

        deck_dir.mkdir(parents=True, exist_ok=True)
        count = sum(
            len(physical_pages)
            for _stage, _sha256, _used_neutral, physical_pages
            in resolved_sections
        )
        cards: list[VisualNovelCard] = []
        physical_pages: list[VisualNovelPage] = []
        index = 0
        for stage, _stage_sha256, _used_neutral, section_pages in resolved_sections:
            for page in section_pages:
                index += 1
                physical_pages.append(page)
                output_path = deck_dir / f"page-{index:03d}.png"
                card_image = _compose_card(
                    stage,
                    page,
                    index=index,
                    count=count,
                    fonts=fonts,
                )
                temporary = deck_dir / (
                    f".page-{index:03d}.{uuid.uuid4().hex}.tmp"
                )
                card_image.save(temporary, format="PNG", optimize=False)
                temporary.replace(output_path)
                cards.append(VisualNovelCard(
                    index=index,
                    count=count,
                    kind=page.kind,
                    speaker=page.speaker,
                    text=page.text,
                    image_path=output_path,
                ))

        transcript = _transcript(physical_pages)
        used_neutral = any(
            section_used_neutral
            for _stage, _sha256, section_used_neutral, _pages
            in resolved_sections
        )
        manifest = {
            "version": 1,
            "deck_id": deck_id,
            "used_neutral_stage": used_neutral,
            "transcript": transcript,
            "cards": [
                {
                    "index": card.index,
                    "count": card.count,
                    "kind": card.kind,
                    "speaker": card.speaker,
                    "text": card.text,
                    "filename": card.image_path.name,
                }
                for card in cards
            ],
        }
        temporary_manifest = deck_dir / (
            f".manifest.{uuid.uuid4().hex}.tmp"
        )
        temporary_manifest.write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        return VisualNovelDeck(
            deck_id=deck_id,
            cards=tuple(cards),
            transcript=transcript,
            manifest_path=manifest_path,
            used_neutral_stage=used_neutral,
        )

    def load_deck(self, deck_id: str) -> VisualNovelDeck | None:
        clean_id = str(deck_id or "").strip().lower()
        if (
            len(clean_id) != 64
            or any(character not in "0123456789abcdef" for character in clean_id)
        ):
            return None
        deck_dir = self.deck_root / clean_id
        manifest_path = deck_dir / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if payload.get("deck_id") != clean_id:
            return None
        cards: list[VisualNovelCard] = []
        for raw in payload.get("cards", []):
            filename = str(raw.get("filename") or "")
            if Path(filename).name != filename or not filename.endswith(".png"):
                return None
            image_path = deck_dir / filename
            if not image_path.is_file():
                return None
            cards.append(VisualNovelCard(
                index=int(raw["index"]),
                count=int(raw["count"]),
                kind=str(raw["kind"]),
                speaker=str(raw.get("speaker") or ""),
                text=str(raw["text"]),
                image_path=image_path,
            ))
        if not cards:
            return None
        return VisualNovelDeck(
            deck_id=clean_id,
            cards=tuple(cards),
            transcript=str(payload.get("transcript") or ""),
            manifest_path=manifest_path,
            used_neutral_stage=bool(payload.get("used_neutral_stage", False)),
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


@dataclass(frozen=True)
class _CardFonts:
    body: ImageFont.FreeTypeFont
    speaker: ImageFont.FreeTypeFont
    counter: ImageFont.FreeTypeFont


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
    for page in pages:
        lines = _wrap_text(page.text, body_font, max_width=924)
        chunks = list(_chunks(lines, 4))
        for chunk in chunks:
            result.append(VisualNovelPage(
                kind=page.kind,
                speaker=page.speaker,
                text="\n".join(chunk),
            ))
    return result


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
            max_width=332,
        )
        speaker_width = min(
            380,
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
