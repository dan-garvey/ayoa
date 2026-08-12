from __future__ import annotations

import base64
import hashlib
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from app.engine.content_asset_bytes import (
    AssetByteResolutionError,
    SAFE_IMAGE_MIME_EXTENSIONS,
    resolve_asset_bytes,
)
from app.engine.content_assets import load_asset_catalog
from app.engine.player_media import PlayerMediaBytes
from app.schemas.content_pack import ContentImageAsset, SafeAssetRevealPayload


logger = logging.getLogger(__name__)

DEFAULT_CLI_ASSET_CACHE_ROOT = Path("app/storage/runtime/asset_cache")
_PACK_DB_METADATA_KEYS = ("db_path", "pack_path", "sqlite_path", "content_db_path")
_MEDIA_ROOT_METADATA_KEYS = (
    "media_root",
    "media_roots",
    "asset_media_root",
    "asset_media_roots",
)
_CACHE_ROOT_METADATA_KEYS = (
    "cache_root",
    "cache_roots",
    "asset_cache_root",
    "asset_cache_roots",
)


class TerminalImageBackend(Protocol):
    name: str

    def is_supported(self) -> bool:
        ...

    def render(self, image: "PreparedCliImageReveal") -> None:
        ...


@dataclass(frozen=True)
class CliImageDisplayOptions:
    cache_root: Path = DEFAULT_CLI_ASSET_CACHE_ROOT
    show_export_path: bool = False


@dataclass(frozen=True)
class PreparedCliImageReveal:
    pov_character_id: str
    cache_path: Path
    filename: str
    mime_type: str
    data: bytes
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class CliImageDisplayResult:
    pov_character_id: str
    displayed: bool = False
    degraded: bool = False
    error_code: str = ""
    backend_name: str = ""
    export_path: Path | None = None


class Iterm2InlineImageBackend:
    """Emit the iTerm2 inline image protocol used by iTerm2 and WezTerm."""

    name = "iterm2-inline-image"

    def __init__(
        self,
        *,
        output: BinaryIO | None = None,
        environ: Mapping[str, str] | None = None,
        is_tty: bool | None = None,
    ) -> None:
        self.output = output or sys.stdout.buffer
        self.environ = dict(os.environ if environ is None else environ)
        self._is_tty = sys.stdout.isatty() if is_tty is None else is_tty

    def is_supported(self) -> bool:
        if not self._is_tty:
            return False
        term = self.environ.get("TERM", "").lower()
        if not term or term == "dumb":
            return False
        if self.environ.get("TMUX"):
            return False
        term_program = self.environ.get("TERM_PROGRAM", "")
        return term_program in {"iTerm.app", "WezTerm"} or bool(
            self.environ.get("WEZTERM_PANE")
        )

    def render(self, image: PreparedCliImageReveal) -> None:
        safe_name = base64.b64encode(image.filename.encode("utf-8")).decode("ascii")
        data = base64.b64encode(image.data).decode("ascii")
        payload = (
            f"\x1b]1337;File=name={safe_name};inline=1;"
            f"size={image.byte_count};preserveAspectRatio=1:{data}\a"
        )
        self.output.write(payload.encode("ascii"))
        self.output.flush()


class CliImageDisplayRenderer:
    def __init__(
        self,
        *,
        backend: TerminalImageBackend | None = None,
        options: CliImageDisplayOptions | None = None,
    ) -> None:
        self.backend = backend or Iterm2InlineImageBackend()
        self.options = options or CliImageDisplayOptions()

    @classmethod
    def from_environment(
        cls,
        *,
        show_export_path: bool = False,
        cache_root: str | Path = DEFAULT_CLI_ASSET_CACHE_ROOT,
    ) -> "CliImageDisplayRenderer":
        return cls(
            options=CliImageDisplayOptions(
                cache_root=Path(cache_root),
                show_export_path=show_export_path,
            )
        )

    def prepare_reveals(
        self,
        response: Any,
        *,
        ckpt: Any,
        session_id: str,
        character_ids: set[str],
    ) -> dict[str, list[PreparedCliImageReveal | CliImageDisplayResult]]:
        per_pov = getattr(response, "per_player_asset_reveals", None) or {}
        if not isinstance(per_pov, Mapping) or not per_pov:
            return {}

        sources = _asset_runtime_sources(ckpt)
        prepared: dict[str, list[PreparedCliImageReveal | CliImageDisplayResult]] = {}
        for raw_cid, payloads in per_pov.items():
            cid = str(raw_cid).strip()
            if not cid or cid not in character_ids:
                continue
            if not isinstance(payloads, Sequence):
                continue
            items: list[PreparedCliImageReveal | CliImageDisplayResult] = []
            for payload_value in payloads:
                try:
                    payload = _coerce_payload(payload_value)
                    item = self._prepare_payload(
                        payload,
                        ckpt_sources=sources,
                        session_id=session_id,
                        pov_character_id=cid,
                    )
                except AssetByteResolutionError as exc:
                    logger.warning(
                        "cli image reveal resolution failed: code=%s",
                        exc.code,
                    )
                    item = CliImageDisplayResult(
                        pov_character_id=cid,
                        error_code="resolution_failed",
                        backend_name=self.backend.name,
                    )
                except Exception:
                    logger.exception("cli image reveal preparation failed")
                    item = CliImageDisplayResult(
                        pov_character_id=cid,
                        error_code="preparation_failed",
                        backend_name=self.backend.name,
                    )
                items.append(item)
            if items:
                prepared[cid] = items
        return prepared

    def render_prepared(
        self,
        item: PreparedCliImageReveal | CliImageDisplayResult,
    ) -> CliImageDisplayResult:
        if isinstance(item, CliImageDisplayResult):
            return item
        if not self.backend.is_supported():
            return CliImageDisplayResult(
                pov_character_id=item.pov_character_id,
                degraded=True,
                error_code="unsupported_terminal",
                backend_name=self.backend.name,
                export_path=(
                    item.cache_path if self.options.show_export_path else None
                ),
            )
        try:
            self.backend.render(item)
        except Exception:
            logger.exception(
                "cli image backend render failed: backend=%s",
                self.backend.name,
            )
            return CliImageDisplayResult(
                pov_character_id=item.pov_character_id,
                degraded=True,
                error_code="render_failed",
                backend_name=self.backend.name,
                export_path=(
                    item.cache_path if self.options.show_export_path else None
                ),
            )
        return CliImageDisplayResult(
            pov_character_id=item.pov_character_id,
            displayed=True,
            backend_name=self.backend.name,
        )

    def prepare_generated(
        self,
        media: PlayerMediaBytes,
        *,
        session_id: str,
        pov_character_id: str,
        cache_root: str | Path | None = None,
    ) -> PreparedCliImageReveal:
        """Prepare an already validated runtime illustration for display."""

        cache_path = write_cli_safe_asset_cache(
            media,
            session_id=session_id,
            cache_root=cache_root or self.options.cache_root,
        )
        return PreparedCliImageReveal(
            pov_character_id=pov_character_id,
            cache_path=cache_path,
            filename=cache_path.name,
            mime_type=media.mime_type,
            data=media.data,
            sha256=media.sha256,
            byte_count=media.byte_count,
        )

    def _prepare_payload(
        self,
        payload: SafeAssetRevealPayload,
        *,
        ckpt_sources: Mapping[str, "_PackAssetRuntimeSource"],
        session_id: str,
        pov_character_id: str,
    ) -> PreparedCliImageReveal:
        source = ckpt_sources.get(payload.pack_id.strip())
        if source is None:
            raise AssetByteResolutionError(
                "missing_pack_catalog",
                pack_id=payload.pack_id,
                asset_id=payload.asset_id,
            )
        asset = source.assets.get(_asset_key(payload.pack_id, payload.asset_id))
        if asset is None:
            raise AssetByteResolutionError(
                "missing_asset_catalog_row",
                pack_id=payload.pack_id,
                asset_id=payload.asset_id,
            )
        resolved = resolve_asset_bytes(
            payload,
            asset,
            media_roots={payload.pack_id: source.media_roots},
            cache_roots={payload.pack_id: source.cache_roots},
        )
        cache_path = write_cli_safe_asset_cache(
            resolved,
            session_id=session_id,
            cache_root=self.options.cache_root,
        )
        return PreparedCliImageReveal(
            pov_character_id=pov_character_id,
            cache_path=cache_path,
            filename=cache_path.name,
            mime_type=resolved.mime_type,
            data=resolved.data,
            sha256=resolved.sha256,
            byte_count=resolved.byte_count,
        )


def write_cli_safe_asset_cache(
    resolved: PlayerMediaBytes,
    *,
    session_id: str,
    cache_root: str | Path = DEFAULT_CLI_ASSET_CACHE_ROOT,
) -> Path:
    extension = SAFE_IMAGE_MIME_EXTENSIONS[resolved.mime_type]
    session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    root = Path(cache_root) / f"session-{session_key}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    path = root / f"{resolved.sha256}{extension}"
    if not path.exists() or path.read_bytes() != resolved.data:
        path.write_bytes(resolved.data)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path


@dataclass(frozen=True)
class _PackAssetRuntimeSource:
    assets: dict[str, ContentImageAsset]
    media_roots: tuple[Path, ...]
    cache_roots: tuple[Path, ...]


def _asset_runtime_sources(ckpt: Any) -> dict[str, _PackAssetRuntimeSource]:
    session = getattr(ckpt, "session", None)
    content_state = getattr(session, "content_state", {}) if session else {}
    if not isinstance(content_state, Mapping):
        return {}
    sources: dict[str, _PackAssetRuntimeSource] = {}
    for pack_key, pack_state in content_state.items():
        pack_id = _pack_id(pack_key, pack_state)
        if not pack_id:
            continue
        metadata = _metadata(pack_state)
        db_path = _first_path(metadata, _PACK_DB_METADATA_KEYS)
        if db_path is None:
            continue
        assets = load_asset_catalog(db_path, pack_id=pack_id)
        if not assets:
            continue
        media_roots = [*_paths_from_metadata(metadata, _MEDIA_ROOT_METADATA_KEYS)]
        cache_roots = [*_paths_from_metadata(metadata, _CACHE_ROOT_METADATA_KEYS)]
        media_roots.extend(_default_media_roots(db_path))
        sources[pack_id] = _PackAssetRuntimeSource(
            assets=assets,
            media_roots=tuple(dict.fromkeys(path.resolve() for path in media_roots)),
            cache_roots=tuple(dict.fromkeys(path.resolve() for path in cache_roots)),
        )
    return sources


def _coerce_payload(value: Any) -> SafeAssetRevealPayload:
    if isinstance(value, SafeAssetRevealPayload):
        return value
    if isinstance(value, Mapping):
        return SafeAssetRevealPayload(**dict(value))
    raise AssetByteResolutionError("invalid_payload")


def _metadata(pack_state: Any) -> Mapping[str, Any]:
    raw = (
        pack_state.get("metadata")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "metadata", None)
    )
    return raw if isinstance(raw, Mapping) else {}


def _pack_id(pack_key: Any, pack_state: Any) -> str:
    raw = (
        pack_state.get("pack_id")
        if isinstance(pack_state, Mapping)
        else getattr(pack_state, "pack_id", "")
    )
    return str(raw or pack_key or "").strip()


def _first_path(metadata: Mapping[str, Any], keys: Sequence[str]) -> Path | None:
    for key in keys:
        raw = metadata.get(key)
        if raw:
            return Path(str(raw)).expanduser()
    return None


def _paths_from_metadata(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str | Path):
            values = [raw]
        elif isinstance(raw, Sequence):
            values = list(raw)
        else:
            continue
        for value in values:
            if value:
                paths.append(Path(str(value)).expanduser())
    return paths


def _default_media_roots(db_path: Path) -> tuple[Path, ...]:
    parent = db_path.expanduser().parent
    return (
        parent / "media",
        parent / "asset_media",
        parent / "assets",
    )


def _asset_key(pack_id: str, asset_id: str) -> str:
    return f"{pack_id.strip()}::{asset_id.strip()}"
