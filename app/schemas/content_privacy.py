from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


REDACTED_IMPORT_SENTINEL = "[redacted private module material]"

FORBIDDEN_MODULE_METADATA_KEYS = {
    "asset_cache_root",
    "asset_cache_roots",
    "asset_media_root",
    "asset_media_roots",
    "cache_root",
    "cache_roots",
    "content_db_path",
    "db_path",
    "delivery_ref",
    "dm_notes",
    "file_path",
    "hidden_labels",
    "local_path",
    "media_root",
    "media_roots",
    "pack_path",
    "path",
    "player_display_payload",
    "protected_excerpt",
    "raw_bytes",
    "raw_ocr",
    "raw_source_path",
    "raw_text",
    "sqlite_path",
    "source_path",
    "source_ref",
}

_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)+[^/\s]+")
_ASSET_REF_RE = re.compile(r"\basset://[^\s\"')\]}]+", re.IGNORECASE)
_FILE_URI_RE = re.compile(r"\bfile://[^\s\"')\]}]+", re.IGNORECASE)
_IMAGE_PAYLOAD_RE = re.compile(
    r"\bdata:image[^\s\"')\]}]*|\b(?:image_bytes|input_image|image_url)\b",
    re.IGNORECASE,
)
_SOURCE_PDF_RE = re.compile(r"\b[^\s\"']+\.pdf\b", re.IGNORECASE)
_PRIVATE_STORAGE_RE = re.compile(r"\bprivate_extractions/[^\s\"')\]}]*", re.IGNORECASE)
_FORBIDDEN_FIELD_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(key) for key in sorted(FORBIDDEN_MODULE_METADATA_KEYS))
    + r")\b(?:\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+))?",
    re.IGNORECASE,
)
_UNSAFE_TEXT_PATTERNS = (
    _FILE_URI_RE,
    _ASSET_REF_RE,
    _IMAGE_PAYLOAD_RE,
    _PRIVATE_STORAGE_RE,
    _SOURCE_PDF_RE,
    _FORBIDDEN_FIELD_RE,
    _ABSOLUTE_PATH_RE,
)


def contains_imported_asset_sentinel(value: str) -> bool:
    return any(pattern.search(value or "") for pattern in _UNSAFE_TEXT_PATTERNS)


def redact_imported_asset_text(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    for pattern in _UNSAFE_TEXT_PATTERNS:
        text = pattern.sub(REDACTED_IMPORT_SENTINEL, text)
    return " ".join(text.split())


def sanitize_player_safe_text(
    value: str,
    *,
    protected_terms: Sequence[str] = (),
) -> str:
    text = " ".join((value or "").split())
    if not text:
        return ""
    if contains_imported_asset_sentinel(text):
        return ""
    if any(term and term in text for term in protected_terms):
        return ""
    return text


def sanitize_module_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).strip()
            if key_text.lower() in FORBIDDEN_MODULE_METADATA_KEYS:
                continue
            sanitized_value = sanitize_module_metadata(item)
            if sanitized_value is not None:
                sanitized[key_text] = sanitized_value
        return sanitized
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := sanitize_module_metadata(item)) is not None
        ]
    if isinstance(value, str):
        if contains_imported_asset_sentinel(value):
            return None
        return " ".join(value.split())
    return value
