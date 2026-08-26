from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


REDACTED_IMPORT_SENTINEL = "[redacted private module material]"
PRIVATE_RUNTIME_METADATA_CONTEXT = "include_private_runtime_metadata"

FORBIDDEN_MODULE_METADATA_KEYS = {
    "action_palette",
    "actions",
    "asset_cache_root",
    "asset_cache_roots",
    "asset_media_root",
    "asset_media_roots",
    "cache_root",
    "cache_roots",
    "catalog",
    "catalog_build_hash",
    "content_pack_domain_catalog",
    "content_db_path",
    "cooldowns",
    "constraints",
    "db_path",
    "delivery_ref",
    "dm_notes",
    "domain_catalog",
    "engine_overlay",
    "escalation_thresholds",
    "file_path",
    "front_dossier_records",
    "front_dossiers",
    "front_resources",
    "goals",
    "hidden_labels",
    "hidden_plans",
    "hidden_resources",
    "imported_front_dossiers",
    "imported_fronts",
    "knowledge_channels",
    "knows",
    "local_path",
    "media_root",
    "media_roots",
    "minion_refs",
    "minions",
    "pack_path",
    "path",
    "player_display_payload",
    "pressure",
    "projection_hash",
    "projection_schema_version",
    "protected_excerpt",
    "raw_bytes",
    "raw_ocr",
    "raw_source_path",
    "raw_text",
    "resources",
    "restraints",
    "router_lookup_catalog",
    "router_knowledge_index",
    "router_knowledge_packets",
    "villains",
    "sqlite_path",
    "source_path",
    "source_ref",
}

_UNIX_PATH_SEGMENT = r"[^/\s\"')\]}>;,]+"
_WINDOWS_PATH_SEGMENT = r"[^\\/\s\"')\]}>;,]+"
_REPO_PATH_ROOTS = (
    r"app|tests|scripts|infra|experiments|audits|samples|stories|\.agents|"
    r"\.beads|\.cursor|private_extractions"
)
_PRIVATE_FILE_SUFFIXES = (
    r"bmp|cfg|ckpt|csv|db|gif|ini|jpe?g|jsonl?|log|md|pdf|png|py|"
    r"safetensors|sqlite3?|svg|toml|tsv|txt|webp|ya?ml"
)
_UNQUOTED_PRIVATE_FILE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?:/[^\r\n\"'<>|]*?|"
    rf"[A-Za-z]:[\\/][^\r\n\"'<>|]*?|\\\\[^\r\n\"'<>|]*?|"
    rf"(?:\.{{1,2}}[\\/])?(?:{_REPO_PATH_ROOTS})"
    rf"[\\/][^\r\n\"'<>|]*?)\.(?:{_PRIVATE_FILE_SUFFIXES})"
    rf"(?=$|[\s,;:)\]}}])",
    re.IGNORECASE,
)
_QUOTED_PRIVATE_PATH_RE = re.compile(
    rf"(?P<quote>[\"'])(?:/(?:[^/\r\n\"']+/[^\r\n\"']+|"
    rf"[^/\r\n\"']+\.(?:{_PRIVATE_FILE_SUFFIXES}))|"
    rf"[A-Za-z]:[\\/][^\r\n\"']+|"
    rf"\\\\[^\r\n\"']+|(?:\.{{1,2}}[\\/])?(?:{_REPO_PATH_ROOTS})"
    rf"[\\/][^\r\n\"']+)(?P=quote)",
    re.IGNORECASE,
)
_UNIX_ABSOLUTE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])/(?:{_UNIX_PATH_SEGMENT}/)+"
    rf"{_UNIX_PATH_SEGMENT}"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/]"
    rf"(?:{_WINDOWS_PATH_SEGMENT}[\\/])*{_WINDOWS_PATH_SEGMENT}"
    rf"|(?<![\\A-Za-z0-9_.-])\\\\{_WINDOWS_PATH_SEGMENT}\\"
    rf"(?:{_WINDOWS_PATH_SEGMENT}\\)*{_WINDOWS_PATH_SEGMENT}",
    re.IGNORECASE,
)
_REPO_INTERNAL_RELATIVE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])"
    rf"(?:\.{{1,2}}[\\/])?"
    rf"(?:{_REPO_PATH_ROOTS})"
    rf"[\\/](?:{_WINDOWS_PATH_SEGMENT}[\\/])*{_WINDOWS_PATH_SEGMENT}",
    re.IGNORECASE,
)
_ASSET_REF_RE = re.compile(r"\basset://[^\s\"')\]}]+", re.IGNORECASE)
_FILE_URI_RE = re.compile(r"\bfile://[^\s\"')\]}]+", re.IGNORECASE)
_IMAGE_PAYLOAD_RE = re.compile(
    r"\bdata:image[^\s\"')\]}]*|\b(?:image_bytes|input_image|image_url)\b",
    re.IGNORECASE,
)
_SOURCE_PDF_RE = re.compile(r"\b[^\s\"']+\.pdf\b", re.IGNORECASE)
_PRIVATE_STORAGE_RE = re.compile(
    r"\bprivate_extractions[\\/][^\s\"')\]}]*",
    re.IGNORECASE,
)
_FORBIDDEN_FIELD_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(key) for key in sorted(FORBIDDEN_MODULE_METADATA_KEYS))
    + r")\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+)",
    re.IGNORECASE,
)
_COMPACT_CONTENT_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_.:/@+-])"
    r"[A-Za-z][A-Za-z0-9_.-]*:"
    r"[A-Za-z][A-Za-z0-9_.\/-]+@(?:sha256:)?[A-Za-z0-9_.:-]+",
    re.IGNORECASE,
)
_CONTENT_RECORD_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:actor|agent_context|area|enc|front|handout|hazard|loc|map|stat|table|"
    r"treasure|trap)\.[A-Za-z0-9_.-]+"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_SHA256_HASH_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sha256:[A-Za-z0-9_.:-]{8,}"
    r"|(?<![A-Fa-f0-9])[A-Fa-f0-9]{64}(?![A-Fa-f0-9])",
    re.IGNORECASE,
)
_CONTENT_METADATA_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)"
    r"(?:Reviewed\s+)?(?:content\s+refs?|known_refs|content_hash(?:es)?|"
    r"pack_id|source_fingerprint|agent_context(?:_slice)?(?:_ref)?)"
    r"[^.!?\n]*(?:[.!?]|$)",
    re.IGNORECASE,
)
_UNSAFE_TEXT_PATTERNS = (
    _QUOTED_PRIVATE_PATH_RE,
    _UNQUOTED_PRIVATE_FILE_PATH_RE,
    _FILE_URI_RE,
    _ASSET_REF_RE,
    _IMAGE_PAYLOAD_RE,
    _PRIVATE_STORAGE_RE,
    _SOURCE_PDF_RE,
    _FORBIDDEN_FIELD_RE,
    _WINDOWS_ABSOLUTE_PATH_RE,
    _UNIX_ABSOLUTE_PATH_RE,
    _REPO_INTERNAL_RELATIVE_PATH_RE,
)
_UNSAFE_CONTENT_METADATA_PATTERNS = (
    _COMPACT_CONTENT_REF_RE,
    _CONTENT_RECORD_REF_RE,
    _SHA256_HASH_RE,
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


def redact_imported_content_metadata_text(
    value: str,
    *,
    protected_terms: Sequence[str] = (),
) -> str:
    """Remove source-tracking metadata from prose-facing agent context."""

    text = str(value or "")
    if not text:
        return ""
    protected = {str(item or "").strip() for item in protected_terms}
    for term in sorted(protected, key=len, reverse=True):
        if len(term) < 8:
            continue
        text = re.sub(re.escape(term), REDACTED_IMPORT_SENTINEL, text)
    for pattern in _UNSAFE_TEXT_PATTERNS:
        text = pattern.sub(REDACTED_IMPORT_SENTINEL, text)
    text = _CONTENT_METADATA_SENTENCE_RE.sub(" ", text)
    for pattern in _UNSAFE_CONTENT_METADATA_PATTERNS:
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
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return sanitize_module_metadata(value.model_dump(mode="json"))
        except TypeError:
            return sanitize_module_metadata(value.model_dump())
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


def should_include_private_runtime_metadata(context: Any) -> bool:
    return (
        isinstance(context, Mapping)
        and context.get(PRIVATE_RUNTIME_METADATA_CONTEXT) is True
    )
