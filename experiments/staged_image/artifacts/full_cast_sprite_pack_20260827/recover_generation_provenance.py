#!/usr/bin/env python3
"""Recover exact built-in image-generation inputs from Codex rollout logs.

The experiment was intentionally executed one candidate at a time.  Each
successful generation call was followed by a copy from Codex's generated image
store into this artifact tree.  This script pairs those adjacent operations,
preserves the exact prompt string and reference list, and hash-validates both
the generated source and the landed artifact.

This is coding-time provenance tooling only.  It is not imported by Ayoa and
does not add image reading to the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


IMAGEGEN_NAME = "tools.image_gen__imagegen"
GENERATED_RE = re.compile(
    r"/home/dan/\.codex/generated_images/[^\s\\\"'`;]+\.png"
)
DESTINATION_RE = re.compile(
    r"experiments/staged_image/artifacts/"
    r"(?:full_cast_sprite_pack_20260827|mirelle_rowan_sprite_pack_v2_20260827)/"
    r"generation_raw/[^\s\\\"'`;]+\.png"
)


@dataclass(frozen=True)
class ExecEvent:
    line: int
    timestamp: str | None
    source: str


@dataclass(frozen=True)
class Invocation:
    prompt: str | None
    prompt_exact: bool
    prompt_expression: str
    references: tuple[str, ...]
    reference_expression: str | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(items: Iterator[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def skip_quoted(source: str, start: int) -> int:
    """Return the first offset after one JS quote/template literal."""
    quote = source[start]
    index = start + 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    return len(source)


def skip_comment(source: str, start: int) -> int:
    if source.startswith("//", start):
        end = source.find("\n", start + 2)
        return len(source) if end < 0 else end + 1
    if source.startswith("/*", start):
        end = source.find("*/", start + 2)
        return len(source) if end < 0 else end + 2
    return start


def code_occurrences(source: str, needle: str) -> list[int]:
    offsets: list[int] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "'\"`":
            index = skip_quoted(source, index)
            continue
        comment_end = skip_comment(source, index)
        if comment_end != index:
            index = comment_end
            continue
        if source.startswith(needle, index):
            offsets.append(index)
            index += len(needle)
            continue
        index += 1
    return offsets


def balanced_contents(source: str, opening: int, left: str, right: str) -> tuple[str, int]:
    if source[opening] != left:
        raise ValueError(f"expected {left!r} at offset {opening}")
    depth = 1
    index = opening + 1
    while index < len(source):
        char = source[index]
        if char in "'\"`":
            index = skip_quoted(source, index)
            continue
        comment_end = skip_comment(source, index)
        if comment_end != index:
            index = comment_end
            continue
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
        index += 1
    raise ValueError(f"unterminated {left}{right} starting at {opening}")


def decode_js_literal(source: str, start: int) -> tuple[str, int, bool]:
    quote = source[start]
    if quote not in "'\"`":
        raise ValueError("not a JS string literal")
    chars: list[str] = []
    exact = True
    index = start + 1
    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "v": "\v",
        "0": "\0",
        "\\": "\\",
        "'": "'",
        '"': '"',
        "`": "`",
        "$": "$",
    }
    while index < len(source):
        char = source[index]
        if char == quote:
            return "".join(chars), index + 1, exact
        if quote == "`" and source.startswith("${", index):
            exact = False
        if char != "\\":
            chars.append(char)
            index += 1
            continue
        index += 1
        if index >= len(source):
            exact = False
            break
        escaped = source[index]
        if escaped == "u" and index + 4 < len(source):
            token = source[index + 1 : index + 5]
            try:
                chars.append(chr(int(token, 16)))
                index += 5
                continue
            except ValueError:
                exact = False
        if escaped == "x" and index + 2 < len(source):
            token = source[index + 1 : index + 3]
            try:
                chars.append(chr(int(token, 16)))
                index += 3
                continue
            except ValueError:
                exact = False
        if escaped == "\n":
            index += 1
            continue
        chars.append(escapes.get(escaped, escaped))
        if escaped not in escapes:
            exact = False
        index += 1
    return "".join(chars), index, False


def skip_space(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def find_property_value(source: str, name: str) -> int | None:
    for offset in code_occurrences(source, name):
        before = source[offset - 1] if offset else ""
        after_index = offset + len(name)
        after = source[after_index] if after_index < len(source) else ""
        if (before.isalnum() or before in "_$") or (after.isalnum() or after in "_$"):
            continue
        cursor = skip_space(source, after_index)
        if cursor < len(source) and source[cursor] == ":":
            return skip_space(source, cursor + 1)
    return None


def resolve_variable_literal(script: str, name: str) -> tuple[str | None, bool]:
    declaration = re.compile(rf"\b(?:const|let|var)\s+{re.escape(name)}\s*=")
    for match in declaration.finditer(script):
        cursor = skip_space(script, match.end())
        if cursor < len(script) and script[cursor] in "'\"`":
            value, _, exact = decode_js_literal(script, cursor)
            return value, exact
    return None, False


def parse_prompt(argument: str, script: str) -> tuple[str | None, bool, str]:
    cursor = find_property_value(argument, "prompt")
    if cursor is None:
        return None, False, "unresolved prompt shorthand or dynamic expression"
    if argument[cursor] in "'\"`":
        value, end, exact = decode_js_literal(argument, cursor)
        return value, exact, argument[cursor:end]
    token = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", argument[cursor:])
    if token:
        name = token.group(0)
        value, exact = resolve_variable_literal(script, name)
        return value, exact, name
    expression = argument[cursor:].split(",", 1)[0].strip()
    return None, False, expression


def parse_string_array(source: str, cursor: int) -> tuple[str, ...]:
    cursor = skip_space(source, cursor)
    if cursor >= len(source) or source[cursor] != "[":
        return ()
    contents, _ = balanced_contents(source, cursor, "[", "]")
    values: list[str] = []
    index = 0
    while index < len(contents):
        if contents[index] in "'\"`":
            value, index, exact = decode_js_literal(contents, index)
            if exact and "${" not in value:
                values.append(value)
            continue
        index += 1
    return tuple(values)


def parse_references(argument: str, script: str) -> tuple[tuple[str, ...], str | None]:
    cursor = find_property_value(argument, "referenced_image_paths")
    if cursor is None:
        return (), None
    if argument[cursor] == "[":
        contents, end = balanced_contents(argument, cursor, "[", "]")
        return parse_string_array("[" + contents + "]", 0), argument[cursor:end]
    token = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", argument[cursor:])
    if not token:
        return (), argument[cursor:].split(",", 1)[0].strip()
    name = token.group(0)
    declaration = re.compile(rf"\b(?:const|let|var)\s+{re.escape(name)}\s*=")
    for match in declaration.finditer(script):
        value_cursor = skip_space(script, match.end())
        if value_cursor < len(script) and script[value_cursor] == "[":
            return parse_string_array(script, value_cursor), name
    return (), name


def invocations(script: str) -> list[Invocation]:
    result: list[Invocation] = []
    for offset in code_occurrences(script, IMAGEGEN_NAME):
        cursor = skip_space(script, offset + len(IMAGEGEN_NAME))
        if cursor >= len(script) or script[cursor] != "(":
            continue
        argument, _ = balanced_contents(script, cursor, "(", ")")
        prompt, exact, expression = parse_prompt(argument, script)
        references, reference_expression = parse_references(argument, script)
        result.append(
            Invocation(
                prompt=prompt,
                prompt_exact=exact and prompt is not None and "${" not in prompt,
                prompt_expression=expression,
                references=references,
                reference_expression=reference_expression,
            )
        )
    return result


def load_exec_events(path: Path) -> list[ExecEvent]:
    events: list[ExecEvent] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = item.get("payload", {})
            if not (
                item.get("type") == "response_item"
                and payload.get("type") == "custom_tool_call"
                and payload.get("name") == "exec"
            ):
                continue
            events.append(
                ExecEvent(
                    line=line_number,
                    timestamp=item.get("timestamp"),
                    source=payload.get("input", ""),
                )
            )
    return events


def image_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {
            "path": str(path),
            "sha256": sha256(path),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "bytes": path.stat().st_size,
        }


def reference_metadata(reference: str, repo_root: Path) -> dict[str, object]:
    path = Path(reference)
    candidates = [path]
    if path.is_absolute():
        for prefix in (Path("/home/dan/ayoa-worktrees/mirelle-rowan-sprite-pack"), Path("/home/dan/ayoa")):
            try:
                candidates.append(repo_root / path.relative_to(prefix))
            except ValueError:
                pass
    else:
        candidates.append(repo_root / path)
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    item: dict[str, object] = {"invocation_path": reference}
    if resolved is not None:
        item.update(image_metadata(resolved))
    else:
        item["exists"] = False
    return item


def recover_rollout(path: Path, repo_root: Path) -> list[dict[str, object]]:
    events = load_exec_events(path)
    generation_indexes = [
        index for index, event in enumerate(events) if invocations(event.source)
    ]
    recovered: list[dict[str, object]] = []
    for position, event_index in enumerate(generation_indexes):
        event = events[event_index]
        calls = invocations(event.source)
        next_index = (
            generation_indexes[position + 1]
            if position + 1 < len(generation_indexes)
            else len(events)
        )
        sources: list[str] = []
        destinations: list[str] = []
        for later in events[event_index + 1 : next_index]:
            if "/home/dan/.codex/generated_images/" not in later.source:
                continue
            sources.extend(GENERATED_RE.findall(later.source))
            destinations.extend(DESTINATION_RE.findall(later.source))
        sources = unique(iter(sources))
        destinations = unique(iter(destinations))
        association = (
            "exact_order"
            if len(calls) == len(sources) == len(destinations)
            else "count_mismatch"
        )
        for index, call in enumerate(calls):
            source = sources[index] if index < len(sources) else None
            destination = destinations[index] if index < len(destinations) else None
            record: dict[str, object] = {
                "source_rollout": path.name,
                "source_line": event.line,
                "timestamp": event.timestamp,
                "invocation_index": index,
                "association": association,
                "prompt_exact": call.prompt_exact,
                "prompt_expression": call.prompt_expression,
                "prompt": call.prompt,
                "prompt_sha256": (
                    hashlib.sha256(call.prompt.encode("utf-8")).hexdigest()
                    if call.prompt is not None
                    else None
                ),
                "references": [
                    reference_metadata(reference, repo_root)
                    for reference in call.references
                ],
                "reference_expression": call.reference_expression,
                "generated_source": source,
                "destination": destination,
            }
            if source and Path(source).exists():
                record["generated_source_metadata"] = image_metadata(Path(source))
            if destination:
                landed = repo_root / destination
                if landed.exists():
                    record["destination_metadata"] = image_metadata(landed)
                    if source and Path(source).exists():
                        record["source_destination_hash_match"] = (
                            sha256(Path(source)) == sha256(landed)
                        )
            recovered.append(record)
    return recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("rollouts", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    records: list[dict[str, object]] = []
    for rollout in args.rollouts:
        records.extend(recover_rollout(rollout, repo_root))
    records.sort(
        key=lambda record: (
            str(record.get("destination") or "~"),
            str(record["source_rollout"]),
            int(record["source_line"]),
            int(record["invocation_index"]),
        )
    )
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    exact_landed = [
        record
        for record in records
        if record.get("destination_metadata") and record.get("prompt_exact")
    ]
    landed = [record for record in records if record.get("destination_metadata")]
    hash_matches = [
        record for record in landed if record.get("source_destination_hash_match") is True
    ]
    print(f"records={len(records)}")
    print(f"landed={len(landed)}")
    print(f"exact_prompt_landed={len(exact_landed)}")
    print(f"source_destination_hash_matches={len(hash_matches)}")
    print(f"output={output}")
    print(f"output_sha256={sha256(output)}")


if __name__ == "__main__":
    main()
