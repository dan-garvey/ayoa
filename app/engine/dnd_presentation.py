from __future__ import annotations

from typing import Any


def combat_status_lines(
    view: Any,
    *,
    markdown: bool = False,
    include_map: bool = True,
    max_chars: int | None = None,
) -> list[str]:
    if not getattr(view, "active", False):
        return [getattr(view, "message", "") or "No active combat."]

    current_id = str(getattr(view, "current_participant_id", "") or "")
    lines: list[str] = []
    round_number = int(getattr(view, "round_number", 0) or 0)
    turn_number = int(getattr(view, "turn_number", 0) or 0)
    if round_number:
        header = f"Round {round_number}"
        if turn_number:
            header += f" · Turn {turn_number}"
        lines.append(header)
    message = str(getattr(view, "message", "") or "").strip()
    if message:
        lines.append(message)

    participants = list(getattr(view, "participants", []) or [])
    for participant in participants:
        lines.append(combat_participant_line(
            participant,
            current_id=current_id,
            markdown=markdown,
        ))
    if not participants:
        lines.append("(no participants)")

    if include_map:
        append_limited_lines(
            lines,
            getattr(view, "map_lines", ()) or (),
            max_chars=max_chars,
            truncation="... map truncated.",
        )
    return lines


def combat_participant_line(
    participant: Any,
    *,
    current_id: str = "",
    markdown: bool = False,
) -> str:
    character_id = str(getattr(participant, "character_id", "") or "")
    name = str(getattr(participant, "name", "") or "") or character_id
    marker = ">" if (
        bool(getattr(participant, "current", False))
        or (character_id and character_id == current_id)
    ) else "-"

    bits: list[str] = []
    hp_current = getattr(participant, "hp_current", None)
    hp_max = getattr(participant, "hp_max", None)
    hp_temporary = int(getattr(participant, "hp_temporary", 0) or 0)
    if hp_current is not None:
        hp_text = str(hp_current)
        if hp_max is not None:
            hp_text += f"/{hp_max}"
        if hp_temporary:
            hp_text += f" (+{hp_temporary})"
        bits.append(f"HP {hp_text}")
    armor_class = getattr(participant, "armor_class", None)
    if armor_class is not None:
        bits.append(f"AC {armor_class}")
    initiative = getattr(participant, "initiative", None)
    if initiative is not None:
        bits.append(f"Init {initiative}")

    defeat_state = str(getattr(participant, "defeat_state", "") or "active")
    if defeat_state != "active":
        state = defeat_state
        if state == "down":
            state += (
                f" ({int(getattr(participant, 'death_save_successes', 0) or 0)}S/"
                f"{int(getattr(participant, 'death_save_failures', 0) or 0)}F)"
            )
        bits.append(state)

    conditions = list(getattr(participant, "conditions", []) or [])
    if conditions:
        bits.append(", ".join(str(condition) for condition in conditions))
    active_effects = list(getattr(participant, "active_effects", []) or [])
    if active_effects:
        bits.append(f"Effects: {', '.join(str(effect) for effect in active_effects)}")
    pending = str(getattr(participant, "pending_initiating_action", "") or "")
    if bool(getattr(participant, "current", False)) and pending:
        bits.append(f"Declared: {pending}")

    if markdown:
        label = f"**{name}** (`{character_id}`)"
    else:
        label = f"{name} ({character_id})"
    suffix = f" - {'; '.join(bits)}" if bits else ""
    return f"{marker} {label}{suffix}"


def append_limited_lines(
    lines: list[str],
    extra_lines: Any,
    *,
    max_chars: int | None = None,
    truncation: str = "... truncated.",
) -> None:
    clean = [str(line).strip() for line in extra_lines if str(line).strip()]
    if not clean:
        return
    lines.append("")
    for line in clean:
        if max_chars is not None and len("\n".join([*lines, line])) > max_chars:
            if len("\n".join([*lines, truncation])) <= max_chars:
                lines.append(truncation)
            break
        lines.append(line)


def currency_line(currency: dict[str, Any], *, separator: str = ", ") -> str:
    parts = []
    for key in ("pp", "gp", "ep", "sp", "cp"):
        value = int_or(currency.get(key), 0)
        if value:
            parts.append(f"{value} {key}")
    return separator.join(parts)


def inventory_item_line(
    item: dict[str, Any],
    *,
    bullet: bool = False,
    include_id: bool = True,
    include_attuned: bool = False,
) -> str:
    qty = int_or(item.get("quantity"), 1)
    prefix = f"{qty}x " if qty != 1 else ""
    kind = str(item.get("kind") or "").replace("_", " ")
    item_id = str(item.get("id") or item.get("item_id") or "")
    suffix_bits = [kind]
    if include_attuned and item.get("attuned"):
        suffix_bits.append("attuned")
    if include_id:
        suffix_bits.append(item_id)
    suffix = " (" + ", ".join(bit for bit in suffix_bits if bit) + ")"
    bullet_prefix = "- " if bullet else ""
    return (
        f"{bullet_prefix}{prefix}{item.get('name') or 'Item'}"
        f"{suffix if suffix != ' ()' else ''}"
    )


def loot_item_line(item: Any) -> str:
    qty = int_or(getattr(item, "quantity", 1), 1)
    prefix = f"{qty}x " if qty != 1 else ""
    kind = str(getattr(item, "kind", "") or "").replace("_", " ")
    suffix = f" ({kind})" if kind else ""
    notes = str(getattr(item, "notes", "") or "").strip()
    if notes:
        suffix += f" - {notes}"
    return f"{prefix}{getattr(item, 'name', 'Item')}{suffix}"


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
