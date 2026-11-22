"""Shared helpers for heterogeneous source parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_SENTINEL_STRINGS = {"", "n/a", "na", "none", "null", "unknown", "-", "--"}


def normalize_key(value: Any) -> str:
    """``"Date of Birth"`` / ``"dateOfBirth"`` -> ``"dateofbirth"``."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def pick(mapping: dict[str, Any], *aliases: str, default: Any = None) -> Any:
    """Case/format-insensitive lookup across several possible key names."""
    if not isinstance(mapping, dict):
        return default
    normalized = {normalize_key(key): value for key, value in mapping.items()}
    for alias in aliases:
        key = normalize_key(alias)
        if key in normalized:
            value = normalized[key]
            if not _is_sentinel(value):
                return value
    return default


def _is_sentinel(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _SENTINEL_STRINGS
    return False


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return [value]


def read_text(path: Path) -> str:
    """Read a text file, tolerating UTF-8 BOMs and legacy Windows encodings."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json_records(path: Path) -> tuple[list[Any], str | None]:
    """Load JSON, JSON-lines or a wrapped ``{"patients": [...]}`` export."""
    text = read_text(path)
    if not text.strip():
        return [], "file is empty"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try NDJSON before giving up.
        records: list[Any] = []
        errors = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
        if records:
            return records, f"{errors} malformed JSON line(s) skipped" if errors else None
        return [], f"invalid JSON: {exc.msg} at line {exc.lineno}"

    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        for key in ("patients", "records", "data", "entries", "items"):
            candidate = pick(payload, key)
            if isinstance(candidate, list):
                return candidate, None
        return [payload], None
    return [payload], "unexpected top-level JSON type"


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if not _is_sentinel(value):
            return value
    return None


def flatten_address(value: Any) -> dict[str, Any]:
    """Normalize flat or FHIR-ish address structures into flat fields."""
    if isinstance(value, dict):
        line = pick(value, "line", "address_line", "street", "address1", "text")
        if isinstance(line, list):
            line = ", ".join(str(part) for part in line if part)
        return {
            "address_line": line,
            "city": pick(value, "city"),
            "state": pick(value, "state", "province"),
            "postal_code": pick(value, "postal_code", "postalCode", "zip", "zipcode"),
        }
    if isinstance(value, str) and value.strip():
        return {"address_line": value.strip(), "city": None, "state": None, "postal_code": None}
    return {"address_line": None, "city": None, "state": None, "postal_code": None}


def truncate(value: Any, limit: int = 180) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def iter_chunks(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
