"""File-building helpers shared by the test modules."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[list[object]], header: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path


def note(
    mrn: str,
    patient: str,
    dob: str,
    date: str,
    body: str,
    note_type: str = "Progress Note",
) -> str:
    return (
        f"=== NOTE ===\nMRN: {mrn}\nPatient: {patient}\nDOB: {dob}\n"
        f"Date: {date}\nType: {note_type}\n---\n{body}\n"
    )
