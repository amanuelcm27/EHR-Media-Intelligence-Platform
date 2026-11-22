"""Plain-text / scanned-note adapter.

Expected shape (headers are optional but recommended)::

    === NOTE ===
    MRN: MRN-100234
    Patient: Vance, Eleanor
    DOB: 02/11/1948
    Date: 2024-03-02
    Type: Discharge Summary
    Author: Dr. A. Patel
    ---
    Chief Complaint: worsening shortness of breath ...

Multiple notes may live in one file. Notes without an MRN are still parsed;
the pipeline attempts to identify them from name + DOB or quarantines them
with an explicit report entry.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import normalize_key, read_text
from .raw import IngestionIssue, RawDocument, RawPatient, SourceResult

_BLOCK_SEPARATOR = re.compile(
    r"(?m)^\s*(?:={3,}\s*(?:NOTE|DOCUMENT|RECORD)?\s*={3,}"
    r"|-{3,}\s*(?:NOTE|DOCUMENT)\s*-{3,}"
    r"|\[(?:NOTE|DOCUMENT)\]\s*)\s*$"
)
_HEADER_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /_#.()\-]{0,40})\s*:\s*(.+?)\s*$")
_BODY_SEPARATOR = re.compile(r"^\s*-{3,}\s*$")
_HEADER_KEYS = {
    "mrn",
    "medicalrecordnumber",
    "medicalrecordno",
    "patient",
    "patientname",
    "name",
    "dob",
    "dateofbirth",
    "birthdate",
    "date",
    "dateofservice",
    "dos",
    "type",
    "documenttype",
    "notetype",
    "author",
    "provider",
    "clinician",
    "title",
    "subject",
    "encounter",
    "encounterid",
    "visit",
    "facility",
    "sex",
    "gender",
    "age",
    "language",
}


class TextNotesSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = self.path.name

    def load(self) -> SourceResult:
        result = SourceResult()
        text = read_text(self.path)
        if not text.strip():
            result.issues.append(
                IngestionIssue(self.name, "$", "text_empty", "Text file is empty", "warning")
            )
            return result

        blocks = [block for block in _BLOCK_SEPARATOR.split(text) if block.strip()]
        if not blocks:
            blocks = [text]

        groups: dict[str, list[tuple[dict[str, str], str, str]]] = {}
        for index, block in enumerate(blocks):
            row_ref = f"{self.name}#note{index + 1}"
            header, body = self._parse_block(block)
            if not body.strip():
                result.issues.append(
                    IngestionIssue(
                        self.name,
                        row_ref,
                        "text_empty_body",
                        "Note block has no body text",
                        "warning",
                    )
                )
                continue
            mrn = header.get("mrn")
            patient_name = header.get("patient") or header.get("patientname") or header.get("name")
            dob = header.get("dob") or header.get("dateofbirth") or header.get("birthdate")
            key = (
                f"mrn:{mrn.strip()}"
                if mrn
                else f"name:{normalize_key(patient_name or '')}|{dob or ''}|{index}"
            )
            if not mrn:
                result.issues.append(
                    IngestionIssue(
                        self.name,
                        row_ref,
                        "text_missing_mrn",
                        "Note has no MRN; identity will be inferred from name/DOB if possible",
                        "warning",
                    )
                )
            groups.setdefault(key, []).append((header, body, row_ref))

        for key, entries in groups.items():
            header, _, _ = entries[0]
            full_name = header.get("patient") or header.get("patientname") or header.get("name")
            patient = RawPatient(
                mrn=header.get("mrn"),
                full_name=full_name,
                birth_date=header.get("dob") or header.get("dateofbirth") or header.get("birthdate"),
                gender=header.get("sex") or header.get("gender"),
                preferred_language=header.get("language"),
                source=self.name,
                source_format="text",
                row_ref=entries[0][2],
                raw={"header": header},
            )
            for entry_header, body, row_ref in entries:
                doc_type = (
                    entry_header.get("type")
                    or entry_header.get("documenttype")
                    or entry_header.get("notetype")
                )
                authored_on = (
                    entry_header.get("date")
                    or entry_header.get("dateofservice")
                    or entry_header.get("dos")
                )
                patient.documents.append(
                    RawDocument(
                        text=body,
                        doc_type=doc_type,
                        title=entry_header.get("title") or entry_header.get("subject"),
                        authored_on=authored_on,
                        author=entry_header.get("author")
                        or entry_header.get("provider")
                        or entry_header.get("clinician"),
                        encounter_ref=entry_header.get("encounter")
                        or entry_header.get("encounterid")
                        or entry_header.get("visit"),
                        source=self.name,
                        source_format="text",
                    )
                )
            result.records.append(patient)
        return result

    # ------------------------------------------------------------------
    def _parse_block(self, block: str) -> tuple[dict[str, str], str]:
        lines = block.splitlines()
        header: dict[str, str] = {}
        body_start = len(lines)
        in_header = True
        for index, line in enumerate(lines):
            if not in_header:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if _BODY_SEPARATOR.match(stripped):
                body_start = index + 1
                in_header = False
                break
            match = _HEADER_LINE.match(line)
            if match and normalize_key(match.group(1)) in _HEADER_KEYS:
                header[normalize_key(match.group(1))] = match.group(2).strip()
                continue
            body_start = index
            in_header = False
            break
        body = "\n".join(lines[body_start:]).strip()
        # Fallback: identity keys embedded in the first lines of an otherwise
        # unstructured scanned note.
        for line in lines[:25]:
            match = _HEADER_LINE.match(line)
            if not match:
                continue
            key = normalize_key(match.group(1))
            if key in _HEADER_KEYS:
                header.setdefault(key, match.group(2).strip())
        return header, body
