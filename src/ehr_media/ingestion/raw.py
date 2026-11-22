"""Raw, source-shaped records produced by the ingestion adapters.

These types intentionally contain unvalidated values (dates may be strings,
lab values may be ``"<0.5 mg/dL"``). Cleaning and validation happen in
:mod:`ehr_media.ingestion.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class RawDocument:
    text: str
    doc_type: str | None = None
    title: str | None = None
    authored_on: Any = None
    author: str | None = None
    encounter_ref: str | None = None
    source: str = ""
    source_format: str = "json"


@dataclass(slots=True)
class RawLab:
    test_name: str
    value: Any = None
    unit: str | None = None
    reference_range: str | None = None
    collected_at: Any = None
    status: str | None = None
    loinc_code: str | None = None
    source: str = ""
    source_format: str = "csv"


@dataclass(slots=True)
class RawEncounter:
    class_code: str | None = None
    reason: str | None = None
    start_date: Any = None
    end_date: Any = None
    source: str = ""


@dataclass(slots=True)
class RawPatient:
    mrn: Any = None
    given_name: str | None = None
    family_name: str | None = None
    middle_name: str | None = None
    suffix: str | None = None
    full_name: str | None = None
    birth_date: Any = None
    gender: Any = None
    phone: str | None = None
    email: str | None = None
    address_line: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    preferred_language: str | None = None
    documents: list[RawDocument] = field(default_factory=list)
    labs: list[RawLab] = field(default_factory=list)
    encounters: list[RawEncounter] = field(default_factory=list)
    source: str = ""
    source_format: str = "json"
    row_ref: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionIssue:
    source: str
    row_ref: str
    code: str
    message: str
    severity: str = "warning"
    raw_excerpt: str | None = None


@dataclass(slots=True)
class SourceResult:
    records: list[RawPatient] = field(default_factory=list)
    issues: list[IngestionIssue] = field(default_factory=list)


class Source(Protocol):
    """A file-backed ingestion adapter."""

    name: str

    def load(self) -> SourceResult:  # pragma: no cover - protocol definition
        ...
