"""Canonical, validated intermediate representation.

This is the contract between ingestion/cleaning (Task 1) and FHIR mapping
(Task 2). Every cleaned patient is a :class:`PatientRecord`; every change made
by the cleaning pipeline is captured in its ``audit`` list so a reviewer can
reconstruct exactly what happened to each field.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .ids import mrn_display

Gender = Literal["male", "female", "other", "unknown"]
AuditAction = Literal[
    "normalized",
    "default_applied",
    "conflict_resolved",
    "duplicate_removed",
    "inferred",
    "validated",
    "quarantined",
]
Severity = Literal["info", "warning", "error", "critical"]
DocumentType = Literal[
    "progress_note",
    "consultation",
    "discharge_summary",
    "imaging_report",
    "lab_report",
    "scanned_note",
    "other",
]
LabInterpretation = Literal["normal", "low", "high", "critical", "unknown"]
EncounterClass = Literal["AMB", "IMP", "EMER", "OBSENC"]


class AuditEntry(BaseModel):
    """One field-level change made by the cleaning pipeline."""

    model_config = ConfigDict(extra="forbid")

    field: str
    action: AuditAction
    before: str | None = None
    after: str | None = None
    rule: str
    source: str
    at: datetime


class DataQualityFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: Severity = "warning"
    field: str | None = None


class Anomaly(BaseModel):
    """A clinically relevant anomaly detected in a document or lab."""

    model_config = ConfigDict(extra="forbid")

    code: str
    description: str
    severity: Severity = "warning"


class ClinicalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    patient_id: str
    doc_type: DocumentType = "other"
    title: str
    text: str
    authored_on: date | None = None
    author: str | None = None
    encounter_id: str | None = None
    source: str
    source_format: str
    content_hash: str
    anomalies: list[Anomaly] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class LabResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    patient_id: str
    test_name: str
    loinc_code: str | None = None
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    reference_low: float | None = None
    reference_high: float | None = None
    interpretation: LabInterpretation = "unknown"
    collected_at: date | None = None
    status: Literal["final", "preliminary", "corrected", "cancelled", "unknown"] = "unknown"
    source: str
    content_hash: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_abnormal(self) -> bool:
        return self.interpretation in {"low", "high", "critical"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_value(self) -> str:
        if self.value is not None:
            rendered = f"{self.value:g}"
            return f"{rendered} {self.unit}".strip() if self.unit else rendered
        return self.value_text or "—"


class Encounter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    patient_id: str
    class_code: EncounterClass = "AMB"
    reason: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    source: str = "derived"


class PatientRecord(BaseModel):
    """Canonical patient with cleaned demographics and clinical media."""

    model_config = ConfigDict(extra="forbid")

    id: str
    mrn: str
    given_name: str
    family_name: str
    middle_name: str | None = None
    suffix: str | None = None
    birth_date: date | None = None
    gender: Gender = "unknown"
    phone: str | None = None
    email: str | None = None
    address_line: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    preferred_language: str | None = None
    documents: list[ClinicalDocument] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    encounters: list[Encounter] = Field(default_factory=list)
    data_quality_flags: list[DataQualityFlag] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    ingested_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_name(self) -> str:
        parts = [self.given_name, self.middle_name, self.family_name, self.suffix]
        return " ".join(part for part in parts if part)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mrn_display(self) -> str:
        return mrn_display(self.mrn)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def abnormal_labs(self) -> list[LabResult]:
        return [lab for lab in self.lab_results if lab.is_abnormal]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def latest_document_date(self) -> date | None:
        dates = [doc.authored_on for doc in self.documents if doc.authored_on]
        dates += [lab.collected_at for lab in self.lab_results if lab.collected_at]
        return max(dates) if dates else None

    def age_at(self, on: date | None = None) -> int | None:
        if self.birth_date is None:
            return None
        on = on or date.today()
        return (
            on.year
            - self.birth_date.year
            - ((on.month, on.day) < (self.birth_date.month, self.birth_date.day))
        )

    def flag_codes(self) -> set[str]:
        return {flag.code for flag in self.data_quality_flags}
