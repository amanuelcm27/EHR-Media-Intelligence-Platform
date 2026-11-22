"""Typed summary artifacts returned by the summarization service."""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "AI-generated summary for clinical decision support only. It may be "
    "incomplete or inaccurate and must be verified against the source record "
    "by a qualified clinician before any clinical use."
)


class SummarySections(BaseModel):
    chief_concern: str
    diagnoses: list[str] = Field(default_factory=list)
    recent_media: list[str] = Field(default_factory=list)
    flagged_anomalies: list[str] = Field(default_factory=list)


class MediaItem(BaseModel):
    kind: str
    title: str
    date: date_type | None = None
    detail: str | None = None


class SummaryEvidence(BaseModel):
    """Structured facts extracted from the record before any LLM call."""

    patient_id: str
    mrn: str
    patient_name: str
    age: int | None = None
    gender: str = "unknown"
    chief_concern: str
    diagnoses: list[str] = Field(default_factory=list)
    recent_media: list[MediaItem] = Field(default_factory=list)
    flagged_anomalies: list[str] = Field(default_factory=list)
    document_count: int = 0
    lab_count: int = 0
    encounter_count: int = 0
    latest_date: date_type | None = None
    record_hash: str


class ProviderResult(BaseModel):
    sections: SummarySections
    confidence: float
    provider: str
    model: str
    fallback: bool = False
    error: str | None = None
    raw_response: str | None = None


class ClinicalSummary(BaseModel):
    patient_id: str
    patient_name: str
    mrn: str
    record_hash: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str
    model: str
    sections: SummarySections
    summary: str
    word_count: int
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: Literal["low", "moderate", "high"]
    disclaimer: str = DISCLAIMER
    cached: bool = False
    fallback: bool = False
    error: str | None = None
    latency_ms: float | None = None

    def to_text(self) -> str:
        return self.summary


class SummaryBatchResult(BaseModel):
    summaries: list[ClinicalSummary] = Field(default_factory=list)
    duration_ms: float = 0.0
    cached: int = 0
    generated: int = 0
    fallbacks: int = 0
    providers: dict[str, int] = Field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.summaries)

    def by_patient(self, patient_id: str) -> ClinicalSummary | None:
        return next(
            (summary for summary in self.summaries if summary.patient_id == patient_id),
            None,
        )
