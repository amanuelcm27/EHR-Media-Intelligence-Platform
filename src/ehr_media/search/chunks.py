"""Build searchable chunks from cleaned patients and generated summaries.

Each chunk maps back to one FHIR resource so the API can return a resource
type, a resource id, and a link to the patient detail drawer.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from pydantic import BaseModel, Field

from ..ids import report_id_for, stable_id
from ..models import ClinicalDocument, LabResult, PatientRecord
from ..summaries.models import ClinicalSummary

DOCUMENT_LABELS = {
    "progress_note": "Progress note",
    "consultation": "Consultation note",
    "discharge_summary": "Discharge summary",
    "imaging_report": "Imaging report",
    "lab_report": "Laboratory report",
    "scanned_note": "Scanned note",
    "other": "Clinical document",
}

ENCOUNTER_LABELS = {
    "AMB": "Ambulatory encounter",
    "IMP": "Inpatient encounter",
    "EMER": "Emergency encounter",
    "OBSENC": "Observation encounter",
}


class Chunk(BaseModel):
    chunk_id: str
    patient_id: str
    patient_name: str
    mrn: str
    resource_type: str
    resource_id: str
    title: str | None = None
    effective_date: str | None = None
    section: str | None = None
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)

    def store_payload(self) -> dict:
        payload = self.model_dump()
        payload.pop("metadata", None)
        return payload


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    size = max(size, 100)
    overlap = max(0, min(overlap, size // 2))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _chunk(
    patient: PatientRecord,
    *,
    resource_type: str,
    resource_id: str,
    title: str | None,
    effective_date: date | None,
    section: str,
    text: str,
    index: int = 0,
    metadata: dict[str, str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=stable_id("chunk", resource_type, resource_id, index, length=20),
        patient_id=patient.id,
        patient_name=patient.display_name,
        mrn=patient.mrn,
        resource_type=resource_type,
        resource_id=resource_id,
        title=title,
        effective_date=_date_iso(effective_date),
        section=section,
        text=text.strip(),
        metadata=metadata or {},
    )


def _document_chunks(
    patient: PatientRecord, document: ClinicalDocument, size: int, overlap: int
) -> list[Chunk]:
    label = DOCUMENT_LABELS.get(document.doc_type, "Clinical document")
    heading = f"{document.title} ({label}"
    if document.authored_on:
        heading += f", {document.authored_on.isoformat()}"
    heading += ")."
    body = f"{heading} {document.text}"
    pieces = _split_text(body, size, overlap)
    return [
        _chunk(
            patient,
            resource_type="DocumentReference",
            resource_id=document.id,
            title=document.title,
            effective_date=document.authored_on,
            section="note",
            text=piece,
            index=index,
            metadata={"doc_type": document.doc_type, "author": document.author or ""},
        )
        for index, piece in enumerate(pieces)
    ]


def _lab_chunks(patient: PatientRecord, lab: LabResult) -> list[Chunk]:
    parts = [f"Lab result: {lab.test_name}."]
    if lab.value is not None:
        parts.append(f"Value {lab.display_value}.")
    elif lab.value_text:
        parts.append(f"Value {lab.value_text}.")
    if lab.interpretation != "unknown":
        parts.append(f"Interpretation {lab.interpretation}.")
    if lab.reference_range:
        parts.append(f"Reference range {lab.reference_range}.")
    if lab.collected_at:
        parts.append(f"Collected {lab.collected_at.isoformat()}.")
    if lab.status != "unknown":
        parts.append(f"Status {lab.status}.")
    return [
        _chunk(
            patient,
            resource_type="Observation",
            resource_id=lab.id,
            title=lab.test_name,
            effective_date=lab.collected_at,
            section="lab",
            text=" ".join(parts),
            metadata={"interpretation": lab.interpretation, "loinc": lab.loinc_code or ""},
        )
    ]


def _report_chunks(patient: PatientRecord) -> list[Chunk]:
    grouped: dict[str, list[LabResult]] = {}
    for lab in patient.lab_results:
        key = lab.collected_at.isoformat() if lab.collected_at else "undated"
        grouped.setdefault(key, []).append(lab)

    chunks: list[Chunk] = []
    for key, group in grouped.items():
        collected = group[0].collected_at
        report_id = report_id_for(patient.id, key)
        abnormal = [lab for lab in group if lab.is_abnormal]
        results = "; ".join(
            f"{lab.test_name} {lab.display_value}"
            + (f" ({lab.interpretation})" if lab.is_abnormal else "")
            for lab in sorted(group, key=lambda item: item.test_name)
        )
        text = f"Laboratory panel with {len(group)} result(s)."
        if collected:
            text += f" Collected {collected.isoformat()}."
        text += f" Results: {results}."
        if abnormal:
            text += f" {len(abnormal)} abnormal result(s)."
        chunks.append(
            _chunk(
                patient,
                resource_type="DiagnosticReport",
                resource_id=report_id,
                title="Laboratory panel",
                effective_date=collected,
                section="report",
                text=text,
            )
        )
    return chunks


def _encounter_chunks(patient: PatientRecord) -> list[Chunk]:
    chunks: list[Chunk] = []
    for encounter in patient.encounters:
        label = ENCOUNTER_LABELS.get(encounter.class_code, "Encounter")
        text = f"{label}."
        if encounter.reason:
            text += f" Reason: {encounter.reason}."
        if encounter.start_date:
            text += f" Start {encounter.start_date.isoformat()}."
        if encounter.end_date:
            text += f" End {encounter.end_date.isoformat()}."
        chunks.append(
            _chunk(
                patient,
                resource_type="Encounter",
                resource_id=encounter.id,
                title=encounter.reason or label,
                effective_date=encounter.start_date,
                section="encounter",
                text=text,
                metadata={"class": encounter.class_code},
            )
        )
    return chunks


def _patient_chunk(
    patient: PatientRecord, summary: ClinicalSummary | None
) -> Chunk:
    age = patient.age_at()
    demographics = (
        f"Patient {patient.display_name}"
        + (f", {age}-year-old" if age is not None else "")
        + f" {patient.gender}. MRN {patient.mrn}."
    )
    conditions: list[str] = []
    if summary is not None:
        conditions = summary.sections.diagnoses[:8]
    if conditions:
        demographics += " Documented conditions: " + "; ".join(conditions) + "."
    if patient.latest_document_date:
        demographics += f" Latest record {patient.latest_document_date.isoformat()}."
    return _chunk(
        patient,
        resource_type="Patient",
        resource_id=patient.id,
        title="Patient demographics",
        effective_date=patient.latest_document_date,
        section="patient",
        text=demographics,
    )


def _summary_chunk(patient: PatientRecord, summary: ClinicalSummary) -> Chunk:
    return _chunk(
        patient,
        resource_type="Summary",
        resource_id=patient.id,
        title="AI clinical summary",
        effective_date=patient.latest_document_date,
        section="summary",
        text=summary.summary,
        metadata={
            "provider": summary.provider,
            "confidence": str(summary.confidence),
        },
    )


def build_chunks(
    patients: Iterable[PatientRecord],
    summaries: dict[str, ClinicalSummary] | None = None,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    include_summaries: bool = True,
) -> list[Chunk]:
    """Return deterministic, resource-linked chunks for the whole corpus."""
    summaries = summaries or {}
    chunks: list[Chunk] = []
    for patient in sorted(patients, key=lambda item: item.mrn):
        summary = summaries.get(patient.id)
        chunks.append(_patient_chunk(patient, summary))
        for document in sorted(
            patient.documents,
            key=lambda item: (item.authored_on or date.min, item.id),
            reverse=True,
        ):
            chunks.extend(
                _document_chunks(patient, document, chunk_size, chunk_overlap)
            )
        for lab in sorted(
            patient.lab_results,
            key=lambda item: (item.collected_at or date.min, item.test_name),
            reverse=True,
        ):
            chunks.extend(_lab_chunks(patient, lab))
        chunks.extend(_report_chunks(patient))
        chunks.extend(_encounter_chunks(patient))
        if include_summaries and summary is not None:
            chunks.append(_summary_chunk(patient, summary))
    chunks.sort(key=lambda chunk: chunk.chunk_id)
    return chunks
