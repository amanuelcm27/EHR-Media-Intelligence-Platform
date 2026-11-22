"""Deterministic extraction of the facts a clinical summary must cover.

The LLM never sees the raw record; it receives this compact, structured
evidence block. That keeps prompts small, makes the offline fallback useful,
and means the summary can be audited against the same facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any

from ..models import ClinicalDocument, LabResult, PatientRecord
from .models import MediaItem, SummaryEvidence

_LABEL_RE = re.compile(
    r"(?:chief complaint|chief concern|presenting complaint|reason for visit)"
    r"\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
_ADMISSION_RE = re.compile(
    r"(?:admitted with|presented with|presents with|reports|evaluation for)\s+"
    r"([^.;\n]{5,160})",
    re.IGNORECASE,
)
_DX_RE = re.compile(r"(?:diagnoses?|assessment|impression)\s*[:\-]\s*(.+)", re.IGNORECASE)
_DX_STOP_RE = re.compile(
    r"\b(plan|disposition|follow[- ]?up|hospital course|medications?)\s*:",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DX_SPLIT = re.compile(r"[;,]|(?<=[a-z0-9])\.\s+")

_PLAN_PREFIXES = (
    "plan",
    "continue",
    "start",
    "stop",
    "repeat",
    "wean",
    "treated",
    "disposition",
    "follow",
    "admit",
    "transfer",
    "patient",
    "monitor",
)

# Modifiers that should not make two mentions of the same condition distinct.
_DX_MODIFIERS = {
    "improving",
    "improved",
    "resolving",
    "resolved",
    "worsening",
    "acute",
    "chronic",
    "stable",
    "mild",
    "moderate",
    "severe",
    "suspected",
    "possible",
    "likely",
    "therapeutic",
    "after",
}

_CONDITION_HINT = re.compile(
    r"\b(sepsis|failure|infarct|stroke|pneumonia|appendicitis|fracture|embolism|"
    r"diabetes|hypertension|arrhythmia|fibrillation|asthma|copd|cancer|carcinoma|"
    r"lymphoma|leukemia|infection|abscess|thrombosis|hemorrhage|aneurysm|"
    r"pancreatitis|cholecystitis|diverticulitis|meningitis|encephalopathy|renal|"
    r"kidney|hepatic|anemia|thrombocytopenia|neutropenia|hyperkalemia|"
    r"hypokalemia|hyponatremia|acidosis|alkalosis)\b",
    re.IGNORECASE,
)

_SKIP_FLAG_CODES = {"critical_lab", "abnormal_lab"}

_TYPE_LABELS = {
    "progress_note": "Progress note",
    "consultation": "Consultation",
    "discharge_summary": "Discharge summary",
    "imaging_report": "Imaging report",
    "lab_report": "Laboratory report",
    "scanned_note": "Scanned note",
    "other": "Clinical document",
}


def record_hash_for(patient: PatientRecord) -> str:
    """Hash only the clinical content that should invalidate a cached summary."""
    payload: dict[str, Any] = {
        "id": patient.id,
        "mrn": patient.mrn,
        "given_name": patient.given_name,
        "family_name": patient.family_name,
        "middle_name": patient.middle_name,
        "suffix": patient.suffix,
        "birth_date": patient.birth_date,
        "gender": patient.gender,
        "documents": [
            {
                "id": document.id,
                "doc_type": document.doc_type,
                "title": document.title,
                "authored_on": document.authored_on,
                "author": document.author,
                "text": document.text,
            }
            for document in sorted(patient.documents, key=lambda item: item.id)
        ],
        "labs": [
            {
                "id": lab.id,
                "test_name": lab.test_name,
                "value": lab.value,
                "value_text": lab.value_text,
                "unit": lab.unit,
                "reference_range": lab.reference_range,
                "interpretation": lab.interpretation,
                "collected_at": lab.collected_at,
                "status": lab.status,
            }
            for lab in sorted(patient.lab_results, key=lambda item: item.id)
        ],
        "encounters": [
            {
                "id": encounter.id,
                "class_code": encounter.class_code,
                "reason": encounter.reason,
                "start_date": encounter.start_date,
                "end_date": encounter.end_date,
            }
            for encounter in sorted(patient.encounters, key=lambda item: item.id)
        ],
        "flags": sorted(
            (flag.code, flag.message, flag.severity)
            for flag in patient.data_quality_flags
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def extract_evidence(patient: PatientRecord) -> SummaryEvidence:
    documents = sorted(
        patient.documents,
        key=lambda item: (item.authored_on or date.min, item.id),
        reverse=True,
    )
    labs = sorted(
        patient.lab_results,
        key=lambda item: (item.collected_at or date.min, item.id),
        reverse=True,
    )
    return SummaryEvidence(
        patient_id=patient.id,
        mrn=patient.mrn,
        patient_name=patient.display_name,
        age=patient.age_at(),
        gender=patient.gender,
        chief_concern=_chief_concern(documents),
        diagnoses=_diagnoses(documents, patient),
        recent_media=_recent_media(documents, labs),
        flagged_anomalies=_flagged_anomalies(patient, documents, labs),
        document_count=len(documents),
        lab_count=len(labs),
        encounter_count=len(patient.encounters),
        latest_date=patient.latest_document_date,
        record_hash=record_hash_for(patient),
    )


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_sentence(text: str, limit: int = 240) -> str:
    text = _clean(text)
    if not text:
        return ""
    sentence = _SENTENCE_SPLIT.split(text, maxsplit=1)[0].strip()
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + "…"
    return sentence.rstrip(" .;,") + "."


def _chief_concern(documents: list[ClinicalDocument]) -> str:
    for document in documents:
        match = _LABEL_RE.search(document.text)
        if match:
            value = _first_sentence(match.group(1), limit=220)
            if value:
                return value
    for document in documents:
        match = _ADMISSION_RE.search(document.text)
        if match:
            value = _first_sentence(match.group(1), limit=180)
            if value:
                return f"Patient {value}."
    return "No chief concern explicitly documented; review recent media."


def _diagnoses(documents: list[ClinicalDocument], patient: PatientRecord) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = _clean(value).strip(" .;:,-")
        if not value or len(value) < 3 or len(value) > 110:
            return
        lowered = value.casefold()
        if lowered.startswith(_PLAN_PREFIXES):
            return
        if "therapeutic" in lowered or lowered.startswith("inr "):
            return
        key = " ".join(
            token
            for token in re.findall(r"[a-z0-9]+", lowered)
            if token not in _DX_MODIFIERS
        )
        if not key or key in seen:
            return
        seen.add(key)
        found.append(value[0].upper() + value[1:])

    for document in documents:
        for match in _DX_RE.finditer(document.text):
            fragment = match.group(1)
            stop = _DX_STOP_RE.search(fragment)
            if stop:
                fragment = fragment[: stop.start()]
            for piece in _DX_SPLIT.split(fragment[:400]):
                add(piece)
        # Document titles such as "Discharge Summary - Sepsis" are useful when
        # a note has no explicit Assessment/Diagnoses label. Skip department
        # or topic suffixes ("Progress Note - Cardiology").
        if " - " in document.title:
            suffix = document.title.split(" - ", 1)[1]
            if document.doc_type in {"discharge_summary", "consultation"} or _CONDITION_HINT.search(
                suffix
            ):
                add(suffix)

    if not found:
        abnormal = [lab for lab in patient.lab_results if lab.is_abnormal]
        if abnormal:
            names = ", ".join(lab.test_name for lab in abnormal[:4])
            add(f"Abnormal results requiring review ({names})")
    return found[:8]


def _recent_media(
    documents: list[ClinicalDocument], labs: list[LabResult]
) -> list[MediaItem]:
    items: list[MediaItem] = []
    for document in documents[:4]:
        items.append(
            MediaItem(
                kind=_TYPE_LABELS.get(document.doc_type, "Clinical document"),
                title=document.title,
                date=document.authored_on,
                detail=_first_sentence(document.text, limit=180) or None,
            )
        )

    # Group the latest labs into one panel item so the summary stays compact.
    by_date: dict[date, list[LabResult]] = {}
    for lab in labs:
        if lab.collected_at:
            by_date.setdefault(lab.collected_at, []).append(lab)
    for collected, group in list(sorted(by_date.items(), reverse=True))[:2]:
        abnormal = [lab for lab in group if lab.is_abnormal]
        detail = "; ".join(
            f"{lab.test_name} {lab.display_value}"
            + (f" ({lab.interpretation})" if lab.is_abnormal else "")
            for lab in (abnormal or group)[:5]
        )
        items.append(
            MediaItem(
                kind="Lab panel",
                title=f"{len(group)} laboratory result(s)",
                date=collected,
                detail=detail or None,
            )
        )
    return items[:6]


def _flagged_anomalies(
    patient: PatientRecord,
    documents: list[ClinicalDocument],
    labs: list[LabResult],
) -> list[str]:
    ranked: list[tuple[int, str]] = []
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    seen: set[str] = set()

    def add(message: str, severity: str) -> None:
        message = _clean(message).strip(" .;")
        if not message or message.casefold() in seen:
            return
        seen.add(message.casefold())
        ranked.append((severity_rank.get(severity, 3), message))

    for lab in labs:
        if lab.interpretation in {"critical", "high", "low"}:
            when = f" on {lab.collected_at.isoformat()}" if lab.collected_at else ""
            add(
                f"{lab.test_name} {lab.display_value} ({lab.interpretation}){when}",
                "critical" if lab.interpretation == "critical" else "warning",
            )
    for document in documents:
        for anomaly in document.anomalies:
            when = f", {document.authored_on.isoformat()}" if document.authored_on else ""
            add(f"{anomaly.description} — {document.title}{when}", anomaly.severity)
    for flag in patient.data_quality_flags:
        if flag.severity in {"critical", "warning"} and flag.code not in _SKIP_FLAG_CODES:
            add(flag.message, flag.severity)

    ranked.sort(key=lambda item: item[0])
    return [message for _, message in ranked[:8]]
