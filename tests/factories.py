"""Reusable synthetic patient factories for the search/API tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable

from ehr_media.ids import (
    content_hash,
    document_id_for,
    encounter_id_for,
    observation_id_for,
    patient_id_for_mrn,
)
from ehr_media.models import ClinicalDocument, Encounter, LabResult, PatientRecord


def make_patient(
    *,
    mrn: str = "00777777",
    given_name: str = "Grace",
    family_name: str = "Hopper",
    birth_date: date = date(1955, 12, 9),
    gender: str = "female",
    note_title: str = "Cardiology progress note",
    note_text: str = (
        "Chief Complaint: chest pain and palpitations. "
        "Assessment: atrial fibrillation, hypertension. "
        "Plan: start apixaban, cardiology referral."
    ),
    note_date: date = date(2026, 4, 1),
    doc_type: str = "progress_note",
    labs: Iterable[dict] | None = None,
    encounter_reason: str = "Chest pain",
    encounter_date: date = date(2026, 4, 1),
) -> PatientRecord:
    patient_id = patient_id_for_mrn(mrn)
    encounter = Encounter(
        id=encounter_id_for(patient_id, f"AMB|{encounter_date}|{encounter_reason}"),
        patient_id=patient_id,
        class_code="AMB",
        reason=encounter_reason,
        start_date=encounter_date,
        end_date=encounter_date,
        source="json",
    )
    digest = content_hash(note_title, note_date.isoformat(), note_text)
    document = ClinicalDocument(
        id=document_id_for(patient_id, digest),
        patient_id=patient_id,
        doc_type=doc_type,  # type: ignore[arg-type]
        title=note_title,
        text=note_text,
        authored_on=note_date,
        author="Dr. Test",
        encounter_id=encounter.id,
        source="json",
        source_format="json",
        content_hash=digest,
    )

    lab_specs = list(labs) if labs is not None else [
        {
            "test_name": "Potassium",
            "value": 6.4,
            "unit": "mmol/L",
            "reference_range": "3.5-5.1",
            "reference_low": 3.5,
            "reference_high": 5.1,
            "interpretation": "critical",
            "collected_at": note_date,
        }
    ]
    lab_results: list[LabResult] = []
    for spec in lab_specs:
        lab_digest = content_hash(
            patient_id, spec["test_name"], spec.get("collected_at"), spec.get("value")
        )
        lab_results.append(
            LabResult(
                id=observation_id_for(patient_id, lab_digest),
                patient_id=patient_id,
                test_name=spec["test_name"],
                value=spec.get("value"),
                value_text=spec.get("value_text"),
                unit=spec.get("unit"),
                reference_range=spec.get("reference_range"),
                reference_low=spec.get("reference_low"),
                reference_high=spec.get("reference_high"),
                interpretation=spec.get("interpretation", "unknown"),
                collected_at=spec.get("collected_at"),
                status=spec.get("status", "final"),
                source="csv",
                content_hash=lab_digest,
            )
        )

    return PatientRecord(
        id=patient_id,
        mrn=mrn,
        given_name=given_name,
        family_name=family_name,
        birth_date=birth_date,
        gender=gender,  # type: ignore[arg-type]
        documents=[document],
        lab_results=lab_results,
        encounters=[encounter],
        ingested_at=datetime(2026, 11, 22, tzinfo=timezone.utc),
    )


def three_topic_patients() -> list[PatientRecord]:
    return [
        make_patient(
            mrn="00700001",
            given_name="Eleanor",
            family_name="Vance",
            note_title="Discharge Summary - Heart Failure",
            note_text=(
                "Chief Complaint: worsening shortness of breath. "
                "Assessment: acute decompensated heart failure, hyperkalemia. "
                "Plan: IV furosemide and daily weights."
            ),
            doc_type="discharge_summary",
            labs=[
                {
                    "test_name": "BNP",
                    "value": 1450,
                    "unit": "pg/mL",
                    "reference_range": "<100",
                    "interpretation": "critical",
                    "collected_at": date(2026, 2, 14),
                },
                {
                    "test_name": "Potassium",
                    "value": 6.8,
                    "unit": "mmol/L",
                    "reference_range": "3.5-5.1",
                    "interpretation": "critical",
                    "collected_at": date(2026, 2, 14),
                },
            ],
            encounter_reason="Heart failure",
            encounter_date=date(2026, 2, 14),
        ),
        make_patient(
            mrn="00700002",
            given_name="Lena",
            family_name="Petrova",
            note_title="Surgical Consultation - Appendicitis",
            note_text=(
                "Chief Complaint: right lower quadrant pain. "
                "Assessment: acute appendicitis without perforation. "
                "Plan: laparoscopic appendectomy after antibiotics."
            ),
            note_date=date(2026, 4, 2),
            doc_type="consultation",
            labs=[
                {
                    "test_name": "White Blood Cell Count",
                    "value": 15.2,
                    "unit": "10^3/uL",
                    "reference_range": "4.0-11.0",
                    "interpretation": "high",
                    "collected_at": date(2026, 4, 2),
                }
            ],
            encounter_reason="Appendicitis",
            encounter_date=date(2026, 4, 2),
        ),
        make_patient(
            mrn="00700003",
            given_name="James",
            family_name="OConnor",
            note_title="ED Follow-up Note - Pneumonia",
            note_text=(
                "Chief Complaint: fever and productive cough. "
                "Assessment: sepsis secondary to pneumonia. "
                "Plan: ceftriaxone and azithromycin."
            ),
            note_date=date(2026, 2, 3),
            labs=[
                {
                    "test_name": "Lactate",
                    "value": 2.8,
                    "unit": "mmol/L",
                    "reference_range": "0.5-2.0",
                    "interpretation": "high",
                    "collected_at": date(2026, 2, 3),
                }
            ],
            encounter_reason="Pneumonia",
            encounter_date=date(2026, 2, 3),
        ),
    ]
