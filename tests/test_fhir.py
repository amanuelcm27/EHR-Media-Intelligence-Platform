"""Task 2 tests: FHIR mapping, three-gate validation and SQLite persistence."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ehr_media.config import Settings
from ehr_media.fhir import FhirMapper, FhirPipeline, FhirValidator, SqliteStore
from ehr_media.ids import (
    document_id_for,
    encounter_id_for,
    observation_id_for,
    patient_id_for_mrn,
)
from ehr_media.models import ClinicalDocument, Encounter, LabResult, PatientRecord


def build_patient() -> PatientRecord:
    mrn = "00999999"
    patient_id = patient_id_for_mrn(mrn)
    encounter = Encounter(
        id=encounter_id_for(patient_id, "AMB|2026-04-01|Annual physical"),
        patient_id=patient_id,
        class_code="AMB",
        reason="Annual physical",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 1),
        source="json",
    )
    note_text = (
        "Chief Complaint: chest pain and palpitations.\n"
        "Assessment: atrial fibrillation, hypertension.\n"
        "Plan: start apixaban, cardiology referral."
    )
    document = ClinicalDocument(
        id=document_id_for(patient_id, "note-hash"),
        patient_id=patient_id,
        doc_type="progress_note",
        title="Cardiology progress note",
        text=note_text,
        authored_on=date(2026, 4, 1),
        author="Dr. Ada Lovelace",
        encounter_id=encounter.id,
        source="json",
        source_format="json",
        content_hash="note-hash",
    )
    labs = [
        LabResult(
            id=observation_id_for(patient_id, "potassium-hash"),
            patient_id=patient_id,
            test_name="Potassium",
            loinc_code="2823-3",
            value=6.4,
            unit="mmol/L",
            reference_range="3.5-5.1",
            reference_low=3.5,
            reference_high=5.1,
            interpretation="critical",
            collected_at=date(2026, 4, 1),
            status="final",
            source="csv",
            content_hash="potassium-hash",
        ),
        LabResult(
            id=observation_id_for(patient_id, "creatinine-hash"),
            patient_id=patient_id,
            test_name="Creatinine",
            loinc_code="2160-0",
            value=0.9,
            unit="mg/dL",
            reference_range="0.6-1.2",
            reference_low=0.6,
            reference_high=1.2,
            interpretation="normal",
            collected_at=date(2026, 4, 1),
            status="final",
            source="csv",
            content_hash="creatinine-hash",
        ),
    ]
    return PatientRecord(
        id=patient_id,
        mrn=mrn,
        given_name="Grace",
        family_name="Hopper",
        birth_date=date(1955, 12, 9),
        gender="female",
        phone="+12125550123",
        email="grace.hopper@example.org",
        city="New York",
        state="NY",
        postal_code="10001",
        preferred_language="English",
        documents=[document],
        lab_results=labs,
        encounters=[encounter],
        ingested_at=datetime(2026, 11, 22, tzinfo=timezone.utc),
    )


@pytest.fixture
def patient() -> PatientRecord:
    return build_patient()


def _entries(bundle: dict) -> list[dict]:
    return [entry["resource"] for entry in bundle["entry"]]


def _by_type(bundle: dict, resource_type: str) -> list[dict]:
    return [
        resource
        for resource in _entries(bundle)
        if resource.get("resourceType") == resource_type
    ]


def test_bundle_contains_expected_resources_and_references(patient: PatientRecord) -> None:
    bundle = FhirMapper(patient).build_bundle()

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(_by_type(bundle, "Patient")) == 1
    assert len(_by_type(bundle, "Encounter")) == 1
    assert len(_by_type(bundle, "DocumentReference")) == 1
    assert len(_by_type(bundle, "Observation")) == 2
    assert len(_by_type(bundle, "DiagnosticReport")) == 1

    patient_resource = _by_type(bundle, "Patient")[0]
    assert patient_resource["gender"] == "female"
    assert patient_resource["birthDate"] == "1955-12-09"
    assert patient_resource["identifier"][0]["value"] == "00999999"

    reference_ids = {
        f"{resource['resourceType']}/{resource['id']}" for resource in _entries(bundle)
    }
    for resource in _entries(bundle):
        if resource["resourceType"] == "Patient":
            continue
        assert resource["subject"]["reference"] == f"Patient/{patient.id}"
    document = _by_type(bundle, "DocumentReference")[0]
    assert document["context"]["encounter"][0]["reference"] in reference_ids
    assert document["content"][0]["attachment"]["data"]
    assert (
        base64.b64decode(document["content"][0]["attachment"]["data"]).decode("utf-8")
        == patient.documents[0].text
    )

    report = _by_type(bundle, "DiagnosticReport")[0]
    assert report["status"] == "final"
    for result in report["result"]:
        assert result["reference"] in reference_ids
    observations = {resource["id"]: resource for resource in _by_type(bundle, "Observation")}
    potassium = next(
        resource for resource in observations.values()
        if resource["code"]["coding"][0]["code"] == "2823-3"
    )
    assert potassium["interpretation"][0]["coding"][0]["code"] == "AA"
    assert potassium["valueQuantity"]["value"] == 6.4


def test_valid_bundle_passes_all_three_gates(patient: PatientRecord) -> None:
    bundle = FhirMapper(patient).build_bundle()
    validation = FhirValidator().validate(bundle, patient)

    assert validation.valid, [issue.model_dump() for issue in validation.issues]
    assert validation.error_count == 0
    assert validation.resource_counts == {
        "DiagnosticReport": 1,
        "DocumentReference": 1,
        "Encounter": 1,
        "Observation": 2,
        "Patient": 1,
    }
    assert validation.bundle_id == bundle["id"]


def test_invalid_gender_is_caught_by_schema_and_business_rules(
    patient: PatientRecord,
) -> None:
    bundle = FhirMapper(patient).build_bundle()
    _by_type(bundle, "Patient")[0]["gender"] = "xenomorph"

    validation = FhirValidator().validate(bundle)

    assert not validation.valid
    codes = {issue.code for issue in validation.issues}
    assert "invalid_gender" in codes
    assert any(issue.source == "r4-json-schema" for issue in validation.issues)


def test_unresolved_reference_is_reported(patient: PatientRecord) -> None:
    bundle = FhirMapper(patient).build_bundle()
    # Drop the encounter the note points at; the DocumentReference now dangles.
    bundle["entry"] = [
        entry
        for entry in bundle["entry"]
        if entry["resource"]["resourceType"] != "Encounter"
    ]

    validation = FhirValidator().validate(bundle, patient)

    assert not validation.valid
    assert "unresolved_reference" in {issue.code for issue in validation.issues}


def test_wrong_subject_reference_is_reported(patient: PatientRecord) -> None:
    bundle = FhirMapper(patient).build_bundle()
    _by_type(bundle, "DocumentReference")[0]["subject"]["reference"] = "Patient/other"

    validation = FhirValidator().validate(bundle, patient)

    assert not validation.valid
    assert "invalid_subject" in {issue.code for issue in validation.issues}


def test_pipeline_writes_artifacts_and_persists(
    settings: Settings, patient: PatientRecord
) -> None:
    settings.ensure_dirs()
    pipeline = FhirPipeline(settings)
    result = pipeline.run([patient], write_artifacts=True, persist=True)

    assert result.total_bundles == 1
    assert result.valid_bundles == 1
    assert result.report is not None
    assert result.report.total_errors == 0
    assert result.duration_ms > 0

    bundle_path = settings.fhir_dir / f"{patient.id}.json"
    assert bundle_path.exists()
    written = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert written["id"] == result.bundles[patient.id]["id"]
    assert result.paths["report_json"].exists()
    assert result.paths["report_md"].exists()
    assert result.paths["manifest"].exists()

    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["summary"]["bundles"] == 1
    assert manifest["bundles"][0]["valid"] is True

    store = SqliteStore(settings.db_path)
    assert store.get_patient(patient.id)["mrn"] == "00999999"
    assert store.get_bundle(patient.id)["id"] == written["id"]
    assert len(store.get_resources(patient.id)) == 6
    assert store.get_validation(patient.id)["valid"] is True
    assert store.validation_summary() == {
        "bundles": 1,
        "valid": 1,
        "invalid": 0,
        "errors": 0,
        "warnings": 0,
    }


def test_report_counts_invalid_bundles(settings: Settings, patient: PatientRecord) -> None:
    settings.ensure_dirs()
    pipeline = FhirPipeline(settings)
    good = FhirMapper(patient).build_bundle()
    broken = json.loads(json.dumps(good))
    _by_type(broken, "Patient")[0]["gender"] = "xenomorph"
    broken["id"] = "bundle-broken"

    # Exercise the report aggregation directly; the pipeline always maps from
    # valid PatientRecord objects, so an invalid bundle is injected here.
    validations = [
        pipeline.validate(good, patient),
        pipeline.validate(broken),
    ]
    from ehr_media.fhir import ValidationReport

    report = ValidationReport.from_validations(validations)

    assert report.total_bundles == 2
    assert report.valid_bundles == 1
    assert report.invalid_bundles == 1
    assert report.total_errors >= 1
    assert report.resource_counts["Patient"] == 2
