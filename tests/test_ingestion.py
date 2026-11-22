"""End-to-end ingestion/cleaning tests (Task 1)."""

from __future__ import annotations

import json
from pathlib import Path

from ehr_media.ingestion import IngestionPipeline

from tests.helpers import note, write_csv, write_json, write_text


def run(paths: list[Path]):
    return IngestionPipeline.from_paths(paths).run()


def test_duplicate_rows_merge_and_dedupe(tmp_path: Path) -> None:
    text = "Chief Complaint: cough. Assessment: acute bronchitis. Plan: rest and fluids."
    payload = {
        "patients": [
            {
                "mrn": "MRN-900001",
                "first_name": "Test",
                "last_name": "Patient",
                "dob": "01/02/1980",
                "gender": "M",
                "documents": [{"type": "Progress Note", "date": "2026-01-01", "text": text}],
                "labs": [{"test_name": "HGB", "value": "13.2", "date": "2026-01-01"}],
            },
            {
                "mrn": "900001",
                "first_name": "Test",
                "last_name": "Patient",
                "dob": "1980-01-02",
                "gender": "male",
                "documents": [{"type": "Progress Note", "date": "2026-01-01", "text": text}],
                "labs": [
                    {
                        "test_name": "Hemoglobin",
                        "value": 13.2,
                        "unit": "g/dL",
                        "reference_range": "12-17.5",
                        "date": "2026-01-01",
                    }
                ],
            },
        ]
    }
    result = run([write_json(tmp_path / "ehr.json", payload)])

    assert len(result.patients) == 1
    patient = result.patients[0]
    assert patient.mrn == "00900001"
    assert patient.gender == "male"
    assert len(patient.documents) == 1
    assert len(patient.lab_results) == 1
    assert any(entry.action == "duplicate_removed" for entry in patient.audit)
    assert result.report["stats"]["duplicates_removed"]["documents"] == 1
    assert result.report["stats"]["duplicates_removed"]["labs"] == 1


def test_conflicting_identifiers_resolved_by_source_priority(tmp_path: Path) -> None:
    payload = {
        "patients": [
            {
                "mrn": "900002",
                "first_name": "Jane",
                "last_name": "Doe",
                "dob": "1980-01-02",
                "gender": "F",
            }
        ]
    }
    notes = note(
        "900002",
        "Doe, Janet",
        "01/03/1980",
        "2026-01-01",
        "Chief Complaint: fatigue. Assessment: iron deficiency anemia. Plan: iron studies.",
    )
    result = run(
        [
            write_json(tmp_path / "ehr.json", payload),
            write_text(tmp_path / "notes.txt", notes),
        ]
    )

    assert len(result.patients) == 1
    patient = result.patients[0]
    assert patient.given_name == "Jane"  # JSON outranks text
    assert patient.birth_date.isoformat() == "1980-01-02"
    codes = patient.flag_codes()
    assert "conflicting_given_name" in codes
    assert "conflicting_birth_date" in codes
    assert any(entry.action == "conflict_resolved" for entry in patient.audit)
    assert {conflict["field"] for conflict in result.report["conflicts"]} == {
        "given_name",
        "birth_date",
    }


def test_missing_mrn_is_synthesized_or_rejected(tmp_path: Path) -> None:
    payload = {
        "patients": [
            {
                "first_name": "No",
                "last_name": "Identifier",
                "dob": "1990-05-05",
                "gender": "F",
                "documents": [
                    {
                        "type": "Progress Note",
                        "date": "2026-02-01",
                        "text": "Chief Complaint: headache. Assessment: migraine. Plan: hydration.",
                    }
                ],
            },
            {"documents": []},
        ]
    }
    result = run([write_json(tmp_path / "ehr.json", payload)])

    assert len(result.patients) == 1
    patient = result.patients[0]
    assert patient.mrn.startswith("SYN")
    assert "synthetic_mrn" in patient.flag_codes()
    assert result.report["stats"]["rejected_records"] == 1
    assert any(issue.code == "missing_identifier" for issue in result.issues)


def test_invalid_dates_values_and_missing_fields_are_quarantined(tmp_path: Path) -> None:
    payload = {
        "patients": [
            {
                "mrn": "900004",
                "first_name": "Broken",
                "last_name": "Record",
                "gender": "xenomorph",
                "documents": [
                    {
                        "type": "Progress Note",
                        "date": "13/45/2025",
                        "text": "Chief Complaint: dizziness. Assessment: orthostatic hypotension.",
                    }
                ],
                "labs": [
                    {
                        "test_name": "Mystery Test",
                        "value": "abc",
                        "reference_range": "garbage",
                        "date": "not-a-date",
                    }
                ],
            }
        ]
    }
    result = run([write_json(tmp_path / "ehr.json", payload)])
    patient = result.patients[0]

    assert patient.birth_date is None
    assert patient.gender == "unknown"
    codes = patient.flag_codes()
    assert {"missing_birth_date", "invalid_gender", "missing_gender"} <= codes
    assert "invalid_document_date" in codes
    assert patient.documents[0].authored_on is None

    lab = patient.lab_results[0]
    assert lab.value is None
    assert lab.value_text == "ABC"
    assert lab.collected_at is None
    assert lab.interpretation == "unknown"
    assert {"invalid_lab_date", "non_numeric_lab_value", "invalid_reference_range"} <= codes
    assert any(entry.action == "quarantined" for entry in patient.audit)


def test_csv_lab_cleaning_infers_units_and_dedupes(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "labs.csv",
        [
            ["900010", "HGB", "11,8", "", "", "03/05/2026", "final"],
            ["MRN-900010", "HGB", "11,8", "", "", "2026-03-05", "final"],
            ["900010", "", "5.0", "mmol/L", "", "2026-03-05", "final"],
        ],
        ["mrn", "test_name", "value", "unit", "reference_range", "collected_at", "status"],
    )
    result = run([csv_path])

    assert len(result.patients) == 1
    patient = result.patients[0]
    assert len(patient.lab_results) == 1
    lab = patient.lab_results[0]
    assert lab.test_name == "Hemoglobin"
    assert lab.value == 11.8  # comma decimal, not eleven hundred
    assert lab.unit == "g/dL"  # inferred from catalog
    assert lab.reference_range == "12-17.5"
    assert lab.interpretation == "low"
    assert any(entry.rule == "lab.unit_from_catalog" for entry in patient.audit)
    assert any(issue.code == "csv_missing_test_name" for issue in result.issues)


def test_text_note_without_mrn_uses_name_and_dob(tmp_path: Path) -> None:
    notes = note(
        "UNKNOWN",
        "Solo, Test",
        "02/02/1982",
        "2026-01-01",
        "Chief Complaint: fatigue. Assessment: hypothyroidism. Plan: TSH and free T4.",
    )
    result = run([write_text(tmp_path / "note.txt", notes)])

    assert len(result.patients) == 1
    patient = result.patients[0]
    assert patient.mrn.startswith("SYN")
    assert patient.display_name == "Test Solo"
    assert len(patient.documents) == 1
    assert patient.encounters  # derived so FHIR references always resolve


def test_report_and_cleaned_files_are_written(tmp_path: Path, settings) -> None:
    payload = {
        "patients": [
            {
                "mrn": "900011",
                "first_name": "Report",
                "last_name": "Writer",
                "dob": "1970-01-01",
                "gender": "F",
            }
        ]
    }
    result = run([write_json(tmp_path / "ehr.json", payload)])
    paths = result.write(settings)

    assert paths["patients"].exists()
    assert paths["report"].exists()
    cleaned = json.loads(paths["patients"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert cleaned[0]["mrn"] == "00900011"
    assert report["stats"]["patients"] == 1


def test_demo_dataset_smoke() -> None:
    result = IngestionPipeline.from_paths("data/raw").run()

    assert len(result.patients) >= 50
    assert sum(len(p.documents) for p in result.patients) >= 60
    assert sum(len(p.lab_results) for p in result.patients) >= 150
    assert result.report["stats"]["duplicates_removed"]["documents"] >= 1
    assert result.report["stats"]["duplicates_removed"]["labs"] >= 5
    assert result.report["conflicts"]

    codes = {flag.code for p in result.patients for flag in p.data_quality_flags}
    assert {
        "critical_lab",
        "missing_birth_date",
        "synthetic_mrn",
        "possible_duplicate_patient",
        "conflicting_birth_date",
    } <= codes

    # Every document points at an encounter that exists for that patient.
    for patient in result.patients:
        encounter_ids = {encounter.id for encounter in patient.encounters}
        for document in patient.documents:
            assert document.encounter_id in encounter_ids
