"""Unit tests for the deterministic normalizers (Task 1 edge cases)."""

from __future__ import annotations

from datetime import date

import pytest

from ehr_media.normalizers import (
    detect_anomalies,
    interpret_lab,
    lookup_lab,
    normalize_doc_type,
    normalize_gender,
    normalize_lab_name,
    normalize_mrn,
    normalize_name,
    normalize_phone,
    normalize_state,
    parse_date,
    parse_lab_value,
    parse_reference_range,
    split_comma_name,
)


class TestDates:
    def test_iso_and_compact(self) -> None:
        assert parse_date("2026-03-05").value == date(2026, 3, 5)
        assert parse_date("20260305").value == date(2026, 3, 5)
        assert parse_date("2026-03-05T14:22:00Z").value == date(2026, 3, 5)

    def test_us_and_unambiguous_day_first(self) -> None:
        us = parse_date("03/05/2026")
        assert us.value == date(2026, 3, 5)
        assert us.rule == "date.month_first"
        assert us.warning is not None  # ambiguous input is surfaced, not hidden

        dmy = parse_date("18/04/1975")
        assert dmy.value == date(1975, 4, 18)
        assert dmy.rule == "date.day_first"

    def test_named_month_and_two_digit_year(self) -> None:
        assert parse_date("5-Mar-2026").value == date(2026, 3, 5)
        assert parse_date("March 5, 2026").value == date(2026, 3, 5)
        assert parse_date("3/5/26").value == date(2026, 3, 5)

    def test_invalid_future_and_sentinel_dates(self) -> None:
        assert parse_date("13/45/2025").value is None
        assert parse_date("13/45/2025").rule == "date.invalid"
        assert parse_date("2999-01-01").rule == "date.future"
        assert parse_date("N/A").value is None
        assert parse_date("").value is None

    def test_implausible_year(self) -> None:
        assert parse_date("1850-01-01").rule == "date.out_of_range"


class TestDemographics:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("M", "male"),
            ("1", "male"),
            ("male", "male"),
            ("F ", "female"),
            ("2", "female"),
            ("Non-binary", "other"),
            ("U", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_gender_codes(self, raw: str, expected: str) -> None:
        assert normalize_gender(raw).value == expected

    def test_unrecognized_gender_is_flagged(self) -> None:
        parsed = normalize_gender("xenomorph")
        assert parsed.value == "unknown"
        assert parsed.recognized is False
        assert parsed.warning

    def test_mrn_formats(self) -> None:
        assert normalize_mrn("MRN-1234").value == "00001234"
        assert normalize_mrn("  mrn 1234 ").value == "00001234"
        assert normalize_mrn(100455).value == "00100455"
        assert normalize_mrn("A-7781").value == "A7781"
        assert normalize_mrn("N/A").value is None

    def test_names(self) -> None:
        assert normalize_name("  SMITH ,  JOHN ") == "Smith, John"
        assert normalize_name("Dr. Eleanor Vance") == "Eleanor Vance"
        assert normalize_name("o'connor") == "O'Connor"
        assert normalize_name("de la cruz") == "De la Cruz"
        assert split_comma_name("Bell, Marcus") == ("Marcus", "Bell")

    def test_phone_and_state(self) -> None:
        assert normalize_phone("(212) 555-0142")[0] == "+12125550142"
        assert normalize_phone("1-646-555-0198")[0] == "+16465550198"
        assert normalize_phone("555-CALL")[0] is None
        assert normalize_state("New York") == "NY"
        assert normalize_state("ny") == "NY"
        assert normalize_state("Ontario") is None


class TestLabs:
    def test_value_qualifiers_and_comma_decimals(self) -> None:
        parsed = parse_lab_value("<0.5 mg/dL")
        assert parsed.value == 0.5
        assert parsed.qualifier == "<"
        assert parsed.unit == "mg/dL"

        assert parse_lab_value("11,8").value == 11.8
        assert parse_lab_value("1,234").value == 1234.0
        assert parse_lab_value(">200").value == 200.0
        assert parse_lab_value("NEGATIVE").value_text == "NEGATIVE"

    def test_reference_ranges(self) -> None:
        assert parse_reference_range("3.5-5.1")[:2] == (3.5, 5.1)
        assert parse_reference_range("<100")[:2] == (None, 100.0)
        assert parse_reference_range(">60")[:2] == (60.0, None)
        assert parse_reference_range("3.5 - 5.1 mmol/L")[:2] == (3.5, 5.1)
        assert parse_reference_range("garbage")[:2] == (None, None)

    def test_alias_catalog_and_loinc(self) -> None:
        assert normalize_lab_name("HGB") == ("Hemoglobin", "718-7")
        assert normalize_lab_name("a1c") == ("Hemoglobin A1c", "4548-4")
        # Regression: UACR must not be mistaken for serum creatinine.
        assert normalize_lab_name("Urine albumin-creatinine ratio")[0] == (
            "Urine Albumin-Creatinine Ratio"
        )
        assert lookup_lab("serum creatinine").loinc == "2160-0"  # type: ignore[union-attr]

    def test_interpretation_including_critical_thresholds(self) -> None:
        assert interpret_lab("Potassium", 6.8, 3.5, 5.1) == "critical"
        assert interpret_lab("Hemoglobin", 11.8, 12.0, 17.5) == "low"
        assert interpret_lab("Glucose", 168, 70, 140) == "high"
        assert interpret_lab("Sodium", 139, 135, 145) == "normal"
        assert interpret_lab("Sodium", None, 135, 145) == "unknown"


class TestTextAnalysis:
    def test_document_type_inference(self) -> None:
        assert normalize_doc_type("Discharge Summary") == "discharge_summary"
        assert normalize_doc_type("CT Chest") == "imaging_report"
        assert normalize_doc_type("Scanned Note") == "scanned_note"

    def test_anomaly_detection_respects_negation(self) -> None:
        codes = {a.code for a in detect_anomalies("No acute infarct. Pneumonia present.")}
        assert "infection_imaging_finding" in codes
        assert "acute_neuro_finding" not in codes

        critical = detect_anomalies("Patient became septic with a positive blood culture.")
        assert {a.code for a in critical} >= {"sepsis_mention", "positive_blood_culture"}
