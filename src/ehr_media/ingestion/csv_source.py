"""CSV adapter for lab exports and flat patient lists.

The same adapter handles two common exports:

* **Lab results** - one row per analyte (``mrn,test_name,value,unit,range,date``)
* **Patient roster** - one row per patient (``mrn,first_name,last_name,dob,...``)

Schema is detected from the header row. Rows are never silently dropped:
malformed rows become :class:`IngestionIssue` entries and the pipeline report
counts them.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .common import normalize_key, read_text, truncate
from .raw import IngestionIssue, RawDocument, RawLab, RawPatient, SourceResult

_MRN_KEYS = {"mrn", "medicalrecordnumber", "medicalrecordno", "patientid", "acct", "account"}
_TEST_KEYS = {"testname", "test", "analyte", "labtest", "loincdisplay", "component", "testcode"}
_VALUE_KEYS = {"value", "result", "resultvalue", "labvalue", "observedvalue"}
_UNIT_KEYS = {"unit", "units", "uom", "unitofmeasure"}
_RANGE_KEYS = {"referencerange", "refrange", "normalrange", "reference", "range"}
_DATE_KEYS = {
    "collectedat",
    "collectiondate",
    "date",
    "resultdate",
    "collected",
    "timestamp",
    "dateofservice",
    "dos",
}
_STATUS_KEYS = {"status", "resultstatus"}
_GIVEN_KEYS = {"firstname", "givenname", "given", "first"}
_FAMILY_KEYS = {"lastname", "familyname", "family", "surname", "last"}
_FULL_NAME_KEYS = {"patientname", "name", "patient"}
_DOB_KEYS = {"dob", "dateofbirth", "birthdate", "birthday"}
_GENDER_KEYS = {"gender", "sex"}
_PHONE_KEYS = {"phone", "phonenumber", "telephone", "mobile"}
_EMAIL_KEYS = {"email", "emailaddress"}
_ADDRESS_KEYS = {"address", "addressline", "street", "address1"}
_CITY_KEYS = {"city", "town"}
_STATE_KEYS = {"state", "province"}
_ZIP_KEYS = {"zip", "zipcode", "postalcode", "postcode"}
_LANGUAGE_KEYS = {"language", "preferredlanguage"}
_NOTE_KEYS = {"note", "notes", "text", "document", "narrative", "notetext"}
_DOC_TYPE_KEYS = {"doctype", "documenttype", "notetype", "type"}
_TITLE_KEYS = {"title", "subject", "documenttitle"}
_AUTHOR_KEYS = {"author", "provider", "clinician"}


class CsvSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = self.path.name

    def load(self) -> SourceResult:
        result = SourceResult()
        text = read_text(self.path)
        if not text.strip():
            result.issues.append(
                IngestionIssue(self.name, "$", "csv_empty", "CSV file is empty", "warning")
            )
            return result

        reader = csv.DictReader(io.StringIO(text), delimiter=self._delimiter(text))
        if not reader.fieldnames:
            result.issues.append(
                IngestionIssue(self.name, "$", "csv_no_header", "CSV has no header row", "error")
            )
            return result

        header_map = {field: normalize_key(field) for field in reader.fieldnames if field}
        normalized_headers = set(header_map.values())
        is_lab = bool(normalized_headers & _TEST_KEYS) and bool(
            normalized_headers & _VALUE_KEYS
        )
        is_patient = bool(normalized_headers & _MRN_KEYS) and bool(
            normalized_headers & (_GIVEN_KEYS | _FAMILY_KEYS | _FULL_NAME_KEYS | _DOB_KEYS)
        )
        if not is_lab and not is_patient:
            result.issues.append(
                IngestionIssue(
                    self.name,
                    "$",
                    "csv_unknown_schema",
                    "CSV schema not recognized: expected lab columns (test_name/value) "
                    "or patient columns (mrn + name/dob)",
                    "error",
                )
            )
            return result

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not (value or "").strip() for value in row.values()):
                continue
            normalized = {
                header_map[key]: (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
                if key in header_map
            }
            row_ref = f"{self.name}:{line_number}"
            patient = self._patient_from_row(normalized, row_ref)
            if not patient.mrn and not patient.full_name and not patient.family_name:
                result.issues.append(
                    IngestionIssue(
                        self.name,
                        row_ref,
                        "csv_missing_identifier",
                        "Row has neither an MRN nor a patient name",
                        "warning",
                        truncate(row),
                    )
                )
            if is_lab:
                lab = self._lab_from_row(normalized, row_ref, result)
                if lab:
                    patient.labs.append(lab)
            note_text = self._first(normalized, _NOTE_KEYS)
            if note_text and len(note_text) > 5:
                patient.documents.append(
                    RawDocument(
                        text=note_text,
                        doc_type=self._first(normalized, _DOC_TYPE_KEYS),
                        title=self._first(normalized, _TITLE_KEYS),
                        authored_on=self._first(normalized, _DATE_KEYS),
                        author=self._first(normalized, _AUTHOR_KEYS),
                        source=self.name,
                        source_format="csv",
                    )
                )
            result.records.append(patient)
        return result

    @staticmethod
    def _delimiter(text: str) -> str:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","

    @staticmethod
    def _first(row: dict[str, Any], keys: set[str]) -> Any:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return value
        return None

    def _patient_from_row(self, row: dict[str, Any], row_ref: str) -> RawPatient:
        return RawPatient(
            mrn=self._first(row, _MRN_KEYS),
            given_name=self._first(row, _GIVEN_KEYS),
            family_name=self._first(row, _FAMILY_KEYS),
            full_name=self._first(row, _FULL_NAME_KEYS),
            birth_date=self._first(row, _DOB_KEYS),
            gender=self._first(row, _GENDER_KEYS),
            phone=self._first(row, _PHONE_KEYS),
            email=self._first(row, _EMAIL_KEYS),
            address_line=self._first(row, _ADDRESS_KEYS),
            city=self._first(row, _CITY_KEYS),
            state=self._first(row, _STATE_KEYS),
            postal_code=self._first(row, _ZIP_KEYS),
            preferred_language=self._first(row, _LANGUAGE_KEYS),
            source=self.name,
            source_format="csv",
            row_ref=row_ref,
            raw=dict(row),
        )

    def _lab_from_row(
        self, row: dict[str, Any], row_ref: str, result: SourceResult
    ) -> RawLab | None:
        test_name = self._first(row, _TEST_KEYS)
        if not test_name:
            result.issues.append(
                IngestionIssue(
                    self.name,
                    row_ref,
                    "csv_missing_test_name",
                    "Lab row skipped: no test name/analyte",
                    "error",
                    truncate(row),
                )
            )
            return None
        value = self._first(row, _VALUE_KEYS)
        if value is None:
            result.issues.append(
                IngestionIssue(
                    self.name,
                    row_ref,
                    "csv_missing_lab_value",
                    f"Lab row for {test_name!r} has no value",
                    "warning",
                )
            )
        return RawLab(
            test_name=str(test_name),
            value=value,
            unit=self._first(row, _UNIT_KEYS),
            reference_range=self._first(row, _RANGE_KEYS),
            collected_at=self._first(row, _DATE_KEYS),
            status=self._first(row, _STATUS_KEYS),
            source=self.name,
            source_format="csv",
        )
