"""JSON EHR export adapter.

Accepts a wide range of shapes: a list of patients, ``{"patients": [...]}``,
NDJSON, nested ``{"patient": {...}, "documents": [...]}`` objects, and
FHIR-flavoured fields (identifier lists, HumanName lists, referenceRange
objects). Unknown/extra keys are preserved in ``RawPatient.raw`` for audit.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from ..ingestion.common import (
    as_list,
    first_non_empty,
    flatten_address,
    load_json_records,
    pick,
)
from ..ingestion.raw import (
    IngestionIssue,
    RawDocument,
    RawEncounter,
    RawLab,
    RawPatient,
    SourceResult,
)

_DOCUMENT_KEYS = ("documents", "docs", "media", "notes", "reports", "clinical_documents")
_LAB_KEYS = ("labs", "lab_results", "labResults", "results", "observations", "diagnostics")
_ENCOUNTER_KEYS = ("encounters", "visits", "admissions")
_INLINE_NOTE_KEYS = {
    "notetext",
    "clinicalnote",
    "progressnote",
    "dischargesummary",
    "imagingreport",
    "consultationnote",
    "consultnote",
    "scannednote",
    "notebody",
}


class JsonEhrSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = self.path.name

    def load(self) -> SourceResult:
        result = SourceResult()
        records, warning = load_json_records(self.path)
        if warning:
            result.issues.append(
                IngestionIssue(
                    source=self.name,
                    row_ref="$",
                    code="json_parse_warning",
                    message=warning,
                    severity="warning",
                )
            )
        for index, item in enumerate(records):
            row_ref = f"{self.name}#{index}"
            if not isinstance(item, dict):
                result.issues.append(
                    IngestionIssue(
                        source=self.name,
                        row_ref=row_ref,
                        code="json_record_not_object",
                        message="Skipped JSON record: expected an object",
                        severity="error",
                        raw_excerpt=str(item)[:180],
                    )
                )
                continue
            try:
                result.records.append(self._parse_patient(item, row_ref))
            except Exception as exc:  # defensive: one bad row must not stop ingestion
                result.issues.append(
                    IngestionIssue(
                        source=self.name,
                        row_ref=row_ref,
                        code="json_record_error",
                        message=f"Could not parse JSON record: {exc}",
                        severity="error",
                        raw_excerpt=str(item)[:180],
                    )
                )
        return result

    # ------------------------------------------------------------------
    def _parse_patient(self, item: dict[str, Any], row_ref: str) -> RawPatient:
        nested = pick(item, "patient", "demographics")
        base: dict[str, Any] = {}
        if isinstance(nested, dict):
            base.update(nested)
        base.update({key: value for key, value in item.items() if key != "patient"})

        given, family, middle, suffix = self._parse_name(base)
        address = flatten_address(
            first_non_empty(
                pick(base, "address", "home_address", "mailing_address"),
                {
                    "line": pick(base, "address_line", "address1", "street"),
                    "city": pick(base, "city"),
                    "state": pick(base, "state", "province"),
                    "postal_code": pick(base, "postal_code", "zip", "zipcode"),
                },
            )
        )
        phone = self._parse_telecom(base, "phone", "telecom", "phone_number", "mobile")
        email = self._parse_telecom(base, "email", "email_address")
        full_name = pick(base, "full_name", "patient_name", "display_name")
        if not isinstance(full_name, str):
            full_name = None

        patient = RawPatient(
            mrn=self._parse_identifier(base),
            given_name=given,
            family_name=family,
            middle_name=middle,
            suffix=suffix,
            full_name=full_name,
            birth_date=pick(base, "birth_date", "date_of_birth", "dob", "birthDate"),
            gender=pick(base, "gender", "sex", "administrative_gender"),
            phone=phone,
            email=email,
            address_line=address["address_line"],
            city=address["city"],
            state=address["state"],
            postal_code=address["postal_code"],
            preferred_language=pick(base, "preferred_language", "language", "locale"),
            source=self.name,
            source_format="json",
            row_ref=row_ref,
            raw=item,
        )

        patient.documents = self._parse_documents(base)
        patient.labs = self._parse_labs(base)
        patient.encounters = self._parse_encounters(base)
        patient.documents.extend(self._parse_inline_notes(base))
        return patient

    # -- demographics ---------------------------------------------------
    @staticmethod
    def _parse_identifier(base: dict[str, Any]) -> Any:
        value = pick(
            base,
            "mrn",
            "medical_record_number",
            "medicalRecordNumber",
            "patient_id",
            "patientId",
            "identifier",
        )
        fallback: Any = None
        for candidate in as_list(value):
            if isinstance(candidate, dict):
                candidate_value = candidate.get("value")
                if not candidate_value:
                    continue
                if fallback is None:
                    fallback = candidate_value
                type_coding = candidate.get("type") or {}
                coding = type_coding.get("coding") if isinstance(type_coding, dict) else None
                labels = " ".join(
                    str(code.get("display") or code.get("code") or "")
                    for code in as_list(coding)
                    if isinstance(code, dict)
                ).lower()
                system = str(candidate.get("system") or "").lower()
                if "mrn" in labels or "medical record" in labels or "mrn" in system:
                    return candidate_value
            elif candidate:
                return candidate
        return fallback

    @staticmethod
    def _parse_name(base: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
        name_value = pick(base, "name", "patient_name", "full_name", "display_name")
        if isinstance(name_value, list):
            name_value = name_value[0] if name_value else None
        if isinstance(name_value, dict):
            given_list = as_list(name_value.get("given"))
            return (
                given_list[0] if given_list else None,
                name_value.get("family"),
                given_list[1] if len(given_list) > 1 else None,
                name_value.get("suffix") if not isinstance(name_value.get("suffix"), list) else (name_value.get("suffix") or [None])[0],
            )
        given = pick(base, "given_name", "given", "first_name", "firstname", "first")
        family = pick(base, "family_name", "family", "last_name", "lastname", "last", "surname")
        middle = pick(base, "middle_name", "middlename", "middle")
        suffix = pick(base, "suffix", "name_suffix")
        if not given and not family and isinstance(name_value, str):
            return None, None, None, suffix
        return given, family, middle, suffix

    @staticmethod
    def _parse_telecom(base: dict[str, Any], *keys: str) -> str | None:
        value = pick(base, *keys)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    system = str(entry.get("system") or "").lower()
                    if keys[0] == "email" and system and system != "email":
                        continue
                    if keys[0] == "phone" and system and system not in {"phone", "sms", "mobile"}:
                        continue
                    if entry.get("value"):
                        return entry["value"]
                elif entry:
                    return str(entry)
            return None
        if isinstance(value, dict):
            return value.get("value")
        return value

    # -- documents ------------------------------------------------------
    def _parse_documents(self, base: dict[str, Any]) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for key in _DOCUMENT_KEYS:
            for entry in as_list(pick(base, key)):
                document = self._parse_document(entry)
                if document:
                    documents.append(document)
        return documents

    def _parse_document(self, entry: Any) -> RawDocument | None:
        if isinstance(entry, str):
            return RawDocument(
                text=entry, source=self.name, source_format="json"
            ) if entry.strip() else None
        if not isinstance(entry, dict):
            return None

        text = pick(
            entry,
            "text",
            "content",
            "body",
            "note",
            "narrative",
            "transcript",
            "report_text",
            "note_text",
            "value",
        )
        if not text:
            attachment = pick(entry, "attachment", "content_attachment")
            if isinstance(attachment, dict):
                text = self._decode_attachment(attachment)
                if not text:
                    text = attachment.get("title") or attachment.get("url")
        if isinstance(text, dict):
            text = text.get("text") or text.get("value") or text.get("div")
        if not text or not str(text).strip():
            return None

        author = pick(entry, "author", "author_name", "provider", "clinician")
        if isinstance(author, dict):
            author = author.get("display") or author.get("text")
        elif isinstance(author, list):
            author = next(
                (
                    item.get("display") if isinstance(item, dict) else str(item)
                    for item in author
                    if item
                ),
                None,
            )

        return RawDocument(
            text=str(text),
            doc_type=pick(entry, "doc_type", "document_type", "type", "kind", "category", "note_type"),
            title=pick(entry, "title", "subject", "document_title", "description"),
            authored_on=pick(
                entry,
                "authored_on",
                "date",
                "date_of_service",
                "service_date",
                "document_date",
                "created_at",
                "datetime",
                "effective_date",
            ),
            author=author,
            encounter_ref=pick(entry, "encounter_id", "encounter", "visit_id"),
            source=self.name,
            source_format="json",
        )

    @staticmethod
    def _decode_attachment(attachment: dict[str, Any]) -> str | None:
        data = attachment.get("data")
        if isinstance(data, str) and data:
            try:
                decoded = base64.b64decode(data)
                return decoded.decode("utf-8", errors="replace")
            except (binascii.Error, ValueError):
                return None
        return None

    def _parse_inline_notes(self, base: dict[str, Any]) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for key, value in base.items():
            normalized = "".join(ch for ch in key.lower() if ch.isalnum())
            if normalized in _INLINE_NOTE_KEYS and isinstance(value, str) and len(value.strip()) > 20:
                documents.append(
                    RawDocument(
                        text=value,
                        doc_type=key,
                        source=self.name,
                        source_format="json",
                    )
                )
        return documents

    # -- labs -----------------------------------------------------------
    def _parse_labs(self, base: dict[str, Any]) -> list[RawLab]:
        labs: list[RawLab] = []
        for key in _LAB_KEYS:
            for entry in as_list(pick(base, key)):
                lab = self._parse_lab(entry)
                if lab:
                    labs.append(lab)
        return labs

    def _parse_lab(self, entry: Any) -> RawLab | None:
        if isinstance(entry, str):
            return None
        if not isinstance(entry, dict):
            return None

        test = pick(entry, "test_name", "test", "name", "analyte", "loinc_display", "display")
        if isinstance(test, dict):
            test = test.get("text") or test.get("display")
            if not test:
                coding = as_list(test.get("coding") if isinstance(test, dict) else None)
                test = coding[0].get("display") if coding and isinstance(coding[0], dict) else None

        value: Any = pick(entry, "value", "result", "value_quantity", "valueQuantity", "result_value")
        unit = pick(entry, "unit", "units", "uom", "unit_of_measure")
        if isinstance(value, dict):
            unit = unit or value.get("unit")
            value = value.get("value", value.get("text"))

        loinc = pick(entry, "loinc_code", "loinc", "code")
        if isinstance(loinc, dict):
            coding = as_list(loinc.get("coding"))
            loinc = coding[0].get("code") if coding and isinstance(coding[0], dict) else loinc.get("code")

        if not test:
            return None
        return RawLab(
            test_name=str(test),
            value=value,
            unit=str(unit) if unit else None,
            reference_range=self._parse_reference_range(
                pick(entry, "reference_range", "ref_range", "reference", "normal_range", "referenceRange")
            ),
            collected_at=pick(
                entry,
                "collected_at",
                "date",
                "effective_date",
                "result_date",
                "timestamp",
                "effectiveDateTime",
            ),
            status=pick(entry, "status"),
            loinc_code=str(loinc) if loinc else None,
            source=self.name,
            source_format="json",
        )

    @staticmethod
    def _parse_reference_range(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            low = value.get("low")
            high = value.get("high")
            low = low.get("value") if isinstance(low, dict) else low
            high = high.get("value") if isinstance(high, dict) else high
            if low is not None and high is not None:
                return f"{low}-{high}"
            if high is not None:
                return f"<{high}"
            if low is not None:
                return f">{low}"
            return value.get("text")
        if isinstance(value, list) and value:
            return JsonEhrSource._parse_reference_range(value[0])
        return str(value)

    # -- encounters -----------------------------------------------------
    def _parse_encounters(self, base: dict[str, Any]) -> list[RawEncounter]:
        encounters: list[RawEncounter] = []
        for key in _ENCOUNTER_KEYS:
            for entry in as_list(pick(base, key)):
                if not isinstance(entry, dict):
                    continue
                class_value = pick(entry, "class", "class_code", "encounter_class", "type")
                if isinstance(class_value, dict):
                    class_value = class_value.get("code") or class_value.get("display")
                period = pick(entry, "period")
                start = pick(entry, "start_date", "start", "admission_date", "date")
                end = pick(entry, "end_date", "end", "discharge_date")
                if isinstance(period, dict):
                    start = start or period.get("start")
                    end = end or period.get("end")
                encounters.append(
                    RawEncounter(
                        class_code=str(class_value) if class_value else None,
                        reason=pick(entry, "reason", "reason_code", "chief_complaint", "description"),
                        start_date=start,
                        end_date=end,
                        source=self.name,
                    )
                )
        return encounters
