"""Map canonical :class:`PatientRecord` objects to FHIR R4 resources.

Resources are emitted as plain FHIR JSON dictionaries so they can be stored,
validated against the official R4 JSON Schema, and returned by the API without
serialization surprises. ``fhir.resources`` R4B models are used by the
validator for structural checks.

Resource references use relative RESTful references (``Patient/{id}``,
``Encounter/{id}``) and every reference points to an entry inside the same
Bundle.
"""

from __future__ import annotations

import base64
from datetime import date, datetime, timezone
from typing import Any

from ..ids import bundle_id_for, report_id_for
from ..models import ClinicalDocument, Encounter, LabResult, PatientRecord

MRN_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/identifier/mrn"
ENCOUNTER_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/identifier/encounter"
DOCUMENT_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/identifier/document"
REPORT_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/identifier/diagnostic-report"
BUNDLE_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/identifier/bundle"
DATA_QUALITY_SYSTEM = "https://ehr-media-intelligence.example.org/fhir/data-quality"

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
V3_ACT_CODE = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
V3_OBSERVATION_INTERPRETATION = (
    "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
)
OBSERVATION_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
V2_0074 = "http://terminology.hl7.org/CodeSystem/v2-0074"
US_CORE_DOCUMENT_CATEGORY = (
    "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category"
)
DATA_ABSENT_REASON = "http://terminology.hl7.org/CodeSystem/data-absent-reason"

ENCOUNTER_CLASS_DISPLAY = {
    "AMB": "ambulatory",
    "IMP": "inpatient encounter",
    "EMER": "emergency",
    "OBSENC": "observation encounter",
}

DOCUMENT_TYPE_CODING = {
    "progress_note": ("11506-3", "Progress note"),
    "consultation": ("11488-4", "Consult note"),
    "discharge_summary": ("18842-5", "Discharge summary"),
    "imaging_report": ("18748-4", "Diagnostic imaging report"),
    "lab_report": ("11502-2", "Laboratory report"),
    "scanned_note": ("34109-9", "Note"),
    "other": ("34109-9", "Note"),
}

INTERPRETATION_CODING = {
    "normal": ("N", "Normal"),
    "low": ("L", "Low"),
    "high": ("H", "High"),
    "critical": ("AA", "Critical abnormal"),
}


def _instant(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return f"{value.isoformat()}T00:00:00Z"


def _reference(resource_type: str, resource_id: str) -> dict[str, str]:
    return {"reference": f"{resource_type}/{resource_id}"}


def _codeable(
    system: str | None,
    code: str | None,
    display: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    concept: dict[str, Any] = {}
    if system and code:
        concept["coding"] = [{"system": system, "code": code, "display": display or code}]
    if text:
        concept["text"] = text
    return concept


class FhirMapper:
    """Builds a FHIR R4 Bundle for one cleaned patient."""

    def __init__(self, patient: PatientRecord) -> None:
        self.patient = patient
        self._encounters = list(patient.encounters)
        self._observations = [self._observation(lab) for lab in patient.lab_results]
        self._reports = self._diagnostic_reports(patient.lab_results)
        self._documents = [self._document_reference(doc) for doc in patient.documents]

    # ------------------------------------------------------------------
    def build_bundle(self) -> dict[str, Any]:
        resources: list[dict[str, Any]] = [self._patient()]
        resources.extend(self._encounter(encounter) for encounter in self._encounters)
        resources.extend(self._observations)
        resources.extend(self._reports)
        resources.extend(self._documents)

        entries = []
        for resource in resources:
            resource_type = resource["resourceType"]
            resource_id = resource["id"]
            entries.append(
                {
                    "fullUrl": f"{resource_type}/{resource_id}",
                    "resource": resource,
                }
            )
        return {
            "resourceType": "Bundle",
            "id": bundle_id_for(self.patient.id),
            "identifier": {"system": BUNDLE_SYSTEM, "value": self.patient.mrn},
            "type": "collection",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entry": entries,
        }

    # ------------------------------------------------------------------
    def _patient(self) -> dict[str, Any]:
        patient = self.patient
        name: dict[str, Any] = {
            "use": "official",
            "family": patient.family_name,
            "given": [patient.given_name],
        }
        if patient.middle_name:
            name["given"].append(patient.middle_name)
        if patient.suffix:
            name["suffix"] = [patient.suffix]

        resource: dict[str, Any] = {
            "resourceType": "Patient",
            "id": patient.id,
            "meta": {
                "tag": [
                    {
                        "system": DATA_QUALITY_SYSTEM,
                        "code": flag.code,
                        "display": flag.message[:120],
                    }
                    for flag in patient.data_quality_flags
                ]
            },
            "identifier": [
                {
                    "use": "official",
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                                "display": "Medical record number",
                            }
                        ]
                    },
                    "system": MRN_SYSTEM,
                    "value": patient.mrn,
                }
            ],
            "active": True,
            "name": [name],
            "gender": patient.gender,
        }
        if patient.birth_date:
            resource["birthDate"] = patient.birth_date.isoformat()

        telecom = []
        if patient.phone:
            telecom.append({"system": "phone", "value": patient.phone, "use": "home"})
        if patient.email:
            telecom.append({"system": "email", "value": patient.email, "use": "home"})
        if telecom:
            resource["telecom"] = telecom

        address: dict[str, Any] = {"use": "home"}
        if patient.address_line:
            address["line"] = [patient.address_line]
        if patient.city:
            address["city"] = patient.city
        if patient.state:
            address["state"] = patient.state
        if patient.postal_code:
            address["postalCode"] = patient.postal_code
        if len(address) > 1:
            resource["address"] = [address]

        if patient.preferred_language:
            resource["communication"] = [
                {
                    "language": {
                        "coding": [
                            {
                                "system": "urn:ietf:bcp:47",
                                "code": "en",
                                "display": "English",
                            }
                        ],
                        "text": patient.preferred_language,
                    },
                    "preferred": True,
                }
            ]
        return resource

    def _encounter(self, encounter: Encounter) -> dict[str, Any]:
        resource: dict[str, Any] = {
            "resourceType": "Encounter",
            "id": encounter.id,
            "identifier": [{"system": ENCOUNTER_SYSTEM, "value": encounter.id}],
            "status": "finished",
            "class": {
                "system": V3_ACT_CODE,
                "code": encounter.class_code,
                "display": ENCOUNTER_CLASS_DISPLAY[encounter.class_code],
            },
            "subject": _reference("Patient", self.patient.id),
        }
        if encounter.reason:
            resource["type"] = [{"text": encounter.reason}]
        period: dict[str, Any] = {}
        if encounter.start_date:
            period["start"] = _instant(encounter.start_date)
        if encounter.end_date:
            period["end"] = _instant(encounter.end_date)
        if period:
            resource["period"] = period
        return resource

    def _document_reference(self, document: ClinicalDocument) -> dict[str, Any]:
        loinc_code, loinc_display = DOCUMENT_TYPE_CODING.get(
            document.doc_type, DOCUMENT_TYPE_CODING["other"]
        )
        encoded = base64.b64encode(document.text.encode("utf-8")).decode("ascii")
        attachment: dict[str, Any] = {
            "contentType": "text/plain; charset=utf-8",
            "data": encoded,
            "title": document.title,
            "size": len(document.text.encode("utf-8")),
        }
        if document.authored_on:
            attachment["creation"] = _instant(document.authored_on)
        resource: dict[str, Any] = {
            "resourceType": "DocumentReference",
            "id": document.id,
            "identifier": [{"system": DOCUMENT_SYSTEM, "value": document.id}],
            "status": "current",
            "docStatus": "final",
            "type": {
                "coding": [
                    {"system": LOINC, "code": loinc_code, "display": loinc_display}
                ],
                "text": document.title,
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": US_CORE_DOCUMENT_CATEGORY,
                            "code": "clinical-note",
                            "display": "Clinical Note",
                        }
                    ]
                }
            ],
            "subject": _reference("Patient", self.patient.id),
            "description": document.title,
            "content": [{"attachment": attachment}],
        }
        if document.authored_on:
            resource["date"] = _instant(document.authored_on)
        if document.author:
            # Display-only reference avoids dangling Practitioner references.
            resource["author"] = [{"display": document.author}]
        if document.encounter_id:
            context: dict[str, Any] = {
                "encounter": [_reference("Encounter", document.encounter_id)]
            }
            if document.authored_on:
                context["period"] = {"start": _instant(document.authored_on)}
            resource["context"] = context
        return resource

    def _observation(self, lab: LabResult) -> dict[str, Any]:
        resource: dict[str, Any] = {
            "resourceType": "Observation",
            "id": lab.id,
            "identifier": [{"system": DOCUMENT_SYSTEM, "value": lab.id}],
            "status": lab.status if lab.status != "corrected" else "corrected",
            "category": [
                {
                    "coding": [
                        {
                            "system": OBSERVATION_CATEGORY,
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": _codeable(
                LOINC if lab.loinc_code else None,
                lab.loinc_code,
                lab.test_name,
                lab.test_name,
            ),
            "subject": _reference("Patient", self.patient.id),
        }
        encounter = self._encounter_for(lab.collected_at)
        if encounter:
            resource["encounter"] = _reference("Encounter", encounter.id)
        if lab.collected_at:
            resource["effectiveDateTime"] = _instant(lab.collected_at)

        if lab.value is not None:
            unit = lab.unit or "1"
            resource["valueQuantity"] = {
                "value": lab.value,
                "unit": unit,
                "system": UCUM,
                "code": unit,
            }
        elif lab.value_text:
            resource["valueString"] = lab.value_text
        else:
            resource["dataAbsentReason"] = _codeable(
                DATA_ABSENT_REASON, "unknown", "Unknown", "No result value recorded"
            )

        interpretation = INTERPRETATION_CODING.get(lab.interpretation)
        if interpretation:
            resource["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": V3_OBSERVATION_INTERPRETATION,
                            "code": interpretation[0],
                            "display": interpretation[1],
                        }
                    ]
                }
            ]

        reference_range: dict[str, Any] = {}
        if lab.reference_low is not None:
            reference_range["low"] = {
                "value": lab.reference_low,
                "unit": lab.unit or "1",
                "system": UCUM,
                "code": lab.unit or "1",
            }
        if lab.reference_high is not None:
            reference_range["high"] = {
                "value": lab.reference_high,
                "unit": lab.unit or "1",
                "system": UCUM,
                "code": lab.unit or "1",
            }
        if lab.reference_range:
            reference_range["text"] = lab.reference_range
        if reference_range:
            resource["referenceRange"] = [reference_range]
        return resource

    def _diagnostic_reports(self, labs: list[LabResult]) -> list[dict[str, Any]]:
        grouped: dict[str, list[LabResult]] = {}
        for lab in labs:
            key = lab.collected_at.isoformat() if lab.collected_at else "undated"
            grouped.setdefault(key, []).append(lab)

        reports: list[dict[str, Any]] = []
        for key, group in grouped.items():
            collected = group[0].collected_at
            report_id = report_id_for(self.patient.id, key)
            abnormal = [lab for lab in group if lab.is_abnormal]
            critical = [lab for lab in group if lab.interpretation == "critical"]
            conclusion_parts = []
            if critical:
                conclusion_parts.append(
                    f"{len(critical)} critical result(s): "
                    + ", ".join(f"{lab.test_name} {lab.display_value}" for lab in critical[:4])
                )
            if abnormal:
                conclusion_parts.append(
                    f"{len(abnormal)} abnormal result(s) of {len(group)} total"
                )
            if not abnormal:
                conclusion_parts.append(
                    f"{len(group)} result(s) within reference range where available"
                )

            statuses = {lab.status for lab in group}
            if statuses <= {"final", "corrected"}:
                status = "final"
            elif "preliminary" in statuses:
                status = "preliminary"
            elif statuses == {"cancelled"}:
                status = "cancelled"
            else:
                status = "unknown"

            resource: dict[str, Any] = {
                "resourceType": "DiagnosticReport",
                "id": report_id,
                "identifier": [{"system": REPORT_SYSTEM, "value": report_id}],
                "status": status,
                "category": [
                    {
                        "coding": [
                            {"system": V2_0074, "code": "LAB", "display": "Laboratory"}
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": LOINC,
                            "code": "11502-2",
                            "display": "Laboratory report",
                        }
                    ],
                    "text": "Laboratory panel",
                },
                "subject": _reference("Patient", self.patient.id),
                "result": [
                    _reference("Observation", lab.id)
                    for lab in sorted(group, key=lambda item: item.test_name)
                ],
                "conclusion": "; ".join(conclusion_parts)[:1000],
            }
            if collected:
                resource["effectiveDateTime"] = _instant(collected)
                resource["issued"] = _instant(collected)
            encounter = self._encounter_for(collected)
            if encounter:
                resource["encounter"] = _reference("Encounter", encounter.id)
            reports.append(resource)
        reports.sort(key=lambda item: item.get("effectiveDateTime", ""), reverse=True)
        return reports

    def _encounter_for(self, on: date | None) -> Encounter | None:
        if not self._encounters:
            return None
        if on is None:
            return self._encounters[0]
        same_day = [enc for enc in self._encounters if enc.start_date == on]
        if same_day:
            return same_day[0]
        earlier = [
            enc for enc in self._encounters if enc.start_date is not None and enc.start_date <= on
        ]
        if earlier:
            return max(earlier, key=lambda enc: enc.start_date or date.min)
        return self._encounters[0]


def map_patient(patient: PatientRecord) -> dict[str, Any]:
    """Convenience wrapper used by the pipeline and tests."""
    return FhirMapper(patient).build_bundle()
