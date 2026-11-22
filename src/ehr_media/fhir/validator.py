"""FHIR R4 validation: structural models, official JSON Schema, references.

Three independent gates are applied to every Bundle:

1. ``fhir.resources`` R4B model parsing (types, cardinality, date formats).
2. The official HL7 FHIR R4 JSON Schema (enumerations, patterns, required fields).
3. Project-specific referential-integrity and clinical business rules.

Every failure becomes a structured :class:`ValidationIssue` with a severity,
path, resource identity and human-readable message, so the generated report is
actionable rather than a stack trace.
"""

from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

from jsonschema import Draft7Validator
from pydantic import BaseModel, Field

from ..ids import bundle_id_for
from ..models import PatientRecord
from .mapper import MRN_SYSTEM

try:  # fhir.resources R4B models (Pydantic v1 compatibility API)
    from fhir.resources.R4B.bundle import Bundle as R4Bundle
    from fhir.resources.R4B.diagnosticreport import DiagnosticReport as R4DiagnosticReport
    from fhir.resources.R4B.documentreference import DocumentReference as R4DocumentReference
    from fhir.resources.R4B.encounter import Encounter as R4Encounter
    from fhir.resources.R4B.observation import Observation as R4Observation
    from fhir.resources.R4B.patient import Patient as R4Patient
except Exception:  # pragma: no cover - dependency is declared in pyproject
    R4Bundle = R4DiagnosticReport = R4DocumentReference = None  # type: ignore[assignment]
    R4Encounter = R4Observation = R4Patient = None  # type: ignore[assignment]

SCHEMA_PATH = Path(__file__).parent / "schemas" / "fhir-r4.schema.json.gz"

MODEL_MAP = {
    "Patient": R4Patient,
    "Encounter": R4Encounter,
    "DocumentReference": R4DocumentReference,
    "DiagnosticReport": R4DiagnosticReport,
    "Observation": R4Observation,
    "Bundle": R4Bundle,
}

GENDERS = {"male", "female", "other", "unknown"}
ENCOUNTER_STATUSES = {
    "planned", "arrived", "triaged", "in-progress", "onleave", "finished",
    "cancelled", "entered-in-error", "unknown",
}
DOCUMENT_STATUSES = {"current", "superseded", "entered-in-error"}
DOC_STATUSES = {"preliminary", "final", "amended", "entered-in-error"}
REPORT_STATUSES = {
    "registered", "partial", "preliminary", "final", "amended", "corrected",
    "appended", "cancelled", "entered-in-error", "unknown",
}
OBSERVATION_STATUSES = {
    "registered", "preliminary", "final", "amended", "corrected", "cancelled",
    "entered-in-error", "unknown",
}
INTERPRETATION_CODES = {"N", "L", "H", "AA", "A", "HU", "LU", "LL", "HH"}
ENCOUNTER_CLASSES = {"AMB", "IMP", "EMER", "OBSENC"}

Severity = Literal["error", "warning", "info"]
ValidationSource = Literal[
    "fhir.resources", "r4-json-schema", "referential-integrity", "business-rule"
]


class ValidationIssue(BaseModel):
    severity: Severity
    code: str
    path: str
    message: str
    resource_type: str | None = None
    resource_id: str | None = None
    source: ValidationSource


class BundleValidation(BaseModel):
    patient_id: str
    mrn: str
    bundle_id: str
    valid: bool
    resource_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


class ValidationReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "HL7 FHIR R4 (4.0.1)"
    validators: list[str] = Field(
        default_factory=lambda: [
            "fhir.resources R4B model parsing",
            "HL7 FHIR R4 JSON Schema",
            "referential integrity + clinical business rules",
        ]
    )
    total_bundles: int = 0
    valid_bundles: int = 0
    invalid_bundles: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    resource_counts: dict[str, int] = Field(default_factory=dict)
    bundles: list[BundleValidation] = Field(default_factory=list)

    @classmethod
    def from_validations(cls, validations: list[BundleValidation]) -> ValidationReport:
        counts: dict[str, int] = {}
        for validation in validations:
            for resource_type, count in validation.resource_counts.items():
                counts[resource_type] = counts.get(resource_type, 0) + count
        return cls(
            total_bundles=len(validations),
            valid_bundles=sum(1 for validation in validations if validation.valid),
            invalid_bundles=sum(1 for validation in validations if not validation.valid),
            total_errors=sum(validation.error_count for validation in validations),
            total_warnings=sum(validation.warning_count for validation in validations),
            resource_counts=dict(sorted(counts.items())),
            bundles=validations,
        )

    def write(self, json_path: Path, markdown_path: Path | None = None) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if markdown_path:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_markdown(self), encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


@lru_cache
def load_r4_schema() -> dict[str, Any]:
    with gzip.open(SCHEMA_PATH, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache
def r4_bundle_validator() -> Draft7Validator:
    schema = load_r4_schema()
    bundle_schema = dict(schema["definitions"]["Bundle"])
    bundle_schema["definitions"] = schema["definitions"]
    return Draft7Validator(bundle_schema)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class FhirValidator:
    def validate(self, bundle: dict[str, Any], patient: PatientRecord | None = None) -> BundleValidation:
        issues: list[ValidationIssue] = []
        issues.extend(self._validate_models(bundle))
        issues.extend(self._validate_schema(bundle))
        issues.extend(self._validate_references(bundle))
        issues.extend(self._validate_business_rules(bundle))

        issues = self._dedupe(issues)
        issues.sort(key=lambda issue: (issue.severity != "error", issue.path, issue.code))
        counts: dict[str, int] = {}
        for entry in bundle.get("entry", []):
            resource_type = entry.get("resource", {}).get("resourceType", "Unknown")
            counts[resource_type] = counts.get(resource_type, 0) + 1

        patient_id = patient.id if patient else self._patient_id(bundle)
        mrn = patient.mrn if patient else self._mrn(bundle)
        return BundleValidation(
            patient_id=patient_id,
            mrn=mrn,
            bundle_id=str(bundle.get("id") or bundle_id_for(patient_id)),
            valid=not any(issue.severity == "error" for issue in issues),
            resource_counts=dict(sorted(counts.items())),
            issues=issues,
        )

    # -- 1. fhir.resources models --------------------------------------
    def _validate_models(self, bundle: dict[str, Any]) -> Iterator[ValidationIssue]:
        if R4Bundle is None:
            yield ValidationIssue(
                severity="warning",
                code="validator_unavailable",
                path="$",
                message="fhir.resources is not installed; skipped model validation",
                source="fhir.resources",
            )
            return
        try:
            R4Bundle.parse_obj(bundle)
        except Exception as exc:  # pydantic v1 compatibility layer
            for error in self._pydantic_errors(exc):
                yield ValidationIssue(
                    severity="error",
                    code=f"model_{error.get('type', 'validation')}",
                    path=f"Bundle.{error.get('loc', '$')}",
                    message=str(error.get("msg", "model validation failed"))[:500],
                    source="fhir.resources",
                )

        for index, entry in enumerate(bundle.get("entry", [])):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            model = MODEL_MAP.get(resource_type)
            if model is None:
                yield ValidationIssue(
                    severity="warning",
                    code="unknown_resource_type",
                    path=f"entry[{index}].resource.resourceType",
                    message=f"Unsupported FHIR resource type {resource_type!r}",
                    resource_type=resource_type,
                    resource_id=resource.get("id"),
                    source="fhir.resources",
                )
                continue
            try:
                model.parse_obj(resource)
            except Exception as exc:
                for error in self._pydantic_errors(exc):
                    yield ValidationIssue(
                        severity="error",
                        code=f"model_{error.get('type', 'validation')}",
                        path=f"{resource_type}/{resource.get('id')}.{error.get('loc', '$')}",
                        message=str(error.get("msg", "model validation failed"))[:500],
                        resource_type=resource_type,
                        resource_id=resource.get("id"),
                        source="fhir.resources",
                    )

    @staticmethod
    def _pydantic_errors(exc: Exception) -> list[dict[str, Any]]:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                return [dict(error) for error in errors()]
            except Exception:  # pragma: no cover
                pass
        return [{"loc": "$", "msg": str(exc), "type": "validation"}]

    # -- 2. official R4 JSON Schema -------------------------------------
    def _validate_schema(self, bundle: dict[str, Any]) -> Iterator[ValidationIssue]:
        validator = r4_bundle_validator()
        for error in sorted(validator.iter_errors(bundle), key=lambda item: list(item.path)):
            resource_type, resource_id = self._resource_for_path(bundle, list(error.absolute_path))
            yield ValidationIssue(
                severity="error",
                code=f"schema_{error.validator or 'validation'}",
                path=self._json_path(list(error.absolute_path)),
                message=error.message[:700],
                resource_type=resource_type,
                resource_id=resource_id,
                source="r4-json-schema",
            )

    # -- 3. referential integrity ---------------------------------------
    def _validate_references(self, bundle: dict[str, Any]) -> Iterator[ValidationIssue]:
        entries = bundle.get("entry", [])
        if not entries:
            yield ValidationIssue(
                severity="error",
                code="empty_bundle",
                path="entry",
                message="Bundle has no entries",
                source="referential-integrity",
            )
            return

        by_reference: dict[str, dict[str, Any]] = {}
        full_urls: set[str] = set()
        for index, entry in enumerate(entries):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if not resource_id:
                yield ValidationIssue(
                    severity="error",
                    code="missing_resource_id",
                    path=f"entry[{index}].resource.id",
                    message=f"{resource_type or 'Resource'} entry has no id",
                    resource_type=resource_type,
                    source="referential-integrity",
                )
                continue
            reference = f"{resource_type}/{resource_id}"
            if reference in by_reference:
                yield ValidationIssue(
                    severity="error",
                    code="duplicate_resource",
                    path=f"entry[{index}]",
                    message=f"Duplicate resource {reference}",
                    resource_type=resource_type,
                    resource_id=resource_id,
                    source="referential-integrity",
                )
            by_reference[reference] = resource

            full_url = entry.get("fullUrl")
            if full_url:
                if full_url in full_urls:
                    yield ValidationIssue(
                        severity="error",
                        code="duplicate_full_url",
                        path=f"entry[{index}].fullUrl",
                        message=f"Duplicate fullUrl {full_url}",
                        resource_type=resource_type,
                        resource_id=resource_id,
                        source="referential-integrity",
                    )
                full_urls.add(full_url)
                if full_url != reference:
                    yield ValidationIssue(
                        severity="warning",
                        code="full_url_mismatch",
                        path=f"entry[{index}].fullUrl",
                        message=f"fullUrl {full_url!r} does not match {reference!r}",
                        resource_type=resource_type,
                        resource_id=resource_id,
                        source="referential-integrity",
                    )

        patient_refs = [
            reference for reference in by_reference if reference.startswith("Patient/")
        ]
        if len(patient_refs) != 1:
            yield ValidationIssue(
                severity="error",
                code="patient_count",
                path="entry",
                message=f"Bundle must contain exactly one Patient, found {len(patient_refs)}",
                source="referential-integrity",
            )
        patient_ref = patient_refs[0] if patient_refs else None

        for path, reference in self._iter_references(bundle):
            if reference.startswith("#"):
                continue
            if reference not in by_reference:
                yield ValidationIssue(
                    severity="error",
                    code="unresolved_reference",
                    path=path,
                    message=f"Reference {reference!r} does not resolve inside the Bundle",
                    source="referential-integrity",
                )

        for index, entry in enumerate(entries):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if resource_type in {
                "Encounter",
                "DocumentReference",
                "DiagnosticReport",
                "Observation",
            }:
                subject = (resource.get("subject") or {}).get("reference")
                if patient_ref and subject != patient_ref:
                    yield ValidationIssue(
                        severity="error",
                        code="invalid_subject",
                        path=f"entry[{index}].resource.subject.reference",
                        message=f"{resource_type} subject must reference {patient_ref}",
                        resource_type=resource_type,
                        resource_id=resource_id,
                        source="referential-integrity",
                    )
            if resource_type == "DiagnosticReport":
                for result_index, result in enumerate(resource.get("result", [])):
                    reference = result.get("reference", "")
                    target = by_reference.get(reference)
                    if target is None or target.get("resourceType") != "Observation":
                        yield ValidationIssue(
                            severity="error",
                            code="invalid_report_result",
                            path=f"entry[{index}].resource.result[{result_index}]",
                            message=f"DiagnosticReport.result {reference!r} must point to an Observation",
                            resource_type=resource_type,
                            resource_id=resource_id,
                            source="referential-integrity",
                        )
            if resource_type == "DocumentReference":
                context = resource.get("context") or {}
                for enc_index, encounter_ref in enumerate(context.get("encounter", [])):
                    reference = encounter_ref.get("reference", "")
                    target = by_reference.get(reference)
                    if target is None or target.get("resourceType") != "Encounter":
                        yield ValidationIssue(
                            severity="error",
                            code="invalid_document_encounter",
                            path=f"entry[{index}].resource.context.encounter[{enc_index}]",
                            message=f"DocumentReference context {reference!r} must point to an Encounter",
                            resource_type=resource_type,
                            resource_id=resource_id,
                            source="referential-integrity",
                        )

    # -- 4. clinical business rules -------------------------------------
    def _validate_business_rules(self, bundle: dict[str, Any]) -> Iterator[ValidationIssue]:
        if bundle.get("type") != "collection":
            yield ValidationIssue(
                severity="error",
                code="bundle_type",
                path="type",
                message="Bundle.type must be 'collection' for a per-patient export",
                source="business-rule",
            )
        timestamp = bundle.get("timestamp")
        if timestamp and not self._valid_instant(timestamp):
            yield ValidationIssue(
                severity="error",
                code="invalid_timestamp",
                path="timestamp",
                message=f"Bundle.timestamp {timestamp!r} is not a valid FHIR instant",
                source="business-rule",
            )

        for index, entry in enumerate(bundle.get("entry", [])):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            path = f"entry[{index}].resource"
            if resource_type == "Patient":
                yield from self._validate_patient(resource, path)
            elif resource_type == "Encounter":
                yield from self._validate_encounter(resource, path)
            elif resource_type == "DocumentReference":
                yield from self._validate_document(resource, path)
            elif resource_type == "DiagnosticReport":
                yield from self._validate_report(resource, path)
            elif resource_type == "Observation":
                yield from self._validate_observation(resource, path)

    def _validate_patient(self, resource: dict[str, Any], path: str) -> Iterator[ValidationIssue]:
        if not any(
            identifier.get("system") == MRN_SYSTEM and identifier.get("value")
            for identifier in resource.get("identifier", [])
        ):
            yield ValidationIssue(
                severity="error",
                code="patient_missing_mrn",
                path=f"{path}.identifier",
                message=f"Patient has no MRN identifier in system {MRN_SYSTEM}",
                resource_type="Patient",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if not any(name.get("family") for name in resource.get("name", [])):
            yield ValidationIssue(
                severity="warning",
                code="patient_missing_family_name",
                path=f"{path}.name",
                message="Patient has no family name (placeholder may have been used)",
                resource_type="Patient",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if resource.get("gender") not in GENDERS:
            yield ValidationIssue(
                severity="error",
                code="invalid_gender",
                path=f"{path}.gender",
                message=f"Patient.gender {resource.get('gender')!r} must be one of {sorted(GENDERS)}",
                resource_type="Patient",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        birth_date = resource.get("birthDate")
        if birth_date:
            parsed = self._parse_date(birth_date)
            if parsed is None:
                yield ValidationIssue(
                    severity="error",
                    code="invalid_birth_date",
                    path=f"{path}.birthDate",
                    message=f"Patient.birthDate {birth_date!r} is not a valid FHIR date",
                    resource_type="Patient",
                    resource_id=resource.get("id"),
                    source="business-rule",
                )
            elif parsed > date.today():
                yield ValidationIssue(
                    severity="error",
                    code="future_birth_date",
                    path=f"{path}.birthDate",
                    message="Patient.birthDate is in the future",
                    resource_type="Patient",
                    resource_id=resource.get("id"),
                    source="business-rule",
                )

    def _validate_encounter(self, resource: dict[str, Any], path: str) -> Iterator[ValidationIssue]:
        if resource.get("status") not in ENCOUNTER_STATUSES:
            yield ValidationIssue(
                severity="error",
                code="invalid_encounter_status",
                path=f"{path}.status",
                message=f"Encounter.status {resource.get('status')!r} is not valid",
                resource_type="Encounter",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        class_code = (resource.get("class") or {}).get("code")
        if class_code not in ENCOUNTER_CLASSES:
            yield ValidationIssue(
                severity="warning",
                code="unexpected_encounter_class",
                path=f"{path}.class.code",
                message=f"Encounter.class {class_code!r} is outside the expected AMB/IMP/EMER/OBSENC set",
                resource_type="Encounter",
                resource_id=resource.get("id"),
                source="business-rule",
            )

    def _validate_document(self, resource: dict[str, Any], path: str) -> Iterator[ValidationIssue]:
        if resource.get("status") not in DOCUMENT_STATUSES:
            yield ValidationIssue(
                severity="error",
                code="invalid_document_status",
                path=f"{path}.status",
                message=f"DocumentReference.status {resource.get('status')!r} is not valid",
                resource_type="DocumentReference",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if "docStatus" in resource and resource.get("docStatus") not in DOC_STATUSES:
            yield ValidationIssue(
                severity="error",
                code="invalid_doc_status",
                path=f"{path}.docStatus",
                message=f"DocumentReference.docStatus {resource.get('docStatus')!r} is not valid",
                resource_type="DocumentReference",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        content = resource.get("content") or []
        if not content:
            yield ValidationIssue(
                severity="error",
                code="document_missing_content",
                path=f"{path}.content",
                message="DocumentReference has no content",
                resource_type="DocumentReference",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        for index, item in enumerate(content):
            attachment = item.get("attachment") or {}
            if not attachment.get("contentType"):
                yield ValidationIssue(
                    severity="error",
                    code="attachment_missing_content_type",
                    path=f"{path}.content[{index}].attachment.contentType",
                    message="Attachment.contentType is required",
                    resource_type="DocumentReference",
                    resource_id=resource.get("id"),
                    source="business-rule",
                )
            if not attachment.get("data") and not attachment.get("url"):
                yield ValidationIssue(
                    severity="error",
                    code="attachment_missing_data",
                    path=f"{path}.content[{index}].attachment",
                    message="Attachment must contain inline data or a URL",
                    resource_type="DocumentReference",
                    resource_id=resource.get("id"),
                    source="business-rule",
                )
        if "date" in resource and not self._valid_instant(resource["date"]):
            yield ValidationIssue(
                severity="error",
                code="invalid_document_date",
                path=f"{path}.date",
                message=f"DocumentReference.date {resource.get('date')!r} is not a valid FHIR instant",
                resource_type="DocumentReference",
                resource_id=resource.get("id"),
                source="business-rule",
            )

    def _validate_report(self, resource: dict[str, Any], path: str) -> Iterator[ValidationIssue]:
        if resource.get("status") not in REPORT_STATUSES:
            yield ValidationIssue(
                severity="error",
                code="invalid_report_status",
                path=f"{path}.status",
                message=f"DiagnosticReport.status {resource.get('status')!r} is not valid",
                resource_type="DiagnosticReport",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if not resource.get("code"):
            yield ValidationIssue(
                severity="error",
                code="report_missing_code",
                path=f"{path}.code",
                message="DiagnosticReport.code is required",
                resource_type="DiagnosticReport",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if not resource.get("result"):
            yield ValidationIssue(
                severity="warning",
                code="report_without_results",
                path=f"{path}.result",
                message="DiagnosticReport has no linked Observations",
                resource_type="DiagnosticReport",
                resource_id=resource.get("id"),
                source="business-rule",
            )

    def _validate_observation(self, resource: dict[str, Any], path: str) -> Iterator[ValidationIssue]:
        if resource.get("status") not in OBSERVATION_STATUSES:
            yield ValidationIssue(
                severity="error",
                code="invalid_observation_status",
                path=f"{path}.status",
                message=f"Observation.status {resource.get('status')!r} is not valid",
                resource_type="Observation",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if not resource.get("code"):
            yield ValidationIssue(
                severity="error",
                code="observation_missing_code",
                path=f"{path}.code",
                message="Observation.code is required",
                resource_type="Observation",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        if not any(
            key in resource
            for key in ("valueQuantity", "valueString", "valueCodeableConcept", "dataAbsentReason")
        ):
            yield ValidationIssue(
                severity="error",
                code="observation_missing_value",
                path=path,
                message="Observation must carry a value or dataAbsentReason",
                resource_type="Observation",
                resource_id=resource.get("id"),
                source="business-rule",
            )
        for index, interpretation in enumerate(resource.get("interpretation", [])):
            for coding in interpretation.get("coding", []):
                if coding.get("code") not in INTERPRETATION_CODES:
                    yield ValidationIssue(
                        severity="warning",
                        code="unknown_interpretation_code",
                        path=f"{path}.interpretation[{index}]",
                        message=f"Interpretation code {coding.get('code')!r} is unusual",
                        resource_type="Observation",
                        resource_id=resource.get("id"),
                        source="business-rule",
                    )
        for index, reference_range in enumerate(resource.get("referenceRange", [])):
            low = (reference_range.get("low") or {}).get("value")
            high = (reference_range.get("high") or {}).get("value")
            if low is not None and high is not None and low > high:
                yield ValidationIssue(
                    severity="error",
                    code="invalid_reference_range",
                    path=f"{path}.referenceRange[{index}]",
                    message="Reference range low value exceeds high value",
                    resource_type="Observation",
                    resource_id=resource.get("id"),
                    source="business-rule",
                )

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _iter_references(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
        if isinstance(value, dict):
            if isinstance(value.get("reference"), str):
                yield path, value["reference"]
            for key, child in value.items():
                yield from FhirValidator._iter_references(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from FhirValidator._iter_references(child, f"{path}[{index}]")

    @staticmethod
    def _resource_for_path(
        bundle: dict[str, Any], path: list[Any]
    ) -> tuple[str | None, str | None]:
        if len(path) >= 3 and path[0] == "entry" and isinstance(path[1], int):
            resource = bundle.get("entry", [])[path[1]].get("resource", {})
            return resource.get("resourceType"), resource.get("id")
        return "Bundle", bundle.get("id")

    @staticmethod
    def _json_path(path: list[Any]) -> str:
        rendered = ""
        for part in path:
            if isinstance(part, int):
                rendered += f"[{part}]"
            else:
                rendered += f".{part}" if rendered else str(part)
        return rendered or "$"

    @staticmethod
    def _parse_date(value: str) -> date | None:
        try:
            return date.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError:
                return None

    @classmethod
    def _valid_instant(cls, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    @staticmethod
    def _dedupe(issues: list[ValidationIssue]) -> list[ValidationIssue]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[ValidationIssue] = []
        for issue in issues:
            key = (issue.code, issue.path, issue.message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)
        return unique

    @staticmethod
    def _patient_id(bundle: dict[str, Any]) -> str:
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                return str(resource.get("id", "unknown"))
        return "unknown"

    @staticmethod
    def _mrn(bundle: dict[str, Any]) -> str:
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Patient":
                continue
            for identifier in resource.get("identifier", []):
                if identifier.get("system") == MRN_SYSTEM:
                    return str(identifier.get("value", "unknown"))
        return "unknown"


def render_markdown(report: ValidationReport) -> str:
    lines = [
        "# FHIR R4 Validation Report",
        "",
        f"Generated: `{report.generated_at.isoformat()}`",
        f"Schema: **{report.schema_version}**",
        f"Validators: {'; '.join(report.validators)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Bundles | {report.total_bundles} |",
        f"| Valid | {report.valid_bundles} |",
        f"| Invalid | {report.invalid_bundles} |",
        f"| Errors | {report.total_errors} |",
        f"| Warnings | {report.total_warnings} |",
        "",
    ]
    if report.resource_counts:
        lines.extend(["### Resource counts", "", "| Resource | Count |", "|---|---:|"])
        lines.extend(
            f"| {resource_type} | {count} |"
            for resource_type, count in report.resource_counts.items()
        )
        lines.append("")

    invalid = [bundle for bundle in report.bundles if not bundle.valid]
    if not invalid:
        lines.append("**All Bundles passed structural, schema, and reference validation.**")
        return "\n".join(lines) + "\n"

    lines.extend(["## Invalid Bundles", ""])
    for bundle in invalid:
        lines.extend(
            [
                f"### {bundle.mrn} (`{bundle.patient_id}`)",
                "",
                f"Bundle: `{bundle.bundle_id}` · errors: **{bundle.error_count}** · "
                f"warnings: {bundle.warning_count}",
                "",
                "| Severity | Source | Code | Path | Message |",
                "|---|---|---|---|---|",
            ]
        )
        for issue in bundle.issues:
            message = issue.message.replace("|", "\\|")
            lines.append(
                f"| {issue.severity} | {issue.source} | `{issue.code}` | "
                f"`{issue.path}` | {message} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
