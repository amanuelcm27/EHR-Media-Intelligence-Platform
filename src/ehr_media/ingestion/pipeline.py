"""Cleaning + merging pipeline (Task 1).

Raw records from every source are normalized into drafts, merged by canonical
MRN, de-duplicated, and emitted as validated :class:`PatientRecord` objects
with a field-level audit trail and a machine-readable ingestion report.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings, get_settings
from ..ids import (
    content_hash,
    document_id_for,
    encounter_id_for,
    observation_id_for,
    patient_id_for_mrn,
)
from ..models import (
    Anomaly,
    AuditEntry,
    ClinicalDocument,
    DataQualityFlag,
    DocumentType,
    Encounter,
    EncounterClass,
    LabResult,
    PatientRecord,
)
from ..normalizers import (
    LAB_CATALOG,
    clean_text,
    detect_anomalies,
    interpret_lab,
    is_blank,
    lookup_lab,
    normalize_doc_type,
    normalize_document_text,
    normalize_email,
    normalize_encounter_class,
    normalize_gender,
    normalize_lab_name,
    normalize_mrn,
    normalize_name,
    normalize_phone,
    normalize_state,
    normalize_title,
    parse_date,
    parse_lab_value,
    parse_reference_range,
    split_comma_name,
    utcnow,
)
from .csv_source import CsvSource
from .json_source import JsonEhrSource
from .raw import IngestionIssue, RawDocument, RawPatient, Source, SourceResult
from .text_source import TextNotesSource

SOURCE_PRIORITY = {"json": 30, "csv": 20, "text": 10}

DEMOGRAPHIC_FIELDS = (
    "given_name",
    "family_name",
    "middle_name",
    "suffix",
    "birth_date",
    "gender",
    "phone",
    "email",
    "address_line",
    "city",
    "state",
    "postal_code",
    "preferred_language",
)
IDENTITY_FIELDS = {"given_name", "family_name", "birth_date", "gender"}

DOCUMENT_TITLES: dict[str, str] = {
    "progress_note": "Progress note",
    "consultation": "Consultation note",
    "discharge_summary": "Discharge summary",
    "imaging_report": "Imaging report",
    "lab_report": "Laboratory report",
    "scanned_note": "Scanned note",
    "other": "Clinical document",
}

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Internal draft structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Candidate:
    value: Any
    raw: Any
    source: str
    source_format: str
    priority: int
    order: int
    rule: str


@dataclass(slots=True)
class _DraftDocument:
    doc_type: DocumentType
    title: str
    text: str
    authored_on: date | None
    author: str | None
    encounter_ref: str | None
    source: str
    source_format: str
    content_hash: str
    anomalies: list[Anomaly]
    audit: list[AuditEntry]


@dataclass(slots=True)
class _DraftLab:
    test_name: str
    loinc_code: str | None
    value: float | None
    value_text: str | None
    unit: str | None
    reference_range: str | None
    reference_low: float | None
    reference_high: float | None
    interpretation: str
    collected_at: date | None
    status: str
    source: str
    content_hash: str


@dataclass(slots=True)
class _DraftEncounter:
    class_code: EncounterClass
    reason: str | None
    start_date: date | None
    end_date: date | None
    source: str


@dataclass(slots=True)
class _Draft:
    mrn: str | None
    order: int
    source: str
    source_format: str
    row_ref: str
    fields: dict[str, list[_Candidate]] = field(default_factory=lambda: defaultdict(list))
    flags: list[DataQualityFlag] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    documents: list[_DraftDocument] = field(default_factory=list)
    labs: list[_DraftLab] = field(default_factory=list)
    encounters: list[_DraftEncounter] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    synthetic_mrn: bool = False


@dataclass
class IngestionResult:
    patients: list[PatientRecord]
    issues: list[IngestionIssue]
    report: dict[str, Any]

    def write(self, settings: Settings | None = None) -> dict[str, Path]:
        """Persist cleaned records + reports and return the written paths."""
        settings = settings or get_settings()
        settings.ensure_dirs()
        paths = {
            "patients": settings.cleaned_dir / "patients.json",
            "report": settings.reports_dir / "ingestion_report.json",
        }
        paths["patients"].write_text(
            json.dumps(
                [patient.model_dump(mode="json") for patient in self.patients],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["report"].write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    def __init__(self, sources: Iterable[Source]) -> None:
        self.sources = list(sources)
        self._duplicate_counts = {"documents": 0, "labs": 0}

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path] | str | Path,
        *,
        data_dir: str | Path | None = None,
    ) -> IngestionPipeline:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files: list[Path] = []
        for item in paths:
            path = Path(item)
            if not path.is_absolute() and data_dir is not None:
                path = Path(data_dir) / path
            if path.is_dir():
                files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            elif path.is_file():
                files.append(path)
        sources: list[Source] = []
        for file in files:
            suffix = file.suffix.lower()
            if suffix in {".json", ".jsonl", ".ndjson"}:
                sources.append(JsonEhrSource(file))
            elif suffix in {".csv", ".tsv"}:
                sources.append(CsvSource(file))
            elif suffix in {".txt", ".text", ".md"}:
                sources.append(TextNotesSource(file))
        return cls(sources)

    def run(self) -> IngestionResult:
        started = time.perf_counter()
        self._duplicate_counts = {"documents": 0, "labs": 0}
        issues: list[IngestionIssue] = []
        drafts: list[_Draft] = []
        source_counts: Counter[str] = Counter()
        raw_count = 0
        order = 0

        for source in self.sources:
            try:
                result: SourceResult = source.load()
            except Exception as exc:  # pragma: no cover - defensive
                issues.append(
                    IngestionIssue(
                        source=getattr(source, "name", "unknown"),
                        row_ref="$",
                        code="source_failed",
                        message=f"Source failed to load: {exc}",
                        severity="error",
                    )
                )
                continue
            issues.extend(result.issues)
            for raw in result.records:
                raw_count += 1
                source_counts[raw.source] += 1
                drafts.append(self._build_draft(raw, order))
                order += 1

        # Resolve missing MRNs before grouping (name + DOB -> deterministic ID).
        for draft in drafts:
            if draft.mrn is None:
                synthetic = self._synthetic_mrn(draft)
                if synthetic:
                    draft.mrn = synthetic
                    draft.synthetic_mrn = True
                    draft.flags = [f for f in draft.flags if f.code != "missing_mrn"]
                    draft.flags.append(
                        DataQualityFlag(
                            code="synthetic_mrn",
                            message="MRN missing; deterministic identifier derived from name + DOB",
                            severity="info",
                            field="mrn",
                        )
                    )
                    draft.audit.append(
                        self._audit(
                            "mrn",
                            "inferred",
                            None,
                            synthetic,
                            "mrn.derived_from_name_and_dob",
                            draft.source,
                        )
                    )
                else:
                    issues.append(
                        IngestionIssue(
                            source=draft.source,
                            row_ref=draft.row_ref,
                            code="missing_identifier",
                            message="Record rejected: no MRN and insufficient name/DOB to identify patient",
                            severity="error",
                        )
                    )

        grouped: dict[str, list[_Draft]] = defaultdict(list)
        rejected = 0
        for draft in drafts:
            if draft.mrn is None:
                rejected += 1
                continue
            grouped[draft.mrn].append(draft)

        patients: list[PatientRecord] = []
        conflicts: list[dict[str, Any]] = []
        for mrn, group in grouped.items():
            patient, patient_conflicts = self._merge_group(mrn, group)
            patients.append(patient)
            conflicts.extend(patient_conflicts)

        patients.sort(key=lambda p: p.mrn)
        duplicate_candidates = self._flag_duplicate_patients(patients)

        stats = {
            "sources": dict(source_counts),
            "source_files": [getattr(source, "name", "unknown") for source in self.sources],
            "raw_records": raw_count,
            "rejected_records": rejected,
            "patients": len(patients),
            "documents": sum(len(p.documents) for p in patients),
            "lab_results": sum(len(p.lab_results) for p in patients),
            "encounters": sum(len(p.encounters) for p in patients),
            "conflicts": len(conflicts),
            "audit_entries": sum(len(p.audit) for p in patients),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        stats["duplicates_removed"] = dict(self._duplicate_counts)

        flag_counts = Counter(flag.code for p in patients for flag in p.data_quality_flags)
        stats["data_quality_flags"] = dict(sorted(flag_counts.items()))

        report = {
            "generated_at": stats["generated_at"],
            "stats": stats,
            "issues": [self._issue_dict(issue) for issue in issues],
            "conflicts": [
                {key: value for key, value in conflict.items() if not key.startswith("_")}
                for conflict in conflicts
            ],
            "duplicate_candidates": duplicate_candidates,
            "patients": [
                {
                    "mrn": patient.mrn,
                    "name": patient.display_name,
                    "documents": len(patient.documents),
                    "lab_results": len(patient.lab_results),
                    "encounters": len(patient.encounters),
                    "flags": sorted(patient.flag_codes()),
                }
                for patient in patients
            ],
        }
        return IngestionResult(patients=patients, issues=issues, report=report)

    # ------------------------------------------------------------------
    # Draft construction
    # ------------------------------------------------------------------
    def _build_draft(self, raw: RawPatient, order: int) -> _Draft:
        priority = SOURCE_PRIORITY.get(raw.source_format, 10)
        draft = _Draft(
            mrn=None,
            order=order,
            source=raw.source,
            source_format=raw.source_format,
            row_ref=raw.row_ref,
        )
        draft.sources.add(raw.source)

        mrn_parse = normalize_mrn(raw.mrn)
        draft.mrn = mrn_parse.value
        if mrn_parse.value is None:
            draft.flags.append(
                DataQualityFlag(
                    code="missing_mrn",
                    message="Record has no usable MRN",
                    severity="error",
                    field="mrn",
                )
            )
        elif mrn_parse.changed:
            draft.audit.append(
                self._audit(
                    "mrn",
                    "normalized",
                    str(raw.mrn),
                    mrn_parse.value,
                    mrn_parse.rule,
                    raw.source,
                )
            )
        if mrn_parse.warning and mrn_parse.value:
            draft.flags.append(
                DataQualityFlag(
                    code="mrn_warning",
                    message=mrn_parse.warning,
                    severity="warning",
                    field="mrn",
                )
            )

        given, family = raw.given_name, raw.family_name
        if isinstance(raw.full_name, str) and raw.full_name.strip():
            derived_given, derived_family = split_comma_name(raw.full_name)
            if not given and derived_given:
                given = derived_given
            if not family and derived_family:
                family = derived_family
        if not given and not family:
            draft.flags.append(
                DataQualityFlag(
                    code="missing_name",
                    message="No patient name supplied by this source",
                    severity="warning",
                    field="name",
                )
            )

        self._add_name(draft, "given_name", given, raw.full_name or given, raw, priority)
        self._add_name(draft, "family_name", family, raw.full_name or family, raw, priority)
        self._add_name(draft, "middle_name", raw.middle_name, raw.middle_name, raw, priority)
        self._add_name(draft, "suffix", raw.suffix, raw.suffix, raw, priority)

        date_parse = parse_date(raw.birth_date)
        if date_parse.value is not None:
            self._add(
                draft,
                "birth_date",
                date_parse.value,
                raw.birth_date,
                raw,
                priority,
                date_parse.rule,
            )
            if date_parse.warning:
                draft.flags.append(
                    DataQualityFlag(
                        code="ambiguous_birth_date",
                        message=date_parse.warning,
                        severity="info",
                        field="birth_date",
                    )
                )
        elif not is_blank(raw.birth_date):
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_birth_date",
                    message=date_parse.warning or "Birth date could not be parsed",
                    severity="warning",
                    field="birth_date",
                )
            )
            draft.audit.append(
                self._audit(
                    "birth_date",
                    "quarantined",
                    str(raw.birth_date),
                    None,
                    date_parse.rule,
                    raw.source,
                )
            )

        gender_parse = normalize_gender(raw.gender)
        if gender_parse.recognized:
            self._add(
                draft,
                "gender",
                gender_parse.value,
                raw.gender,
                raw,
                priority,
                gender_parse.rule,
            )
        elif is_blank(raw.gender):
            draft.flags.append(
                DataQualityFlag(
                    code="missing_gender",
                    message="Gender not documented; defaulted to 'unknown'",
                    severity="info",
                    field="gender",
                )
            )
        else:
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_gender",
                    message=gender_parse.warning or f"Unrecognized gender {raw.gender!r}",
                    severity="warning",
                    field="gender",
                )
            )
            draft.audit.append(
                self._audit(
                    "gender",
                    "quarantined",
                    str(raw.gender),
                    "unknown",
                    gender_parse.rule,
                    raw.source,
                )
            )

        phone, phone_warning = normalize_phone(raw.phone)
        if phone:
            self._add(draft, "phone", phone, raw.phone, raw, priority, "phone.e164")
        elif phone_warning:
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_phone", message=phone_warning, severity="warning", field="phone"
                )
            )

        email, email_warning = normalize_email(raw.email)
        if email:
            self._add(draft, "email", email, raw.email, raw, priority, "email.lowercase")
        elif email_warning:
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_email", message=email_warning, severity="warning", field="email"
                )
            )

        for field_name in ("address_line", "city", "postal_code", "preferred_language"):
            value = getattr(raw, field_name)
            text = clean_text(value)
            if text:
                self._add(draft, field_name, text, value, raw, priority, f"{field_name}.trimmed")

        state = normalize_state(raw.state)
        if state:
            self._add(draft, "state", state, raw.state, raw, priority, "state.usps")
        elif not is_blank(raw.state):
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_state",
                    message=f"Unrecognized state {raw.state!r}",
                    severity="warning",
                    field="state",
                )
            )

        for raw_document in raw.documents:
            document = self._normalize_document(raw_document, raw.source)
            if document:
                draft.documents.append(document)
                if not is_blank(raw_document.authored_on) and document.authored_on is None:
                    draft.flags.append(
                        DataQualityFlag(
                            code="invalid_document_date",
                            message=f"Document date {raw_document.authored_on!r} could not be parsed",
                            severity="warning",
                            field="document.authored_on",
                        )
                    )

        for raw_lab in raw.labs:
            lab = self._normalize_lab(raw_lab, draft)
            if lab:
                draft.labs.append(lab)

        for raw_encounter in raw.encounters:
            start = parse_date(raw_encounter.start_date)
            end = parse_date(raw_encounter.end_date)
            draft.encounters.append(
                _DraftEncounter(
                    class_code=normalize_encounter_class(raw_encounter.class_code),
                    reason=clean_text(raw_encounter.reason) or None,
                    start_date=start.value,
                    end_date=end.value,
                    source=raw.source,
                )
            )

        return draft

    def _add_name(
        self,
        draft: _Draft,
        field_name: str,
        value: Any,
        raw_value: Any,
        raw: RawPatient,
        priority: int,
    ) -> None:
        normalized = normalize_name(value)
        if normalized:
            self._add(
                draft, field_name, normalized, raw_value, raw, priority, f"{field_name}.title_case"
            )

    def _add(
        self,
        draft: _Draft,
        field_name: str,
        value: Any,
        raw_value: Any,
        raw: RawPatient,
        priority: int,
        rule: str,
    ) -> None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        draft.fields[field_name].append(
            _Candidate(
                value=value,
                raw=raw_value,
                source=raw.source,
                source_format=raw.source_format,
                priority=priority,
                order=draft.order,
                rule=rule,
            )
        )

    def _normalize_document(
        self, raw: RawDocument, source: str
    ) -> _DraftDocument | None:
        text = normalize_document_text(raw.text)
        if not text:
            return None
        doc_type = normalize_doc_type(raw.doc_type or raw.title)
        date_parse = parse_date(raw.authored_on)
        authored_on = date_parse.value
        audit: list[AuditEntry] = []
        if not is_blank(raw.authored_on):
            if authored_on is None:
                audit.append(
                    self._audit(
                        "document.authored_on",
                        "quarantined",
                        str(raw.authored_on),
                        None,
                        date_parse.rule,
                        source,
                    )
                )
            elif str(raw.authored_on).strip() != authored_on.isoformat():
                audit.append(
                    self._audit(
                        "document.authored_on",
                        "normalized",
                        str(raw.authored_on),
                        authored_on.isoformat(),
                        date_parse.rule,
                        source,
                    )
                )
        title = normalize_title(
            raw.title,
            fallback=f"{DOCUMENT_TITLES.get(doc_type, 'Clinical document')}"
            + (f" ({authored_on.isoformat()})" if authored_on else ""),
        )
        digest = content_hash(
            doc_type,
            authored_on.isoformat() if authored_on else "",
            " ".join(text.lower().split()),
        )
        return _DraftDocument(
            doc_type=doc_type,
            title=title,
            text=text,
            authored_on=authored_on,
            author=clean_text(raw.author) or None,
            encounter_ref=raw.encounter_ref,
            source=source,
            source_format=raw.source_format,
            content_hash=digest,
            anomalies=detect_anomalies(text),
            audit=audit,
        )

    def _normalize_lab(self, raw: Any, draft: _Draft) -> _DraftLab | None:
        if is_blank(raw.test_name):
            return None
        name, catalog_loinc = normalize_lab_name(raw.test_name)
        parsed = parse_lab_value(raw.value)
        entry = lookup_lab(name)

        if str(raw.test_name).strip() != name:
            draft.audit.append(
                self._audit(
                    "lab.test_name",
                    "normalized",
                    str(raw.test_name),
                    name,
                    "lab.catalog_alias",
                    raw.source,
                )
            )

        unit = clean_text(raw.unit) or parsed.unit
        if not unit and entry and entry.unit:
            unit = entry.unit
            draft.audit.append(
                self._audit(
                    "lab.unit",
                    "inferred",
                    None,
                    unit,
                    "lab.unit_from_catalog",
                    raw.source,
                )
            )
        elif raw.unit and parsed.unit and clean_text(raw.unit) != parsed.unit:
            draft.flags.append(
                DataQualityFlag(
                    code="conflicting_lab_unit",
                    message=f"Unit column {raw.unit!r} differs from value unit {parsed.unit!r}; kept column",
                    severity="info",
                    field="lab.unit",
                )
            )

        reference_low, reference_high, range_rule = parse_reference_range(raw.reference_range)
        reference_range = clean_text(raw.reference_range) or None
        if reference_low is None and reference_high is None:
            if entry and (entry.low is not None or entry.high is not None):
                reference_low, reference_high = entry.low, entry.high
                reference_range = self._format_range(reference_low, reference_high)
                draft.audit.append(
                    self._audit(
                        "lab.reference_range",
                        "inferred",
                        None,
                        reference_range,
                        "lab.reference_range_from_catalog",
                        raw.source,
                    )
                )
            elif reference_range:
                draft.flags.append(
                    DataQualityFlag(
                        code="invalid_reference_range",
                        message=f"Could not parse reference range {reference_range!r}",
                        severity="warning",
                        field="lab.reference_range",
                    )
                )
                reference_range = None

        date_parse = parse_date(raw.collected_at)
        if is_blank(raw.collected_at):
            draft.flags.append(
                DataQualityFlag(
                    code="missing_lab_date",
                    message=f"Lab {name!r} has no collection date",
                    severity="warning",
                    field="lab.collected_at",
                )
            )
        elif date_parse.value is None:
            draft.flags.append(
                DataQualityFlag(
                    code="invalid_lab_date",
                    message=date_parse.warning or "Lab collection date could not be parsed",
                    severity="warning",
                    field="lab.collected_at",
                )
            )
            draft.audit.append(
                self._audit(
                    "lab.collected_at",
                    "quarantined",
                    str(raw.collected_at),
                    None,
                    date_parse.rule,
                    raw.source,
                )
            )

        if parsed.value is None and parsed.value_text is None and is_blank(raw.value):
            draft.flags.append(
                DataQualityFlag(
                    code="missing_lab_value",
                    message=f"Lab {name!r} has no result value",
                    severity="warning",
                    field="lab.value",
                )
            )
        elif parsed.warning:
            draft.flags.append(
                DataQualityFlag(
                    code="non_numeric_lab_value",
                    message=f"Lab {name!r}: {parsed.warning}",
                    severity="info",
                    field="lab.value",
                )
            )

        interpretation = interpret_lab(
            name,
            parsed.value,
            reference_low,
            reference_high,
            qualifier=parsed.qualifier,
            explicit_flag=parsed.flag,
        )
        status = self._normalize_lab_status(raw.status)
        if status == "cancelled":
            interpretation = "unknown"

        return _DraftLab(
            test_name=name,
            loinc_code=raw.loinc_code or catalog_loinc,
            value=parsed.value,
            value_text=parsed.value_text,
            unit=unit or None,
            reference_range=reference_range,
            reference_low=reference_low,
            reference_high=reference_high,
            interpretation=interpretation,
            collected_at=date_parse.value,
            status=status,
            source=raw.source,
            content_hash=content_hash(
                name.lower(),
                date_parse.value.isoformat() if date_parse.value else "",
                parsed.value,
                parsed.value_text,
                unit,
            ),
        )

    @staticmethod
    def _normalize_lab_status(value: Any) -> str:
        text = clean_text(value).lower()
        if text in {"final", "f", "completed", "complete", "verified"}:
            return "final"
        if text in {"prelim", "preliminary", "p", "partial"}:
            return "preliminary"
        if text in {"corrected", "c", "amended"}:
            return "corrected"
        if text in {"cancelled", "canceled", "x"}:
            return "cancelled"
        return "unknown"

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_group(
        self, mrn: str, drafts: list[_Draft]
    ) -> tuple[PatientRecord, list[dict[str, Any]]]:
        drafts = sorted(drafts, key=lambda d: d.order)
        patient_id = patient_id_for_mrn(mrn)
        conflicts: list[dict[str, Any]] = []
        audit: list[AuditEntry] = []
        flags: list[DataQualityFlag] = []

        chosen: dict[str, _Candidate | None] = {}
        for field_name in DEMOGRAPHIC_FIELDS:
            candidates = [c for draft in drafts for c in draft.fields.get(field_name, [])]
            selected, conflict_values = self._choose(candidates)
            chosen[field_name] = selected
            if selected is None:
                continue
            if self._changed(selected):
                audit.append(
                    self._audit(
                        field_name,
                        "normalized",
                        str(selected.raw) if selected.raw is not None else None,
                        str(selected.value),
                        selected.rule,
                        selected.source,
                    )
                )
            if field_name in IDENTITY_FIELDS and len(conflict_values) > 1:
                rendered = "; ".join(f"{value} ({source})" for value, source in conflict_values)
                flags.append(
                    DataQualityFlag(
                        code=f"conflicting_{field_name}",
                        message=f"Conflicting {field_name.replace('_', ' ')}: {rendered}. "
                        f"Kept {selected.value!r} from {selected.source}.",
                        severity="warning",
                        field=field_name,
                    )
                )
                audit.append(
                    self._audit(
                        field_name,
                        "conflict_resolved",
                        ", ".join(str(value) for value, _ in conflict_values),
                        str(selected.value),
                        "conflict.source_priority",
                        selected.source,
                    )
                )
                conflicts.append(
                    {
                        "mrn": mrn,
                        "field": field_name,
                        "values": [str(value) for value, _ in conflict_values],
                        "sources": [source for _, source in conflict_values],
                        "chosen": str(selected.value),
                        "chosen_source": selected.source,
                    }
                )

        for draft in drafts:
            flags.extend(draft.flags)
            audit.extend(draft.audit)

        given_name = self._field_value(chosen, "given_name") or "Unknown"
        family_name = self._field_value(chosen, "family_name") or "Unknown"
        birth_date = self._field_value(chosen, "birth_date")
        gender = self._field_value(chosen, "gender") or "unknown"

        if given_name == "Unknown" or family_name == "Unknown":
            flags.append(
                DataQualityFlag(
                    code="missing_name",
                    message="Patient name incomplete; placeholder 'Unknown' inserted",
                    severity="warning",
                    field="name",
                )
            )
            audit.append(
                self._audit(
                    "name",
                    "default_applied",
                    None,
                    "Unknown",
                    "name.placeholder",
                    drafts[0].source,
                )
            )
        if birth_date is None:
            flags.append(
                DataQualityFlag(
                    code="missing_birth_date",
                    message="Date of birth not documented; FHIR Patient.birthDate omitted",
                    severity="warning",
                    field="birth_date",
                )
            )
        if gender == "unknown":
            flags.append(
                DataQualityFlag(
                    code="missing_gender",
                    message="Gender unknown; FHIR Patient.gender set to 'unknown'",
                    severity="info",
                    field="gender",
                )
            )

        documents, duplicate_docs, document_audit = self._merge_documents(
            patient_id, drafts
        )
        labs, duplicate_labs = self._merge_labs(patient_id, drafts)
        encounters = self._merge_encounters(patient_id, drafts, documents, labs)

        # Link each document to the closest encounter so subject/encounter
        # references always resolve inside the Bundle.
        for document in documents:
            document.encounter_id = self._nearest_encounter(document.authored_on, encounters)
        audit.extend(document_audit)

        if duplicate_docs or duplicate_labs:
            audit.append(
                self._audit(
                    "record",
                    "duplicate_removed",
                    f"{duplicate_docs + duplicate_labs} duplicate item(s)",
                    "0 duplicates",
                    "dedupe.content_hash",
                    drafts[0].source,
                )
            )

        critical_labs = [lab for lab in labs if lab.interpretation == "critical"]
        if critical_labs:
            summary = ", ".join(
                f"{lab.test_name} {lab.display_value} ({lab.interpretation})"
                for lab in critical_labs[:4]
            )
            flags.append(
                DataQualityFlag(
                    code="critical_lab",
                    message=f"{len(critical_labs)} critical lab result(s): {summary}",
                    severity="critical",
                    field="lab_results",
                )
            )
        abnormal_count = sum(1 for lab in labs if lab.is_abnormal)
        if abnormal_count and not critical_labs:
            flags.append(
                DataQualityFlag(
                    code="abnormal_lab",
                    message=f"{abnormal_count} abnormal lab result(s)",
                    severity="warning",
                    field="lab_results",
                )
            )

        critical_docs = [
            doc for doc in documents if any(a.severity == "critical" for a in doc.anomalies)
        ]
        if critical_docs:
            flags.append(
                DataQualityFlag(
                    code="critical_document_finding",
                    message=f"{len(critical_docs)} document(s) contain critical clinical language",
                    severity="critical",
                    field="documents",
                )
            )
        if not documents:
            flags.append(
                DataQualityFlag(
                    code="no_documents",
                    message="No clinical documents available for this patient",
                    severity="info",
                    field="documents",
                )
            )
        if not labs:
            flags.append(
                DataQualityFlag(
                    code="no_labs",
                    message="No lab results available for this patient",
                    severity="info",
                    field="lab_results",
                )
            )
        if not encounters:
            flags.append(
                DataQualityFlag(
                    code="no_encounter",
                    message="No encounter supplied; an encounter was derived",
                    severity="info",
                    field="encounters",
                )
            )
        elif all(encounter.source == "derived" for encounter in encounters):
            flags.append(
                DataQualityFlag(
                    code="derived_encounter",
                    message="No encounter supplied by source; encounter derived from record dates",
                    severity="info",
                    field="encounters",
                )
            )

        flags = self._dedupe_flags(self._drop_resolved_flags(flags, birth_date, gender, given_name, family_name))

        patient = PatientRecord(
            id=patient_id,
            mrn=mrn,
            given_name=given_name,
            family_name=family_name,
            middle_name=self._field_value(chosen, "middle_name"),
            suffix=self._field_value(chosen, "suffix"),
            birth_date=birth_date,
            gender=gender,
            phone=self._field_value(chosen, "phone"),
            email=self._field_value(chosen, "email"),
            address_line=self._field_value(chosen, "address_line"),
            city=self._field_value(chosen, "city"),
            state=self._field_value(chosen, "state"),
            postal_code=self._field_value(chosen, "postal_code"),
            preferred_language=self._field_value(chosen, "preferred_language"),
            documents=documents,
            lab_results=labs,
            encounters=encounters,
            data_quality_flags=flags,
            audit=audit,
            sources=sorted({source for draft in drafts for source in draft.sources}),
            ingested_at=utcnow(),
        )
        self._duplicate_counts = {
            "documents": self._duplicate_counts.get("documents", 0) + duplicate_docs,
            "labs": self._duplicate_counts.get("labs", 0) + duplicate_labs,
        }
        return patient, conflicts

    @staticmethod
    def _choose(candidates: list[_Candidate]) -> tuple[_Candidate | None, list[tuple[Any, str]]]:
        if not candidates:
            return None, []
        ordered = sorted(candidates, key=lambda c: (-c.priority, c.order))
        selected = ordered[0]
        distinct: list[tuple[Any, str]] = []
        seen: set[str] = set()
        for candidate in ordered:
            key = str(candidate.value).lower()
            if key not in seen:
                seen.add(key)
                distinct.append((candidate.value, candidate.source))
        return selected, distinct

    @staticmethod
    def _changed(candidate: _Candidate) -> bool:
        if candidate.raw is None:
            return False
        raw = str(candidate.raw).strip()
        value = candidate.value.isoformat() if isinstance(candidate.value, date) else str(candidate.value)
        return raw != value

    @staticmethod
    def _field_value(chosen: dict[str, _Candidate | None], field_name: str) -> Any:
        candidate = chosen.get(field_name)
        return candidate.value if candidate else None

    @staticmethod
    def _drop_resolved_flags(
        flags: list[DataQualityFlag],
        birth_date: Any,
        gender: str,
        given_name: str,
        family_name: str,
    ) -> list[DataQualityFlag]:
        resolved: set[str] = set()
        if birth_date is not None:
            resolved |= {"missing_birth_date", "invalid_birth_date"}
        if gender != "unknown":
            resolved |= {"missing_gender", "invalid_gender"}
        if given_name != "Unknown" and family_name != "Unknown":
            resolved.add("missing_name")
        return [flag for flag in flags if flag.code not in resolved]

    @staticmethod
    def _dedupe_flags(flags: list[DataQualityFlag]) -> list[DataQualityFlag]:
        seen: set[tuple[str, str | None]] = set()
        unique: list[DataQualityFlag] = []
        for flag in flags:
            key = (flag.code, flag.field)
            if key in seen:
                continue
            seen.add(key)
            unique.append(flag)
        return sorted(
            unique, key=lambda f: (-_SEVERITY_ORDER.get(f.severity, 0), f.code)
        )

    def _merge_documents(
        self, patient_id: str, drafts: list[_Draft]
    ) -> tuple[list[ClinicalDocument], int, list[AuditEntry]]:
        by_hash: dict[str, _DraftDocument] = {}
        duplicate_count = 0
        audit: list[AuditEntry] = []
        for draft in drafts:
            for document in draft.documents:
                existing = by_hash.get(document.content_hash)
                if existing is None:
                    by_hash[document.content_hash] = document
                    continue
                duplicate_count += 1
                if existing.authored_on is None and document.authored_on is not None:
                    existing.authored_on = document.authored_on
                if existing.author is None and document.author:
                    existing.author = document.author
                if existing.title == DOCUMENT_TITLES.get(existing.doc_type, "Clinical document"):
                    existing.title = document.title
                audit.append(
                    self._audit(
                        "document",
                        "duplicate_removed",
                        f"{document.title} from {document.source}",
                        existing.title,
                        "dedupe.document_content_hash",
                        document.source,
                    )
                )
        documents = [
            ClinicalDocument(
                id=document_id_for(patient_id, document.content_hash),
                patient_id=patient_id,
                doc_type=document.doc_type,
                title=document.title,
                text=document.text,
                authored_on=document.authored_on,
                author=document.author,
                encounter_id=None,
                source=document.source,
                source_format=document.source_format,
                content_hash=document.content_hash,
                anomalies=document.anomalies,
            )
            for document in by_hash.values()
        ]
        for document in by_hash.values():
            audit.extend(document.audit)
        documents.sort(
            key=lambda d: (d.authored_on is not None, d.authored_on or date.min, d.title),
            reverse=True,
        )
        return documents, duplicate_count, audit

    def _merge_labs(
        self, patient_id: str, drafts: list[_Draft]
    ) -> tuple[list[LabResult], int]:
        by_hash: dict[str, _DraftLab] = {}
        duplicate_count = 0
        for draft in drafts:
            for lab in draft.labs:
                if lab.content_hash in by_hash:
                    duplicate_count += 1
                    existing = by_hash[lab.content_hash]
                    if existing.reference_range is None and lab.reference_range:
                        existing.reference_range = lab.reference_range
                        existing.reference_low = lab.reference_low
                        existing.reference_high = lab.reference_high
                    continue
                by_hash[lab.content_hash] = lab
        labs = [
            LabResult(
                id=observation_id_for(patient_id, lab.content_hash),
                patient_id=patient_id,
                test_name=lab.test_name,
                loinc_code=lab.loinc_code,
                value=lab.value,
                value_text=lab.value_text,
                unit=lab.unit,
                reference_range=lab.reference_range,
                reference_low=lab.reference_low,
                reference_high=lab.reference_high,
                interpretation=lab.interpretation,  # type: ignore[arg-type]
                collected_at=lab.collected_at,
                status=lab.status,  # type: ignore[arg-type]
                source=lab.source,
                content_hash=lab.content_hash,
            )
            for lab in by_hash.values()
        ]
        labs.sort(
            key=lambda lab: (lab.collected_at is not None, lab.collected_at or date.min, lab.test_name),
            reverse=True,
        )
        return labs, duplicate_count

    def _merge_encounters(
        self,
        patient_id: str,
        drafts: list[_Draft],
        documents: list[ClinicalDocument],
        labs: list[LabResult],
    ) -> list[Encounter]:
        merged: dict[tuple[Any, Any, Any], _DraftEncounter] = {}
        for draft in drafts:
            for encounter in draft.encounters:
                key = (encounter.class_code, encounter.start_date, encounter.reason)
                merged.setdefault(key, encounter)

        dates = [doc.authored_on for doc in documents if doc.authored_on]
        dates += [lab.collected_at for lab in labs if lab.collected_at]
        earliest = min(dates) if dates else None
        latest = max(dates) if dates else None

        if not merged and (documents or labs):
            class_code: EncounterClass = (
                "IMP"
                if any(doc.doc_type == "discharge_summary" for doc in documents)
                else "AMB"
            )
            merged[("derived", None, None)] = _DraftEncounter(
                class_code=class_code,
                reason="Clinical media encounter (derived)",
                start_date=earliest,
                end_date=latest,
                source="derived",
            )
        if not merged:
            merged[("derived", None, None)] = _DraftEncounter(
                class_code="AMB",
                reason="Registration encounter (derived)",
                start_date=None,
                end_date=None,
                source="derived",
            )

        encounters: list[Encounter] = []
        for encounter in merged.values():
            start = encounter.start_date or earliest
            end = encounter.end_date
            if start and end and end < start:
                end = start
            encounters.append(
                Encounter(
                    id=encounter_id_for(
                        patient_id,
                        f"{encounter.class_code}|{start}|{encounter.reason or ''}",
                    ),
                    patient_id=patient_id,
                    class_code=encounter.class_code,
                    reason=encounter.reason,
                    start_date=start,
                    end_date=end,
                    source=encounter.source,
                )
            )
        encounters.sort(
            key=lambda e: (e.start_date is not None, e.start_date or date.min), reverse=True
        )
        return encounters

    @staticmethod
    def _nearest_encounter(
        document_date: date | None, encounters: list[Encounter]
    ) -> str | None:
        if not encounters:
            return None
        if document_date is not None:
            same_day = [e for e in encounters if e.start_date == document_date]
            if same_day:
                return same_day[0].id
            earlier = [
                e for e in encounters if e.start_date is not None and e.start_date <= document_date
            ]
            if earlier:
                return max(earlier, key=lambda e: e.start_date or date.min).id
        return encounters[0].id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _synthetic_mrn(self, draft: _Draft) -> str | None:
        given = self._peek(draft, "given_name")
        family = self._peek(draft, "family_name")
        dob = self._peek(draft, "birth_date")
        if dob is None or (not given and not family):
            return None
        digest = content_hash(
            (family or "").lower(), (given or "").lower(), dob.isoformat(), length=10
        )
        return f"SYN{digest.upper()}"

    @staticmethod
    def _peek(draft: _Draft, field_name: str) -> Any:
        candidates = draft.fields.get(field_name, [])
        if not candidates:
            return None
        best = sorted(candidates, key=lambda c: (-c.priority, c.order))[0]
        return best.value

    def _flag_duplicate_patients(self, patients: list[PatientRecord]) -> list[dict[str, Any]]:
        seen: dict[tuple[str, str, date], PatientRecord] = {}
        pairs: list[dict[str, Any]] = []
        for patient in patients:
            if patient.birth_date is None:
                continue
            if patient.given_name == "Unknown" or patient.family_name == "Unknown":
                continue
            key = (
                patient.family_name.lower(),
                patient.given_name.lower(),
                patient.birth_date,
            )
            other = seen.get(key)
            if other is None:
                seen[key] = patient
                continue
            pairs.append(
                {
                    "mrn_a": other.mrn,
                    "mrn_b": patient.mrn,
                    "reason": "same normalized name + date of birth with different MRNs",
                    "action": "flagged_only",
                }
            )
            for current, counterpart in ((other, patient), (patient, other)):
                current.data_quality_flags.append(
                    DataQualityFlag(
                        code="possible_duplicate_patient",
                        message=f"Possible duplicate of {counterpart.display_name} "
                        f"({counterpart.mrn_display}): same name and DOB",
                        severity="warning",
                        field="mrn",
                    )
                )
                current.audit.append(
                    self._audit(
                        "mrn",
                        "validated",
                        current.mrn,
                        counterpart.mrn,
                        "dedupe.name_dob_candidate",
                        "pipeline",
                    )
                )
        return pairs

    @staticmethod
    def _format_range(low: float | None, high: float | None) -> str | None:
        if low is not None and high is not None:
            return f"{low:g}-{high:g}"
        if high is not None:
            return f"<{high:g}"
        if low is not None:
            return f">{low:g}"
        return None

    @staticmethod
    def _audit(
        field_name: str,
        action: str,
        before: str | None,
        after: str | None,
        rule: str,
        source: str,
    ) -> AuditEntry:
        return AuditEntry(
            field=field_name,
            action=action,  # type: ignore[arg-type]
            before=before,
            after=after,
            rule=rule,
            source=source,
            at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _issue_dict(issue: IngestionIssue) -> dict[str, Any]:
        return {
            "source": issue.source,
            "row_ref": issue.row_ref,
            "code": issue.code,
            "message": issue.message,
            "severity": issue.severity,
            "raw_excerpt": issue.raw_excerpt,
        }
