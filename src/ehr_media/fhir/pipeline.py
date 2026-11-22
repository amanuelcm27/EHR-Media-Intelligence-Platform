"""Task 2 orchestration: map, validate, export and persist FHIR Bundles.

The pipeline is deliberately side-effect explicit. Callers can run it against
in-memory :class:`PatientRecord` objects (tests) or point it at the Task 1
ingestion pipeline (CLI / API startup) and choose whether artifacts and SQLite
should be written.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings, get_settings
from ..models import PatientRecord
from .mapper import FhirMapper
from .store import SqliteStore
from .validator import BundleValidation, FhirValidator, ValidationReport


@dataclass
class FhirPipelineResult:
    """Everything produced by one mapping + validation run."""

    bundles: dict[str, dict[str, Any]] = field(default_factory=dict)
    validations: list[BundleValidation] = field(default_factory=list)
    report: ValidationReport | None = None
    paths: dict[str, Path] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def total_bundles(self) -> int:
        return len(self.bundles)

    @property
    def valid_bundles(self) -> int:
        return sum(1 for validation in self.validations if validation.valid)

    @property
    def invalid_bundles(self) -> int:
        return self.total_bundles - self.valid_bundles

    def bundle_for(self, patient_id: str) -> dict[str, Any] | None:
        return self.bundles.get(patient_id)

    def summary(self) -> dict[str, Any]:
        return {
            "bundles": self.total_bundles,
            "valid": self.valid_bundles,
            "invalid": self.invalid_bundles,
            "errors": sum(validation.error_count for validation in self.validations),
            "warnings": sum(validation.warning_count for validation in self.validations),
            "duration_ms": round(self.duration_ms, 1),
        }


class FhirPipeline:
    """Maps cleaned patients to validated FHIR R4 Bundles."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SqliteStore | None = None,
        mapper: type[FhirMapper] | None = None,
        validator: FhirValidator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.mapper_cls = mapper or FhirMapper
        self.validator = validator or FhirValidator()

    # ------------------------------------------------------------------
    def map_patient(self, patient: PatientRecord) -> dict[str, Any]:
        return self.mapper_cls(patient).build_bundle()

    def validate(
        self, bundle: dict[str, Any], patient: PatientRecord | None = None
    ) -> BundleValidation:
        return self.validator.validate(bundle, patient)

    def run(
        self,
        patients: Iterable[PatientRecord],
        *,
        write_artifacts: bool = True,
        persist: bool = True,
    ) -> FhirPipelineResult:
        started = time.perf_counter()
        patients = list(patients)
        result = FhirPipelineResult()

        for patient in patients:
            bundle = self.map_patient(patient)
            validation = self.validate(bundle, patient)
            result.bundles[patient.id] = bundle
            result.validations.append(validation)

        result.report = ValidationReport.from_validations(result.validations)
        result.duration_ms = (time.perf_counter() - started) * 1000

        if write_artifacts:
            result.paths.update(self._write_artifacts(result))

        if persist:
            store = self.store or SqliteStore(self.settings.db_path)
            store.initialize()
            for patient in patients:
                validation = next(
                    item for item in result.validations if item.patient_id == patient.id
                )
                store.save_bundle(patient, result.bundles[patient.id], validation)
            result.paths["database"] = store.path
            self.store = store

        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    def run_from_paths(
        self,
        paths: Iterable[str | Path] | str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        write_artifacts: bool = True,
        persist: bool = True,
    ) -> FhirPipelineResult:
        """Convenience entry point: ingest raw files, then map + validate."""
        from ..ingestion.pipeline import IngestionPipeline

        paths = paths if paths is not None else self.settings.data_dir
        data_dir = data_dir if data_dir is not None else self.settings.data_dir
        ingestion = IngestionPipeline.from_paths(paths, data_dir=data_dir)
        ingestion_result = ingestion.run()
        if write_artifacts:
            ingestion_result.write(self.settings)
        return self.run(
            ingestion_result.patients,
            write_artifacts=write_artifacts,
            persist=persist,
        )

    # ------------------------------------------------------------------
    def _write_artifacts(self, result: FhirPipelineResult) -> dict[str, Path]:
        settings = self.settings
        settings.ensure_dirs()
        paths: dict[str, Path] = {}

        for patient_id, bundle in result.bundles.items():
            path = settings.fhir_dir / f"{patient_id}.json"
            path.write_text(
                json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            paths[patient_id] = path

        report = result.report
        assert report is not None  # populated by run()
        report_json = settings.reports_dir / "fhir_validation_report.json"
        report_md = settings.reports_dir / "fhir_validation_report.md"
        report.write(report_json, report_md)
        paths["report_json"] = report_json
        paths["report_md"] = report_md

        manifest = settings.fhir_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "schema_version": report.schema_version,
                    "summary": result.summary(),
                    "resource_counts": report.resource_counts,
                    "bundles": [
                        {
                            "patient_id": validation.patient_id,
                            "mrn": validation.mrn,
                            "bundle_id": validation.bundle_id,
                            "valid": validation.valid,
                            "errors": validation.error_count,
                            "warnings": validation.warning_count,
                            "resources": validation.resource_counts,
                            "file": f"{validation.patient_id}.json",
                        }
                        for validation in result.validations
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["manifest"] = manifest
        return paths
