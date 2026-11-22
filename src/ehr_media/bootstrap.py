"""One-command platform build and startup reuse logic.

``build_platform`` runs every task in order: ingest/clean -> FHIR map/validate
-> summarize -> embed/index. ``ensure_platform`` reuses an existing database
and only rebuilds the pieces that are missing.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from .config import Settings, get_settings
from .fhir import FhirPipeline, SqliteStore
from .ingestion.pipeline import IngestionPipeline
from .models import PatientRecord
from .search import EmbeddingProvider, SearchIndexer, SearchService
from .summaries import ClinicalSummarizer, ClinicalSummary

_PATIENT_COMPUTED = {
    "display_name",
    "mrn_display",
    "abnormal_labs",
    "latest_document_date",
}
_DOCUMENT_COMPUTED = {"word_count"}
_LAB_COMPUTED = {"is_abnormal", "display_value"}


def patient_from_payload(payload: dict[str, Any]) -> PatientRecord:
    """Rebuild a PatientRecord from SQLite JSON, dropping computed fields."""
    data = {key: value for key, value in payload.items() if key not in _PATIENT_COMPUTED}
    data["documents"] = [
        {key: value for key, value in document.items() if key not in _DOCUMENT_COMPUTED}
        for document in data.get("documents", [])
    ]
    data["lab_results"] = [
        {key: value for key, value in lab.items() if key not in _LAB_COMPUTED}
        for lab in data.get("lab_results", [])
    ]
    return PatientRecord.model_validate(data)


def load_patients_from_store(store: SqliteStore) -> list[PatientRecord]:
    patients: list[PatientRecord] = []
    for row in store.list_patients():
        payload = store.get_patient(row["patient_id"])
        if payload:
            patients.append(patient_from_payload(payload))
    return patients


def load_summaries_from_store(
    store: SqliteStore, patients: Iterable[PatientRecord]
) -> dict[str, ClinicalSummary]:
    summaries: dict[str, ClinicalSummary] = {}
    for patient in patients:
        payload = store.get_summary(patient.id)
        if payload:
            summaries[patient.id] = ClinicalSummary.model_validate(payload)
    return summaries


def build_platform(
    settings: Settings | None = None,
    *,
    store: SqliteStore | None = None,
    embedder: EmbeddingProvider | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Run ingestion, FHIR, summaries and the search index end to end."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    started = time.perf_counter()
    store = store or SqliteStore(settings.db_path)
    store.initialize()

    ingestion = IngestionPipeline.from_paths(settings.data_dir)
    ingestion_result = ingestion.run()
    if write_artifacts:
        ingestion_result.write(settings)

    fhir_result = FhirPipeline(settings, store=store).run(
        ingestion_result.patients,
        write_artifacts=write_artifacts,
        persist=True,
    )
    summary_result = ClinicalSummarizer(
        settings, store=store, persist=True
    ).summarize_all(
        ingestion_result.patients,
        write_artifacts=write_artifacts,
    )
    index_result = SearchIndexer(
        settings, store=store, embedder=embedder
    ).build(
        ingestion_result.patients,
        summary_result.summaries,
    )
    return {
        "status": "built",
        "patients": len(ingestion_result.patients),
        "documents": sum(len(patient.documents) for patient in ingestion_result.patients),
        "labs": sum(len(patient.lab_results) for patient in ingestion_result.patients),
        "fhir": fhir_result.summary(),
        "summaries": {
            "count": summary_result.count,
            "generated": summary_result.generated,
            "cached": summary_result.cached,
            "fallbacks": summary_result.fallbacks,
            "providers": summary_result.providers,
        },
        "index": index_result.summary(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def ensure_platform(
    settings: Settings | None = None,
    *,
    store: SqliteStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Build only what is missing, then return current platform stats."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = store or SqliteStore(settings.db_path)
    store.initialize()
    stats = store.stats()

    if stats["patients"] == 0 or stats["bundles"] == 0:
        return build_platform(settings, store=store, embedder=embedder)

    patients = load_patients_from_store(store)
    summaries = load_summaries_from_store(store, patients)
    if len(summaries) < len(patients):
        summarizer = ClinicalSummarizer(settings, store=store, persist=True)
        for patient in patients:
            if patient.id not in summaries:
                summaries[patient.id] = summarizer.summarize(patient)

    vectors_path = settings.index_dir / "vectors.npy"
    indexed = store.get_meta("chunk_count")
    if not vectors_path.exists() or not indexed or int(indexed) != store.chunk_count():
        index_result = SearchIndexer(
            settings, store=store, embedder=embedder
        ).build(patients, summaries)
        return {
            "status": "index_rebuilt",
            "patients": len(patients),
            "summaries": len(summaries),
            "index": index_result.summary(),
        }
    return {
        "status": "reused",
        "patients": len(patients),
        "summaries": len(summaries),
        "chunks": store.chunk_count(),
    }


def platform_stats(settings: Settings | None = None, store: SqliteStore | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    store = store or SqliteStore(settings.db_path)
    stats = store.stats()
    stats["index"] = {
        "chunk_count": store.get_meta("chunk_count"),
        "provider": store.get_meta("embedding_provider"),
        "model": store.get_meta("embedding_model"),
        "backend": store.get_meta("vector_backend"),
        "built_at": store.get_meta("built_at"),
    }
    return stats


def search_service(
    settings: Settings | None = None,
    *,
    store: SqliteStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> SearchService:
    settings = settings or get_settings()
    return SearchService(
        settings,
        store=store or SqliteStore(settings.db_path),
        embedder=embedder,
    )
