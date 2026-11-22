"""Task 4 tests: chunks, embeddings, index persistence and search behavior."""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pytest

from ehr_media.config import Settings
from ehr_media.fhir import SqliteStore
from ehr_media.search import (
    HashingEmbeddingProvider,
    SearchIndexUnavailable,
    SearchIndexer,
    SearchService,
    VectorIndex,
    build_chunks,
    load_index,
)
from ehr_media.search.service import _normalize_resource_types
from ehr_media.summaries import ClinicalSummarizer, OfflineSummaryProvider
from tests.factories import make_patient, three_topic_patients


def build_corpus(
    settings: Settings,
    patients,
    *,
    with_summaries: bool = True,
):
    store = SqliteStore(settings.db_path)
    store.initialize()
    embedder = HashingEmbeddingProvider(settings.embedding_dim)
    summaries = {}
    if with_summaries:
        summarizer = ClinicalSummarizer(
            settings, store=store, provider=OfflineSummaryProvider(), persist=True
        )
        summaries = {patient.id: summarizer.summarize(patient) for patient in patients}
    result = SearchIndexer(
        settings, store=store, embedder=embedder, backend="numpy"
    ).build(patients, summaries)
    return store, embedder, result, summaries


def test_build_chunks_covers_every_resource_type() -> None:
    patients = three_topic_patients()
    summarizer = ClinicalSummarizer(persist=False)
    summaries = {patient.id: summarizer.summarize(patient) for patient in patients}

    chunks = build_chunks(patients, summaries, chunk_size=300, chunk_overlap=60)

    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.resource_type for chunk in chunks} == {
        "DiagnosticReport",
        "DocumentReference",
        "Encounter",
        "Observation",
        "Patient",
        "Summary",
    }
    assert {chunk.patient_id for chunk in chunks} == {patient.id for patient in patients}
    assert all(chunk.patient_name and chunk.mrn for chunk in chunks)
    assert any("shortness of breath" in chunk.text.lower() for chunk in chunks)
    assert any(chunk.effective_date == "2026-02-14" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunks_split_long_documents() -> None:
    patient = make_patient(
        note_text=" ".join(["dyspnea and edema"] * 200),
    )
    chunks = build_chunks([patient], chunk_size=300, chunk_overlap=60)
    document_chunks = [
        chunk for chunk in chunks if chunk.resource_type == "DocumentReference"
    ]

    assert len(document_chunks) > 1
    assert len({chunk.chunk_id for chunk in document_chunks}) == len(document_chunks)
    assert all(len(chunk.text) <= 320 for chunk in document_chunks)


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    embedder = HashingEmbeddingProvider(384)
    texts = [
        "heart failure with shortness of breath",
        "appendicitis with right lower quadrant pain",
    ]

    first = embedder.encode(texts)
    second = embedder.encode(texts)
    assert first.shape == (2, 384)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)

    query = embedder.encode_query("heart failure")
    heart = float(query @ first[0])
    appendix = float(query @ first[1])
    assert heart > appendix


def test_vector_index_returns_descending_scores_and_honors_mask() -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )
    index = VectorIndex(vectors, backend="numpy")

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), 3)
    assert [position for position, _ in hits] == [0, 1, 2]
    assert hits[0][1] >= hits[1][1] >= hits[2][1]

    masked = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        3,
        allowed=np.array([False, True, True]),
    )
    assert {position for position, _ in masked} == {1, 2}


def test_indexer_persists_chunks_vectors_and_manifest(settings: Settings) -> None:
    settings.ensure_dirs()
    patients = three_topic_patients()
    store, embedder, result, _summaries = build_corpus(settings, patients)

    assert result.chunk_count > 10
    assert result.dimension == settings.embedding_dim
    assert result.provider == "hashing"
    assert result.backend == "numpy"
    assert result.paths["vectors"].exists()
    assert result.paths["manifest"].exists()
    assert store.chunk_count() == result.chunk_count
    assert store.get_meta("chunk_count") == result.chunk_count
    assert store.get_meta("embedding_provider") == "hashing"

    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == result.chunk_count
    assert manifest["resource_types"]["DocumentReference"] >= 3

    chunks, vectors, meta = load_index(settings, store)
    assert len(chunks) == len(vectors) == result.chunk_count
    assert vectors.shape[1] == settings.embedding_dim
    assert meta["vector_backend"] == "numpy"


def test_search_returns_ranked_top_five_with_ai_snippets(settings: Settings) -> None:
    settings.ensure_dirs()
    patients = three_topic_patients()
    store, embedder, _result, _summaries = build_corpus(settings, patients)
    service = SearchService(settings, store=store, embedder=embedder)

    response = service.search("heart failure shortness of breath", top_k=5)

    assert response.query == "heart failure shortness of breath"
    assert response.top_k == 5
    assert 1 <= response.count <= 5
    assert response.took_ms < 2000
    assert response.index["provider"] == "hashing"
    scores = [result.score for result in response.results]
    assert scores == sorted(scores, reverse=True)
    assert response.results[0].patient_name == "Eleanor Vance"
    assert response.results[0].resource_type in {
        "DocumentReference",
        "Summary",
        "Patient",
        "DiagnosticReport",
        "Observation",
        "Encounter",
    }
    assert response.results[0].snippet
    assert response.results[0].ai_snippet
    assert all(result.rank == index for index, result in enumerate(response.results, 1))
    assert all(result.mrn and result.patient_id for result in response.results)


def test_search_filters_by_resource_type_date_and_patient(settings: Settings) -> None:
    settings.ensure_dirs()
    patients = three_topic_patients()
    store, embedder, _result, _summaries = build_corpus(settings, patients)
    service = SearchService(settings, store=store, embedder=embedder)
    eleanor = patients[0]

    labs = service.search("potassium", resource_types=["lab"], top_k=10)
    assert labs.results
    assert {result.resource_type for result in labs.results} == {"Observation"}

    notes = service.search("heart", resource_types=["notes"], top_k=10)
    assert notes.results
    assert {result.resource_type for result in notes.results} == {"DocumentReference"}

    february = service.search(
        "heart failure",
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 28),
        top_k=10,
    )
    assert february.results
    assert all(
        result.effective_date and result.effective_date.startswith("2026-02")
        for result in february.results
    )

    for_patient = service.search("heart failure", patient_id=eleanor.id, top_k=10)
    assert for_patient.results
    assert {result.patient_id for result in for_patient.results} == {eleanor.id}

    empty = service.search(
        "heart failure", resource_types=["Encounter"], start_date="2020-01-01", end_date="2020-01-02"
    )
    assert empty.results == []
    assert empty.count == 0


def test_search_validation(settings: Settings) -> None:
    settings.ensure_dirs()
    patients = three_topic_patients()
    store, embedder, _result, _summaries = build_corpus(settings, patients)
    service = SearchService(settings, store=store, embedder=embedder)

    with pytest.raises(ValueError):
        service.search("   ")
    with pytest.raises(ValueError):
        service.search("heart", resource_types=["xray"])
    with pytest.raises(ValueError):
        service.search("heart", start_date="2026-05-01", end_date="2026-01-01")
    with pytest.raises(ValueError):
        service.search("heart", start_date="not-a-date")

    assert _normalize_resource_types(["lab", "Observations", "note"]) == [
        "Observation",
        "DocumentReference",
    ]


def test_search_requires_a_built_index(settings: Settings) -> None:
    settings.ensure_dirs()
    service = SearchService(settings, embedder=HashingEmbeddingProvider(settings.embedding_dim))
    with pytest.raises(SearchIndexUnavailable):
        service.search("heart failure")


def test_search_latency_for_fifty_patients(settings: Settings) -> None:
    settings.ensure_dirs()
    patients = []
    for index in range(50):
        patients.append(
            make_patient(
                mrn=f"006{index:05d}",
                given_name=f"Patient{index}",
                family_name="Synthetic",
                note_title=f"Progress note {index}",
                note_text=(
                    f"Patient {index} reports chest pain and dyspnea. "
                    "Assessment: heart failure with volume overload. "
                    "Plan: diuresis and echocardiogram."
                    if index % 2 == 0
                    else f"Patient {index} reports abdominal pain and fever. "
                    "Assessment: appendicitis. Plan: surgery."
                ),
                labs=[
                    {
                        "test_name": "Potassium",
                        "value": 4.0 + (index % 20) / 10,
                        "unit": "mmol/L",
                        "reference_range": "3.5-5.1",
                        "interpretation": "normal",
                        "collected_at": date(2026, 1, 1),
                    }
                ],
            )
        )
    store, embedder, result, _ = build_corpus(settings, patients, with_summaries=False)
    service = SearchService(settings, store=store, embedder=embedder)

    started = time.perf_counter()
    response = service.search("heart failure volume overload", top_k=5)
    wall_ms = (time.perf_counter() - started) * 1000

    assert result.chunk_count >= 50 * 4
    assert response.count == 5
    assert response.took_ms < 2000
    assert wall_ms < 2000
    assert response.results[0].patient_id in {patient.id for patient in patients}
