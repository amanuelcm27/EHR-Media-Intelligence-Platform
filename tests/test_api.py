"""API tests for search, patient detail and FHIR endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ehr_media.api import create_app
from ehr_media.config import Settings
from ehr_media.fhir import FhirPipeline, SqliteStore
from ehr_media.search import HashingEmbeddingProvider, SearchIndexer
from ehr_media.summaries import ClinicalSummarizer, OfflineSummaryProvider
from tests.factories import three_topic_patients


def prepare_platform(settings: Settings) -> SqliteStore:
    settings.ensure_dirs()
    store = SqliteStore(settings.db_path)
    store.initialize()
    patients = three_topic_patients()
    FhirPipeline(settings, store=store).run(
        patients, write_artifacts=False, persist=True
    )
    summarizer = ClinicalSummarizer(
        settings, store=store, provider=OfflineSummaryProvider(), persist=True
    )
    summaries = {patient.id: summarizer.summarize(patient) for patient in patients}
    SearchIndexer(
        settings,
        store=store,
        embedder=HashingEmbeddingProvider(settings.embedding_dim),
        backend="numpy",
    ).build(patients, summaries)
    return store


@pytest.fixture
def client(settings: Settings):
    prepare_platform(settings)
    app = create_app(
        settings,
        auto_build=False,
        embedder=HashingEmbeddingProvider(settings.embedding_dim),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_filters(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["stats"]["patients"] == 3
    assert payload["stats"]["bundles"] == 3
    assert payload["stats"]["valid_bundles"] == 3
    assert payload["index"]["provider"] == "hashing"
    assert payload["index"]["backend"] == "numpy"

    filters = client.get("/filters")
    assert filters.status_code == 200
    body = filters.json()
    assert set(body["resource_types"]) == {
        "DiagnosticReport",
        "DocumentReference",
        "Encounter",
        "Observation",
        "Patient",
        "Summary",
    }
    assert body["date_range"]["min"] <= body["date_range"]["max"]
    assert body["patients"] == 3


def test_search_endpoint_returns_top_five(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "heart failure shortness of breath", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "heart failure shortness of breath"
    assert body["count"] <= 5
    assert body["took_ms"] < 2000
    assert body["results"]
    assert body["results"][0]["patient_name"] == "Eleanor Vance"
    assert body["results"][0]["score"] > 0
    assert body["results"][0]["ai_snippet"]
    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["results"][0]["rank"] == 1


def test_search_endpoint_filters_and_validation(client: TestClient) -> None:
    labs = client.post(
        "/search",
        json={"query": "potassium", "top_k": 10, "resource_types": ["lab"]},
    )
    assert labs.status_code == 200
    assert {result["resource_type"] for result in labs.json()["results"]} == {
        "Observation"
    }

    dated = client.post(
        "/search",
        json={
            "query": "heart failure",
            "start_date": "2026-02-01",
            "end_date": "2026-02-28",
            "top_k": 10,
        },
    )
    assert dated.status_code == 200
    assert all(
        result["effective_date"].startswith("2026-02")
        for result in dated.json()["results"]
    )

    assert client.post("/search", json={"query": "  "}).status_code == 422
    assert (
        client.post("/search", json={"query": "heart", "top_k": 99}).status_code == 422
    )
    assert (
        client.post(
            "/search", json={"query": "heart", "resource_types": ["xray"]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/search",
            json={
                "query": "heart",
                "start_date": "2026-05-01",
                "end_date": "2026-01-01",
            },
        ).status_code
        == 422
    )


def test_patient_list_and_detail(client: TestClient) -> None:
    listing = client.get("/patients")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 3
    assert body["count"] == 3
    assert all(patient["has_summary"] for patient in body["patients"])
    assert all(patient["mrn_display"].startswith("MRN-") for patient in body["patients"])

    searched = client.get("/patients", params={"q": "Vance"})
    assert searched.status_code == 200
    assert searched.json()["total"] == 1
    patient_id = searched.json()["patients"][0]["patient_id"]

    detail = client.get(f"/patients/{patient_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["patient"]["id"] == patient_id
    assert payload["summary"]["patient_id"] == patient_id
    assert "Chief concern:" in payload["summary"]["summary"]
    assert payload["validation"]["valid"] is True
    assert payload["resource_counts"]["Patient"] == 1
    assert payload["resource_counts"]["DocumentReference"] == 1
    assert payload["resources"]
    assert payload["documents"][0]["text"]


def test_patient_summary_bundle_and_resource_endpoints(client: TestClient) -> None:
    patient_id = client.get("/patients").json()["patients"][0]["patient_id"]

    summary = client.get(f"/patients/{patient_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["word_count"] <= 200

    refreshed = client.get(
        f"/patients/{patient_id}/summary", params={"force": "true"}
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["cached"] is False

    bundle = client.get(f"/patients/{patient_id}/bundle")
    assert bundle.status_code == 200
    bundle_body = bundle.json()
    assert bundle_body["resourceType"] == "Bundle"
    assert bundle_body["type"] == "collection"

    document = next(
        entry["resource"]
        for entry in bundle_body["entry"]
        if entry["resource"]["resourceType"] == "DocumentReference"
    )
    resource = client.get(f"/fhir/{document['id']}")
    assert resource.status_code == 200
    assert resource.json()["id"] == document["id"]


def test_404_responses(client: TestClient) -> None:
    assert client.get("/patients/does-not-exist").status_code == 404
    assert client.get("/patients/does-not-exist/bundle").status_code == 404
    assert client.get("/patients/does-not-exist/summary").status_code == 404
    assert client.get("/fhir/does-not-exist").status_code == 404


def test_missing_index_returns_503(settings: Settings) -> None:
    settings.ensure_dirs()
    app = create_app(settings, auto_build=False)
    with TestClient(app) as test_client:
        response = test_client.post("/search", json={"query": "heart failure"})
    assert response.status_code == 503
    assert "index" in response.json()["detail"].lower()


def test_root_reports_api_when_frontend_missing(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "EHR Media Intelligence Platform" in response.text
