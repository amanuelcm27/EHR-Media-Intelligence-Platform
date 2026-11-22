"""FastAPI application for the EHR Media Intelligence Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..bootstrap import ensure_platform, patient_from_payload
from ..config import Settings, get_settings
from ..fhir import SqliteStore
from ..ids import mrn_display
from ..search import (
    SearchIndexUnavailable,
    SearchResponse,
    SearchService,
    VALID_RESOURCE_TYPES,
)
from ..summaries import ClinicalSummarizer

logger = logging.getLogger("ehr_media.api")


@dataclass
class Services:
    settings: Settings
    store: SqliteStore
    search: SearchService
    summarizer: ClinicalSummarizer


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    resource_types: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    patient_id: str | None = None
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class PatientListResponse(BaseModel):
    total: int
    count: int
    limit: int
    offset: int
    patients: list[dict[str, Any]]


class PatientDetailResponse(BaseModel):
    patient: dict[str, Any]
    summary: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    resource_counts: dict[str, int] = Field(default_factory=dict)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)


router = APIRouter()


def get_services(request: Request) -> Services:
    return request.app.state.services


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    services: Services = request.app.state.services
    stats = services.store.stats()
    return {
        "status": "ok",
        "version": __version__,
        "build": getattr(request.app.state, "build_info", None),
        "stats": stats,
        "index": {
            "chunk_count": services.store.get_meta("chunk_count"),
            "provider": services.store.get_meta("embedding_provider"),
            "model": services.store.get_meta("embedding_model"),
            "backend": services.store.get_meta("vector_backend"),
            "built_at": services.store.get_meta("built_at"),
        },
    }


@router.get("/filters")
def filters(services: Services = Depends(get_services)) -> dict[str, Any]:
    chunks = services.store.get_chunks()
    resource_types = sorted({chunk["resource_type"] for chunk in chunks})
    dates = sorted(
        chunk["effective_date"] for chunk in chunks if chunk.get("effective_date")
    )
    return {
        "resource_types": resource_types,
        "all_resource_types": VALID_RESOURCE_TYPES,
        "date_range": {
            "min": dates[0] if dates else None,
            "max": dates[-1] if dates else None,
        },
        "patients": len(services.store.list_patients()),
        "chunks": len(chunks),
    }


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, services: Services = Depends(get_services)) -> SearchResponse:
    try:
        return services.search.search(
            payload.query,
            top_k=payload.top_k,
            resource_types=payload.resource_types,
            start_date=payload.start_date,
            end_date=payload.end_date,
            patient_id=payload.patient_id,
            min_score=payload.min_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SearchIndexUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/patients", response_model=PatientListResponse)
def list_patients(
    q: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    services: Services = Depends(get_services),
) -> PatientListResponse:
    rows = services.store.list_patients()
    if q:
        needle = q.strip().casefold()
        rows = [
            row
            for row in rows
            if needle in row["display_name"].casefold()
            or needle in row["mrn"].casefold()
            or needle in mrn_display(row["mrn"]).casefold()
        ]
    total = len(rows)
    page = rows[offset : offset + limit]
    patients = [
        {
            **row,
            "mrn_display": mrn_display(row["mrn"]),
            "has_summary": services.store.get_summary(row["patient_id"]) is not None,
        }
        for row in page
    ]
    return PatientListResponse(
        total=total,
        count=len(patients),
        limit=limit,
        offset=offset,
        patients=patients,
    )


@router.get("/patients/{patient_id}", response_model=PatientDetailResponse)
def get_patient(
    patient_id: str,
    refresh_summary: bool = Query(default=False),
    services: Services = Depends(get_services),
) -> PatientDetailResponse:
    payload = services.store.get_patient(patient_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id!r} not found")
    patient = patient_from_payload(payload)

    summary = services.store.get_summary(patient_id)
    if summary is None or refresh_summary:
        try:
            summary = services.summarizer.summarize(patient, force=refresh_summary).model_dump(
                mode="json"
            )
        except Exception as exc:  # summary failure must not hide the record
            logger.warning("Summary generation failed for %s: %s", patient_id, exc)
            summary = None

    resources = services.store.get_resources(patient_id)
    counts: dict[str, int] = {}
    for resource in resources:
        resource_type = resource.get("resourceType", "Unknown")
        counts[resource_type] = counts.get(resource_type, 0) + 1

    documents = [
        {
            "id": document.id,
            "title": document.title,
            "doc_type": document.doc_type,
            "authored_on": document.authored_on.isoformat()
            if document.authored_on
            else None,
            "author": document.author,
            "word_count": document.word_count,
            "text": document.text,
            "anomalies": [anomaly.model_dump() for anomaly in document.anomalies],
        }
        for document in patient.documents
    ]

    return PatientDetailResponse(
        patient=patient.model_dump(mode="json"),
        summary=summary,
        validation=services.store.get_validation(patient_id),
        resource_counts=dict(sorted(counts.items())),
        resources=resources,
        documents=documents,
    )


@router.get("/patients/{patient_id}/bundle")
def get_bundle(
    patient_id: str, services: Services = Depends(get_services)
) -> dict[str, Any]:
    bundle = services.store.get_bundle(patient_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"No Bundle for patient {patient_id!r}")
    return bundle


@router.get("/patients/{patient_id}/summary")
def get_summary(
    patient_id: str,
    force: bool = Query(default=False),
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    payload = services.store.get_patient(patient_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id!r} not found")
    patient = patient_from_payload(payload)
    try:
        summary = services.summarizer.summarize(patient, force=force)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Summary generation failed: {exc}") from exc
    return summary.model_dump(mode="json")


@router.get("/fhir/{resource_id}")
def get_resource(
    resource_id: str, services: Services = Depends(get_services)
) -> dict[str, Any]:
    resource = services.store.get_resource(resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail=f"FHIR resource {resource_id!r} not found")
    return resource


def create_app(
    settings: Settings | None = None,
    *,
    auto_build: bool = True,
    embedder: Any | None = None,
) -> Any:
    from fastapi import FastAPI

    settings = settings or get_settings()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: Any):
        store = SqliteStore(settings.db_path)
        store.initialize()
        build_info: dict[str, Any] = {"status": "skipped"}
        if auto_build:
            try:
                build_info = ensure_platform(settings, store=store, embedder=embedder)
                logger.info("Platform ready: %s", build_info)
            except Exception as exc:  # keep the API up and report the failure
                logger.exception("Platform build failed")
                build_info = {"status": "error", "error": str(exc)}
        search_service = SearchService(settings, store=store, embedder=embedder)
        try:
            search_service.ensure_ready()
        except Exception as exc:  # index can be built later; search returns 503
            logger.warning("Search index is not ready at startup: %s", exc)
        app.state.services = Services(
            settings=settings,
            store=store,
            search=search_service,
            summarizer=ClinicalSummarizer(settings, store=store, persist=True),
        )
        app.state.build_info = build_info
        yield

    app = FastAPI(
        title="EHR Media Intelligence Platform",
        version=__version__,
        description=(
            "Synthetic EHR ingestion, FHIR R4 normalization, AI summaries and "
            "semantic search for clinical media."
        ),
        lifespan=lifespan,
    )
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)

    if settings.frontend_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(settings.frontend_dir)),
            name="static",
        )

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        frontend = settings.frontend_index
        if frontend.exists():
            return FileResponse(frontend)
        return JSONResponse(
            {
                "message": "EHR Media Intelligence Platform API",
                "docs": "/docs",
                "search": "POST /search",
                "frontend": "Run the Tailwind build to generate frontend/index.html",
            }
        )

    return app


app = create_app()
