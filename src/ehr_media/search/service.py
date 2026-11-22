"""Semantic search over indexed clinical chunks."""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..fhir.store import SqliteStore
from .embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    resolve_embedding_provider,
)
from .index import VectorIndex, load_index

_RESOURCE_ALIASES = {
    "documentreference": "DocumentReference",
    "document": "DocumentReference",
    "documents": "DocumentReference",
    "note": "DocumentReference",
    "notes": "DocumentReference",
    "observation": "Observation",
    "observations": "Observation",
    "lab": "Observation",
    "labs": "Observation",
    "lab result": "Observation",
    "diagnosticreport": "DiagnosticReport",
    "report": "DiagnosticReport",
    "reports": "DiagnosticReport",
    "panel": "DiagnosticReport",
    "encounter": "Encounter",
    "encounters": "Encounter",
    "visit": "Encounter",
    "summary": "Summary",
    "summaries": "Summary",
    "clinicalsummary": "Summary",
    "ai summary": "Summary",
    "patient": "Patient",
    "patients": "Patient",
}
VALID_RESOURCE_TYPES = sorted(set(_RESOURCE_ALIASES.values()))
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SearchIndexUnavailable(RuntimeError):
    pass


class SearchFilters(BaseModel):
    resource_types: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    patient_id: str | None = None


class SearchResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    patient_id: str
    patient_name: str
    mrn: str
    resource_type: str
    resource_id: str
    title: str | None = None
    effective_date: str | None = None
    section: str | None = None
    snippet: str
    text: str
    ai_snippet: str | None = None
    summary_confidence: float | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    count: int
    took_ms: float
    filters: SearchFilters
    results: list[SearchResult] = Field(default_factory=list)
    index: dict[str, Any] = Field(default_factory=dict)


def _terms(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1}


def _normalize_resource_types(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        canonical = _RESOURCE_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unknown resource type {value!r}; expected one of "
                + ", ".join(VALID_RESOURCE_TYPES)
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _snippet(text: str, query: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    query_terms = _terms(query)
    best = ""
    best_score = -1
    for sentence in _SENTENCE_RE.split(text):
        terms = _terms(sentence)
        score = len(query_terms & terms)
        if score > best_score:
            best, best_score = sentence, score
    if best_score <= 0:
        best = text
    best = best.strip()
    if len(best) > limit:
        best = best[:limit].rsplit(" ", 1)[0] + "…"
    return best


class SearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SqliteStore | None = None,
        embedder: EmbeddingProvider | None = None,
        chunks: list[dict[str, Any]] | None = None,
        vectors: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder
        self._chunks = chunks
        self._vectors = vectors
        self._meta = meta
        self._index: VectorIndex | None = None
        self._summary_cache: dict[str, dict[str, Any] | None] = {}

    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        if self._chunks is not None and self._vectors is not None:
            if self._index is None:
                self._index = VectorIndex(
                    self._vectors, backend=self._meta_backend()
                )
            if self.embedder is None:
                self.embedder = self._build_embedder()
            return
        store = self._ensure_store()
        store.initialize()
        try:
            chunks, vectors, meta = load_index(self.settings, store)
        except (FileNotFoundError, ValueError) as exc:
            raise SearchIndexUnavailable(
                f"Search index is not ready: {exc}. Run `python -m ehr_media.cli build`."
            ) from exc
        self._chunks = chunks
        self._vectors = vectors
        self._meta = meta
        self._index = VectorIndex(vectors, backend=self._meta_backend())
        if self.embedder is None:
            self.embedder = self._build_embedder()

    def _meta_backend(self) -> str:
        meta = self._meta or {}
        backend = str(meta.get("vector_backend") or self.settings.vector_backend)
        return backend if backend in {"auto", "faiss", "numpy"} else "auto"

    def _build_embedder(self) -> EmbeddingProvider:
        meta = self._meta or {}
        provider = str(meta.get("embedding_provider") or "")
        model = str(meta.get("embedding_model") or "")
        dimension = int(meta.get("embedding_dimension") or self.settings.embedding_dim)
        try:
            if provider.startswith("hashing"):
                return HashingEmbeddingProvider(dimension)
            if provider == "sentence-transformers":
                return SentenceTransformerEmbeddingProvider(
                    model or self.settings.embedding_model
                )
            if provider == "openai":
                return OpenAIEmbeddingProvider(self.settings)
            return resolve_embedding_provider(self.settings)
        except EmbeddingProviderError as exc:
            raise SearchIndexUnavailable(
                f"Embedding provider {provider or 'auto'!r} is unavailable: {exc}"
            ) from exc

    def _ensure_store(self) -> SqliteStore:
        if self.store is None:
            self.store = SqliteStore(self.settings.db_path)
        return self.store

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        resource_types: Iterable[str] | None = None,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        patient_id: str | None = None,
        min_score: float | None = None,
    ) -> SearchResponse:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty")
        self.ensure_ready()
        assert self._chunks is not None and self._index is not None
        started = time.perf_counter()

        top_k = int(top_k or self.settings.default_top_k)
        top_k = max(1, min(top_k, self.settings.max_top_k))
        canonical_types = _normalize_resource_types(resource_types)
        start = _coerce_date(start_date, "start_date")
        end = _coerce_date(end_date, "end_date")
        if start and end and start > end:
            raise ValueError("start_date must be on or before end_date")

        filters = SearchFilters(
            resource_types=canonical_types,
            start_date=start,
            end_date=end,
            patient_id=patient_id,
        )
        allowed = self._filter_mask(filters)
        if allowed is not None and not allowed.any():
            return self._response(query, top_k, filters, started, [])

        embedder = self.embedder
        assert embedder is not None
        query_vector = embedder.encode_query(query)
        if self._index.dimension and query_vector.shape[0] != self._index.dimension:
            raise SearchIndexUnavailable(
                f"Query dimension {query_vector.shape[0]} does not match index "
                f"dimension {self._index.dimension}"
            )

        candidate_k = min(len(self._chunks), max(top_k * 6, top_k))
        raw = self._index.search(query_vector, candidate_k, allowed=allowed)
        query_terms = _terms(query)
        weight = max(0.0, min(self.settings.hybrid_weight, 1.0))
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for position, cosine in raw:
            chunk = self._chunks[position]
            lexical = 0.0
            if query_terms:
                lexical = len(query_terms & _terms(chunk["text"])) / len(query_terms)
            combined = (1.0 - weight) * cosine + weight * lexical
            if min_score is not None and combined < min_score:
                continue
            scored.append((combined, cosine, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        deduped: list[tuple[float, float, dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        for combined, cosine, chunk in scored:
            key = (chunk["resource_type"], chunk["resource_id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append((combined, cosine, chunk))
            if len(deduped) >= top_k:
                break

        results: list[SearchResult] = []
        for rank, (combined, _cosine, chunk) in enumerate(deduped, start=1):
            summary = self._summary_for(chunk["patient_id"])
            ai_snippet = None
            confidence = None
            if summary:
                ai_snippet = _summary_snippet(summary)
                confidence = summary.get("confidence")
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(float(combined), 4),
                    chunk_id=chunk["chunk_id"],
                    patient_id=chunk["patient_id"],
                    patient_name=chunk["patient_name"],
                    mrn=chunk["mrn"],
                    resource_type=chunk["resource_type"],
                    resource_id=chunk["resource_id"],
                    title=chunk.get("title"),
                    effective_date=chunk.get("effective_date"),
                    section=chunk.get("section"),
                    snippet=_snippet(chunk["text"], query),
                    text=chunk["text"][:1600],
                    ai_snippet=ai_snippet,
                    summary_confidence=confidence,
                )
            )
        return self._response(query, top_k, filters, started, results)

    # ------------------------------------------------------------------
    def _filter_mask(self, filters: SearchFilters) -> np.ndarray | None:
        assert self._chunks is not None
        if (
            not filters.resource_types
            and filters.start_date is None
            and filters.end_date is None
            and filters.patient_id is None
        ):
            return None
        allowed = np.zeros(len(self._chunks), dtype=bool)
        types = set(filters.resource_types)
        start = filters.start_date.isoformat() if filters.start_date else None
        end = filters.end_date.isoformat() if filters.end_date else None
        for index, chunk in enumerate(self._chunks):
            if types and chunk["resource_type"] not in types:
                continue
            if filters.patient_id and chunk["patient_id"] != filters.patient_id:
                continue
            effective = chunk.get("effective_date")
            if start or end:
                if not effective:
                    continue
                if start and effective < start:
                    continue
                if end and effective > end:
                    continue
            allowed[index] = True
        return allowed

    def _summary_for(self, patient_id: str) -> dict[str, Any] | None:
        if patient_id in self._summary_cache:
            return self._summary_cache[patient_id]
        summary = None
        if self.store is not None:
            try:
                summary = self.store.get_summary(patient_id)
            except Exception:  # pragma: no cover - defensive
                summary = None
        self._summary_cache[patient_id] = summary
        return summary

    def _response(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters,
        started: float,
        results: list[SearchResult],
    ) -> SearchResponse:
        meta = self._meta or {}
        return SearchResponse(
            query=query,
            top_k=top_k,
            count=len(results),
            took_ms=round((time.perf_counter() - started) * 1000, 1),
            filters=filters,
            results=results,
            index={
                "provider": meta.get("embedding_provider"),
                "model": meta.get("embedding_model"),
                "dimension": meta.get("embedding_dimension"),
                "backend": meta.get("vector_backend"),
                "chunk_count": meta.get("chunk_count"),
                "built_at": meta.get("built_at"),
            },
        )


def _coerce_date(value: date | str | None, field: str) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)") from exc


def _summary_snippet(summary: dict[str, Any], limit: int = 240) -> str | None:
    text = summary.get("summary") or ""
    if not text:
        sections = summary.get("sections") or {}
        text = sections.get("chief_concern") or ""
    text = " ".join(str(text).split())
    if not text:
        return None
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text
