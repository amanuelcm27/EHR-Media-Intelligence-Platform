"""Vector index construction and persistence (FAISS when available, NumPy otherwise)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import Settings, get_settings
from ..fhir.store import SqliteStore
from ..models import PatientRecord
from ..summaries.models import ClinicalSummary
from .chunks import Chunk, build_chunks
from .embeddings import EmbeddingProvider, resolve_embedding_provider

try:  # optional acceleration
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class VectorIndex:
    """Cosine-similarity index over normalized vectors."""

    def __init__(
        self,
        vectors: np.ndarray | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        self.vectors = (
            np.asarray(vectors, dtype=np.float32)
            if vectors is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.reshape(1, -1)
        self._faiss = None
        self.backend = "numpy"
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError(f"Unknown vector backend {backend!r}")
        if backend in {"auto", "faiss"} and faiss is not None and len(self.vectors):
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
            self.backend = "faiss"

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        if not len(self.vectors) or top_k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{self.dimension}"
            )
        if allowed is not None:
            indices = np.flatnonzero(allowed)
            if not len(indices):
                return []
            scores = self.vectors[indices] @ query
            order = np.argsort(-scores)[:top_k]
            return [
                (int(indices[position]), float(scores[position])) for position in order
            ]

        if self._faiss is not None:
            scores, positions = self._faiss.search(query.reshape(1, -1), top_k)
            return [
                (int(position), float(score))
                for score, position in zip(scores[0], positions[0])
                if position >= 0
            ]
        scores = self.vectors @ query
        order = np.argsort(-scores)[:top_k]
        return [(int(position), float(scores[position])) for position in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path, *, backend: str = "auto") -> VectorIndex:
        vectors = np.load(path)
        return cls(vectors, backend=backend)


@dataclass
class IndexResult:
    chunk_count: int
    dimension: int
    provider: str
    model: str
    backend: str
    duration_ms: float
    paths: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "dimension": self.dimension,
            "provider": self.provider,
            "model": self.model,
            "backend": self.backend,
            "duration_ms": round(self.duration_ms, 1),
        }


class SearchIndexer:
    """Embeds chunks, writes them to SQLite, and saves the vector matrix."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SqliteStore | None = None,
        embedder: EmbeddingProvider | None = None,
        backend: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder
        self.backend = backend or self.settings.vector_backend
        self._resolved_embedder: EmbeddingProvider | None = embedder

    def _embedder(self) -> EmbeddingProvider:
        if self._resolved_embedder is None:
            self._resolved_embedder = resolve_embedding_provider(self.settings)
        return self._resolved_embedder

    def _ensure_store(self) -> SqliteStore:
        if self.store is None:
            self.store = SqliteStore(self.settings.db_path)
            self.store.initialize()
        return self.store

    def build(
        self,
        patients: Iterable[PatientRecord],
        summaries: dict[str, ClinicalSummary] | Iterable[ClinicalSummary] | None = None,
        *,
        write_artifacts: bool = True,
    ) -> IndexResult:
        started = time.perf_counter()
        settings = self.settings
        settings.ensure_dirs()
        store = self._ensure_store()
        if isinstance(summaries, dict):
            summary_map = summaries
        elif summaries is None:
            summary_map = {}
        else:
            summary_map = {summary.patient_id: summary for summary in summaries}

        chunks = build_chunks(
            patients,
            summary_map,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        embedder = self._embedder()
        vectors = embedder.encode(
            [chunk.text for chunk in chunks], batch_size=settings.embedding_batch_size
        )
        index = VectorIndex(vectors, backend=self.backend)
        if vectors.shape[1] != embedder.dimension and len(chunks):
            raise RuntimeError(
                f"Embedding dimension {vectors.shape[1]} does not match provider "
                f"dimension {embedder.dimension}"
            )

        store.replace_chunks([chunk.store_payload() for chunk in chunks])
        vectors_path = settings.index_dir / "vectors.npy"
        index.save(vectors_path)

        built_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "embedding_provider": embedder.name,
            "embedding_model": embedder.model,
            "embedding_dimension": int(vectors.shape[1]) if vectors.size else embedder.dimension,
            "chunk_count": len(chunks),
            "vector_backend": index.backend,
            "built_at": built_at,
        }
        for key, value in meta.items():
            store.set_meta(key, value)

        manifest = settings.index_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    **meta,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "resource_types": _resource_type_counts(chunks),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = IndexResult(
            chunk_count=len(chunks),
            dimension=meta["embedding_dimension"],
            provider=embedder.name,
            model=embedder.model,
            backend=index.backend,
            duration_ms=(time.perf_counter() - started) * 1000,
            paths={"vectors": vectors_path, "manifest": manifest, "database": store.path},
        )
        return result


def _resource_type_counts(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.resource_type] = counts.get(chunk.resource_type, 0) + 1
    return dict(sorted(counts.items()))


def load_index(
    settings: Settings,
    store: SqliteStore,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Load chunks, vectors and metadata, validating that they still align."""
    chunks = store.get_chunks()
    vectors_path = settings.index_dir / "vectors.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vector index not found at {vectors_path}")
    vectors = np.load(vectors_path)
    meta = {
        key: store.get_meta(key)
        for key in (
            "embedding_provider",
            "embedding_model",
            "embedding_dimension",
            "chunk_count",
            "vector_backend",
            "built_at",
        )
    }
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector mismatch: {len(chunks)} chunks but {len(vectors)} vectors"
        )
    return chunks, vectors, meta
