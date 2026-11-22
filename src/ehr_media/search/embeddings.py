"""Embedding providers with a deterministic offline implementation.

``sentence-transformers`` is used when available; otherwise a signed
feature-hashing embedder keeps search fully functional without a model
download. Both return L2-normalized float32 vectors so cosine similarity is a
plain dot product.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from ..config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "patient",
    "clinical",
    "note",
    "result",
    "results",
    "reported",
    "documented",
    "history",
}


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    name: str = "embedding"
    model: str = "unknown"
    dimension: int = 0

    @abstractmethod
    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbeddingProvider(EmbeddingProvider):
    """Signed feature hashing over word unigrams, bigrams and char trigrams."""

    name = "hashing"
    model = "hashing-tf-v1"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = max(64, dimension)

    def _tokens(self, text: str) -> list[tuple[str, float]]:
        words = [
            token
            for token in _TOKEN_RE.findall(text.lower())
            if token not in _STOPWORDS and len(token) > 1
        ]
        features: list[tuple[str, float]] = [(word, 1.0) for word in words]
        features.extend(
            (f"{left} {right}", 0.55)
            for left, right in zip(words, words[1:])
        )
        for word in words:
            if len(word) >= 6:
                padded = f"#{word}#"
                features.extend(
                    (padded[index : index + 3], 0.25)
                    for index in range(len(padded) - 2)
                )
        return features

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature, weight in self._tokens(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            bucket = number % self.dimension
            sign = 1.0 if (number >> 63) & 1 else -1.0
            vector[bucket] += sign * weight
        return vector

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalize(np.vstack([self._encode_one(text) for text in texts]))


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EmbeddingProviderError(
                "sentence-transformers is not installed"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # model download / cache failure
            raise EmbeddingProviderError(
                f"Could not load sentence-transformer model {model_name!r}: {exc}"
            ) from exc
        self.model = model_name
        dimension_getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        self.dimension = int(
            dimension_getter()
            if callable(dimension_getter)
            else self._model.encode(["dimension probe"]).shape[1]
        )

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalize(np.asarray(vectors))


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"
    model = "text-embedding-3-small"
    URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise EmbeddingProviderError("OPENAI_API_KEY is not set")
        self.api_key = settings.openai_api_key.get_secret_value()
        self.timeout = settings.llm_timeout_seconds
        self.dimension = 1536

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        batch_size = max(1, min(batch_size, 100))
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    self.URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                if response.status_code >= 400:
                    raise EmbeddingProviderError(
                        f"OpenAI embeddings failed (HTTP {response.status_code}): "
                        f"{response.text[:300]}"
                    )
                payload = response.json()
                ordered = sorted(payload.get("data", []), key=lambda item: item["index"])
                if len(ordered) != len(batch):
                    raise EmbeddingProviderError("OpenAI embeddings returned a partial batch")
                vectors.extend(item["embedding"] for item in ordered)
        self.dimension = len(vectors[0]) if vectors else self.dimension
        return _normalize(np.asarray(vectors, dtype=np.float32))


def resolve_embedding_provider(
    settings: Settings, requested: str | None = None
) -> EmbeddingProvider:
    choice = (requested or settings.embedding_provider or "auto").lower()
    if choice == "hashing":
        return HashingEmbeddingProvider(settings.embedding_dim)
    if choice == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    if choice == "openai":
        return OpenAIEmbeddingProvider(settings)
    if choice != "auto":
        raise EmbeddingProviderError(f"Unknown embedding provider {choice!r}")
    try:
        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    except EmbeddingProviderError:
        provider = HashingEmbeddingProvider(settings.embedding_dim)
        provider.name = "hashing (auto fallback)"
        return provider
