"""Task 4: embeddings, vector index and semantic search."""

from .chunks import Chunk, build_chunks
from .embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    resolve_embedding_provider,
)
from .index import IndexResult, SearchIndexer, VectorIndex, load_index
from .service import (
    SearchFilters,
    SearchIndexUnavailable,
    SearchResponse,
    SearchResult,
    SearchService,
    VALID_RESOURCE_TYPES,
)

__all__ = [
    "Chunk",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "HashingEmbeddingProvider",
    "IndexResult",
    "OpenAIEmbeddingProvider",
    "SearchFilters",
    "SearchIndexUnavailable",
    "SearchIndexer",
    "SearchResponse",
    "SearchResult",
    "SearchService",
    "SentenceTransformerEmbeddingProvider",
    "VALID_RESOURCE_TYPES",
    "VectorIndex",
    "build_chunks",
    "load_index",
    "resolve_embedding_provider",
]
