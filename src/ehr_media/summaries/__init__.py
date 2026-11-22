"""Task 3: AI clinical summaries with caching and offline fallback."""

from .evidence import extract_evidence, record_hash_for
from .models import (
    DISCLAIMER,
    ClinicalSummary,
    MediaItem,
    ProviderResult,
    SummaryBatchResult,
    SummaryEvidence,
    SummarySections,
)
from .providers import (
    AnthropicSummaryProvider,
    OfflineSummaryProvider,
    OpenAISummaryProvider,
    SummaryProvider,
    SummaryProviderError,
    parse_summary_json,
    resolve_provider,
)
from .summarizer import ClinicalSummarizer, compose_summary, summarize_patient

__all__ = [
    "DISCLAIMER",
    "AnthropicSummaryProvider",
    "ClinicalSummarizer",
    "ClinicalSummary",
    "MediaItem",
    "OfflineSummaryProvider",
    "OpenAISummaryProvider",
    "ProviderResult",
    "SummaryBatchResult",
    "SummaryEvidence",
    "SummaryProvider",
    "SummaryProviderError",
    "SummarySections",
    "compose_summary",
    "extract_evidence",
    "parse_summary_json",
    "record_hash_for",
    "resolve_provider",
    "summarize_patient",
]
