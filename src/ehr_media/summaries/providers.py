"""Summary providers: Anthropic, OpenAI and a deterministic offline fallback.

The offline provider is a first-class implementation, not a stub. It makes the
demo and test suite runnable without API keys while still producing every
required section (chief concern, diagnoses, recent media, flagged anomalies).
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import Settings
from .evidence import SummaryEvidence
from .models import ProviderResult, SummarySections

SYSTEM_PROMPT = (
    "You are a clinical summarization assistant. You receive structured evidence "
    "extracted from one synthetic patient record. Use only that evidence; never "
    "invent diagnoses, dates, medications, or results. Write for a busy clinician. "
    "Return one JSON object with exactly these keys: chief_concern (string), "
    "diagnoses (array of strings), recent_media (array of strings), "
    "flagged_anomalies (array of strings), confidence (number from 0 to 1 reflecting "
    "how complete the evidence is). Keep all text combined under 170 words."
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_VERSION = "2023-06-01"


class SummaryProviderError(RuntimeError):
    """Raised when a provider cannot produce a usable summary."""


class SummaryProvider(ABC):
    name: str = "provider"
    model: str = "unknown"

    @abstractmethod
    def summarize(self, evidence: SummaryEvidence) -> ProviderResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Offline / deterministic provider
# ---------------------------------------------------------------------------


class OfflineSummaryProvider(SummaryProvider):
    name = "offline"
    model = "rule-based-v1"

    def summarize(self, evidence: SummaryEvidence) -> ProviderResult:
        diagnoses = evidence.diagnoses or ["No diagnosis explicitly documented"]
        recent_media = [
            f"{item.kind}: {item.title}"
            + (f" ({item.date.isoformat()})" if item.date else "")
            for item in evidence.recent_media
        ] or ["No clinical media available"]
        anomalies = evidence.flagged_anomalies or ["No flagged anomalies"]
        return ProviderResult(
            sections=SummarySections(
                chief_concern=evidence.chief_concern,
                diagnoses=diagnoses,
                recent_media=recent_media,
                flagged_anomalies=anomalies,
            ),
            confidence=self._confidence(evidence),
            provider=self.name,
            model=self.model,
        )

    @staticmethod
    def _confidence(evidence: SummaryEvidence) -> float:
        score = 0.42
        if evidence.document_count:
            score += 0.12
        if evidence.document_count >= 2:
            score += 0.06
        if evidence.lab_count:
            score += 0.10
        if evidence.lab_count >= 4:
            score += 0.05
        if evidence.diagnoses:
            score += 0.10
        if evidence.flagged_anomalies:
            score += 0.05
        if evidence.latest_date is not None:
            score += 0.04
        if evidence.age is not None:
            score += 0.02
        if not evidence.document_count:
            score = min(score, 0.40)
        return round(min(score, 0.88), 2)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def build_prompt(evidence: SummaryEvidence) -> str:
    evidence_json = json.dumps(evidence.model_dump(mode="json"), indent=2, default=str)
    return (
        "Summarize this synthetic patient record for a clinician. Cover the chief "
        "concern, diagnoses, recent clinical media, and flagged anomalies. "
        "Do not add facts that are not in the evidence.\n\n"
        f"EVIDENCE:\n{evidence_json}\n\n"
        "Return only the JSON object described in the system prompt."
    )


class _HttpSummaryProvider(SummaryProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout = settings.llm_timeout_seconds
        self.max_retries = max(0, settings.llm_max_retries)

    def _post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code >= 500:
                    raise SummaryProviderError(
                        f"{self.name} returned HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                if response.status_code >= 400:
                    raise SummaryProviderError(
                        f"{self.name} rejected the request (HTTP {response.status_code}): "
                        f"{response.text[:300]}"
                    )
                return response.json()
            except (httpx.HTTPError, ValueError, SummaryProviderError) as exc:
                last_error = exc
                if isinstance(exc, SummaryProviderError) and "HTTP 4" in str(exc):
                    break
                if attempt < self.max_retries:
                    time.sleep(min(1.5 * (attempt + 1), 3.0))
        raise SummaryProviderError(str(last_error or f"{self.name} request failed"))

    def summarize(self, evidence: SummaryEvidence) -> ProviderResult:
        raw = self._complete(build_prompt(evidence))
        sections, confidence = parse_summary_json(raw)
        return ProviderResult(
            sections=sections,
            confidence=confidence,
            provider=self.name,
            model=self.model,
            raw_response=raw,
        )

    @abstractmethod
    def _complete(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicSummaryProvider(_HttpSummaryProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.anthropic_api_key is None:
            raise SummaryProviderError("ANTHROPIC_API_KEY / EHR_ANTHROPIC_API_KEY is not set")
        self.api_key = settings.anthropic_api_key.get_secret_value()
        self.model = settings.anthropic_model

    def _complete(self, prompt: str) -> str:
        data = self._post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            payload={
                "model": self.model,
                "max_tokens": 900,
                "temperature": 0.2,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise SummaryProviderError("anthropic returned an empty response")
        return text


class OpenAISummaryProvider(_HttpSummaryProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.openai_api_key is None:
            raise SummaryProviderError("OPENAI_API_KEY / EHR_OPENAI_API_KEY is not set")
        self.api_key = settings.openai_api_key.get_secret_value()
        self.model = settings.openai_model

    def _complete(self, prompt: str) -> str:
        data = self._post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            payload={
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 900,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        choices = data.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        if not text.strip():
            raise SummaryProviderError("openai returned an empty response")
        return text


# ---------------------------------------------------------------------------
# Parsing / selection
# ---------------------------------------------------------------------------


def parse_summary_json(raw: str) -> tuple[SummarySections, float]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SummaryProviderError("provider response did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SummaryProviderError(f"provider returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SummaryProviderError("provider JSON must be an object")

    chief = payload.get("chief_concern") or payload.get("chiefConcern") or ""
    sections = SummarySections(
        chief_concern=_as_text(chief) or "No chief concern documented",
        diagnoses=_as_list(payload.get("diagnoses")),
        recent_media=_as_list(payload.get("recent_media") or payload.get("recentMedia")),
        flagged_anomalies=_as_list(
            payload.get("flagged_anomalies") or payload.get("flaggedAnomalies")
        ),
    )
    confidence = _as_confidence(payload.get("confidence"))
    return sections, confidence


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;]+", value)
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = [value]
    return [str(item).strip(" .;-") for item in parts if str(item).strip()]


def _as_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.6
    if number > 1.0:  # some models return percentages
        number /= 100.0
    return round(min(max(number, 0.05), 0.98), 2)


def resolve_provider(settings: Settings, requested: str | None = None) -> SummaryProvider:
    """Resolve ``auto|anthropic|openai|offline`` into a provider instance."""
    choice = (requested or settings.llm_provider or "auto").lower()
    if choice == "offline":
        return OfflineSummaryProvider()
    if choice == "anthropic":
        return AnthropicSummaryProvider(settings)
    if choice == "openai":
        return OpenAISummaryProvider(settings)
    if choice != "auto":
        raise SummaryProviderError(f"Unknown summary provider {choice!r}")
    if settings.anthropic_api_key is not None:
        return AnthropicSummaryProvider(settings)
    if settings.openai_api_key is not None:
        return OpenAISummaryProvider(settings)
    return OfflineSummaryProvider()
