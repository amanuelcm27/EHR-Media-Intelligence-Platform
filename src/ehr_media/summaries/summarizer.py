"""Clinical summary service with record-hash caching and offline fallback."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..config import Settings, get_settings
from ..fhir.store import SqliteStore
from ..models import PatientRecord
from .evidence import extract_evidence
from .models import (
    ClinicalSummary,
    ProviderResult,
    SummaryBatchResult,
    SummarySections,
)
from .providers import (
    OfflineSummaryProvider,
    SummaryProvider,
    SummaryProviderError,
    resolve_provider,
)

MAX_WORDS = 200


def confidence_label(confidence: float) -> str:
    if confidence < 0.5:
        return "low"
    if confidence < 0.75:
        return "moderate"
    return "high"


def _tidy(value: str) -> str:
    return " ".join(value.split()).strip(" .;-")


def _limit_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    if limit <= 0:
        return "…"
    return " ".join(words[:limit]).rstrip(" ,;:.") + "…"


def compose_summary(
    sections: SummarySections,
    *,
    max_words: int = MAX_WORDS,
) -> str:
    """Render sections as a compact narrative and enforce the word limit."""
    diagnoses = [_tidy(item) for item in sections.diagnoses if _tidy(item)]
    media = [_tidy(item) for item in sections.recent_media if _tidy(item)]
    anomalies = [_tidy(item) for item in sections.flagged_anomalies if _tidy(item)]

    chunks = [
        f"Chief concern: {_tidy(sections.chief_concern) or 'Not documented'}.",
        "Diagnoses: " + ("; ".join(diagnoses) if diagnoses else "None documented") + ".",
        "Recent media: " + ("; ".join(media) if media else "None available") + ".",
        "Flagged anomalies: "
        + ("; ".join(anomalies) if anomalies else "None flagged")
        + ".",
    ]
    narrative = " ".join(chunk for chunk in chunks if chunk)
    if len(narrative.split()) <= max_words:
        return narrative

    # Allocate a budget per section so one long list cannot crowd out the rest.
    budgets = (45, 55, 50, 40)
    trimmed = [
        f"Chief concern: {_limit_words(_tidy(sections.chief_concern), budgets[0])}.",
        f"Diagnoses: {_limit_words('; '.join(diagnoses), budgets[1])}.",
        f"Recent media: {_limit_words('; '.join(media), budgets[2])}.",
        f"Flagged anomalies: {_limit_words('; '.join(anomalies), budgets[3])}.",
    ]
    narrative = " ".join(trimmed)
    return _limit_words(narrative, max_words)


class ClinicalSummarizer:
    """Generates, caches and persists one summary per patient record version."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SqliteStore | None = None,
        provider: SummaryProvider | None = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.persist = persist
        self.provider_error: str | None = None
        if provider is not None:
            self.provider = provider
        else:
            try:
                self.provider = resolve_provider(self.settings)
            except SummaryProviderError as exc:
                self.provider = OfflineSummaryProvider()
                self.provider_error = str(exc)
        self._memory_cache: dict[tuple[str, str], ClinicalSummary] = {}

    # ------------------------------------------------------------------
    def summarize(
        self,
        patient: PatientRecord,
        *,
        force: bool = False,
    ) -> ClinicalSummary:
        started = time.perf_counter()
        evidence = extract_evidence(patient)

        if not force:
            cached = self._get_cached(patient.id, evidence.record_hash)
            if cached is not None:
                cached.cached = True
                return cached

        fallback = False
        error: str | None = None
        try:
            result = self.provider.summarize(evidence)
        except Exception as exc:  # provider outage, bad key, invalid JSON
            fallback = True
            error = f"{type(exc).__name__}: {exc}"
            result = OfflineSummaryProvider().summarize(evidence)
            result.fallback = True
            result.error = error

        if self.provider_error and result.provider == "offline":
            fallback = True
            error = error or self.provider_error
            result.fallback = True
            result.error = error

        narrative = compose_summary(
            result.sections, max_words=self.settings.max_summary_words
        )
        summary = ClinicalSummary(
            patient_id=patient.id,
            patient_name=patient.display_name,
            mrn=patient.mrn,
            record_hash=evidence.record_hash,
            provider=result.provider,
            model=result.model,
            sections=result.sections,
            summary=narrative,
            word_count=len(narrative.split()),
            confidence=result.confidence,
            confidence_label=confidence_label(result.confidence),  # type: ignore[arg-type]
            cached=False,
            fallback=fallback or result.fallback,
            error=error or result.error,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        self._save(summary)
        return summary

    def summarize_all(
        self,
        patients: Iterable[PatientRecord],
        *,
        force: bool = False,
        write_artifacts: bool = True,
    ) -> SummaryBatchResult:
        started = time.perf_counter()
        result = SummaryBatchResult()
        for patient in patients:
            summary = self.summarize(patient, force=force)
            result.summaries.append(summary)
            if summary.cached:
                result.cached += 1
            else:
                result.generated += 1
            if summary.fallback:
                result.fallbacks += 1
            result.providers[summary.provider] = result.providers.get(summary.provider, 0) + 1

        if write_artifacts:
            self.write_artifacts(result)
        result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    # ------------------------------------------------------------------
    def write_artifacts(self, result: SummaryBatchResult) -> dict[str, Path]:
        self.settings.ensure_dirs()
        paths: dict[str, Path] = {}
        for summary in result.summaries:
            path = self.settings.summaries_dir / f"{summary.patient_id}.json"
            path.write_text(
                json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            paths[summary.patient_id] = path

        manifest = self.settings.summaries_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "count": result.count,
                    "cached": result.cached,
                    "generated": result.generated,
                    "fallbacks": result.fallbacks,
                    "providers": result.providers,
                    "summaries": [
                        {
                            "patient_id": summary.patient_id,
                            "mrn": summary.mrn,
                            "patient_name": summary.patient_name,
                            "provider": summary.provider,
                            "model": summary.model,
                            "confidence": summary.confidence,
                            "confidence_label": summary.confidence_label,
                            "word_count": summary.word_count,
                            "cached": summary.cached,
                            "fallback": summary.fallback,
                            "file": f"{summary.patient_id}.json",
                        }
                        for summary in result.summaries
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["manifest"] = manifest
        return paths

    # ------------------------------------------------------------------
    def _get_cached(self, patient_id: str, record_hash: str) -> ClinicalSummary | None:
        memory = self._memory_cache.get((patient_id, record_hash))
        if memory is not None:
            return memory.model_copy()
        if self.store is None and not self.persist:
            return None
        store = self._ensure_store()
        cached = store.get_cached_summary(patient_id, record_hash)
        if cached is None:
            return None
        summary = ClinicalSummary.model_validate(cached)
        self._memory_cache[(patient_id, record_hash)] = summary
        return summary.model_copy()

    def _save(self, summary: ClinicalSummary) -> None:
        self._memory_cache[(summary.patient_id, summary.record_hash)] = summary.model_copy()
        if not self.persist and self.store is None:
            return
        store = self._ensure_store()
        payload = summary.model_dump(mode="json")
        store.save_summary(payload)
        store.save_cached_summary(payload)

    def _ensure_store(self) -> SqliteStore:
        if self.store is None:
            self.store = SqliteStore(self.settings.db_path)
            self.store.initialize()
        return self.store


def summarize_patient(
    patient: PatientRecord,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> ClinicalSummary:
    """Convenience wrapper for scripts and tests."""
    return ClinicalSummarizer(settings).summarize(patient, force=force)
