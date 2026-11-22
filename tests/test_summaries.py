"""Task 3 tests: evidence extraction, summaries, caching and fallbacks."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ehr_media.config import Settings
from ehr_media.fhir import SqliteStore
from ehr_media.ids import (
    document_id_for,
    encounter_id_for,
    observation_id_for,
    patient_id_for_mrn,
)
from ehr_media.models import ClinicalDocument, Encounter, LabResult, PatientRecord
from ehr_media.summaries import (
    ClinicalSummarizer,
    OfflineSummaryProvider,
    ProviderResult,
    SummaryProvider,
    SummaryProviderError,
    SummarySections,
    compose_summary,
    extract_evidence,
    parse_summary_json,
    record_hash_for,
    resolve_provider,
)
from ehr_media.summaries.providers import build_prompt


def build_patient() -> PatientRecord:
    mrn = "00888888"
    patient_id = patient_id_for_mrn(mrn)
    encounter = Encounter(
        id=encounter_id_for(patient_id, "AMB|2026-04-01|Chest pain"),
        patient_id=patient_id,
        class_code="AMB",
        reason="Chest pain",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 1),
        source="json",
    )
    document = ClinicalDocument(
        id=document_id_for(patient_id, "note-hash"),
        patient_id=patient_id,
        doc_type="progress_note",
        title="Cardiology progress note",
        text=(
            "Chief Complaint: chest pain and palpitations.\n"
            "Assessment: atrial fibrillation, hypertension.\n"
            "Plan: start apixaban, cardiology referral."
        ),
        authored_on=date(2026, 4, 1),
        author="Dr. Ada Lovelace",
        encounter_id=encounter.id,
        source="json",
        source_format="json",
        content_hash="note-hash",
    )
    labs = [
        LabResult(
            id=observation_id_for(patient_id, "potassium-hash"),
            patient_id=patient_id,
            test_name="Potassium",
            loinc_code="2823-3",
            value=6.4,
            unit="mmol/L",
            reference_range="3.5-5.1",
            reference_low=3.5,
            reference_high=5.1,
            interpretation="critical",
            collected_at=date(2026, 4, 1),
            status="final",
            source="csv",
            content_hash="potassium-hash",
        ),
        LabResult(
            id=observation_id_for(patient_id, "creatinine-hash"),
            patient_id=patient_id,
            test_name="Creatinine",
            loinc_code="2160-0",
            value=0.9,
            unit="mg/dL",
            reference_range="0.6-1.2",
            reference_low=0.6,
            reference_high=1.2,
            interpretation="normal",
            collected_at=date(2026, 4, 1),
            status="final",
            source="csv",
            content_hash="creatinine-hash",
        ),
    ]
    return PatientRecord(
        id=patient_id,
        mrn=mrn,
        given_name="Grace",
        family_name="Hopper",
        birth_date=date(1955, 12, 9),
        gender="female",
        documents=[document],
        lab_results=labs,
        encounters=[encounter],
        ingested_at=datetime(2026, 11, 22, tzinfo=timezone.utc),
    )


class CountingProvider(SummaryProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, evidence) -> ProviderResult:
        self.calls += 1
        return ProviderResult(
            sections=SummarySections(
                chief_concern="Chest pain with palpitations.",
                diagnoses=["Atrial fibrillation", "Hypertension"],
                recent_media=["Progress note (2026-04-01)"],
                flagged_anomalies=["Potassium 6.4 mmol/L (critical)"],
            ),
            confidence=0.81,
            provider=self.name,
            model=self.model,
        )


class FailingProvider(SummaryProvider):
    name = "broken"
    model = "broken-1"

    def summarize(self, evidence) -> ProviderResult:
        raise SummaryProviderError("simulated provider outage")


class VerboseProvider(SummaryProvider):
    name = "verbose"
    model = "verbose-1"

    def summarize(self, evidence) -> ProviderResult:
        return ProviderResult(
            sections=SummarySections(
                chief_concern=" ".join(["dyspnea"] * 300),
                diagnoses=[" ".join(["diagnosis"] * 300)],
                recent_media=[" ".join(["media"] * 300)],
                flagged_anomalies=[" ".join(["anomaly"] * 300)],
            ),
            confidence=0.7,
            provider=self.name,
            model=self.model,
        )


@pytest.fixture
def patient() -> PatientRecord:
    return build_patient()


@pytest.fixture
def offline_settings(settings: Settings) -> Settings:
    settings.anthropic_api_key = None
    settings.openai_api_key = None
    settings.llm_provider = "auto"
    return settings


def test_offline_summary_covers_required_sections(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    summary = ClinicalSummarizer(offline_settings, persist=False).summarize(patient)

    assert summary.provider == "offline"
    assert summary.model == "rule-based-v1"
    assert summary.cached is False
    assert 20 <= summary.word_count <= 200
    assert summary.word_count == len(summary.summary.split())
    assert "Chief concern:" in summary.summary
    assert "Diagnoses:" in summary.summary
    assert "Recent media:" in summary.summary
    assert "Flagged anomalies:" in summary.summary

    assert "chest pain" in summary.sections.chief_concern.lower()
    diagnoses = " | ".join(summary.sections.diagnoses).lower()
    assert "atrial fibrillation" in diagnoses
    assert "hypertension" in diagnoses
    assert summary.sections.recent_media
    assert any("potassium" in item.lower() for item in summary.sections.flagged_anomalies)
    assert 0.0 <= summary.confidence <= 1.0
    assert summary.confidence_label in {"low", "moderate", "high"}
    assert "clinical decision support" in summary.disclaimer.lower()


def test_evidence_extraction_is_structured(patient: PatientRecord) -> None:
    evidence = extract_evidence(patient)

    assert evidence.patient_id == patient.id
    assert evidence.mrn == "00888888"
    assert evidence.age == patient.age_at()
    assert evidence.document_count == 1
    assert evidence.lab_count == 2
    assert evidence.encounter_count == 1
    assert evidence.latest_date == date(2026, 4, 1)
    assert len(evidence.record_hash) == 32
    assert evidence.recent_media[0].kind == "Progress note"
    assert any(item.kind == "Lab panel" for item in evidence.recent_media)


def test_summary_cache_hits_and_force_regenerates(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    provider = CountingProvider()
    summarizer = ClinicalSummarizer(
        offline_settings, provider=provider, persist=True
    )

    first = summarizer.summarize(patient)
    assert first.cached is False
    assert provider.calls == 1

    second = summarizer.summarize(patient)
    assert second.cached is True
    assert second.summary == first.summary
    assert provider.calls == 1

    forced = summarizer.summarize(patient, force=True)
    assert forced.cached is False
    assert provider.calls == 2

    store = SqliteStore(offline_settings.db_path)
    cached = store.get_cached_summary(patient.id, first.record_hash)
    assert cached is not None
    assert cached["summary"] == first.summary
    assert store.get_summary(patient.id)["record_hash"] == first.record_hash


def test_record_hash_changes_when_clinical_content_changes(patient: PatientRecord) -> None:
    before = record_hash_for(patient)
    patient.lab_results[0].value = 7.2
    after = record_hash_for(patient)

    assert before != after


def test_cache_is_invalidated_by_new_record_version(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    provider = CountingProvider()
    summarizer = ClinicalSummarizer(
        offline_settings, provider=provider, persist=True
    )

    summarizer.summarize(patient)
    patient.lab_results[0].value = 7.2
    updated = summarizer.summarize(patient)

    assert provider.calls == 2
    assert updated.cached is False
    assert updated.record_hash == record_hash_for(patient)


def test_provider_failure_falls_back_to_offline(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    summarizer = ClinicalSummarizer(
        offline_settings, provider=FailingProvider(), persist=False
    )
    summary = summarizer.summarize(patient)

    assert summary.provider == "offline"
    assert summary.fallback is True
    assert summary.error is not None and "simulated provider outage" in summary.error
    assert summary.word_count <= 200


def test_verbose_provider_output_is_trimmed(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    summarizer = ClinicalSummarizer(
        offline_settings, provider=VerboseProvider(), persist=False
    )
    summary = summarizer.summarize(patient)

    assert summary.word_count <= offline_settings.max_summary_words
    assert summary.word_count <= 200
    assert summary.summary.endswith(".")


def test_compose_summary_enforces_limit() -> None:
    sections = SummarySections(
        chief_concern=" ".join(["word"] * 500),
        diagnoses=[" ".join(["dx"] * 500)],
        recent_media=[" ".join(["media"] * 500)],
        flagged_anomalies=[" ".join(["flag"] * 500)],
    )
    narrative = compose_summary(sections, max_words=120)

    assert len(narrative.split()) <= 120


def test_parse_summary_json_handles_code_fences_and_percentages() -> None:
    raw = """```json
    {"chief_concern": "Chest pain", "diagnoses": ["AFib", "HTN"],
     "recent_media": "Progress note; Lab panel",
     "flagged_anomalies": ["Hyperkalemia"], "confidence": 82}
    ```"""
    sections, confidence = parse_summary_json(raw)

    assert sections.chief_concern == "Chest pain"
    assert sections.diagnoses == ["AFib", "HTN"]
    assert sections.recent_media == ["Progress note", "Lab panel"]
    assert sections.flagged_anomalies == ["Hyperkalemia"]
    assert confidence == 0.82


def test_parse_summary_json_rejects_non_json() -> None:
    with pytest.raises(SummaryProviderError):
        parse_summary_json("I could not summarize this record.")


def test_provider_resolution(offline_settings: Settings) -> None:
    assert isinstance(resolve_provider(offline_settings), OfflineSummaryProvider)
    with pytest.raises(SummaryProviderError):
        resolve_provider(offline_settings, "anthropic")
    with pytest.raises(SummaryProviderError):
        resolve_provider(offline_settings, "unknown-provider")


def test_prompt_contains_evidence_and_guardrails(patient: PatientRecord) -> None:
    prompt = build_prompt(extract_evidence(patient))

    assert "synthetic patient record" in prompt
    assert "chest pain" in prompt.lower()
    assert "Do not add facts" in prompt
    assert "chief_concern" in prompt


def test_batch_writes_artifacts_and_manifest(
    offline_settings: Settings, patient: PatientRecord
) -> None:
    other = build_patient()
    other.mrn = "00888889"
    other.id = patient_id_for_mrn(other.mrn)
    provider = CountingProvider()
    summarizer = ClinicalSummarizer(
        offline_settings, provider=provider, persist=True
    )

    result = summarizer.summarize_all([patient, other], write_artifacts=True)

    assert result.count == 2
    assert result.generated == 2
    assert result.cached == 0
    assert result.providers == {"fake": 2}
    assert result.duration_ms >= 0
    assert result.by_patient(patient.id) is not None

    summary_path = offline_settings.summaries_dir / f"{patient.id}.json"
    manifest_path = offline_settings.summaries_dir / "manifest.json"
    assert summary_path.exists()
    assert manifest_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["patient_id"] == patient.id
    assert payload["word_count"] <= 200
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == 2
    assert manifest["providers"] == {"fake": 2}

    again = summarizer.summarize_all([patient], write_artifacts=False)
    assert again.cached == 1
