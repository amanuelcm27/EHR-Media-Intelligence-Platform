"""Deterministic, FHIR-safe identifier helpers.

FHIR ``id`` values must match ``[A-Za-z0-9\\-\\.]{1,64}``. We derive IDs from
stable business keys with UUIDv5 so that re-running the pipeline never
produces duplicate resources and references stay stable across runs.
"""

from __future__ import annotations

import hashlib
import re
import uuid

_NAMESPACE = uuid.UUID("6f1c4f0a-6a1c-4d4b-9c0e-2f7a5f3d9b11")

_INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9\-.]")


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    """Return ``<prefix>-<hex>`` deterministically derived from ``parts``."""
    key = "|".join(str(part) for part in parts)
    digest = uuid.uuid5(_NAMESPACE, key).hex[:length]
    safe_prefix = _INVALID_ID_CHARS.sub("", prefix) or "res"
    return f"{safe_prefix}-{digest}"


def patient_id_for_mrn(mrn: str) -> str:
    return stable_id("pat", mrn, length=16)


def encounter_id_for(patient_id: str, key: str) -> str:
    return stable_id("enc", patient_id, key, length=16)


def document_id_for(patient_id: str, content_hash: str) -> str:
    return stable_id("doc", patient_id, content_hash, length=16)


def observation_id_for(patient_id: str, lab_hash: str) -> str:
    return stable_id("obs", patient_id, lab_hash, length=16)


def report_id_for(patient_id: str, report_key: str) -> str:
    return stable_id("dr", patient_id, report_key, length=16)


def bundle_id_for(patient_id: str) -> str:
    return stable_id("bundle", patient_id, length=16)


def content_hash(*parts: object, length: int = 32) -> str:
    """Stable SHA-256 based hash used for de-duplication and cache keys."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def mrn_display(mrn: str) -> str:
    """Human-readable MRN, e.g. ``MRN-00100455``."""
    return f"MRN-{mrn}" if mrn.isdigit() else f"MRN-{mrn.upper()}"
