"""SQLite persistence for cleaned patients, FHIR Bundles, summaries and chunks.

The database is the single source of truth for the API. Files under
``artifacts/fhir`` and ``artifacts/summaries`` are human-inspectable exports;
SQLite keeps retrieval fast and transactional.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..models import PatientRecord

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    mrn TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    birth_date TEXT,
    gender TEXT,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bundles (
    bundle_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    valid INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bundles_patient ON bundles(patient_id);

CREATE TABLE IF NOT EXISTS resources (
    resource_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_json TEXT NOT NULL,
    effective_date TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resources_patient ON resources(patient_id);
CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type);
CREATE INDEX IF NOT EXISTS idx_resources_date ON resources(effective_date);

CREATE TABLE IF NOT EXISTS validation_reports (
    patient_id TEXT PRIMARY KEY,
    bundle_id TEXT NOT NULL,
    valid INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    patient_id TEXT PRIMARY KEY,
    record_hash TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_cache (
    patient_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (patient_id, record_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    mrn TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    title TEXT,
    effective_date TEXT,
    section TEXT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_patient ON chunks(patient_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(resource_type);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resource_date(resource: dict[str, Any]) -> str | None:
    for key in ("date", "effectiveDateTime", "issued"):
        value = resource.get(key)
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
    period = resource.get("period") or {}
    start = period.get("start")
    if isinstance(start, str) and len(start) >= 10:
        return start[:10]
    return None


class SqliteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # Patients / Bundles / resources
    # ------------------------------------------------------------------
    def save_bundle(
        self,
        patient: PatientRecord,
        bundle: dict[str, Any],
        validation: Any | None = None,
    ) -> None:
        now = _utcnow()
        patient_payload = patient.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO patients
                    (patient_id, mrn, display_name, birth_date, gender, data_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    mrn=excluded.mrn,
                    display_name=excluded.display_name,
                    birth_date=excluded.birth_date,
                    gender=excluded.gender,
                    data_json=excluded.data_json,
                    updated_at=excluded.updated_at
                """,
                (
                    patient.id,
                    patient.mrn,
                    patient.display_name,
                    patient.birth_date.isoformat() if patient.birth_date else None,
                    patient.gender,
                    json.dumps(patient_payload, ensure_ascii=False),
                    now,
                ),
            )
            valid = 1 if validation is not None and validation.valid else 0
            error_count = validation.error_count if validation is not None else 0
            warning_count = validation.warning_count if validation is not None else 0
            connection.execute(
                """
                INSERT INTO bundles
                    (bundle_id, patient_id, bundle_json, valid, error_count,
                     warning_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bundle_id) DO UPDATE SET
                    bundle_json=excluded.bundle_json,
                    valid=excluded.valid,
                    error_count=excluded.error_count,
                    warning_count=excluded.warning_count,
                    updated_at=excluded.updated_at
                """,
                (
                    bundle["id"],
                    patient.id,
                    json.dumps(bundle, ensure_ascii=False),
                    valid,
                    error_count,
                    warning_count,
                    now,
                    now,
                ),
            )
            if validation is not None:
                connection.execute(
                    """
                    INSERT INTO validation_reports
                        (patient_id, bundle_id, valid, report_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        bundle_id=excluded.bundle_id,
                        valid=excluded.valid,
                        report_json=excluded.report_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        patient.id,
                        validation.bundle_id,
                        valid,
                        validation.model_dump_json(),
                        now,
                    ),
                )
            connection.execute("DELETE FROM resources WHERE patient_id = ?", (patient.id,))
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                connection.execute(
                    """
                    INSERT OR REPLACE INTO resources
                        (resource_id, patient_id, resource_type, resource_json,
                         effective_date, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resource.get("id"),
                        patient.id,
                        resource.get("resourceType"),
                        json.dumps(resource, ensure_ascii=False),
                        _resource_date(resource),
                        now,
                    ),
                )

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def get_patient_by_mrn(self, mrn: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM patients WHERE mrn = ?", (mrn,)
            ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_patients(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT patient_id, mrn, display_name, birth_date, gender FROM patients "
                "ORDER BY display_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_bundle(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT bundle_json FROM bundles WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return json.loads(row["bundle_json"]) if row else None

    def list_bundles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT bundle_id, patient_id, bundle_json, valid, error_count, warning_count "
                "FROM bundles ORDER BY patient_id"
            ).fetchall()
        return [
            {
                "bundle_id": row["bundle_id"],
                "patient_id": row["patient_id"],
                "bundle": json.loads(row["bundle_json"]),
                "valid": bool(row["valid"]),
                "error_count": row["error_count"],
                "warning_count": row["warning_count"],
            }
            for row in rows
        ]

    def get_resources(self, patient_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT resource_json FROM resources WHERE patient_id = ? "
                "ORDER BY effective_date DESC, resource_type",
                (patient_id,),
            ).fetchall()
        return [json.loads(row["resource_json"]) for row in rows]

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT resource_json FROM resources WHERE resource_id = ?", (resource_id,)
            ).fetchone()
        return json.loads(row["resource_json"]) if row else None

    def get_validation(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM validation_reports WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
        return json.loads(row["report_json"]) if row else None

    def validation_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, SUM(valid) AS valid, "
                "SUM(error_count) AS errors, SUM(warning_count) AS warnings FROM bundles"
            ).fetchone()
        total = int(row["total"] or 0)
        valid = int(row["valid"] or 0)
        return {
            "bundles": total,
            "valid": valid,
            "invalid": total - valid,
            "errors": int(row["errors"] or 0),
            "warnings": int(row["warnings"] or 0),
        }

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def save_summary(self, summary: dict[str, Any]) -> None:
        now = _utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO summaries
                    (patient_id, record_hash, summary_json, provider, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    record_hash=excluded.record_hash,
                    summary_json=excluded.summary_json,
                    provider=excluded.provider,
                    model=excluded.model,
                    updated_at=excluded.updated_at
                """,
                (
                    summary["patient_id"],
                    summary["record_hash"],
                    json.dumps(summary, ensure_ascii=False),
                    summary.get("provider"),
                    summary.get("model"),
                    now,
                    now,
                ),
            )

    def get_summary(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM summaries WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def get_cached_summary(self, patient_id: str, record_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM summary_cache WHERE patient_id = ? AND record_hash = ?",
                (patient_id, record_hash),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def save_cached_summary(self, summary: dict[str, Any]) -> None:
        now = _utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO summary_cache
                    (patient_id, record_hash, summary_json, provider, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    summary["patient_id"],
                    summary["record_hash"],
                    json.dumps(summary, ensure_ascii=False),
                    summary.get("provider"),
                    summary.get("model"),
                    now,
                ),
            )

    # ------------------------------------------------------------------
    # Search chunks / index metadata
    # ------------------------------------------------------------------
    def replace_chunks(self, chunks: list[dict[str, Any]]) -> None:
        now = _utcnow()
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, patient_id, patient_name, mrn, resource_type, resource_id,
                     title, effective_date, section, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["patient_id"],
                        chunk["patient_name"],
                        chunk["mrn"],
                        chunk["resource_type"],
                        chunk["resource_id"],
                        chunk.get("title"),
                        chunk.get("effective_date"),
                        chunk.get("section"),
                        chunk["text"],
                        now,
                    )
                    for chunk in chunks
                ],
            )

    def get_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, patient_id, patient_name, mrn, resource_type, resource_id, "
                "title, effective_date, section, text FROM chunks ORDER BY chunk_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def chunk_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    def set_meta(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO index_meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), _utcnow()),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM index_meta WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            patients = connection.execute("SELECT COUNT(*) AS count FROM patients").fetchone()[
                "count"
            ]
            bundles = connection.execute("SELECT COUNT(*) AS count FROM bundles").fetchone()[
                "count"
            ]
            summaries = connection.execute("SELECT COUNT(*) AS count FROM summaries").fetchone()[
                "count"
            ]
            chunks = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            resources = connection.execute(
                "SELECT resource_type, COUNT(*) AS count FROM resources GROUP BY resource_type"
            ).fetchall()
            validation = connection.execute(
                "SELECT COUNT(*) AS total, SUM(valid) AS valid FROM bundles"
            ).fetchone()
        return {
            "patients": int(patients),
            "bundles": int(bundles),
            "valid_bundles": int(validation["valid"] or 0),
            "invalid_bundles": int(validation["total"] or 0) - int(validation["valid"] or 0),
            "summaries": int(summaries),
            "chunks": int(chunks),
            "resources": {
                row["resource_type"]: int(row["count"])
                for row in sorted(resources, key=lambda item: item["resource_type"])
            },
            "database": str(self.path),
        }
