# EHR Media Intelligence Platform — Write-up

## Problem

Clinicians need one place to search heterogeneous EHR media — notes, labs, reports, encounters and AI summaries — without losing provenance or trust. The assessment asked for a complete pipeline: messy ingestion, FHIR R4 normalization, cached LLM summaries, fast semantic search and an accessible clinician UI.

I built an offline-first Python/FastAPI platform with a Tailwind frontend. Everything runs on synthetic data and can operate without external API keys.

## Approach

**1. Ingestion and cleaning.** JSON, CSV and free-text note loaders attach source and row provenance to every record. A Pydantic pipeline normalizes mixed date formats, gender values, MRNs and contact fields; removes duplicate labs/documents; and flags possible duplicate patients and conflicting identifiers instead of silently picking a winner. Each record receives an audit entry and data-quality flags. The demo ingests 97 raw records into 51 patients, 69 documents, 175 labs and 51 encounters, with 1 rejected row, 2 conflicts and 221 audit entries. Tests cover ambiguous dates, missing fields, duplicates, conflicts and malformed rows.

**2. FHIR R4.** Each patient becomes a Bundle containing `Patient`, `Encounter`, `DocumentReference`, `Observation` and `DiagnosticReport`. Resources are parsed with `fhir.resources` R4B models and then checked against the official HL7 R4 JSON Schema, followed by reference-integrity and clinical business rules. Validation is persisted and exported as JSON/Markdown. All 51 bundles validate with zero errors or warnings.

**3. Summaries.** An evidence extractor pulls chief concern, diagnoses, recent media and flagged anomalies. Anthropic, OpenAI and a deterministic offline provider share one interface; `auto` uses a key if present, otherwise offline. Output is capped at 200 words and includes confidence, a disclaimer and provider metadata. The cache key is patient ID plus a SHA-256 hash of the normalized record, persisted in SQLite, so unchanged records never re-call the LLM.

**4. Search.** Records are chunked into 454 passages, including patient demographics and AI summaries. Embeddings come from `all-MiniLM-L6-v2`, a signed-hashing fallback or OpenAI; vectors live in FAISS when available, otherwise NumPy. `POST /search` returns the top five with scores, snippets, AI snippets, confidence and patient context, and supports resource-type, date, patient and score filters. A small lexical boost helps exact drug/lab terms. Warm queries take roughly 12–40 ms server-side, and a test asserts 50 records complete under two seconds.

**5. UI.** The Tailwind interface provides 250 ms debounced live search, URL state, type chips and date filters. Cards show name/MRN, date, type badge, match score, source snippet and AI summary. A patient drawer exposes summary, documents and FHIR resources with validation status, quality flags and expandable JSON. Loading skeletons, empty, error/retry and index-offline states are covered. Accessibility includes landmarks, a skip link, live regions, visible focus, a focus trap, Escape-to-close, arrow-key result navigation and ARIA tabs.

## Key decisions

- **Offline-first:** optional ML/LLM packages degrade to deterministic hashing embeddings and an extractive summarizer, so the demo always runs.
- **Two-layer FHIR validation:** typed models plus the official schema catch different classes of errors; business rules enforce bundle-local references.
- **SQLite as system of record:** patients, resources, bundles, validation, summaries, chunks and audit live in one inspectable file.
- **Deterministic cache invalidation:** hashing the normalized record avoids stale summaries without manual cache management.
- **Plain-dict FHIR storage:** validated resources are stored as the same JSON clients receive, keeping the API transparent.

## Tradeoffs

SQLite and in-process FAISS are ideal for a single-node demo but not for concurrent production writes or incremental scale. There is no authentication, RBAC, encryption or immutable audit store. The offline summarizer is safe and deterministic but less fluent than a hosted LLM, and no LLM output is clinically validated. Attachments use base64 rather than an object store, and the app is not a full FHIR REST server.

## Validation

`pytest` passes all 73 tests. The FHIR report shows 51/51 valid bundles, 0 errors and 0 warnings. API smoke tests cover health, search, filters, patients, detail, bundle and resource endpoints. UI checks covered live search, filters, drawer tabs, JSON expansion, focus behavior, empty/error states and responsive layouts. The 50-record latency test passes well inside the two-second budget.

## Research and next steps

I referenced the HL7 FHIR R4 specification and JSON Schema, `fhir.resources`, the MiniLM model card, FAISS documentation, the WAI-ARIA Authoring Practices and Tailwind docs. Next I would add SSO/RBAC, Postgres + pgvector with incremental indexing, PHI de-identification before external LLM calls, retrieval/summary evaluation with clinician review, streaming summaries with human approval, a FHIR REST facade and Playwright end-to-end tests.

**Bottom line:** the platform demonstrates an end-to-end, auditable and accessible path from messy records to validated FHIR, trustworthy cached summaries and sub-second semantic retrieval — while remaining fully runnable offline on synthetic data.
