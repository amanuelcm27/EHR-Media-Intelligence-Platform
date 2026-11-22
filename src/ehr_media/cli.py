"""Command line interface for building, searching and serving the platform."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .bootstrap import (
    build_platform,
    ensure_platform,
    patient_from_payload,
    platform_stats,
)
from .config import Settings, get_settings
from .fhir import SqliteStore
from .search import SearchService
from .summaries import ClinicalSummarizer


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    for key, value in (
        ("data_dir", getattr(args, "data_dir", None)),
        ("artifacts_dir", getattr(args, "artifacts_dir", None)),
        ("embedding_provider", getattr(args, "embedding_provider", None)),
        ("vector_backend", getattr(args, "vector_backend", None)),
        ("llm_provider", getattr(args, "llm_provider", None)),
    ):
        if value:
            overrides[key] = value
    return Settings(**overrides) if overrides else get_settings()


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_build(args: argparse.Namespace) -> int:
    settings = _settings(args)
    result = build_platform(settings, write_artifacts=not args.no_artifacts)
    _print(result)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ensure_platform(settings)
    service = SearchService(settings)
    try:
        response = service.search(
            args.query,
            top_k=args.top_k,
            resource_types=args.type or None,
            start_date=args.start,
            end_date=args.end,
            patient_id=args.patient,
        )
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print(response.model_dump(mode="json"))
        return 0
    print(
        f"{response.count} result(s) in {response.took_ms} ms "
        f"[{response.index.get('provider')} / {response.index.get('backend')}]"
    )
    for result in response.results:
        print(
            f"{result.rank}. {result.patient_name} ({result.mrn}) | "
            f"{result.resource_type} | {result.effective_date or 'undated'} | "
            f"score {result.score:.3f}"
        )
        print(f"   {result.snippet}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ensure_platform(settings)
    store = SqliteStore(settings.db_path)
    payload = store.get_patient(args.patient_id)
    if payload is None:
        print(f"Patient {args.patient_id!r} not found", file=sys.stderr)
        return 1
    patient = patient_from_payload(payload)
    summary = ClinicalSummarizer(settings, store=store).summarize(
        patient, force=args.force
    )
    _print(summary.model_dump(mode="json"))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ensure_platform(settings)
    _print(platform_stats(settings))
    return 0


def cmd_patients(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ensure_platform(settings)
    store = SqliteStore(settings.db_path)
    rows = store.list_patients()
    if args.query:
        needle = args.query.casefold()
        rows = [
            row
            for row in rows
            if needle in row["display_name"].casefold() or needle in row["mrn"].casefold()
        ]
    _print(rows[: args.limit])
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    settings = _settings(args)
    app = create_app(settings, auto_build=not args.no_build)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", help="Raw data directory (default: data/raw)")
    parser.add_argument(
        "--artifacts-dir", help="Artifacts directory (default: artifacts)"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "hashing", "sentence-transformers", "openai"],
        help="Override the embedding provider",
    )
    parser.add_argument(
        "--vector-backend", choices=["auto", "faiss", "numpy"], help="Vector index backend"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "offline", "anthropic", "openai"],
        help="Override the summary provider",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ehr-media",
        description="EHR media ingestion, FHIR R4, summaries and semantic search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Run ingestion, FHIR, summaries and index")
    _common(build)
    build.add_argument("--no-artifacts", action="store_true", help="Skip JSON artifact exports")
    build.set_defaults(func=cmd_build)

    search = subparsers.add_parser("search", help="Semantic search")
    _common(search)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--type", action="append", help="Resource type filter (repeatable)")
    search.add_argument("--start", help="Start date YYYY-MM-DD")
    search.add_argument("--end", help="End date YYYY-MM-DD")
    search.add_argument("--patient", help="Patient id filter")
    search.add_argument("--json", action="store_true", help="Print full JSON response")
    search.set_defaults(func=cmd_search)

    summarize = subparsers.add_parser("summarize", help="Generate one patient summary")
    _common(summarize)
    summarize.add_argument("patient_id")
    summarize.add_argument("--force", action="store_true")
    summarize.set_defaults(func=cmd_summarize)

    stats = subparsers.add_parser("stats", help="Show database/index statistics")
    _common(stats)
    stats.set_defaults(func=cmd_stats)

    patients = subparsers.add_parser("patients", help="List patients")
    _common(patients)
    patients.add_argument("--query")
    patients.add_argument("--limit", type=int, default=50)
    patients.set_defaults(func=cmd_patients)

    serve = subparsers.add_parser("serve", help="Run the API and frontend")
    _common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--no-build", action="store_true", help="Do not auto-build missing data")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
