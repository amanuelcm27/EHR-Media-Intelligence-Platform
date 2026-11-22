"""Frontend contract tests for the Task 5 clinician UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ehr_media.api import create_app
from ehr_media.config import Settings

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_markup_is_accessible_and_complete() -> None:
    markup = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    # Search and filter affordances.
    assert 'role="search"' in markup
    assert 'id="search-input"' in markup
    assert 'id="type-filters"' in markup
    assert 'data-type="DocumentReference"' in markup
    assert 'data-type="Observation"' in markup
    assert 'id="start-date"' in markup
    assert 'id="end-date"' in markup

    # Accessibility contract.
    assert "skip-link" in markup
    assert 'aria-live="polite"' in markup
    assert 'role="dialog"' in markup
    assert 'aria-modal="true"' in markup
    assert 'role="tablist"' in markup
    assert 'role="tabpanel"' not in markup  # panels are created by app.js

    # Static assets and demo-data warning.
    assert "/static/app.css" in markup
    assert "/static/app.js" in markup
    assert "no real patient data" in markup.lower()


def test_frontend_script_covers_key_interactions() -> None:
    script = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "AbortController" in script
    assert "setTimeout" in script
    assert "renderSkeletons" in script
    assert "renderEmpty" in script
    assert "renderError" in script
    assert "trapFocus" in script
    assert "handleTabKeydown" in script
    assert 'event.key === "Escape"' in script
    assert "aria-pressed" in script


def test_tailwind_stylesheet_is_built() -> None:
    stylesheet = (FRONTEND_DIR / "app.css").read_text(encoding="utf-8")

    assert len(stylesheet) > 10_000
    assert ".result-card" in stylesheet
    assert ".filter-chip" in stylesheet
    assert "@media (prefers-reduced-motion" in stylesheet


def test_api_serves_the_frontend_bundle(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "raw",
        artifacts_dir=tmp_path / "artifacts",
        frontend_dir=FRONTEND_DIR,
    )
    app = create_app(settings, auto_build=False)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Clinical media search" in index.text
        assert client.get("/static/app.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
