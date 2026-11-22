"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ehr_media.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw",
        artifacts_dir=tmp_path / "artifacts",
        frontend_dir=tmp_path / "frontend",
    )
