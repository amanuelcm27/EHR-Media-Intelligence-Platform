"""Application configuration.

Settings are read from environment variables (prefix ``EHR_``) and an optional
``.env`` file. All relative paths are resolved against the repository root so
the CLI and the API behave identically no matter which directory they are
started from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EHR_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ------------------------------------------------------------
    data_dir: Path = Field(default=Path("data/raw"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    frontend_dir: Path = Field(default=Path("frontend"))

    # --- embeddings -------------------------------------------------------
    embedding_provider: str = "auto"  # auto|sentence-transformers|hashing|openai
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # --- vector index -----------------------------------------------------
    vector_backend: str = "auto"  # auto|faiss|numpy

    # --- summarization ----------------------------------------------------
    llm_provider: str = "auto"  # auto|anthropic|openai|offline
    anthropic_model: str = "claude-sonnet-4-5"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    max_summary_words: int = 200
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # --- search -----------------------------------------------------------
    default_top_k: int = 5
    max_top_k: int = 20
    hybrid_weight: float = 0.15
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- api --------------------------------------------------------------
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    log_level: str = "info"

    @field_validator(
        "data_dir", "artifacts_dir", "frontend_dir", mode="before"
    )
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    # -- derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.artifacts_dir / "ehr.db"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def fhir_dir(self) -> Path:
        return self.artifacts_dir / "fhir"

    @property
    def summaries_dir(self) -> Path:
        return self.artifacts_dir / "summaries"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def cleaned_dir(self) -> Path:
        return self.artifacts_dir / "cleaned"

    @property
    def frontend_index(self) -> Path:
        return self.frontend_dir / "index.html"

    def ensure_dirs(self) -> None:
        for path in (
            self.artifacts_dir,
            self.reports_dir,
            self.fhir_dir,
            self.summaries_dir,
            self.index_dir,
            self.cleaned_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
