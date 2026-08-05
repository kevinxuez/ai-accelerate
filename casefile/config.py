"""Runtime configuration with conservative, offline-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_roots(name: str, defaults: tuple[Path, ...]) -> tuple[Path, ...]:
    value = os.getenv(name)
    raw = value.split(os.pathsep) if value else [str(path) for path in defaults]
    return tuple(Path(path).expanduser().resolve() for path in raw if path.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _env_path("CASEFILE_DATA_DIR", PACKAGE_ROOT / "data")
    chroma_dir: Path = _env_path("CASEFILE_CHROMA_DIR", PACKAGE_ROOT / "chroma_db")
    rules_dir: Path = _env_path("CASEFILE_RULES_DIR", PACKAGE_ROOT / "rules")
    model: str = os.getenv("CASEFILE_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_base_url: str = os.getenv(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
    )
    min_relevance: float = float(os.getenv("CASEFILE_MIN_RELEVANCE", "0.08"))
    mock_calendar: bool = _env_bool("MOCK_CALENDAR", True)
    google_credentials: Path = _env_path(
        "GOOGLE_CALENDAR_CREDENTIALS", REPO_ROOT / "credentials.json"
    )
    google_token: Path = _env_path("GOOGLE_CALENDAR_TOKEN", REPO_ROOT / "token.json")
    nsda_base_url: str | None = os.getenv("NSDA_BASE_URL") or None
    nsda_api_key: str | None = os.getenv("NSDA_API_KEY") or None
    nsda_timeout_seconds: float = float(os.getenv("NSDA_TIMEOUT_SECONDS", "5"))
    nsda_mock_data: Path = _env_path(
        "NSDA_MOCK_DATA", PACKAGE_ROOT / "providers" / "nsda_mock.json"
    )
    allowed_ingest_roots: tuple[Path, ...] = _env_roots(
        "CASEFILE_INGEST_ROOTS", (REPO_ROOT / "background",)
    )
    max_upload_bytes: int = int(
        os.getenv("CASEFILE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    )
    requests_per_minute: int = int(os.getenv("CASEFILE_REQUESTS_PER_MINUTE", "60"))

    @property
    def cards_path(self) -> Path:
        return self.data_dir / "cards_labeled.json"

    @property
    def progress_path(self) -> Path:
        return self.data_dir / "progress.json"

    @property
    def audit_path(self) -> Path:
        return self.data_dir / "tool_calls.jsonl"

    @property
    def pending_dir(self) -> Path:
        return self.data_dir / ".casefile_pending"

    @property
    def calendar_pending_dir(self) -> Path:
        return self.data_dir / ".calendar_pending"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / ".casefile_uploads"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / ".casefile_sessions"

    @property
    def security_audit_path(self) -> Path:
        return self.data_dir / "security_events.jsonl"

    @property
    def idempotency_path(self) -> Path:
        return self.data_dir / "idempotency.json"

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.calendar_pending_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
