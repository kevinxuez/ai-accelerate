from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from casefile.config import get_settings


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_docx() -> Path:
    return REPO_ROOT / "background" / "Copy of Pro Cards - Crypto.docx"


@pytest.fixture
def isolated_settings(tmp_path):
    settings = replace(
        get_settings(),
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        rules_dir=tmp_path / "rules",
        anthropic_api_key=None,
        mock_calendar=True,
    )
    settings.ensure_runtime_dirs()
    settings.progress_path.write_text("[]\n", encoding="utf-8")
    settings.cards_path.write_text("[]\n", encoding="utf-8")
    settings.rules_dir.mkdir(parents=True, exist_ok=True)
    return settings

