from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path

import pytest

from casefile.agents.graph import compile_four_agent_graph
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.config import get_settings
from casefile.llm import AnthropicJSONClient
from casefile.retrieval import CaseFileIndex, SentenceTransformerEmbedder


ROOT = Path(__file__).resolve().parents[1]


def test_deleted_runtime_and_obsolete_switches_are_absent() -> None:
    assert not (ROOT / "casefile" / "agent").exists()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "casefile").rglob("*.py")
    )
    forbidden = (
        "casefile." + "agent.",
        "mock" + "_calendar",
        "use" + "_model",
        "no" + "_model",
        "Task" + "Plan",
    )
    assert all(value not in source for value in forbidden)


def test_google_provider_requires_its_explicit_dependency(
    tmp_path, monkeypatch
) -> None:
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    settings = replace(
        get_settings(),
        calendar_provider="google",
        google_credentials=credentials,
        nsda_provider="disabled",
        nsda_base_url=None,
    )

    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr("casefile.config.import_module", missing_dependency)
    with pytest.raises(ValueError, match="Calendar.*package extra"):
        settings.validate_configuration()


def test_missing_model_configuration_is_recorded_and_raised() -> None:
    client = AnthropicJSONClient(api_key=None)

    with pytest.raises(CaseFileError) as caught:
        client.complete_json(system="Return JSON.", user="{}")

    assert caught.value.code == ErrorCode.MODEL_CONFIGURATION_ERROR
    assert len(client.calls) == 1
    assert client.calls[0].status == "failed"


def test_missing_langgraph_prevents_four_agent_graph_construction(monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_langgraph(name, *args, **kwargs):
        if name == "langgraph.graph":
            raise ImportError("injected missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_langgraph)
    with pytest.raises(CaseFileError) as caught:
        compile_four_agent_graph(object())

    assert caught.value.code == ErrorCode.CONFIGURATION_ERROR


def test_missing_embedding_assets_prevent_startup(isolated_settings) -> None:
    settings = replace(
        isolated_settings,
        embedding_model_path=isolated_settings.data_dir / "missing-model",
    )

    with pytest.raises(CaseFileError) as caught:
        SentenceTransformerEmbedder(settings)

    assert caught.value.code == ErrorCode.CONFIGURATION_ERROR


def test_chroma_query_failure_is_not_replaced_by_json_ranking(
    isolated_settings,
) -> None:
    class Embedder:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        dimensions = 384

        def embed(self, texts):
            return [[0.0] * self.dimensions for _ in texts]

    class Collection:
        metadata = {
            "casefile_schema_version": 1,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "hnsw:space": "cosine",
        }

        def query(self, **kwargs):
            raise RuntimeError("injected Chroma failure")

    class Client:
        collection = Collection()

        def get_or_create_collection(self, *args, **kwargs):
            return self.collection

        def get_collection(self, *args, **kwargs):
            return self.collection

    isolated_settings.cards_path.write_text(
        '[{"id":"card-1","resolution":"R1","side":"pro",'
        '"source_file":"cards.docx","embedding_text":"consumer protection",'
        '"returned_document":"Citation\\nEvidence","body":"Evidence",'
        '"ingest_status":"ok",'
        '"flags":[]}]\n',
        encoding="utf-8",
    )
    index = CaseFileIndex(
        isolated_settings,
        client=Client(),
        embedder=Embedder(),
    )

    with pytest.raises(CaseFileError) as caught:
        index.search_cards(
            "consumer protection",
            resolution="R1",
            side="pro",
        )

    assert caught.value.code == ErrorCode.RETRIEVAL_UNAVAILABLE
