import json
import threading
from datetime import date
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import agent_memory.query_ui as query_ui
from agent_memory.config import AppConfig
from agent_memory.query.agent import MemoryCitationValidationError, MemoryQueryResult
from agent_memory.query.retrieve import MemorySource


@pytest.fixture
def query_ui_server(tmp_path):
    database_path = tmp_path / "memory.db"
    database_path.touch()
    settings = AppConfig(
        config_path=tmp_path / "config.toml",
        project_root=tmp_path,
        data_dir=tmp_path,
        database_path=database_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_model="gemma4",
        embedding_model="qwen3-embedding",
        embedding_dimensions=3,
        num_ctx=1024,
        num_predict=256,
        keep_alive="1m",
        timeout_seconds=30,
        skip_headline_patterns=(),
    )
    server = query_ui.create_query_server(
        settings,
        database_path,
        model="gemma4",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(url, payload, *, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    request = Request(
        f"{url}/api/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    return urlopen(request)


def test_query_ui_shows_answer_and_source_evidence(
    query_ui_server,
    monkeypatch,
):
    calls = {}
    source = MemorySource(
        citation_id="c1",
        fact_id="fact-cash",
        document_id="doc-quarterly",
        headline="September quarterly",
        published_at="2026-10-13T21:39:00+00:00",
        quote="Cash balance of A$200 million at 30 September 2026.",
        source_path="data/CXO/quarterly.md",
    )

    def fake_answer(
        database_path,
        client,
        question,
        *,
        knowledge_date,
        ticker,
        model,
        chunk_fallback,
    ):
        calls.update(
            database_path=database_path,
            client=client,
            question=question,
            knowledge_date=knowledge_date,
            ticker=ticker,
            model=model,
            chunk_fallback=chunk_fallback,
        )
        return MemoryQueryResult(
            query_id="query-1",
            question=question,
            ticker=ticker,
            knowledge_date=knowledge_date,
            intent="state",
            answer="The cash balance was A$200 million.",
            citations=(source,),
            retrieved_facts=(),
            used_chunk_fallback=False,
            debug={"query_id": "query-1", "available_citation_ids": ["c1"]},
        )

    def fake_client_factory(settings, database_path):
        del settings, database_path
        return object()

    monkeypatch.setattr(query_ui, "LLMClient", fake_client_factory)
    monkeypatch.setattr(query_ui, "answer_memory", fake_answer)

    with urlopen(f"{query_ui_server}/") as response:
        page = response.read().decode("utf-8")
        assert response.headers["Content-Security-Policy"]
    assert 'id="question"' in page
    assert 'id="as-of"' in page
    assert "textContent" in page
    assert ".innerHTML" not in page
    assert "Query debug details" in page
    assert 'makeElement("details", "card debug")' in page
    assert "renderDebug(data.debug)" in page
    assert "data.debug);" in page

    request_body = {
        "question": "What was CXO's cash balance?",
        "ticker": "cxo",
        "as_of": "2026-10-15",
        "chunk_fallback": True,
    }
    with _post(query_ui_server, request_body) as response:
        result = json.loads(response.read())

    assert result["status"] == "answered"
    assert result["answer"] == "The cash balance was A$200 million."
    assert result["debug"]["available_citation_ids"] == ["c1"]
    assert result["citations"] == [
        {
            "citation_id": "c1",
            "document_id": "doc-quarterly",
            "headline": "September quarterly",
            "published_at": "2026-10-13T21:39:00+00:00",
            "quote": "Cash balance of A$200 million at 30 September 2026.",
            "source_path": "data/CXO/quarterly.md",
            "fact_id": "fact-cash",
        }
    ]
    assert calls["question"] == request_body["question"]
    assert calls["ticker"] == "CXO"
    assert calls["knowledge_date"] == date(2026, 10, 15)
    assert calls["chunk_fallback"] is True
    assert calls["model"] == "gemma4"


def test_query_ui_returns_debug_on_citation_validation_failure(
    query_ui_server, monkeypatch
):
    debug = {
        "query_id": "failed-query",
        "available_citation_ids": ["c1"],
        "answer_attempts": [
            {
                "raw_response": '{"answer":"<script>alert(1)</script>","citation_ids":[]}',
                "validation_error": "a non-abstaining answer has no citations",
            }
        ],
    }

    def fake_answer(*args, **kwargs):
        raise MemoryCitationValidationError(
            "Memory answer failed citation validation after one correction: "
            "a non-abstaining answer has no citations",
            debug=debug,
        )

    monkeypatch.setattr(query_ui, "LLMClient", lambda *args: object())
    monkeypatch.setattr(query_ui, "answer_memory", fake_answer)
    with pytest.raises(HTTPError) as failure:
        _post(
            query_ui_server,
            {"question": "Cash?", "ticker": "CXO", "as_of": "2026-10-15"},
        )
    assert failure.value.code == 422
    payload = json.loads(failure.value.read())
    assert "a non-abstaining answer has no citations" in payload["error"]
    assert payload["debug"] == debug
    assert "answer" not in payload


def test_query_ui_keeps_regular_validation_errors_without_debug(
    query_ui_server, monkeypatch
):
    def fake_answer(*args, **kwargs):
        raise ValueError("Query planner selected an entity outside the allowed set")

    monkeypatch.setattr(query_ui, "LLMClient", lambda *args: object())
    monkeypatch.setattr(query_ui, "answer_memory", fake_answer)
    with pytest.raises(HTTPError) as failure:
        _post(
            query_ui_server,
            {"question": "Cash?", "ticker": "CXO", "as_of": "2026-10-15"},
        )
    assert failure.value.code == 422
    assert json.loads(failure.value.read()) == {
        "error": "Query planner selected an entity outside the allowed set"
    }


def test_query_ui_rejects_invalid_date_and_cross_origin_requests(query_ui_server):
    with pytest.raises(HTTPError) as invalid_date:
        _post(
            query_ui_server,
            {
                "question": "Question",
                "ticker": "CXO",
                "as_of": "15 October 2026",
            },
            origin=query_ui_server,
        )
    assert invalid_date.value.code == 400

    with pytest.raises(HTTPError) as cross_origin:
        _post(
            query_ui_server,
            {
                "question": "Question",
                "ticker": "CXO",
                "as_of": "2026-10-15",
            },
            origin="http://evil.example",
        )
    assert cross_origin.value.code == 403
