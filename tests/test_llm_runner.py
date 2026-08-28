from __future__ import annotations

import json
from pathlib import Path

from deid_battery.runners import llm


ROOT = Path(__file__).resolve().parents[1]


def test_committed_prompt_assets_build_system_and_user_messages() -> None:
    template, example, labels = llm._load_assets(ROOT / "prompts")

    system, user = llm._prompts(
        "Patiënt Jan Janssens kwam vandaag.", labels, template, example
    )

    assert len(labels) == 15
    assert "Name:Patient" in system
    assert "Jan Janssens" in system
    assert user == "Originele tekst:\nPatiënt Jan Janssens kwam vandaag.\n\nJSON:"


def test_thinking_response_is_parsed_and_aligned() -> None:
    content = (
        "<think>reasoning that must not enter the parser</think>"
        '{"spans":[{"annotated_text":"Jan Janssens","label":"Name:Patient"}]}'
    )
    text = "Patiënt Jan Janssens kwam vandaag."

    spans = llm._extract(text, llm._parse(content))

    assert spans == [
        {
            "begin": 8,
            "end": 20,
            "label": "Name:Patient",
            "text": "Jan Janssens",
            "category": "Name",
            "subtype": "Patient",
        }
    ]


def test_truncated_json_salvages_complete_span_items() -> None:
    content = (
        '{"spans": ['
        '{"annotated_text":"01/02/2025","label":"Date"},'
        '{"annotated_text":"unfinished"'
    )

    assert llm._parse(content) == [
        {"annotated_text": "01/02/2025", "label": "Date"}
    ]


def test_thinking_request_uses_recorded_sampling_without_strict_schema(
    monkeypatch,
) -> None:
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"spans": []}'}}]}

    def fake_post(url, *, json, timeout):
        captured.update({"url": url, "payload": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    response_format = llm._schema(["Date"])

    result = llm._annotate(
        "http://127.0.0.1:11434/v1",
        "system",
        "user",
        response_format,
        "qwen3:8b",
        0.6,
        0.95,
        8000,
        True,
        None,
        attempts=1,
    )

    assert json.loads(result) == {"spans": []}
    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] == 600
    assert captured["payload"]["temperature"] == 0.6
    assert captured["payload"]["top_p"] == 0.95
    assert captured["payload"]["max_tokens"] == 8000
    assert "response_format" not in captured["payload"]
