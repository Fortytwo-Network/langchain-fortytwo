import json

import pytest


def test_chat_destination_auth_and_usage(model_factory):
    model, requests = model_factory(base_url="https://stage.example.test/v1")
    result = model.invoke("Hello")
    request = requests[0]
    assert str(request.url) == "https://stage.example.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer synthetic-test-key"
    assert json.loads(request.content) == {
        "model": "qwen/qwen3.8-27b",
        "stream": False,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    assert result.content == "Hello"
    assert {
        key: result.usage_metadata[key] for key in ("input_tokens", "output_tokens", "total_tokens")
    } == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert result.response_metadata["model_provider"] == "fortytwo"


async def test_async_chat(model_factory):
    model, requests = model_factory()
    result = await model.ainvoke("Hello")
    assert result.content == "Hello"
    assert requests[0].url.path == "/v1/chat/completions"


@pytest.mark.parametrize(
    "options",
    [
        {"use_responses_api": True},
        {"output_version": "responses/v1"},
        {"reasoning": {"effort": "low"}},
        {"previous_response_id": "resp-test"},
        {"tools": [{"type": "web_search_preview"}]},
        {"reasoning_effort": "invalid"},
        {"reasoning_effort": []},
        {"reasoning_effort": {}},
    ],
)
def test_responses_and_invalid_effort_never_send_http(model_factory, options):
    model, requests = model_factory()
    with pytest.raises(ValueError):
        model.invoke("Hello", **options)
    assert not requests


@pytest.mark.parametrize(
    "options",
    [
        {"use_responses_api": True},
        {"output_version": "responses/v1"},
        {"reasoning": {"effort": "low"}},
        {"use_previous_response_id": True},
    ],
)
def test_responses_constructor_rejected(options):
    from langchain_fortytwo import ChatFortytwo

    with pytest.raises(ValueError):
        ChatFortytwo(api_key="test", **options)
