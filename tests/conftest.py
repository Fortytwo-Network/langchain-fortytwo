"""Network-free HTTP fixtures; live tests supply their own explicit environment."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from langchain_fortytwo import ChatFortytwo


def completion(message: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen/qwen3.8-27b",
        "choices": [
            {
                "index": 0,
                "message": message or {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def sse(chunks: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def model_factory() -> Iterator[Callable[..., tuple[ChatFortytwo, list[httpx.Request]]]]:
    clients: list[httpx.Client] = []

    def create(
        response: dict[str, Any] | bytes | None = None,
        *,
        status: int = 200,
        **kwargs: Any,
    ) -> tuple[ChatFortytwo, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if isinstance(response, bytes):
                return httpx.Response(
                    status, content=response, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(status, json=response or completion())

        client = httpx.Client(transport=httpx.MockTransport(handler))
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        model = ChatFortytwo(
            api_key="synthetic-test-key",
            http_client=client,
            http_async_client=async_client,
            max_retries=0,
            **kwargs,
        )
        return model, requests

    yield create
    for client in clients:
        client.close()
