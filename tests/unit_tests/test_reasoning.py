import json

import pytest
from conftest import completion, sse
from langchain_core.messages import AIMessageChunk
from openai.types.chat import ChatCompletion


@pytest.mark.parametrize("reasoning", ["Think carefully", ""])
@pytest.mark.parametrize("sdk_object", [False, True])
def test_reasoning_in_response(model_factory, reasoning, sdk_object):
    model, _ = model_factory()
    data = completion({"role": "assistant", "content": "Answer", "reasoning": reasoning})
    response = ChatCompletion.model_validate(data) if sdk_object else data
    message = model._create_chat_result(response).generations[0].message
    assert message.additional_kwargs["reasoning_content"] == reasoning
    assert message.content == "Answer"


def test_absent_reasoning_and_multiple_choices(model_factory):
    model, _ = model_factory()
    data = completion()
    data["choices"].append(
        {
            "index": 1,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Two", "reasoning": "Second thought"},
        }
    )
    messages = model._create_chat_result(data).generations
    assert "reasoning_content" not in messages[0].message.additional_kwargs
    assert messages[1].message.additional_kwargs["reasoning_content"] == "Second thought"


def chunks():
    result = []
    for delta in [
        {"role": "assistant", "reasoning": ""},
        {"reasoning": "Think "},
        {"reasoning": "first"},
        {"content": "Answer"},
    ]:
        result.append(
            {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen/qwen3.8-27b",
                "choices": [{"index": 0, "delta": delta}],
            }
        )
    result.append(
        {
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "qwen/qwen3.8-27b",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_merges_reasoning_and_usage(model_factory, asynchronous):
    model, requests = model_factory(sse(chunks()))
    outputs = (
        [c async for c in model.astream("Hello")] if asynchronous else list(model.stream("Hello"))
    )
    combined = outputs[0]
    for chunk in outputs[1:]:
        combined += chunk
    assert combined.content == "Answer"
    assert combined.additional_kwargs["reasoning_content"] == "Think first"
    assert combined.usage_metadata["total_tokens"] == 5
    assert any(c.usage_metadata for c in outputs)
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_every_effort_passes_through(model_factory, effort):
    model, requests = model_factory(reasoning_effort=effort)
    model.invoke("Hello")
    model.invoke("Hello", reasoning_effort=effort)
    assert all(json.loads(r.content)["reasoning_effort"] == effort for r in requests)


def test_usage_only_and_null_delta(model_factory):
    model, _ = model_factory()
    result = model._convert_chunk_to_generation_chunk(chunks()[-1], AIMessageChunk, None)
    assert result.message.usage_metadata["total_tokens"] == 5
    assert (
        model._convert_chunk_to_generation_chunk(
            {"choices": [{"delta": None}]}, AIMessageChunk, None
        )
        is None
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("reasoning", ["Think first", "", None, 42])
def test_reasoning_in_direct_and_structured_stream_events(model_factory, wrapped, reasoning):
    model, requests = model_factory()
    chunk = {"choices": [{"index": 0, "delta": {"content": "Answer", "reasoning": reasoning}}]}
    event = {"type": "chunk", "chunk": chunk} if wrapped else chunk
    result = model._convert_chunk_to_generation_chunk(event, AIMessageChunk, None)
    assert result.message.content == "Answer"
    assert result.message.response_metadata["model_provider"] == "fortytwo"
    if isinstance(reasoning, str):
        assert result.message.additional_kwargs["reasoning_content"] == reasoning
    else:
        assert "reasoning_content" not in result.message.additional_kwargs
    assert requests == []
