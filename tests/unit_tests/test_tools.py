import json

import pytest
from conftest import completion
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field


class Weather(BaseModel):
    """Look up the weather in a city."""

    city: str = Field(description="City name")


@pytest.mark.parametrize("tool_content", ["Sunny", [{"type": "text", "text": "Sunny"}]])
async def test_tool_call_and_result_history(model_factory, tool_content):
    model, requests = model_factory(
        completion(
            {
                "role": "assistant",
                "content": None,
                "reasoning": "Use weather",
                "tool_calls": [
                    {
                        "id": "call-weather",
                        "type": "function",
                        "function": {"name": "Weather", "arguments": '{"city":"Dubai"}'},
                    }
                ],
            }
        )
    )
    bound = model.bind_tools([Weather])
    result = await bound.ainvoke("Weather in Dubai?")
    assert result.tool_calls[0] == {
        "name": "Weather",
        "args": {"city": "Dubai"},
        "id": "call-weather",
        "type": "tool_call",
    }
    bound.invoke(
        [
            ("human", "Weather in Dubai?"),
            result,
            ToolMessage(content=tool_content, tool_call_id="call-weather"),
        ]
    )
    body = json.loads(requests[1].content)
    assert body["messages"][-1] == {
        "role": "tool",
        "content": tool_content,
        "tool_call_id": "call-weather",
    }
    assert body["messages"][-2]["tool_calls"][0]["id"] == "call-weather"
    assert body["tools"][0]["function"]["name"] == "Weather"
