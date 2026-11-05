# langchain-fortytwo

LangChain chat integration for Fortytwo's OpenAI-compatible Chat Completions API.
The model is `qwen/qwen3.8-27b`. The integration preserves Fortytwo's reasoning
field alongside the answer, supports tools and images, and requests strict JSON
schemas when using JSON-schema structured output.

This is an unpublished source candidate. Install a built wheel as shown below.

## Install a local build

Python 3.10 or later is required. Build the reviewed source with `uv build`, then
install the wheel from its `dist` directory.

```sh
uv pip install dist/langchain_fortytwo-0.1.0-py3-none-any.whl
export FORTYTWO_API_KEY="your-api-key"
```

The intended package name for publication is `langchain-fortytwo`.

## Chat and streaming

```python
import asyncio
from langchain_fortytwo import ChatFortytwo

model = ChatFortytwo()
answer = model.invoke("What is 23 multiplied by 17?")
print(answer.content)
print(answer.additional_kwargs.get("reasoning_content", ""))

for chunk in model.stream("Explain the answer briefly."):
    print(chunk.content, end="")

async def main():
    answer = await model.ainvoke("Say hello.")
    print(answer.content)
    async for chunk in model.astream("Count to three."):
        print(chunk.content, end="")

asyncio.run(main())
```

`reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`
and `max`. Omitting it keeps the API default. Reasoning appears in
`additional_kwargs["reasoning_content"]` on answers and chunks; it is kept out
of the answer text. Usage metadata is retained, including final streaming usage.

## Tools

```python
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

class Weather(BaseModel):
    """Look up the weather in a city."""
    city: str = Field(description="City name")

with_tools = model.bind_tools([Weather])
messages = [("human", "Use Weather to look up Dubai.")]
answer = with_tools.invoke(messages)
if answer.tool_calls:
    call = answer.tool_calls[0]
    # Replace this demonstration result with the result of your own tool.
    result = ToolMessage(content="Sunny, 30 C", tool_call_id=call["id"])
    print(with_tools.invoke([*messages, answer, result]).content)
```

The application executes tools. The model returns tool calls; this package does
not run arbitrary functions on the application's behalf.

## Strict structured output

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str

city = model.with_structured_output(City, method="json_schema").invoke(
    "Name the city Dubai and its country."
)
print(city.name, city.country)
```

JSON-schema mode defaults to `strict=True`. Explicit `strict=False` can construct
a LangChain wrapper but raises an actionable error when invoked, before HTTP.
Native JSON-schema mode accepts Pydantic v2 classes, TypedDict and dictionary
schemas. Pydantic v1 classes are explicitly unsupported in this mode and raise
an actionable error before HTTP. The default method remains function calling,
whose behaviour is unchanged.
Pydantic results are validated locally; dictionary schemas rely on the API's
strict schema validation.

## Images

```python
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage

data = base64.b64encode(Path("image.png").read_bytes()).decode()
message = HumanMessage(content=[
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
])
print(model.invoke([message]).content)
```

The API supports PNG, JPEG and WebP images. GIF is not supported. This package
preserves image blocks rather than converting unsupported formats.

## Configuration and transport

Production defaults to `https://api.fortytwo.network/v1`. Explicit `api_key` and
`base_url` take precedence over `FORTYTWO_API_KEY` and `FORTYTWO_API_BASE`.
OpenAI API keys, base URLs, organisation/project values and LangSmith gateway
settings are not used as fallbacks. A missing Fortytwo key raises a local error.
`timeout`, `max_retries`, `http_client` and `http_async_client` are supported.
Preconfigured SDK-client injection is rejected because it could bypass the
integration's endpoint, credential and header guarantees.

This is a Chat Completions integration. Responses API options, built-in
Responses tools and Anthropic Messages calls are not supported.

The proposed User-Agent begins with `langchain-fortytwo/<installed-version>`.
A caller-supplied User-Agent follows the marker. Other caller headers are
preserved; per-call headers override constructor headers case-insensitively.
The marker cannot be removed through per-call headers. Backend logging and
attribution are controlled by the service; emitting a header does not confirm
that it is retained or used by the backend.

## Local verification

```sh
uv sync --frozen --all-groups
uv run --no-sync pytest --disable-socket --allow-unix-socket tests/unit_tests -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src
uv build
```

Standard tests are pinned to
`langchain-tests==1.1.9`; supported features are not skipped to disguise failures.
The unsupported v1-class exclusion is narrowly scoped to one inherited native
JSON-schema case; supported v2, TypedDict and dictionary cases remain enabled.

