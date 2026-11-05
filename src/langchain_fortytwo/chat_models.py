"""Fortytwo chat model with reasoning and strict schema support."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from typing import Any, Literal

import openai
from langchain_core.language_models import LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.fortytwo.network/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_RESPONSES_OPTIONS = frozenset(
    {
        "previous_response_id",
        "use_previous_response_id",
        "reasoning",
        "truncation",
        "context_management",
        "include",
        "store",
    }
)


def _headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    """Preserve caller identification after the proposed package marker."""
    marker = f"langchain-fortytwo/{version('langchain-fortytwo')}"
    result: dict[str, str] = {}
    caller_agent: str | None = None
    for name, value in (headers or {}).items():
        if name.lower() == "user-agent":
            if not isinstance(value, str):
                raise ValueError("The Fortytwo User-Agent marker cannot be removed.")
            caller_agent = value
        else:
            result[name.lower()] = value
    if caller_agent and caller_agent != marker and not caller_agent.startswith(marker + " "):
        marker += " " + caller_agent
    elif caller_agent:
        marker = caller_agent
    result["User-Agent"] = marker
    return result


def _check_chat_options(options: Mapping[str, Any]) -> None:
    for name in _RESPONSES_OPTIONS:
        if options.get(name) is not None and options.get(name) is not False:
            raise ValueError(f"{name} is not supported by Fortytwo Chat Completions.")
    if options.get("use_responses_api") is True:
        raise ValueError("Fortytwo supports Chat Completions, not the Responses API.")
    if options.get("output_version") == "responses/v1":
        raise ValueError("Fortytwo does not support the Responses output format.")
    effort = options.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or effort not in _EFFORTS):
        raise ValueError("Unsupported Fortytwo reasoning_effort.")
    for tool in options.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") != "function":
            raise ValueError("Fortytwo supports function tools only.")


def _strict_response_format(payload: dict[str, Any]) -> None:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = dict(response_format.get("json_schema", {}))
        if schema.get("strict") is False:
            raise ValueError("Fortytwo json_schema requires strict=True.")
        schema["strict"] = True
        payload["response_format"] = {**response_format, "json_schema": schema}


class ChatFortytwo(BaseChatOpenAI):
    """Call Fortytwo through LangChain's chat interface.

    Set `FORTYTWO_API_KEY`, or pass `api_key`. The `base_url` argument and
    `FORTYTWO_API_BASE` override the production endpoint. SDK-client injection
    is unsupported; use `http_client` and `http_async_client` for transports.
    """

    model_config = ConfigDict(hide_input_in_errors=True)
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    openai_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("FORTYTWO_API_KEY", default=None),
        alias="api_key",
        repr=False,
    )
    openai_api_base: str = Field(
        default_factory=from_env("FORTYTWO_API_BASE", default=DEFAULT_API_BASE),
        alias="base_url",
    )
    reasoning_effort: ReasoningEffort | None = None
    stream_usage: bool = True
    use_responses_api: bool = False

    @model_validator(mode="before")
    @classmethod
    def _check_client_and_protocol(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for name in ("client", "async_client", "root_client", "root_async_client"):
                if values.get(name) is not None:
                    raise ValueError("Inject HTTP transports instead of preconfigured SDK clients.")
            _check_chat_options(values)
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Build both clients without OpenAI credential or gateway fallback."""
        if not self.openai_api_key or not self.openai_api_key.get_secret_value():
            raise ValueError("Set FORTYTWO_API_KEY or pass a non-empty api_key.")
        if self.n is not None and (self.n < 1 or (self.n > 1 and self.streaming)):
            raise ValueError("n must be positive and must be 1 when streaming.")
        if not self.openai_api_base:
            raise ValueError("Fortytwo base_url must not be empty.")
        if self.openai_proxy:
            raise ValueError("Configure a proxy through the supplied HTTP transports.")
        _check_chat_options(
            {
                name: getattr(self, name, None)
                for name in _RESPONSES_OPTIONS | {"use_responses_api", "output_version"}
            }
        )
        _check_chat_options(self.model_kwargs)
        headers = _headers(self.default_headers)
        self.default_headers = headers
        params: dict[str, Any] = {
            "api_key": self.openai_api_key.get_secret_value(),
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": headers,
            "default_query": self.default_query,
            "organization": "",
            "project": "",
            "admin_api_key": "",
            "webhook_secret": "",
        }
        if self.max_retries is not None:
            params["max_retries"] = self.max_retries
        if self.root_client is None:
            self.root_client = openai.OpenAI(http_client=self.http_client, **params)
            for name in ("organization", "project", "admin_api_key", "webhook_secret"):
                setattr(self.root_client, name, None)
            self.client = self.root_client.chat.completions
        if self.root_async_client is None:
            self.root_async_client = openai.AsyncOpenAI(
                http_client=self.http_async_client, **params
            )
            for name in ("organization", "project", "admin_api_key", "webhook_secret"):
                setattr(self.root_async_client, name, None)
            self.async_client = self.root_async_client.chat.completions
        self._add_version("langchain-fortytwo", version("langchain-fortytwo"))
        return self

    @property
    def _llm_type(self) -> str:
        return "chat-fortytwo"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "FORTYTWO_API_KEY"}

    @property
    def _uses_gateway(self) -> bool:
        return False

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "fortytwo"
        return params

    def _use_responses_api(self, payload: dict[str, Any]) -> bool:
        _check_chat_options(payload)
        return False

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.pop("_fortytwo_json_schema_strict", None) is False:
            raise ValueError("Fortytwo json_schema requires strict=True.")
        _check_chat_options(kwargs)
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _check_chat_options(payload)
        _strict_response_format(payload)
        if "extra_headers" in payload:
            overrides = payload["extra_headers"] or {}
            overridden_names = {name.lower() for name in overrides}
            merged = {
                name: value
                for name, value in (self.default_headers or {}).items()
                if name.lower() not in overridden_names
            }
            payload["extra_headers"] = _headers({**merged, **overrides})
        return payload

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for generation, choice in zip(result.generations, data.get("choices", []), strict=False):
            reasoning = choice.get("message", {}).get("reasoning")
            if isinstance(reasoning, str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
            if isinstance(generation.message, AIMessage):
                generation.message.response_metadata["model_provider"] = "fortytwo"
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if result is not None and isinstance(result.message, AIMessageChunk):
            result.message.response_metadata["model_provider"] = "fortytwo"
            # Match LangChain's direct and beta structured-stream event shapes.
            choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
            if choices:
                reasoning = (choices[0].get("delta") or {}).get("reasoning")
                if isinstance(reasoning, str):
                    result.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel] | type | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Bind a schema, requiring strict JSON-schema responses when selected."""
        if method == "json_schema":
            if isinstance(schema, type) and issubclass(schema, BaseModelV1):
                raise ValueError("Use Pydantic v2 for json_schema, or select function_calling.")
            strict = True if strict is None else strict
            if strict is False:
                # Pydantic response formats retain the class rather than a dict,
                # so preserve strictness until our pre-HTTP request boundary.
                kwargs["_fortytwo_json_schema_strict"] = False
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )
