import json

import openai
import pytest
from pydantic import SecretStr

from langchain_fortytwo import ChatFortytwo


def test_defaults_and_secret_representation(monkeypatch):
    monkeypatch.delenv("FORTYTWO_API_BASE", raising=False)
    model = ChatFortytwo(api_key="synthetic-test-key")
    assert model.model_name == "qwen/qwen3.8-27b"
    assert model.openai_api_base == "https://api.fortytwo.network/v1"
    assert isinstance(model.openai_api_key, SecretStr)
    assert "synthetic-test-key" not in repr(model)
    assert "synthetic-test-key" not in json.dumps(model.to_json())


def test_explicit_and_env_precedence(monkeypatch):
    monkeypatch.setenv("FORTYTWO_API_KEY", "fortytwo-env-test")
    monkeypatch.setenv("FORTYTWO_API_BASE", "https://env.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-test")
    monkeypatch.setenv("OPENAI_API_BASE", "https://wrong.example.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-sdk.example.test/v1")
    for name in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ADMIN_KEY", "OPENAI_WEBHOOK_SECRET"):
        monkeypatch.setenv(name, "wrong-provider-setting")
    monkeypatch.setenv("LANGSMITH_GATEWAY_BASE_URL", "https://wrong-gateway.example.test")
    env_model = ChatFortytwo()
    assert env_model.root_client.api_key == "fortytwo-env-test"
    assert str(env_model.root_async_client.base_url) == "https://env.example.test/v1/"
    for client in (env_model.root_client, env_model.root_async_client):
        assert client.organization is None
        assert client.project is None
        assert client.admin_api_key is None
        assert client.webhook_secret is None
    explicit = ChatFortytwo(api_key="explicit-test", base_url="https://explicit.example.test/v1")
    assert explicit.root_client.api_key == "explicit-test"
    assert explicit.root_async_client.api_key == "explicit-test"
    assert str(explicit.root_client.base_url) == "https://explicit.example.test/v1/"


@pytest.mark.parametrize("api_key", [None, ""])
def test_openai_credentials_cannot_fill_missing_key(monkeypatch, api_key):
    monkeypatch.delenv("FORTYTWO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-test")
    with pytest.raises(ValueError, match="FORTYTWO_API_KEY"):
        ChatFortytwo(api_key=api_key)


@pytest.mark.parametrize("field", ["client", "async_client", "root_client", "root_async_client"])
def test_sdk_injection_rejected(field):
    with pytest.raises(ValueError, match="HTTP transports"):
        ChatFortytwo(api_key="test", **{field: object()})


def test_auth_error_contract(model_factory):
    model, _ = model_factory(
        {"error": {"message": "Rejected synthetic key", "type": "authentication_error"}}, status=401
    )
    with pytest.raises(openai.AuthenticationError):
        model.invoke("Hello")


def test_api_validation_error_contract(model_factory):
    model, _ = model_factory(
        {"error": {"message": "Invalid synthetic input", "type": "invalid_request_error"}},
        status=400,
    )
    with pytest.raises(openai.BadRequestError):
        model.invoke("Hello")


def test_parameter_constraints():
    with pytest.raises(ValueError, match="n must"):
        ChatFortytwo(api_key="test", n=0)
    with pytest.raises(ValueError, match="n must"):
        ChatFortytwo(api_key="test", n=2, streaming=True)


def test_validation_diagnostics_hide_credentials():
    with pytest.raises(ValueError) as failure:
        ChatFortytwo(api_key="synthetic-sensitive-key", client=object())
    assert "synthetic-sensitive-key" not in str(failure.value)
    assert "synthetic-sensitive-key" not in repr(failure.value)
