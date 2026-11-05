from importlib.metadata import version

import pytest


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_default_header_merging(model_factory, asynchronous):
    original = {"user-agent": "my-app/2", "X-Custom": "preserved"}
    model, requests = model_factory(default_headers=original)
    if asynchronous:
        await model.ainvoke("Hello")
    else:
        model.invoke("Hello")
    request = requests[0]
    assert (
        request.headers["user-agent"]
        == f"langchain-fortytwo/{version('langchain-fortytwo')} my-app/2"
    )
    assert request.headers["x-custom"] == "preserved"
    assert original == {"user-agent": "my-app/2", "X-Custom": "preserved"}


def test_per_call_headers_keep_defaults_and_case_insensitive_override(model_factory):
    model, requests = model_factory(
        default_headers={"User-Agent": "constructor/1", "X-Other": "kept"}
    )
    model.invoke("Hello", extra_headers={"X-Custom": "per-call"})
    assert requests[0].headers["user-agent"].endswith(" constructor/1")
    override = {"USER-AGENT": "invocation/3", "x-other": "changed"}
    model.invoke("Hello", extra_headers=override)
    assert requests[1].headers["user-agent"] == "langchain-fortytwo/0.1.0 invocation/3"
    assert requests[1].headers["x-other"] == "changed"
    assert override["USER-AGENT"] == "invocation/3"


def test_null_marker_cannot_be_removed(model_factory):
    model, requests = model_factory()
    with pytest.raises(ValueError, match="cannot be removed"):
        model.invoke("Hello", extra_headers={"User-Agent": None})
    assert not requests
