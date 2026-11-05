import json

import pytest
from conftest import completion
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("strict", [None, True])
async def test_json_schema_defaults_strict_sync_async(model_factory, strict):
    model, requests = model_factory(
        completion({"role": "assistant", "content": '{"answer":"yes"}'})
    )
    bound = model.with_structured_output(Answer, method="json_schema", strict=strict)
    assert bound.invoke("Answer").answer == "yes"
    assert (await bound.ainvoke("Answer")).answer == "yes"
    assert all(
        json.loads(r.content)["response_format"]["json_schema"]["strict"] is True for r in requests
    )


def test_direct_schema_default_and_reject_false(model_factory):
    model, requests = model_factory()
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    model.invoke("Answer", response_format=schema)
    assert json.loads(requests[0].content)["response_format"]["json_schema"]["strict"] is True
    assert "strict" not in schema["json_schema"]
    with pytest.raises(ValueError, match="strict=True"):
        model.invoke(
            "Answer", response_format={"type": "json_schema", "json_schema": {"strict": False}}
        )
    assert len(requests) == 1


def test_strict_false_and_v1_rejected_before_http(model_factory):
    model, requests = model_factory()
    wrapper = model.with_structured_output(Answer, method="json_schema", strict=False)
    with pytest.raises(ValueError, match="strict=True"):
        wrapper.invoke("Answer")

    class V1Answer(BaseModelV1):
        answer: str

    with pytest.raises(ValueError, match="Pydantic v2"):
        model.with_structured_output(V1Answer, method="json_schema")
    assert not requests


def test_dict_schema_and_include_raw(model_factory):
    model, requests = model_factory(
        completion({"role": "assistant", "content": '{"answer":"yes"}'})
    )
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = model.with_structured_output(schema, method="json_schema", include_raw=True).invoke(
        "Answer"
    )
    assert result["parsed"] == {"answer": "yes"}
    assert result["parsing_error"] is None
    assert result["raw"].content == '{"answer":"yes"}'


def test_invalid_pydantic_result_reports_error(model_factory):
    model, _ = model_factory(completion({"role": "assistant", "content": '{"missing":"answer"}'}))
    with pytest.raises(ValidationError):
        model.with_structured_output(Answer, method="json_schema").invoke("Answer")
