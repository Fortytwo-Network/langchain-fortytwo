import json

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.parametrize("kind", ["png", "jpeg", "webp"])
@pytest.mark.parametrize("standard_block", [False, True])
def test_supported_image_blocks_preserved(model_factory, kind, standard_block):
    model, requests = model_factory()
    data = "c3ludGhldGljLWltYWdl"
    block = (
        {"type": "image", "base64": data, "mime_type": f"image/{kind}"}
        if standard_block
        else {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}}
    )
    model.invoke([HumanMessage(content=[{"type": "text", "text": "Describe"}, block])])
    content = json.loads(requests[0].content)["messages"][0]["content"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/{kind};base64,{data}"},
    }
