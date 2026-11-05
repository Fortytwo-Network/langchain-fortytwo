from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_fortytwo import ChatFortytwo


class TestChatFortytwoStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self):
        return ChatFortytwo

    @property
    def chat_model_params(self):
        return {"api_key": "synthetic-standard-test-key"}

    @property
    def init_from_env_params(self):
        return (
            {
                "FORTYTWO_API_KEY": "env-test-key",
                "FORTYTWO_API_BASE": "https://env.example.test/v1",
            },
            {},
            {"openai_api_key": "env-test-key", "openai_api_base": "https://env.example.test/v1"},
        )

    @property
    def structured_output_kwargs(self):
        return {"method": "json_schema"}

    @property
    def supports_image_inputs(self):
        return True
