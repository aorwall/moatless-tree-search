import json
import anthropic.types
from moatless.completion.model import Usage, Completion
import pytest
from moatless.completion.completion import CompletionModel, LLMResponseFormat


class TestCompletion:
    def test_from_llm_completion_with_dict_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {
            "choices": [{"message": {"content": "Test output"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 5

    def test_from_llm_completion_with_anthropic_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = anthropic.types.Message(
            id="msg_123",
            model="claude-3.5-sonnet",
            type="message",
            role="assistant",
            content=[
                anthropic.types.TextBlock(text="Test output", type="text"),
                anthropic.types.ToolUseBlock(
                    id="tool_1", input={"query": "test"}, name="search", type="tool_use"
                ),
            ],
            usage=anthropic.types.Usage(input_tokens=10, output_tokens=20),
        )
        model = "claude-3.5-sonnet"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response.model_dump()
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 20

    def test_from_llm_completion_with_missing_usage(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {"choices": [{"message": {"content": "Test output"}}]}
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert completion.usage is None

    def test_from_llm_completion_with_unexpected_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = "Unexpected string response"
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion is None

    def test_from_llm_completion_with_multiple_messages(self):
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        completion_response = {
            "choices": [
                {"message": {"content": "I'm doing well, thank you for asking!"}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 20
        assert completion.usage.completion_tokens == 10


def test_serialization_deserialization():
    model = CompletionModel(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        response_format=LLMResponseFormat.TOOLS,
    )

    serialized = model.model_dump()
    assert serialized["response_format"] == "tool_call"

    deserialized = CompletionModel.model_validate(serialized)
    assert deserialized.response_format == LLMResponseFormat.TOOLS

    # Check if it's JSON serializable
    json_string = json.dumps(serialized, indent=2)
    print(json_string)
    assert json_string  # This will raise an error if not serializable
