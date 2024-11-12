import json
import logging
from typing import Optional, Any, Union, Self

import litellm
from instructor import OpenAISchema
from litellm import cost_per_token, NotFoundError
from pydantic import BaseModel, model_validator, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str = Field(..., description="The role of the sender")
    content: Optional[str] = Field(None, description="The message content")


class ToolCall(BaseModel):
    name: str = Field(..., description="The name of the tool being called")
    type: Optional[str] = Field(None, description="The type of tool call")
    input: Optional[dict[str, Any]] = Field(None, description="The input parameters for the tool")


class AssistantMessage(Message):
    role: str = Field("assistant", description="The role of the assistant")
    content: Optional[str] = Field(None, description="The assistant's message content")
    tool_call: Optional[ToolCall] = Field(None, description="Tool call made by the assistant")


class UserMessage(Message):
    role: str = Field("user", description="The role of the user")
    content: str = Field(..., description="The user's message content")


class Usage(BaseModel):
    completion_cost: float = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_completion_response(
        cls, completion_response: dict | BaseModel, model: str
    ) -> Union["Usage", None]:
        if isinstance(completion_response, BaseModel) and hasattr(
            completion_response, "usage"
        ):
            usage = completion_response.usage.model_dump()
        elif isinstance(completion_response, dict) and "usage" in completion_response:
            usage = completion_response["usage"]
        else:
            logger.warning(
                f"No usage info available in completion response: {completion_response}"
            )
            return None

        logger.debug(f"Usage: {json.dumps(usage, indent=2)}")

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)

        if usage.get("cache_creation_input_tokens"):
            prompt_tokens += usage["cache_creation_input_tokens"]

        completion_tokens = usage.get("completion_tokens") or usage.get(
            "output_tokens", 0
        )

        if usage.get("prompt_cache_hit_tokens"):
            cached_tokens = usage["prompt_cache_hit_tokens"]
        elif usage.get("cache_read_input_tokens"):
            cached_tokens = usage["cache_read_input_tokens"]
        else:
            cached_tokens = 0

        try:
            cost = litellm.completion_cost(
                completion_response=completion_response, model=model
            )
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = prompt_cost + completion_cost
            except NotFoundError as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0
            except Exception as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0

        return cls(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            completion_cost=self.completion_cost + other.completion_cost,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def __str__(self) -> str:
        return (
            f"Usage(cost: ${self.completion_cost:.4f}, "
            f"completion tokens: {self.completion_tokens}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"cached tokens: {self.cached_tokens})"
        )

    @model_validator(mode='before')
    @classmethod
    def fix_null_tokens(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if not value:
                    data[key] = 0

        return data

class Completion(BaseModel):
    model: str
    input: list[dict] | None = None
    response: dict[str, Any] | None = None
    usage: Usage | None = None

    @classmethod
    def from_llm_completion(
        cls, input_messages: list[dict], completion_response: Any, model: str
    ) -> Optional["Completion"]:
        if isinstance(completion_response, BaseModel):
            response = completion_response.model_dump()
        elif isinstance(completion_response, dict):
            response = completion_response
        else:
            logger.error(
                f"Unexpected completion response type: {type(completion_response)}"
            )
            return None

        usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            usage=usage,
        )


class StructuredOutput(OpenAISchema):

    @classmethod
    def model_validate_json(
            cls,
            json_data: str | bytes | bytearray,
            **kwarg,
    ) -> Self:
        message = json_data
        logger.info(f"parse_json() Original message: {repr(message)}")

        # Clean control characters from the message, preserving tabs and newlines
        cleaned_message = ''.join(char for char in message if ord(char) >= 32 or char in '\n\r')
        if cleaned_message != message:
            logger.info(f"parse_json() Cleaned control chars: {repr(message)} -> {repr(cleaned_message)}")
        message = cleaned_message

        # Extract JSON using the new function
        message, all_jsons = extract_json_from_message(message)
        if len(all_jsons) > 1:
            logger.warning(f"Found multiple JSON objects, using the first one. All found: {all_jsons}")
        if all_jsons:
            logger.info(f"parse_json() Extracted JSON: {repr(message)}")

        # Normalize line endings to \n
        if isinstance(message, str):
            message = message.replace('\r\n', '\n').replace('\r', '\n')

        logger.debug(f"parse_json() Final message to validate: {repr(message)}")

        __tracebackhide__ = True
        return super().model_validate_json(
            message if isinstance(message, str) else json.dumps(message),
            **kwarg
        )

def extract_json_from_message(message: str) -> tuple[dict | str, list[dict]]:
    """
    Extract JSON from a message, handling both code blocks and raw JSON.
    Returns a tuple of (selected_json_dict, all_found_json_dicts).

    Args:
        message: The message to parse

    Returns:
        tuple[dict | str, list[dict]]: (The selected JSON dict to use or original message, List of all JSON dicts found)
    """
    all_found_jsons = []

    # First try to find ```json blocks
    try:
        current_pos = 0
        while True:
            start = message.find("```json", current_pos)
            if start == -1:
                break
            start += 7  # Move past ```json
            end = message.find("```", start)
            if end == -1:
                break
            potential_json = message[start:end].strip()
            try:
                json_dict = json.loads(potential_json)
                all_found_jsons.append(json_dict)
            except json.JSONDecodeError:
                pass
            current_pos = end + 3

        if all_found_jsons:
            return all_found_jsons[0], all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract JSON from code blocks: {e}")

    # If no ```json blocks found, try to find raw JSON objects
    try:
        current_pos = 0
        while True:
            start = message.find("{", current_pos)
            if start == -1:
                break
            # Try to parse JSON starting from each { found
            for end in range(len(message), start, -1):
                try:
                    potential_json = message[start:end]
                    json_dict = json.loads(potential_json)
                    all_found_jsons.append(json_dict)
                    break
                except json.JSONDecodeError:
                    continue
            if not all_found_jsons:  # If no valid JSON found, move past this {
                current_pos = start + 1
            else:
                current_pos = end

        if all_found_jsons:
            return all_found_jsons[0], all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract raw JSON objects: {e}")

    return message, all_found_jsons
