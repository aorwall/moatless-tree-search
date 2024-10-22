import json
import logging
import os
import random
import string
from enum import Enum
from typing import Optional, Union, List, Tuple, Any

import instructor
import litellm
import openai
from anthropic import Anthropic, AnthropicBedrock, NOT_GIVEN
from anthropic.types import ToolUseBlock
from instructor import OpenAISchema
from instructor.exceptions import InstructorRetryException
from litellm.types.utils import ModelResponse
from openai import AzureOpenAI, OpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field, model_validator

from moatless.completion.model import Message, Completion

logger = logging.getLogger(__name__)


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    STRUCTURED_OUTPUT = "structured_output"


class CompletionModel(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        2000, description="The maximum number of tokens to generate"
    )
    model_base_url: Optional[str] = Field(
        default=None, description="The base URL for the model API"
    )
    model_api_key: Optional[str] = Field(
        default=None, description="The API key for the model"
    )
    response_format: LLMResponseFormat = Field(
        LLMResponseFormat.TOOLS, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )

    @model_validator(mode="after")
    def validate_response_format(self):
        if self.model.startswith("claude") or self.model.startswith("anthropic"):
            self.response_format = LLMResponseFormat.ANTHROPIC_TOOLS
        elif self.model in [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
        ]:
            self.response_format = LLMResponseFormat.STRUCTURED_OUTPUT
        elif self.model == "deepseek/deepseek-chat":
            self.response_format = LLMResponseFormat.JSON
        else:
            try:
                support_function_calling = litellm.supports_function_calling(
                    model=self.model
                )
            except Exception as e:
                support_function_calling = False

            if not support_function_calling:
                logger.debug(
                    f"The model {self.model} doens't support function calling, set response format to JSON"
                )
                self.response_format = LLMResponseFormat.JSON

        return self

    def create_completion(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[type[OpenAISchema]] | None = None,
    ) -> Tuple[OpenAISchema, Completion]:
        completion_messages = self._map_completion_messages(messages)

        if self.response_format != LLMResponseFormat.ANTHROPIC_TOOLS:
            completion_messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            if self.response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
                action_args, completion_response = self._anthropic_completion(
                    completion_messages, system_prompt, actions
                )
            elif self.response_format == LLMResponseFormat.STRUCTURED_OUTPUT:
                action_args, completion_response = self._openai_completion(
                    completion_messages, actions
                )
            elif self.response_format == LLMResponseFormat.TOOLS:
                action_args, completion_response = self._litellm_tool_completion(
                    completion_messages, actions
                )
            else:
                action_args, completion_response = self._instructor_completion(
                    completion_messages, actions
                )
        except InstructorRetryException as e:
            logger.warning(
                f"Failed to get completion response from LLM. {e}\n\nCompletion: {e.last_completion}"
            )
            raise e
        except Exception as e:
            logger.warning(
                f"Failed to get completion response from LLM. {e}."
                f"Input messages:\n {json.dumps(completion_messages, indent=2)}"
            )
            raise e

        completion = Completion.from_llm_completion(
            input_messages=completion_messages,
            completion_response=completion_response,
            model=self.model,
        )

        return action_args, completion

    def _litellm_tool_completion(
        self,
        messages: list[dict],
        actions: list[type[OpenAISchema]],
        is_retry: bool = False,
    ) -> Tuple[OpenAISchema, ModelResponse]:
        litellm.drop_params = True

        tools = []
        for action in actions:
            tools.append(openai.pydantic_function_tool(action))

        completion_response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            base_url=self.model_base_url,
            api_key=self.model_api_key,
            stop=self.stop_words,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        tool_args, tool_name, retry_message = None, None, None
        if (
            not completion_response.choices[0].message.tool_calls
            and completion_response.choices[0].message.content
        ):
            if "```json" in completion_response.choices[0].message.content:
                logger.info(
                    f"Found no tool call but JSON in completion response, will try to parse"
                )

                try:
                    action_request = self.action_type().from_response(
                        completion_response, mode=instructor.Mode.TOOLS
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse JSON as tool call, will try to parse as JSON "
                    )

                    try:
                        action_request = self.action_type().from_response(
                            completion_response, mode=instructor.Mode.JSON
                        )
                    except Exception as e:
                        logger.exception(
                            f"Failed to parse JSON as tool call from completion response: {completion_response.choices[0].message.content}"
                        )
                        raise e

                return action_request, completion_response
            elif completion_response.choices[0].message.content.startswith("{"):
                tool_args = json.loads(completion_response.choices[0].message.content)

            if tool_args:
                if "name" in tool_args:
                    tool_name = tool_args.get("name")

                if "parameters" in tool_args:
                    tool_args = tool_args["parameters"]

        elif completion_response.choices[0].message.tool_calls[0]:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_args = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name

        if not tool_args:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}"
                )
                raise ValueError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            )
            if not retry_message:
                retry_message = "You must response with a tool call."
            messages.append({"role": "user", "content": retry_message})
            return self._litellm_tool_completion(messages, is_retry=True)

        action_request = self.action_type().from_tool_call(
            tool_args=tool_args, tool_name=tool_name
        )
        return action_request, completion_response

    def create_text_completion(self, messages: List[Message], system_prompt: str):
        completion_messages = self._map_completion_messages(messages)

        if self.response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
            response, completion_response = self._anthropic_completion(
                completion_messages, system_prompt
            )
        else:
            completion_messages.insert(0, {"role": "system", "content": system_prompt})
            response, completion_response = self._litellm_text_completion(
                completion_messages
            )

        completion = Completion.from_llm_completion(
            input_messages=completion_messages,
            completion_response=completion_response,
            model=self.model,
        )

        return response, completion

    def input_messages(
        self, content: str, completion: Completion | None, feedback: str | None = None
    ):
        messages = []
        tool_call_id = None

        if completion:
            messages = completion.input

            response_message = completion.response["choices"][0]["message"]
            if response_message.get("tool_calls"):
                tool_call_id = response_message.get("tool_calls")[0]["id"]
                last_response = {
                    "role": response_message["role"],
                    "tool_calls": response_message["tool_calls"],
                }
            else:
                last_response = {
                    "role": response_message["role"],
                    "content": response_message["content"],
                }
            messages.append(last_response)

            if response_message.get("tool_calls"):
                tool_call_id = response_message.get("tool_calls")[0]["id"]

        if tool_call_id:
            new_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        else:
            new_message = {
                "role": "user",
                "content": content,
            }

        if feedback:
            new_message["content"] += "\n\n" + feedback

        messages.append(new_message)
        return messages

    def _litellm_text_completion(
        self, messages: list[dict]
    ) -> Tuple[str, ModelResponse]:
        litellm.drop_params = True

        completion_response = litellm.completion(
            model=self.model,
            base_url=self.model_base_url,
            api_key=self.model_api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words,
            messages=messages,
        )
        return completion_response.choices[0].message.content, completion_response

    def _instructor_completion(
        self,
        messages: list[dict],
        actions: List[type[OpenAISchema]] | None = None,
        is_retry: bool = False,
    ) -> Tuple[OpenAISchema, ModelResponse]:
        if self.response_format == LLMResponseFormat.JSON:
            client = instructor.from_litellm(
                litellm.completion, mode=instructor.Mode.MD_JSON
            )
        else:
            client = instructor.from_litellm(
                litellm.completion, mode=instructor.Mode.TOOLS
            )

        if actions:

            class TakeAction(OpenAISchema):
                action: Union[tuple(actions)] = Field(...)

                class Config:
                    smart_union = True

            action_type = TakeAction
        else:
            action_type = None

        try:
            take_action, completion_response = (
                client.chat.completions.create_with_completion(
                    model=self.model,
                    base_url=self.model_base_url,
                    api_key=self.model_api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    response_model=action_type,
                )
            )

            if take_action.action:
                return take_action.action, completion_response
            else:
                logger.warning(
                    f"No action returned in response: {take_action}. Completion response: {completion_response}"
                )
                raise ValueError(f"No action returned in response: {take_action}.")

        except InstructorRetryException as e:
            logger.error(json.dumps(e.messages, indent=2))
            raise e
        except Exception as e:
            logger.exception(f"Failed to get completion response from litellm: {e}")
            raise e

    def function_call_system_prompt(self):
        return """You are an AI language model tasked with transforming unstructured messages wrapped in the XML tag <message> into structured tool calls. Your guidelines are:

         * Do not change, omit, or add any information from the original message.
         * Focus solely on converting the existing information into the correct tool call format.
         * Extract all relevant details necessary for the tool call without altering their meaning.
         * Ignore planned steps in the tool call
         * Provide the reasoning in the scratch_pad field

        Your response should be the tool call generated from the provided unstructured message, adhering strictly to these instructions."""

    def function_call_prompt(self, llm_response: str):
        content = "<message>\n"
        content += llm_response
        content += "\n</message>"
        return content

    def _openai_completion(
        self,
        messages: list[dict],
        actions: List[type[OpenAISchema]] | None = None,
        response_format: type[OpenAISchema] | None = None,
        is_retry: bool = False,
    ):
        if os.getenv("AZURE_API_KEY"):
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version="2024-07-01-preview",
                azure_endpoint=os.getenv("AZURE_API_BASE"),
            )
        else:
            client = OpenAI()

        tools = []
        if actions:
            for action in actions:
                schema = action.openai_schema
                tools.append(openai.pydantic_function_tool(action, name=schema["name"], description=schema["description"]))

        try:
            if actions:
                completion_response = client.beta.chat.completions.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    tool_choice="required",
                    tools=tools,
                    parallel_tool_calls=False,
                )
            else:
                completion_response = client.beta.chat.completions.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    response_format=response_format,
                )
        except LengthFinishReasonError as e:
            logger.error(
                f"Failed to parse completion response. Completion: {e.completion.model_dump_json(indent=2)}"
            )
            from moatless.actions.reject import Reject

            # TODO: Raise exception instead?
            return Reject(
                rejection_reason=f"Failed to generate action: {e}"
            ), e.completion

        if not actions:
            response = completion_response.choices[0].message.parsed
            return response, completion_response

        elif not completion_response.choices[0].message.tool_calls:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}"
                )
                raise RuntimeError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            )
            messages.append(
                {"role": "user", "content": "You must response with a tool call."}
            )
            return self._openai_completion(messages, actions, response_format, is_retry)
        else:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            action_request = tool_call.function.parsed_arguments
            return action_request, completion_response

    def _anthropic_completion(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        actions: List[type[OpenAISchema]] | None = None,
    ) -> Tuple[OpenAISchema | str, Any]:
        if self.model.startswith("anthropic"):
            anthropic_client = AnthropicBedrock()
        else:
            anthropic_client = Anthropic()

        if actions:
            tools = []
            tool_choice = {"type": "any"}
            for action in actions:
                tools.append(action.anthropic_schema)
        else:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN

        completion_response = anthropic_client.beta.prompt_caching.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[
                {
                    "text": system_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tool_choice=tool_choice,
            tools=tools,
            messages=messages,
        )

        try:
            if not actions:
                return completion_response.content[0].text, completion_response
            for block in completion_response.content:
                if isinstance(block, ToolUseBlock):
                    action = None
                    for check_action in actions:
                        logger.info(f"Checking action {check_action.name} == {block.name}")
                        if check_action.name == block.name:
                            action = check_action
                            break

                    if not action:
                        raise ValueError(f"Unknown action {block.name}")

                    action_args = action.model_validate(block.input)

                    # TODO: We only support one action at the moment
                    return action_args, completion_response

                else:
                    logger.warning(f"Unexpected block {block}]")

        except Exception as e:
            logger.exception(
                f"Failed to parse action request from completion response. Completion: {completion_response}"
            )
            raise e

    def _map_completion_messages(self, messages: list[Message]) -> list[dict]:
        tool_call_id = None
        completion_messages = []
        for message in messages:
            if message.role == "user":
                if tool_call_id and self.response_format in [
                    LLMResponseFormat.TOOLS,
                    LLMResponseFormat.STRUCTURED_OUTPUT,
                ]:
                    completion_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                elif tool_call_id and self.response_format in [
                    LLMResponseFormat.TOOLS,
                    LLMResponseFormat.STRUCTURED_OUTPUT,
                ]:
                    completion_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "tool_use_id": tool_call_id,
                                    "content": message.content,
                                    "type": "tool_result",
                                }
                            ],
                        }
                    )
                elif tool_call_id and self.response_format in [
                    LLMResponseFormat.ANTHROPIC_TOOLS,
                ]:
                    completion_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "tool_use_id": tool_call_id,
                                    "content": message.content,
                                    "type": "tool_result",
                                    # FIME "cache_control": {"type": "ephemeral"}
                                }
                            ],
                        }
                    )
                else:
                    completion_messages.append(
                        {"role": "user", "content": message.content}
                    )
            elif message.role == "assistant":
                if message.tool_call:
                    tool_call_id = generate_call_id()
                    if self.response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
                        completion_messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "id": tool_call_id,
                                        "input": message.tool_call.input,
                                        "type": "tool_use",
                                        "name": message.tool_call.name,
                                    }
                                ],
                            }
                        )
                    elif self.response_format in [
                        LLMResponseFormat.TOOLS,
                        LLMResponseFormat.STRUCTURED_OUTPUT,
                    ]:
                        completion_messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": message.tool_call.name,
                                            "arguments": json.dumps(
                                                message.tool_call.input
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                    else:
                        json_content = json.dumps(message.tool_call.input, indent=2)

                        # TODO Only if self.model.startswith("deepseek"): ?
                        json_content = f"```json\n{json_content}\n```"

                        completion_messages.append(
                            {
                                "role": "assistant",
                                "content": json_content,
                            }
                        )

                else:
                    tool_call_id = None
                    completion_messages.append(
                        {"role": "assistant", "content": message.content}
                    )

        return completion_messages

    def _get_tool_call(self, completion_response) -> Tuple[str, dict]:
        if (
            not completion_response.choices[0].message.tool_calls
            and completion_response.choices[0].message.content
        ):
            if "```json" in completion_response.choices[0].message.content:
                content = completion_response.choices[0].message.content
                json_start = content.index("```json") + 7
                json_end = content.rindex("```")
                json_content = content[json_start:json_end].strip()
            elif completion_response.choices[0].message.content.startswith("{"):
                json_content = completion_response.choices[0].message.content
            else:
                return None, None

            tool_call = json.loads(json_content)
            return tool_call.get("name"), tool_call

        elif completion_response.choices[0].message.tool_calls:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_dict = json.loads(tool_call.function.arguments)
            return tool_call.function.name, tool_dict

        return None

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
        return dump

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "response_format" in obj:
            obj["response_format"] = LLMResponseFormat(obj["response_format"])
        return super().model_validate(obj)


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
