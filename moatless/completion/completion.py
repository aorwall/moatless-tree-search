import json
import logging
import os
import random
import string
from enum import Enum
from typing import Optional, Union, List, Tuple, Any, Self

import anthropic
import instructor
from instructor.utils import extract_json_from_codeblock
from jsonschema import ValidationError
import litellm
import openai
import tenacity
from anthropic import Anthropic, AnthropicBedrock, NOT_GIVEN
from anthropic.types import ToolUseBlock, TextBlock
from anthropic.types.beta import BetaToolUseBlock, BetaTextBlock, BetaMessageParam, BetaCacheControlEphemeralParam
from instructor import OpenAISchema
from instructor.exceptions import InstructorRetryException
from litellm.types.utils import ModelResponse
from openai import AzureOpenAI, OpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field, model_validator
from litellm.exceptions import BadRequestError, NotFoundError, AuthenticationError, APIError
from moatless.completion.model import Message, Completion
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)

class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    STRUCTURED = "structured_output"


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
        # Always use JSON response format for deepseek chat as it isn't reliable with tools
        if self.model == "deepseek/deepseek-chat":
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

    @property
    def supports_anthropic_prompt_caching(self):
        return self.model.startswith("claude-3-5-")

    @property
    def supports_anthropic_computer_use(self):
        return "claude-3-5-sonnet-20241022" in self.model

    @property
    def use_anthropic_client(self):
        """Skip LiteLLM and use Anthropic's client for beta features"""
        return "claude-3-5" in self.model

    @property
    def use_openai_client(self):
        """Skip LiteLLm and use OpenAI's client for beta features"""
        return self.model in  [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
        ]

    def create_completion(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[type[OpenAISchema]] | None = None,
    ) -> Tuple[OpenAISchema, Completion]:
        if not system_prompt:
            raise ValueError("System prompt is required")

        completion_messages = self._map_completion_messages(messages)

        if not self.supports_anthropic_computer_use and not self.supports_anthropic_prompt_caching:
            completion_messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            if self.use_anthropic_client:
                action_args, completion_response = self._anthropic_completion(
                    completion_messages, system_prompt, actions
                )
            elif self.use_openai_client:
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
                f"Instructor failed after {e.n_attempts} attempts. Last completion: {e.last_completion}. Messages: {e.messages}")
            raise CompletionRejectError(
                f"Failed to parse response. Did {e.n_attempts} attempts. Error: {e}",
                last_completion=e.last_completion,
                messages=e.messages
            )
        except CompletionRejectError as e:
            raise e
        except Exception as e:
            if isinstance(e, APIError):
                logger.error(
                    f"Request failed. self.model: {self.model}, base_url: {self.model_base_url}. Model: {e.model}, Provider {e.llm_provider}. Litellm {e.litellm_debug_info}. Exception {e.message}"
                )
            else:
                logger.error(f"Failed to get completion response from litellm: {e}")
            raise CompletionRuntimeError(
                f"Failed to get completion response: {e}",
            ) from e

        completion = Completion.from_llm_completion(
            input_messages=completion_messages,
            completion_response=completion_response,
            model=self.model,
        )

        if "stop_reason" in completion.response and completion.response["stop_reason"] == "max_tokens":
            raise CompletionRejectError(
                f"Max tokens reached in completion response",
                last_completion=completion_response
            )

        return action_args, completion

    def create_text_completion(self, messages: List[Message], system_prompt: str):
        completion_messages = self._map_completion_messages(messages)

        if self.supports_anthropic_computer_use or self.supports_anthropic_prompt_caching:
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

    def create_completion_with_response_model(
        self,
        messages: list[Message],
        system_prompt: str,
        response_model: type[OpenAISchema]
    ) -> Tuple[OpenAISchema, ModelResponse]:
        if not self.response_format == LLMResponseFormat.JSON:
            return self.create_completion(messages, system_prompt, actions=[response_model])

        completion_messages = self._map_completion_messages(messages)
        completion_messages.insert(0, {"role": "system", "content": system_prompt})

        client = instructor.from_litellm(
            litellm.completion, mode=instructor.Mode.JSON
        )

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type((APIError, BadRequestError, NotFoundError, AuthenticationError)),
            stop=tenacity.stop_after_attempt(3),
            #before=lambda x: logger.info(x),
            #after=lambda x: logger.info(x),
        )
        response, completion_response = (
            client.chat.completions.create_with_completion(
                model=self.model,
                api_base=self.model_base_url,
                api_key=self.model_api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words,
                messages=completion_messages,
                response_model=response_model,
                max_retries=retries,  # type: ignore
            )
        )

        completion = Completion.from_llm_completion(
            input_messages=completion_messages,
            completion_response=completion_response,
            model=self.model,
        )

        return response, completion

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
            api_base=self.model_base_url,
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
            api_base=self.model_base_url,
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
                action_type: str = Field(..., description="The type of action being taken")

                @model_validator(mode='before')
                def validate_action(cls, data: dict) -> dict:
                    action_type = data.get('action_type')
                    if not action_type:
                        return data

                    if len(actions) == 1:
                        data['action'] = actions[0].model_validate(data['action'])
                    else:
                        avalabile_actions = [action for action in actions if hasattr(action, "name")]

                        if not avalabile_actions:
                            raise CompletionRuntimeError(f"No actions found in {actions}")

                        # Find the correct action class based on action_type
                        action_class = next(
                            (action for action in avalabile_actions if action.name == action_type),
                            None
                        )
                        if not action_class:
                            action_names = [action.name for action in avalabile_actions]
                            raise ValidationError(f"Unknown action type: {action_type}. Available actions: {', '.join(action_names)}")

                        # Validate the action data using the specific action class
                        data['action'] = action_class.model_validate(data['action'])
                    return data

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

                    try:
                        # Look specifically for json if "json" in message:
                        start = message.index("json") + 7 # Find the next closing
                        end = message.find("```", start)
                        if end != -1:
                            json_message = message[start:end].strip()
                            logger.info(f"parse_json() Extracted JSON from codeblock: {repr(json_message)}")
                            message = json_message
                    except Exception as e:
                        logger.warning(f"Failed to extract JSON from message: {e}")

                    # Extract JSON from codeblock if present
                    json_message = extract_json_from_codeblock(message)
                    if json_message != message:
                        logger.info(f"parse_json() Extracted JSON: {repr(json_message)}")
                    message = json_message

                    # Normalize line endings to \n
                    message = message.replace('\r\n', '\n').replace('\r', '\n')

                    logger.debug(f"parse_json() Final message to validate: {repr(message)}")

                    __tracebackhide__ = True
                    return super().model_validate_json(
                        message,
                        **kwarg
                    )

                #class Config:
                #    smart_union = True

            action_type = TakeAction
        else:
            action_type = None

        def log_retry(state: tenacity.RetryCallState):
            if state.attempt_number > 1:
                logger.warning(state)

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type((APIError, BadRequestError, NotFoundError, AuthenticationError)),
            stop=tenacity.stop_after_attempt(3),
            after=log_retry,
        )
        take_action, completion_response = (
            client.chat.completions.create_with_completion(
                model=self.model,
                api_base=self.model_base_url,
                api_key=self.model_api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words,
                messages=messages,
                response_model=action_type,
                max_retries=retries,  # type: ignore
            )
        )

        if take_action.action:
            return take_action.action, completion_response
        else:
            logger.warning(
                f"No action returned in response: {take_action}. Completion response: {completion_response}"
            )
            raise CompletionRejectError(
                f"No action returned in response: {take_action}.",
                last_completion=completion_response
            )

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
                tools.append(
                    openai.pydantic_function_tool(
                        action, name=schema["name"], description=schema["description"]
                    )
                )

        try:
            if actions:
                completion_response = client.beta.chat.completions.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words,
                    messages=messages,
                    #tool_choice="required",
                    tools=tools,
                    parallel_tool_calls=True,
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

        if actions:
            tools = []
            tool_choice = {"type": "any"}
            for action in actions:
                if hasattr(action, "name") and action.name == "str_replace_editor":
                    tools.append({
                        "name": "str_replace_editor",
                        "type": "text_editor_20241022"
                    })
                else:
                    schema = action.anthropic_schema

                    # Remove scratch pad field, use regular text block for thoughts
                    if "scratch_pad" in schema["input_schema"]["properties"]:
                        del schema["input_schema"]["properties"]["scratch_pad"]

                    tools.append(schema)

        else:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN

        system_message = {
            "text": system_prompt,
            "type": "text"
        }

        if "anthropic" in self.model:
            anthropic_client = AnthropicBedrock()
            betas = ["computer-use-2024-10-22"]
        else:
            anthropic_client = Anthropic()
            betas = ["computer-use-2024-10-22", "prompt-caching-2024-07-31"]
            _inject_prompt_caching(messages)
            system_message["cache_control"] = {"type": "ephemeral"}

        completion_response = None
        retry_message = None
        for i in range(2):
            if i > 0:
                logger.warning(f"Retrying completion request: {retry_message} (attempt {i})")

            try:
                completion_response = anthropic_client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[system_message],
                    # tool_choice=tool_choice,
                    tools=tools,
                    messages=messages,
                    betas=betas
                )
            except anthropic.BadRequestError as e:
                logger.error(
                    f"Failed to create completion: {e}. Input messages: {json.dumps(messages, indent=2)}"
                )
                raise CompletionRuntimeError(f"Failed to create completion: {e}") from e

            try:
                text = None
                if not actions:
                    return completion_response.content[0].text, completion_response
                for block in completion_response.content:
                    if isinstance(block, ToolUseBlock) or isinstance(block, BetaToolUseBlock):
                        action = None
                        for check_action in actions:
                            if check_action.openai_schema["name"] == block.name:
                                action = check_action
                                break

                        if not action:
                            raise ValueError(f"Unknown action {block.name}")

                        action_args = action.model_validate(block.input)

                        if hasattr(action_args, "scratch_pad") and text and not action_args.scratch_pad:
                            action_args.scratch_pad = text

                        # TODO: We only support one action at the moment
                        return action_args, completion_response
                    elif isinstance(block, TextBlock) or isinstance(block, BetaTextBlock):
                        text = block.text
                    else:
                        logger.warning(f"Unexpected block {block}]")

                retry_message = f"You're an autonomous agent that can't communicate with the user. Please provide a tool call."
            except anthropic.APIError as e:
                if hasattr(e, "status_code"):
                    raise CompletionRuntimeError(f"Failed to call Anthropic API. Status code: {e.status_code}, Response: {e.body}") from e
                else:
                    raise CompletionRuntimeError(f"Failed to call Anthropic API. {e}") from e
            except Exception as e:
                retry_message = f"The request was invalid. Please try again. Error: {e}"

            messages.append(
                {
                    "role": "assistant",
                    "content": completion_response.content
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": retry_message
                }
            )

        if not text:
            text = f"No action found in completion response"

        raise CompletionRejectError(text, last_completion=completion_response)

    def _map_completion_messages(self, messages: list[Message]) -> list[dict]:
        tool_call_id = None
        completion_messages = []
        for i, message in enumerate(messages):
            if message.role == "user":
                if tool_call_id and self.use_anthropic_client:
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
                    tool_call_id = None
                elif tool_call_id and self.response_format in [
                    LLMResponseFormat.TOOLS
                ]:
                    completion_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                    tool_call_id = None
                elif tool_call_id and self.response_format in [
                    LLMResponseFormat.TOOLS
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
                    tool_call_id = None
                else:
                    completion_messages.append(
                        {"role": "user", "content": message.content}
                    )
            elif message.role == "assistant":
                if message.tool_call:
                    tool_call_id = f"call_{i}"
                    content = []
                    if self.use_anthropic_client:
                        tool_input = message.tool_call.input.copy()

                        # Scratch pad is provided as a message instead of part of the tool call
                        if "scratch_pad" in message.tool_call.input:
                            scratch_pad = tool_input["scratch_pad"]
                            del tool_input["scratch_pad"]
                            if scratch_pad:
                                content.append(
                                    {
                                        "type": "text",
                                        "text": scratch_pad,
                                    }
                                )

                        content.append(
                            {
                                "id": tool_call_id,
                                "input": tool_input,
                                "type": "tool_use",
                                "name": message.tool_call.name,
                            }
                        )
                        completion_messages.append(
                            {
                                "role": "assistant",
                                "content": content
                            }
                        )
                    elif self.response_format in [
                        LLMResponseFormat.TOOLS,
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
                        action_json = {
                            "action": message.tool_call.input,
                            "action_type": message.tool_call.name
                        }
                        json_content = json.dumps(action_json, indent=2)

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

    @model_validator(mode="after")
    def set_api_key(self) -> "CompletionModel":
        """
        Update the model with the API key from en vars if model base URL is set but API key is not as we don't persist the API key.
        """
        if self.model_base_url and not self.model_api_key:
            self.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")

        return self



def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        # message["role"] == "user" and
        if isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break
