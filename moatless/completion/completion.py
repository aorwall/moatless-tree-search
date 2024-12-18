import json
import logging
import os
from enum import Enum
from textwrap import dedent
from typing import Optional, Union, List, Tuple, Any

import anthropic
import instructor
import litellm
import openai
import tenacity
from anthropic import Anthropic, AnthropicBedrock, NOT_GIVEN
from anthropic.types import ToolUseBlock, TextBlock
from anthropic.types.beta import (
    BetaToolUseBlock,
    BetaTextBlock,
    BetaMessageParam,
    BetaCacheControlEphemeralParam,
)
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.types.llms.anthropic import AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
from litellm.types.utils import ModelResponse
from openai import AzureOpenAI, OpenAI, LengthFinishReasonError
from pyarrow import system_memory_pool
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Message, Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError, CompletionError

logger = logging.getLogger(__name__)


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"

class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""
    structured_outputs: List[StructuredOutput] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)

    @classmethod
    def create(cls,
               text: str | None = None,
               output: List[StructuredOutput] | StructuredOutput | None = None,
               completion: Completion | None = None) -> 'CompletionResponse':
        if isinstance(output, StructuredOutput):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = None

        return cls(text_response=text, structured_outputs=outputs, completion=completion)

    @property
    def structured_output(self) -> Optional[StructuredOutput]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 0:
            logger.warning("Multiple structured outputs found in completion response, returning the first one")
        return self.structured_outputs[0] if self.structured_outputs else None


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
        default=None, 
        description="The API key for the model",
        exclude=True
    )
    response_format: Optional[LLMResponseFormat] = Field(
        None, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Additional metadata for the completion model"
    )

    @model_validator(mode="after")
    def validate_response_format(self):
        if not self.response_format:
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
                else:
                    self.response_format = LLMResponseFormat.TOOLS
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
        return self.model in [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
        ]

    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]] | type[StructuredOutput]
    ) -> CompletionResponse:
        if not system_prompt:
            raise ValueError("System prompt is required")

        completion_messages = messages
        completion_response = None
        try:
            if self.use_anthropic_client:
                return self._anthropic_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.response_format == LLMResponseFormat.REACT and isinstance(
                response_model, list
            ):
                return self._litellm_react_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.response_format == LLMResponseFormat.TOOLS:
                return self._litellm_tool_completion(
                    completion_messages, system_prompt, response_model
                )
            else:
                action_args, completion_response = self._litellm_completion(
                    completion_messages, system_prompt, response_model
                )
        except CompletionError as e:
            raise e
        except Exception as e:
            if isinstance(e, APIError):
                logger.exception(
                    f"Request failed.{e.litellm_debug_info}. Input messages: {json.dumps(messages, indent=2)}"
                )
                if e.status_code >= 500:
                    raise CompletionRejectError(
                        f"Failed to create completion: {e}",
                        messages=completion_messages,
                        last_completion=completion_response,
                    ) from e

            else:
                logger.exception(f"Failed to create completion. Input messages: {json.dumps(messages, indent=2)}")

            raise CompletionRuntimeError(
                f"Failed to get completion response: {e}",
                messages=completion_messages,
                last_completion=completion_response,
            ) from e

        if completion_response:
            completion = Completion.from_llm_completion(
                input_messages=completion_messages,
                completion_response=completion_response,
                model=self.model,
            )
        else:
            completion = None

        if (
            "stop_reason" in completion.response
            and completion.response["stop_reason"] == "max_tokens"
        ):
            raise CompletionRejectError(
                f"Max tokens reached in completion response",
                messages=completion_messages,
                last_completion=completion_response,
            )

        raise RuntimeError("Shouldnt reach this point")

    def _litellm_tool_completion(
            self,
            messages: list[dict],
            system_prompt: str,
            response_model: type[StructuredOutput] | List[type[StructuredOutput]] | None,
    ) -> CompletionResponse:
        litellm.drop_params = True
        messages.insert(0, {"role": "system", "content": system_prompt})

        if isinstance(response_model, list):
            tools = [r.openai_schema(thoughts_in_action=True) for r in response_model]
        elif response_model:
            tools = [response_model.openai_schema()]
        else:
            tools = None

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            llm_completion_response = None
            try:
                llm_completion_response = litellm.completion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    api_base=self.model_base_url,
                    api_key=self.model_api_key,
                    stop=self.stop_words,
                    tools=tools,
                    tool_choice="required",
                    messages=messages,
                    metadata=self.metadata or {},
                )

                logger.info(json.dumps(messages, indent=2))
                logger.info(json.dumps(llm_completion_response.model_dump(), indent=2))
                if not llm_completion_response or not llm_completion_response.choices:
                    raise CompletionRuntimeError("No completion response or choices returned")

                content = llm_completion_response.choices[0].message.content

                def get_response_model(tool_name: str):
                    if isinstance(response_model, list):
                        for r in response_model:
                            if r.name == tool_name:
                                return r
                    else:
                        return response_model

                response_objects = []

                invalid_function_names = []
                if llm_completion_response.choices[0].message.tool_calls:
                    for tool_call in llm_completion_response.choices[0].message.tool_calls:
                        tool_args = json.loads(tool_call.function.arguments)
                        action = get_response_model(tool_call.function.name)

                        if not action:
                            logger.warning(f"Invalid action name: {tool_call.function.name}")
                            invalid_function_names.append(tool_call.function.name)
                            continue

                        response_object = action.model_validate(tool_args)
                        response_objects.append(response_object)

                    if invalid_function_names:
                        available_actions = [r.name for r in response_model]
                        raise ValueError(f"Unknown functions {invalid_function_names}. Available functions: {available_actions}")

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=llm_completion_response,
                    model=self.model,
                )

                return CompletionResponse.create(text=content, output=response_objects, completion=completion)

            except ValidationError as e:
                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. Fix the errors, exceptions found\n{e}",
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=llm_completion_response,
                    messages=messages,
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        structured_output: type[StructuredOutput] | list[type[StructuredOutput]],
    ) -> CompletionResponse:
        if not structured_output:
            raise CompletionRuntimeError(f"Response model is required for completion")

        if isinstance(structured_output, list) and len(structured_output) > 1:
            avalabile_actions = [
                action for action in structured_output if hasattr(action, "name")
            ]
            if not avalabile_actions:
                raise CompletionRuntimeError(f"No actions found in {structured_output}")

            class TakeAction(StructuredOutput):
                action: Union[tuple(structured_output)] = Field(...)
                action_type: str = Field(
                    ..., description="The type of action being taken"
                )

                @model_validator(mode="before")
                def validate_action(cls, data: dict) -> dict:
                    if not isinstance(data, dict):
                        raise ValidationError("Expected dictionary input")
                        
                    action_type = data.get("action_type")
                    if not action_type:
                        return data

                    # Find the correct action class based on action_type
                    action_class = next(
                        (
                            action
                            for action in avalabile_actions
                            if action.name == action_type
                        ),
                        None,
                    )
                    if not action_class:
                        action_names = [action.name for action in avalabile_actions]
                        raise ValidationError(
                            f"Unknown action type: {action_type}. Available actions: {', '.join(action_names)}"
                        )

                    # Validate the action data using the specific action class
                    action_data = data.get("action")
                    if not action_data:
                        raise ValidationError("Action data is required")
                        
                    data["action"] = action_class.model_validate(action_data)
                    return data

            response_model = TakeAction
        else:
            response_model = structured_output

        system_prompt += dedent(f"""\n# Response format
You must respond with only a JSON object that match the following json_schema:\n

{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself.""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            completion_response = None

            try:
                completion_response = litellm.completion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    api_base=self.model_base_url,
                    api_key=self.model_api_key,
                    stop=self.stop_words,
                    messages=messages,
                    response_format={"type": "json_object"},
                    metadata=self.metadata or {},
                )

                logger.info(json.dumps(messages, indent=2))
                logger.info(json.dumps(completion_response.model_dump(), indent=2))
                if not completion_response or not completion_response.choices:
                    raise CompletionRuntimeError("No completion response or choices returned")

                if isinstance(
                    completion_response.choices[0].message.content, BaseModel
                ):
                    assistant_message = completion_response.choices[
                        0
                    ].message.content.model_dump()
                else:
                    assistant_message = completion_response.choices[0].message.content

                if not assistant_message:
                    raise CompletionRuntimeError("Empty response from model")

                messages.append({"role": "assistant", "content": assistant_message})

                response = response_model.from_response(
                    completion_response, mode=instructor.Mode.JSON
                )

                if hasattr(response, "action"):
                    return response.action, completion_response

                return response, completion_response

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. Fix the errors, exceptions found\n{e}",
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                )
            except Exception as e:
                logger.exception(f"Completion attempt failed with error: {e}. Will retry.")
                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _validate_react_format(self, response_text: str):
        # Split into lines and remove empty ones
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Count occurrences of each section
        thought_count = sum(1 for line in lines if line.startswith("Thought:"))
        action_count = sum(1 for line in lines if line.startswith("Action:"))

        # Check for multiple action blocks
        if thought_count > 1 or action_count > 1:
            logger.warning(f"Multiple Thought or Action sections found in response: {response_text}")

        # Check if all sections exist
        if thought_count < 1 or action_count < 1:
            raise ValueError("Response must have one 'Thought:' and 'Action:' section")

        # Find the starting lines for each section
        thought_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Thought:")), -1
        )
        action_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Action:")), -1
        )

        # Check if sections are in correct order
        if not (thought_line < action_line):
            raise ValueError("Sections must be in order: Thought, Action")

    def _litellm_react_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        actions: list[type[StructuredOutput]],
    ) -> CompletionResponse:
        action_input_schemas = []

        for action in actions:
            action_input_schemas.append(f" * {action.name} {action.format_schema_for_llm()}")
            
        system_prompt += dedent(f"""\n# Response format

Use the following format:

Thought: You should always think about what to do
Action: The action to take followed by the input arguments based on the schema below

Use one of the following actions and provide input arguments matching the schema.
                            
{'\n\n'.join(action_input_schemas)}

Important: Do not include multiple Thought-Action blocks. Do not include code blocks or additional text outside of this format.
""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            response_text, completion_response = self._litellm_text_completion(messages)

            try:
                self._validate_react_format(response_text)

                thought_start = response_text.find("Thought:")
                action_start = response_text.find("Action:")

                if thought_start == -1 or action_start == -1:
                    raise ValueError("Missing Thought or Action sections")

                thought = response_text[thought_start + 8 : action_start].strip()
                action_input = response_text[action_start + 7:].strip()

                # Extract action name and input
                action_lines = action_input.split('\n', 1)
                if len(action_lines) < 2:
                    raise ValueError("Missing action name and input")

                action_name = action_lines[0].strip()
                action_input = action_lines[1].strip()

                # Find the matching action class
                action_class = next((a for a in actions if a.name == action_name), None)
                if not action_class:
                    action_names = [a.name for a in actions]
                    raise ValueError(
                        f"Unknown action: {action_name}. Available actions: {', '.join(action_names)}"
                    )

                # Check if input appears to be XML format
                if action_input.strip().startswith("<") or action_input.strip().startswith("```xml"):
                    try:
                        action_request = action_class.model_validate_xml(action_input)
                    except Exception as e:
                        format_example = action_class.format_schema_for_llm() if hasattr(action_class, 'format_schema_for_llm') else ""
                        raise ValueError(
                            f"Invalid XML format for {action_name}. Error: {e}\n\n"
                            f"Expected format:\n{format_example}"
                        )
                else:
                    # Otherwise, try to validate as JSON
                    try:
                        action_request = action_class.model_validate_json(action_input)
                    except Exception as e:
                        schema = action_class.model_json_schema()
                        if "thoughts" in schema["properties"]:
                            del schema["properties"]["thoughts"]
                        raise ValueError(
                            f"Invalid format for {action_name}. Error: {e}\n\n"
                            f"Expected JSON schema:\n{json.dumps(schema, indent=2)}"
                        )

                action_request.thoughts = thought
                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                )

                return CompletionResponse(
                    structured_outputs=[action_request],
                    completion=completion
                )

            except Exception as e:
                logger.warning(f"ReAct parsing failed: {e}. Response: {response_text}")
                messages.append({"role": "assistant", "content": response_text})

                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. {e}",
                    }
                )

                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_text_completion(self, messages: list[dict]) -> Tuple[str, ModelResponse]:
        litellm.drop_params = True

        completion_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "metadata": self.metadata or {},  # Always pass at least an empty dict
        }

        if self.model_base_url:
            completion_kwargs["api_base"] = self.model_base_url
        if self.model_api_key:
            completion_kwargs["api_key"] = self.model_api_key
        if self.stop_words:
            completion_kwargs["stop"] = self.stop_words

        completion_response = litellm.completion(**completion_kwargs)
        return completion_response.choices[0].message.content, completion_response

    def _anthropic_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        response_model: type[StructuredOutput]
        | List[type[StructuredOutput]]
        | None = None,
    ) -> CompletionResponse:
        tools = []
        tool_choice = {"type": "any"}

        actions = []
        if not response_model:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN
        else:
            if isinstance(response_model, list):
                actions = response_model
            elif response_model:
                actions = [response_model]

            for action in actions:
                if hasattr(action, "name") and action.name == "str_replace_editor":
                    tools.append(
                        {"name": "str_replace_editor", "type": "text_editor_20241022"}
                    )
                else:
                    schema = action.anthropic_schema()

                    # Remove scratch pad field, use regular text block for thoughts
                    if "thoughts" in schema["input_schema"]["properties"]:
                        del schema["input_schema"]["properties"]["thoughts"]

                    tools.append(schema)

        system_message = {"text": system_prompt, "type": "text"}

        anthropic_messages = anthropic_messages_pt(
            model=self.model,
            messages=messages,
            llm_provider="anthropic",
        )
        if "anthropic" in self.model:
            anthropic_client = AnthropicBedrock()
            betas = ["computer-use-2024-10-22"] #, "prompt-caching-2024-07-31"]
            extra_headers = { } #"X-Amzn-Bedrock-explicitPromptCaching": "enabled"}
        else:
            anthropic_client = Anthropic()
            extra_headers = {}
            betas = ["computer-use-2024-10-22", "prompt-caching-2024-07-31"]
            _inject_prompt_caching(anthropic_messages)
            system_message["cache_control"] = {"type": "ephemeral"}


        completion_response = None
        retry_message = None
        for i in range(2):
            if i > 0:
                logger.warning(
                    f"Retrying completion request: {retry_message} (attempt {i})"
                )

            try:
                completion_response = anthropic_client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[system_message],
                    # tool_choice=tool_choice,
                    tools=tools,
                    messages=anthropic_messages,
                    betas=betas,
                    extra_headers=extra_headers
                )
            except anthropic.BadRequestError as e:
                logger.exception(
                    f"Failed to create completion: {e}. Input messages: {json.dumps(messages, indent=2)}"
                )
                last_completion = (
                    completion_response.model_dump() if completion_response else None
                )
                raise CompletionRuntimeError(
                    f"Failed to create completion: {e}",
                    last_completion=last_completion,
                    messages=messages,
                ) from e

            def get_response_format(name: str):
                if len(actions) == 1:
                    return actions[0]
                else:
                    for check_action in actions:
                        if check_action.name == block.name:
                            return check_action
                return None

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )

            tool_call_id = None
            try:
                text = None

                if not actions:
                    return CompletionResponse(text=completion_response.content[0].text, completion_response=completion_response)

                for block in completion_response.content:
                    if isinstance(block, ToolUseBlock) or isinstance(
                        block, BetaToolUseBlock
                    ):
                        action = None

                        tool_call_id = block.id

                        if len(actions) == 1:
                            action = actions[0]
                        else:
                            for check_action in actions:
                                if check_action.name == block.name:
                                    action = check_action
                                    break

                        if not action:
                            raise ValueError(f"Unknown action {block.name}")

                        action_args = action.model_validate(block.input)

                        if (
                            hasattr(action_args, "thoughts")
                            and text
                            and not action_args.thoughts
                        ):
                            action_args.thoughts = text

                        # TODO: We only support one action at the moment
                        return CompletionResponse.create(text=text, output=[action_args], completion=completion)

                    elif isinstance(block, TextBlock) or isinstance(
                        block, BetaTextBlock
                    ):
                        text = block.text
                        # Extract thoughts from <thoughts> tags if present
                        if "<thoughts>" in text and "</thoughts>" in text:
                            start = text.index("<thoughts>") + len("<thoughts>")
                            end = text.index("</thoughts>")
                            text = text[start:end].strip()
                    else:
                        logger.warning(f"Unexpected block {block}]")

                logger.warning(f"No tool call found in completion response. Response text: {text}")
                retry_message = f"You're an autonomous agent that can't communicate with the user. Please provide a tool call."
            except anthropic.APIError as e:
                if hasattr(e, "status_code"):
                    raise CompletionRuntimeError(
                        f"Failed to call Anthropic API. Status code: {e.status_code}, Response: {e.body}"
                    ) from e
                else:
                    raise CompletionRuntimeError(
                        f"Failed to call Anthropic API. {e}"
                    ) from e
            except ValidationError as e:
                logger.exception(f"Failed to validate: {json.dumps(completion_response.model_dump(), indent=2)}")
                retry_message = f"The request was invalid. Please try again. Error: {e}"
            except Exception as e:
                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                    messages=messages,
                    last_completion=completion_response,
                ) from e

            response_content = [
                block.model_dump() for block in completion_response.content
            ]
            messages.append({"role": "assistant", "content": response_content})

            user_message = self._create_user_message(tool_call_id, retry_message)
            messages.append(user_message)

        raise CompletionRejectError(
            f"Failed to create completion: {retry_message}",
            messages=messages,
            last_completion=completion_response,
        )

    def _map_completion_messages(self, messages: list[Message]) -> list[dict]:
        tool_call_id = None
        completion_messages = []
        for i, message in enumerate(messages):
            if message.role == "user":
                user_message = self._create_user_message(tool_call_id, message.content)
                completion_messages.append(user_message)
                tool_call_id = None
            elif message.role == "assistant":
                if message.tool_call:
                    tool_call_id = message.tool_call_id
                    content = []
                    if self.use_anthropic_client:
                        tool_input = message.tool_call.input.copy()

                        # Scratch pad is provided as a message instead of part of the tool call
                        if "thoughts" in message.tool_call.input:
                            thoughts = tool_input["thoughts"]
                            del tool_input["thoughts"]
                            if thoughts:
                                content.append(
                                    {
                                        "type": "text",
                                        "text": f"<thoughts>\n{thoughts}\n</thoughts>",
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
                            {"role": "assistant", "content": content}
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
                            "action_type": message.tool_call.name,
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

    def _create_user_message(self, tool_call_id: str | None, content: str) -> dict:
        if tool_call_id and self.use_anthropic_client:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_call_id,
                        "content": f"<observation>\n{content}\n</observation>",
                        "type": "tool_result",
                    }
                ],
            }
        elif tool_call_id and self.response_format in [LLMResponseFormat.TOOLS]:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        elif tool_call_id and self.response_format in [LLMResponseFormat.TOOLS]:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_call_id,
                        "content": content,
                        "type": "tool_result",
                    }
                ],
            }

        else:
            return {"role": "user", "content": content}

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
    messages: list[Union[AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam]],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        # message["role"] == "user" and
        if isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break
