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
from litellm.types.utils import ModelResponse
from openai import AzureOpenAI, OpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Message, Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError, CompletionError

logger = logging.getLogger(__name__)


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


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
    response_format: LLMResponseFormat = Field(
        LLMResponseFormat.TOOLS, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Additional metadata for the completion model"
    )

    @model_validator(mode="after")
    def validate_response_format(self):
        if self.response_format == LLMResponseFormat.TOOLS:
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
        return "claude-3-5-" in self.model

    @property
    def supports_anthropic_computer_use(self):
        # Haiku doesn't support computer use
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
        messages: List[Message],
        system_prompt: str,
        response_model: List[type[StructuredOutput]]
        | type[StructuredOutput]
        | None = None,
    ) -> Tuple[StructuredOutput, Completion]:
        # Ensure messages is initialized as a list
        completion_messages = []
        
        # Convert messages to the format expected by litellm
        if isinstance(messages, list):
            completion_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
        else:
            # Handle single message case
            completion_messages = [
                {"role": messages.role, "content": messages.content}
            ]

        # Insert system prompt at the beginning
        if system_prompt:
            completion_messages.insert(0, {"role": "system", "content": system_prompt})

        completion_response = None
        try:
            if self.use_anthropic_client:
                action_args, completion_response = self._anthropic_completion(
                    completion_messages, system_prompt, response_model
                )
            elif response_model is None:
                completion_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )
                action_args, completion_response = self._litellm_text_completion(
                    completion_messages,
                )
            elif self.response_format == LLMResponseFormat.REACT and isinstance(
                response_model, list
            ):
                action_args, completion_response = self._litellm_react_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.use_openai_client:
                action_args, completion_response = self._openai_completion(
                    completion_messages, system_prompt, response_model
                )
            elif self.response_format == LLMResponseFormat.TOOLS:
                action_args, completion_response = self._litellm_tool_completion(
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
                logger.error(
                    f"Request failed. self.model: {self.model}, base_url: {self.model_base_url}. Model: {e.model}, Provider {e.llm_provider}. Litellm {e.litellm_debug_info}. Exception {e.message}"
                )
                if e.status_code >= 500:
                    raise CompletionRejectError(
                        f"Failed to create completion: {e}",
                        messages=completion_messages,
                        last_completion=completion_response,
                    ) from e

            else:
                logger.error(f"Failed to get completion response from litellm: {e}")

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

        return action_args, completion

    def create_text_completion(self, messages: List[Message], system_prompt: str):
        completion_messages = self._map_completion_messages(messages)

        if (
            self.supports_anthropic_computer_use
            or self.supports_anthropic_prompt_caching
        ):
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

    def _litellm_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        structured_output: type[StructuredOutput] | list[type[StructuredOutput]],
    ) -> Tuple[StructuredOutput, ModelResponse]:
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

                response = response_model.from_response(completion_response)

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
    ) -> Tuple[StructuredOutput, ModelResponse]:
        action_input_schemas = []

        for action in actions:
            action_input_schemas.append(f" * {action.get_name()} {action.format_schema_for_llm()}")
            
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
                action_class = next((a for a in actions if a.name.fget(a) == action_name), None)
                if not action_class:
                    action_names = [a.name.fget(a) for a in actions]
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
                return action_request, completion_response

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

    def _litellm_tool_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        response_model: type[StructuredOutput]| List[type[StructuredOutput]],
        is_retry: bool = False,
    ) -> Tuple[StructuredOutput, ModelResponse]:
        litellm.drop_params = True
        messages.insert(0, {"role": "system", "content": system_prompt})

        if isinstance(response_model, list):
            tools = [r.openai_schema(thoughts_in_action=True) for r in response_model]
        else:
            tools = [response_model.openai_schema()]

        completion_response = litellm.completion(
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
            return self._litellm_tool_completion(messages, system_prompt, response_model, is_retry=True)

        action = None
        if isinstance(response_model, list):
            for r in response_model:
                if r.model_json_schema()["title"] == tool_name:
                    action = r
                    break
        else:
            action = response_model

        if not action:
            available_actions = [r.model_json_schema()["title"] for r in response_model]
            raise ValueError(f"Unknown action {tool_name}. Available acitons: {available_actions}")

        action_args = action.model_validate(tool_args)

        if (
                hasattr(action_args, "thoughts")
                and completion_response.choices[0].message.content
                and not action_args.thoughts
        ):
            action_args.thoughts = completion_response.choices[0].message.content

        return action_args, completion_response

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

    def _openai_completion(
        self,
        messages: list[dict],
        system_prompt: str,
        actions: List[type[StructuredOutput]] | None = None,
        response_format: type[StructuredOutput] | None = None,
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

        messages.insert(0, {"role": "system", "content": system_prompt})

        tools = []
        if actions:
            for action in actions:
                schema = action.openai_schema()
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
                    # tool_choice="required",
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
        response_model: type[StructuredOutput]
        | List[type[StructuredOutput]]
        | None = None,
    ) -> Tuple[StructuredOutput | str, Any]:
        tools = []
        tool_choice = {"type": "any"}
        actions = []

        print(messages)

        # Ensure messages are properly formatted
        sanitized_messages = []
        for message in messages:
            if isinstance(message.get('content'), (list, dict)):
                # Convert complex content to string
                content = json.dumps(message['content']) if isinstance(message['content'], dict) else str(message['content'])
            else:
                content = message.get('content', '')
                
            sanitized_messages.append({
                "role": message["role"],
                "content": content
            })

        if response_model:
            if isinstance(response_model, list):
                actions = response_model
            else:
                actions = [response_model]

            for action in actions:
                try:
                    # Skip computer use tools for Haiku
                    if hasattr(action, "name") and action.name == "str_replace_editor":
                        if not self.supports_anthropic_computer_use:
                            continue
                        tools.append(
                            {"name": "str_replace_editor", "type": "text_editor_20241022"}
                        )
                    else:
                        # Use model_json_schema instead of model_dump for class schemas
                        schema = action.model_json_schema()
                        if "thoughts" in schema.get("properties", {}):
                            del schema["properties"]["thoughts"]
                        tools.append({
                            "name": getattr(action, "name", action.__name__.lower()),
                            "parameters": schema
                        })
                except Exception as e:
                    logger.warning(f"Failed to generate schema for action {action}: {e}")
                    continue

        if not tools:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN

        # Fix system message format
        if system_prompt:
            system_message = system_prompt
        else:
            system_message = None

        if "anthropic" in self.model:
            anthropic_client = AnthropicBedrock()
            extra_headers = {}
        else:
            anthropic_client = Anthropic()
            extra_headers = {}

        # Only enable betas for models that support them
        betas = []
        if self.supports_anthropic_computer_use:
            betas.append("computer-use-2024-10-22")
        if self.supports_anthropic_prompt_caching:
            betas.append("prompt-caching-2024-07-31")
            _inject_prompt_caching(sanitized_messages)

        try:
            logger.debug(f"Sending messages to Anthropic: {json.dumps(messages, indent=2)}")
            completion_response = anthropic_client.beta.messages.create(
                model=self.model,
                messages=sanitized_messages,
                system=system_message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=tools,
                tool_choice=tool_choice,
                extra_headers=extra_headers,
                betas=betas if betas else None,
            )
            logger.debug(f"Received response from Anthropic: {completion_response}")

            # Parse the completion response
            if response_model:
                if completion_response.content:
                    content = completion_response.content[0].text
                    if isinstance(response_model, list):
                        # If response_model is a list, use the first model that successfully parses
                        for model_class in response_model:
                            try:
                                structured_output = model_class.from_response(completion_response)
                                return structured_output, completion_response
                            except Exception as e:
                                logger.debug(f"Failed to parse with {model_class.__name__}: {e}")
                                continue
                        raise ValueError("Failed to parse response with any of the provided models")
                    else:
                        structured_output = response_model.from_response(completion_response)
                        return structured_output, completion_response
                else:
                    logger.warning("No content in completion response")
                    return None, completion_response
            else:
                if completion_response.content:
                    return completion_response.content[0].text, completion_response
                else:
                    logger.warning("No content in completion response")
                    return None, completion_response

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}", exc_info=True)
            raise

    def _map_completion_messages(self, messages: list[Message]) -> list[dict]:
        if self.use_anthropic_client:
            # Simplified format for Anthropic
            completion_messages = []
            for message in messages:
                if message.role == "user":
                    completion_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif message.role == "assistant":
                    if message.content:
                        completion_messages.append({
                            "role": "assistant", 
                            "content": message.content
                        })
                    elif message.tool_call:
                        # Format tool calls appropriately for Anthropic
                        tool_content = message.tool_call.input
                        if "thoughts" in tool_content:
                            thoughts = tool_content.pop("thoughts")
                            content = f"{thoughts}\n\n"
                        else:
                            content = ""
                        content += json.dumps(tool_content, indent=2)
                        completion_messages.append({
                            "role": "assistant",
                            "content": content
                        })
            return completion_messages
        else:
            # Existing logic for other models
            ...

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
    messages: list[BetaMessageParam],
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