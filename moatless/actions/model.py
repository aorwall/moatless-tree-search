import importlib
import json
import logging
import pkgutil
from abc import ABC
from typing import Dict, Type, Any, Optional, Self

from instructor import OpenAISchema
from instructor.utils import extract_json_from_codeblock, classproperty
from openai.types.chat import ChatCompletion
from pydantic import Field, BaseModel, model_validator

from moatless.completion.model import ToolCall, Completion

logger = logging.getLogger(__name__)


_action_args: Dict[str, Type["ActionArguments"]] = {}


class ActionArguments(OpenAISchema, ABC):
    scratch_pad: str = Field(description="Your reasoning for the action.")

    class Config:
        title = "Action"

    @classproperty
    def name(cls):
        return cls.Config.title if hasattr(cls.Config, "title") else cls.__name__
    
    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, input=self.model_dump())

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def equals(self, other: "ActionArguments") -> bool:
        return self.model_dump(exclude={"scratch_pad"}) == other.model_dump(
            exclude={"scratch_pad"}
        )

    def to_prompt(self):
        prompt = f"Action: {self.name}\n"
        prompt += "\n".join(
            [f"  {k}: {v}" for k, v in self.model_dump(exclude={"scratch_pad"}).items()]
        )
        return prompt

    @model_validator(mode='before')
    @classmethod
    def fix_scratch_pad(cls, data: Any) -> Any:
        """Allow scratch_pad to be null."""
        if isinstance(data, dict):
            if not data.get("scratch_pad"):
                data["scratch_pad"] = ""

        return data

    @model_validator(mode='before')
    @classmethod
    def fix_null_fields(cls, data: Any) -> Any:
        """Allow scratch_pad to be null."""
        if isinstance(data, dict):
            for key, value in data.items():
                if value == "null":
                    data[key] = None

        return data

    @classmethod
    def get_action_args(cls, action_name: str) -> Type["ActionArguments"]:
        """
        Dynamically import and return the appropriate ActionArguments class for the given action.
        """
        if not _action_args:
            cls._load_action_args()

        action_args = _action_args.get(action_name)
        if action_args:
            return action_args

        raise ValueError(f"Unknown action: {action_name}")

    @classmethod
    def _load_action_args(cls):
        actions_package = importlib.import_module("moatless.actions")

        for _, module_name, _ in pkgutil.iter_modules(actions_package.__path__):
            full_module_name = f"moatless.actions.{module_name}"
            module = importlib.import_module(full_module_name)
            for name, obj in module.__dict__.items():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ActionArguments)
                    and obj != ActionArguments
                ):
                    _action_args[name] = obj

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionArguments":
        if isinstance(obj, dict):
            obj = obj.copy()
            action_args_class_path = obj.pop("action_args_class", None)
            if action_args_class_path:
                module_name, class_name = action_args_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                action_args_class = getattr(module, class_name)
                return action_args_class.model_validate(obj)
        return super().model_validate(obj)


    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwarg,
    ) -> Self:
        message = json_data

        # Clean control characters from the message
        cleaned_message = ''.join(char for char in message if ord(char) >= 32 or char in '\n\r\t')
        if cleaned_message != message:
            logger.info(f"parse_json() Cleaned control chars: {repr(message)} -> {repr(cleaned_message)}")
        message = cleaned_message

        # Extract JSON from codeblock if present
        json_message = extract_json_from_codeblock(message)
        if json_message != message:
            logger.info(f"parse_json() Extracted JSON: {repr(json_message)}")
        message = json_message

        logger.debug(f"parse_json() Final message to validate: {repr(message)}")

        __tracebackhide__ = True
        return super().model_validate_json(
            message,
            **kwarg
        )


    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message.content or ""
        logger.info(f"parse_json() Original message: {repr(message)}")

        # Clean control characters from the message
        cleaned_message = ''.join(char for char in message if ord(char) >= 32 or char in '\n\r\t')
        if cleaned_message != message:
            logger.info(f"parse_json() Cleaned control chars: {repr(message)} -> {repr(cleaned_message)}")
        message = cleaned_message
        
        # Extract JSON from codeblock if present
        json_message = extract_json_from_codeblock(message)
        if json_message != message:
            logger.info(f"parse_json() Extracted JSON: {repr(json_message)}")
        message = json_message

        logger.debug(f"parse_json() Final message to validate: {repr(message)}")
        return cls.model_validate_json(
            message,
            context=validation_context,
            strict=strict,
        )


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str


class Observation(BaseModel):
    message: str = Field(
        description="The message returned to the agent, will be displayed in chat histoy."
    )
    extra: Optional[str] = Field(
        None,
        description="Extra information to be returned to the agent, will not be displayed in chat history.",
    )
    terminal: bool = Field(
        False, description="Indicates if this action results in a terminal state"
    )
    expect_correction: bool = Field(
        False, description="Indicates that a the action arguments was inccorect and we expect a correction"
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, description="Additional properties"
    )
    execution_completion: Optional[Completion] = Field(
        None, description="Completion created when executing the action"
    )

    @classmethod
    def create(cls, message: str, terminal: bool = False):
        return cls(message=message, terminal=terminal)


class FewShotExample(BaseModel):
    user_input: str = Field(..., description="The user's input/question")
    action: ActionArguments = Field(..., description="The expected response as ActionArguments")

    @classmethod
    def create(cls, user_input: str, action: ActionArguments) -> "FewShotExample":
        return cls(user_input=user_input, action=action)

