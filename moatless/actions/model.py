import importlib
import logging
import pkgutil
from abc import ABC
from typing import Dict, Type, Any, Optional

from instructor import OpenAISchema
from instructor.utils import extract_json_from_codeblock
from openai.types.chat import ChatCompletion
from pydantic import Field, BaseModel

from moatless.completion.model import ToolCall, Completion

logger = logging.getLogger(__name__)


_action_args: Dict[str, Type["ActionArguments"]] = {}


class ActionArguments(OpenAISchema, ABC):
    scratch_pad: str = Field(..., description="Your reasoning for the action.")

    class Config:
        title = 'Action'

    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, input=self.model_dump())

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def equals(self, other: "Action") -> bool:
        return self.model_dump(exclude={"scratch_pad"}) == other.model_dump(
            exclude={"scratch_pad"}
        )

    def name(self):
        return self.Config.title if hasattr(self.Config, 'title') else self.__class__.__name__

    def to_prompt(self):
        prompt = f"Action: {self.name()}\n"
        prompt += "\n".join(
            [
                f"  {k}: {v}"
                for k, v in self.model_dump(exclude={"scratch_pad"}).items()
            ]
        )
        return prompt

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
                if isinstance(obj, type) and issubclass(obj, ActionArguments) and obj != ActionArguments:
                    _action_args[name] = obj

    @classmethod
    def model_validate(cls: Type["Action"], obj: Any) -> "Action":
        if isinstance(obj, dict) and "action_name" in obj:
            action_name = obj.pop("action_name")
            return cls.create_action(action_name, **obj)
        return super().model_validate(obj)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["action_name"] = self.__class__.__name__
        return data

    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message.content or ""

        # Because Qwen-2.5-72B-Instruct keeps adding those to the responses...
        if "\x00" in message:
            logger.info(f"parse_json() Replace \x00 in: {message}")
            message = message.replace("\x00", "")
        message = extract_json_from_codeblock(message)

        return cls.model_validate_json(
            message,
            context=validation_context,
            strict=strict,
        )


class ActionOutput(BaseModel):
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
        False, description="Indicates that a correction is expected after this action"
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
