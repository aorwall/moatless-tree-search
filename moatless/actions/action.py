import importlib
import logging
import pkgutil
from typing import Any, Optional, List, Dict, Type, Tuple, ClassVar

from instructor import OpenAISchema
from instructor.utils import classproperty, extract_json_from_codeblock
from openai.types.chat import ChatCompletion
from pydantic import Field, BaseModel, PrivateAttr

from moatless.completion import Completion, ToolCall
from moatless.file_context import FileContext
from moatless.schema import RewardScaleEntry
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

class ActionArguments(OpenAISchema):
    pass



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


class Action(BaseModel):

    name: str
    description: str
    args_schema: Type[ActionArguments]

    def __init__(self, **data):
        super().__init__(**data)

    def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> ActionOutput:
        """
        Execute the action.
        """

        message = self._execute(file_context=file_context)
        return ActionOutput.create(message)

    def _execute(self, file_context: FileContext | None = None) -> str | None:
        """
        Execute the action and return the updated FileContext.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, input=self.model_dump())

    @property
    def log_name(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

    def equals(self, other: "Action") -> bool:
        return self.model_dump(exclude={"scratch_pad"}) == other.model_dump(
            exclude={"scratch_pad"}
        )

    def to_prompt(self):
        prompt = f"Action: {self.__class__.__name__}\n"
        prompt += "\n".join(
            [
                f"  {k}: {v}"
                for k, v in self.model_dump(exclude={"scratch_pad"}).items()
            ]
        )
        return prompt

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

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        if trajectory_length < 3:
            return [
                "Exploratory Actions: Recognize that initial searches and information-gathering steps are essential and should not be heavily penalized if they don't yield immediate results.",
                "Appropriateness of Action: Evaluate if the action is logical given the agent's current knowledge and the early stage of problem-solving.",
            ]
        else:
            return [
                "Solution Quality: Assess the logical changes, contextual fit, and overall improvement without introducing new issues.",
                "Progress Assessment: Evaluate the agent's awareness of solution history, detection of repetitive actions, and planned next steps.",
                "Repetitive Actions: Detect if the agent is repeating the same unsuccessful actions without making progress and penalize accordingly.",
            ]

    def get_reward_scale(self, trajectory_length) -> List[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The action significantly advances the solution.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The action contributes positively towards solving the problem.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The action is acceptable but may have some issues.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The action has minimal impact or minor negative consequences.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The action is inappropriate or shows a lack of progress.",
            ),
            RewardScaleEntry(
                min_value=-100,
                max_value=-50,
                description="The action is counterproductive or demonstrates persistent repetition without learning.",
            ),
        ]

    @classmethod
    def create_action(cls, action_name: str, **data) -> "Action":
        """
        Dynamically import and create the appropriate action instance.
        """
        action_class = cls.get_action_class(action_name)
        return action_class(**data)

    @staticmethod
    def get_action_class(action_name: str) -> Type["Action"]:
        """
        Dynamically import and return the appropriate action class from all modules in moatless.actions.
        """
        actions_package = importlib.import_module("moatless.actions")
        
        for _, module_name, _ in pkgutil.iter_modules(actions_package.__path__):
            full_module_name = f"moatless.actions.{module_name}"
            module = importlib.import_module(full_module_name)
            action_class = getattr(module, action_name, None)
            if action_class and issubclass(action_class, Action):
                return action_class

        raise ValueError(f"Unknown action: {action_name}")

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

    @staticmethod
    def generate_reward_scale_entries(
        descriptions: List[Tuple[int, int, str]]
    ) -> List[RewardScaleEntry]:
        """
        Generate a list of RewardScaleEntry objects based on the provided descriptions.

        Args:
            descriptions: A list of tuples, each containing (min_value, max_value, description)

        Returns:
            A list of RewardScaleEntry objects
        """
        return [
            RewardScaleEntry(min_value=min_val, max_value=max_val, description=desc)
            for min_val, max_val, desc in descriptions
        ]

    def get_reward_range(self, trajectory_length: int) -> Tuple[int, int]:
        """
        Get the minimum and maximum reward values for this action.

        Args:
            trajectory_length: The length of the current trajectory

        Returns:
            A tuple containing the minimum and maximum reward values
        """
        reward_scale = self.get_reward_scale(trajectory_length)
        min_reward = min(entry.min_value for entry in reward_scale)
        max_reward = max(entry.max_value for entry in reward_scale)
        return min_reward, max_reward

    def get_value_function_prompt(self) -> str:
        """
        Get the base prompt for the value function.
        This method can be overridden in subclasses to provide action-specific prompts.
        """
        return """Your role is to evaluate the **last executed action** of the search tree that our AI agents are traversing, to help us determine the best trajectory to solve a programming issue. The agent is responsible for identifying and modifying the correct file(s) in response to the problem statement.

At this stage, the agent is still working on the solution. Your task is twofold:
1. **Evaluation**: Assess whether the change done by the **last executed action** is appropriate for addressing the problem and whether the agent is on the right path to resolving the issue.
2. **Alternative Feedback**: Independently of your evaluation, provide guidance for an alternative problem-solving branch. This ensures parallel exploration of different solution paths.
"""


class ActionExecution(BaseModel):
    action: Action = Field(..., description="The executed action")
    output: Optional[ActionOutput] = Field(
        None, description="The output of the executed action"
    )
    build_completion: Optional[Completion] = Field(
        None, description="The completion prompt to build the action"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["action"] = self.action.model_dump(**kwargs)
        data["output"] = self.output.model_dump(**kwargs) if self.output else None
        if not kwargs.get("exclude") or "build_completion" not in kwargs.get("exclude"):
            data["build_completion"] = (
                self.build_completion.model_dump(**kwargs)
                if self.build_completion
                else None
            )
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        if (
            isinstance(obj, dict)
            and "action" in obj
            and isinstance(obj["action"], dict)
        ):
            action_data = obj["action"]
            if "action_name" in action_data:
                action_name = action_data.pop("action_name")
                obj["action"] = Action.create_action(action_name, **action_data)
        return super().model_validate(obj, **kwargs)
