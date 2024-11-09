import logging
from typing import List, Type, Tuple, Dict, Any, Optional
import importlib
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.actions.reject import RejectArgs, Reject
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Message, AssistantMessage, UserMessage, Completion
from moatless.exceptions import RuntimeError, RejectError, CompletionError, CompletionRejectError
from moatless.index.code_index import CodeIndex
from moatless.node import Node, MessageHistoryType
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class ActionAgent(BaseModel):
    system_prompt: Optional[str] = Field(
        None, description="System prompt to be used for generating completions"
    )
    actions: List[Action] = Field(default_factory=list)
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Determines how message history is generated"
    )
    include_extra_history: bool = Field(
        default=True,
        description="Whether to include extra execution details in message history"
    )
    include_file_context: bool = Field(
        default=False,
        description="Whether to include the full file context in the last message"
    )
    include_git_patch: bool = Field(
        default=False,
        description="Whether to include the full git patch in the last message"
    )

    _completion: CompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        completion: CompletionModel,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        **data
    ):
        actions = actions or []
        super().__init__(
            actions=actions, 
            system_prompt=system_prompt,
            **data
        )
        self.set_actions(actions)
        self._completion = completion

    def set_actions(self, actions: List[Action]):
        self.actions = actions
        self._action_map = {action.args_schema: action for action in actions}

    @model_validator(mode="after")
    def verify_system_prompt(self) -> "ActionAgent":
        if self.system_prompt == "":
            self.system_prompt = None
        return self

    @model_validator(mode="after")
    def verify_actions(self) -> "ActionAgent":
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValidationError(
                    f"Invalid action type: {type(action)}. Expected Action subclass."
                )
            if not hasattr(action, "args_schema"):
                raise ValidationError(
                    f"Action {action.__class__.__name__} is missing args_schema attribute"
                )
        return self

    def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""
        try:
            self.generate_action(node)
            duplicate_node = node.find_duplicate()
            if duplicate_node:
                node.is_duplicate = True
                logger.info(
                    f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution."
                )
                return
            self.execute_action(node)
        except RejectError as e:
            logger.warning(f"Node{node.node_id}: Action rejected: {e.message}")
            rejection_reason = str(e)
            node.action = RejectArgs(rejection_reason=rejection_reason)
            self.execute_action(node)

            if isinstance(e, CompletionRejectError):
                if not node.observation.properties:
                    node.observation.properties = {}
                node.observation.properties["last_completion"] = e.last_completion

        except RuntimeError as e:
            logger.error(f"Node{node.node_id}: Runtime error: {e.message}")
            raise  #

        logger.info(
            f"Node{node.node_id}: Executed action: {node.action.name}. "
            f"Terminal: {node.observation.terminal if node.observation else False}. "
            f"Output: {node.observation.message if node.observation else None}"
        )

    def generate_action(self, node: Node):
        """Generate an action for the node."""
        if node.action:
            logger.info(f"Node{node.node_id}: Action already generated. Skipping.")
            return

        possible_actions = self.determine_possible_actions(node)
        # possible_actions = self.actions
        node.possible_actions = [action.name for action in possible_actions]

        system_prompt = self.generate_system_prompt(possible_actions)
        messages = node.generate_message_history(
            message_history_type=self.message_history_type,
            include_extra_history=self.include_extra_history,
            include_file_context=self.include_file_context,
            include_git_patch=self.include_git_patch,
        )

        action_args = [action.args_schema for action in possible_actions]

        try:
            action, completion_response = self._completion.create_completion(
                messages, system_prompt=system_prompt, actions=action_args
            )
        except ValidationError as e:
            logger.warning(f"Failed to create completion: {e}")
            raise RejectError(e.message) from e
        node.action = action
        node.completions["build_action"] = completion_response

    def execute_action(self, node: Node):
        """Execute the generated action."""
        if node.observation:
            logger.info(f"Node{node.node_id}: Observation already generated. Skipping.")
            return

        action = self._action_map.get(type(node.action))
        if action:
            node.observation = action.execute(node.action, node.file_context)

            if node.observation.execution_completion:
                node.completions["execute_action"] = (
                    node.observation.execution_completion
                )
        else:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self._action_map.keys()}"
            )
            raise Exception(f"Action {node.action} not found in action map.")

    def determine_possible_actions(self, node: Node) -> List[Action]:
        """Determine which actions that the agent can take based on the current node state."""
        actions = self.actions
        logger.debug(
            f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in actions]}"
        )
        return actions

    def generate_system_prompt(self, possible_actions: List[Action]) -> str:
        """Generate a system prompt for the agent."""
        return self.system_prompt



    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        dump["actions"] = []
        dump["agent_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        dump["message_history_type"] = self.message_history_type.value
        for action in self.actions:
            action_dump = action.model_dump(**kwargs)
            action_dump["action_class"] = (
                f"{action.__class__.__module__}.{action.__class__.__name__}"
            )
            dump["actions"].append(action_dump)
        return dump

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion", None)
            agent_class_path = obj.pop("agent_class", None)
            
            if "message_history_type" in obj:
                obj["message_history_type"] = MessageHistoryType(obj["message_history_type"])

            if completion_data:
                completion = CompletionModel.model_validate(completion_data)
            else:
                completion = None

            if repository:
                actions = [
                    Action.from_dict(
                        action_data,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                    for action_data in obj.get("actions", [])
                ]
            else:
                logger.debug(f"No repository provided, skip initiating actions")
                actions = []

            if agent_class_path:
                module_name, class_name = agent_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                instance = agent_class(actions=actions, completion=completion)
            else:
                instance = cls(actions=actions, completion=completion)

            instance._action_map = {action.args_schema: action for action in actions}
            return instance

        return super().model_validate(obj)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    @completion.setter
    def completion(self, value: CompletionModel):
        self._completion = value
