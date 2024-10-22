import logging
from typing import List, Type, Tuple, Dict, Any, Optional

from pydantic import BaseModel, Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments
from moatless.actions.reject import RejectArgs
from moatless.completion.completion import (
    CompletionModel,
)
from moatless.completion.model import Message, AssistantMessage, UserMessage, Completion
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    system_prompt: Optional[str] = Field(
        None, description="System prompt to be used for generating completions"
    )
    actions: List[Action] = Field(default_factory=list)

    _completion: CompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        completion: CompletionModel | None = None
    ):
        actions = actions or []
        actions_map = {action.args_schema: action for action in actions}
        super().__init__(actions=actions, system_prompt=system_prompt)
        self._completion = completion
        self._action_map = actions_map

    def run(self, node: Node):
        self._generate_action(node)
        self._execute_action(node)

        logger.info(
            f"Node{node.node_id}: Executed action: {node.action.name}. "
            f"Terminal: {node.observation.terminal}. "
            f"Output: {node.observation.message}"
        )

    def _generate_action(self, node: Node):
        """
        Generate an action
        """
        completion_response = None
        try:
            possible_actions = self._determine_possible_actions(node)
            node.possible_actions = [action.name for action in possible_actions]

            system_prompt = self._create_system_prompt(possible_actions)
            messages = self._create_messages(node)

            action_args, completion_response = self._generate_action_args(
                system_prompt, messages, possible_actions
            )

            node.action = action_args
            node.completions["build_action"] = completion_response

            duplicate_node = node.find_duplicate()
            if duplicate_node:
                logger.info(
                    f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution."
                )
                node.is_duplicate = True
                return

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Error generating action.")
            error_message = f"Failed to generate action: {str(e)}"
            node.action = RejectArgs(
                rejection_reason=error_message,
                scratch_pad=f"An error occurred during action generation: {error_message}",
            )

    def _execute_action(self, node: Node):
        action = self._action_map.get(type(node.action))
        if action:
            node.observation = action.execute(node.action, node.file_context)

            if node.observation.execution_completion:
                node.completions["execute_action"] = node.observation.execution_completion

    def _create_system_prompt(self, possible_actions: List[Action]) -> str:
        return self.system_prompt or ""

    def _create_messages(self, node: Node) -> list[Message]:
        messages: list[Message] = []

        last_node = None
        previous_nodes = node.get_trajectory()[:-1]
        for previous_node in previous_nodes:
            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

            content = ""
            if previous_node.message:
                # TODO: Don't hardcode the issue tag...
                content = f"<issue>\n{previous_node.message}\n</issue>"

            if previous_node.observation:
                content += previous_node.observation.message

            if not content:
                logger.warning(f"Node{previous_node.node_id}: No content to add to messages")

            messages.append(UserMessage(content=content))

            last_node = previous_node

        if last_node.observation and last_node.observation.extra:
            messages[-1].content += "\n" + last_node.observation.extra

        # TODO: Only add the updated file context per node
        if node.file_context:
            if node.file_context.is_empty():
                file_context_str = "No files added to file context yet."
            else:
                file_context_str = node.file_context.create_prompt(
                    show_span_ids=False,
                    show_line_numbers=True,
                    exclude_comments=False,
                    show_outcommented_code=True,
                    outcomment_code_comment="... rest of the code",
                )

            messages[-1].content = (
                f"# Current file context:\n\n<file_context>\n{file_context_str}\n</file_context>\n\n"
                + messages[-1].content
            )

        if node.feedback:
            logger.info(f"Node{node.node_id}: Feedback provided: {node.feedback}")
            messages[-1].content += f"\n\n{node.feedback}"

        return messages

    def _generate_action_args(
        self, system_prompt: str, messages: List[Message], actions: List[Action]
    ) -> Tuple[ActionArguments, Completion]:
        try:
            action_args = []
            for action in actions:
                if not isinstance(action, Action):
                    raise TypeError(
                        f"Invalid action type: {type(action)}. Expected Action subclass."
                    )
                if not hasattr(action, "args_schema"):
                    raise AttributeError(
                        f"Action {action.__class__.__name__} is missing args_schema attribute"
                    )
                action_args.append(action.args_schema)

            return self._completion.create_completion(
                messages, system_prompt=system_prompt, actions=action_args
            )
        except Exception as e:
            logger.exception(f"Error in _generate_action_args: {str(e)}")
            problematic_actions = [
                f"{action.__class__.__name__} (type: {type(action)})"
                for action in actions
                if not isinstance(action, Action) or not hasattr(action, "args_schema")
            ]
            if problematic_actions:
                error_message = f"The following actions are invalid or missing args_schema attribute: {', '.join(problematic_actions)}"
            else:
                error_message = (
                    "Unknown error occurred while generating action arguments"
                )
            raise RuntimeError(error_message) from e

    def _determine_possible_actions(self, node: Node) -> List[Action]:
        actions = self.actions
        logger.debug(
            f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in actions]}"
        )
        return actions

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        dump["actions"] = [action.model_dump(**kwargs) for action in self.actions]
        return dump

    @classmethod
    def model_validate(cls, obj: Any, repository: Repository = None, runtime: Any = None, code_index: CodeIndex = None) -> "Agent":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion", None)

            if completion_data:
                completion = CompletionModel.model_validate(completion_data)
            else:
                completion = None

            if repository:
                actions = [
                    Action.model_validate(action_data, repository=repository, runtime=runtime, code_index=code_index)
                    for action_data in obj.get("actions", [])
                ]
            else:
                logger.info(f"No repository provided, skip iniating acitons")
                actions = []

            instance = cls(actions=actions, completion=completion)
            instance._action_map = {action.args_schema: action for action in actions}
            return instance

        return super().model_validate(obj)

    # Add a property for completion to allow access while maintaining it as a private attribute
    @property
    def completion(self) -> CompletionModel:
        return self._completion

    @completion.setter
    def completion(self, value: CompletionModel):
        self._completion = value
