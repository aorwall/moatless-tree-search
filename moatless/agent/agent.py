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
from moatless.exceptions import RuntimeError, RejectError, CompletionError
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class MessageHistoryType(Enum):
    MESSAGES = "messages"  # Provides all messages in sequence
    SUMMARY = "summary"  # Generates one message with summarized history


class ActionAgent(BaseModel):
    system_prompt: Optional[str] = Field(
        None, description="System prompt to be used for generating completions"
    )
    actions: List[Action] = Field(default_factory=list)
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.SUMMARY,
        description="Determines how message history is generated"
    )
    include_extra_history: bool = Field(
        default=False,
        description="Whether to include extra execution details in message history"
    )

    _completion: CompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        completion: CompletionModel,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        message_history_type: MessageHistoryType = MessageHistoryType.SUMMARY,
        include_extra_history: bool = False,
    ):
        actions = actions or []
        actions_map = {action.args_schema: action for action in actions}
        super().__init__(
            actions=actions, 
            system_prompt=system_prompt,
            message_history_type=message_history_type,
            include_extra_history=include_extra_history
        )
        self._completion = completion
        self._action_map = actions_map

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
        node.possible_actions = [action.name for action in possible_actions]

        system_prompt = self.generate_system_prompt(possible_actions)
        messages = self.generate_message_history(node)

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
                f"Node{node.node_id}: Action {node.action} not found in action map. "
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

    def generate_message_history(self, node: Node) -> list[Message]:
        """Generate message history based on the configured message_history_type."""
        messages = [UserMessage(content=f"<task>\n{node.get_root().message}\n</task>\n\n")]
        
        previous_nodes = node.get_trajectory()[:-1]
        if self.message_history_type == MessageHistoryType.SUMMARY:
            messages.extend(self.generate_summary_history(previous_nodes, node))
        else:  # MessageHistoryType.FULL
            messages.extend(self.generate_full_history(previous_nodes, node))
            
        return messages

    def generate_summary_history(self, previous_nodes: List[Node], current_node: Node) -> list[Message]:
        """Generate a single message containing summarized history."""
        formatted_history: List[str] = []
        counter = 0
        
        # Generate history
        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = f"\n## {counter}. Action: {previous_node.action.name}\n"
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    formatted_state += f"\n\nOutput: {previous_node.observation.message}"
                    if (i == len(previous_nodes) - 1) or self.include_extra_history:
                        if previous_node.observation.extra:
                            formatted_state += "\n\n"
                            formatted_state += previous_node.observation.extra
                    formatted_history.append(formatted_state)
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")

        content = "Below is the history of previously executed actions and their outputs.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"
        
        content += self._format_file_context(current_node)
        content += self._format_git_patch(current_node)
        content += self._format_feedback(current_node)
        return [UserMessage(content=content)]

    def generate_full_history(self, previous_nodes: List[Node], current_node: Node) -> list[Message]:
        """Generate a sequence of messages representing the full conversation history."""
        messages: list[Message] = []
        
        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

                content = ""
                if previous_node.message:
                    content = f"<issue>\n{previous_node.message}\n</issue>"

                if previous_node.observation:
                    content += previous_node.observation.message
                    if (i == len(previous_nodes) - 1) or self.include_extra_history:
                        if previous_node.observation.extra:
                            content += "\n" + previous_node.observation.extra

                if not content:
                    logger.warning(
                        f"Node{previous_node.node_id}: No content to add to messages"
                    )

                messages.append(UserMessage(content=content))

        # Add file context, git patch, and feedback as the final message
        context = self._format_file_context(current_node)
        git_patch = self._format_git_patch(current_node)
        feedback = self._format_feedback(current_node)
        
        if context or git_patch or feedback:
            messages.append(UserMessage(content=context + git_patch + feedback))
            
        return messages

    def _format_file_context(self, node: Node) -> str:
        """Generate formatted string for file context."""
        if not node.file_context:
            return ""
        
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
        return f"# Current file context:\n\n<file_context>\n{file_context_str}\n</file_context>\n\n"

    def _format_git_patch(self, node: Node) -> str:
        """Generate formatted string for git patch."""
        if not node.file_context:
            return ""

        full_patch = node.file_context.generate_git_patch()
        message = "Changes made to the codebase so far:\n"
        if full_patch.strip():
            message += "<git_patch>\n"
            message += full_patch
            message += "\n</git_patch>\n\n"
        else:
            message += "<git_patch>\n"
            message += "No changes made yet."
            message += "\n</git_patch>\n\n"
        return message

    def _format_feedback(self, node: Node) -> str:
        """Generate formatted string for feedback."""
        if not node.feedback:
            return ""
        
        logger.info(f"Node{node.node_id}: Feedback provided: {node.feedback}")
        return f"\n\n{node.feedback}"

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
