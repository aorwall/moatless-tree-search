import logging
from typing import List, Type, Tuple

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments
from moatless.actions.reject import Reject
from moatless.completion.completion import (
    CompletionModel,
)
from moatless.completion.model import Message, AssistantMessage, UserMessage, Completion
from moatless.node import Node
from moatless.settings import ModelSettings, Settings
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        actions: List[Action] | None = None,
        completion: CompletionModel | None = None,
        model_settings: ModelSettings | None = None,
    ):
        if completion:
            self.completion = completion
        else:
            model_settings = model_settings or Settings.default_model
            self.completion = CompletionModel.from_settings(model_settings)

        self.actions = actions
        self.action_map = {action.args_schema: action for action in actions} if actions else {}

    def run(self, node: Node):
        self._generate_action(node)
        self._execute_action(node)
            
        logger.info(
            f"Node{node.node_id}: Executed action: {node.action.name}. "
            f"Terminal: {node.output.terminal}. "
            f"Output: {node.output.message}"
        )

    def _generate_action(self, node: Node):
        """
        Generate an action
        """
        completion_response = None
        try:            
            node.possible_actions = self._determine_possible_actions(node)
            system_prompt = self._create_system_prompt(node.possible_actions)
            messages = self._create_messages(node)

            action_args, completion_response = self._generate_action_args(system_prompt, messages, node.possible_actions)

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
            node.action = Reject(rejection_reason=f"Failed to generate action: {e}")

    def _execute_action(self, node: Node) -> Tuple[ActionArguments, Completion]:
        action = self.action_map.get(type(node.action))
        if action:
            output = action.execute(node.action, node.file_context)
            node.output = output

            if output.execution_completion:
                node.completions["execute_action"] = output.execution_completion

            logger.info(
                f"Node{node.node_id}: Executed action: {action.name}. "
                f"Terminal: {node.output.terminal}. "
                f"Output: {node.output.message}"
            )

    def _create_system_prompt(self, possible_actions: List[Type[ActionArguments]]) -> str:
        return ""

    def _create_messages(self, node: Node) -> list[Message]:
        messages: list[Message] = []

        last_node = None
        previous_nodes = node.get_trajectory()[:-1]
        for previous_node in previous_nodes:
            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

            content = previous_node.message or ""

            messages.append(UserMessage(content=content))

            last_node = previous_node

        if last_node.output and last_node.output.extra:
            messages[-1].content += "\n" + last_node.output.extra

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
                f"# Current file context:\n\n<file_context>\n{file_context_str}\n</file_context>\n\nFunction response:\n"
                + messages[-1].content
            )

        if node.feedback:
            logger.info(f"Node{node.node_id}: Feedback provided: {node.feedback}")
            messages[-1].content += f"\n\n{node.feedback}"

        return messages

    def _generate_action_args(
        self, system_prompt: str, messages: List[Message], actions: List[Type[Action]]
    ) -> Tuple[ActionArguments, Completion]:
        return self.completion.create_completion(
            messages, system_prompt=system_prompt, actions=actions
        )

    def _determine_possible_actions(self, node: Node) -> List[Type[ActionArguments]]:
        return [action.args_schema for action in self.actions]
