import logging
from typing import List, Type, Tuple

from moatless.actions.action import Action
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.completion import (
    CompletionModel,
    LLMResponseFormat,
    Message,
    UserMessage,
    AssistantMessage,
    Completion,
    ToolCall,
)
from moatless.node import Node
from moatless.settings import ModelSettings, Settings
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        workspace: Workspace,
        actions: List[Type[Action]] | None = None,
        completion: CompletionModel | None = None,
        model_settings: ModelSettings | None = None,
    ):
        if completion:
            self.completion = completion
        else:
            model_settings = model_settings or Settings.default_model
            self.completion = CompletionModel.from_settings(model_settings)

        self.workspace = workspace
        self.actions = actions

    def generate_action(self, node: Node) -> Tuple[Action, Completion | None]:
        """
        Build and execute the action for the given node and apply the results to the node.
        """
        completion_response = None
        try:
            node.possible_actions = self._determine_possible_actions(node)
            action, completion_response = self._generate(node)
            logger.info(f"Node{node.node_id}: Generated action: {action.name}")

            # TODO: Configure this for each type of action
            if hasattr(action, "_completion_model"):
                action._completion_model = CompletionModel.from_settings(
                    Settings.default_model
                )

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Error generating action.")
            action = Reject(rejection_reason=f"Failed to generate action: {e}")

        return action, completion_response

    def _generate(self, node: Node) -> Tuple[Action, Completion]:
        system_prompt = self._create_system_prompt(node.possible_actions)
        messages = self._create_messages(node)
        return self._generate_action(system_prompt, messages, node.possible_actions)

    def _create_system_prompt(self, possible_actions: List[Type[Action]]) -> str:
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

    def _generate_action(
        self, system_prompt: str, messages: List[Message], actions: List[Type[Action]]
    ) -> Tuple[Action, Completion]:
        action, completion = self.completion.create_completion(
            messages, system_prompt=system_prompt, actions=actions
        )
        action.set_workspace(self.workspace)
        return action, completion

    def _determine_possible_actions(self, node: Node) -> List[Type[Action]]:
        return self.actions
