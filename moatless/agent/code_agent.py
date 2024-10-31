import logging
from typing import List, Type

from moatless.actions.action import Action
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.run_tests import RunTests
from moatless.agent.agent import ActionAgent
from moatless.agent.code_prompts import SYSTEM_PROMPT, FEW_SHOT_JSON
from moatless.completion.completion import (
    LLMResponseFormat,
)
from moatless.node import Node

logger = logging.getLogger(__name__)


class CodingAgent(ActionAgent):
    def _create_system_prompt(self, possible_actions: List[Type[Action]]) -> str:
        if self.system_prompt:
            prompt = self.system_prompt
        else:
            prompt = SYSTEM_PROMPT

        if self.completion.response_format == LLMResponseFormat.JSON:
            prompt += "\n\n" + FEW_SHOT_JSON

        return prompt

    def _determine_possible_actions(self, node: Node) -> List[Action]:
        possible_actions = self.actions.copy()

        # Remove RequestCodeChange and RunTests if there's no file context
        if node.file_context.is_empty():
            possible_actions = [
                action
                for action in possible_actions
                if action.__class__ not in [RequestCodeChange, RunTests]
            ]

        # Remove RunTests if it was just executed in the parent node
        if (
            node.parent
            and node.parent.action
            and node.parent.action.__class__ == RunTests
        ):
            possible_actions = [
                action for action in possible_actions if action.__class__ != RunTests
            ]

        # Remove Finish and Reject if there's no file context or no code changes
        if not node.file_context.has_patch():
            possible_actions = [
                action
                for action in possible_actions
                if action.__class__ not in [Finish, Reject]
            ]

        # Remove actions that have been marked as duplicates
        if node.parent:
            siblings = [
                child for child in node.parent.children if child.node_id != node.node_id
            ]
            duplicate_actions = set(
                child.action.__class__ for child in siblings if child.is_duplicate
            )
            possible_actions = [
                action
                for action in possible_actions
                if action.__class__ not in duplicate_actions
            ]

        logger.info(
            f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in possible_actions]}"
        )
        return possible_actions
