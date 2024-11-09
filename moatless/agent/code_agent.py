import logging
from typing import List, Type
import json

from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, RequestMoreContext
from moatless.actions.action import Action
from moatless.actions.apply_change_and_test import ApplyCodeChangeAndTest
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.edit import ClaudeEditTool
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.agent.agent import ActionAgent
from moatless.agent.code_prompts import SYSTEM_PROMPT, CLAUDE_PROMPT, SIMPLE_CODE_PROMPT
from moatless.completion.completion import (
    LLMResponseFormat, CompletionModel,
)
from moatless.completion.model import Message
from moatless.index import CodeIndex
from moatless.node import Node, MessageHistoryType
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

logger = logging.getLogger(__name__)


class CodingAgent(ActionAgent):

    def generate_system_prompt(self, possible_actions: List[Type[Action]]) -> str:
        if self.system_prompt:
            prompt = self.system_prompt
        else:
            prompt = SYSTEM_PROMPT

        if self.completion.response_format == LLMResponseFormat.JSON:
            few_shot_examples = []
            for action in possible_actions:
                examples = action.get_few_shot_examples()
                if examples:
                    few_shot_examples.extend(examples)

            if few_shot_examples:
                prompt += "\n\nHere are some examples of how to use the available actions:\n\n"
                for example in few_shot_examples:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name
                    }
                    prompt += f"User: {example.user_input}\nAssistant:\n{json.dumps(action_json, indent=2)}\n"

        return prompt

    def determine_possible_actions(self, node: Node) -> List[Action]:
        possible_actions = self.actions.copy()

        # Remove RequestCodeChange and RunTests if there's no file context
        if node.file_context.is_empty():
            possible_actions = [
                action
                for action in possible_actions
                if action.__class__ not in [ApplyCodeChangeAndTest, RequestCodeChange, RunTests]
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
                child.action.name for child in siblings if child.is_duplicate
            )
            possible_actions = [
                action
                for action in possible_actions
                if action.name not in duplicate_actions
            ]

        logger.info(
            f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in possible_actions]}"
        )

        return possible_actions

    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: CompletionModel,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        edit_completion_model: CompletionModel | None = None,
    ):
        system_prompt = None

        if completion_model.supports_anthropic_computer_use:
            actions = create_claude_coding_actions(
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                completion_model=completion_model
            )
            system_prompt = CLAUDE_PROMPT
        else:
            actions = create_coding_actions(
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                identify_completion_model=completion_model,
                edit_completion_model=edit_completion_model or completion_model,
            )

        if not runtime:
            system_prompt = SIMPLE_CODE_PROMPT

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_history_type=MessageHistoryType.MESSAGES,
            include_extra_history=True,
            include_file_context=False,
            include_git_patch=False,
        )


def create_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    runtime: RuntimeEnvironment | None = None,
    identify_completion_model: CompletionModel | None = None,
    edit_completion_model: CompletionModel | None = None,
):
    find_class = FindClass(code_index=code_index, repository=repository, completion_model=identify_completion_model)
    find_function = FindFunction(code_index=code_index, repository=repository, completion_model=identify_completion_model)
    find_code_snippet = FindCodeSnippet(code_index=code_index, repository=repository, completion_model=identify_completion_model)
    semantic_search = SemanticSearch(code_index=code_index, repository=repository, completion_model=identify_completion_model)
    request_context = RequestMoreContext(repository=repository)

    actions = [
        semantic_search,
        find_class,
        find_function,
        find_code_snippet,
        request_context,
    ]

    if runtime:
        request_code_change = ApplyCodeChangeAndTest(
            code_index=code_index, repository=repository, runtime=runtime, completion_model=edit_completion_model
        )
        #actions.append(
        #    RunTests(code_index=code_index, repository=repository, runtime=runtime)
        #)
    else:
        request_code_change = RequestCodeChange(
            repository=repository, completion_model=edit_completion_model
        )

    actions.append(request_code_change)
    actions.append(Finish())
    actions.append(Reject())

    return actions


def create_claude_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    runtime: RuntimeEnvironment | None = None,
    completion_model: CompletionModel | None = None,
):
    find_class = FindClass(code_index=code_index, repository=repository, completion_model=completion_model)
    find_function = FindFunction(code_index=code_index, repository=repository, completion_model=completion_model)
    find_code_snippet = FindCodeSnippet(code_index=code_index, repository=repository, completion_model=completion_model)
    semantic_search = SemanticSearch(code_index=code_index, repository=repository, completion_model=completion_model)
    request_context = RequestMoreContext(repository=repository)
    request_code_change = ClaudeEditTool(code_index=code_index, repository=repository, runtime=runtime)

    actions = [
        semantic_search,
        find_class,
        find_function,
        find_code_snippet,
        request_context,
        request_code_change
    ]

    actions.append(Finish())
    actions.append(Reject())

    return actions
