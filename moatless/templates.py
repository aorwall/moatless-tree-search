from typing import Optional

from moatless.actions.apply_change_and_test import ApplyCodeChangeAndTest
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.edit import ClaudeEditTool
from moatless.actions.find_class import FindClass
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.run_tests import RunTests
from moatless.actions.semantic_search import SemanticSearch
from moatless.agent.code_agent import CodingAgent
from moatless.agent.code_prompts import SIMPLE_CODE_PROMPT
from moatless.completion.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.search_tree import SearchTree

def create_basic_coding_actions(
    repository: Repository,
    code_index: CodeIndex,
    completion_model: CompletionModel
):
    find_class = FindClass(code_index=code_index, repository=repository, completion_model=completion_model)
    find_function = FindFunction(code_index=code_index, repository=repository, completion_model=completion_model)
    find_code_snippet = FindCodeSnippet(code_index=code_index, repository=repository, completion_model=completion_model)
    semantic_search = SemanticSearch(code_index=code_index, repository=repository, completion_model=completion_model)
    request_context = RequestMoreContext(repository=repository)
    request_code_change = RequestCodeChange(repository=repository, completion_model=completion_model)
    finish = Finish()
    reject = Reject()

    return [
        find_class,
        find_function,
        find_code_snippet,
        request_context,
        request_code_change,
        semantic_search,
        finish,
        reject,
    ]

def create_basic_coding_tree(
    message: str,
    repository: Repository,
    code_index: CodeIndex,
    completion_model: CompletionModel,
    max_iterations: int = 10,
    max_depth: int = 10,
    max_cost: float | None = None,
    perist_path: str | None = None,
):
    find_class = FindClass(code_index=code_index, repository=repository, completion_model=completion_model)
    find_function = FindFunction(code_index=code_index, repository=repository, completion_model=completion_model)
    find_code_snippet = FindCodeSnippet(code_index=code_index, repository=repository, completion_model=completion_model)
    semantic_search = SemanticSearch(code_index=code_index, repository=repository, completion_model=completion_model)
    request_context = RequestMoreContext(repository=repository)
    request_code_change = RequestCodeChange(
        repository=repository, completion_model=completion_model
    )
    finish = Finish()
    reject = Reject()

    actions = [
        find_class,
        find_function,
        find_code_snippet,
        request_context,
        request_code_change,
        semantic_search,
        finish,
        reject,
    ]

    file_context = FileContext(repo=repository)
    agent = CodingAgent(
        actions=actions, completion=completion_model, system_prompt=SIMPLE_CODE_PROMPT
    )
    return SearchTree.create(
        message=message,
        agent=agent,
        file_context=file_context,
        max_expansions=1,
        max_iterations=max_iterations,
        max_depth=max_depth,
        max_cost=max_cost,
        persist_path=perist_path,
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
    request_code_change = ClaudeEditTool()

    actions = [
        semantic_search,
        find_class,
        find_function,
        find_code_snippet,
        request_context,
        request_code_change
    ]

    if runtime:
        actions.append(
            RunTests(code_index=code_index, repository=repository, runtime=runtime)
        )

    actions.append(Finish())
    actions.append(Reject())

    return actions

def create_mcts_coding_tree(
    message: str,
    repository: Repository,
    code_index: CodeIndex,
    runtime: RuntimeEnvironment | None = None,
    completion_model: CompletionModel | None = None,
    max_iterations: int = 100,
    max_expansions: int = 3,
    max_depth: int = 20,
    max_cost: Optional[float] = None,
):
    actions = create_coding_actions()
    file_context = FileContext(repo=repository)
    agent = CodingAgent(actions=actions, completion=completion_model)
    return SearchTree.create(
        message=message,
        agent=agent,
        file_context=file_context,
        max_expansions=1,
        max_iterations=max_iterations,
        max_depth=max_depth,
        max_cost=max_cost,
    )
