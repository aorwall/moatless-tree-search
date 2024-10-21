from typing import Optional
from moatless.actions.code_change import RequestCodeChange
from moatless.agent.code_agent import CodingAgent
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.actions.find_class import FindClass
from moatless.actions.find_function import FindFunction
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.semantic_search import SemanticSearch
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.agent.agent import Agent
from moatless.completion.completion import CompletionModel
from moatless.search_tree import SearchTree
from moatless.settings import ModelSettings

def create_basic_coding_tree(
        message: str,
        repository: Repository,
        code_index: CodeIndex,
        model_settings: ModelSettings,
        max_iterations: int = 10,
        max_depth: int = 10,
        max_cost: Optional[float] = None
    ):
    completion_model = CompletionModel.from_settings(model_settings)
    find_class = FindClass(code_index=code_index, repository=repository)
    find_function = FindFunction(code_index=code_index, repository=repository)
    find_code_snippet = FindCodeSnippet(code_index=code_index, repository=repository)
    semantic_search = SemanticSearch(code_index=code_index, repository=repository)
    request_context = RequestMoreContext(repository=repository)
    request_code_change = RequestCodeChange(repository=repository, completion_model=completion_model)
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
        reject
    ]

    file_context = FileContext(repo=repository)
    agent = CodingAgent(actions=actions, completion=completion_model)
    return SearchTree.create(message=message, agent=agent, file_context=file_context, max_expansions=1, max_iterations=max_iterations, max_depth=max_depth, max_cost=max_cost)
