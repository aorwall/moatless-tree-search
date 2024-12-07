import json
from typing import List
from unittest.mock import Mock

from moatless.actions.request_context import RequestMoreContext

from moatless.actions.code_change import RequestCodeChange
from moatless.actions.find_class import FindClass
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.actions.search_base import SearchBaseAction
from moatless.actions.semantic_search import SemanticSearch
from moatless.agent.code_agent import CodingAgent
from moatless.completion.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.node import Node
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.runtime.runtime import TestResult
from moatless.templates import create_coding_actions


class MockCodeIndex(CodeIndex):
    def __init__(self):
        pass


class MockRuntimeEnvironment(RuntimeEnvironment):
    def run_tests(self, test_files: List[str] | None = None) -> list[TestResult]:
        return []


def test_determine_possible_actions():
    # Create a mock node with an empty file context
    repository = InMemRepository()
    node = Node.stub(file_context=FileContext(repo=repository))

    actions = create_coding_actions(repository=repository)
    agent = CodingAgent(actions=actions, completion=Mock(CompletionModel))

    # Test 1: Basic case with empty file context
    actions = agent.determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
    }

    # Test 2: Non-empty file context without code changes
    mock_repo = InMemRepository()
    mock_repo.save_file("test.py", "hello")
    file_context = FileContext(repo=mock_repo)
    file_context.add_span_to_context("test.py", "hello")
    node = Node.stub(file_context=file_context)
    actions = agent.determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
        RequestCodeChange,
        RunTests,
    }

    # Test 3: Non-empty file context with code changes
    mock_repo = InMemRepository()
    mock_repo.save_file("test.py", "hello")
    file_context = FileContext(repo=mock_repo)
    file_context.add_span_to_context("test.py", "hello")
    file_context.get_file("test.py").apply_changes("updated")
    node = Node.stub(file_context=file_context)
    actions = agent.determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
        RequestCodeChange,
        RunTests,
        Finish,
        Reject,
    }

    # Test 4: With a finished child
    finished_child = Node.stub(action=Finish())
    node.children.append(finished_child)
    actions = agent.determine_possible_actions(node)
    assert Finish not in actions


def test_determine_possible_actions_duplicate_actions():
    node = Node(node_id=4, file_context=FileContext(repo=InMemRepository()))
    agent = CodingAgent(completion=Mock(CompletionModel))

    duplicate_action = SemanticSearch(scratch_pad="", query="foo")
    parent = Node(node_id=1)
    first_child = Node(node_id=2, action=duplicate_action, is_duplicate=True)
    duplicate_child = Node(node_id=4, action=duplicate_action, is_duplicate=True)
    parent.children.append(first_child)
    parent.children.append(duplicate_child)
    node.parent = parent
    actions = agent.determine_possible_actions(node)
    assert SemanticSearch not in actions


def test_dump_and_load_coding_agent():
    repository = InMemRepository()
    code_index = MockCodeIndex()
    runtime = MockRuntimeEnvironment()

    completion_model = CompletionModel(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )

    actions = [
        FindClass(repository=repository, code_index=code_index),
        FindFunction(repository=repository, code_index=code_index),
        FindCodeSnippet(repository=repository, code_index=code_index),
        SemanticSearch(repository=repository, code_index=code_index),
        RequestMoreContext(repository=repository),
        RequestCodeChange(repository=repository, completion_model=completion_model),
        RunTests(repository=repository, code_index=code_index, runtime=runtime),
        Finish(),
        Reject(),
    ]

    original_agent = CodingAgent(actions=actions, completion=completion_model)

    dumped_agent = json.dumps(original_agent.model_dump(), indent=2)
    print(dumped_agent)

    # Load the agent from JSON
    loaded_agent_data = json.loads(dumped_agent)
    loaded_agent = CodingAgent.model_validate(loaded_agent_data)

    # Manually set the dependencies after loading
    for action in loaded_agent.actions:
        if isinstance(action, SearchBaseAction):
            action._repository = repository
            action._code_index = code_index
        if hasattr(action, "runtime"):
            action.runtime = runtime

    # Assert that the number of actions is the same
    assert len(original_agent.actions) == len(loaded_agent.actions)

    # Assert that all action types are preserved
    original_action_types = set(type(action) for action in original_agent.actions)
    loaded_action_types = set(type(action) for action in loaded_agent.actions)
    assert original_action_types == loaded_action_types

    # Assert that the _action_map is correctly populated
    assert set(original_agent._action_map.keys()) == set(
        loaded_agent._action_map.keys()
    )

    # Assert that the completion model is preserved
    assert isinstance(loaded_agent.completion, CompletionModel)
    assert loaded_agent.completion.model == original_agent.completion.model
    assert loaded_agent.completion.max_tokens == original_agent.completion.max_tokens
    assert loaded_agent.completion.temperature == original_agent.completion.temperature
    assert not loaded_agent.completion.model_api_key

    # Check if the dependencies are correctly set for each action
    for action in loaded_agent.actions:
        if hasattr(action, "_repository"):
            assert action._repository == repository
        if hasattr(action, "_code_index"):
            assert action._code_index == code_index
        if hasattr(action, "_runtime"):
            assert action._runtime == runtime

    for original_action, loaded_action in zip(
        original_agent.actions, loaded_agent.actions
    ):
        assert type(original_action) == type(loaded_action)
        assert original_action.name == loaded_action.name


def test_create_system_prompt_with_few_shot_examples():
    # Setup
    repository = InMemRepository()
    code_index = MockCodeIndex()
    completion_model = Mock(CompletionModel)
    completion_model.response_format = "json"
    
    actions = [
        FindClass(repository=repository, code_index=code_index),
        FindFunction(repository=repository, code_index=code_index),
        RequestCodeChange(repository=repository, completion_model=completion_model)
    ]
    
    agent = CodingAgent(actions=actions, completion=completion_model)
    
    # Get the system prompt
    prompt = agent.generate_system_prompt([FindClass, FindFunction, RequestCodeChange])
    print(prompt)
    
    # Verify the prompt structure
    assert "Here are some examples of how to use the available actions:" in prompt
    
    # Verify FindClass example
    assert "auth/*.py" in prompt
    assert "UserAuthentication" in prompt
    
    # Verify FindFunction example
    assert "Find the calculate_interest function" in prompt
    assert "validate_token method in the JWTAuthenticator class" in prompt
    
    # Verify RequestCodeChange example
    assert "Add error handling to the process_payment method" in prompt
    assert "Add import for the logging module" in prompt
    
    # Verify JSON format
    assert "```json" in prompt
    assert "scratch_pad" in prompt
    assert "class_name" in prompt
    assert "function_name" in prompt
    assert "file_path" in prompt
    assert "action_type" in prompt
    
    # Verify action types are present
    assert "FindClass" in prompt
    assert "FindFunction" in prompt
    assert "RequestCodeChange" in prompt
