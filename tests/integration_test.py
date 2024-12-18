import os

import pytest
from dotenv import load_dotenv

from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import load_instance, create_repository
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.index import CodeIndex
from moatless.schema import MessageHistoryType
from moatless.search_tree import SearchTree

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",  # "azure/gpt-4o",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.parametrize(
    "model",
    [
        # "claude-3-5-sonnet-20241022",
        # "claude-3-5-haiku-20241022",
        # "anthropic.claude-3-5-sonnet-20241022-v2:0",
        # "anthropic.claude-3-5-haiku-20241022-v1:0",
        # "claude-3-5-sonnet-20241022",
        # "azure/gpt-4o-mini",
        # "gpt-4o-2024-08-06",
        # "deepseek/deepseek-chat",
        "gemini/gemini-2.0-flash-exp"
    ],
)
@pytest.mark.llm_integration
def test_basic_coding_tree(model):
    completion_model = CompletionModel(model=model, temperature=0.0, response_format=LLMResponseFormat.REACT)

    instance = load_instance("django__django-16379")
    repository = create_repository(instance)

    index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
    code_index = CodeIndex.from_index_name(
        instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
    )

    agent = CodingAgent.create(
        completion_model=completion_model,
        repository=repository,
        code_index=code_index,
        message_history_type=MessageHistoryType.REACT
    )

    persist_path = f"itegration_test_{model.replace('.', '_').replace('/', '_')}.json"

    search_tree = SearchTree.create(
        instance["problem_statement"],
        agent=agent,
        repository=repository,
        max_expansions=1,
        max_iterations=15,
        persist_path=persist_path
    )

    search_tree.maybe_persist()
    node = search_tree.run_search()
    print(node.message)
    search_tree.maybe_persist()
    assert node.action
    assert node.action.name == "Finish"
    assert search_tree.is_finished()
    # print(json.dumps(search_tree.root.model_dump(), indent=2))

