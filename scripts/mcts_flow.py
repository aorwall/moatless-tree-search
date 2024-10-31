import logging
import os

from dotenv import load_dotenv

from moatless.agent import CodingAgent
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.discriminator import AgentDiscriminator
from moatless.feedback import FeedbackGenerator
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, RequestMoreContext, RequestCodeChange, Finish, Reject, RunTests
from moatless.value_function import ValueFunction
from testbeds.sdk import TestbedSDK
from moatless.runtime.testbed import TestbedEnvironment

logging.basicConfig(level=logging.INFO)

load_dotenv()

index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")

instance = get_moatless_instance("django__django-14608")

completion_model = CompletionModel(model="gpt-4o-mini", temperature=0.7)

repository = create_repository(instance, repo_base_dir="/tmp/repos")

code_index = CodeIndex.from_index_name(
    instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
)

file_context = FileContext(repo=repository)


selector = BestFirstSelector()

value_function = ValueFunction(completion=completion_model)

discriminator = AgentDiscriminator(
    completion=completion_model,
    n_agents=5,
    n_rounds=3,
)

feedback = FeedbackGenerator()

runtime = TestbedEnvironment(
    testbed_sdk=TestbedSDK(),
    repository=repository,
    instance=instance
)

actions = [
    FindClass(code_index=code_index, repository=repository),
    FindFunction(code_index=code_index, repository=repository),
    FindCodeSnippet(code_index=code_index, repository=repository),
    SemanticSearch(code_index=code_index, repository=repository),
    RequestMoreContext(repository=repository),
    RequestCodeChange(repository=repository, completion_model=completion_model),
    RunTests(code_index=code_index, repository=repository, runtime=runtime),
    Finish(),
    Reject()
]

agent = CodingAgent(actions=actions, completion=completion_model)

search_tree = SearchTree.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    selector=selector,
    value_function=value_function,
    discriminator=discriminator,
    feedback_generator=feedback,
    max_iterations=50,
    max_expansions=3,
    max_depth=25,
    persist_path="traj.json",
)

search_tree.maybe_persist()

node = search_tree.run_search()

search_tree.maybe_persist()
print(node.observation.message)
