# Moatless Tree Search 

### Code for paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/html/2410.20285v1)

Note: The original development code can be found at [https://github.com/a-antoniades/swe-planner](https://github.com/a-antoniades/swe-planner). It is only intended for reproducing the results in the paper. This is a clean refactor with a modular design, which will be maintained and extended.

<div align="center">

[![License](https://img.shields.io/badge/LICENSE-APACHE_LICENSE_2.0-yellow?style=flat-square&labelColor=lightgrey)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2408.08435-B31B1B?style=flat-square)](https://arxiv.org/html/2410.20285v1)
[![Streamlit](https://img.shields.io/badge/STREAMLIT-7C4DFF?style=flat-square)](https://streamlit.moatless.ai/)
[![Twitter](https://img.shields.io/badge/TWITTER-00ACEE?style=flat-square)](https://twitter.com/your-handle)
[![YouTube](https://img.shields.io/badge/YOUTUBE-FF0000?style=flat-square)](https://www.youtube.com/watch?v=VcEHX_TNDgQ)
</div>

<div align="center">
  <a href="assets/method.pdf" target="_blank">
    <img src="./assets/method.png" alt="Method Diagram" width="90%">
  </a>

  <p>Overview of SWE-Search showing the tree search process, where states (nodes) and actions (edges) are evaluated using contextual information and value function feedback to guide expansion.</p>
</div>

## Installation

Install the package:

```bash
pip install moatless-tree-search
```

### Environment Setup

Before running the evaluation, you'll need:
1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.
3. (Optional) Access to a testbed environment - see [moatless-testbeds](https://github.com/aorwall/moatless-testbeds) for setup instructions

You can configure these settings by either:

1. Create a `.env` file in the project root (copy from `.env.example`):

   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. Or export the variables directly:
   
   ```bash
   # Directory for storing vector index store files  
   export INDEX_STORE_DIR="/tmp/index_store"    

   # Directory for storing clonedrepositories 
   export REPO_DIR="/tmp/repos"

   # Required: At least one LLM provider API key
   export OPENAI_API_KEY="<your-key>"
   export ANTHROPIC_API_KEY="<your-key>"
   export HUGGINGFACE_API_KEY="<your-key>"
   export DEEPSEEK_API_KEY="<your-key>"

   # ...or Base URL for custom LLM API service (optional)
   export CUSTOM_LLM_API_BASE="<your-base-url>"
   export CUSTOM_LLM_API_KEY="<your-key>"

   # Required: API Key for Voyage Embeddings
   export VOYAGE_API_KEY="<your-key>"

   # Optional: Configuration for testbed environment (https://github.com/aorwall/moatless-testbeds)
   export TESTBED_API_KEY="<your-key>"
   export TESTBED_BASE_URL="<your-base-url>"
   ```


## Streamlit

To launch the Streamlit app, run:

```bash
streamlit run -m moatless.streamlit_app
```

The following badges are used to indicate the status of a node:

| Badge | Shape | Color | Description |
|-------|-------|-------|-------------|
| ⭐ | Star | Green | Node is marked as resolved |
| ❌ | X | Red | Invalid edits or failed tests |
| 🟢 | Circle | Green | Correct code spans present in the context |
| 🟡 | Circle | Yellow | Either:<br>• Found files but not spans<br>• Found spans but in wrong files<br>|

## Evaluation

To run the evaluation script, use the following command:

```bash
python -m moatless.benchmark.run_evaluation \
        --model "gpt-4o-mini-2024-07-18" \
        --repo_base_dir /tmp/repos \ 
        --eval_dir "./evaluations" \
        --eval_name mts \
        --temp 0.7 \
        --num_workers 1 \
        --use_testbed \
        --feedback \
        --max_iterations 100 \
        --max_expansions 5
```

You can optionally set the `--instance_id` to evaluate on a specific instance or a list of instances.

## Examples

### Example: Basic Flow
Basic setup similar to the moatless-tools agent.

```python
from moatless.agent import CodingAgent
from moatless.agent.code_prompts import SIMPLE_CODE_PROMPT
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.search_tree import SearchTree
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, RequestMoreContext, RequestCodeChange, Finish, Reject

index_store_dir = "/tmp/index_store"
repo_base_dir = "/tmp/repos"
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")

completion_model = CompletionModel(model="gpt-4o", temperature=0.0)

repository = create_repository(instance)

code_index = CodeIndex.from_index_name(
    instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
)

actions = [
    FindClass(code_index=code_index, repository=repository),
    FindFunction(code_index=code_index, repository=repository),
    FindCodeSnippet(code_index=code_index, repository=repository),
    SemanticSearch(code_index=code_index, repository=repository),
    RequestMoreContext(repository=repository),
    RequestCodeChange(repository=repository, completion_model=completion_model),
    Finish(),
    Reject()
]

file_context = FileContext(repo=repository)
agent = CodingAgent(actions=actions, completion=completion_model, system_prompt=SIMPLE_CODE_PROMPT)

search_tree = SearchTree.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    max_expansions=1,
    max_iterations=50
)

node = search_tree.run_search()
print(node.observation.message)
```

### Example: MCTS Flow

How to setup the evaluation flow with MCTS and testbeds.

```python
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

index_store_dir = "/tmp/index_store"
repo_base_dir = "/tmp/repos"
persist_path = "trajectory.json"

instance = get_moatless_instance("django__django-16379")

completion_model = CompletionModel(model="gpt-4o-mini", temperature=0.7)

repository = create_repository(instance, repo_base_dir=repo_base_dir)

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
    max_iterations=100,
    max_expansions=3,
    max_depth=25,
    persist_path=persist_path,
)

node = search_tree.run_search()
print(node.observation.message)
```
