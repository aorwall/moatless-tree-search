# Moatless Tree Search

### Code for paper [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement](https://arxiv.org/abs/2410.20285)

Note: The original development code can be found at [github.com/a-antoniades/swe-search](https://github.com/a-antoniades/swe-search). It is only intended for reproducing the results in the paper. This is a clean refactor with a modular design, which will be maintained and extended.

##

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-4D9885.svg?style=flat&logo=apache)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2408.08435-B31B1B.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2410.20285)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.moatless.ai/)
[![YouTube](https://img.shields.io/badge/Video-YouTube-FF0000.svg?style=flat&logo=youtube)](https://www.youtube.com/watch?v=VcEHX_TNDgQ)
[![Twitter](https://img.shields.io/badge/Tweet-X-1DA1F2.svg?style=flat&logo=twitter)](https://x.com/anton_iades/status/1852022811113697307)
[![Discord](https://img.shields.io/badge/👾-Discord-7289DA?style=flat)](https://discord.gg/74VX8ppBEg)
</div>

<div align="center">
  <a href="assets/method.pdf" target="_blank">
    <img src="./assets/method.png" alt="Method Diagram" width="100%">
  </a>

  <p>Overview of SWE-Search showing the tree search process, where states (nodes) and actions (edges) are evaluated using contextual information and value function feedback to guide expansion.</p>
</div>

## Installation

Install the package:

```shell
pip install moatless-tree-search
```

### Environment Setup

Before running the evaluation, you'll need:

1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.

**Prerequisites: Docker installed**

Follow these steps to set up the environment using Docker:

1. Clone this repository:
   ```shell
   git clone https://github.com/aorwall/moatless-tree-search.git
   ```

2. Create a `.env` file in the moatless-api directory:

    ```shell
   cp .env.example .env
   ```
   
   ```shell
   # Point to this repository's swesearch directory to extend with SWE-Search components
   # Replace with your actual path to this repository
   export MOATLESS_COMPONENTS_PATH="/path/to/moatless-tree-search"
   
   # Directory where configuration and trajectories will be saved
   # Can be set to the existing .moatless directory in this repo
   export MOATLESS_DIR="/path/to/your/swe-search-2/.moatless"
   
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

   ```

3. Run the server with Docker Compose:
   ```shell
   cd moatless-tree-search
   docker-compose up -d
   ```

4. Check if the server is running by visiting `http://localhost:8000` in your web browser. Go to `http://localhost:5173/settings/components` to verify that all expected components have been initialized.

5. Check logs:
   ```shell
   docker-compose logs api
   ```

6. To shut down the server:
   ```shell
   docker-compose down
   ```

## Evaluation

To run the evaluation script:

**FIXME: Needs to be updated to new version**

```shell
moatless-evaluate \
    --model "gpt-4o-mini" \
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

You can optionally set the `--instance_ids` to evaluate on a specific instance or a list of instances.

Use `--use_testbed` if you got access to a testbed environment. Otherwise, tests will not be run.

## Development

Install with UV:

```shell
uv sync
```

### Apple Silicon

To install the dependencies on Apple Silicon (Mac M1/M2/M3 etc.), you need the following workaround to get the `graphviz` package working:

```shell
brew install graphviz
export CFLAGS="-I $(brew --prefix graphviz)/include"
export LDFLAGS="-L $(brew --prefix graphviz)/lib"
poetry install --with dev
```

## Examples

**FIXME: Needs to be updated to new version**

### Example: Basic Flow

Basic setup similar to the moatless-tools agent.

```python
from moatless.agent.code_agent import CodingAgent
from moatless.agent.code_prompts import SIMPLE_CODE_PROMPT
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.search_tree import SearchTree
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, StringReplace, CreateFile, AppendString, RunTests, Finish, Reject

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
    ViewCode(repository=repository),
    StringReplace(repository=repository, code_index=code_index),
    CreateFile(repository=repository, code_index=code_index),
    AppendString(repository=repository, code_index=code_index),
    RunTests(repository=repository, code_index=code_index),
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
from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion import CompletionModel
from moatless.discriminator import AgentDiscriminator
from moatless.feedback import FeedbackGenerator
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, Reject, RunTests, StringReplace, CreateFile
from moatless.value_function.base import ValueFunction
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
    ViewCode(repository=repository),
    StringReplace(repository=repository, code_index=code_index),
    CreateFile(repository=repository, code_index=code_index),
    RunTests(repository=repository, code_index=code_index),
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

### Citation

```bibtex
@misc{antoniades2024swesearchenhancingsoftwareagents,
      title={SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement}, 
      author={Antonis Antoniades and Albert Örwall and Kexun Zhang and Yuxi Xie and Anirudh Goyal and William Wang},
      year={2024},
      eprint={2410.20285},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.20285}, 
}
```
