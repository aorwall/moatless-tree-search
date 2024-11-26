import logging
import os
from enum import verify

import litellm
from dotenv import load_dotenv
from moatless.benchmark.evaluation import Evaluation, TreeSearchSettings
from moatless.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.schema import MessageHistoryType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

load_dotenv()

def verify_model(model: str, api_key: str = None, base_url: str = None):
    evaluations_dir = "./evaluations"
    # set env vars
    litellm.callbacks = [LogHandler()]

    if api_key and base_url:
        os.environ["CUSTOM_LLM_API_BASE"] = base_url
        os.environ["CUSTOM_LLM_API_KEY"] = api_key

    completion_model = CompletionModel(
        model=model,
        temperature=0.0,
        max_tokens=2000,
        model_api_key=api_key,
        model_base_url=base_url,
        response_format=LLMResponseFormat.REACT
    )
    tree_search_settings = TreeSearchSettings(
        max_iterations=30,
        max_expansions=1,
        agent_message_history_type=MessageHistoryType.REACT,
        model=completion_model
    )

    evaluation = Evaluation(
        evaluations_dir="/home/albert/repos/albert/moatless-tree-search/evaluations",
        evaluation_name="20241125_Qwen2.5-Coder-32B-Instruct_basic_flow_2",
        num_workers=10,
        use_testbed=True,
        repo_base_dir="/tmp/repos",
        settings=tree_search_settings,
        dataset_name="princeton-nlp/SWE-bench_Lite",
    )

    instance_ids = [
        "astropy__astropy-12907",
        "django__django-11583",
        "django__django-11815",
        "django__django-12708",
        "django__django-13028",
        "django__django-13401",
        "django__django-14999",
        "django__django-15790",
        "django__django-16595",
        "django__django-17051",
        "matplotlib__matplotlib-23314",
        "matplotlib__matplotlib-24149",
        "matplotlib__matplotlib-24970",
        "matplotlib__matplotlib-26011",
        "mwaskom__seaborn-3010",
        "psf__requests-1963",
        "pytest-dev__pytest-5692",
        "scikit-learn__scikit-learn-11281",
        "scikit-learn__scikit-learn-25570",
        "sphinx-doc__sphinx-8721",
        "sympy__sympy-12481",
        "sympy__sympy-15011",
        "sympy__sympy-15609",
        "sympy__sympy-15678",
        "sympy__sympy-18057",
        "sympy__sympy-18189",
        "sympy__sympy-18532",
        "sympy__sympy-21055",
        "sympy__sympy-22714"
    ]

    instance_ids = ["django__django-11964", "django__django-11999", "sympy__sympy-20154"]

    evaluation.run_evaluation(
        split="combo",
        #instance_ids=instance_ids,
        #exclude_instance_ids=["sympy__sympy-17655"],
        #max_resolved=24,
        min_resolved=1
    )

#verify_model("gpt-4o-mini-2024-07-18")
#verify_model("deepseek/deepseek-chat")


#verify_model("openrouter/qwen/qwen-2.5-72b-instruct")
verify_model("openai/Qwen/Qwen2.5-Coder-32B-Instruct")

#verify_model("openai/Qwen/Qwen2.5-72B-Instruct")
#verify_model("claude-3-5-sonnet-20241022")
#verify_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
#verify_model("claude-3-5-haiku-20241022")