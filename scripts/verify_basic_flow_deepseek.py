import logging
import os
from enum import verify

import litellm

from moatless.benchmark.evaluation import Evaluation, TreeSearchSettings
from moatless.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.node import MessageHistoryType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")


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
        max_tokens=4000,
        model_api_key=api_key,
        model_base_url=base_url,
        response_format=LLMResponseFormat.REACT
    )
    tree_search_settings = TreeSearchSettings(
        max_iterations=30,
        max_expansions=1,
        use_edit_actions=True,
        #feedback_type="reward",
        agent_message_history_type=MessageHistoryType.REACT,
        model=completion_model
    )

    evaluation = Evaluation(
        evaluations_dir="/home/albert/repos/albert/moatless-tree-search/evaluations",
        evaluation_name="20241120_Qwen2.5-Coder-32B-Instruct_basic_1",
        num_workers=5,
        use_testbed=True,
        repo_base_dir="/tmp/repos2",
        settings=tree_search_settings,
        dataset_name="princeton-nlp/SWE-bench_Lite",
    )

    instance_ids = ["django__django-11049", "django__django-11179", "django__django-13230", "django__django-14382", "django__django-13447", "django__django-12453", "django__django-13933", "django__django-16041", "django__django-16046", "django__django-16873", "psf__requests-863", "scikit-learn__scikit-learn-13584", "scikit-learn__scikit-learn-14894", "sympy__sympy-14774", "sympy__sympy-23117"]
    instance_ids = ["sympy__sympy-13971"]
    evaluation.run_evaluation(
        split="lite",
        #instance_ids=instance_ids,
        exclude_instance_ids=["sympy__sympy-17655"],
        max_resolved=25,
        min_resolved=20
    )

#verify_model("gpt-4o-mini-2024-07-18")
#verify_model("deepseek/deepseek-chat")

#verify_model("openai/Qwen/Qwen2.5-72B-Instruct", api_key="b584e39754fd3799235f22ac9b537c88", base_url="http://avior.mlfoundry.com/live-inference/v1")

#verify_model("openrouter/qwen/qwen-2.5-72b-instruct")
verify_model("openai/Qwen/Qwen2.5-Coder-32B-Instruct", api_key="b584e39754fd3799235f22ac9b537c88", base_url="http://avior.mlfoundry.com/live-inference/v1")

#verify_model("openai/Qwen/Qwen2.5-72B-Instruct")
#verify_model("claude-3-5-sonnet-20241022")
#verify_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
#verify_model("claude-3-5-haiku-20241022")