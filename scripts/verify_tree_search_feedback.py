import logging
import os
from enum import verify

import litellm
from tokenizers.models import Model

from moatless.benchmark.evaluation import TreeSearchSettings, Evaluation, create_evaluation_name
from moatless.completion.completion import LLMResponseFormat, CompletionModel
from moatless.completion.log_handler import LogHandler
from moatless.schema import MessageHistoryType
from moatless.selector import BestFirstSelector, LLMSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

def verify_model(model: str, api_key: str = None, base_url: str = None):
    evaluations_dir = "./evaluations"
    litellm.callbacks = [LogHandler()]

    if api_key and base_url:
        os.environ["CUSTOM_LLM_API_BASE"] = base_url
        os.environ["CUSTOM_LLM_API_KEY"] = api_key

    agent_model = CompletionModel(
        model=model,
        temperature=0.7,
        max_tokens=2000,
        model_api_key=api_key,
        model_base_url=base_url,
        response_format=LLMResponseFormat.TOOLS
    )

    model = CompletionModel(
        model=model,
        temperature=0.0,
        max_tokens=2000,
        response_format=LLMResponseFormat.TOOLS
     )

    # Continue exploiting leaf nodes until finished or reward is below 0
    selector = BestFirstSelector(
        high_value_threshold=1,
        high_value_leaf_bonus_constant=500,
        depth_bonus_factor=200,
        use_average_reward=True
    )

    tree_search_settings = TreeSearchSettings(
        max_iterations=125,
        max_expansions=5,
        min_finished_nodes=3,
        max_finished_nodes=5,
        reward_threshold=90,
        provide_feedback=True,
        max_cost=1.0,
        debate=False,
        best_first=True,
        use_edit_actions=True,
        feedback_type="agent",
        value_function="coding",
        agent_message_history_type=MessageHistoryType.MESSAGES,
        model=model,
        agent_model=agent_model
    )

    evaluation_name = tree_search_settings.create_evaluation_name()

    evaluation = Evaluation(
        settings=tree_search_settings,
        evaluations_dir=evaluations_dir,
        evaluation_name="20241216_haiku",
        max_file_context_tokens=16000,
        num_workers=4,
        use_testbed=True,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        selector=selector
    )

    instance_ids = [
        "django__django-11848",
        "django__django-13401",
        "matplotlib__matplotlib-26020",
        "scikit-learn__scikit-learn-25570"
    ]

    instance_ids = [
        "astropy__astropy-12907",
        "django__django-11999",
        "django__django-14672"
    ]

    instance_ids = [
        "django__django-14016",
        "scikit-learn__scikit-learn-25570"
    ]
    instance_ids = ["django__django-11049", "django__django-11179", "django__django-13230", "django__django-14382", "django__django-13447", "django__django-12453", "django__django-13933", "django__django-16041", "django__django-16046", "django__django-16873", "psf__requests-863", "scikit-learn__scikit-learn-13584", "scikit-learn__scikit-learn-14894", "sympy__sympy-14774", "sympy__sympy-23117"]
    instance_ids = ["django__django-16379"]
    evaluation.run_evaluation(
        split="lite",
        instance_ids=instance_ids,
        #max_resolved=20,
        #min_resolved=10
    )

#verify_model("azure/gpt-4o-mini")
#
#verify_model("gpt-4o-mini-2024-07-18")
#verify_model("deepseek/deepseek-chat")

# HYperbolic
#verify_model("openai/Qwen/Qwen2.5-72B-Instruct", api_key="", base_url="https://api.hyperbolic.xyz/v1")
#verify_model("openai/Qwen/Qwen2-Coder-32B-Instruct ", api_key="", base_url="https://api.hyperbolic.xyz/v1")

#verify_model("openrouter/qwen/qwen-2.5-72b-instruct")

#verify_model("openai/Qwen/Qwen2.5-72B-Instruct")
#verify_model("claude-3-5-sonnet-20241022")
#verify_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
verify_model("anthropic.claude-3-5-haiku-20241022-v1:0")
#verify_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
#verify_model("claude-3-5-haiku-20241022")
