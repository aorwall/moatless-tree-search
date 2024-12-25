import logging
import os
from enum import verify

import litellm
from tokenizers.models import Model

from moatless.benchmark.evaluation import TreeSearchSettings, Evaluation, create_evaluation_name
from moatless.completion.completion import LLMResponseFormat, CompletionModel
from moatless.completion.log_handler import LogHandler
from moatless.discriminator import MeanAwardDiscriminator
from moatless.feedback.novel_solution_feedback import NovelSolutionFeedbackAgent
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
        temperature=0.0,
        max_tokens=4000,
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
        minimum_reward_threshold=-50,
        high_value_threshold=-25,
        high_value_leaf_bonus_constant=500,
        depth_bonus_factor=200,
        use_average_reward=True
    )

    tree_search_settings = TreeSearchSettings(
        max_iterations=80,
        max_expansions=3,
        min_finished_nodes=2,
        max_finished_nodes=3,
        reward_threshold=95,
        provide_feedback=True,
        max_cost=1.0,
        debate=False,
        best_first=True,
        use_edit_actions=True,
        discriminator=MeanAwardDiscriminator(),
        feedback_generator=NovelSolutionFeedbackAgent(
            completion_model=agent_model
        ),
        value_function="coding",
        agent_message_history_type=MessageHistoryType.MESSAGES,
        model=model,
        agent_model=agent_model
    )

    evaluation_name = tree_search_settings.create_evaluation_name()

    evaluation = Evaluation(
        settings=tree_search_settings,
        evaluations_dir=evaluations_dir,
        evaluation_name="20241222_haiku_tree_search_novel_feedback_temp_0_2",
        max_file_context_tokens=16000,
        num_workers=8,
        use_testbed=True,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        selector=selector
    )

    instance_ids = [
        #"django__django-11848",
        "django__django-11964",
        #"django__django-14999"
    ]
    instance_ids = ["pylint-dev__pylint-7080", "sympy__sympy-21379"] #, "django__django-11999", "sympy__sympy-20154"]

    evaluation.run_evaluation(
        split="combo",
        #instance_ids=instance_ids,
        #exclude_instance_ids=["sympy__sympy-16792"],
        max_resolved=8,
        min_resolved=2
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
