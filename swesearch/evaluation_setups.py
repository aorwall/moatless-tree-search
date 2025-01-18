
from moatless.agent.settings import AgentSettings
from moatless.benchmark.schema import TreeSearchSettings
from moatless.completion.base import BaseCompletionModel, LLMResponseFormat
from moatless.schema import CompletionModelSettings, MessageHistoryType
from moatless.selector.simple import SimpleSelector
from swesearch.discriminator import BestRewardDiscriminator
from swesearch.feedback.novel_solution_feedback import NovelSolutionPlanner
from swesearch.value_function.comparison import ComparisonValueFunction


def create_evaluation_setup(config: dict) -> TreeSearchSettings:

    completion_model = CompletionModelSettings(
        model=config["model"],
        temperature=config.get("temperature", 0.0),
        max_tokens=4000,
        model_api_key=config.get("api_key"),
        model_base_url=config.get("base_url"),
        response_format=config.get("response_format"),
        thoughts_in_action=config.get("thoughts_in_action", False),
    )

    agent_settings = AgentSettings(
        completion_model=completion_model,
        message_history_type=config.get("message_history_type", MessageHistoryType.MESSAGES),
        system_prompt=None,
        thoughts_in_action=config.get("thoughts_in_action", False),
    )

    value_completion_model = BaseCompletionModel.create(
        # model="o1-mini",
        model="deepseek/deepseek-chat",
        temperature=0.0,
        max_tokens=4000,
        model_api_key=config.get("api_key"),
        model_base_url=config.get("base_url"),
        response_format=LLMResponseFormat.JSON,
        message_cache=False
    )

    feedback_completion_model = BaseCompletionModel.create(
        # model="o1-preview",
        model="deepseek/deepseek-chat",
        temperature=0.0,
        max_tokens=4000,
        model_api_key=config.get("api_key"),
        model_base_url=config.get("base_url"),
        response_format=LLMResponseFormat.JSON,
        message_cache=False
    )

    value_function = ComparisonValueFunction(completion_model=value_completion_model)
    feedback_generator = NovelSolutionPlanner(completion_model=feedback_completion_model)
    discriminator = BestRewardDiscriminator()
    
    tree_search_settings = TreeSearchSettings(
        max_iterations=80,
        max_expansions=4,
        max_cost=4.0,
        max_depth=10,
        min_finished_nodes=2,
        max_finished_nodes=4,
        reward_threshold=95,
        model=completion_model,
        agent_settings=agent_settings,
        value_function = value_function,
        selector=SimpleSelector(),
        feedback_generator=feedback_generator,
        discriminator=discriminator,
    )

    return tree_search_settings