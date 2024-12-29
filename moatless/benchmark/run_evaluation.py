import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from moatless.benchmark.evaluation_v2 import (
    create_evaluation_name,
    Evaluation,
    TreeSearchSettings,
    BestFirstSelector
)
from moatless.completion.completion import CompletionModel
from moatless.completion.completion import LLMResponseFormat
from moatless.schema import MessageHistoryType
from moatless.benchmark.schema import EvaluationDatasetSplit

logger = logging.getLogger(__name__)

def load_dataset_split(split: str) -> Optional[EvaluationDatasetSplit]:
    """Load a dataset split from the datasets directory."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets", f"{split}_dataset.json")
    if not os.path.exists(dataset_path):
        return None
    
    with open(dataset_path) as f:
        data = json.load(f)
        return EvaluationDatasetSplit(**data)

def get_missing_instances(evaluation_dir: str, instance_ids: list[str]) -> list[str]:
    """Get instances that haven't been evaluated yet."""
    if not os.path.exists(evaluation_dir):
        return instance_ids
    
    # Get all evaluated instances from the evaluation directory
    evaluated_instances = set()
    for instance_dir in os.listdir(evaluation_dir):
        if os.path.isdir(os.path.join(evaluation_dir, instance_dir)):
            evaluated_instances.add(instance_dir)
    
    # Return instances that haven't been evaluated
    return [instance_id for instance_id in instance_ids if instance_id not in evaluated_instances]

def evaluate_search_and_code(
    instance_ids: Optional[list] = None,
    repo_base_dir: str | None = None,
    use_testbed: bool = False,
    num_workers: int = 4,
    evaluation_name=None,
    evaluations_dir=None,
    date=None,
    tree_search_settings: TreeSearchSettings = None,
    repos: Optional[list[str]] = None,
    split: str = "lite_and_verified_solvable",
    overwrite: bool = False,
    high_value_threshold: float = 50.0,
    high_value_leaf_bonus_constant: float = 50.0,
    use_average_reward: bool = False,
    **kwargs,
):
    selector = None

    temperature = tree_search_settings.model.temperature

    if evaluation_name is None:
        evaluation_name = create_evaluation_name(
            model=tree_search_settings.model.model,
            date=date,
            max_expansions=tree_search_settings.max_expansions,
            debate=tree_search_settings.debate,
            provide_feedback=tree_search_settings.provide_feedback,
            temp_bias=temperature,
            use_testbed=use_testbed,
        )

    # Load instances from dataset if specified
    if split:
        dataset = load_dataset_split(split)
        if dataset is None:
            raise ValueError(f"Dataset split '{split}' not found")
        instance_ids = dataset.instance_ids
    
    # If evaluation exists, only evaluate missing instances
    if not overwrite and evaluations_dir and evaluation_name:
        eval_dir = os.path.join(evaluations_dir, evaluation_name)
        instance_ids = get_missing_instances(eval_dir, instance_ids)
        if not instance_ids:
            logger.info("All instances in the dataset have already been evaluated")
            return eval_dir

    # Expect models with prefix openai/ to be custom
    if tree_search_settings.model.model.startswith("openai/"):
        tree_search_settings.model.model_base_url = os.getenv("CUSTOM_LLM_API_BASE")
        tree_search_settings.model.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")

    logger.info("Evaluation Parameters:")
    logger.info(f"Evalation dir: {evaluations_dir}")
    logger.info(f"Evaluation Name: {evaluation_name}")
    logger.info(f"Dataset Split: {split}")
    logger.info(f"Model: {tree_search_settings.model.model}")
    logger.info(f"Model Base URL: {tree_search_settings.model.model_base_url}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Tree Search Settings:")
    logger.info(f"  Max Expansions: {tree_search_settings.max_expansions}")
    logger.info(f"  Max Iterations: {tree_search_settings.max_iterations}")
    logger.info(f"  Min Finished Nodes: {tree_search_settings.min_finished_nodes}")
    logger.info(f"  Provide Feedback: {tree_search_settings.provide_feedback}")
    if tree_search_settings.value_function_model:
        logger.info(
            f"  Value Function Model: {tree_search_settings.value_function_model.model}"
        )
        logger.info(
            f"  Value Function Model Temperature: {tree_search_settings.value_function_model.temperature}"
        )
    logger.info(f"Max Cost: {tree_search_settings.max_cost}")
    logger.info(f"Max iterations: {tree_search_settings.max_iterations}")
    logger.info(f"Number of Workers: {num_workers}")
    logger.info(f"Use Testbed: {use_testbed}")
    logger.info(f"Number of instances to evaluate: {len(instance_ids) if instance_ids else 0}")

    evaluation = Evaluation(
        settings=tree_search_settings,
        selector=selector,
        evaluations_dir=evaluations_dir,
        evaluation_name=evaluation_name,
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        num_workers=num_workers,
        use_testbed=use_testbed,
        overwrite=overwrite,
    )

    evaluation.run_evaluation(
        instance_ids=instance_ids,
        repos=repos,
        split=split,
    )

    return os.path.join(evaluations_dir, evaluation_name)


# Create a function to ensure the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as e:
        pass

    parser = argparse.ArgumentParser(
        description="Run Moatless evaluation on SWE-Bench instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g., gpt-4, claude-3-opus-20240229)",
    )
    required.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory to store evaluation results",
    )

    # Model settings
    model_group = parser.add_argument_group("model settings")
    model_group.add_argument(
        "--temp", type=float, default=0.2, help="Temperature for model sampling"
    )
    model_group.add_argument(
        "--format",
        type=str,
        choices=["tools", "json", "react"],
        default="tools",
        help="Response format for the model"
    )

    # Search settings
    search_group = parser.add_argument_group("search settings")
    search_group.add_argument(
        "--max_expansions",
        type=int,
        default=30,
        help="Maximum number of expansions per node",
    )
    search_group.add_argument(
        "--min_finished_nodes",
        type=int,
        default=3,
        help="Minimum number of finished nodes before stopping",
    )
    search_group.add_argument(
        "--max_finished_nodes",
        type=int,
        default=5,
        help="Maximum number of finished nodes before stopping",
    )
    search_group.add_argument(
        "--max_iterations", type=int, default=100, help="Maximum number of iterations"
    )
    search_group.add_argument(
        "--max_cost",
        type=float,
        default=2.0,
        help="Maximum cost allowed for the search",
    )
    search_group.add_argument(
        "--reward_threshold",
        type=int,
        default=90,
        help="Minimum reward threshold to consider before finishing",
    )
    search_group.add_argument(
        "--sample_first",
        action="store_true",
        help="Use sampling instead of best-first search",
    )

    # Features
    features_group = parser.add_argument_group("features")
    features_group.add_argument(
        "--debate", action="store_true", help="Enable agent debate"
    )
    features_group.add_argument(
        "--feedback", action="store_true", help="Enable feedback generation"
    )
    features_group.add_argument(
        "--feedback_type",
        type=str,
        choices=["reward", "agent", None],
        default=None,
        help="Type of feedback generator to use",
    )
    features_group.add_argument(
        "--use_testbed", action="store_true", help="Enable testbed for running tests"
    )
    features_group.add_argument(
        "--use_edit_actions",
        action="store_true",
        help="Enable edit actions for the coding agent"
    )

    # Runtime settings
    runtime_group = parser.add_argument_group("runtime settings")
    runtime_group.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers"
    )
    runtime_group.add_argument(
        "--repo_base_dir",
        type=str,
        default=os.getenv("REPO_DIR", "/tmp/repos"),
        help="Base directory for repositories",
    )

    # Instance selection
    instance_group = parser.add_argument_group("instance selection")
    instance_group.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to evaluate",
    )
    instance_group.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="Filter instances by repository names",
    )
    instance_group.add_argument(
        "--split",
        type=str,
        choices=["verified", "lite", "easy", "lite_and_verified", "lite_and_verified_solvable"],
        default="lite_and_verified_solvable",
        help="Dataset split to use (verified, lite, easy, lite_and_verified, lite_and_verified_solvable)",
    )

    # Other settings
    other_group = parser.add_argument_group("other settings")
    other_group.add_argument(
        "--eval_name", type=str, default=None, help="Custom name for the evaluation"
    )
    other_group.add_argument(
        "--date", type=str, default=None, help="Custom date for the evaluation name"
    )
    other_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results if they exist"
    )

    selector_group = parser.add_argument_group("selector settings")
    selector_group.add_argument(
        "--high_value_threshold",
        type=float,
        default=50.0,
        help="Threshold for considering a node's reward as high value"
    )
    selector_group.add_argument(
        "--high_value_leaf_bonus_constant",
        type=float,
        default=50.0,
        help="Bonus constant for high-value leaf nodes"
    )
    selector_group.add_argument(
        "--use_average_reward",
        action="store_true",
        help="Use average reward across trajectory instead of node reward"
    )

    args = parser.parse_args()

    # Verify environment variables
    if args.use_testbed:
        if not os.getenv("TESTBED_API_KEY") or not os.getenv("TESTBED_BASE_URL"):
            parser.error(
                "--use_testbed requires TESTBED_API_KEY and TESTBED_BASE_URL environment variables"
            )

    # Verify model-specific requirements
    if args.model.startswith("openai/"):
        if not os.getenv("CUSTOM_LLM_API_BASE") or not os.getenv("CUSTOM_LLM_API_KEY"):
            parser.error(
                "Custom OpenAI models require CUSTOM_LLM_API_BASE and CUSTOM_LLM_API_KEY environment variables"
            )
    elif args.model.startswith("gpt"):
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OpenAI models require OPENAI_API_KEY environment variable")
    elif args.model.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            parser.error(
                "Anthropic models require ANTHROPIC_API_KEY environment variable"
            )

    # Verify directories exist or can be created
    try:
        os.makedirs(args.eval_dir, exist_ok=True)
        os.makedirs(args.repo_base_dir, exist_ok=True)
    except Exception as e:
        parser.error(f"Failed to create directories: {e}")

    # Update logging configuration
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.eval_name is not None:
        log_filename = f"run_evaluation_{args.eval_name}_{current_time}.log"
        error_log_filename = f"run_evaluation_{args.eval_name}_{current_time}_error.log"
    else:
        log_filename = f"run_evaluation_{current_time}.log"
        error_log_filename = f"run_evaluation_{current_time}_error.log"

    # Ensure the directory for log files exists
    log_dir = os.path.join(args.eval_dir, "logs")
    ensure_dir(os.path.join(log_dir, log_filename))
    ensure_dir(os.path.join(log_dir, error_log_filename))

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    try:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error creating file handler for logging: {e}")
        print("Continuing without file logging...")

    # Create error file handler which logs error messages
    try:
        error_file_handler = logging.FileHandler(
            os.path.join(log_dir, error_log_filename)
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_file_handler.setFormatter(error_file_formatter)
        logger.addHandler(error_file_handler)
    except Exception as e:
        print(f"Error creating error file handler for logging: {e}")
        print("Continuing without error file logging...")

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Adjust log levels for specific loggers
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("moatless").setLevel(logging.WARNING)
    logging.getLogger("moatless.benchmark.evaluation").setLevel(logging.INFO)
    logging.getLogger("moatless.benchmark.run_evaluation").setLevel(logging.INFO)
    # logging.getLogger("mcts_tree").setLevel(logging.INFO)

    def get_response_format(format_str: str) -> LLMResponseFormat:
        format_map = {
            "tools": LLMResponseFormat.TOOLS,
            "json": LLMResponseFormat.JSON,
            "react": LLMResponseFormat.REACT
        }
        return format_map[format_str]

    def get_message_history_type(format_str: str) -> MessageHistoryType:
        # Only use REACT for react format, use MESSAGES for all others
        return MessageHistoryType.REACT if format_str == "react" else MessageHistoryType.MESSAGES

    model_settings = CompletionModel(
        model=args.model, 
        temperature=args.temp, 
        max_tokens=3000,
        response_format=get_response_format(args.format)
    )

    tree_search_settings = TreeSearchSettings(
        max_iterations=args.max_iterations,
        max_expansions=args.max_expansions,
        min_finished_nodes=args.min_finished_nodes,
        max_finished_nodes=args.max_finished_nodes,
        max_cost=args.max_cost,
        reward_threshold=args.reward_threshold,
        provide_feedback=args.feedback,
        feedback_type=args.feedback_type,
        debate=args.debate,
        best_first=True,
        model=model_settings,
        agent_message_history_type=get_message_history_type(args.format)
    )

    evaluate_search_and_code(
        evaluation_name=args.eval_name,
        evaluations_dir=args.eval_dir,
        repo_base_dir=args.repo_base_dir,
        tree_search_settings=tree_search_settings,
        instance_ids=args.instance_ids,
        repos=args.repos,
        date=args.date,
        use_testbed=args.use_testbed,
        num_workers=args.num_workers,
        best_first=not args.sample_first,
        split=args.split,
        high_value_threshold=args.high_value_threshold,
        high_value_leaf_bonus_constant=args.high_value_leaf_bonus_constant,
        use_average_reward=args.use_average_reward,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
