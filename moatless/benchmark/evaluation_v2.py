import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, Union, Callable, List
from dataclasses import dataclass

import random
import litellm
from pydantic import BaseModel, Discriminator, Field, ConfigDict
from tqdm.auto import tqdm

from moatless.agent.agent import ActionAgent
from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.report import (
    BenchmarkResult,
    to_dataframe,
    create_sha256_hash,
    to_result,
)
from moatless.benchmark.swebench import (
    create_repository,
    create_index,
)
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel
from moatless.completion.log_handler import LogHandler
from moatless.discriminator import Discriminator, AgentDiscriminator
from moatless.feedback.feedback import FeedbackGenerator
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.feedback.reward_feedback import RewardFeedbackGenerator
from moatless.schema import MessageHistoryType
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector, SoftmaxSelector, Selector, LLMSelector, FeedbackSelector
from moatless.value_function.base import ValueFunction
from moatless.value_function.coding import CodingValueFunction
from moatless.benchmark.instance_collections import sampled_50_instances
from moatless.agent.settings import AgentSettings

logger = logging.getLogger(__name__)

__all__ = [
    'TreeSearchSettings',
    'Evaluation',
    'create_evaluation_name',
    'InstanceStatus',
    'EvaluationStatus',
    'EvaluationEvent'
]


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, MessageHistoryType):
            return obj.value
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class DebateSettings(BaseModel):
    n_agents: int = Field(
        8,
        description="The number of agents to debate the rewards to transitions.",
    )
    n_rounds: int = Field(
        3,
        description="The number of rounds to debate the rewards to transitions.",
    )


class TreeSearchSettings(BaseModel):
    max_expansions: int = Field(
        3,
        description="The maximum number of expansions of one state.",
    )

    max_iterations: int = Field(
        100,
        description="The maximum number of iterations to run the tree search.",
    )

    max_cost: float = Field(
        4,
        description="The maximum cost spent on tokens before finishing.",
    )

    min_finished_nodes: Optional[int] = Field(
        2,
        description="The minimum number of finished nodes to consider before finishing",
    )

    max_finished_nodes: Optional[int] = Field(
        3,
        description="The maximum number of finished nodes to consider before finishing",
    )

    reward_threshold: Optional[int] = Field(
        None,
        description="The min reward threshold to consider before finishing.",
    )

    max_depth: int = Field(
        20,
        description="The maximum depth for one trajectory in simulations.",
    )

    model: Optional[CompletionModel] = Field(
        default=None,
        description="The default model.",
    )

    agent_settings: AgentSettings = Field(
        ...,
        description="Settings for creating the agent"
    )

    selector: Optional[Selector] = Field(default=None, description="Custom selector for tree search")

    value_function: Optional[ValueFunction] = Field(
        None,
        description="The value function to use for the tree search.",
    )

    discriminator: Optional[Discriminator] = Field(
        None,
        description="The discriminator to use for the tree search.",
    )   

    feedback_generator: Optional[FeedbackGenerator] = Field(
        None,
        description="The feedback generator to use for the tree search.",
    )


class InstanceStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"

class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class EvaluationEvent:
    """Event emitted by the evaluation process"""
    evaluation_name: str
    event_type: str
    data: Any

class EvaluationInstance(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    instance_id: str = Field(description="Unique identifier for the instance")
    status: InstanceStatus = Field(default=InstanceStatus.PENDING, description="Current status of the instance")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the instance was created")
    started_at: Optional[datetime] = Field(default=None, description="When evaluation started")
    completed_at: Optional[datetime] = Field(default=None, description="When evaluation completed")
    submission: Optional[str] = Field(default=None, description="The submitted patch")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")
    resolved: Optional[bool] = Field(default=None, description="Whether the instance was resolved")
    duration: Optional[float] = Field(default=None, description="Time taken to evaluate in seconds")
    benchmark_result: Optional[BenchmarkResult] = Field(default=None, description="Benchmark result for this instance")

    def start(self):
        self.status = InstanceStatus.STARTED
        self.started_at = datetime.now(timezone.utc)

    def complete(self, submission: Optional[str] = None, resolved: Optional[bool] = None, benchmark_result: Optional[BenchmarkResult] = None):
        self.status = InstanceStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.submission = submission
        self.resolved = resolved
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        if benchmark_result:
            self.benchmark_result = benchmark_result

    def fail(self, error: str):
        self.status = InstanceStatus.ERROR
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()

class Evaluation(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value,
        }
    )

    evaluations_dir: str = Field(description="Directory where evaluations are stored")
    evaluation_name: str = Field(description="Name of the evaluation")
    settings: TreeSearchSettings = Field(description="Tree search settings")
    dataset_name: str = Field(default="princeton-nlp/SWE-bench_Lite", description="Name of the dataset")
    repo_base_dir: Union[str, None] = Field(default=None, description="Base directory for repositories")
    num_workers: int = Field(default=1, description="Number of workers for parallel processing")
    use_testbed: bool = Field(default=False, description="Whether to use testbed")
    instances: dict[str, EvaluationInstance] = Field(default_factory=dict, description="Dictionary of instances by instance_id")
    start_time: Optional[datetime] = Field(default=None, description="When the evaluation started")
    finish_time: Optional[datetime] = Field(default=None, description="When the evaluation finished")
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING, description="Current status of the evaluation")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._event_handlers: List[Callable[[EvaluationEvent], None]] = []

    @classmethod
    def load(cls, evaluations_dir: str, evaluation_name: str) -> "Evaluation":
        """Load an evaluation from a file."""
        logger.info(f"Loading evaluation {evaluation_name} from {evaluations_dir}")
        eval_path = os.path.join(evaluations_dir, evaluation_name, "evaluation.json")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
            
        try:
            with open(eval_path, 'r') as f:
                data = json.load(f)
                return cls.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in evaluation file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading evaluation: {e}")

    def add_event_handler(self, handler: Callable[[EvaluationEvent], None]):
        """Add an event handler to receive evaluation events"""
        self._event_handlers.append(handler)

    def emit_event(self, event_type: str, data: Any = None):
        """Emit an event to all registered handlers"""
        logger.info(f"Emitting event {event_type} with data {data}")
        event = EvaluationEvent(
            evaluation_name=self.evaluation_name,
            event_type=event_type,
            data=data
        )
        for handler in self._event_handlers:
            handler(event)

    @classmethod
    def create(
        cls,
        evaluations_dir: str,
        evaluation_name: str,
        settings: TreeSearchSettings,
        split: str = "lite",
        instance_ids: list[str] | None = None,
        exclude_instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        ignore_repos: list[str] | None = None,
        min_resolved: Optional[int] = None,
        max_resolved: Optional[int] = None,
        **kwargs
    ) -> "Evaluation":
        # Load and filter instances based on split
        if split == "combo":
            # Load both lite and verified datasets
            lite_path = os.path.join(os.path.dirname(__file__), "swebench_lite_all_evaluations.json")
            verified_path = os.path.join(os.path.dirname(__file__), "swebench_verified_all_evaluations.json")
            
            with open(lite_path) as f:
                lite_instances = json.load(f)
            with open(verified_path) as f:
                verified_instances = json.load(f)
                
            # Get instance IDs that exist in both datasets
            lite_ids = {instance["instance_id"] for instance in lite_instances}
            verified_ids = {instance["instance_id"] for instance in verified_instances}
            common_ids = lite_ids.intersection(verified_ids)
            
            # Use instances from lite dataset that exist in both
            raw_instances = [instance for instance in lite_instances if instance["instance_id"] in common_ids]
            logger.info(f"Found {len(raw_instances)} instances that exist in both lite and verified datasets")
        else:
            file_path = os.path.join(os.path.dirname(__file__), f"swebench_lite_all_evaluations.json")
            with open(file_path) as f:
                raw_instances = json.load(f)
            logger.info(f"Loaded {len(raw_instances)} instances from {file_path}")

        random.shuffle(raw_instances)

        # Apply all filters
        if instance_ids:
            raw_instances = [
                instance
                for instance in raw_instances
                if instance["instance_id"] in instance_ids
            ]
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances filtered by instance_ids"
            )

        if exclude_instance_ids:
            raw_instances = [
                instance
                for instance in raw_instances
                if instance["instance_id"] not in exclude_instance_ids
            ]
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances filtered by exclude_instance_ids"
            )

        if min_resolved is not None:
            raw_instances = [
                instance
                for instance in raw_instances
                if len(instance["resolved_by"]) >= min_resolved
                or (
                    min_resolved == 1
                    and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0
                )
            ]
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances filtered by min_resolved >= {min_resolved}"
            )

        if max_resolved is not None:
            raw_instances = [
                instance
                for instance in raw_instances
                if len(instance["resolved_by"]) <= max_resolved
            ]
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances filtered by max_resolved <= {max_resolved}"
            )
        
        if split == "sampled_50_instances":
            raw_instances = [
                instance for instance in raw_instances 
                if instance["instance_id"] in sampled_50_instances
            ]
            logger.info(f"Running evaluation for {len(raw_instances)} instances from sampled_50_instances")

        if repos:
            raw_instances = [
                instance for instance in raw_instances if instance["repo"] in repos
            ]
            logger.info(
                f"Running evaluation for {len(raw_instances)} instances filtered by repos"
            )

        if ignore_repos:
            raw_instances = [
                instance
                for instance in raw_instances
                if instance["repo"] not in ignore_repos
            ]
            if raw_instances:
                logger.info(
                    f"Running evaluation for {len(raw_instances)} instances after filtering by ignore_repos"
                )

        # After all filters, apply random sampling if requested
        if split == "random":
            raw_instances = random.sample(raw_instances, min(50, len(raw_instances)))
            logger.info(f"Randomly selected {len(raw_instances)} instances from filtered dataset")

        random.shuffle(raw_instances)

        # Create instances dictionary
        instances = {
            instance["instance_id"]: EvaluationInstance(instance_id=instance["instance_id"])
            for instance in raw_instances
        }

        # Create evaluation object with filtered instances
        evaluation = cls(
            evaluations_dir=evaluations_dir,
            evaluation_name=evaluation_name,
            settings=settings,
            instances=instances,
            **kwargs
        )
        
        if not os.path.exists(evaluation.evaluation_dir):
            os.makedirs(evaluation.evaluation_dir, exist_ok=True)
            
        evaluation._save_evaluation_state()
        return evaluation

    @property
    def evaluation_dir(self):
        return os.path.join(self.evaluations_dir, self.evaluation_name)
    
    def _run_evaluation(self, rerun_errors: bool = False):
        # Create evaluation directory if it doesn't exist
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        self.start_time = datetime.now(timezone.utc)
        self.status = EvaluationStatus.RUNNING
        self.emit_event("evaluation_started")
        error = 0

        results = []
        logger.info(f"Processing {len(self.instances)} instances with {self.num_workers} workers")

        # If rerun_errors is True, reset error instances and remove their directories
        if rerun_errors:
            for instance_id, instance in self.instances.items():
                if instance.status == InstanceStatus.ERROR:
                    # Reset instance status
                    instance.status = InstanceStatus.PENDING
                    instance.started_at = None
                    instance.completed_at = None
                    instance.error = None
                    instance.duration = None
                    instance.benchmark_result = None
                    
                    # Remove instance directory if it exists
                    instance_dir = os.path.join(self.evaluation_dir, instance_id)
                    if os.path.exists(instance_dir):
                        shutil.rmtree(instance_dir)

            self._save_evaluation_state()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.evaluate_instance, instance_id)
                for instance_id in self.instances.keys()
            ]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.emit_event("instance_completed", result)
                        # Save evaluation state after each instance
                        self._save_evaluation_state()
                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")
                    self.emit_event("instance_error", {"error": traceback.format_exc()})
                    # Save evaluation state even if there was an error
                    self._save_evaluation_state()

        logger.info(f"Completed processing with {error} errors")
        self.status = EvaluationStatus.COMPLETED if error == 0 else EvaluationStatus.ERROR
        self.finish_time = datetime.now(timezone.utc)
        self.emit_event("evaluation_completed", {
            "total_instances": len(self.instances),
            "errors": error
        })

    def _save_evaluation_state(self):
        """Save the current state of the evaluation to a file."""
        eval_file = os.path.join(self.evaluation_dir, "evaluation.json")
        with open(eval_file, "w") as f:
            json.dump(self.model_dump(), f, cls=DateTimeEncoder, indent=2)
        logger.debug("Saved evaluation state to %s", eval_file)

    def evaluate_instance(self, instance_id: str):
        """Evaluate a single instance."""
        try:
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            instance = self.instances[instance_id]
            instance.start()
            self.emit_event("instance_started", {"instance_id": instance_id})
            
            # Create instance directory and evaluation result
            instance_dir = os.path.join(self.evaluation_dir, instance_id)
            trajectory_path = os.path.join(instance_dir, "trajectory.json")
            eval_result_path = os.path.join(instance_dir, "eval_result.json")
            
            # Create directories if they don't exist
            os.makedirs(instance_dir, exist_ok=True)
            
            # Initialize or load eval_result
            eval_result = {
                "node_results": {},
                "status": "started",
                "start_time": datetime.now(timezone.utc).isoformat()
            }
            
            if os.path.exists(eval_result_path):
                try:
                    with open(eval_result_path) as f:
                        eval_result = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            metadata: dict[str, Any] = {
                    "evaluation_name": self.evaluation_name,
                    "instance_id": instance_id,
                }

            repository = create_repository(
                moatless_instance, repo_base_dir=self.repo_base_dir
            )
            code_index = create_index(moatless_instance, repository=repository)

            if self.use_testbed:
                from moatless.runtime.testbed import TestbedEnvironment

                run_id = hashlib.sha256(self.evaluation_name.encode()).hexdigest()[:8]

                runtime = TestbedEnvironment(
                    repository=repository,
                    instance=moatless_instance,
                    dataset_name=self.dataset_name,
                    run_id=run_id,
                )
            else:
                runtime = None
            
            agent = CodingAgent.create(
                completion_model=self.settings.agent_settings.completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=self.settings.agent_settings.message_history_type,
            )

            # Create search tree with trajectory file in instance directory
            tree = SearchTree.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                selector=self.settings.selector,
                agent=agent,
                value_function=self.settings.value_function,
                max_iterations=self.settings.max_iterations,
                max_expansions=self.settings.max_expansions,
                max_cost=self.settings.max_cost,
                persist_path=trajectory_path,
                metadata=metadata
            )

            # Add event handler to forward tree events
            def tree_event_handler(event):
                if event["event_type"] == "tree_iteration":
                    # Generate benchmark result after each iteration
                    from moatless.benchmark.report import to_result
                    benchmark_result = to_result(tree)
                    instance.benchmark_result = benchmark_result
                    
                    # Emit event with benchmark result
                    self.emit_event("tree_progress", {
                        "instance_id": instance_id,
                        "iteration": event["data"]["iteration"],
                        "total_cost": event["data"]["total_cost"],
                        "best_reward": event["data"]["best_reward"],
                        "finished_nodes": event["data"]["finished_nodes"],
                        "total_nodes": event["data"]["total_nodes"],
                        "best_node_id": event["data"]["best_node_id"],
                        "benchmark_result": benchmark_result.dict() if benchmark_result else None
                    })

            tree.add_event_handler(tree_event_handler)

            # Run search
            start_time = time.time()
            try:
                best_node = tree.run_search()
                
                # Generate final benchmark result
                from moatless.benchmark.report import to_result
                benchmark_result = to_result(tree)
                
                # Update eval_result with completion info
                eval_result["status"] = "completed"
                eval_result["duration"] = time.time() - start_time
                eval_result["benchmark_result"] = benchmark_result.dict() if benchmark_result else None
                
                if best_node:
                    patch = best_node.file_context.generate_git_patch()
                    eval_result["selected_node"] = best_node.node_id
                    eval_result["patch"] = patch
                
                # Complete instance with result
                instance.complete(resolved=bool(best_node), benchmark_result=benchmark_result)
                self.emit_event("instance_completed", {
                    "instance_id": instance_id,
                    "resolved": instance.resolved,
                    "benchmark_result": benchmark_result.dict() if benchmark_result else None
                })
                
            except Exception as e:
                eval_result["status"] = "error"
                eval_result["error"] = str(e)
                eval_result["duration"] = time.time() - start_time
                logger.error(f"Error in search tree execution: {e}")
                logger.error(traceback.format_exc())
                raise
            
            finally:
                # Save evaluation result
                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)
                
                # Clean up
                # if repository:
                #    shutil.rmtree(repository.repo_dir, ignore_errors=True)
                
                del runtime
                del repository
                del tree
                gc.collect()

        except Exception as e:
            instance.error = str(e)
            instance.complete(resolved=False)
            self.emit_event("instance_error", {
                "instance_id": instance_id,
                "error": str(e)
            })
            raise

    def _create_completion_model(
        self, model_settings: CompletionModel | None = None
    ) -> CompletionModel:
        return model_settings or self.settings.model


    def _save_json_report(self, results: list[BenchmarkResult]):
        json_results = [result.model_dump() for result in results]
        with open(f"{self.evaluation_dir}/report.json", "w") as f:
            json.dump(json_results, f, indent=2)

    def read_trajectory(self, path) -> dict | None:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        else:
            return None

    def get_actions(self, trajectory: dict):
        actions = []
        for transition in trajectory["transitions"]:
            for action in transition["actions"]:
                actions.append(action["action"])
        return actions

    

def create_evaluation_name(
    model: str,
    date,
    max_expansions=None,
    **kwargs,
):
    if date:
        date_str = date
    else:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    # Make model name URL-safe (only alphanumeric and underscores)
    model_name = model.split("/")[-1]
    # Replace any non-alphanumeric chars with underscore
    model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    # Remove repeated underscores and any leading/trailing underscores
    model_name = "_".join(filter(None, model_name.split("_"))).strip("_")

    model_name = f"{date_str}_{model_name}"

    if max_expansions:
        model_name += f"_max_exp{max_expansions}"

    for key, value in kwargs.items():
        # Convert key-value pairs to URL-safe format
        safe_value = "".join(c if c.isalnum() else "_" for c in str(value))
        safe_value = "_".join(filter(None, safe_value.split("_"))).strip("_")
        model_name += f"_{key}_{safe_value}"
    return model_name.lower()  # Convert to lowercase for consistency
