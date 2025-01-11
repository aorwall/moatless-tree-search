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
from moatless.benchmark.schema import (
    TreeSearchSettings,
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    EvaluationEvent,
)
from moatless.exceptions import RuntimeError

from moatless.benchmark.swebench import (
    create_repository,
    create_index,
)
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.discriminator import Discriminator, AgentDiscriminator
from moatless.feedback.feedback import FeedbackGenerator
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.feedback.reward_feedback import RewardFeedbackGenerator
from moatless.runtime.testbed import TestbedEnvironment
from moatless.schema import MessageHistoryType
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector, SoftmaxSelector, Selector, LLMSelector, FeedbackSelector
from moatless.value_function.base import ValueFunction
from moatless.value_function.coding import CodingValueFunction
from moatless.benchmark.instance_collections import sampled_50_instances
from moatless.agent.settings import AgentSettings
from moatless.benchmark.repository import EvaluationRepository, EvaluationFileRepository

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)

__all__ = [
    'TreeSearchSettings',
    'Evaluation',
    'create_evaluation_name',
    'InstanceStatus',
    'EvaluationStatus',
    'EvaluationEvent'
]

class EvaluationRunner:
    def __init__(
        self,
        repository: EvaluationFileRepository,
        evaluation: Evaluation,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        repo_base_dir: Union[str, None] = None,
        num_workers: int = 1,
        use_testbed: bool = False,
        rerun_errors: bool = True
    ):
        self._event_handlers: List[Callable[[EvaluationEvent], None]] = []
        self._event_lock = threading.Lock()  # Add lock for event handlers
        self._repo_lock = threading.Lock()  # Add lock for repository operations
        self.repository = repository
        self.evaluation = evaluation
        self.dataset_name = dataset_name
        self.repo_base_dir = repo_base_dir
        self.num_workers = num_workers
        self.use_testbed = use_testbed
        self.rerun_errors = rerun_errors

    def add_event_handler(self, handler: Callable[[EvaluationEvent], None]):
        """Add an event handler to receive evaluation events"""
        with self._event_lock:
            self._event_handlers.append(handler)

    def emit_event(self, evaluation_name: str, event_type: str, data: Any = None):
        """Emit an event to all registered handlers"""
        logger.info(f"Emitting event {event_type}")
        event = EvaluationEvent(
            evaluation_name=evaluation_name,
            event_type=event_type,
            data=data
        )
        with self._event_lock:
            for handler in self._event_handlers:
                handler(event)

    def _save_evaluation(self, evaluation: Evaluation):
        """Thread-safe wrapper for saving evaluation"""
        with self._repo_lock:
            self.repository.save_evaluation(evaluation)

    def _save_instance(self, evaluation_name: str, instance: EvaluationInstance):
        """Thread-safe wrapper for saving instance"""
        with self._repo_lock:
            self.repository.save_instance(evaluation_name, instance)

    def _load_instance(self, evaluation_name: str, instance_id: str) -> Optional[EvaluationInstance]:
        return self.repository.load_instance(evaluation_name, instance_id)

    def run_evaluation(self, evaluation: Evaluation | None = None, rerun_errors: bool = False, instance_ids: List[str] = []):
        """Run the evaluation process."""

        if not evaluation:
            evaluation = self.evaluation

        # Create evaluation directory if it doesn't exist
        os.makedirs(self.repository.get_evaluation_dir(evaluation.evaluation_name), exist_ok=True)

        evaluation.start_time = datetime.now(timezone.utc)
        evaluation.status = EvaluationStatus.RUNNING
        self._save_evaluation(evaluation)
        self.emit_event(evaluation.evaluation_name, "evaluation_started")
        error = 0

        results = []
        
        logger.info(f"Processing {len(instance_ids)} instances with {self.num_workers} workers. Rerun error {rerun_errors}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.evaluate_instance, evaluation, instance_id)
                for instance_id in instance_ids
            ]

            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.emit_event(evaluation.evaluation_name, "instance_completed", result.model_dump())
                        # Save evaluation state after each instance
                        self._save_evaluation(evaluation)
                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")
                    self.emit_event(evaluation.evaluation_name, "instance_error", {"error": traceback.format_exc()})
                    # Save evaluation state even if there was an error
                    self._save_evaluation(evaluation)

        logger.info(f"Completed processing with {error} errors")
        evaluation.status = EvaluationStatus.COMPLETED if error == 0 else EvaluationStatus.ERROR
        evaluation.finish_time = datetime.now(timezone.utc)
        self._save_evaluation(evaluation)
        
        # Create final evaluation response
        # self.generate_and_save_evaluation_response(evaluation)
            
        self.emit_event(evaluation.evaluation_name, "evaluation_completed", {
            "total_instances": len(instance_ids),
            "errors": error
        })

    def evaluate_instance(self, evaluation: Evaluation, instance_id: str):
        """Evaluate a single instance."""
        runtime = None
        repository = None
        search_tree = None
        eval_result = None
        try:
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            instance = self._load_instance(evaluation.evaluation_name, instance_id)
            if not instance:
                instance = EvaluationInstance(instance_id=instance_id)
            
            instance_dir = self.repository.get_instance_dir(evaluation.evaluation_name, instance_id)
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
                        logger.info(f"Loading eval_result from {eval_result_path}")
                        eval_result = json.load(f)
                except json.JSONDecodeError:
                    pass

            search_tree = self.create_and_run_search_tree(
                problem_statement=problem_statement,
                evaluation_name=evaluation.evaluation_name,
                instance=instance,
                moatless_instance=moatless_instance,
                trajectory_path=trajectory_path,
                evaluation_settings=evaluation.settings,
            )

            try:
                # Evaluate all leaf nodes if using testbed
                start_time = time.time()
                if self.use_testbed:
                    logger.info(f"Evaluating nodes for instance {instance_id}")
                    eval_result = self.evaluate_nodes(
                        instance_id=instance_id,
                        instance=moatless_instance,
                        search_tree=search_tree,
                        eval_result=eval_result
                    )
            except RuntimeError as e:
                raise e

            except Exception as e:
                eval_result["status"] = "error"
                eval_result["error"] = traceback.format_exc()
                eval_result["duration"] = time.time() - start_time
                logger.exception(f"Error when evaluating nodes for instance {instance_id}")

            benchmark_result = to_result(search_tree, eval_report=eval_result)

            # Complete instance with result
            instance.complete(resolved=benchmark_result.resolved, benchmark_result=benchmark_result)
            self.emit_event(evaluation.evaluation_name, "instance_completed", {
                "instance_id": instance_id,
                "resolved": instance.resolved,
                "benchmark_result": benchmark_result.dict() if benchmark_result else None
            })
            return benchmark_result
    
        except Exception as e:
            stacktrace = traceback.format_exc()
            instance.fail(error=stacktrace)
            self._save_instance(evaluation.evaluation_name, instance)  # Save failed state
            self.emit_event(evaluation.evaluation_name, "instance_error", {
                "instance_id": instance_id,
                "error": str(e)
            })
            raise
        finally:
            # Save final instance state using thread-safe method
            self._save_instance(evaluation.evaluation_name, instance)
            
            if eval_result:
                # Save evaluation result
                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)
            
            # Clean up
            del runtime
            del repository
            del search_tree
            del eval_result
            gc.collect()


    def evaluate_nodes(
        self,
        instance_id: str,
        instance: dict,
        search_tree: SearchTree,
        eval_result: dict,
    ):
        """Evaluate all leaf nodes using the testbed."""
        leaf_nodes = search_tree.get_leaf_nodes()
        patch_results = {}

        # Filter out already evaluated nodes
        unevaluated_nodes = [
            node
            for node in leaf_nodes
            if str(node.node_id) not in eval_result.get("node_results", {})
        ]

        if not unevaluated_nodes:
            logger.info(
                f"All {len(leaf_nodes)} nodes for instance {instance_id} have already been evaluated"
            )
            return eval_result

        logger.info(
            f"Found {len(leaf_nodes) - len(unevaluated_nodes)} already evaluated nodes, "
            f"will evaluate remaining {len(unevaluated_nodes)} nodes for instance {instance_id}"
        )
        repository = create_repository(
            instance, repo_base_dir=self.repo_base_dir
        )
        # TODO: Set run_id on testbed environment
        run_id = hashlib.sha256(self.evaluation.evaluation_name.encode()).hexdigest()[:8]
        runtime = TestbedEnvironment(
            repository=repository,
            instance=instance,
            dataset_name=self.dataset_name,
            # run_id=run_id,
        )

        for i, leaf_node in enumerate(unevaluated_nodes):
            logger.info(
                f"Evaluate Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id}"
            )

            if str(leaf_node.node_id) in eval_result["node_results"]:
                logger.info(
                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} that has already been evaluated"
                )
                continue

            patch = leaf_node.file_context.generate_git_patch(ignore_tests=True)
            if patch and patch.strip():
                patch_hash = create_sha256_hash(patch)

                if patch_hash in patch_results:
                    logger.info(
                        f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} as patch has already been evaluated."
                    )
                    eval_result["node_results"][str(leaf_node.node_id)] = patch_results[patch_hash]
                else:
                    start_time = time.time()
                    result = runtime.evaluate(patch=patch)
                    if not result:
                        logger.error(f"Error in evaluating patch for {instance_id}")
                        continue

                    eval_result["node_results"][str(leaf_node.node_id)] = result.model_dump()
                    patch_results[patch_hash] = eval_result["node_results"][str(leaf_node.node_id)]
                    logger.info(
                        f"Evaluated patch for node {leaf_node.node_id} in {time.time() - start_time} seconds (resolved: {result.resolved})"
                    )
            else:
                logger.info(
                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} with no patch."
                )

        return eval_result

    def create_and_run_search_tree(
        self,
        problem_statement: str,
        evaluation_name: str,
        instance: EvaluationInstance,
        moatless_instance: dict,
        trajectory_path: str,
        evaluation_settings: TreeSearchSettings,
    ) -> SearchTree:
        """Create and run a search tree for the given problem instance."""
        metadata: dict[str, Any] = {
            "evaluation_name": evaluation_name,
            "instance_id": instance.instance_id,
        }

        search_tree = None
        rerun_tree = False
        if os.path.exists(trajectory_path):
            try:
                persisted_tree = SearchTree.from_file(
                    trajectory_path,
                )

                if self.rerun_errors:
                    leaf_nodes = persisted_tree.get_leaf_nodes()
                    for leaf_node in leaf_nodes:
                        if leaf_node.error or (leaf_node.action and leaf_node.action.name == "Error"):
                            rerun_tree = True
                            break
                
                if persisted_tree.is_finished() and not rerun_tree:
                    logger.info(f"Found completed search tree for {instance.instance_id}")
                    return persisted_tree
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse search tree from {trajectory_path}. Will remove file to start over. Error: {e}")
                os.remove(trajectory_path)
                
        repository = create_repository(
            moatless_instance, repo_base_dir=self.repo_base_dir
        )
        code_index = create_index(moatless_instance, repository=repository)

        runtime = None
        if self.use_testbed:
            from moatless.runtime.testbed import TestbedEnvironment
            run_id = hashlib.sha256(evaluation_name.encode()).hexdigest()[:8]
            runtime = TestbedEnvironment(
                repository=repository,
                instance=moatless_instance,
                dataset_name=self.dataset_name,
                run_id=run_id,
            )

        # Load search tree from file again and set repository, runtime and code index
        if os.path.exists(trajectory_path):
            search_tree = SearchTree.from_file(
                trajectory_path,
                repository=repository,
                runtime=runtime,
                code_index=code_index,
            )
            completion_model = evaluation_settings.agent_settings.completion_model.clone()
            completion_model.metadata = {"instance_id": instance.instance_id}

            search_tree.agent = CodingAgent.create(
                completion_model=completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=evaluation_settings.agent_settings.message_history_type,
                thoughts_in_action=evaluation_settings.agent_settings.thoughts_in_action,
            )
        else:
            completion_model = evaluation_settings.agent_settings.completion_model.clone()
            completion_model.metadata = {"instance_id": instance.instance_id}

            agent = CodingAgent.create(
                completion_model=completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=evaluation_settings.agent_settings.message_history_type,
                thoughts_in_action=evaluation_settings.agent_settings.thoughts_in_action,
            )
            search_tree = SearchTree.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                selector=evaluation_settings.selector,
                agent=agent,
                value_function=evaluation_settings.value_function,
                max_iterations=evaluation_settings.max_iterations,
                max_expansions=evaluation_settings.max_expansions,
                max_cost=evaluation_settings.max_cost,
                persist_path=trajectory_path,
                metadata=metadata
            )

        if self.rerun_errors:
            leaf_nodes = search_tree.get_leaf_nodes()
            for leaf_node in leaf_nodes:
                if (leaf_node.error or (leaf_node.action and leaf_node.action.name == "Error")) and leaf_node.parent:
                    # Remove error node from parent's children
                    leaf_node.parent.children = [c for c in leaf_node.parent.children if c.node_id != leaf_node.node_id]
                    logger.info(f"Removed error node {leaf_node.node_id} from parent {leaf_node.parent.node_id} on instance {instance.instance_id}")

        def tree_event_handler(event):
            logger.info(f"Got event {event['event_type']}")
            if event["event_type"] == "tree_iteration":
                instance.usage = search_tree.total_usage()
                instance.iterations = len(search_tree.root.get_all_nodes())
                self._save_instance(evaluation_name, instance)

                logger.info("Emit event tree_progress")
                self.emit_event(evaluation_name, "tree_progress", {
                    "instance_id": instance.instance_id,
                })

        instance.start()
        self.emit_event(evaluation_name, "instance_started", {"instance_id": instance.instance_id})
                
        # Save initial instance state
        self._save_instance(evaluation_name, instance)

        search_tree.add_event_handler(tree_event_handler)
        search_tree.run_search()
        return search_tree

def create_evaluation_name(
    model: str,
    date: str | None = None,
    max_expansions: int | None = None,
    response_format: LLMResponseFormat | None = None,
    message_history: MessageHistoryType | None = None,
    thoughts_in_action: bool | None = None,
) -> str:
    """Create a unique name for an evaluation."""
    if not date:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean model name
    model_name = model.replace("/", "_").replace("-", "_")
    
    # Build name components
    components = [model_name, date]
    
    if max_expansions is not None:
        components.append(f"exp_{max_expansions}")
        
    if response_format:
        components.append(f"fmt_{response_format.value}")
        
    if message_history:
        components.append(f"hist_{message_history.value}")
    
    if thoughts_in_action:
        components.append("thoughts-in-action")

    return "_".join(components)
