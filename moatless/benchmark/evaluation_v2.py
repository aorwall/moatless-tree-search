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
from moatless.benchmark.repository import EvaluationRepository, EvaluationFileRepository

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
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        repo_base_dir: Union[str, None] = None,
        num_workers: int = 1,
        use_testbed: bool = False,
    ):
        self._event_handlers: List[Callable[[EvaluationEvent], None]] = []
        self.repository = repository
        self.dataset_name = dataset_name
        self.repo_base_dir = repo_base_dir
        self.num_workers = num_workers
        self.use_testbed = use_testbed

    def add_event_handler(self, handler: Callable[[EvaluationEvent], None]):
        """Add an event handler to receive evaluation events"""
        self._event_handlers.append(handler)

    def emit_event(self, evaluation_name: str, event_type: str, data: Any = None):
        """Emit an event to all registered handlers"""
        logger.info(f"Emitting event {event_type}")
        event = EvaluationEvent(
            evaluation_name=evaluation_name,
            event_type=event_type,
            data=data
        )
        for handler in self._event_handlers:
            handler(event)

    def run_evaluation(self, evaluation: Evaluation, rerun_errors: bool = False):
        """Run the evaluation process."""
        # Create evaluation directory if it doesn't exist
        os.makedirs(self.repository.get_evaluation_dir(evaluation.evaluation_name), exist_ok=True)
        
        evaluation.start_time = datetime.now(timezone.utc)
        evaluation.status = EvaluationStatus.RUNNING
        self.repository.save_evaluation(evaluation)
        self.emit_event(evaluation.evaluation_name, "evaluation_started")
        error = 0

        results = []
        instances = self.repository.list_instances(evaluation.evaluation_name)
        logger.info(f"Processing {len(instances)} instances with {self.num_workers} workers")

        # If rerun_errors is True, reset error instances and remove their directories
        if rerun_errors:
            for instance in instances:
                if instance.status == InstanceStatus.ERROR:
                    # Reset instance status
                    instance.status = InstanceStatus.PENDING
                    instance.started_at = None
                    instance.completed_at = None
                    instance.error = None
                    instance.duration = None
                    instance.benchmark_result = None
                    
                    # Remove instance directory
                    self.repository.delete_instance(evaluation.evaluation_name, instance.instance_id)
                    # Save updated instance
                    self.repository.save_instance(evaluation.evaluation_name, instance)

            self.repository.save_evaluation(evaluation)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.evaluate_instance, evaluation, instance.instance_id)
                for instance in instances
            ]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.emit_event(evaluation.evaluation_name, "instance_completed", result)
                        # Save evaluation state after each instance
                        self.repository.save_evaluation(evaluation)
                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")
                    self.emit_event(evaluation.evaluation_name, "instance_error", {"error": traceback.format_exc()})
                    # Save evaluation state even if there was an error
                    self.repository.save_evaluation(evaluation)

        logger.info(f"Completed processing with {error} errors")
        evaluation.status = EvaluationStatus.COMPLETED if error == 0 else EvaluationStatus.ERROR
        evaluation.finish_time = datetime.now(timezone.utc)
        self.repository.save_evaluation(evaluation)
        self.emit_event(evaluation.evaluation_name, "evaluation_completed", {
            "total_instances": len(instances),
            "errors": error
        })

    def evaluate_instance(self, evaluation: Evaluation, instance_id: str):
        """Evaluate a single instance."""
        try:
            moatless_instance = get_moatless_instance(instance_id=instance_id)
            problem_statement = f"<task>\nSolve the following reported issue in the {moatless_instance['repo']} repository:\n\n{moatless_instance['problem_statement']}\n</task>"

            instance = self.repository.load_instance(evaluation.evaluation_name, instance_id)
            if not instance:
                instance = EvaluationInstance(instance_id=instance_id)
            
            instance.start()
            self.emit_event(evaluation.evaluation_name, "instance_started", {"instance_id": instance_id})
            
            # Create instance directory and evaluation result
            instance_dir = self.repository.get_instance_dir(evaluation.evaluation_name, instance_id)
            trajectory_path = os.path.join(instance_dir, "trajectory.json")
            eval_result_path = os.path.join(instance_dir, "eval_result.json")
            
            # Create directories if they don't exist
            os.makedirs(instance_dir, exist_ok=True)

            # Save initial instance state
            self.repository.save_instance(evaluation.evaluation_name, instance)
            
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
                    "evaluation_name": evaluation.evaluation_name,
                    "instance_id": instance_id,
                }

            repository = create_repository(
                moatless_instance, repo_base_dir=self.repo_base_dir
            )
            code_index = create_index(moatless_instance, repository=repository)

            if self.use_testbed:
                from moatless.runtime.testbed import TestbedEnvironment

                run_id = hashlib.sha256(evaluation.evaluation_name.encode()).hexdigest()[:8]

                runtime = TestbedEnvironment(
                    repository=repository,
                    instance=moatless_instance,
                    dataset_name=self.dataset_name,
                    run_id=run_id,
                )
            else:
                runtime = None
            
            agent = CodingAgent.create(
                completion_model=evaluation.settings.agent_settings.completion_model,
                repository=repository,
                code_index=code_index,
                runtime=runtime,
                message_history_type=evaluation.settings.agent_settings.message_history_type,
            )

            # Create search tree with trajectory file in instance directory
            tree = SearchTree.create(
                message=problem_statement,
                repository=repository,
                runtime=runtime,
                selector=evaluation.settings.selector,
                agent=agent,
                value_function=evaluation.settings.value_function,
                max_iterations=evaluation.settings.max_iterations,
                max_expansions=evaluation.settings.max_expansions,
                max_cost=evaluation.settings.max_cost,
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
                    self.emit_event(evaluation.evaluation_name, "tree_progress", {
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
                
                if best_node:
                    eval_result["selected_node"] = best_node.node_id

                # Evaluate all leaf nodes if using testbed
                if self.use_testbed:
                    eval_result = self.evaluate_nodes(
                        evaluation_name=evaluation.evaluation_name,
                        instance_id=instance_id,
                        instance=moatless_instance,
                        tree=tree,
                        eval_result=eval_result,
                        eval_result_path=eval_result_path,
                        runtime=runtime,
                        repository=repository
                    )

                benchmark_result = to_result(tree, eval_report=eval_result)

                # Complete instance with result
                instance.complete(resolved=benchmark_result.resolved, benchmark_result=benchmark_result)
                self.emit_event(evaluation.evaluation_name, "instance_completed", {
                    "instance_id": instance_id,
                    "resolved": instance.resolved,
                    "benchmark_result": benchmark_result.dict() if benchmark_result else None
                })

            except Exception as e:
                eval_result["status"] = "error"
                eval_result["error"] = traceback.format_exc()
                eval_result["duration"] = time.time() - start_time
                logger.exception(f"Error in search tree execution")
                
                raise
            
            finally:
                # Save final instance state
                self.repository.save_instance(evaluation.evaluation_name, instance)
                
                # Save evaluation result
                with open(eval_result_path, "w") as f:
                    json.dump(eval_result, f, indent=2)
                
                # Clean up
                del runtime
                del repository
                del tree
                gc.collect()

        except Exception as e:
            instance.error = str(e)
            instance.complete(resolved=False)
            self.repository.save_instance(evaluation.evaluation_name, instance)  # Save failed state
            self.emit_event(evaluation.evaluation_name, "instance_error", {
                "instance_id": instance_id,
                "error": str(e)
            })
            raise

    def evaluate_nodes(
        self,
        evaluation_name: str,
        instance_id: str,
        instance: dict,
        tree: SearchTree,
        eval_result: dict,
        eval_result_path: str,
        runtime: Any = None,
        repository: Any = None,
    ):
        """Evaluate all leaf nodes using the testbed."""
        leaf_nodes = tree.get_leaf_nodes()
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
            return

        logger.info(
            f"Found {len(leaf_nodes) - len(unevaluated_nodes)} already evaluated nodes, "
            f"will evaluate remaining {len(unevaluated_nodes)} nodes for instance {instance_id}"
        )

        # Create runtime if not provided
        if not runtime and repository:
            from moatless.runtime.testbed import TestbedEnvironment
            runtime = TestbedEnvironment(
                repository=repository,
                instance=instance,
                dataset_name=self.dataset_name,
                enable_cache=True,
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
                    eval_result["node_results"][leaf_node.node_id] = patch_results[patch_hash]
                else:
                    start_time = time.time()
                    result = runtime.evaluate(patch=patch)
                    if not result:
                        logger.error(f"Error in evaluating patch for {instance_id}")
                        continue

                    eval_result["node_results"][leaf_node.node_id] = result.model_dump()
                    patch_results[patch_hash] = result.model_dump()
                    logger.info(
                        f"Evaluated patch in {time.time() - start_time} seconds (resolved: {result.resolved})"
                    )
            else:
                logger.info(
                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} with no patch."
                )

            with open(eval_result_path, "w") as f:
                json.dump(eval_result, f, indent=2)

            return eval_result

    def read_trajectory(self, path) -> dict | None:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        else:
            return None


def create_evaluation_name(
    model: str,
    date: str | None = None,
    max_expansions: int | None = None,
    response_format: LLMResponseFormat | None = None,
    message_history: MessageHistoryType | None = None,
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

    return "_".join(components)
