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
from typing import Optional, Any

import litellm
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.report import (
    BenchmarkResult,
    to_dataframe,
    create_sha256_hash,
)
from moatless.benchmark.swebench import (
    create_repository,
    create_index,
)
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel
from moatless.completion.log_handler import LogHandler
from moatless.discriminator import AgentDiscriminator
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.feedback.reward_feedback import RewardFeedbackGenerator
from moatless.schema import MessageHistoryType
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector, SoftmaxSelector, Selector
from moatless.value_function.coding import CodingValueFunction

logger = logging.getLogger(__name__)


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
        0.5,
        description="The maximum cost spent on token before finishing.",
    )

    model: Optional[CompletionModel] = Field(
        default=None,
        description="The default model.",
    )

    agent_model: Optional[CompletionModel] = Field(
        default=None,
        description="The model the agent will use.",
    )

    value_function_model: Optional[CompletionModel] = Field(
        None,
        description="The model to use for building actions.",
    )

    min_finished_nodes: Optional[int] = Field(
        2,
        description="The minimum number of finished nodes to consider before finishing",
    )
    max_finished_nodes: Optional[int] = Field(
        2,
        description="The maximum number of finished nodes to consider before finishing",
    )

    reward_threshold: Optional[int] = Field(
        None,
        description="The min reward threshold to consider before finishing.",
    )

    provide_feedback: bool = Field(
        True,
        description="Whether to provide feedback from previosly evaluated transitions.",
    )

    debate: bool = Field(
        True,
        description="Whether to debate the rewards to transitions.",
    )

    debate_n_agents: Optional[int] = Field(
        8,
        description="The number of agents to debate the rewards to transitions.",
    )

    debate_n_rounds: Optional[int] = Field(
        3,
        description="The number of rounds to debate the rewards to transitions.",
    )

    best_first: bool = Field(
        True,
        description="Whether to use best first search.",
    )

    exploration_constant: float = Field(
        1.41,
        description="The exploration constant for UCT.",
    )

    max_depth: int = Field(
        20,
        description="The maximum depth for one trajectory in simulations.",
    )

    debate_settings: DebateSettings | None = Field(
        None,
        description="The settings for the debate.",
    )

    feedback_type: Optional[str] = Field(
        None,
        description="Type of feedback generator to use ('reward', 'agent', or None).",
    )

    agent_message_history_type: MessageHistoryType = Field(
        MessageHistoryType.MESSAGES,
        description="Determines how message history is generated for the agent.",
    )

    def create_evaluation_name(
        self,
        date: str | None = None,
    ) -> str:
        if date:
            date_str = date
        else:
            date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

        # Make model name URL-safe (only alphanumeric and underscores)
        model_name = self.model.model.split("/")[-1]
        # Replace any non-alphanumeric chars with underscore
        model_name = "".join(c if c.isalnum() else "_" for c in model_name)
        # Remove repeated underscores and any leading/trailing underscores
        model_name = "_".join(filter(None, model_name.split("_"))).strip("_")

        # Start with date and model
        name = f"{date_str}_{model_name}"

        # Add all float and bool fields from the model's schema
        schema = self.model_json_schema()
        for field_name, field in schema["properties"].items():
            if field.get("type") in ["number", "boolean"]:
                # Convert field name to lowercase and replace spaces with underscores
                safe_field = field_name.lower().replace(" ", "_")
                value = getattr(self, field_name)
                # For boolean fields, just include the name if True
                if field["type"] == "boolean" and value:
                    name += f"_{safe_field}"
                # For float fields, include the value
                elif field["type"] == "number":
                    name += f"_{safe_field}_{value}"

        return name.lower()  # Convert to lowercase for consistency


class Evaluation:
    def __init__(
        self,
        evaluations_dir: str,
        evaluation_name: str,
        settings: TreeSearchSettings,
        max_file_context_tokens: int = 16000,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        repo_base_dir: str | None = None,
        report_mode: str | None = None,
        litellm_callback: Optional[str] = None,
        num_workers: int = 1,
        use_testbed: bool = False,
        selector: Selector | None = None,
    ):
        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.report_mode = report_mode
        self.dataset_name = dataset_name
        self.evaluation_name = evaluation_name

        self.use_testbed = use_testbed

        self.settings = settings
        self.max_file_context_tokens = max_file_context_tokens

        self.selector = selector

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        logger.info(f"Evaluation directory: {self.evaluation_dir}")
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(self.evaluation_dir)

        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"

        self.repo_base_dir = repo_base_dir or os.getenv("REPO_DIR", "/tmp/repos")

        if litellm_callback:
            litellm.success_callback = [litellm_callback]
            litellm.failure_callback = [litellm_callback]

        self.status_file = f"{self.evaluation_dir}/status_summary.json"
        self.event_file = f"{self.evaluation_dir}/event_log.json"
        self.file_lock = threading.Lock()
        self.statuses = defaultdict(dict)
        self.events = defaultdict(list)

    def update_status(self, instance_id: str, status: str):
        with self.file_lock:
            if instance_id not in self.statuses:
                self.statuses[instance_id] = {
                    "created": datetime.now().isoformat(),
                }

            self.statuses[instance_id].update(
                {"last_updated": datetime.now().isoformat(), "status": status}
            )
            self._save_statuses()

    def log_event(self, instance_id: str, event: str):
        with self.file_lock:
            self.events[instance_id].append(
                {"timestamp": datetime.now().isoformat(), "event": event}
            )
            self._save_events()

    def _save_statuses(self):
        with open(self.status_file, "w") as f:
            json.dump(self.statuses, f, indent=2)

    def _save_events(self):
        with open(self.event_file, "w") as f:
            json.dump(self.events, f, indent=2)

    def run_evaluation(
        self,
        split: str = "lite",
        instance_ids: list[str] | None = None,
        exclude_instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        ignore_repos: list[str] | None = None,
        min_resolved: Optional[int] = None,
        max_resolved: Optional[int] = None,
    ):
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
            instances = [instance for instance in lite_instances if instance["instance_id"] in common_ids]
            logger.info(f"Found {len(instances)} instances that exist in both lite and verified datasets")
        else:
            file_path = os.path.join(os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json")
            with open(file_path) as f:
                instances = json.load(f)
            logger.info(f"Loaded {len(instances)} instances from {file_path}")

        random.shuffle(instances)


        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by instance_ids"
            )

        if exclude_instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] not in exclude_instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by exclude_instance_ids"
            )

        if min_resolved is not None:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= min_resolved
                or (
                    min_resolved == 1
                    and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0
                )
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by min_resolved >= {min_resolved}"
            )

        if max_resolved is not None:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) <= max_resolved
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by max_resolved <= {max_resolved}"
            )

        if repos:
            instances = [
                instance for instance in instances if instance["repo"] in repos
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by repos"
            )

        if ignore_repos:
            instances = [
                instance
                for instance in instances
                if instance["repo"] not in ignore_repos
            ]

            if instances:
                logger.info(
                    f"Running evaluation for {len(instances)} instances after filtering by ignore_repos"
                )

        return self._run_evaluation(instances)

    def run_single_instance(
        self,
        instance_id: str,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split="test",
    ) -> BenchmarkResult:
        instance = get_moatless_instance(instance_id, split)
        return self.evaluate_instance(instance)

    def evaluate_instance(self, instance: dict):
        instance_id = instance["instance_id"]
        instance_dir = os.path.join(self.evaluation_dir, f"{instance_id}")
        trajectory_path = os.path.join(instance_dir, "trajectory.json")

        if not os.path.exists(self.evaluation_dir):
            os.makedirs(trajectory_path)

        log_dir = os.path.join(instance_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        eval_result_path = os.path.join(instance_dir, "eval_result.json")
        if os.path.exists(eval_result_path):
            try:
                with open(eval_result_path) as f:
                    eval_result = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse eval result from {eval_result_path}. Will remove file to start over. Error: {e}"
                )
                os.remove(eval_result_path)
                eval_result = {
                    "node_results": {},
                }
        else:
            eval_result = {
                "node_results": {},
            }

        logger.info(f"Evaluating {instance_id}")
        problem_statement = f"<task>\n{instance['problem_statement']}\n</task>"

        runtime = None
        repository = None

        self.update_status(instance_id, "started")
        self.log_event(instance_id, "evaluate_instance_initiated")

        try:
            search_tree = None

            if os.path.exists(trajectory_path):
                try:
                    persisted_tree = SearchTree.from_file(trajectory_path)
                    if persisted_tree.is_finished():
                        logger.info(f"Found completed search tree for {instance_id}")
                        search_tree = persisted_tree
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse search tree from {trajectory_path}. Will remove file to start over. Error: {e}"
                    )
                    os.remove(trajectory_path)

            if not search_tree:
                self.log_event(instance_id, "workspace_creation_started")

                self.log_event(instance_id, "workspace_created")

                metadata: dict[str, Any] = {
                    "evaluation_name": self.evaluation_name,
                    "instance_id": instance["instance_id"],
                }

                repository = create_repository(
                    instance, repo_base_dir=self.repo_base_dir
                )
                code_index = create_index(instance, repository=repository)

                if self.use_testbed:
                    from moatless.runtime.testbed import TestbedEnvironment

                    run_id = hashlib.sha256(self.evaluation_name.encode()).hexdigest()[:8]

                    runtime = TestbedEnvironment(
                        repository=repository,
                        instance=instance,
                        log_dir=log_dir,
                        dataset_name=self.dataset_name,
                        run_id=run_id,
                    )
                else:
                    runtime = None

                if os.path.exists(trajectory_path):
                    search_tree = SearchTree.from_file(
                        trajectory_path,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                else:
                    completion_model = self._create_completion_model(
                        self.settings.agent_model
                    )

                    agent = CodingAgent.create(
                        completion_model=completion_model,
                        repository=repository,
                        code_index=code_index,
                        runtime=runtime,
                        message_history_type=self.settings.agent_message_history_type,
                    )

                    agent_role = f"""You are an autonomous AI assistant and a core member of the development team for the {instance["repo"]} project. As a senior developer on the team, you have deep knowledge of the codebase and best practices."""
                    agent.system_prompt = f"{agent_role}\n\n{agent.system_prompt}"

                    if self.selector:
                        selector = self.selector
                    elif self.settings.best_first:
                        selector = BestFirstSelector()
                    else:
                        selector = SoftmaxSelector()

                    if self.settings.max_expansions > 1:
                        value_function = CodingValueFunction(
                            completion_model=self._create_completion_model(
                                self.settings.value_function_model
                            )
                        )

                        # discriminator = MeanAwardDiscriminator()
                        discriminator = AgentDiscriminator(
                            completion=self._create_completion_model(),
                            n_agents=self.settings.debate_n_agents,
                            n_rounds=self.settings.debate_n_rounds,
                        )

                        if self.settings.provide_feedback:
                            if self.settings.feedback_type == "agent":
                                feedback = FeedbackAgent(
                                    completion_model=self._create_completion_model()
                                )
                            elif self.settings.feedback_type == "reward":
                                feedback = RewardFeedbackGenerator()
                            else:
                                feedback = None
                        else:
                            feedback = None
                    else:
                        value_function = None
                        discriminator = None
                        feedback = None

                    search_tree = SearchTree.create(
                        message=problem_statement,
                        repository=repository,
                        runtime=runtime,
                        agent=agent,
                        selector=selector,
                        value_function=value_function,
                        discriminator=discriminator,
                        feedback_generator=feedback,
                        max_expansions=self.settings.max_expansions,
                        max_iterations=self.settings.max_iterations,
                        max_depth=self.settings.max_depth,
                        min_finished_nodes=self.settings.min_finished_nodes,
                        max_finished_nodes=self.settings.max_finished_nodes,
                        max_cost=self.settings.max_cost,
                        reward_threshold=self.settings.reward_threshold,
                        metadata=metadata,
                        persist_path=trajectory_path,
                    )

            best_node = None
            start_time = time.time()
            try:
                self.log_event(instance_id, "search_tree_execution_started")

                if search_tree and "error" in eval_result:
                    del eval_result["error"]
                    with open(eval_result_path, "w") as f:
                        json.dump(eval_result, f, indent=2)

                search_tree.run_search()
                best_node = search_tree.get_best_trajectory()
                self.log_event(instance_id, "search_tree_execution_completed")
                if best_node:
                    self.save_prediction(
                        instance_id, best_node.file_context.generate_git_patch()
                    )
                eval_result["status"] = "completed"

                leaf_nodes = search_tree.get_leaf_nodes()
                patch_results = {}
                logger.info(
                    f"Will evaluate {len(leaf_nodes)} leaf nodes for instance {instance_id}"
                )

                if "node_results" not in eval_result:
                    eval_result["node_results"] = {}

                if self.use_testbed:
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
                    else:
                        logger.info(
                            f"Found {len(leaf_nodes) - len(unevaluated_nodes)} already evaluated nodes, "
                            f"will evaluate remaining {len(unevaluated_nodes)} nodes for instance {instance_id}"
                        )

                        if not runtime:
                            repository = create_repository(
                                instance, repo_base_dir=self.repo_base_dir
                            )
                            from testbeds.sdk import TestbedSDK
                            from moatless.runtime.testbed import TestbedEnvironment

                            runtime = TestbedEnvironment(
                                testbed_sdk=TestbedSDK(),
                                repository=repository,
                                instance=instance,
                                log_dir=log_dir,
                                enable_cache=True,
                            )

                        for i, leaf_node in enumerate(unevaluated_nodes):
                            logger.info(
                                f"Evaluate Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id}"
                            )

                            if str(leaf_node.node_id) in eval_result["node_results"]:
                                logger.info(
                                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} that has already been evaluated "
                                )
                                continue

                            patch = leaf_node.file_context.generate_git_patch()
                            if patch and patch.strip():
                                patch_hash = create_sha256_hash(patch)

                                if patch_hash in patch_results:
                                    logger.info(
                                        f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} as patch has already been evaluated."
                                    )
                                    eval_result["node_results"][leaf_node.node_id] = (
                                        patch_results[patch_hash]
                                    )
                                else:
                                    start_time = time.time()
                                    result = runtime.evaluate(patch=patch)
                                    if not result:
                                        logger.error(
                                            f"Error in evaluating patch for {instance_id}"
                                        )
                                        continue

                                    eval_result["node_results"][leaf_node.node_id] = (
                                        result.model_dump()
                                    )
                                    patch_results[patch_hash] = result.model_dump()
                                    logger.info(
                                        f"Evaluated patch in {time.time() - start_time} seconds (resolved: {result.resolved})"
                                    )
                            else:
                                logger.info(
                                    f"Skip Node{leaf_node.node_id} {i + 1}/{len(unevaluated_nodes)} for instance {instance_id} with no patch."
                                )

                            if best_node and leaf_node.node_id == best_node.node_id:
                                self.save_prediction(instance_id, patch)
                                eval_result["selected_node"] = leaf_node.node_id

                                if eval_result["node_results"].get(leaf_node.node_id):
                                    eval_result["resolved"] = eval_result[
                                        "node_results"
                                    ][leaf_node.node_id]["resolved"]

                                    if eval_result.get("resolved"):
                                        logger.info(
                                            f"Resolved {instance['instance_id']}"
                                        )
                                    else:
                                        logger.info(
                                            f"Could not resolve {instance['instance_id']}"
                                        )

                            with open(eval_result_path, "w") as f:
                                json.dump(eval_result, f, indent=2)

                if "error" in eval_result:
                    del eval_result["error"]

            except Exception:
                eval_result["error"] = traceback.format_exc()
                eval_result["status"] = "error"
                logging.exception(f"Error in evaluation of {instance['instance_id']} ")
            finally:
                eval_result["duration"] = time.time() - start_time
                search_tree.persist(trajectory_path)

            self.log_event(instance_id, "evaluation_completed")
            self.update_status(instance_id, eval_result["status"])

            return eval_result

        except Exception:
            logger.exception(f"Error in processing instance {instance_id}")
            self.log_event(instance_id, "evaluation_error")
            self.update_status(instance_id, "error")
            return None

        finally:
            with open(eval_result_path, "w") as f:
                json.dump(eval_result, f, indent=2)

            # Clean up
            if repository:
                shutil.rmtree(repository.repo_dir, ignore_errors=True)

            del runtime
            del repository
            del search_tree
            gc.collect()

    def save_prediction(self, instance_id, submission):
        with self.file_lock:
            prediction = {
                "model_name_or_path": self.evaluation_name,
                "instance_id": instance_id,
                "model_patch": submission,
            }
            with open(self.predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

    def _create_completion_model(
        self, model_settings: CompletionModel | None = None
    ) -> CompletionModel:
        return model_settings or self.settings.model

    def _to_csv_report(self, results: list[BenchmarkResult]):
        df = to_dataframe(results, self.report_mode)
        df.to_csv(
            f"{self.evaluation_dir}/result.csv",
            index=False,
            sep=",",
            decimal=",",
            quoting=1,
        )

    def _run_evaluation(self, instances: list[dict]):
        error = 0

        with open(self.predictions_path, "w") as file:
            file.write("")

        results = []

        logger.info(
            f"Processing {len(instances)} instances with {self.num_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self.evaluate_instance, instance)
                for instance in instances
            ]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    result = future.result()
                    # TODO
                    # if result:
                    #    results.append(result)
                    #    # self._to_csv_report(results)
                    #    self._save_json_report(results)
                    # else:
                    #    error += 1

                    # stats = self._create_stats(results)
                    # pbar.set_postfix(stats)

                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")

        logger.info(f"Completed processing with {error} errors")
        self.update_status("all", "evaluation_completed")

    def _create_stats(self, results):
        stats = {}
        if results:
            stats["avg_duration"] = sum(r.duration for r in results) / len(results)
            stats["avg_cost"] = sum(r.total_cost for r in results) / len(results)
            stats["total_cost"] = sum(r.total_cost for r in results)

            identified = sum(
                1
                for r in results
                if r.status in ["identified", "planned", "edited", "resolved"]
            )
            resolved = sum(1 for r in results if r.status in ["resolved"])
            error = sum(1 for r in results if r.status == "error")

            if identified > 0:
                stats["identified"] = f"{(identified / len(results)) * 100:.2f}%"
            if resolved > 0:
                stats["resolved"] = f"{(resolved / len(results)) * 100:.2f}%"
            stats["error"] = error

        return stats

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
