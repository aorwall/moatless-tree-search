import concurrent.futures
import gc
import json
import logging
import os
import shutil
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Any, List

import litellm
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from moatless.agent.code_agent import CodingAgent
from moatless.agent.code_prompts import SIMPLE_CODE_PROMPT, SYSTEM_PROMPT
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
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.discriminator import MeanAwardDiscriminator, AgentDiscriminator
from moatless.feedback import FeedbackGenerator
from moatless.file_context import FileContext
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector, SoftmaxSelector
from moatless.templates import create_coding_actions
from moatless.value_function import ValueFunction

logger = logging.getLogger(__name__)


class ModelSettings(BaseModel):
    model: str = Field(
        ...,
        description="The model to use for completions.",
    )
    temperature: float = Field(
        0.0,
        description="The temperature to use for completions.",
    )
    base_url: Optional[str] = Field(
        None,
        description="The base URL for the model API.",
    )
    api_key: Optional[str] = Field(
        None,
        description="The API key for the model API.",
    )
    response_format: Optional[LLMResponseFormat] = Field(
        LLMResponseFormat.TOOLS,
        description="The response format for the model API.",
    )


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

    model: Optional[ModelSettings] = Field(
        default=None,
        description="The default model.",
    )

    agent_model: Optional[ModelSettings] = Field(
        default=None,
        description="The model the agent will use.",
    )

    value_function_model: Optional[ModelSettings] = Field(
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
        100,
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
    ):
        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.report_mode = report_mode
        self.dataset_name = dataset_name
        self.evaluation_name = evaluation_name

        self.use_testbed = use_testbed

        self.settings = settings
        self.max_file_context_tokens = max_file_context_tokens

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
        resolved_by: Optional[int] = None,
        instance_ids: list[str] | None = None,
        ignore_repos: list[str] | None = None,
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json"
        )
        with open(file_path) as f:
            instances = json.load(f)

        instances = sorted(instances, key=lambda x: len(x["resolved_by"]), reverse=True)
        logger.info(f"Loaded {len(instances)} instances from {file_path}")

        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by instance_ids"
            )

        if resolved_by:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= resolved_by
                or (
                    resolved_by == 1
                    and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0
                )
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by resolved_by >= {resolved_by}"
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
            with open(eval_result_path) as f:
                eval_result = json.load(f)
        else:
            eval_result = {
                "node_results": {},
            }

        logger.info(f"Evaluating {instance_id}")
        problem_statement = instance["problem_statement"]

        runtime = None
        repository = None

        self.update_status(instance_id, "started")
        self.log_event(instance_id, "evaluate_instance_initiated")

        try:
            search_tree = None

            if os.path.exists(trajectory_path):
                persisted_tree = SearchTree.from_file(trajectory_path)
                if persisted_tree.is_finished():
                    logger.info(f"Found completed search tree for {instance_id}")
                    search_tree = persisted_tree

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
                    from testbeds.sdk import TestbedSDK
                    from moatless.runtime.testbed import TestbedEnvironment

                    runtime = TestbedEnvironment(
                        testbed_sdk=TestbedSDK(),
                        repository=repository,
                        instance=instance,
                        log_dir=log_dir,
                    )
                    system_prompt = SYSTEM_PROMPT # Use default system prompt
                else:
                    runtime = None
                    system_prompt = SIMPLE_CODE_PROMPT # Use simple code prompt

                if os.path.exists(trajectory_path):
                    search_tree = SearchTree.from_file(
                        trajectory_path,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                else:
                    actions = create_coding_actions(
                        repository=repository,
                        code_index=code_index,
                        runtime=runtime,
                        edit_completion_model=self._create_completion_model(),
                    )
                    agent = CodingAgent(
                        completion=self._create_completion_model(
                            self.settings.agent_model
                        ),
                        actions=actions,
                    )

                    if self.settings.best_first:
                        selector = BestFirstSelector()
                    else:
                        selector = SoftmaxSelector()

                    value_function = ValueFunction(
                        completion=self._create_completion_model(
                            self.settings.value_function_model
                        )
                    )

                    # discriminator = MeanAwardDiscriminator()
                    discriminator = AgentDiscriminator(
                        completion_model=self._create_completion_model(),
                        n_agents=self.settings.debate_n_agents,
                        n_rounds=self.settings.debate_n_rounds,
                    )

                    if self.settings.provide_feedback:
                        feedback = FeedbackGenerator()
                    else:
                        feedback = None

                    file_context = FileContext(repo=repository)

                    search_tree = SearchTree.create(
                        message=problem_statement,
                        file_context=file_context,
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
                search_tree.run_search()
                best_node = search_tree.get_best_trajectory()
                self.log_event(instance_id, "search_tree_execution_completed")
                eval_result["status"] = "completed"
            except Exception:
                eval_result["error"] = traceback.format_exc()
                eval_result["status"] = "error"
                logging.exception(f"Error in evaluation of {instance['instance_id']} ")
            finally:
                eval_result["duration"] = time.time() - start_time
                search_tree.persist(trajectory_path)

            finished_nodes = search_tree.get_finished_nodes()
            patch_results = {}
            logger.info(
                f"Will evaluate {len(finished_nodes)} finished nodes for instance {instance_id}"
            )

            if "node_results" not in eval_result:
                eval_result["node_results"] = {}

            if self.use_testbed:
                if len(eval_result.get("node_results")) == len(
                    search_tree.get_finished_nodes()
                ):
                    logger.info(
                        f"Already evaluated results for {len(search_tree.get_finished_nodes())} nodes in {instance_id}"
                    )
                else:
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
                        )

                    for i, finished_node in enumerate(finished_nodes):
                        logger.info(
                            f"Evaluate finished Node{finished_node.node_id} {i+1}/{len(finished_nodes)} for instance {instance_id}"
                        )

                        if finished_node.node_id in eval_result["node_results"]:
                            continue

                        patch = finished_node.file_context.generate_git_patch()
                        patch_hash = create_sha256_hash(patch)

                        if patch:
                            if patch_hash in patch_results:
                                logger.info(
                                    f"Use already evaluated patch for Node{finished_node.node_id} in {instance_id}"
                                )
                                eval_result["node_results"][finished_node.node_id] = (
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

                                eval_result["node_results"][finished_node.node_id] = (
                                    result.model_dump()
                                )
                                patch_results[patch_hash] = result.model_dump()
                                logger.info(
                                    f"Evaluated patch in {time.time() - start_time} seconds (resolved: {result.resolved})"
                                )

                        if best_node and finished_node.node_id == best_node.node_id:
                            self.save_prediction(instance_id, patch)
                            eval_result["selected_node"] = finished_node.node_id

                            if eval_result["node_results"].get(finished_node.node_id):
                                eval_result["resolved"] = eval_result["node_results"][
                                    finished_node.node_id
                                ]["resolved"]

                                if eval_result.get("resolved"):
                                    logger.info(f"Resolved {instance['instance_id']}")
                                else:
                                    logger.info(
                                        f"Could not resolve {instance['instance_id']}"
                                    )

                        with open(eval_result_path, "w") as f:
                            json.dump(eval_result, f, indent=2)

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
        self, model_settings: ModelSettings | None = None
    ) -> CompletionModel:
        model_settings = model_settings or self.settings.model
        return CompletionModel(
            model=model_settings.model,
            temperature=model_settings.temperature,
            model_base_url=model_settings.base_url,
            model_api_key=model_settings.api_key,
            response_format=model_settings.response_format,
        )

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
    model_name = model.split("/")[-1]
    model_name = f"{date_str}_{model_name}"
    if max_expansions:
        model_name += f"_max_exp{max_expansions}"
    for key, value in kwargs.items():
        model_name += f"_{key}_{value}"
    return model_name
