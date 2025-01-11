import glob
import logging
import logging
import os
from typing import Tuple, List, Dict
import json
import shutil
from collections import Counter
from datetime import datetime

from moatless.benchmark.report import BenchmarkResult, to_result, to_dataframe, to_trajectory_dataframe, read_reports
from moatless.benchmark.utils import get_moatless_instances
from moatless.search_tree import SearchTree
from moatless.benchmark.report_utils import create_evaluation_response, create_instance_dto, create_instance_response, load_resolution_rates
from moatless.benchmark.repository import EvaluationFileRepository
from moatless_tools.schema import EvaluationSettingsDTO, EvaluationResponseDTO, InstanceItemDTO

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.WARNING,
)

logger = logging.getLogger(__name__)

instance_splits: Dict[str, List[str]] = {}

def load_dataset_splits():
    """Load all dataset splits and map instance IDs to their splits."""
    global instance_splits
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    
    for dataset_file in glob.glob(os.path.join(datasets_dir, "*_dataset.json")):
        try:
            with open(dataset_file) as f:
                dataset = json.load(f)
                split_name = dataset["name"]
                
                # Add this split to each instance's list of splits
                for instance_id in dataset["instance_ids"]:
                    if instance_id not in instance_splits:
                        instance_splits[instance_id] = []
                    instance_splits[instance_id].append(split_name)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_file}: {e}")



def get_trajectories(dir: str) -> list[Tuple[SearchTree, str]]:
    trajectories = []
    for root, _, files in os.walk(dir):
        trajectory_path = os.path.join(root, "trajectory.json")
        if not os.path.exists(trajectory_path):
            logger.warning(f"Trajectory file not found: {trajectory_path}")
            continue

        try:
            rel_path = os.path.relpath(root, dir)
            # Check if file is empty
            if os.stat(trajectory_path).st_size == 0:
                logger.warning(f"Empty trajectory file: {trajectory_path}")
                continue

            trajectory = SearchTree.from_file(trajectory_path)
            trajectories.append((trajectory, rel_path))
        except Exception as e:
            logger.exception(f"Failed to load trajectory from {trajectory_path}: {e}")
    return trajectories


def summarize_with_rates(counter: Dict[str, int], total_count: int) -> List[Tuple[str, int, float]]:
    """Convert raw counts to (item, count, rate) tuples, sorted by rate."""
    return [(item, count, (count / total_count) * 100) 
            for item, count in sorted(counter.items(), key=lambda x: (x[1] / total_count), reverse=True)]


def summarize_flags_by_resolution(results: List[BenchmarkResult]) -> Tuple[List[Tuple[str, int, float]], List[Tuple[str, int, float]]]:
    resolved_counter = Counter()
    unresolved_counter = Counter()
    resolved_count = 0
    unresolved_count = 0
    
    for result in results:
        if result.resolved:
            resolved_count += 1
            for flag in result.flags:
                resolved_counter[flag] += 1
        else:
            unresolved_count += 1
            for flag in result.flags:
                unresolved_counter[flag] += 1
    
    resolved_stats = summarize_with_rates(resolved_counter, resolved_count) if resolved_count > 0 else []
    unresolved_stats = summarize_with_rates(unresolved_counter, unresolved_count) if unresolved_count > 0 else []
    
    return resolved_stats, unresolved_stats


def get_evaluations_dir():
    """Get the current evaluations directory from environment."""
    return os.getenv("MOATLESS_DIR", "./evals")

def get_repository() -> EvaluationFileRepository:
    """Get repository instance with current evaluations directory."""
    return EvaluationFileRepository(get_evaluations_dir())

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def generate_report(dir: str):
    result_path = os.path.join(dir, "result.json")
    predictions_path = os.path.join(dir, "all_preds2.jsonl")
    with open(predictions_path, "w") as file:
        file.write("")

    load_dataset_splits()

    external_result = None
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            external_result = json.load(f)
    
    if not external_result:
        raise ValueError("External result not found")

    trajectories = get_trajectories(dir)
    print(f"Trajectories: {len(trajectories)}")
    if not trajectories:
        raise ValueError("No trajectories found")
    instances = get_moatless_instances()

    # Load resolution rates once
    resolution_rates = load_resolution_rates()
    
    duplicted_sarch = 0
    results = []
    instance_items = []  # Keep track of all instance items
    
    for trajectory, rel_path in trajectories:
        instance_id = trajectory.metadata["instance_id"]

        instance = instances.get(instance_id)
        if not instance:
            logger.error(f"Instance {instance_id} not found")
            continue

        splits = instance_splits.get(instance_id, [])
        eval_report = None
        eval_result_file = os.path.join(dir, instance_id, "eval_result.json")
        try:
            if os.path.exists(eval_result_file):
                with open(eval_result_file, "r") as f:
                    eval_report = json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load eval report from {eval_result_file}. : {e}")
        
        # Process trajectory and create result
        result = to_result(trajectory, eval_report, external_result)
        results.append(result)

        # Create instance DTO for list view
        instance_item = create_instance_dto(
            result=result,
            resolution_rates=resolution_rates,
            splits=splits,
        )
        instance_items.append(instance_item)

        # Create and save full instance response DTO
        instance_response = create_instance_response(
            search_tree=trajectory,
            instance=instance,
            eval_result=eval_report,
            resolution_rates=resolution_rates,
            splits=splits,
            result=result  # Pass the result to get metrics
        )
        instance_response_path = os.path.join(dir, instance_id, "instance_response.json")
        print(f"Save {instance_response_path}")
        with open(instance_response_path, "w") as f:
            json.dump(instance_response.model_dump(), f, indent=2, cls=DateTimeEncoder)


        # Handle predictions if best node exists
        best_node = trajectory.get_best_trajectory()
        if best_node:
            prediction = {
                "model_name_or_path": trajectory.metadata.get("evaluation_name", dir.split("/")[-1]),
                "instance_id": instance_id,
                "model_patch": best_node.file_context.generate_git_patch(),
            }
            with open(predictions_path, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

    # Create and save partial evaluation response after each instance
    evaluation_name = dir.split("/")[-1]
    first_tree = trajectories[0][0] if trajectories else None
    partial_evaluation_response = create_evaluation_response(
        evaluation_name, 
        instance_items,
        first_tree=first_tree
    )
    
    # Save current state of evaluation_response.json
    with open(os.path.join(dir, "evaluation_response.json"), "w") as f:
        json.dump(partial_evaluation_response.model_dump(), f, indent=2, cls=DateTimeEncoder)

    # Save the results as report.json
    report_path = os.path.join(dir, "report.json")
    with open(report_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)

    # Use the saved results directly
    df = to_dataframe(results)

    # Add a new column for success status
    df['success_status'] = df.apply(lambda row:
        'Resolved' if row['status'] == 'resolved' else
        'Running w/ Resolved' if row['status'] == 'running' and row['resolved_solutions'] > 0 else
        'Running w/ Solution' if row['status'] == 'running' and row['solutions'] > 0 else
        'Partially Resolved' if row['resolved_solutions'] > 0 else
        'Running' if row['status'] == 'running' else
        'Rejected' if row['status'] == 'rejected' else
        'Error' if row['status'] == 'error' else
        'Failed', axis=1)

    # Format llmonkeys_rate as a percentage
    df['llmonkeys_rate'] = df['llmonkeys_rate'].apply(lambda x: f"{x * 100:.1f}%")

    # Calculate summary statistics
    total_trajectories = len(df)
    status_counts = df['success_status'].value_counts()
    total_cost = df['total_cost'].sum()
    total_prompt_tokens = df['prompt_tokens'].sum()
    total_completion_tokens = df['completion_tokens'].sum()
    total_transitions = df['all_transitions'].sum()
    avg_cost = df['total_cost'].mean()
    avg_prompt_tokens = df['prompt_tokens'].mean()
    avg_completion_tokens = df['completion_tokens'].mean()
    avg_transitions = df['all_transitions'].mean()
    filtered_df = df[~df['status'].isin(['running', 'error', 'rejected'])]
    avg_duration = filtered_df['duration'].mean()

    # Print summary statistics
    print("\nSummary:")
    print(f"Total Trajectories: {total_trajectories}")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Transitions: {total_transitions}")
    print(f"Average Cost per Trajectory: ${avg_cost:.2f}")
    print(f"Average Prompt Tokens per Trajectory: {avg_prompt_tokens:.2f}")
    print(f"Average Completion Tokens per Trajectory: {avg_completion_tokens:.2f}")
    print(f"Average Transitions per Trajectory: {avg_transitions:.2f}")
    print(f"Average Duration: {avg_duration:.2f} s")
    
    # Print status distribution
    print("\nStatus Distribution:")
    for status, count in status_counts.items():
        print(f"{status}: {count}")
        
    # Print resolved status distribution
    print("\nResolved Status Distribution:")
    resolved_counts = df['resolved'].value_counts()
    for resolved_status, count in resolved_counts.items():
        status_text = "Resolved" if resolved_status == True else "Not Resolved" if resolved_status == False else "Unknown"
        print(f"{status_text}: {count}")

    # Print combined status and resolved distribution
    print("\nDetailed Status Distribution:")
    combined_counts = df.groupby(['status', 'resolved']).size().unstack(fill_value=0)
    print(combined_counts)

    # Summarize and print flags with rates
    resolved_flag_stats, unresolved_flag_stats = summarize_flags_by_resolution(results)
    
    print("\nFlags Summary for Resolved Trajectories:")
    for flag, count, rate in resolved_flag_stats:
        print(f"{flag}: {count} ({rate:.1f}%)")
    
    print("\nFlags Summary for Unresolved Trajectories:")
    for flag, count, rate in unresolved_flag_stats:
        print(f"{flag}: {count} ({rate:.1f}%)")

    # Calculate and print rate differences for flags
    print("\nFlags Rate Differences (Unresolved - Resolved):")
    resolved_rates = {flag: rate for flag, _, rate in resolved_flag_stats}
    unresolved_rates = {flag: rate for flag, _, rate in unresolved_flag_stats}
    all_flags = set(resolved_rates.keys()) | set(unresolved_rates.keys())
    
    rate_diffs = [(flag, 
                   unresolved_rates.get(flag, 0) - resolved_rates.get(flag, 0))
                  for flag in all_flags]
    for flag, diff in sorted(rate_diffs, key=lambda x: abs(x[1]), reverse=True):
        if diff != 0:
            print(f"{flag}: {diff:+.1f}%")

    df.to_csv(os.path.join(dir, "report.csv"), index=False)

    # to csv
    df = to_trajectory_dataframe(results)
    df.to_csv(os.path.join(dir, "trajectories.csv"), index=False)

    # to json
    with open(os.path.join(dir, "report.json"), "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)

if __name__ == "__main__":
    import sys
    directory = sys.argv[1]
    generate_report(directory)
