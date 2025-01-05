from typing import List, Dict, Optional
from datetime import datetime
from moatless.benchmark.evaluation_v2 import Evaluation, InstanceStatus
from moatless.benchmark.report import BenchmarkResult
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.file_context import FileContext
from moatless.search_tree import SearchTree
from moatless_tools.schema import (
    EvaluationResponseDTO, EvaluationSettingsDTO, InstanceItemDTO, UsageDTO,
    InstanceResponseDTO, NodeDTO, ActionDTO, ObservationDTO, CompletionDTO, ActionStepDTO, FileContextDTO, FileContextSpanDTO, FileContextFileDTO
)
from moatless.node import Node as MoatlessNode
from testbeds.schema import TestStatus


def load_resolution_rates() -> Dict[str, float]:
    """Load resolution rates from the resolved submissions data."""
    import os
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "resolved_submissions.json")
    try:
        with open(dataset_path) as f:
            resolved_data = json.load(f)
            return {
                instance_id: len(data["resolved_submissions"]) / data["no_of_submissions"] 
                if data["no_of_submissions"] > 0 else 0.0
                for instance_id, data in resolved_data.items()
            }
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load resolution rates data")
        return {}

def create_evaluation_response(evaluation: Evaluation, instance_items: List[InstanceItemDTO]) -> EvaluationResponseDTO:
    """Create EvaluationResponseDTO from evaluation and instance items."""
    # Calculate totals from instance items
    total_cost = sum(item.completionCost or 0 for item in instance_items)
    prompt_tokens = sum(item.promptTokens or 0 for item in instance_items)
    completion_tokens = sum(item.completionTokens or 0 for item in instance_items)
    
    return EvaluationResponseDTO(
        name=evaluation.evaluation_name,
        status=evaluation.status,
        isActive=False,
        settings=EvaluationSettingsDTO(
            model=evaluation.settings.model.model,
            temperature=evaluation.settings.model.temperature,
            maxIterations=evaluation.settings.max_iterations,
            responseFormat=evaluation.settings.agent_settings.completion_model.response_format,
            maxCost=evaluation.settings.max_cost,
        ),
        startedAt=evaluation.start_time if hasattr(evaluation, 'start_time') else None,
        totalCost=total_cost,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        totalInstances=len(instance_items),
        completedInstances=sum(1 for i in instance_items if i.status == "completed"),
        errorInstances=sum(1 for i in instance_items if i.status == "error"),
        resolvedInstances=sum(1 for i in instance_items if i.resolved is True),
        failedInstances=sum(1 for i in instance_items if i.status == "failed"),
        instances=instance_items
    )

def derive_instance_status(result: BenchmarkResult) -> str:
    """Derive instance status from a BenchmarkResult."""
    return (
        "resolved" if result.resolved is True else
        "failed" if result.resolved is False else 
        "error" if result.status == "error" else
        "completed" if result.status == "completed" else
        "running" if result.status == "running" else
        "pending"
    )

def create_instance_dto(result: BenchmarkResult, resolution_rates: Dict[str, float], splits: List[str] = None) -> InstanceItemDTO:
    """Create InstanceItemDTO from a BenchmarkResult."""
    status = derive_instance_status(result)

    return InstanceItemDTO(
        instanceId=result.instance_id,
        status=status,
        duration=result.duration,
        resolved=result.resolved,
        error=result.error if result.error else None,
        iterations=result.all_transitions,
        completionCost=result.total_cost,
        promptTokens=result.prompt_tokens,
        completionTokens=result.completion_tokens,
        resolutionRate=resolution_rates.get(result.instance_id, None),
        splits=splits or [],
        flags=result.flags
    ) 

def convert_moatless_node_to_api_node(node: MoatlessNode) -> NodeDTO:
    """Convert a Moatless Node to an API Node model."""
    # Collect warnings and errors
    warnings = []
    errors = []
    
    # Check observation properties for errors and warnings
    if node.observation:
        # Add fail_reason as an error if it exists
        if node.observation.properties.get("fail_reason"):
            errors.append(node.observation.properties["fail_reason"])
            
        # Add flags as warnings if they exist
        if node.observation.properties.get("flags"):
            warnings.extend(node.observation.properties["flags"])
            
        # Count test failures/errors
        failed_tests = 0
        error_tests = 0
        
        # Count from observation test results
        if node.observation.properties.get("test_results"):
            for test in node.observation.properties["test_results"]:
                if test["status"] == "ERROR":
                    error_tests += 1
                elif test["status"] == "FAILED":
                    failed_tests += 1

        # Count from file context test results
        if node.file_context and node.file_context.test_files:
            for test_file in node.file_context.test_files:
                for result in test_file.test_results:
                    if result.status == TestStatus.ERROR:
                        error_tests += 1
                    elif result.status == TestStatus.FAILED:
                        failed_tests += 1

        # Add single warning if there are any test issues
        if failed_tests > 0 or error_tests > 0:
            warning_parts = []
            if failed_tests > 0:
                warning_parts.append(f"{failed_tests} failed")
            if error_tests > 0:
                warning_parts.append(f"{error_tests} errors")
            warnings.append(f"Tests: {' and '.join(warning_parts)}")

    # Check file context for test failures/errors if available
    if node.file_context and node.file_context.test_files:
        for test_file in node.file_context.test_files:
            for result in test_file.test_results:
                if result.status in [TestStatus.ERROR, TestStatus.FAILED]:
                    warnings.append(f"Test {result.status.lower()} in {test_file.file_path}: {result.message}")

    # Convert action steps
    action_steps = []
    for step in node.action_steps:
        # Convert action
        action = ActionDTO(
            name=step.action.name,
            thoughts=getattr(step.action, "thoughts", None),
            properties=step.action.model_dump(exclude={"thoughts", "name"})
        )

        # Convert observation
        observation = None
        if step.observation:
            observation = ObservationDTO(
                message=step.observation.message,
                summary=step.observation.summary,
                properties=step.observation.properties if hasattr(step.observation, "properties") else {},
                expectCorrection=step.observation.expect_correction if hasattr(step.observation, "expect_correction") else False
            )

        # Convert completion
        completion = None
        if step.completion and step.completion.usage:
            usage = UsageDTO(
                completionCost=step.completion.usage.completion_cost,
                promptTokens=step.completion.usage.prompt_tokens,
                completionTokens=step.completion.usage.completion_tokens,
                cachedTokens=step.completion.usage.cached_tokens
            )
            tokens = []
            if step.completion.usage.prompt_tokens:
                tokens.append(f"{step.completion.usage.prompt_tokens}↑")
            if step.completion.usage.completion_tokens:
                tokens.append(f"{step.completion.usage.completion_tokens}↓")
            if step.completion.usage.cached_tokens:
                tokens.append(f"{step.completion.usage.cached_tokens}⚡")
            
            completion = CompletionDTO(
                type="action_step",
                usage=usage,
                tokens=" ".join(tokens)
            )

        action_steps.append(ActionStepDTO(
            thoughts=getattr(step.action, "thoughts", None),
            action=action,
            observation=observation,
            completion=completion
        ))

    # Convert completions
    completions = {}
    for completion_type, completion in node.completions.items():
        if completion and completion.usage:
            usage = UsageDTO(
                completionCost=completion.usage.completion_cost,
                promptTokens=completion.usage.prompt_tokens,
                completionTokens=completion.usage.completion_tokens,
                cachedTokens=completion.usage.cached_tokens
            )
            tokens = []
            if completion.usage.prompt_tokens:
                tokens.append(f"{completion.usage.prompt_tokens}↑")
            if completion.usage.completion_tokens:
                tokens.append(f"{completion.usage.completion_tokens}↓")
            if completion.usage.cached_tokens:
                tokens.append(f"{completion.usage.cached_tokens}⚡")
            
            completions[completion_type] = CompletionDTO(
                type=completion_type,
                usage=usage,
                tokens=" ".join(tokens)
            )

    # Convert file context if exists
    file_context = None
    if node.file_context:
        file_context = file_context_to_dto(node.file_context)

    # Get completion usage if exists
    completion_usage = None
    if hasattr(node, "completion") and node.completion and node.completion.usage:
        completion_usage = UsageDTO(
            completionCost=node.completion.usage.completion_cost,
            promptTokens=node.completion.usage.prompt_tokens,
            completionTokens=node.completion.usage.completion_tokens,
            cachedTokens=node.completion.usage.cached_tokens
        )

    return NodeDTO(
        nodeId=node.node_id,
        actionSteps=action_steps,
        assistantMessage=node.assistant_message,
        userMessage=node.user_message,
        completionUsage=completion_usage,
        completions=completions,
        fileContext=file_context,
        warnings=warnings,
        errors=errors
    )

def create_instance_response(
    search_tree: SearchTree,
    instance: dict,
    eval_result: Optional[dict] = None,
    resolution_rates: Dict[str, float] = None,
    splits: List[str] = None,
    result: Optional[BenchmarkResult] = None
) -> InstanceResponseDTO:
    """Create InstanceResponseDTO from a SearchTree and instance data."""
    nodes = []
    for moatless_node in search_tree.root.get_all_nodes():
        nodes.append(convert_moatless_node_to_api_node(moatless_node))

    instance_id = search_tree.metadata.get("instance_id")
    
    status = derive_instance_status(result) if result else "pending"
    
    return InstanceResponseDTO(
        nodes=nodes,
        totalNodes=len(nodes),
        instance=instance,
        evalResult=eval_result,
        status=status,
        duration=result.duration if result else None,
        resolved=result.resolved if result else None,
        error=result.error if result else None,
        iterations=result.all_transitions if result else None,
        completionCost=result.total_cost if result else None,
        promptTokens=result.prompt_tokens if result else None,
        completionTokens=result.completion_tokens if result else None,
        resolutionRate=resolution_rates.get(instance_id) if resolution_rates else None,
        splits=splits or [],
        flags=result.flags if result else None
    )
    

def file_context_to_dto(file_context: FileContext) -> FileContextDTO:
    if not file_context:
        return None
    """Convert FileContext to FileContextDTO."""
    files = []
    for context_file in file_context.files:
        files.append(FileContextFileDTO(
            file_path=context_file.file_path,
            patch=context_file.patch,
            spans=[FileContextSpanDTO(**span.model_dump()) for span in context_file.spans],
            show_all_spans=context_file.show_all_spans,
            tokens=context_file.context_size(),
            is_new=context_file._is_new,
            was_edited=context_file.was_edited
        ))

    return FileContextDTO(
        summary=file_context.create_summary(),
        testSummary=file_context.get_test_summary() if file_context.test_files else None,
        testResults=[result.model_dump() for test_file in file_context.test_files 
                    for result in test_file.test_results] if file_context.test_files else None,
        patch=file_context.generate_git_patch(),
        files=files
    )
    
