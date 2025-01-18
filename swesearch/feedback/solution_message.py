from typing import List, Optional, Sequence

from moatless.completion.schema import ChatCompletionUserMessage
from moatless.node import Node
import logging

logger = logging.getLogger(__name__)

def _format_history(node: Node) -> str:
    """Format the action history for a node."""
    if not node.action:
        return ""
        
    previous_nodes = node.get_trajectory()[:-1]
    formatted_history: List[str] = []
    counter = 0
    for previous_node in previous_nodes:
        if previous_node.action:
            counter += 1
            formatted_state = f"\n## {counter}. Action: {previous_node.action.name}\n"
            formatted_state += previous_node.action.to_prompt()

            if previous_node.observation:
                formatted_state += f"\n\nObservation: {previous_node.observation.summary}"
                formatted_history.append(formatted_state)
            else:
                logger.warning(f"No output found for Node{previous_node.node_id}")

    if formatted_history:
        return "\nHistory of actions and their outputs:\n<history>\n" + "\n".join(formatted_history) + "\n</history>\n"
    return ""

def _format_completion_reason(node: Node) -> str:
    """Format the completion reasoning for a node."""
    message = "\n<reasoning_for_completion>\n"
    if not node.action or node.action.name != "Finish":
        message += "The coding agent did not finish the task."
    else:
        message += node.action.finish_reason
    message += "\n</reasoning_for_completion>\n"
    return message

def _format_changes(node: Node) -> str:
    """Format the changes made by a node."""
    full_patch = node.file_context.generate_git_patch()
    return f"\nChanges made to the codebase:\n<git_patch>\n{full_patch}\n</git_patch>\n"


def _format_test_output(node: Node) -> str:
    """Format test output for a node."""
    if not node.file_context.has_runtime or not node.file_context.test_files:
        return ""
        
    test_output = "\n<test_output>\nThe following tests were run:\n"
    for test_file in node.file_context.test_files:
        test_output += f"* {test_file.file_path}\n"
        
    failure_details = node.file_context.get_test_failure_details(max_tokens=4000)
    if failure_details:
        test_output += failure_details + "\n\n"

    test_output += node.file_context.get_test_summary()
    test_output += "\n</test_output>\n"
    return test_output

def _format_reward(node: Node) -> str:
    """Format reward information for a node."""
    if not node.reward:
        return ""
        
    message = f"\n### Reward: {node.reward.value}/100\n"
    if node.reward.explanation:
        message += f"Reward Explanation: {node.reward.explanation}\n"
    return message


def format_task_and_context(root_node: Node, show_file_context: bool = True) -> str:
    """Format the original task and file context from all solutions."""
    message = "The task to solve:\n"
    message += root_node.message + "\n\n"
    
    if show_file_context:
        message += "The code context for all solutions:\n"
        original_context = root_node.file_context.clone()
        
        # Collect all file context from all solutions
        for node in root_node.get_leaf_nodes():
            for file in node.file_context.files:
                original_context.add_spans_to_context(file.file_path, file.span_ids)
                
        message += original_context.create_prompt(
            show_outcommented_code=True,
            exclude_comments=False,
            outcomment_code_comment="... code not in context",
        )
        
    return message

def _format_solution_context(node: Node) -> str:
    """Format the updated file context for a single solution."""
    message = "\nUpdated file context:\n<updated_file_context>\n"
    for file in node.file_context.files:
        if file.patch:
            message += file.to_prompt(
                show_outcommented_code=True,
                exclude_comments=True,
                outcomment_code_comment="... code not in context"
            )
    message += "</updated_file_context>\n"
    return message

def format_single_solution(
    node: Node,
    show_history: bool = True,
    show_file_context: bool = True,
) -> str:
    """Format a single solution's details.
    
    Args:
        node: The solution node to format
        solution_number: Optional number for this solution (used in multi-solution context)
        show_history: Whether to show action history
        show_file_context: Whether to show file context
        is_current: Whether this is the current solution being evaluated
    """
    if not node.file_context.has_patch(ignore_tests=True):
        logger.warning(f"No changes made to the codebase for Node{node.node_id}")
        return ""
        
    message = "\nStatus: " + ("Finished" if node.is_finished() else "Abandoned")

    if show_history:
        message += _format_history(node)
        
    message += _format_completion_reason(node)
    message += _format_changes(node)
    message += "\n" + "-" * 80 + "\n"
    
    if show_file_context:
        message += _format_solution_context(node)
        
    message += _format_test_output(node)
    message += _format_reward(node)

    return message

def create_solution_message(
    nodes: Sequence[Node],
    show_history: bool = True,
    show_file_context: bool = True,
    exclude_node_id: Optional[int] = None,
) -> str:
    """Create a message describing one or more solution attempts."""
    if not nodes:
        return ""
        
    # Start with task and complete context
    message = format_task_and_context(nodes[0].get_root(), show_file_context)
    message += "\n\nSolutions:\n"
    
    solution_number = 1
    for node in nodes:
        if exclude_node_id and node.node_id == exclude_node_id:
            continue
        
        solution_text = f"\n# Solution {solution_number} (Node{node.node_id})\n"
        solution_text = format_single_solution(
            node=node,
            show_history=show_history,
            show_file_context=show_file_context
        )
        if solution_text:  # Only increment if solution was actually formatted
            message += solution_text
            solution_number += 1

    return message
