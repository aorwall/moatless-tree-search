import logging
from typing import Optional, Tuple

from moatless.actions.search_base import SearchBaseArgs
from moatless.completion.model import Completion
from moatless.node import Node
from moatless.value_function.base import ValueFunction
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)


class CodingValueFunction(ValueFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        if node.observation.expect_correction and self.correction_award is not None:
            # Start with the base correction award
            correction_penalty = 0
            current_node = node.parent

            # Check parent nodes for expect_correction
            while current_node and current_node.observation and current_node.observation.expect_correction:
                if (
                    current_node.observation
                    and current_node.observation.expect_correction
                ):
                    correction_penalty += 25
                current_node = current_node.parent

            # Calculate final reward with penalty, minimum of -100
            final_reward = max(-100, self.correction_award - correction_penalty)
            logger.info(
                f"Expecting a correction, base reward {self.correction_award}, penalty {correction_penalty}, final reward {final_reward}"
            )
            return Reward(value=final_reward, explanation="Expects a correction"), None

        if node.action.name == "Reject":
            logger.info(f"Reject action, assigning reward -100")
            return Reward(value=-100, explanation="Reject action"), None

        if node.observation.properties:
            fail_reason = node.observation.properties.get("fail_reason")
            if fail_reason:
                logger.info(f"Action failed with reason: {fail_reason}")
                if fail_reason == "no_search_hits":
                    return Reward(
                        value=-100, explanation="Search returned no results"
                    ), None
                elif fail_reason == "file_not_found":
                    return Reward(
                        value=-25,
                        explanation="Requested file does not exist in repository",
                    ), None
                elif fail_reason == "is_directory":
                    return Reward(
                        value=-50,
                        explanation="Requested path is a directory, not a file",
                    ), None
                elif fail_reason == "invalid_file":
                    return Reward(
                        value=-50,
                        explanation="File exists but could not be parsed or is empty",
                    ), None
                elif fail_reason == "too_many_tokens":
                    return Reward(
                        value=-25,
                        explanation="Requested context exceeds token limit, needs to be more specific",
                    ), None
                elif fail_reason == "no_spans_found":
                    return Reward(
                        value=-50,
                        explanation="No code spans found matching the request",
                    ), None
                elif fail_reason == "no_spans_added":
                    return Reward(
                        value=-25, explanation="Requested spans were already in context"
                    ), None

                # File creation failures
                elif fail_reason == "file_exists":
                    return Reward(
                        value=-25, explanation="Cannot create file - it already exists"
                    ), None
                elif fail_reason == "file_not_found":
                    return Reward(
                        value=-100, explanation="Cannot modify file - it doesn't exist"
                    ), None

                # String replacement failures
                elif fail_reason == "string_not_found":
                    return Reward(
                        value=-50, explanation="String to replace was not found in file"
                    ), None
                elif fail_reason == "multiple_occurrences":
                    return Reward(
                        value=-25,
                        explanation="Multiple occurrences of string found - need more context",
                    ), None
                elif fail_reason == "string_already_exists":
                    return Reward(
                        value=-50, explanation="New string already exists in file"
                    ), None
                elif fail_reason == "no_changes":
                    return Reward(
                        value=-50,
                        explanation="Old and new strings are identical - no changes needed",
                    ), None
                elif fail_reason == "indentation_differs":
                    return Reward(
                        value=-25,
                        explanation="Content matches but indentation is incorrect",
                    ), None
                elif fail_reason == "line_breaks_differs":
                    return Reward(
                        value=-25,
                        explanation="Content matches but line breaks are incorrect",
                    ), None
                elif fail_reason == "multiple_format_matches":
                    return Reward(
                        value=-25,
                        explanation="Multiple potential matches with different formatting found",
                    ), None

                # Generic failure
                else:
                    return Reward(value=-100, explanation="Action failed"), None

            if isinstance(node.action, SearchBaseArgs):
                if not node.observation.properties.get("new_span_ids"):
                    return Reward(
                        value=-50,
                        explanation="Search returned results but did not add any new spans to the context",
                    ), None

                return Reward(
                    value=100,
                    explanation="Search returned results and added new spans to the context",
                ), None

            test_results = node.observation.properties.get("test_results", [])
            if test_results:
                failure_count = sum(
                    1 for result in test_results if result["status"] == "FAILED"
                )
                error_count = sum(
                    1 for result in test_results if result["status"] == "ERROR"
                )
                total_tests = len(test_results)
                passed_count = total_tests - failure_count - error_count

                # Get previous test results
                previous_failure_count = None
                previous_error_count = None
                previous_reward = None
                parent_node = node.parent
                if (
                    parent_node
                    and parent_node.observation
                    and parent_node.observation.properties
                    and parent_node.reward
                    and parent_node.observation.properties.get("test_results") is not None
                ):
                    prev_test_results = parent_node.observation.properties.get(
                        "test_results", []
                    )
                    previous_failure_count = sum(
                        1
                        for result in prev_test_results
                        if result["status"] == "FAILED"
                    )
                    previous_error_count = sum(
                        1 for result in prev_test_results if result["status"] == "ERROR"
                    )
                    previous_reward = parent_node.reward.value

                if previous_reward is None:
                    # No previous test runs found
                    if failure_count == 0 and error_count == 0:
                        return Reward(
                            value=100, explanation=f"All {passed_count} tests passing"
                        ), None
                    return Reward(
                        value=50,
                        explanation=f"First test run with {failure_count} failures and {error_count} errors",
                    ), None

                # Compare with previous failures and errors
                if (
                    failure_count > previous_failure_count
                    or error_count > previous_error_count
                ):
                    new_value = max(-100, previous_reward - 50)
                    return Reward(
                        value=new_value,
                        explanation=f"Test failures/errors increased: failures {previous_failure_count}->{failure_count}, errors {previous_error_count}->{error_count}",
                    ), None
                elif (
                    failure_count < previous_failure_count
                    and error_count <= previous_error_count
                ):
                    new_value = min(75, previous_reward + 25)
                    return Reward(
                        value=new_value,
                        explanation=f"Test failures decreased: {previous_failure_count}->{failure_count}, errors {previous_error_count}->{error_count}",
                    ), None
                else:
                    new_value = max(-100, previous_reward - 25)
                    return Reward(
                        value=new_value,
                        explanation=f"No improvement in test results: failures {failure_count}, errors {error_count}",
                    ), None

        if node.action.name == "ViewCode":
            # Check files in properties for new span IDs
            files_info = node.observation.properties.get("files", {}).values()
            files_with_new_spans = [
                file for file in files_info if file.get("new_span_ids")
            ]

            if len(files_with_new_spans) == len(files_info) and files_with_new_spans:
                return Reward(
                    value=100,
                    explanation="Successfully added relevant code context for all requested files",
                ), None
            elif files_with_new_spans:
                return Reward(
                    value=50,
                    explanation="Successfully added some new code context, but not for all requested files",
                ), None

            return Reward(
                value=-25, explanation="Request completed but no new context was needed"
            ), None

        return super().get_reward(node)
