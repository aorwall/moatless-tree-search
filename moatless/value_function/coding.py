import logging
from typing import Optional, Tuple

from moatless.actions.search_base import SearchBaseArgs
from moatless.completion.model import Completion
from moatless.node import Node
from moatless.value_function.base import ValueFunction
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)

FAILURE_REWARDS = {
    'MAJOR': {
        'file_not_found': 'Requested file does not exist in repository',
        'is_directory': 'Requested path is a directory, not a file',
        'invalid_file': 'File exists but could not be parsed or is empty',
        'no_spans_found': 'No code spans found matching the request',
        'string_not_found': 'String to replace was not found in file',
        'string_already_exists': 'New string already exists in file',
        'no_changes': 'Old and new strings are identical - no changes needed',
    },

    'MINOR': {
        'too_many_tokens': 'Requested context exceeds token limit, needs to be more specific',
        'no_spans_added': 'Requested spans were already in context',
        'no_search_hits': 'Search returned no results',
        'file_exists': 'Cannot create file - it already exists',
        'multiple_occurrences': 'Multiple occurrences of string found - need more context',
        'indentation_differs': 'Content matches but indentation is incorrect',
        'line_breaks_differs': 'Content matches but line breaks are incorrect',
        'multiple_format_matches': 'Multiple potential matches with different formatting found',
    }
}

FAILURE_VALUES = {
    'MAJOR': -50,
    'MINOR': -25
}

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
                
                # Find the severity and explanation for the failure
                for severity, failures in FAILURE_REWARDS.items():
                    if fail_reason in failures:
                        return Reward(
                            value=FAILURE_VALUES[severity],
                            explanation=failures[fail_reason]
                        ), None
                
                # Default case for unknown failures
                return Reward(value=-100, explanation="Action failed"), None

            if isinstance(node.action, SearchBaseArgs):
                if not node.observation.properties.get("new_span_ids"):  # TODO Use fail reason?
                    return Reward(
                        value=-25,
                        explanation="Search returned results but did not add any new spans to the context",
                    ), None

                return Reward(
                    value=50,
                    explanation="Search returned results and added new spans to the context",
                ), None

            if node.file_context.was_edited():
                passed_count, failure_count, error_count = node.file_context.get_test_counts()

                # Get previous test results
                previous_failure_count = 0
                previous_error_count = 0
                previous_passed_count = 0
                previous_reward = 0
                parent_node = node.parent
                if parent_node and parent_node.file_context:
                    previous_passed_count, previous_failure_count, previous_error_count = parent_node.file_context.get_test_counts()
                    if parent_node.reward:
                        previous_reward = parent_node.reward.value
                    previous_total_tests = previous_passed_count + previous_failure_count + previous_error_count

                # Compare with previous test results
                total_tests = passed_count + failure_count + error_count
                if total_tests == 0:
                    return Reward(
                        value=25, explanation="No tests run"
                    ), None
                elif failure_count == 0 and error_count == 0:
                    return Reward(
                        value=100, explanation=f"All {passed_count} tests passing"
                    ), None
                elif previous_total_tests == 0:
                    return Reward(
                        value=50,
                        explanation=f"First test run with {failure_count} failures and {error_count} errors",
                    ), None
                elif failure_count > previous_failure_count or error_count > previous_error_count:
                    new_value = max(-100, previous_reward - 50)
                    return Reward(
                        value=new_value,
                        explanation=f"Test failures/errors increased: failures {previous_failure_count}->{failure_count}, errors {previous_error_count}->{error_count}",
                    ), None
                elif failure_count < previous_failure_count and error_count <= previous_error_count:
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
                    value=50,
                    explanation="Successfully added relevant code context for all requested files",
                ), None
            elif files_with_new_spans:
                return Reward(
                    value=25,
                    explanation="Successfully added some new code context, but not for all requested files",
                ), None

            return Reward(
                value=-25, explanation="Request completed but no new context was needed"
            ), None

        return super().get_reward(node)
