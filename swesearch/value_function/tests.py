import logging
from typing import Optional, Tuple

from moatless.actions.search_base import SearchBaseArgs
from moatless.node import Node, ActionStep
from moatless.value_function.base import BaseValueFunction
from moatless.node import Reward

logger = logging.getLogger(__name__)


class TestsValueFunction(BaseValueFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def get_reward(self, node: Node) -> Optional[Reward]:
        for action_step in node.action_steps:
            if action_step.action.name == "RunTests":
                return self.get_reward_for_run_tests(node)
        
        return None
        
    def get_reward_for_run_tests(self, node: Node) -> Optional[Reward]:
        if not node.file_context:
            return None
        
        # Get current test results
        passed_count, failure_count, error_count = (
            node.file_context.get_test_counts()
        )
        total_tests = passed_count + failure_count + error_count

        # Get previous test results
        previous_failure_count = 0
        previous_error_count = 0
        previous_reward = 100
        parent_node = node.parent
        if (
            parent_node
            and parent_node.file_context
            and parent_node.file_context.was_edited()
        ):
            (
                previous_passed_count,
                previous_failure_count,
                previous_error_count,
            ) = parent_node.file_context.get_test_counts()
            if parent_node.reward:
                previous_reward = parent_node.reward.value

        if total_tests == 0:
            return Reward(value=50, explanation=f"No tests run")
        elif failure_count == 0 and error_count == 0:
            return Reward(
                value=100, explanation=f"All {passed_count} tests passing"
            )
        elif (
            failure_count > previous_failure_count
            or error_count > previous_error_count
        ):
            new_value = max(-100, previous_reward - 25)
            return Reward(
                value=new_value,
                explanation=f"Test failures increased: {previous_failure_count}->{failure_count}, errors {previous_error_count}->{error_count}",
            )
        elif (
            failure_count < previous_failure_count
            and error_count <= previous_error_count
        ):
            new_value = min(75, previous_reward + 25)
            return Reward(
                value=new_value,
                explanation=f"Test failures decreased: {previous_failure_count}->{failure_count}, errors {previous_error_count}->{error_count}",
            )
        else:
            new_value = max(-100, previous_reward - 25)
            return Reward(
                value=new_value,
                explanation=f"No improvement in test results: failures {failure_count}, errors {error_count}",
            )

    