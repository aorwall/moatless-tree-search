from typing import List, Any

from moatless.actions.schema import ActionArguments
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.node import FeedbackData, Node
from moatless.value_function.base import Reward

import logging

logger = logging.getLogger(__name__)

class RewardFeedbackGenerator(BaseFeedbackGenerator):
    async def generate_feedback(self, node: Node) -> FeedbackData | None:
        logger.info(f"Generating reward feedback for node {node.node_id}")
        if not node.parent:
            logger.info(f"Node {node.node_id} has no parent")
            return None
        
        visited_siblings = [sibling for sibling in node.parent.children if sibling.reward and sibling.node_id != node.node_id]
        if not visited_siblings:
            logger.info(f"No visited siblings for node {node.node_id}")
            return None

        # Pick last child to always use new feedback
        last_sibling = visited_siblings[-1]
        
        if not last_sibling.action:
            logger.warning(f"Last sibling {last_sibling.node_id} has no action")
            return None
        if not last_sibling.reward:
            logger.warning(f"Last sibling {last_sibling.node_id} has no reward")
            return None        
        
        feedback = self._create_message_alt_action(last_sibling.reward, last_sibling.action)
        return FeedbackData(
            feedback=feedback
        )  # type: ignore

    def _create_message_alt_action(self, reward: Reward, action: ActionArguments):
        FEEDBACK_PROMPT = """The following information describes an action taken in a parallel branch of problem-solving, not in your current trajectory. This action represents an approach taken by a different agent in an entirely separate problem-solving branch. It is not part of your own history or decision-making process. This information is provided solely to inform your decision-making and inspire potential improvements to your approach.

<Alternative_Branch_Action>: An action executed in a completely separate problem-solving branch, distinct from your current path. This action shows how a different agent addressed the same problem you're working on, but in a parallel decision tree. It is not a previous action in your own sequence of decisions.

<Feedback>: The evaluation feedback provided on the Alternative Branch Action. It consists of:
1) An <Assigned_Value>: A numerical score ranging from -100 (lowest) to 100 (highest), indicating the perceived effectiveness of the action in that separate branch.
2) An <Explanation>: A detailed written evaluation of the Alternative Branch Action, analyzing its strengths, weaknesses, and overall impact on solving the problem in that particular branch. This feedback does not reflect on your own actions or decisions.
"""

        feedback = [
            FEEDBACK_PROMPT,
            "<Alternative_Branch_Action>",
            action.to_prompt(),
            "</Alternative_Branch_Action>",
            "",
            f"<Assigned_Value>{reward.value}</Assigned_Value>" "<Explanation>",
            reward.explanation,
            "</Explanation>",
            "",
            "Based on this alternative branch information, propose a new action for your current trajectory.",
        ]

        return "\n".join(feedback)
