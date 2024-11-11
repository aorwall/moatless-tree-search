from moatless.actions.code_change import RequestCodeChange
from moatless.actions.run_tests import RunTests
from moatless.node import Node
from moatless.feedback.feedback import FeedbackGenerator
from typing import List, Any

import logging
logger = logging.getLogger(__name__)


# class FeedbackGenerator(BaseModel):
#     def generate_feedback(self, node: Node) -> str | None:
#         if node.reward:
#             if node.reward.feedback:
#                 return self._create_message_feedback_llmselector(node)

#         # else:

class RewardFeedbackGenerator(FeedbackGenerator):
    def generate_feedback(self, node: Node, actions: List[Any] = None) -> str | None:
        if node.reward:
            if node.reward.feedback:
                return self._create_message_feedback_llmselector(node)
        
        else:
            visited_children = [child for child in node.children if child.reward]
            if not visited_children:
                return None

            # Pick last child to always use new feedback
            last_child = visited_children[-1]
            if last_child.action.name in [RunTests.name]:
                return self._create_message(last_child)
            elif (
                last_child.action.name not in [RequestCodeChange.name]
                and last_child.reward.feedback
            ):
                return self._create_message_feedback(last_child)
            else:
                return self._create_message_alt_action(last_child)

    def _create_message(self, node: Node, feedback: str | None = None):
        prompt = "Feedback from a parallel problem-solving branch is provided within the <feedback> tag. Carefully review this feedback and use it to adjust your search parameters, ensuring that you implement a different search strategy from previous attempts. "
        prompt += "\n\n<feedback>\n"
        prompt += feedback if feedback is not None else node.reward.explanation
        prompt += "\n</feedback>"
        return prompt

    def _create_message_alt_action(self, node: Node):
        FEEDBACK_PROMPT = """The following information describes an action taken in a parallel branch of problem-solving, not in your current trajectory. This action represents an approach taken by a different agent in an entirely separate problem-solving branch. It is not part of your own history or decision-making process. This information is provided solely to inform your decision-making and inspire potential improvements to your approach.

<Alternative_Branch_Action>: An action executed in a completely separate problem-solving branch, distinct from your current path. This action shows how a different agent addressed the same problem you're working on, but in a parallel decision tree. It is not a previous action in your own sequence of decisions.

<Feedback>: The evaluation feedback provided on the Alternative Branch Action. It consists of:
1) An <Assigned_Value>: A numerical score ranging from -100 (lowest) to 100 (highest), indicating the perceived effectiveness of the action in that separate branch.
2) An <Explanation>: A detailed written evaluation of the Alternative Branch Action, analyzing its strengths, weaknesses, and overall impact on solving the problem in that particular branch. This feedback does not reflect on your own actions or decisions.
"""

        feedback = [
            FEEDBACK_PROMPT,
            "<Alternative_Branch_Action>",
            node.action.to_prompt(),
            "</Alternative_Branch_Action>",
            "",
            f"<Assigned_Value>{node.reward.value}</Assigned_Value>" "<Explanation>",
            node.reward.explanation,
            "</Explanation>",
            "",
            "Based on this alternative branch information, propose a new action for your current trajectory.",
        ]

        return "\n".join(feedback)

    def _create_message_feedback(self, node: Node, feedback: str | None = None):
        prompt = "Feedback from a parallel problem-solving branch is provided within the <feedback> tag. Carefully review this feedback and use it to do a new function call ensuring that you implement a different strategy from previous attempts. "
        prompt += "\n\n<feedback>\n"
        prompt += feedback if feedback is not None else node.reward.feedback
        prompt += "\n</feedback>"
        return prompt
    
    def _create_message_feedback_llmselector(self, node: Node):
        prompt = "This is feedback from an Agent monitoring your progress. Take it into account."
        prompt += "\n\n<feedback>\n"
        prompt += node.reward.feedback
        prompt += "\n</feedback>"
        return prompt 