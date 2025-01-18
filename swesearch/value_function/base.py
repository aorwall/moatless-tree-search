import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from litellm.types.llms.openai import ChatCompletionUserMessage
from nltk.app.wordnet_app import explanation
from pydantic import BaseModel, Field

from moatless.actions.action import Action, RewardScaleEntry
from moatless.completion import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.completion.schema import ResponseSchema
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node, generate_ascii_tree, Reward

from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)


class ProvideReward(ResponseSchema):
    """A structured output for providing reward values and feedback for actions."""

    class Config:
        title = "ProvideReward"

    explanation: Optional[str] = Field(
        default=None,
        description="An explanation and the reasoning behind your decision.",
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
        ge=-100,
        le=100,
    )


class ValueFunction(BaseValueFunction):
    completion_model: BaseCompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )
    message_generator: MessageHistoryGenerator = Field(
        default_factory=lambda: MessageHistoryGenerator(),
        description="Generator for message history",
    )
    correction_award: Optional[int] = Field(
        0,
        description="The reward value to automatically assign when the agent expects a correction.",
    )
    include_search_tree: bool = Field(
        default=False,
        description="Whether to include the search tree visualization in the value prompt",
    )
    coding_value_function: Optional["ValueFunction"] = Field(
        default=None,
        description="Optional CodingValueFunction to provide additional context for value decisions",
    )

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        # First get coding value function result if enabled
        coding_reward = None
        if self.coding_value_function:
            coding_reward, _ = self.coding_value_function.get_reward(node)

        messages = self.message_generator.generate_messages(node)
        if messages is None:
            messages = []  # Ensure we have a valid list

        last_message = ""

        # Handle automatic reward cases by adding them to the message
        if node.observation.expect_correction and self.correction_award is not None:
            last_message += "# Automatic Reward Assessment\n"
            last_message += f"Action expects a correction. Suggested value: {self.correction_award}\n\n"

        if node.action.name in ["Reject", "Error"]:
            last_message += "# Automatic Reward Assessment\n"
            last_message += (
                f"{node.action.name} action detected. Suggested value: -100\n\n"
            )

        # Format the action section
        if node.action.name == "Finish":
            last_message += "# Completion Reasoning\n"
            last_message += "<reasoning_for_completion>\n"
            last_message += node.action.finish_reason
            last_message += "\n</reasoning_for_completion>\n\n"
        else:
            last_message += "# Last Executed Action\n"
            last_message += "The following action was executed and its output is the subject of your evaluation:\n\n"
            last_message += "<executed_action>\n"
            last_message += f"Action: {node.action.name}\n"
            last_message += node.action.to_prompt()
            last_message += "\n## Output\n"
            last_message += node.observation.message
            last_message += "\n</executed_action>\n\n"

        # Format the file context section
        if not node.parent.file_context.is_empty():
            last_message += "# File Context\n"
            last_message += "The following code context was available when executing the action:\n\n"
            last_message += "<file_context>\n"
            last_message += node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            last_message += "\n</file_context>\n\n"

        # Format the git patch section
        full_patch = node.parent.file_context.generate_git_patch()
        if full_patch.strip():
            last_message += "# Previous Changes\n"
            last_message += "Git diff of changes made before this action:\n\n"
            last_message += "<git_patch>\n"
            last_message += full_patch
            last_message += "\n</git_patch>\n\n"

        # Format the search tree section
        if self.include_search_tree:
            last_message += "# Search Tree State\n"
            last_message += "<search_tree>\n"
            ascii_tree = generate_ascii_tree(
                node.get_root(),
                include_explanation=True,
                use_color=False,
                include_diffs=True,
                include_action_details=False,
                include_file_context=False,
            )
            last_message += ascii_tree
            last_message += "\n</search_tree>\n\n"

        # Ensure we append the message only if we have content
        if last_message:
            messages.append(
                ChatCompletionUserMessage(role="user", content=last_message)
            )

        system_prompt = self._create_system_prompt(node, coding_reward)

        # Add defensive check
        if not messages:
            messages = [
                ChatCompletionUserMessage(
                    role="user", content="No message history available"
                )
            ]

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages, system_prompt=system_prompt, response_schema=ProvideReward
            )

            return Reward(
                value = completion_response.structured_output.value,
                explanation = completion_response.structured_output.explanation,
            ), completion_response.completion

        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            raise

    def _create_system_prompt(
        self, node: Node, coding_reward: Optional[Reward] = None
    ) -> str:
        base_prompt = self._build_system_prompt(node)

        if coding_reward:
            base_prompt += """
# Coding Value Function Context
<coding_assessment>
The automated coding value function has provided the following assessment:
* Value: {coding_reward.value}
* Explanation: {coding_reward.explanation}
It's based on coding heuristics, and may not be perfect.

Evaluation Guidelines:
1. Consider the automated assessment above
2. Either reinforce its reasoning or explain why you disagree
3. Provide your own comprehensive evaluation
</coding_assessment>
""".format(coding_reward=coding_reward)

        if self.include_search_tree:
            base_prompt += """
# Search Tree Analysis
<search_tree_guidelines>
* Use the provided search tree visualization to understand the full solution space
* Consider any existing finished states in your evaluation
* Guide the agent toward novel solutions that differ from previous attempts
* Discourage actions that would lead to duplicate or very similar outcomes
</search_tree_guidelines>
"""

        return base_prompt

    def _build_system_prompt(self, node: Node):
        action = Action.get_action_by_args_class(type(node.action))
        trajectory_length = len(node.get_trajectory())

        base_prompt = action.get_value_function_prompt()

        criteria_list = action.get_evaluation_criteria(trajectory_length)
        reward_scale_list = action.get_reward_scale(trajectory_length)
        min_value, max_value = action.get_reward_range(trajectory_length)

        evaluation_criteria_text = ValueFunction._format_evaluation_criteria(
            criteria_list
        )
        reward_scale_text = ValueFunction._format_reward_scale(
            reward_scale_list, min_value, max_value
        )

        prompt = base_prompt + evaluation_criteria_text + reward_scale_text

        prompt += f"""
# Feedback Structure:

* **Explanation**: Offer a detailed explanation and reasoning behind your decision, focusing on the **last executed action**, its relation to previous actions and its impact.
* **Feedback to Alternative Branch**: Offer guidance for a parallel problem-solving branch. Suggest conceptual alternative approaches or strategies without providing actual code implementations. Use the search tree to guide your feedback, particularly by avoiding to suggest actions that would lead to the same or very similar previous outcomes.
* **Reward**: Assign a single integer value between {min_value} and {max_value} based on your confidence in the correctness of the action and its likelihood of eventually leading to resolving the issue.
"""

        if node.possible_actions:
            prompt += "\n\n# Available Actions:\n"
            prompt += (
                "The following actions were available for the agent to choose from:\n\n"
            )
            for action_name in node.possible_actions:
                action = Action.get_action_by_name(action_name)
                try:
                    schema = action.args_schema.model_json_schema()
                    prompt += f"\n\n## **{schema['title']}\n\n{schema['description']}"
                except Exception as e:
                    logger.error(
                        f"Error while building prompt for action {action}: {e}"
                    )

        return prompt

    @staticmethod
    def _format_evaluation_criteria(criteria_list: List[str]) -> str:
        formatted_criteria = "\n# Evaluation Criteria:\n"
        for criterion in criteria_list:
            formatted_criteria += f"* {criterion}\n"
        return formatted_criteria

    @staticmethod
    def _format_reward_scale(
        reward_scale_list: List[RewardScaleEntry], min_value: int, max_value: int
    ) -> str:
        formatted_scale = "\n# Reward Scale and Guidelines:\n"
        sorted_entries = sorted(reward_scale_list, key=lambda x: -x.max_value)

        formatted_scale += f"The reward value must be an integer between {min_value} and {max_value}, where:\n\n"

        for entry in sorted_entries:
            if entry.min_value == entry.max_value:
                formatted_scale += f"* **{entry.min_value}**: {entry.description}\n"
            else:
                formatted_scale += f"* **{entry.min_value} to {entry.max_value}**: {entry.description}\n"

        return formatted_scale

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion_model"] = self.completion_model.model_dump(**kwargs)
        dump["value_function_class"] = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        if self.coding_value_function:
            dump["coding_value_function"] = self.coding_value_function.model_dump(
                **kwargs
            )
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ValueFunction":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion_model", None)
            value_function_class_path = obj.pop("value_function_class", None)
            coding_value_function_data = obj.pop("coding_value_function", None)

            if completion_data:
                obj["completion_model"] = BaseCompletionModel.model_validate(
                    completion_data
                )
            else:
                obj["completion_model"] = None

            if coding_value_function_data:
                from swesearch.value_function.coding import CodingValueFunction

                obj["coding_value_function"] = CodingValueFunction.model_validate(
                    coding_value_function_data
                )

            if value_function_class_path:
                module_name, class_name = value_function_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                value_function_class = getattr(module, class_name)
                instance = value_function_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    def _combine_rewards(self, reward1: Reward, reward2: Reward) -> Reward:
        """Combine two rewards by averaging their values and concatenating explanations."""
        combined_value = (reward1.value + reward2.value) // 2  # Integer division
        combined_explanation = (
            "Combined Assessment:\n"
            f"1. General Assessment: {reward1.explanation}\n"
            f"2. Code Quality Assessment: {reward2.explanation}"
        )
        return Reward(value=combined_value, explanation=combined_explanation)
