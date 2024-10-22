import logging
from typing import List, Optional, Tuple

from pydantic import BaseModel, PrivateAttr

from moatless.completion.completion import CompletionModel
from moatless.completion.model import Message, UserMessage, Completion
from moatless.node import Node, Reward
from moatless.value_function.model import RewardScaleEntry

logger = logging.getLogger(__name__)


class ValueFunction(BaseModel):
    _completion: CompletionModel = PrivateAttr()

    def __init__(self, completion: CompletionModel):
        super().__init__()
        self._completion = completion

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        message = self._create_message(node)
        system_prompt = self._create_system_prompt(node)

        return self._completion.create_completion([message], system_prompt, [Reward])

    def _create_system_prompt(self, node: Node) -> str:
        return self._build_system_prompt(node)

    def _create_message(self, node: Node) -> Message:
        previous_nodes = node.get_trajectory()[:-1]

        message = f"<problem_statement>\n"
        message += node.get_root().message
        message += "\n</problem_statement>\n\n"

        formatted_history: List[str] = []
        counter = 0
        for previous_node in previous_nodes:
            if previous_node.action:
                counter += 1
                formatted_state = (
                    f"\n## {counter}. Action: {previous_node.action.name}\n"
                )
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    formatted_state += f"\n\nOutput: {previous_node.observation.message}"
                    formatted_history.append(formatted_state)
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")

        if formatted_history:
            message += "<history>\n"
            message += "\n".join(formatted_history)
            message += "\n</history>\n\n"

        if node.action.name == "Finish":
            message += "<reasoning_for_completion>\n"
            message += node.action.finish_reason
            message += "</reasoning_for_completion>\n"
        else:
            message += "## Last Executed Action:\n"
            message += "\n\n<executed_action>\n"
            message += node.action.to_prompt()
            message += f"\n\n**Output:**\n{node.observation.message}"
            if node.observation.extra:
                message += f"\n{node.observation.extra}"
            message += "\n</executed_action>\n\n"

        message += "\n<file_context>\n"
        if node.file_context and not node.file_context.is_empty():
            message += node.file_context.create_prompt(
                show_outcommented_code=True,
                exclude_comments=True,
                outcomment_code_comment="... code not in context",
            )
        else:
            message += "No files added to file context yet."
        message += "\n</file_context>\n\n"

        full_patch = node.file_context.generate_git_patch()
        if full_patch.strip():
            message += "<git_patch>\n"
            message += full_patch
            message += "\n</git_patch>\n\n"
        else:
            message += "<git_patch>\n"
            message += "No changes made yet."
            message += "\n</git_patch>\n\n"

        return UserMessage(content=message)

    def _build_system_prompt(self, node: Node):
        action = node.action
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
* **Feedback to Alternative Branch**: Offer guidance for a parallel problem-solving branch. Suggest conceptual alternative approaches or strategies without providing actual code implementations.
* **Reward**: Assign a single integer value between {min_value} and {max_value} based on your confidence in the correctness of the action and its likelihood of resolving the issue.
"""

        if node.possible_actions:
            prompt += "\n\n# Possible Actions:\n"
            prompt += (
                "The following actions were available for the agent to choose from:\n\n"
            )
            for action in node.possible_actions:
                try:
                    schema = action.model_json_schema()
                    prompt += f"* **{schema['title']}**: {schema['description']}"
                except Exception as e:
                    logger.error(
                        f"Error while building prompt for action {action}: {e}"
                    )

        return prompt

    @staticmethod
    def _format_evaluation_criteria(criteria_list: List[str]) -> str:
        formatted_criteria = "# Evaluation Criteria:\n\n"
        for criterion in criteria_list:
            formatted_criteria += f"* {criterion}\n"
        return formatted_criteria

    @staticmethod
    def _format_reward_scale(
        reward_scale_list: List[RewardScaleEntry], min_value: int, max_value: int
    ) -> str:
        formatted_scale = "# Reward Scale and Guidelines:\n\n"
        sorted_entries = sorted(reward_scale_list, key=lambda x: -x.max_value)

        formatted_scale += f"The reward value must be an integer between {min_value} and {max_value}, where:\n\n"

        for entry in sorted_entries:
            if entry.min_value == entry.max_value:
                formatted_scale += f"* **{entry.min_value}**: {entry.description}\n"
            else:
                formatted_scale += f"* **{entry.min_value} to {entry.max_value}**: {entry.description}\n"

        return formatted_scale

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump() if self._completion else None
        return dump

    @classmethod
    def model_validate(cls, obj):
        completion = None
        if "completion" in obj and obj["completion"]:
            completion = CompletionModel.model_validate(obj["completion"])
        return cls(completion=completion)
