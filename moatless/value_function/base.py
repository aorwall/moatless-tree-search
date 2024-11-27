import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, PrivateAttr, Field

from moatless.actions.action import Action, RewardScaleEntry
from moatless.completion.completion import CompletionModel, Message
from moatless.completion.model import UserMessage, Completion
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node
from moatless.schema import MessageHistoryType
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)


class ValueFunction(BaseModel):
    completion_model: CompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )
    message_generator: MessageHistoryGenerator = Field(
        default_factory=lambda: MessageHistoryGenerator(),
        description="Generator for message history"
    )
    correction_award: Optional[int] = Field(
        0,
        description="The reward value to automatically assign when the agent expects a correction.",
    )

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        if node.observation.expect_correction and self.correction_award is not None:
            logger.info(
                f"Expecting a correction, assigning reward {self.correction_award}"
            )
            return Reward(
                value=self.correction_award, explanation="Expects a correction"
            ), None

        if node.action.name == "Reject":
            logger.info(f"Reject action, assigning reward -100")
            return Reward(value=-100, explanation="Reject action"), None

        if node.action.name == "Error":
            logger.info(f"Error action, assigning reward -100")
            return Reward(value=-100, explanation="Error action"), None


        messages = self.message_generator.generate(node)

        last_message = ""

        if node.action.name == "Finish":
            last_message += "<reasoning_for_completion>\n"
            last_message += node.action.finish_reason
            last_message += "</reasoning_for_completion>\n"
        else:
            last_message += "## Last Executed Action:\n"
            last_message += "Here is the most recent action that was executed and its output. This is the subject of your evaluation.\n"
            last_message += "\n<executed_action>\n"
            last_message += node.action.to_prompt()
            last_message += f"\n\n**Output:**\n{node.observation.message}"
            last_message += "\n</executed_action>\n\n"

        if not node.parent.file_context.is_empty():
            last_message += (
                "The file context the agent had access to when executing the new action"
            )
            last_message += node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

        full_patch = node.parent.file_context.generate_git_patch()
        if full_patch.strip():
            last_message += "\n\nThe git diff of the already made changes before executing the action:\n"
            last_message += "<git_patch>\n"
            last_message += full_patch
            last_message += "\n</git_patch>\n"

        messages.append(UserMessage(content=last_message))

        system_prompt = self._create_system_prompt(node)

        return self.completion_model.create_completion(
            messages=messages, system_prompt=system_prompt, response_model=Reward
        )

    def _create_system_prompt(self, node: Node) -> str:
        return self._build_system_prompt(node)

    def _create_message(self, node: Node) -> Message:
        previous_nodes = node.get_trajectory()[:-1]

        message = node.get_root().message

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
                    formatted_state += (
                        f"\n\nOutput: {previous_node.observation.message}"
                    )
                    formatted_history.append(formatted_state)
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")

        if formatted_history:
            message += "Below is the history of previously executed actions and their outputs that led up to the current state.\n"
            message += "<history>\n"
            message += "\n".join(formatted_history)
            message += "\n</history>\n\n"

        if node.action.name == "Finish":
            message += "<reasoning_for_completion>\n"
            message += node.action.finish_reason
            message += "</reasoning_for_completion>\n"
        else:
            message += "## Last Executed Action:\n"
            message += "Here is the most recent action that was executed and its output. This is the subject of your evaluation.\n"
            message += "\n<executed_action>\n"
            message += node.action.to_prompt()
            message += f"\n\n**Output:**\n{node.observation.message}"
            if node.observation.extra:
                message += f"\n{node.observation.extra}"
            message += "\n</executed_action>\n\n"

        message += "Current state of relevant files and code context after the last executed action:\n"
        message += "<file_context>\n"
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
        message += "Changes made to the codebase so far:\n"
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
* **Feedback to Alternative Branch**: Offer guidance for a parallel problem-solving branch. Suggest conceptual alternative approaches or strategies without providing actual code implementations.
* **Reward**: Assign a single integer value between {min_value} and {max_value} based on your confidence in the correctness of the action and its likelihood of resolving the issue.
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
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ValueFunction":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion_model", None)
            value_function_class_path = obj.pop("value_function_class", None)

            if completion_data:
                obj["completion_model"] = CompletionModel.model_validate(completion_data)
            else:
                obj["completion_model"] = None

            if value_function_class_path:
                module_name, class_name = value_function_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                value_function_class = getattr(module, class_name)
                instance = value_function_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)
