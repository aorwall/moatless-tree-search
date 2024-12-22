import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import BaseModel, PrivateAttr, Field

from moatless.actions.action import Action, RewardScaleEntry
from moatless.completion.completion import CompletionModel, Message
from moatless.completion.model import Completion, StructuredOutput
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node, generate_ascii_tree
from moatless.value_function.base import ValueFunction
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)

class ProvideReward(StructuredOutput):
    """Provide a reward value and feedback for a finished solution."""

    explanation: str = Field(
        ..., 
        description="Provide a detailed analysis of how well the solution solves the original task. Consider functionality, correctness, and completeness. Focus on evaluating the end result rather than the process."
    )
    
    feedback: str = Field(
        ..., 
        description="Write a direct message to a new AI agent that will attempt to solve this task from scratch. The agent has no knowledge of the current solution. Suggest high-level approaches and strategies they should consider. Focus on conceptual guidance rather than specific implementation details. This feedback will be used as initial strategic guidance for their completely fresh attempt."
    )
    value: int = Field(
        ...,
        description="A single integer value based on how well the solution addresses the original requirements",
        ge=-100,
        le=100
    )


class TerminalValueFunction(BaseModel):
    """Value function for evaluating finished solutions.
    
    This class evaluates complete solutions to determine how well they solve the original task.
    It provides:
    - A numerical reward value (-100 to 100)
    - An explanation analyzing the solution quality
    - Feedback suggesting alternative approaches for future attempts
    
    The feedback is designed to guide completely new solution attempts from scratch,
    focusing on high-level strategies rather than specific implementation details.
    This helps explore different approaches to solving the same task.
    
    Note: This value function can only evaluate nodes with a "Finish" action.
    For evaluating intermediate steps, use the base ValueFunction instead.
    """

    completion_model: CompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )
    message_generator: MessageHistoryGenerator = Field(
        default_factory=lambda: MessageHistoryGenerator(),
        description="Generator for message history"
    )
    

    def get_reward(self, node: Node) -> Tuple[Optional[Reward], Optional[Completion]]:
        if node.action.name != "Finish":
            logger.warning(f"TerminalValueFunction can only evaluate finished solutions, but got action {node.action.name}")
            return None, None

        
        user_message = self._create_message(node)

        messages = [user_message]
        
        system_prompt = self._build_system_prompt(node)

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages, 
                system_prompt=system_prompt, 
                response_model=ProvideReward
            )

            if completion_response.structured_output:
                reward = Reward(
                    value=completion_response.structured_output.value,
                    explanation=completion_response.structured_output.explanation,
                    feedback=completion_response.structured_output.feedback
                )
            
                return reward, completion_response.completion
            else:
                logger.error("No structured output found in completion response")
                return None, None

        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            raise

    def _create_message(self, node: Node) -> ChatCompletionUserMessage:
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
                        f"\n\nObservation: {previous_node.observation.summary}"
                    )
                    formatted_history.append(formatted_state)
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")

        if formatted_history:
            message += "\n\nBelow is the history of previously executed actions and their outputs that led up to the finished solution.\n"
            message += "<history>\n"
            message += "\n".join(formatted_history)
            message += "\n</history>\n\n"

        message += "<reasoning_for_completion>\n"
        message += node.action.finish_reason
        message += "</reasoning_for_completion>\n"

        message += "Current state of relevant files and code context in the finished solution:\n"
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
        message += "Changes made to the codebase:\n"
        if full_patch.strip():
            message += "<git_patch>\n"
            message += full_patch
            message += "\n</git_patch>\n\n"
        else:
            message += "<git_patch>\n"
            message += "No changes made yet."
            message += "\n</git_patch>\n\n"

        return ChatCompletionUserMessage(role="user", content=message)

    def _build_system_prompt(self, node: Node):
        action = Action.get_action_by_args_class(type(node.action))
        trajectory_length = len(node.get_trajectory())

        prompt = """You are evaluating a finished solution to determine how well it solves the original task.
Focus on:
1. Whether the solution fully addresses the requirements
2. The quality and correctness of the implementation
3. Any potential issues or limitations

The user message contains the following sections:
- <task>: The original task description
- <history>: Previous actions and their outputs that led to the solution
- <reasoning_for_completion>: The reasoning for considering this solution complete
- <file_context>: Current state of relevant files and code
- <git_patch>: All changes made to the codebase"""

        criteria_list = action.get_evaluation_criteria(trajectory_length)
        reward_scale_list = action.get_reward_scale(trajectory_length)
        min_value, max_value = action.get_reward_range(trajectory_length)

        evaluation_criteria_text = ValueFunction._format_evaluation_criteria(criteria_list)
        reward_scale_text = ValueFunction._format_reward_scale(reward_scale_list, min_value, max_value)

        prompt += evaluation_criteria_text + reward_scale_text

        prompt += f"\nThe reward value must be an integer between {min_value} and {max_value}."

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
            dump["coding_value_function"] = self.coding_value_function.model_dump(**kwargs)
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "TerminalValueFunction":
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
