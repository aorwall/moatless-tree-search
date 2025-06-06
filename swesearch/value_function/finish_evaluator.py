import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from moatless.node import Node, Reward
from moatless.actions.finish import Finish
from moatless.completion.base import BaseCompletionModel
from moatless.completion.schema import AllMessageValues, ChatCompletionUserMessage, ResponseSchema
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.message_history.base import BaseMemory
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)


class ProvideReward(ResponseSchema):
    """A structured output for providing reward values and feedback for actions."""
    
    explanation: str = Field(
        ...,
        description="An explanation and the reasoning behind your decision."
    )
    value: int = Field(
        ...,
        description="A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
        ge=-100,
        le=100
    )


class FinishEvaluator(BaseValueFunction):
    """
    Specialized value function for evaluating Finish actions.
    
    This evaluator focuses on:
    - Whether the decision to finish was reasonable
    - If tests were added/updated appropriately
    - If changes are correct and complete
    - If the trajectory leading to finish was optimal
    """
    
    completion_model: BaseCompletionModel = Field(
        description="Completion model to be used for generating completions"
    )
    
    async def get_reward(self, node: Node) -> Optional[Reward]:
        if not self.completion_model:
            raise ValueError("Completion model is not set")
        
        # Find the finish action step
        finish_action_step = None
        for action_step in node.action_steps:
            if action_step.action.name == "Finish":
                finish_action_step = action_step
                break
        
        if not finish_action_step:
            logger.info(f"No Finish action found for node {node.node_id}, skipping reward")
            return None
        
        last_message = self._build_finish_evaluation_message(node, finish_action_step)
        
        # Create a proper message and append to list
        if last_message:
            messages = [
                ChatCompletionUserMessage(
                    content=last_message
                )
            ]

        system_prompt = self._create_finish_system_prompt(node)

        # Add defensive check
        if not messages:
            messages = [{"role": "user", "content": "No message history available"}]

        try:
            if self.completion_model and not self.completion_model._initialized:
                self.completion_model.initialize(response_schema=ProvideReward)
            
            completion_response = await self.completion_model.create_completion(
                messages=messages, system_prompt=system_prompt
            )
            
            if not completion_response.structured_output:
                logger.error(f"No structured output received from completion model")
                return None
            elif not isinstance(completion_response.structured_output, ProvideReward):
                logger.error(f"Structured output type is not of type ProvideReward: {type(completion_response.structured_output)}")
                return None
            else:
                reward = Reward(
                    value=completion_response.structured_output.value,
                    explanation=completion_response.structured_output.explanation,
                    completion=completion_response.completion_invocation
                )
                return reward

        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            raise

    def _build_finish_evaluation_message(self, node: Node, finish_action_step) -> str:
        """Build the evaluation message specifically for Finish actions."""
        message = "# Completion Reasoning\n"
        message += "<reasoning_for_completion>\n"
        message += finish_action_step.action.finish_reason
        message += "\n</reasoning_for_completion>\n\n"

        # Add the complete trajectory for evaluation
        trajectory = node.get_trajectory()
        if len(trajectory) > 1:  # More than just the finish action
            message += "# Complete Trajectory Analysis\n"
            message += "The following is the complete sequence of actions taken leading to this finish:\n\n"
            
            for i, trajectory_node in enumerate(trajectory[:-1]):  # Exclude the finish node itself
                if trajectory_node.action_steps:
                    action_step = trajectory_node.action_steps[-1]  # Get the most recent action
                    message += f"## Step {i+1}: {action_step.action.name}\n"
                    message += action_step.action.to_prompt()
                    if action_step.observation and action_step.observation.message:
                        message += f"\n**Output:** {action_step.observation.message[:500]}{'...' if len(action_step.observation.message) > 500 else ''}\n"
                    message += "\n"

        # Format the file context section
        if node.parent and node.parent.file_context:
            message += "# Final File Context\n"
            message += "The following code context represents the final state when finishing:\n\n"
            message += "<file_context>\n"
            message += node.parent.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            message += "\n</file_context>\n\n"

            # Format the git patch section
            full_patch = node.parent.file_context.generate_git_patch()
            if full_patch.strip():
                message += "# All Changes Made\n"
                message += "Git diff of all changes made during the trajectory:\n\n"
                message += "<git_patch>\n"
                message += full_patch
                message += "\n</git_patch>\n\n"


        return message

    def _create_finish_system_prompt(self, node: Node) -> str:
        """Create system prompt specifically for evaluating Finish actions."""
        trajectory_length = len(node.get_trajectory())
        
        # Use the Finish action's specific prompts and criteria
        base_prompt = Finish.get_value_function_prompt()
        if not base_prompt:
            base_prompt = """Your role is to evaluate the executed action of the search tree that our AI agents are traversing, with the goal of ensuring that a complete and verified solution is in place. The agent believes that it has finished solving the programming issue."""

        criteria_list = Finish.get_evaluation_criteria(trajectory_length)
        reward_scale_list = Finish.get_reward_scale(trajectory_length)
        
        min_value, max_value = -100, 100

        evaluation_criteria_text = self._format_evaluation_criteria(criteria_list)
        reward_scale_text = self._format_reward_scale(reward_scale_list, min_value, max_value)

        prompt = base_prompt + evaluation_criteria_text + reward_scale_text

        prompt += f"""
# Feedback Structure:

* **Explanation**: Provide a comprehensive evaluation of the complete trajectory leading to this finish action. Focus on:
  - Whether all aspects of the original issue have been addressed
  - Quality and correctness of the solution implementation
  - Adequacy of test coverage and updates
  - Efficiency of the solution path taken
  - Whether the finish decision was premature or well-timed

* **Reward**: Assign a single integer value between {min_value} and {max_value} based on the overall quality of the solution and trajectory. Remember that scores of 75+ require proper test coverage.
"""

        return prompt

    def _format_evaluation_criteria(self, criteria_list) -> str:
        formatted_criteria = "\n# Evaluation Criteria:\n"
        for criterion in criteria_list:
            formatted_criteria += f"* {criterion}\n"
        return formatted_criteria

    def _format_reward_scale(self, reward_scale_list, min_value: int, max_value: int) -> str:
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
        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump(**kwargs)
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "FinishEvaluator":
        if isinstance(obj, dict):
            obj = obj.copy()
            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj.pop("completion_model"))
            
            return super().model_validate(obj)

        return obj