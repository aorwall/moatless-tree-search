import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from moatless.node import Node, Reward
from moatless.actions.action import Action
from moatless.actions.think import ThinkArgs, Think
from moatless.completion.base import BaseCompletionModel
from moatless.completion.schema import AllMessageValues, ChatCompletionUserMessage, ResponseSchema
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.message_history.base import BaseMemory
from moatless.node import generate_ascii_tree
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


class ThinkEvaluator(BaseValueFunction):
    """
    Specialized value function for evaluating Think actions.
    
    This evaluator focuses on:
    - Quality and depth of reasoning
    - Whether the agent understands the problem correctly
    - If the agent is on the right track or hitting a dead end
    - Appropriateness of timing for the Think action
    """
    
    completion_model: BaseCompletionModel = Field(
        description="Completion model to be used for generating completions"
    )
    
    async def get_reward(self, node: Node) -> Optional[Reward]:
        if not self.completion_model:
            raise ValueError("Completion model is not set")
        
        # Find the think action step
        think_action_step = None
        for action_step in node.action_steps:
            if isinstance(action_step.action, ThinkArgs):
                think_action_step = action_step
                break
        
        if not think_action_step:
            logger.info(f"No Think action found for node {node.node_id}, skipping reward")
            return None
        
        message = self._build_think_evaluation_message(node, think_action_step)
        messages = [
            ChatCompletionUserMessage(
                content=message
            )
        ]

        system_prompt = self._create_think_system_prompt(node)

        try:
            if self.completion_model and not self.completion_model._initialized:
                self.completion_model.initialize(response_schema=ProvideReward)
            
            completion_response = await self.completion_model.create_completion(
                messages=messages, system_prompt=system_prompt
            )
            
            if completion_response.structured_output and isinstance(completion_response.structured_output, ProvideReward):
                reward = Reward(
                    value=completion_response.structured_output.value,
                    explanation=completion_response.structured_output.explanation,
                    completion=completion_response.completion_invocation
                )
                return reward
            else:
                logger.error(f"No structured output received from completion model")
                return None

        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            raise

    def _build_think_evaluation_message(self, node: Node, think_action_step) -> str:
        """Build the evaluation message specifically for Think actions."""
        message = "# Think Action Reasoning\n"
        message += "<agent_reasoning>\n"
        message += think_action_step.action.thought
        message += "\n</agent_reasoning>\n\n"

        # Add trajectory leading to this Think action
        trajectory = node.get_trajectory()
        if len(trajectory) > 1:  # More than just the think action
            message += "# Trajectory Leading to Think Action\n"
            message += "The following actions were taken before this Think action:\n\n"
            
            for i, trajectory_node in enumerate(trajectory[:-1]):  # Exclude the current node
                if trajectory_node.action_steps:
                    action_step = trajectory_node.action_steps[-1]  # Get the most recent action
                    message += f"## Step {i+1}: {action_step.action.name}\n"
                    message += action_step.action.to_prompt()
                    if action_step.observation and action_step.observation.message:
                        message += f"\n**Output:** {action_step.observation.message[:300]}{'...' if len(action_step.observation.message) > 300 else ''}\n"
                    message += "\n"

        # Analyze potential dead ends or progression blockers
        message += "# Dead End Detection\n"
        message += "Consider whether the agent might be:\n"
        message += "- Repeating similar actions without progress\n"
        message += "- Misunderstanding the core problem\n"
        message += "- Focusing on irrelevant details\n"
        message += "- Lacking a clear strategy to move forward\n\n"

        # Format the file context section if available
        if node.parent and node.parent.file_context:
            message += "# Current File Context\n"
            message += "The following code context was available when thinking:\n\n"
            message += "<file_context>\n"
            message += node.parent.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            message += "\n</file_context>\n\n"

            # Format any changes made so far
            full_patch = node.parent.file_context.generate_git_patch()
            if full_patch.strip():
                message += "# Changes Made So Far\n"
                message += "Git diff of changes made before this Think action:\n\n"
                message += "<git_patch>\n"
                message += full_patch
                message += "\n</git_patch>\n\n"

        return message

    def _create_think_system_prompt(self, node: Node) -> str:
        """Create system prompt specifically for evaluating Think actions."""
        trajectory_length = len(node.get_trajectory())
        
        # Use the Think action's specific prompts and criteria
        base_prompt = Think.get_value_function_prompt()
        if not base_prompt:
            base_prompt = """Your role is to evaluate the Think action executed by the AI agent. The agent used this action to reason about the problem and plan next steps. Evaluate the quality of the reasoning and whether the agent is on the right track or potentially hitting a dead end."""

        criteria_list = Think.get_evaluation_criteria(trajectory_length)
        reward_scale_list = Think.get_reward_scale(trajectory_length)
        
        min_value, max_value = -100, 100

        evaluation_criteria_text = self._format_evaluation_criteria(criteria_list)
        reward_scale_text = self._format_reward_scale(reward_scale_list, min_value, max_value)

        prompt = base_prompt + evaluation_criteria_text + reward_scale_text

        prompt += f"""
# Feedback Structure:

* **Explanation**: Provide a detailed evaluation of the Think action focusing on:
  - Quality and depth of the reasoning
  - Whether the agent demonstrates correct understanding of the problem
  - If the reasoning leads to a clear and appropriate action plan
  - Whether the agent is potentially hitting a dead end or progressing effectively
  - Technical accuracy of the reasoning content

* **Reward**: Assign a single integer value between {min_value} and {max_value} based on the quality of reasoning and its likelihood to lead to effective problem-solving.
"""

        # Add dead end detection guidance
        prompt += """
# Dead End Detection Guidelines:
* **Repetitive Patterns**: Look for signs the agent is stuck in loops or repeating unsuccessful approaches
* **Problem Misunderstanding**: Check if the reasoning reveals fundamental misunderstanding of the issue
* **Lack of Progress**: Evaluate whether the reasoning shows clear path forward or indicates confusion
* **Strategy Clarity**: Assess whether the agent has a coherent strategy or is exploring randomly
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
    def model_validate(cls, obj: Any) -> "ThinkEvaluator":
        if isinstance(obj, dict):
            obj = obj.copy()
            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj.pop("completion_model"))

            return super().model_validate(obj)

        return obj