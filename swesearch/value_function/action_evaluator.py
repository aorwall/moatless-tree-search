import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from moatless.node import Node, Reward
from moatless.actions.schema import ActionArguments
from moatless.actions.action import Action, RewardScaleEntry
from moatless.actions.finish import FinishArgs
from moatless.completion.base import BaseCompletionModel, CompletionResponse
from moatless.completion.schema import ChatCompletionUserMessage, ResponseSchema
from moatless.node import Node
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

class ActionEvaluator(BaseValueFunction):
    completion_model: BaseCompletionModel = Field(
        description="Completion model to be used for generating completions"
    )
    target_actions: List[str] = Field(
        default_factory=lambda: [],
        description="List of action names to run the reward function on. If empty, evaluates all actions.",
    )
    
    async def get_reward(self, node: Node) -> Optional[Reward]:
        if not self.completion_model:
            raise ValueError("Completion model is not set")
        
        last_message = ""

        # Handle multiple action steps or find the most recent relevant one
        if not node.action_steps:
            logger.warning(f"No action steps found for node {node.node_id}, skipping reward")
            return None
        
        # Filter action steps based on target_actions if specified
        relevant_action_steps = []
        if self.target_actions:
            # Filter by target action names
            for action_step in node.action_steps:
                if action_step.action.name in self.target_actions:
                    relevant_action_steps.append(action_step)
        else:
            # If no target actions specified, use all action steps
            relevant_action_steps = node.action_steps
        
        if not relevant_action_steps:
            logger.info(f"No relevant action steps found for node {node.node_id}, skipping reward")
            return None
        
            

        last_message += "# Last Executed Actions\n"
        last_message += "The following actions were executed and its output is the subject of your evaluation:\n\n"
        last_message += "<executed_actions>\n"
        for action_step in relevant_action_steps:
            action_args = action_step.action
            action = Action.get_action_by_args_class(type(action_args))
            if not action:
                raise ValueError(f"Action not found for action arguments: {action_args}")
            
            last_message += f"Action: {action_args.name}\n"
            last_message += action_args.to_prompt()
            last_message += "\n## Output\n"
            
            if action_step.observation and action_step.observation.message:
                last_message += action_step.observation.message
                
        last_message += "\n</executed_actions>\n\n"

        
        # Add the complete trajectory for evaluation
        trajectory = node.get_trajectory()
        if len(trajectory) > 1:  # More than just the finish action
            last_message += "# Complete Trajectory Analysis\n"
            last_message += "The following is the complete sequence of actions taken leading to these actions:\n\n"
            
            for i, trajectory_node in enumerate(trajectory[:-1]):  # Exclude the finish node itself
                if trajectory_node.action_steps:
                    action_step = trajectory_node.action_steps[-1]  # Get the most recent action
                    last_message += f"## Step {i+1}: {action_step.action.name}\n"
                    last_message += action_step.action.to_prompt()
                    if action_step.observation and action_step.observation.message:
                        last_message += f"\n**Output:** {action_step.observation.message[:500]}{'...' if len(action_step.observation.message) > 500 else ''}\n"
                    last_message += "\n"

        # Format the file context section
        if node.parent and node.parent.file_context:
            last_message += "# File Context\n"
            last_message += "The following code context was available when executing the action:\n\n"
            last_message += "<file_context>\n"
            last_message += node.parent.file_context.create_prompt(
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

        messages = [
            ChatCompletionUserMessage(
                ChatCompletionUserMessage(role="user", content=last_message) 
            )
        ]
       
        system_prompt = self._create_system_prompt(node, action)

        # Add defensive check
        if not messages:
            messages = [ChatCompletionUserMessage(role="user", content="No message history available")]  # type: ignore

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

    def _create_system_prompt(
        self, node: Node, action: type[Action]
    ) -> str:
        base_prompt = self._build_system_prompt(node, action)

        return base_prompt
    
    def _build_system_prompt(self, node: Node, action: type[Action]):
        trajectory_length = len(node.get_trajectory())

        base_prompt = action.get_value_function_prompt()
        if not base_prompt:
            base_prompt = self._default_base_prompt()
        
        criteria_list = action.get_evaluation_criteria(trajectory_length)
        if hasattr(action, "get_reward_scale"): 
            reward_scale_list = action.get_reward_scale(trajectory_length)
        else:
            reward_scale_list = []
            
        # Use default range if get_reward_range doesn't exist
        min_value, max_value = -100, 100

        evaluation_criteria_text = self._format_evaluation_criteria(
            criteria_list
        )
        reward_scale_text = self._format_reward_scale(
            reward_scale_list, min_value, max_value
        )

        prompt = base_prompt + evaluation_criteria_text + reward_scale_text

        prompt += f"""
# Feedback Structure:

* **Explanation**: Offer a detailed explanation and reasoning behind your decision, focusing on the **last executed action**, its relation to previous actions and its impact.
* **Reward**: Assign a single integer value between {min_value} and {max_value} based on your confidence in the correctness of the action and its likelihood of eventually leading to resolving the issue.
"""

        if node.possible_actions:
            prompt += "\n\n# Available Actions:\n"
            prompt += (
                "The following actions were available for the agent to choose from:\n\n"
            )
            for action_name in node.possible_actions:
                try:
                    action = Action.get_action_by_name(action_name)
                
                    schema = action.args_schema.model_json_schema()
                    prompt += f"\n\n## **{schema['title']}\n\n{schema['description']}"
                except Exception as e:
                    logger.error(
                        f"Error while building prompt for action {action}: {e}"
                    )

        return prompt
    
    def _default_base_prompt(self) -> str:
        return """Your role is to evaluate the **last executed action** of the search tree that our AI agents are traversing, to help us determine the best trajectory to solve a programming issue. The agent is responsible for identifying and modifying the correct file(s) in response to the problem statement.

Important: While line numbers may be referenced in the initial problem description, they can shift as changes are made to the file. Focus on whether the agent is modifying the correct logical parts of the code, rather than strictly matching the initially mentioned line numbers. What matters is that the right section of code is being modified, even if its current line number differs from what was originally specified.

At this stage, the agent is still working on the solution. Your task is to assess whether the change done by the **last executed action** is appropriate for addressing the problem and whether the agent is on the right path to resolving the issue. Verify that the correct sections of code are being modified, regardless of their current line numbers.
"""

    def _format_evaluation_criteria(
        self,
        criteria_list: List[str]
    ) -> str:
        formatted_criteria = "\n# Evaluation Criteria:\n"
        for criterion in criteria_list:
            formatted_criteria += f"* {criterion}\n"
        return formatted_criteria

    def _format_reward_scale(
        self,
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
        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump(**kwargs)
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionEvaluator":
        if isinstance(obj, dict):
            obj = obj.copy()
            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj.pop("completion_model"))
            
            return super().model_validate(obj)

        return obj
