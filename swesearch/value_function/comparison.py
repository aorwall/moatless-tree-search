import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import BaseModel, Field, model_validator

from moatless.actions.action import Action, RewardScaleEntry
from moatless.completion import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.node import Node, Reward
from moatless.completion.schema import ResponseSchema
from moatless.value_function.base import BaseValueFunction
from swesearch.value_function.base import BaseValueFunction
from swesearch.feedback.solution_message import create_solution_message, format_single_solution, format_task_and_context

logger = logging.getLogger(__name__)


class SolutionEvaluation(ResponseSchema):
    """Evaluation for a single solution."""
    value: int = Field(
        ...,
        description="A single integer value based on how well the solution addresses the original requirements",
        ge=-100,
        le=100,
    )
    explanation: str = Field(
        ...,
        description="Detailed analysis of how well this specific solution solves the task, considering functionality, correctness, completeness, and test coverage",
    )


class ComparisonValueFunction(BaseValueFunction):
    """Value function for evaluating a single solution.

    This class evaluates a solution to:
    - Provide a numerical reward (-100 to 100)
    - Give detailed explanation of the solution
    - Analyze test coverage and quality
    """

    completion_model: BaseCompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )

    show_file_context: bool = Field(
        True, description="Whether to show file context in the prompt"
    )

    show_history: bool = Field(
        False, description="Whether to show action history in the prompt"
    )

    @model_validator(mode="after")
    def validate_completion_model(self):
        if not isinstance(self.completion_model, BaseCompletionModel):
            raise ValueError("completion_model must be a BaseCompletionModel")
        
        self.completion_model = self.completion_model.clone()
        system_prompt = self._build_system_prompt()
        self.completion_model.initialize(system_prompt=system_prompt, response_schema=SolutionEvaluation)


    def get_reward(self, node: Node) -> Tuple[Optional[Reward], Optional[Completion]]:
        """Get reward for a single node.
        
        This method evaluates the node in isolation, considering its implementation,
        test coverage, and effectiveness in solving the task.
        """
        logger.info(f"Evaluating node {node.node_id} with completion model {self.completion_model.model}")
        if not node.terminal:
            logger.info(f"Node {node.node_id} is not terminal, skipping reward calculation")
            return None, None

        # Get all leaf nodes to show previous solutions
        root_node = node.get_root()
        leaf_nodes = [n for n in root_node.get_leaf_nodes() if n != node and n.reward is not None]
        

        messages = []

        # Start with task and complete context
        solution_message = format_task_and_context(root_node, self.show_file_context)
        messages.append(ChatCompletionUserMessage(role="user", content=solution_message))
        
        
        # Add previous solutions for context if they exist
        if leaf_nodes:
            previous_solutions_message = "\n\nPrevious solutions for reference. These solutions are not being evaluated, but are provided for context to compare the current solution to. Reward should be relative to these solutions.\n"
            for i, prev_node in enumerate(leaf_nodes, 1):
                previous_solutions_message += f"\n# Solution {i})\n"
                previous_solutions_message += format_single_solution(
                    node=prev_node,
                    show_history=False,
                    show_file_context=False
                )
            messages.append(ChatCompletionUserMessage(role="user", content=previous_solutions_message))

        # Add current solution to evaluate
        solution_message = "\n\nCurrent solution to evaluate:\n"
        solution_message += format_single_solution(
            node=node,
            show_history=self.show_history,
            show_file_context=self.show_file_context,
        )
        messages.append(ChatCompletionUserMessage(role="user", content=solution_message))
        
        try:
            completion_response = self.completion_model.create_completion(
                messages=messages
            )

            if completion_response.structured_output:
                output = completion_response.structured_output
                logger.info(f"Node{node.node_id}: Value function returned reward {output}")
                reward = Reward(
                    value=output.value,
                    explanation=output.explanation
                )
                return reward, completion_response.completion
            else:
                logger.error("No structured output found in completion response")
                return None, None

        except Exception as e:
            logger.error(f"Error getting solution reward: {e}")
            raise

    def _build_system_prompt(self) -> str:
        prompt = """You are evaluating a single solution to a task. Your role is to:
1. Evaluate the solution based on how well it addresses the requirements
2. Analyze test coverage and quality
3. Provide a detailed explanation of the solution's effectiveness

Pay special attention to testing aspects:
- New tests added to cover the changes
- Test quality and comprehensiveness
- Edge cases and error conditions covered
- Integration with existing test suite
- Test readability and maintainability

The user message contains:
- The original task description
- The current solution to evaluate with its changes"""

        if self.show_history:
            prompt += "\n- History of actions and outputs that led to the solution"

        if self.show_file_context:
            prompt += "\n- The code context the agent was working with"

        prompt += "\n\nReward Scale (-100 to 100):"
        prompt += "\n* 90 to 100: Perfect solution with optimal implementation AND comprehensive test coverage. Tests are well-written, cover all edge cases, and integrate seamlessly with existing tests. Code is elegant, maintainable, and follows best practices."
        prompt += "\n* 75 to 89: Strong solution with good implementation AND proper test coverage. All core functionality is tested, though some edge cases might be missed. Code quality is high but has minor room for improvement."
        prompt += "\n* 50 to 74: Solid solution that works correctly but has gaps in test coverage OR implementation could be more elegant. Core functionality works but may lack comprehensive error handling or edge case tests."
        prompt += "\n* 25 to 49: Partial solution with significant gaps in either implementation or testing. Basic functionality works but lacks proper error handling, edge case coverage, or has maintainability issues."
        prompt += "\n* 0 to 24: Minimal viable solution with major deficiencies in both implementation and testing. Core functionality may work but is fragile and poorly tested."
        prompt += "\n* -49 to -1: Problematic solution with serious flaws in implementation AND inadequate testing. May introduce new bugs or technical debt."
        prompt += "\n* -100 to -50: Completely incorrect solution that fails to address requirements AND lacks proper testing. May break existing functionality or introduce critical issues."

        prompt += "\n\nIn your evaluation, focus on:"
        prompt += "\n1. Correctness and completeness of the implementation"
        prompt += "\n2. Code quality and maintainability"
        prompt += "\n3. Test coverage and quality"
        prompt += "\n4. Edge case handling"
        prompt += "\n5. Integration with existing code"
        prompt += "\n6. Potential issues or risks"

        prompt += "\n\nNote: If this solution has a previous reward, your evaluation should be relative to that reward:"
        prompt += "\n- If the solution has improved, give a higher reward"
        prompt += "\n- If it has gotten worse, give a lower reward"
        prompt += "\n- If it's about the same quality, give a similar reward"

        return prompt 