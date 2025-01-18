import logging
from typing import List, Optional, Sequence

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import BaseModel, Field

from moatless.completion import BaseCompletionModel
from moatless.completion.schema import ResponseSchema
from moatless.discriminator.base import BaseDiscriminator
from moatless.node import Node
from swesearch.feedback.solution_message import create_solution_message

logger = logging.getLogger(__name__)


class ComparativeSolutionEvaluation(ResponseSchema):
    """Comparative evaluation of multiple solutions."""
    comparative_analysis: str = Field(
        ...,
        description="A detailed analysis comparing the solutions, highlighting their relative strengths and weaknesses, with special attention to test coverage and quality",
    )
    best_solution_id: int = Field(
        ...,
        description="The solution_id of the solution that the AI considers the best overall, based on code quality, completeness, effectiveness, and test coverage",
    )
    best_solution_reasoning: str = Field(
        ...,
        description="Detailed explanation of why this solution was chosen as the best, highlighting its key strengths including test coverage and edge case handling",
    )


class ComparisonDiscriminator(BaseDiscriminator):
    """Discriminator that selects the best solution through comparative analysis."""

    completion_model: BaseCompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )

    show_file_context: bool = Field(
        True, description="Whether to show file context in the prompt"
    )

    show_history: bool = Field(
        False, description="Whether to show action history in the prompt"
    )

    def select(self, nodes: List[Node]) -> Optional[Node]:
        """Select the best solution from a list of nodes using comparative evaluation."""
        if len(nodes) <= 1:
            return nodes[0] if nodes else None

        solution_message = "Compare the following solutions and select the best one:\n"
        solution_message += "Original task:\n"
        solution_message += nodes[0].get_root().user_message
        solution_message += "\n\nSolutions to compare:\n"

        solution_message += create_solution_message(
            nodes=nodes,
            show_history=self.show_history,
            show_file_context=self.show_file_context,
        )

        user_message = ChatCompletionUserMessage(role="user", content=solution_message)
        messages = [user_message]
        system_prompt = self._build_system_prompt()
        self.completion_model.initialize(system_prompt=system_prompt, response_schema=ComparativeSolutionEvaluation)

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages
            )

            if completion_response.structured_output:
                output = completion_response.structured_output
                return self._pick_best_node(nodes, output)
            else:
                logger.error("No structured output found in completion response")
                return None

        except Exception as e:
            logger.error(f"Error selecting best solution: {e}")
            raise

    def _pick_best_node(self, nodes: Sequence[Node], output: ComparativeSolutionEvaluation) -> Node:
        """Pick the best node based on solution ID."""
        for node in nodes:
            if node.node_id == output.best_solution_id:
                return node
                
        # Fallback to first node if ID not found
        logger.warning(f"Could not find node with ID {output.best_solution_id}, using first node")
        return nodes[0]

    def _build_system_prompt(self) -> str:
        prompt = """You are comparing multiple solutions to select the best one. Your role is to:
1. Compare the solutions to identify their relative strengths and weaknesses
2. Select the best solution based on overall quality, completeness, and effectiveness
3. Provide detailed reasoning for your selection

When comparing solutions, focus on:
1. Implementation quality and correctness
2. Test coverage and quality
3. Edge case handling
4. Code maintainability
5. Integration with existing code
6. Potential issues or risks

The user message contains multiple solutions, each with:
- The original task description
- Changes made in the solution"""

        if self.show_history:
            prompt += "\n- History of actions and outputs that led to the solution"

        if self.show_file_context:
            prompt += "\n- The code context the agent was working with"

        prompt += "\n\nIn your comparative analysis, focus on:"
        prompt += "\n1. Key differences in approach between solutions"
        prompt += "\n2. Relative effectiveness in meeting requirements"
        prompt += "\n3. Trade-offs between different implementations"
        prompt += "\n4. Test coverage and quality comparison"
        prompt += "\n5. Unique strengths or innovative aspects of each solution"

        prompt += "\n\nWhen selecting the best solution, consider:"
        prompt += "\n1. Overall code quality and maintainability"
        prompt += "\n2. Completeness of the implementation"
        prompt += "\n3. Effectiveness in solving the original problem"
        prompt += "\n4. Test coverage and quality"
        prompt += "\n5. Edge case handling and robustness"
        prompt += "\n6. Potential for future extension"

        return prompt 