import logging
from typing import List, Optional

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import BaseModel, Field

from moatless.actions.action import Action
from moatless.completion.completion import CompletionModel
from moatless.completion.model import StructuredOutput
from moatless.feedback import FeedbackGenerator
from moatless.feedback.feedback_agent import FeedbackData
from moatless.node import Node

logger = logging.getLogger(__name__)


class ImplementationGuidance(BaseModel):
    focus: str = Field(..., description="Area that needs attention")
    consideration: str = Field(..., description="What to keep in mind")
    why: str = Field(..., description="Why this aspect is important")


class TestRequirement(BaseModel):
    scenario: str = Field(..., description="What to verify")
    criteria: str = Field(..., description="How to know it works")


class NovelSolutionFeedback(StructuredOutput):
    """Provide novel solution feedback to the coding agent."""

    implementation_guidance: ImplementationGuidance = Field(
        ...,
        description="A focused suggestion highlighting the most important aspect to consider",
    )
    test_requirements: List[TestRequirement] = Field(
        ...,
        description="2 specific scenarios or conditions that demonstrate success",
        min_items=2,
        max_items=2,
    )


class NovelSolutionFeedbackAgent(FeedbackGenerator):
    completion_model: CompletionModel = Field(
        ..., description="The completion model to use"
    )

    def generate_feedback(
        self, node: Node, actions: List[Action] | None = None
    ) -> Optional[FeedbackData]:
        """Generate guidance-focused feedback for solution implementation."""

        messages = self._create_analysis_messages(node)
        system_prompt = self._create_system_prompt()

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages,
                system_prompt=system_prompt,
                response_model=NovelSolutionFeedback,
            )

            node.completions["feedback"] = completion_response.completion

            if not completion_response.structured_output:
                logger.error("No structured output in completion response")
                return None

            feedback_response = completion_response.structured_output

            feedback_message = "Consider this aspect in your implementation:\n\n"

            feedback_message += "**Key Focus**:\n"
            feedback_message += (
                f" - What: {feedback_response.implementation_guidance.focus}\n"
            )
            feedback_message += f" - Consider: {feedback_response.implementation_guidance.consideration}\n"
            feedback_message += (
                f" - Why: {feedback_response.implementation_guidance.why}\n\n"
            )

            feedback_message += "\nValidate your solution against these scenarios:\n\n"
            for i, test in enumerate(feedback_response.test_requirements, 1):
                feedback_message += f"**Test Scenario {i}**:\n"
                feedback_message += f" - Scenario: {test.scenario}\n"
                feedback_message += f" - Success Criteria: {test.criteria}\n\n"

            return FeedbackData(
                analysis=completion_response.text_response,
                feedback=feedback_message,
                suggested_node_id=None,
            )

        except Exception as e:
            logger.exception(f"Error generating novel solution feedback: {e}")
            return None

    def _create_analysis_messages(
        self, current_node: Node
    ) -> List[ChatCompletionUserMessage]:
        messages = []

        root_node = current_node.get_root()
        task_message = "# Original Task\n"
        task_message += root_node.message
        messages.append(ChatCompletionUserMessage(role="user", content=task_message))

        leaf_nodes = root_node.get_leaf_nodes()

        if leaf_nodes:
            attempts_message = "# Previous Solution Attempts\n"
            for i, node in enumerate(leaf_nodes):
                if node.node_id == current_node.node_id:
                    continue

                attempts_message += f"\n## Attempt {i+1}\n"

                if node.is_finished():
                    attempts_message += "\nStatus: Finished"
                else:
                    attempts_message += "\nStatus: Abandoned"

                trajectory = node.get_trajectory()
                latest_feedback = None
                for n in trajectory:
                    if n.feedback_data:
                        latest_feedback = n.feedback_data.feedback
                        break

                if latest_feedback:
                    attempts_message += "\n\n### Previous Feedback:\n"
                    attempts_message += latest_feedback

                if node.file_context and not node.file_context.is_empty():
                    attempts_message += "\n\n### Final Code State:\n"
                    attempts_message += (
                        "Code identified as relevant and modified in this attempt\n"
                    )
                    attempts_message += "<file_context>\n"
                    attempts_message += node.file_context.create_prompt(
                        show_outcommented_code=True,
                        exclude_comments=True,
                        outcomment_code_comment="... code not in context for this attempt",
                    )
                    attempts_message += "\n</file_context>"

                patch = (
                    node.file_context.generate_git_patch() if node.file_context else ""
                )
                if patch.strip():
                    attempts_message += "\n\n### Changes Made:\n"
                    attempts_message += "<git_patch>\n"
                    attempts_message += patch
                    attempts_message += "\n</git_patch>"

                if node.file_context and node.file_context.test_files:
                    attempts_message += "\n\n### Test results:\n"
                    attempts_message += node.file_context.get_test_summary()

                if node.reward:
                    attempts_message += f"\n\n### Reward: {node.reward.value}/100\n"
                    if node.reward.explanation:
                        attempts_message += (
                            f"Reward Explanation: {node.reward.explanation}\n"
                        )

            messages.append(
                ChatCompletionUserMessage(role="user", content=attempts_message)
            )

        return messages

    def _create_system_prompt(self) -> str:
        return """You are a feedback agent that generates solution guidance for an AI assistant. 
Your role is to provide insights that guide toward successful novel solutions without being 
prescriptive about specific implementations.

You will receive:

1. Original Task Specification
   - The initial problem or feature request to be implemented

2. Previous Solution Attempts, each containing:
   - Status: Whether the attempt was Finished or Abandoned
   - Previous Feedback: Any guidance provided in earlier attempts
   - Code Context: Files and changes identified as relevant by the AI
     Note: <file_context> sections show only the code that the AI identified 
     as relevant and modified in that attempt
   - Test Results: Outcomes of any tests run on the changes
   - Reward Score: An automated evaluation score (-100 to 100) from a value function assessing the solution
   - Reward Explanation: Detailed analysis of why the solution received its score

Analyze the provided context with special attention to:
- Corner cases and edge conditions that might be overlooked
- Missing or incomplete test coverage
- Opportunities for simpler, more maintainable solutions
- Consistency with existing codebase patterns and practices
- Code areas that might be relevant but weren't examined in previous attempts
- Patterns in what led to higher or lower reward scores

Important: Before generating new guidance:
1. Analyze patterns in previous solution approaches
2. Review focus areas from previous feedback
3. Identify unexplored aspects of the problem
4. Look for entirely different angles to approach the task

Actively avoid:
- Similar focus areas as previous feedback
- Guidance that would lead to similar implementations
- Testing scenarios that were previously suggested
- Problem decomposition patterns used before

Seek novel perspectives by:
- Considering different aspects of the problem
- Looking at the task from a new angle
- Focusing on unexplored system interactions
- Thinking about alternative ways to validate success

Provide guidance that:
- Highlights potential edge cases and error conditions
- Encourages comprehensive test coverage
- Favors simple, proven approaches over complex solutions
- Maintains alignment with established code patterns
- Suggests exploration of potentially relevant code areas not yet examined

Structure your response with:
1. A focused implementation suggestion highlighting:
   - Focus area to consider
   - Key considerations (emphasizing simplicity and robustness)
   - Why this aspect is important

2. Two validation scenarios, each including:
   - Specific scenario to verify (including edge cases)
   - Clear success criteria

Keep guidance general enough to allow for creative solutions while ensuring 
critical aspects aren't overlooked. Favor simplicity and consistency over clever solutions."""
