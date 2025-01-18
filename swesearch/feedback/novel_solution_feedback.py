import logging
import traceback
from typing import List, Optional

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import BaseModel, Field, model_validator

from moatless.actions.action import Action
from moatless.completion import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.completion.schema import ResponseSchema
from moatless.exceptions import CompletionError, CompletionRejectError
from swesearch.feedback import FeedbackGenerator
from swesearch.feedback.feedback_agent import FeedbackData
from moatless.node import Node
from swesearch.feedback.solution_message import format_single_solution, format_task_and_context

logger = logging.getLogger(__name__)


class CodeDiscoveryPlan(BaseModel):
    """Plan for discovering relevant code and patterns."""
    target_area: str = Field(
        ..., 
        description="What code areas or functionality to locate (e.g., 'value calculation logic', 'test helper functions')"
    )
    search_strategy: str = Field(
        ..., 
        description="How to find the relevant code (e.g., 'Look for classes implementing X interface', 'Search for utility functions handling Y')"
    )
    rationale: str = Field(
        ..., 
        description="Why this code is important for the solution and what patterns to look for in it"
    )


class ValidationPlan(BaseModel):
    """Plan for validating the solution."""
    scenario: str = Field(
        ..., 
        description="Specific scenario or edge case to verify"
    )
    required_context: str = Field(
        ..., 
        description="What test utilities, helper functions, or patterns to look for to implement this test"
    )
    success_criteria: str = Field(
        ..., 
        description="How to verify this scenario works correctly"
    )


class NovelSolutionPlan(ResponseSchema):
    """Plan for creating a novel solution by discovering and modifying relevant code."""

    solution_reasoning: str = Field(
        ...,
        description="Step-by-step reasoning process for arriving at this solution plan. This should include:\n"
                   "1. Analysis of what worked/didn't work in previous solutions\n"
                   "2. Identification of patterns that could be improved\n"
                   "3. Reasoning about alternative approaches\n"
                   "4. Justification for the chosen strategy\n"
                   "This field is for internal reasoning and won't be shown in the feedback."
    )
    discovery_plan: CodeDiscoveryPlan = Field(
        ...,
        description="Plan for finding the code that needs to be modified or referenced",
    )
    modification_approach: str = Field(
        ...,
        description="High-level approach for modifying the discovered code (e.g., 'Extend the base class with new validation', 'Add error handling around the core logic')"
    )
    potential_challenges: str = Field(
        ...,
        description="What difficulties might arise and what patterns to look for to handle them"
    )
    validation_plan: List[ValidationPlan] = Field(
        ...,
        description="Plan for verifying the solution works correctly",
    )


class NovelSolutionPlanner(FeedbackGenerator):
    """Generates plans for creating novel solutions by analyzing previous attempts."""

    completion_model: BaseCompletionModel = Field(
        ..., description="The completion model to use"
    )

    @model_validator(mode="after")
    def validate_completion_model(self):
        if not isinstance(self.completion_model, BaseCompletionModel):
            raise ValueError("completion_model must be a BaseCompletionModel")
        
        system_prompt = self._create_system_prompt()
        self.completion_model.initialize(system_prompt=system_prompt, response_schema=NovelSolutionPlan)
        return self
    

    def generate_feedback(
        self, node: Node, actions: List[Action] | None = None
    ) -> Optional[FeedbackData]:
        """Generate a plan for implementing a novel solution."""

        messages = self._create_analysis_messages(node)

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages,
            )

            node.completions["feedback"] = completion_response.completion

            if not completion_response.structured_output:
                logger.error("No structured output in completion response")
                return None

            plan = completion_response.structured_output

            plan_message = "Here's a plan for implementing a solution:\n\n"

            plan_message += "**Code Discovery**:\n"
            plan_message += f"- Target Area: {plan.discovery_plan.target_area}\n"
            plan_message += f"- How to Find It: {plan.discovery_plan.search_strategy}\n"
            plan_message += f"- Why Important: {plan.discovery_plan.rationale}\n\n"

            plan_message += f"**Modification Approach**:\n{plan.modification_approach}\n\n"
            
            plan_message += f"**Potential Challenges**:\n{plan.potential_challenges}\n\n"

            plan_message += "**Validation Plan**:\n"
            for i, test in enumerate(plan.validation_plan, 1):
                plan_message += f"\nTest Scenario {i}:\n"
                plan_message += f"- What to Test: {test.scenario}\n"
                plan_message += f"- Required Context: {test.required_context}\n"
                plan_message += f"- Success Criteria: {test.success_criteria}\n"

            return FeedbackData(
                feedback=plan_message,
            )
        except CompletionError as e:
            node.error = traceback.format_exc()
            if e.last_completion:
                logger.error(f"Node{node.node_id}: Generate feedback failed with completion error: {e}")

                node.completions["feedback"] = Completion.from_llm_completion(
                    input_messages=e.messages if hasattr(e, "messages") else [],
                    completion_response=e.last_completion,
                    model=self.completion.model,
                    usage=e.accumulated_usage if hasattr(e, "accumulated_usage") else None,
                )
            else:
                logger.error(f"Node{node.node_id}: Build action failed with error: {e}")

            if isinstance(e, CompletionRejectError):
                return
            else:
                raise e

        except Exception as e:
            logger.exception(f"Error generating novel solution feedback: {e}")
            return None

    def _create_analysis_messages(
        self, current_node: Node
    ) -> List[ChatCompletionUserMessage]:
        messages = []

        # Get root node and all previous solutions
        root_node = current_node.get_root()
        leaf_nodes = [n for n in root_node.get_leaf_nodes() if n != current_node]

        # Start with task and complete context from all solutions
        task_message = format_task_and_context(root_node, show_file_context=True)
        messages.append(ChatCompletionUserMessage(role="user", content=task_message))

        # Add previous solutions if they exist
        if leaf_nodes:
            solution_message = "\nPrevious solution attempts:\n"
            for i, node in enumerate(leaf_nodes, 1):
                solution_message += f"\n# Solution {i} (Node{node.node_id})\n"
                solution_message += format_single_solution(
                    node=node,
                    show_history=True,
                    show_file_context=True
                )
            messages.append(ChatCompletionUserMessage(role="user", content=solution_message))

        return messages

    def _create_system_prompt(self) -> str:
        return """You are a solution planner that creates strategies for implementing novel solutions.
Your role is to analyze previous attempts and create a plan for a fundamentally different approach,
deliberately moving away from patterns seen in past solutions to explore innovative alternatives.

You will receive and analyze:

1. Original Task Specification and Code Context
   - The initial problem or feature request
   - Code context identified as relevant by previous attempts
   Use this to understand what types of code and patterns will be important.

2. Previous Solution Attempts, each containing:
   - Changes made and their outcomes
   - Test results and coverage
   - Reward scores and explanations
   Use this to identify patterns to deliberately avoid and opportunities for innovation.

First, you should reason through a novel solution step by step:
1. Analyze what worked and didn't work in previous solutions
2. Identify recurring patterns across previous solutions that indicate a "conventional" approach
3. Brainstorm radically different approaches that haven't been tried
4. Consider unconventional but promising strategies that could work better
5. Justify why your novel approach could succeed where others haven't
Document this innovative reasoning process in the solution_reasoning field.

Your goal is to break away from incremental improvements and explore fundamentally different approaches:
- Look for unused but relevant parts of the codebase
- Consider alternative architectural patterns
- Think about solving the problem from a completely different angle
- Challenge assumptions made in previous solutions
- Explore more robust or elegant approaches

Then create a plan that guides an agent to:
1. Find different code areas to modify than previous solutions
2. Implement changes in a novel way
3. Test the changes more comprehensively

The plan should focus on:
- What unexplored code could be leveraged
- How to find promising alternative implementation paths
- What patterns from other parts of the codebase could be adapted
- How to modify the code in ways previous solutions haven't attempted
- What new challenges this approach might face
- How to verify it works more robustly

Important aspects to consider:
1. Novel Code Discovery
   - What unexplored functionality could be useful
   - Where to look for alternative approaches
   - What patterns indicate promising new directions
   - Why this different approach could be better

2. Innovative Modification Strategy
   - How to implement changes differently than before
   - What new patterns to introduce
   - How to avoid pitfalls of previous approaches

3. Enhanced Validation Approach
   - What scenarios previous solutions missed
   - What additional test coverage is needed
   - How to verify the solution is more robust

Your plan should be:
- Clearly different from previous attempts
- Specific about new directions to explore
- Explicit about innovative patterns to introduce
- Detailed about comprehensive validation

Structure your response as:
1. Code Discovery Plan:
   - What novel code areas/functionality to leverage
   - How to find unexplored implementation paths
   - Why this different approach could work better

2. Modification Approach:
   - Innovative strategy for changes
   - New patterns to introduce

3. Potential Challenges:
   - What new difficulties might arise
   - How to handle them differently than before

4. Validation Plan (2 scenarios):
   - What edge cases previous solutions missed
   - What new test approaches are needed
   - How to verify more comprehensive coverage

Focus on making the plan novel and innovative, helping the agent
explore fundamentally different approaches to the problem."""
