import logging
from typing import List, Tuple, Optional

from instructor import OpenAISchema
from pydantic import BaseModel, Field

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments
from moatless.node import Node, MessageHistoryType
from moatless.feedback.feedback import FeedbackGenerator
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Message, UserMessage, Completion

logger = logging.getLogger(__name__)


class FeedbackResponse(OpenAISchema):
    key_insights: List[str] = Field(..., description="Key insights to consider")
    feedback: str = Field(..., description="Direct feedback to the AI assistant")
    

class FeedbackAgent(FeedbackGenerator):
    completion_model: CompletionModel = Field(..., description="Completion model used for generating feedback")


    def generate_feedback(self, node: Node, actions: List[Action] | None = None) -> FeedbackResponse | None:
        if not node.parent:
            logger.info(f"Node {node.node_id} has no parent node, skipping feedback generation")
            return None

        sibling_nodes = node.get_sibling_nodes()
        if not sibling_nodes:
            logger.info(f"Node {node.node_id} has no sibling nodes, skipping feedback generation")
            return None

        messages = self._create_analysis_messages(node, sibling_nodes)
        system_prompt = self._create_system_prompt(actions)

        feedback_response, completion_response = self.completion_model.create_completion_with_response_model(
            messages=messages,
            system_prompt=system_prompt,
            response_model=FeedbackResponse
        )
        node.completions["generate_feedback"] = completion_response
        logger.info(f"Feedback generated for node {node.node_id}. {feedback_response.model_dump_json(indent=2)}")
        return feedback_response.feedback

    def _create_analysis_messages(self, current_node: Node, sibling_nodes: List[Node]) -> List[Message]:
        messages = []

        # Message history showing the current state
        current_state = current_node.generate_message_history(
            message_history_type=MessageHistoryType.SUMMARY,
            include_file_context=False,
            include_extra_history=False,
            include_git_patch=False,
        )
        messages.extend(current_state)

        # Add sibling attempts analysis
        sibling_analysis = "# Previous Parallel Attempts\n\n"
        for sibling in sibling_nodes:
            if not sibling.action:
                continue

            sibling_analysis += f"## Attempt {sibling.node_id}\n"
            sibling_analysis += f"**Action Type**: {sibling.action.name}\n"
            sibling_analysis += f"**Approach**:\n{sibling.action.to_prompt()}\n\n"

            if sibling.is_duplicate:
                sibling_analysis += (
                    "\n\n**WARNING: DUPLICATE ATTEMPT DETECTED!**\n"
                    "This attempt was identical to a previous one. "
                    "Repeating this exact approach would be ineffective and should be avoided.\n"
                )
                continue

            if sibling.observation:
                sibling_analysis += f"**Result**:\n{sibling.observation.message}\n\n"

            if sibling.reward:
                sibling_analysis += f"**Reward Value**: {sibling.reward.value}\n"
                if sibling.reward.explanation:
                    sibling_analysis += f"**Analysis**: {sibling.reward.explanation}\n"
            
            sibling_analysis += "\n---\n\n"

        messages.append(UserMessage(content=sibling_analysis))

        return messages
    
    def _create_system_prompt(self, actions: List[Action]) -> str:
        base_prompt = """Your task is to provide strategic feedback to guide the next execution of an action by another AI assistant. 
You are analyzing parallel attempts—these are alternative scenarios that have **NOT** occurred in the current execution branch but provide valuable insights for guiding the next action.

**Context you will receive:**

- The message history leading to the current state.
- Information about parallel attempts at this same decision point (these have **NOT** been executed in the current branch).
- Warnings about any duplicate attempts that have already been tried.

**Your role is to:**

1. **Analyze** these parallel attempts to extract key learnings that can guide the next execution.
2. **Provide feedback** that specifically guides the selection and execution of the next action.
3. **Focus** on suggesting different actions when duplicates are detected.
4. **Strongly discourage** repeating actions that have been flagged as duplicates.

**Instructions:**

- **Key Insights:** List 2-3 specific technical observations from previous attempts.
- **Direct Feedback:** Provide 2-3 concrete suggestions focusing on which action to use and how. For example:
    - Instead of: "Try a different approach"
    - Write: "Use the CreateFile action to generate a new parser instead of modifying the existing one"
- **When duplicates are detected:** Suggest completely different actions."
"""

        if actions:
            base_prompt += "\n\n# Available Actions:\n"
            base_prompt += "The following actions were available for the AI assistant to choose from:\n\n"
            for action in actions:
                try:
                    schema = action.args_schema.model_json_schema()
                    base_prompt += f"\n\n## {schema['title']}\n{schema['description']}"
                except Exception as e:
                    logger.error(f"Error while building prompt for action {action}: {e}")

        return base_prompt