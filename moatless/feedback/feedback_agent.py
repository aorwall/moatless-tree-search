import logging
from typing import List

from pydantic import Field

from moatless.actions.action import Action
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Message, UserMessage, StructuredOutput
from moatless.feedback.feedback import FeedbackGenerator
from moatless.node import Node
from moatless.schema import MessageHistoryType

logger = logging.getLogger(__name__)


class FeedbackResponse(StructuredOutput):
    analysis: str = Field(
        ..., description="Analysis of the task and alternative branch attempts"
    )
    feedback: str = Field(..., description="Direct feedback to the AI assistant")


class FeedbackAgent(FeedbackGenerator):
    completion_model: CompletionModel = Field(
        ..., description="Completion model used for generating feedback"
    )

    def generate_feedback(
        self, node: Node, actions: List[Action] | None = None
    ) -> FeedbackResponse | None:
        if not node.parent:
            logger.info(
                f"Node {node.node_id} has no parent node, skipping feedback generation"
            )
            return None

        sibling_nodes = node.get_sibling_nodes()
        if not sibling_nodes:
            logger.info(
                f"Node {node.node_id} has no sibling nodes, skipping feedback generation"
            )
            return None

        messages = self._create_analysis_messages(node, sibling_nodes)
        system_prompt = self._create_system_prompt(actions)

        try:
            feedback_response, completion_response = (
                self.completion_model.create_completion_with_response_model(
                    messages=messages,
                    system_prompt=system_prompt,
                    response_model=FeedbackResponse,
                )
            )
            node.completions["generate_feedback"] = completion_response

            logger.info(
                f"Feedback generated for node {node.node_id}. {feedback_response.model_dump_json(indent=2)}"
            )
            return feedback_response.feedback
        except Exception as e:
            logger.exception(f"Error while generating feedback for node {node.node_id}")
            return None

    def _create_analysis_messages(
        self, current_node: Node, sibling_nodes: List[Node]
    ) -> List[Message]:
        messages = []

        first_message = current_node.get_root().message
        messages.append(UserMessage(content=first_message))

        # Message history showing the current state
        current_state = current_node.generate_message_history(
            message_history_type=MessageHistoryType.SUMMARY
        )
        messages.extend(current_state)

        # Add sibling attempts analysis
        sibling_analysis = "# Hypothethical Attempts\n\n"
        has_finish_attempt = any(
            sibling.action and sibling.action.name == "Finish"
            for sibling in sibling_nodes
        )

        for i, sibling in enumerate(sibling_nodes):
            if not sibling.action:
                continue

            sibling_analysis += f"## Attemt {i+1}\n"
            sibling_analysis += f"**Action**: {sibling.action.name}\n"
            sibling_analysis += sibling.action.to_prompt()

            if sibling.is_duplicate:
                sibling_analysis += (
                    "\n\n**WARNING: DUPLICATE ATTEMPT DETECTED!**\n"
                    "This attempt was identical to a previous one. "
                    "Repeating this exact approach would be ineffective and should be avoided.\n"
                )
                continue

            if sibling.observation:
                sibling_analysis += f"\n\n**Hypothetical observation**:\n{sibling.observation.message}\n\n"

            # if sibling.reward:
            #    sibling_analysis += f"**Reward Value**: {sibling.reward.value}\n"
            #    if sibling.reward.explanation:
            #        sibling_analysis += f"**Analysis**: {sibling.reward.explanation}\n"

            sibling_analysis += "\n---\n\n"

        if has_finish_attempt:
            sibling_analysis += (
                "\n\n**WARNING: FINISH ACTION HAS ALREADY BEEN ATTEMPTED!**\n"
                "- Trying to finish again would be ineffective\n"
                "- Focus on exploring alternative solutions instead\n\n"
            )

        if not current_node.parent.file_context.is_empty():
            sibling_analysis += "The file context the agent have access to\n\n"
            sibling_analysis += current_node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

        full_patch = current_node.parent.file_context.generate_git_patch()
        if full_patch.strip():
            sibling_analysis += "\n\nThe git diff of the already made changes before doing these attemts:\n"
            sibling_analysis += "<git_patch>\n"
            sibling_analysis += full_patch
            sibling_analysis += "\n</git_patch>\n"

        messages.append(UserMessage(content=sibling_analysis))

        return messages

    def _create_system_prompt(self, actions: List[Action]) -> str:
        base_prompt = """Your task is to provide strategic feedback to guide the next execution of an action by another AI assistant.

**Context you will receive:**

 * Task Description: The main problem or objective that needs to be addressed wrapped in a <task> tag.
 * History: The conversation leading up to the current state.
 * Hypothetical Attempts: Considred actions that NOT been executed in the current branch. They are hypothetical and serve as insights.
 * Warnings: Any duplicate attempts that have already been tried.

**Your role is to:**

 * Analyze What to Do Next: Combine your understanding of the task with insights from considred attempts (which are hypothetical and unexecuted) to determine the best next step.
 * Provide Feedback: Offer strategic guidance that focuses on novel and diverse solutions to address the task directly.
 * Avoid Duplicates: Strongly discourage repeating any actions flagged as duplicates.

**Instructions:**

 * Analysis: Begin with a brief analysis that combines understanding the task and insights from considered (hypothetical) attempts, focusing on what should be done next.
 * Direct Feedback: Provide one concrete and innovative suggestion for the next action, specifying which available action to use (using the exact name from Available Actions) and how it addresses the task.
    
Remember: Focus on the task's objectives and encourage a novel solution that hasn't been explored yet. Use previous attempts as learning points but do not let them constrain your creativity in solving the task. The considered attempts are hypothetical and should inform, but not limit, your suggested action.
"""

        if actions:
            base_prompt += "\n\n# Available Actions:\n"
            base_prompt += "The following actions were available for the AI assistant to choose from:\n\n"
            for action in actions:
                try:
                    schema = action.args_schema.model_json_schema()
                    base_prompt += f"\n\n## {schema['title']}\n{schema['description']}"
                except Exception as e:
                    logger.error(
                        f"Error while building prompt for action {action}: {e}"
                    )

        return base_prompt
