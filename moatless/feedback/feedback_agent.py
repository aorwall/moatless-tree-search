import logging
from typing import List, Dict, Optional
import json
import os
from datetime import datetime

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field, BaseModel

from moatless.actions.action import Action
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Completion, Usage, StructuredOutput
from moatless.feedback import FeedbackGenerator
from moatless.node import Node, generate_ascii_tree
from moatless.schema import MessageHistoryType
from moatless.message_history import MessageHistoryGenerator
from moatless.utils.parse import parse_node_id

logger = logging.getLogger(__name__)


class FeedbackResponse(StructuredOutput):
    analysis: str = Field(
        ..., description="Analysis of the task and alternative branch attempts"
    )
    feedback: str = Field(..., description="Direct feedback to the AI assistant")
    suggested_node_id: Optional[int] = Field(
        None, description="ID of the node that should be expanded next (optional)"
    )
    raw_messages: List[Dict] = Field(default_factory=list, description="Raw messages used to generate feedback")
    system_prompt: str = Field(None, description="System prompt used to generate feedback")
    raw_completion: str = Field(None, description="Raw completion response")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When feedback was generated")


class FeedbackAgent(FeedbackGenerator):
    completion_model: CompletionModel = Field(..., description="The completion model to use")
    instance_dir: str | None = Field(None, description="Base directory for the instance")
    feedback_file: str | None = Field(None, description="Path to the feedback file")

    include_parent_info: bool = Field(True)
    persist_path: str | None = Field(None)
    include_tree: bool = Field(True)
    include_node_suggestion: bool = Field(True)

    def model_post_init(self, __context) -> None:
        """Initialize feedback file after model initialization"""
        super().model_post_init(__context)
        
        # Set instance directory if not provided
        if not self.instance_dir:
            self.instance_dir = os.getcwd()
        
        # Set feedback file path
        if not self.feedback_file:
            # Create instance directory if it doesn't exist
            os.makedirs(self.instance_dir, exist_ok=True)
            self.feedback_file = os.path.join(self.instance_dir, "feedback.txt")

    class Config:
        arbitrary_types_allowed = True

    def generate_feedback(
        self, 
        node: Node, 
        actions: List[Action] | None = None
    ) -> str | None:
        if not node.parent:
            logger.info(
                f"Node {node.node_id} has no parent node, skipping feedback generation"
            )
            return None

        # Only get siblings that have been run (have actions set)
        sibling_nodes = [s for s in node.get_sibling_nodes() if s.action is not None]
        
        # if not sibling_nodes:
        #     logger.info(
        #         f"Node {node.node_id} has no executed sibling nodes, skipping feedback generation"
        #     )
        #     return None

        messages = self._create_analysis_messages(
            node, 
            sibling_nodes,
        )
        system_prompt = self._create_system_prompt(
            actions
        )

        try:
            completion_response = self.completion_model.create_completion(
                messages=messages,
                system_prompt=system_prompt,
                response_model=FeedbackResponse
            )

            logger.debug(f"Raw completion content: {completion_response.completion}")
            feedback_response: FeedbackResponse = completion_response.structured_output

            # FIXME: This shouldn't really be needed?
            try:
                completion_content = completion_response.content
                import json
                import re
                
                json_match = re.search(r'\{.*\}', completion_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    feedback_data = json.loads(json_str)
                    
                    # Only try to parse node_id if it's requested
                    if self.include_node_suggestion:
                        if not feedback_response.suggested_node_id:
                            suggested_node = parse_node_id(completion_content)
                            if suggested_node is not None:
                                feedback_response.suggested_node_id = suggested_node
                        else:
                            # Remove suggested_node_id if present but not requested
                            feedback_response.suggested_node_id = None

                    # Store feedback in node's completions
                    if not hasattr(node, "completions"):
                        node.completions = {}
                    node.completions["feedback"] = completion_response.completion
                    
                    # Save the feedback to file if requested
                    if self.persist_path:
                        self.save_feedback(
                            node=node,
                            feedback=feedback_response,
                            system_prompt=system_prompt,
                            messages=messages,
                            raw_completion=completion_content
                        )

                    feedback_message = (
                        "System Analysis: I've analyzed your previous actions and alternative attempts. "
                        "Here's strategic guidance for your next steps:\n\n"
                        f"Feedback: {feedback_response.analysis}\n\n"
                        "Note: This feedback is based on the outcomes of various solution attempts. "
                        "While alternative attempts mentioned are from separate branches and "
                        "have not affected your current state, you should carefully consider their "
                        "outcomes to inform your decisions. Learn from both successful and failed "
                        "approaches to craft an improved solution that avoids known pitfalls and "
                        "combines effective strategies."
                    )

                    return feedback_message
                    
                else:
                    logger.error("No JSON structure found in completion response")
                    return None
                    
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {je}")
                return None
            except Exception as e:
                logger.error(f"Error parsing feedback response: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None

        except Exception as e:
            logger.error(f"Error while generating feedback: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _create_analysis_messages(
        self, 
        current_node: Node, 
        sibling_nodes: List[Node],
    ) -> List[ChatCompletionUserMessage]:
        messages = []

        # Format tree visualization section
        if self.include_tree:
            tree_message = "# Search Tree Visualization\n"
            tree_message += "<search_tree>\n"
            tree_message += generate_ascii_tree(
                current_node.get_root(),
                current=current_node,
                include_explanation=True,
                use_color=False,
                include_diffs=True,
                include_action_details=False,
                include_file_context=False,
                show_trajectory=True
            )
            tree_message += "\n</search_tree>\n\n"
            messages.append(ChatCompletionUserMessage(role="user", content=tree_message))

        # Format node relationships section
        relationship_message = "# Node Relationships\n"
        relationship_message += "<relationships>\n"
        relationship_message += f"Current Node: {current_node.node_id}\n"
        relationship_message += f"Parent Node: {current_node.parent.node_id if current_node.parent else 'None'}\n"
        relationship_message += f"Sibling Nodes: {[n.node_id for n in current_node.get_sibling_nodes()]}\n"
        relationship_message += f"Child Nodes: {[n.node_id for n in current_node.children]}\n"
        relationship_message += "</relationships>\n\n"
        messages.append(ChatCompletionUserMessage(role="user", content=relationship_message))

        # Format root task section
        root_node = current_node.get_root()
        first_message = "# Original Task\n"
        first_message += "<task>\n"
        first_message += f"Root Node {root_node.node_id}:\n{root_node.message}\n"
        first_message += "</task>\n\n"
        messages.append(ChatCompletionUserMessage(role="user", content=first_message))

        # Format message history section
        message_generator = MessageHistoryGenerator(
            message_history_type=MessageHistoryType.SUMMARY,
            include_file_context=True,
            include_git_patch=True
        )
        history_messages = message_generator.generate(current_node)
        
        # Add node IDs to the message history
        trajectory = current_node.get_trajectory()
        for i, msg in enumerate(history_messages):
            if i < len(trajectory):
                node = trajectory[i]
                parent_id = node.parent.node_id if node.parent else 'None'
                msg.content = f"# Node {node.node_id} (Parent: {parent_id})\n<history>\n{msg.content}\n</history>\n"
        
        messages.extend(history_messages)

        # Format alternative attempts section
        analysis_message = "# alternative Solution Attempts\n"
        has_finish_attempt = False

        for sibling in sibling_nodes:
            if not sibling.action:
                continue

            if sibling.action.name == "Finish":
                has_finish_attempt = True

            analysis_message += f"<attempt_{sibling.node_id}>\n"
            analysis_message += f"Node {sibling.node_id} (Parent: {sibling.parent.node_id if sibling.parent else 'None'})\n"
            analysis_message += f"Action: {sibling.action.name}\n"
            analysis_message += sibling.action.to_prompt()

            if sibling.observation:
                analysis_message += "\nObservation:\n"
                analysis_message += sibling.observation.message
            
            analysis_message += f"\n</attempt_{sibling.node_id}>\n\n"

        if has_finish_attempt:
            analysis_message += "<warning>\n"
            analysis_message += "FINISH ACTION HAS ALREADY BEEN ATTEMPTED!\n"
            analysis_message += "- Trying to finish again would be ineffective\n"
            analysis_message += "- Focus on exploring alternative solutions instead\n"
            analysis_message += "</warning>\n"

        if analysis_message != "# alternative Solution Attempts\n":
            messages.append(ChatCompletionUserMessage(role="user", content=analysis_message))

        return messages

    def _create_system_prompt(
        self, 
        actions: List[Action],
    ) -> str:
        # Store the JSON format based on whether node suggestion is included
        json_format = '''
{
    "analysis": "Brief analysis of parent state and lessons from alternative attempts",
    "feedback": "Clear, actionable guidance for your next action"''' + ('''
    "suggested_node_id": null  // Optional: ID of the node that should be expanded next''' if self.include_node_suggestion else '') + '''
}'''

        # Calculate the starting number based on whether tree is included
        start_num = 2 if self.include_tree else 1

        # Build the prompt using f-strings instead of .format()
        base_prompt = f"""You are a feedback agent that guides an AI assistant's next action.

**Important Note About Line Numbers:**
While line numbers may be referenced in the initial problem description, they can shift as changes are made to the file. 
Focus on whether the agent is modifying the correct logical parts of the code, rather than strictly matching the initially 
mentioned line numbers. What matters is that the right section of code is being modified, even if its current line numberslightly differs from what was originally specified.

**Input Structure:**"""

        if self.include_tree:
            base_prompt += """
1. Tree Visualization: ASCII representation of the entire seaxrch tree showing:
   - Node IDs and their relationships
   - Action types taken at each node
   - Rewards and visit counts
   - Key outcomes and observations
   Use this tree to generate feedback which takes into account the entire search space, mistakes, and outcomes.
   Encourage the AI assistant to explore alternative solutions, *or combine existing solutions* (look at the git patches) and learn from previous attempts.
   Reaching the same finished state multiple times is a waste of resources."""

        base_prompt += f"""
{start_num}. Original Task: The problem to solve

{start_num + 1}. Message History: Chain of executed actions leading to current state

{start_num + 2}. Tree Structure:
   - Parent Node: Your current starting point - the last successfully executed action
   - Current Node: Your branch from the parent, waiting for your next action
   - alternative Sibling Nodes: Other independent solution attempts branching from the same parent
     (These are from different trajectories and have not happened in your current path)

{start_num + 3}. alternative Node Information: Details about other solution attempts, including:
   - Their proposed actions and parameters
   - Their outcomes (from separate, independent trajectories)
   - Warning flags for previously attempted approaches
   
**Your Task:**
1. Analyze the situation:
   - Start from your parent node's current state
   - Consider what other sibling nodes have tried (but remember these are alternative universes, not your history)
   - Learn from their outcomes to avoid repeating unsuccessful approaches
   - Do not assume that a "finish" state means the issue was resolved succesfully, even if the feedback assumes so, since we can never actually be completely certain.
   - Contextualize the feedback based on the entire tree structure and outcomes
   - Constantly inform the agent about alternative approaches that have been tried (files viewed, tests, git diffs) in order to avoid repeating the same actions and learn from them to improve in this trajectory.
   - When referring to previous attempts in alternative trajectories, provide information about what was tried as the agent does not have access to any alternative node information
   - Nodes in the Current Trajectory are marked with a *.

2. Suggest the next action for your branch
3. Optionally suggest which node to expand next by setting suggested_node_id in your response
   - This can help guide the search towards promising branches
   - Leave as null if you have no strong preference

**Required JSON Response Format:**""" + json_format + """

Note: Focus on encouraging the agent to achieve new, novel solutions and avoid approaches that were tried in other branches. 
**Always clearly articulate which of the Nodes/Actions you refer to are within the current node's trajectory (current trajectory), and which are not (alternative), and therefore have no effect on the current node's state.**"""

        # Add available actions if provided
        if actions:
            base_prompt += "\n\n# Available Actions:\n"
            for action in actions:
                try:
                    schema = action.args_schema.model_json_schema()
                    base_prompt += f"\n## {schema['title']}\n{schema['description']}"
                except Exception as e:
                    logger.error(f"Error while building prompt for action {action}: {e}")

        # Only add node suggestion task if requested
        tasks = """1. Analyze the situation:
   - Start from your parent node's current state
   - Consider what other sibling nodes have tried (but remember these are alternative universes, not your history)
   - Learn from their outcomes to avoid repeating unsuccessful approaches
   - Contextualize the feedback based on the entire tree structure and outcomes
2. Suggest the next action for your branch"""

        if self.include_node_suggestion:
            tasks += """
3. Having now reached a finished state, suggest which node to expand next by setting suggested_node_id in your response
   - This can help guide the search towards promising branches, and eventually reach improved solutions."""

        base_prompt += f"\n**Your Task:**\n{tasks}\n\n**Required JSON Response Format:**" + json_format

        return base_prompt

    def save_feedback(
        self,
        node: Node,
        feedback: FeedbackResponse,
        system_prompt: str | None = None,
        messages: List | None = None,
        raw_completion: str | None = None
    ) -> None:
        """Save raw prompts and responses to feedback file"""
        # Setup file path
        if self.persist_path:
            save_dir = os.path.dirname(self.persist_path)
            base_name = os.path.splitext(os.path.basename(self.persist_path))[0]
            self.feedback_file = os.path.join(save_dir, f"{base_name}_feedback.txt")
            os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        feedback_entry = [
            "=" * 80,
            f"Node {node.node_id} - {timestamp}",
            "=" * 80,
            "",
            "SYSTEM PROMPT",
            "-" * 80,
            system_prompt if system_prompt else "No system prompt provided",
            "",
            "MESSAGES",
            "-" * 80,
        ]

        if messages:
            for i, msg in enumerate(messages, 1):
                feedback_entry.extend([
                    f"[Message {i} - {msg.role}]",
                    msg.content,
                    "-" * 40,
                    ""
                ])

        feedback_entry.extend([
            "COMPLETION",
            "-" * 80,
            raw_completion if raw_completion else "No raw completion provided",
            "",
            "=" * 80,
            ""  # Final newline
        ])

        # Write to file in append mode
        with open(self.feedback_file, "a") as f:
            f.write("\n".join(feedback_entry))
        
        logger.info(f"Saved prompts and completion for node {node.node_id} to {self.feedback_file}")
