import logging
from json import JSONDecodeError
from typing import List, Dict, Optional, Any
import json
import os
from datetime import datetime

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field, BaseModel

from moatless.actions.action import Action
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Completion, Usage, StructuredOutput
from moatless.feedback import FeedbackGenerator
from moatless.node import Node, generate_ascii_tree, FeedbackData
from moatless.schema import MessageHistoryType
from moatless.message_history import MessageHistoryGenerator
from moatless.utils.parse import parse_node_id

logger = logging.getLogger(__name__)


class FeedbackResponse(StructuredOutput):
    """Schema for feedback response"""
    name: str = "provide_feedback"
    
    analysis: str = Field(
        ..., description="Brief analysis of parent state and lessons from alternative attempts"
    )
    feedback: str = Field(..., description="Clear, actionable guidance for your next action")
    suggested_node_id: Optional[int] = Field(
        None, description="ID of the node that should be expanded next (optional)"
    )

    @classmethod
    def anthropic_schema(cls) -> Dict[str, Any]:
        """Provide schema in format expected by Anthropic's tool calling"""
        return {
            "type": "custom",
            "name": "provide_feedback",
            "description": "Provide feedback on the current state",
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "Brief analysis of parent state and lessons from alternative attempts"
                    },
                    "feedback": {
                        "type": "string", 
                        "description": "Clear, actionable guidance for your next action"
                    },
                    "suggested_node_id": {
                        "type": ["integer", "null"],
                        "description": "ID of the node that should be expanded next (optional)"
                    }
                },
                "required": ["analysis", "feedback"]
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message objects to dictionaries"""
        return {
            "role": self.role if hasattr(self, "role") else "assistant",
            "content": self.content if hasattr(self, "content") else str(self)
        }

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
    ) -> FeedbackData | None:
        if not node.parent:
            logger.info(
                f"Node {node.node_id} has no parent node, skipping feedback generation"
            )
            return None


        messages = self._create_analysis_messages(
            node, 
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

            # Store the completion in the node
            node.completions["feedback"] = completion_response.completion

            logger.debug(f"Raw completion content: {completion_response.completion}")
            feedback_response: FeedbackResponse = completion_response.structured_output

            # If node suggestions are disabled, set to None
            if not self.include_node_suggestion:
                feedback_response.suggested_node_id = None

            feedback_message = (
                "System Analysis: I've analyzed your previous actions and alternative attempts. "
                "Here's strategic guidance for your next steps:\n\n"
                f"Feedback: {feedback_response.feedback}\n\n"
                "Note: This feedback is based on the outcomes of various solution attempts. "
                "While alternative attempts mentioned are from separate branches and "
                "have not affected your current state, you should carefully consider their "
                "outcomes to inform your decisions. Learn from both successful and failed "
                "approaches to craft an improved solution that avoids known pitfalls and "
                "combines effective strategies."
            )

            # Save feedback to file if requested
            if self.persist_path:
                self.save_feedback(
                    node=node,
                    feedback=FeedbackResponse(
                        analysis=feedback_response.analysis,
                        feedback=feedback_response.feedback,
                        suggested_node_id=feedback_response.suggested_node_id
                    ),
                    system_prompt=system_prompt,
                    messages=messages,
                    raw_completion=completion_response.completion
                )

            return FeedbackData(
                analysis=feedback_response.analysis,
                feedback=feedback_message,
                suggested_node_id=feedback_response.suggested_node_id
            )

        except Exception as e:
            logger.exception(f"Error while generating feedback: {e}")
            return None

    def _create_analysis_messages(
        self, 
        current_node: Node
    ) -> List[ChatCompletionUserMessage]:
        messages = []

        # Only get siblings that have been run (have actions set)
        sibling_nodes = [s for s in current_node.get_sibling_nodes() if s.action is not None]

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
        first_message += root_node.message
        messages.append(ChatCompletionUserMessage(role="user", content=first_message))

        # Format message history section
        message_generator = MessageHistoryGenerator(
            message_history_type=MessageHistoryType.SUMMARY,
            include_file_context=True,
            include_git_patch=True,
            include_root_node=False
        )
        history_messages = message_generator.generate(current_node)

        if history_messages:
            history_messages[0]["content"] = "Below is the history of previously executed actions and their observations before the current node.\n\n" + history_messages[0]["content"]
            messages.extend(history_messages)

        # Format alternative attempts section
        if sibling_nodes:
            analysis_message = "# Alternative Solution Attempts\n"
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

            messages.append(ChatCompletionUserMessage(role="user", content=analysis_message))

        return messages

    def _create_system_prompt(
        self, 
        actions: List[Action],
    ) -> str:
        start_num = 1
        base_prompt = """You are a feedback agent that guides an AI assistant's next action.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: ACTION AGENT LIMITATIONS  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The action agent receiving your feedback:
• CANNOT see the search tree
• Has NO CONTEXT about node relationships
• Only knows about actions in its direct trajectory
• Cannot understand references to nodes without proper context
• Is at a new node that has NO ACTION YET - it needs your guidance for what to do next

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋  REQUIRED FEEDBACK STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CURRENT NODE CONTEXT
   You must start by describing:
   • Position in tree: "You are at Node X, which is [position relative to root]"
   • Current state: "Your node is currently empty and awaiting your first action"
   • Parent context: "Your parent node (Node Y) [describe what parent did]"
   • Relationship to solutions: "There are [N] terminal nodes in [relationship] branches"

Note: The current node is ALWAYS empty and awaiting its first action - never describe 
it as having done something already.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  CORRECT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Node Context:
"You are at Node 8, which is your first action from the root. Your node is currently 
empty and awaiting your first action. Your parent (Node 1) performed a FindCodeSnippet 
action that didn't add new context. There are three terminal nodes in parallel branches 
(Nodes 7, 9, and 14) that have reached finished states with different approaches."

❌  INCORRECT EXAMPLES - DO NOT USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "Node 8 is empty and expandable"
• "The current node needs to explore improvements"
• "We should validate the existing solution"
• Any description implying the current node has already taken an action
"""
        # Add Input Structure section
        base_prompt += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥  INPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        if self.include_tree:
            base_prompt += """
1. Tree Visualization: ASCII representation showing:
   • Node IDs and relationships
   • Action types at each node
   • Rewards and visit counts
   • Key outcomes and observations"""

        base_prompt += f"""
{start_num}. Original Task: The problem to solve

{start_num + 1}. Message History: Chain of executed actions leading to current state

{start_num + 2}. Tree Structure:
   • Parent Node: Your current starting point - the last successfully executed action
   • Current Node: Your branch from the parent, waiting for your next action
   • Sibling Nodes: Other independent solution attempts branching from the same parent
     (These are from different trajectories and have not happened in your current path)

{start_num + 3}. Alternative Node Information:
   • Their proposed actions and parameters
   • Their outcomes (from separate, independent trajectories)
   • Warning flags for previously attempted approaches"""

        base_prompt += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋  YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━
1. Analyze the situation:
   • Start with current node context (position, state, parent, solutions)
   • Consider sibling attempts (remember these are alternative universes)
   • Learn from outcomes to avoid repeating unsuccessful approaches
   • Contextualize feedback based on tree structure
   • Always explain node relationships and attempts
   • Inform about alternative approaches tried (files, tests, git diffs)

2. Suggest next action:
   • Clear, actionable guidance
   • Based on lessons from other attempts
   • Avoid repeating failed approaches

3. Optionally suggest node to expand:
   • Must explain why this node is promising
   • Leave as null if no strong preference

Remember: Focus on helping the agent achieve novel solutions and, while learning from 
parallel attempts and avoiding re-implementing them. Always provide proper context since the agent cannot see the tree.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: EXPLORATION MANDATE  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your primary goal is to encourage novel solution exploration:
• ALWAYS suggest new approaches, even when existing solutions work well
• Never discourage further exploration just because current solutions are "good enough"
• Actively promote trying completely different strategies and implementations
• Encourage creative combinations of existing approaches into novel solutions
• Look for opportunities to work on entirely new files or components
• Consider radical departures from conventional approaches
• Never assume we have a "working" solution

Examples of encouraging exploration:
1. Fresh Starts
   "While the Redis caching works, let's try a completely new approach:
   • Create a new distributed/sharding.py module
   • Implement a custom consistent hashing algorithm
   • Design a new interface for shard management"

2. Solution Evolution
   "The async query optimization shows promise. Let's expand it:
   • Combine the connection pooling from Node 7 with the caching from Node 12
   • Extend the compiler.py changes to also optimize joins.py and aggregates.py
   • Transform the basic LRU cache into a predictive pre-fetching system"

3. Comprehensive Testing
   • Add concurrent access tests with high load patterns
   • Create chaos testing suite for network partitions
   • Implement performance benchmarks comparing all approaches"

✓ GOOD EXAMPLE:
While Node 3's implementation of parse_http_date() works, let's explore:
• Moving the year calculation logic to a new utils/date_parsing.py module
• Adding property-based testing using hypothesis to generate edge cases
• Implementing a date parsing cache with TTL for repeated requests

✗ BAD EXAMPLE:
We could try:
• Adding more tests
• Optimizing performance
• Improving documentation
• The current solution is complete and correct, making further exploration unnecessary
• Both implementations work well, so we should stop here
• No need to explore further since all tests are passing
"""
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

        for i, msg in enumerate(messages, 1):
            feedback_entry.extend([
                f"[Message {i} - {msg['role']}]",
                msg['content'],
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
