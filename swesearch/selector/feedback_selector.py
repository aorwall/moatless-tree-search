import logging
from typing import List, Optional, Literal

from pydantic import Field

from moatless.completion.model import Completion
from moatless.node import Node

from moatless.selector import BaseSelector

logger = logging.getLogger(__name__)


class FeedbackSelector(BaseSelector):
    """A selector that uses previously generated feedback to make selection decisions."""

    type: Literal["feedback"] = Field("feedback")

    def select(self, nodes: List[Node]) -> Optional[Node]:
        """Select a node based on existing feedback analysis."""
        if not nodes:
            return None

        for node in nodes:
            # Check for existing feedback in node.completions
            if hasattr(node, "completions") and "feedback" in node.completions:
                feedback_completion: Completion = node.completions["feedback"]
                if feedback_completion and feedback_completion.response:
                    try:
                        # Extract feedback response data
                        response_data = feedback_completion.response

                        # If there's a suggested node ID, use it
                        if "suggested_node_id" in response_data:
                            suggested_node_id = response_data["suggested_node_id"]
                            if suggested_node_id == node.node_id:
                                logger.info(
                                    f"Selected explicitly suggested Node{node.node_id}"
                                )
                                return node

                    except Exception as e:
                        logger.warning(
                            f"Error processing feedback for Node{node.node_id}: {e}"
                        )
                        continue

        # Fallback to most recent node if no explicit suggestion
        return max(nodes, key=lambda n: n.node_id)
