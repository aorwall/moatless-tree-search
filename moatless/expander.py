from typing import List
import logging
import random
from pydantic import BaseModel, Field

from moatless.agent.settings import AgentSettings
from moatless.node import Node


logger = logging.getLogger(__name__)

class Expander(BaseModel):

    random_settings: bool = Field(False, description="Whether to select agent settings randomly")
    max_expansions: int = Field(1, description="The maximum number of children to create for each node")

    agent_settings: List[AgentSettings] = Field(
        [],
        description="The settings for the agent model",
    )

    def expand(self, node: Node, search_tree) -> None | Node:
        """Handle all node expansion logic in one place"""
        if node.is_fully_expanded():
            return None

        # Find first unexecuted child if it exists
        for child in node.children:
            if not child.observation:
                logger.info(f"Found unexecuted child {child.node_id} for node {node.node_id}")
                return child

        num_expansions = node.max_expansions or self.max_expansions
        if len(node.children) >= num_expansions:
            logger.info(f"Max expansions reached for node {node.node_id}")
            return None

        # Get agent settings for this expansion
        settings_to_use = self._get_agent_settings(node)
        
        # Create the new node
        child_node = Node(
            node_id=search_tree._generate_unique_id(),
            parent=node,
            file_context=node.file_context.clone() if node.file_context else None,
            max_expansions=self.max_expansions,
            agent_settings=settings_to_use[0] if settings_to_use else None,
        )
        
        # Add child to parent
        node.add_child(child_node)

        # Handle feedback if available
        if search_tree.feedback_generator:
            child_node.message = search_tree.feedback_generator.generate_feedback(
                child_node, 
                search_tree.agent.actions,
                include_parent_info=True,
                persist_path=search_tree.persist_path,
                include_tree=True,
                include_node_suggestion=True
            )
            child_node.feedback = child_node.message
            logger.info(f"Generated feedback for Node{child_node.node_id}")

        logger.info(f"Expanded Node{node.node_id} to new Node{child_node.node_id}")
        return child_node

    def _get_agent_settings(self, node: Node) -> List[AgentSettings]:
        """Get agent settings for a single expansion."""
        if not self.agent_settings:
            return []
        
        if self.random_settings:
            used_settings = {
                child.agent_settings for child in node.children 
                if child.agent_settings is not None
            }
            
            available_settings = [
                setting for setting in self.agent_settings 
                if setting not in used_settings
            ]
            
            settings_pool = available_settings or self.agent_settings
            return [random.choice(settings_pool)]
        else:
            num_children = len(node.children)
            return [self.agent_settings[num_children % len(self.agent_settings)]]

    def _generate_unique_id(self, node: Node):
        return len(node.get_root().get_all_nodes())



