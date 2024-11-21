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

    def expand(self, node: Node) -> None | Node:
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

        # Get single agent setting for this expansion
        settings_to_use = self._get_agent_settings(node)
        
        child_node = Node(
            node_id=self._generate_unique_id(node),
            parent=node,
            file_context=node.file_context.clone() if node.file_context else None,
            max_expansions=self.max_expansions,
            agent_settings=settings_to_use[0] if settings_to_use else None,
        )
        node.add_child(child_node)
        return child_node

    def _get_agent_settings(self, node: Node) -> List[AgentSettings]:
        """
        Get agent settings for a single expansion.
        Returns a list with one item (or empty list if no settings available).
        """
        if not self.agent_settings:
            return []
        
        if self.random_settings:
            # Get settings already used by siblings
            used_settings = {
                child.agent_settings for child in node.children 
                if child.agent_settings is not None
            }
            
            # Try to find unused settings first
            available_settings = [
                setting for setting in self.agent_settings 
                if setting not in used_settings
            ]
            
            # If all settings have been used, use any setting
            settings_pool = available_settings or self.agent_settings
            return [random.choice(settings_pool)]
        else:
            # Original cyclic selection
            num_children = len(node.children)
            return [self.agent_settings[num_children % len(self.agent_settings)]]

    def _generate_unique_id(self, node: Node):
        return len(node.get_root().get_all_nodes())



