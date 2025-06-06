import logging
from typing import Optional

from pydantic import Field

from moatless.node import Node, Reward
from moatless.value_function.base import BaseValueFunction

logger = logging.getLogger(__name__)


class RepeatedActionValueFunction(BaseValueFunction):
    """
    Value function that penalizes repeated actions found in parent nodes.
    
    This value function traverses up the parent chain and checks if any parent node
    has the same ActionArguments as the current node. If repeated actions are found,
    it applies a configurable penalty.
    """
    
    starting_points: int = Field(
        default=50,
        description="Starting points (Y) before applying penalties for repeated actions",
        ge=-100,
        le=100
    )
    
    penalty_per_repeat: int = Field(
        default=25,
        description="Points (X) to subtract for each repeated action found",
        ge=1,
        le=100
    )
    
    async def get_reward(self, node: Node) -> Optional[Reward]:
        """
        Check for repeated actions in the parent chain and apply penalties.
        
        Args:
            node: The node to evaluate
            
        Returns:
            Reward with penalty applied if repeated actions are found, None otherwise
        """
        if not node.action_steps:
            return None
            
        # Get the latest action step from the current node
        current_action_step = node.action_steps[-1]
        current_action = current_action_step.action
        
        repeated_actions = []
        current_node = node.parent
        
        # Traverse up the parent chain to look for repeated actions
        while current_node is not None:
            if current_node.action_steps:
                for action_step in current_node.action_steps:
                    parent_action = action_step.action
                    
                    # Use the equals method from ActionArguments to compare actions
                    # This excludes the "thoughts" field from comparison
                    if current_action.equals(parent_action):
                        repeated_actions.append({
                            "node_id": current_node.node_id,
                            "action_name": parent_action.name,
                            "depth": node.get_depth() - current_node.get_depth()
                        })
            
            current_node = current_node.parent
        
        # If repeated actions were found, apply penalty
        if repeated_actions:
            penalty = len(repeated_actions) * self.penalty_per_repeat
            reward_value = max(-100, self.starting_points - penalty)
            
            # Create detailed explanation
            action_descriptions = []
            for repeat_info in repeated_actions:
                action_descriptions.append(
                    f"Node {repeat_info['node_id']} (depth -{repeat_info['depth']}): {repeat_info['action_name']}"
                )
            
            explanation = (
                f"Repeated action '{current_action.name}' detected. "
                f"Found {len(repeated_actions)} identical action(s) in parent nodes: "
                f"{', '.join(action_descriptions)}. "
                f"Applied penalty: -{penalty} points from {self.starting_points} starting points."
            )
            
            logger.info(f"Node {node.node_id}: {explanation}")
            
            return Reward(
                value=reward_value,
                explanation=explanation,
                tags=["repeated_action", "penalty"]
            )
        
        # No repeated actions found
        return None