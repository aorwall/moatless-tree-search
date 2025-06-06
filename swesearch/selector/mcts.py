from abc import ABC
import difflib
import logging
import math
import random
from dataclasses import dataclass
from typing import List, Set, Type, Literal, Dict, Any, Tuple, Optional
import logging
import re

from moatless.file_context import FileContext
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from moatless.node import Node, Selection
from moatless.selector.base import BaseSelector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class UCTScore:
    final_score: float = 0.0
    exploitation: float = 0.0
    exploration: float = 0.0
    depth_bonus: float = 0.0
    depth_penalty: float = 0.0
    high_value_leaf_bonus: float = 0.0
    high_value_bad_children_bonus: float = 0.0
    high_value_child_penalty: float = 0.0
    high_value_parent_bonus: float = 0.0
    finished_trajectory_penalty: float = 0.0
    diversity_bonus: float = 0.0
    duplicate_child_penalty: float = 0.0
    duplicate_action_penalty: float = 0.0
    unfinished_trajectory_bonus: float = 0.0
    leaf_node_bonus: float = 0.0
    root_exploration_bonus: float = 0.0
    
    def __str__(self):
        components = [
            f"Final Score: {self.final_score:.2f}",
            f"Exploitation: {self.exploitation:.2f}",
            f"Exploration: {self.exploration:.2f}",
            f"Depth Bonus: {self.depth_bonus:.2f}",
            f"Depth Penalty: {self.depth_penalty:.2f}",
            f"High Value Leaf Bonus: {self.high_value_leaf_bonus:.2f}",
            f"High Value Bad Children Bonus: {self.high_value_bad_children_bonus:.2f}",
            f"High Value Child Penalty: {self.high_value_child_penalty:.2f}",
            f"High Value Parent Bonus: {self.high_value_parent_bonus:.2f}",
            f"Finished Trajectory Penalty: {self.finished_trajectory_penalty:.2f}",
            f"Diversity Bonus: {self.diversity_bonus:.2f}",
            f"Duplicate Child Penalty: {self.duplicate_child_penalty:.2f}",
            f"Duplicate Action Penalty: {self.duplicate_action_penalty:.2f}",
            f"Unfinished Trajectory Bonus: {self.unfinished_trajectory_bonus:.2f}",
            f"Leaf Node Bonus: {self.leaf_node_bonus:.2f}",
            f"Root Exploration Bonus: {self.root_exploration_bonus:.2f}",
        ]
        return ", ".join(components)


class MCTSSelector(BaseSelector, ABC):
    exploitation_weight: float = Field(
        default=1.0,
        description="Weight factor for the exploitation term in the UCT score calculation. Higher values favor exploitation over exploration.",
    )
    use_average_reward: bool = Field(
        default=True,
        description="If True, uses average reward across the trajectory for exploitation calculation instead of node reward.",
    )
    exploration_weight: float = Field(
        default=0.0,
        description="Weight factor for the exploration term in the UCT score calculation. Higher values encourage more exploration of less-visited nodes.",
    )
    depth_weight: float = Field(
        default=0.0,
        description="Weight factor for the depth-based components in the UCT score. Affects both the depth bonus and penalty calculations.",
    )
    depth_bonus_factor: float = Field(
        default=0.0,
        description="Factor used in calculating the depth bonus. Higher values increase the bonus for exploring deeper nodes, especially near the root.",
    )
    high_value_threshold: float = Field(
        default=50.0,
        description="Threshold for considering a node's reward as 'high value'. Used in various bonus calculations.",
    )
    low_value_threshold: float = Field(
        default=0.0,
        description="Threshold for considering a node's reward as 'low value'. Used in various penalty calculations.",
    )
    very_high_value_threshold: float = Field(
        default=75.0,
        description="Threshold for considering a node's reward as 'very high value'. Used in the high value child penalty calculation.",
    )
    high_value_leaf_bonus_constant: float = Field(
        default=0.0,
        description="Constant bonus applied to high-value leaf nodes to encourage their exploration.",
    )
    high_value_bad_children_bonus_constant: float = Field(
        default=0.0,
        description="Constant used in calculating the bonus for high-value nodes with low-value children, encouraging 'auto-correction'.",
    )
    high_value_child_penalty_constant: float = Field(
        default=0.0,
        description="Constant used in penalizing nodes with very high-value children to prevent over-exploitation of a single path.",
    )
    finished_trajectory_penalty: float = Field(
        default=0.0,
        description="Penalty applied to nodes on a trajectory that has already finished with a high reward, discouraging revisiting completed paths.",
    )
    check_for_bad_child_actions: List[str] = Field(
        default_factory=lambda: ["RequestCodeChange"],
        description="List of action types to check for when calculating the high value bad children bonus.",
    )
    diversity_weight: float = Field(
        default=0.0,
        description="Weight factor for the diversity bonus. Higher values increase the bonus for nodes with low similarity to other explored nodes.",
    )
    duplicate_child_penalty_constant: float = Field(
        default=0.0,
        description="Constant used in penalizing nodes that have duplicate children. Penalty increases with each duplicate.",
    )
    duplicate_action_penalty_constant: float = Field(
        default=0.0,
        description="Constant used in penalizing nodes that have siblings with the same action name.",
    )
    min_finished_states_ratio: float = Field(
        default=0.0,
        description="Minimum ratio of finished states to total expansions desired"
    )
    leaf_node_bonus: float = Field(
        default=0.0,
        description="Bonus applied to unexpanded leaf nodes when we need more finish states"
    )
    root_exploration_bonus: float = Field(
        default=0.0,
        description="Bonus for exploring from root when other paths are finished"
    )
    unfinished_trajectory_bonus: float = Field(
        default=0.0,
        description="Bonus for exploring unfinished trajectories"
    )
    enable_high_value_parent_bonus: bool = Field(
        default=False,
        description="If True, enables the high value parent bonus"
    )
    prioritize_unrewarded_nodes: bool = Field(
        default=False,
        description="If True, prioritizes nodes without reward objects to continue unevaluated trajectories"
    )
    _similarity_cache: Dict[Tuple[int, int], float] = PrivateAttr(default_factory=dict)

    async def select(self, expandable_nodes: List[Node]) -> Selection:
        raise NotImplementedError("Subclasses must implement the select method.")

    def uct_score(self, node: Node) -> UCTScore:
        """
        Compute the UCT score with additional bonuses and penalties based on node characteristics.
        """
        if node.visits == 0:
            return UCTScore(final_score=0.0)

        exploitation = self.calculate_exploitation(node)
        exploration = self.calculate_exploration(node)
        depth_bonus = self.calculate_depth_bonus(node)
        depth_penalty = self.calculate_depth_penalty(node)
        high_value_leaf_bonus = self.calculate_high_value_leaf_bonus(node)
        high_value_bad_children_bonus = self.calculate_high_value_bad_children_bonus(node)
        high_value_child_penalty = self.calculate_high_value_child_penalty(node)
        high_value_parent_bonus = self.calculate_high_value_parent_bonus(node)
        finished_trajectory_penalty = self.calculate_finished_trajectory_penalty(node)
        diversity_bonus = self.calculate_diversity_bonus(node)
        duplicate_child_penalty = self.calculate_duplicate_child_penalty(node)
        duplicate_action_penalty = self.calculate_duplicate_action_penalty(node)
        unfinished_trajectory_bonus = self.calculate_unfinished_trajectory_bonus(node)
        leaf_node_bonus = self.calculate_leaf_node_bonus(node)
        root_exploration_bonus = self.calculate_root_exploration_bonus(node)
        final_score = (
            exploitation
            + exploration
            + depth_bonus
            - depth_penalty
            + high_value_leaf_bonus
            + high_value_bad_children_bonus
            - high_value_child_penalty
            + high_value_parent_bonus
            - finished_trajectory_penalty
            + diversity_bonus
            - duplicate_child_penalty
            - duplicate_action_penalty
            + unfinished_trajectory_bonus
            + leaf_node_bonus
            + root_exploration_bonus
        )

        return UCTScore(
            final_score=final_score,
            exploitation=exploitation,
            exploration=exploration,
            depth_bonus=depth_bonus,
            depth_penalty=depth_penalty,
            high_value_leaf_bonus=high_value_leaf_bonus,
            high_value_bad_children_bonus=high_value_bad_children_bonus,
            high_value_child_penalty=high_value_child_penalty,
            high_value_parent_bonus=high_value_parent_bonus,
            finished_trajectory_penalty=finished_trajectory_penalty,
            diversity_bonus=diversity_bonus,
            duplicate_child_penalty=duplicate_child_penalty,
            duplicate_action_penalty=duplicate_action_penalty,
            unfinished_trajectory_bonus=unfinished_trajectory_bonus,
            leaf_node_bonus=leaf_node_bonus,
            root_exploration_bonus=root_exploration_bonus,
        )

    def calculate_exploitation(self, node: Node) -> float:
        """
        Calculate the exploitation component of the UCT score.

        Purpose: Favors nodes with higher rewards, encouraging the algorithm to exploit
        known good paths in the search tree.
        """
        if self.use_average_reward:
            reward = node.calculate_mean_reward()
        else:
            reward = node.reward.value if node.reward else 0

        return self.exploitation_weight * reward

    def calculate_exploration(self, node: Node) -> float:
        """
        Calculate the exploration component of the UCT score.

        Purpose: Encourages the exploration of less-visited nodes, ensuring a balance
        between exploitation and exploration in the search process.
        """
        total_visits = node.parent.visits if node.parent else 1
        return self.exploration_weight * math.sqrt(math.log(total_visits) / node.visits)

    def calculate_depth_bonus(self, node: Node) -> float:
        """
        Calculate the depth-based exploration bonus.

        Purpose: Provides an incentive to explore deeper into the search tree,
        particularly for nodes near the root, to encourage thorough exploration.
        """
        depth = node.get_depth()
        if depth == 0:
            return self.depth_bonus_factor * np.exp(-self.depth_weight * (depth - 1))
        return 0

    def calculate_depth_penalty(self, node: Node) -> float:
        """
        Calculate the depth penalty for very deep nodes.

        Purpose: Discourages excessive depth in the search tree, preventing the
        algorithm from getting stuck in overly long paths.
        """
        depth = node.get_depth()
        return self.depth_weight * math.sqrt(depth)

    def calculate_high_value_leaf_bonus(self, node: Node) -> float:
        """
        Calculate the bonus for not expanded nodes with high reward.

        Purpose: Encourages the exploration of promising leaf nodes, potentially
        leading to valuable new paths in the search tree.
        """
        if not node.children and node.reward and node.reward.value >= self.high_value_threshold:
            return self.high_value_leaf_bonus_constant
        return 0

    def calculate_high_value_bad_children_bonus(self, node: Node) -> float:
        """
        Calculate the bonus for nodes with high reward that expanded to low-reward nodes.

        Purpose: Acts as an "auto-correct" mechanism for promising nodes that led to poor
        outcomes, likely due to invalid actions (e.g., syntax errors from incorrect code changes).
        This bonus gives these nodes a second chance, allowing the algorithm to potentially
        recover from or find alternatives to invalid actions.

        The bonus is applied when:
        1. The node has a high reward
        2. It has exactly one child (indicating a single action was taken)
        3. The child action is of a type we want to check (e.g., RequestCodeChange)
        4. The child node has a low reward

        In such cases, we encourage revisiting this node to try different actions,
        potentially leading to better outcomes.
        """
        exploitation = self.calculate_exploitation(node)
        if node.children and exploitation >= self.high_value_threshold:
            child_values = [
                child.reward.value for child in node.children if child.reward
            ]
            if len(child_values) == 1 and any(
                [
                    child.action.__class__.__name__ in self.check_for_bad_child_actions
                    for child in node.children
                ]
            ):
                avg_child_value = sum(child_values) / len(child_values)
                if avg_child_value <= self.low_value_threshold:
                    return (exploitation - avg_child_value) * 5
        return 0

    def calculate_high_value_child_penalty(self, node: Node) -> float:
        """
        Calculate the penalty for nodes with a child with very high reward.

        Purpose: Discourages over-exploitation of a single high-value path, promoting
        exploration of alternative routes in the search tree.
        """
        if node.children:
            child_values = [
                child.reward.value for child in node.children if child.reward
            ]
            max_child_value = max(child_values) if child_values else 0
            if max_child_value >= self.very_high_value_threshold:
                return self.high_value_child_penalty_constant * 1
        return 0

    def calculate_high_value_parent_bonus(self, node: Node) -> float:
        """
        Calculate the bonus for nodes with low reward that haven't been expanded yet but have high reward parents or not rewarded parents.

        Purpose: Encourages exploration of nodes that might be undervalued due to their
        current low reward, especially if they have promising ancestors.
        """
        if not self.enable_high_value_parent_bonus:
            return 0
        
        exploitation = self.calculate_exploitation(node)
        if not node.children:
            if node.parent and (
                not node.parent.reward
                or node.parent.reward.value > self.high_value_threshold
            ):
                if exploitation <= self.low_value_threshold:
                    return self.high_value_threshold - exploitation
        return 0
    
    def calculate_leaf_node_bonus(self, node: Node) -> float:
        """Calculate bonus for leaf nodes when finish ratio is low"""
        # Get ratio of finished states
        root = node.get_root()
        all_nodes = root.get_all_nodes()
        total_expansions = len(all_nodes)
        finished_states = len([n for n in all_nodes if n.is_finished()])
        current_ratio = finished_states / total_expansions if total_expansions > 0 else 0

        # If ratio is too low and this is a leaf node, apply bonus
        if node.is_leaf() and (not self.min_finished_states_ratio or current_ratio < self.min_finished_states_ratio):
            logger.debug(f"Adding leaf node bonus to node {node.node_id} "
                      f"(finish ratio: {current_ratio:.2f})")
            return self.leaf_node_bonus
        return 0.0


    def calculate_finished_trajectory_penalty(self, node: Node) -> float:
        """
        Calculate the penalty for nodes where there are changes and a child node was already finished with high reward.

        Purpose: Discourages revisiting paths that have already led to successful outcomes,
        promoting exploration of new areas in the search space.
        """
        if (
            self.finished_trajectory_penalty
            and node.file_context
            and node.file_context.has_patch()
            and self.is_on_finished_trajectory(node, 100)
        ):
            return self.finished_trajectory_penalty
        return 0

    def is_on_finished_trajectory(
        self, node: Node, min_reward_thresh: int = 100
    ) -> bool:
        """
        Check if the current node is on a trajectory that includes a 'Finish' node.
        """

        for child in node.children:
            if (
                child.is_finished()
                and child.reward
                and child.reward.value >= min_reward_thresh
            ):
                return True

            if self.is_on_finished_trajectory(
                child, min_reward_thresh=min_reward_thresh
            ):
                return True

        return False

    def calculate_diversity_bonus(self, node: Node) -> float:
        """
        Calculate the diversity bonus based on the similarity of the node's solution to already expanded nodes.

        Purpose: Boosts the score for nodes whose solutions have low similarity to other explored nodes,
        encouraging the exploration of novel solutions.
        """
        if not self.diversity_weight:
            return 0

        # Ignore nodes without any code added to file context yet
        if node.file_context is None or node.file_context.is_empty():
            return 0

        expandable_nodes = [
            n
            for n in node.get_root().get_expanded_descendants()
            if n.node_id != node.node_id
        ]

        if not expandable_nodes:
            # No other nodes to compare; return maximum bonus
            return self.diversity_weight

        similarities = []
        for other_node in expandable_nodes:
            similarity = self.get_similarity(node, other_node)
            similarities.append(similarity)

        # Compute the average similarity
        average_similarity = sum(similarities) / len(similarities)

        # Diversity bonus is proportional to (1 - average_similarity)
        diversity_bonus = self.diversity_weight * (1 - average_similarity)

        return diversity_bonus

    def calculate_duplicate_child_penalty(self, node: Node) -> float:
        """
        Calculate penalty for nodes that have duplicate children.
        The penalty increases with each duplicate child.

        Purpose: Discourages exploration of nodes that tend to generate duplicate states,
        as these are likely to be less productive paths in the search space.
        """
        duplicate_count = sum(1 for child in node.children if child.is_duplicate)
        if duplicate_count > 0:
            # Penalty increases quadratically with number of duplicates
            return self.duplicate_child_penalty_constant * (duplicate_count**2)
        return 0

    def calculate_duplicate_action_penalty(self, node: Node) -> float:
        """
        Calculate penalty for nodes that have children with duplicate action names.
        The penalty increases with each duplicate action.

        Purpose: Discourages selecting nodes whose children perform the same type of action
        multiple times, promoting more diverse action sequences.
        """
        if not node.children:
            return 0.0

        # Count occurrences of each action name among children
        action_counts = {}
        for child in node.children:
            if child.action:
                action_name = child.action.__class__.__name__
                action_counts[action_name] = action_counts.get(action_name, 0) + 1

        # Sum up penalties for all action types that have duplicates
        total_penalty = 0.0
        for count in action_counts.values():
            if count > 1:  # Only penalize actions that appear more than once
                # Penalty increases quadratically with number of duplicates
                total_penalty += self.duplicate_action_penalty_constant * (
                    (count - 1) ** 2
                )

        return total_penalty

    def get_similarity(self, node_a: Node, node_b: Node) -> float:
        """
        Retrieve the similarity between two nodes from the cache or compute it if not cached.
        """
        if node_a.file_context is None or node_b.file_context is None:
            return 0.0

        node_ids = (
            min(node_a.node_id, node_b.node_id),
            max(node_a.node_id, node_b.node_id),
        )
        if node_ids in self._similarity_cache:
            return self._similarity_cache[node_ids]

        similarity = calculate_similarity(node_a.file_context, node_b.file_context)
        self._similarity_cache[node_ids] = similarity
        return similarity

    def calculate_root_exploration_bonus(self, node: Node) -> float:
        """Add bonus for root node after the last node was finished"""
        
        if node.get_depth() == 0:  # Is root node
            # Check if we have any finished successful trajectories
            finished_paths = any(
                n.is_finished() and n.reward and n.reward.value >= self.high_value_threshold
                for n in node.get_all_nodes()
            )
            if finished_paths:
                return self.root_exploration_bonus
        return 0

    def calculate_unfinished_trajectory_bonus(self, node: Node) -> float:
        """
        Calculate bonus for nodes that are part of unfinished trajectories.
        Bonus scales with depth relative to deepest unfinished node.
        """
        root = node.get_root()
        unfinished_trajectories = []
        
        def collect_unfinished(n: Node) -> None:
            if not n.children and not n.is_finished():
                trajectory = []
                current = n
                while current:
                    trajectory.insert(0, current)
                    current = current.parent
                unfinished_trajectories.append(trajectory)
                return
            
            for child in n.children:
                collect_unfinished(child)
        
        collect_unfinished(root)
        
        # If no unfinished trajectories, return no bonus
        if not unfinished_trajectories:
            return 0.0
        
        # Find deepest unfinished node for normalization
        max_depth = max(t[-1].get_depth() for t in unfinished_trajectories)
        
        # Check if node is part of any unfinished trajectory
        for trajectory in unfinished_trajectories:
            if node in trajectory:
                depth_ratio = node.get_depth() / max_depth if max_depth > 0 else 0
                return self.unfinished_trajectory_bonus * depth_ratio
            
        return 0.0

    def _create_selection_trace(self, node: Node, score: UCTScore, **extra_trace) -> dict:
        """Create a common trace dictionary with UCT score components and node info."""
        trace = {
            "uct_score": score.final_score,
            "exploitation": score.exploitation,
            "exploration": score.exploration,
            "root_exploration_bonus": score.root_exploration_bonus,
            "depth_bonus": score.depth_bonus,
            "depth_penalty": score.depth_penalty,
            "high_value_leaf_bonus": score.high_value_leaf_bonus,
            "high_value_bad_children_bonus": score.high_value_bad_children_bonus,
            "high_value_child_penalty": score.high_value_child_penalty,
            "high_value_parent_bonus": score.high_value_parent_bonus,
            "finished_trajectory_penalty": score.finished_trajectory_penalty,
            "diversity_bonus": score.diversity_bonus,
            "duplicate_child_penalty": score.duplicate_child_penalty,
            "duplicate_action_penalty": score.duplicate_action_penalty,
            "unfinished_trajectory_bonus": score.unfinished_trajectory_bonus,
            "leaf_node_bonus": score.leaf_node_bonus,
            "visits": node.visits,
            "reward": node.reward.value if node.reward else None
        }
        trace.update(extra_trace)
        return trace

    def _get_top_nodes_info(self, nodes_with_scores: List[Tuple[Node, UCTScore]], probabilities: Optional[List[float]] = None) -> List[dict]:
        """Get information about top 10 nodes for tracing."""
        top_nodes = []
        for i, (node, score) in enumerate(sorted(nodes_with_scores, key=lambda x: x[1].final_score, reverse=True)[:10]):
            node_info = {
                "node_id": node.node_id,
                "visits": node.visits,
                "reward": node.reward.value if node.reward else None,
                "uct_score": score.final_score,
                "exploitation": score.exploitation,
                "exploration": score.exploration
            }
            if probabilities is not None:
                node_info["probability"] = probabilities[nodes_with_scores.index((node, score))]
            top_nodes.append(node_info)
        return top_nodes

    def _handle_single_node(self, node: Node) -> Selection:
        """Handle the case when there's only one expandable node."""
        return Selection(
            node_id=node.node_id,
            reason="Only one expandable node available",
            trace=self._create_selection_trace(node, self.uct_score(node))
        )


class BestFirstSelector(MCTSSelector):

    async def select(self, expandable_nodes: List[Node]) -> Selection:
        if len(expandable_nodes) == 1:
            return self._handle_single_node(expandable_nodes[0])

        # If prioritize_unrewarded_nodes is enabled, select first unrewarded node
        if self.prioritize_unrewarded_nodes:
            for node in expandable_nodes:
                if node.reward is None and node.is_leaf():
                    logger.info(f"Selected unrewarded Node {node.node_id} (prioritizing unrewarded nodes)")
                    return Selection(
                        node_id=node.node_id,
                        reason="First unrewarded node (prioritizing unrewarded trajectories)",
                        trace=self._create_selection_trace(node, self.uct_score(node), unrewarded_selection=True)
                    )

        # Calculate UCT scores with components
        nodes_with_scores = [(node, self.uct_score(node)) for node in expandable_nodes]
        sorted_nodes = sorted(
            nodes_with_scores, key=lambda x: x[1].final_score, reverse=True
        )

        # Log top nodes with detailed score breakdowns
       

        # Select the node with the highest UCT score
        selected_node = sorted_nodes[0][0]
        selected_score = sorted_nodes[0][1]

        logger.info(
            f"Selected Node {selected_node.node_id} with UCT Score: {selected_score.final_score:.2f}"
        )

        # Return Selection with detailed trace information
        trace = self._create_selection_trace(selected_node, selected_score)
        trace["top_nodes"] = self._get_top_nodes_info(nodes_with_scores)
        
        return Selection(
            node_id=selected_node.node_id,
            reason=f"Best UCT score of {selected_score.final_score:.2f}",
            trace=trace
        )


class SoftmaxSelector(MCTSSelector):

    async def select(self, expandable_nodes: List[Node]) -> Selection:
        if len(expandable_nodes) == 1:
            return self._handle_single_node(expandable_nodes[0])
        
        # If prioritize_unrewarded_nodes is enabled, select first unrewarded node
        if self.prioritize_unrewarded_nodes:
            for node in expandable_nodes:
                if node.reward is None and node.is_leaf():
                    logger.info(f"Selected unrewarded Node {node.node_id} (prioritizing unrewarded nodes)")
                    return Selection(
                        node_id=node.node_id,
                        reason="First unrewarded node (prioritizing unrewarded trajectories)",
                        trace=self._create_selection_trace(node, self.uct_score(node), unrewarded_selection=True)
                    )
        
        nodes_with_scores = [(node, self.uct_score(node)) for node in expandable_nodes]
        uct_scores = [score.final_score for _, score in nodes_with_scores]

        # Calculate softmax probabilities
        softmax_scores = np.exp(uct_scores - np.max(uct_scores))
        probabilities = softmax_scores / softmax_scores.sum()

        # Log summary for top nodes (limited to 10)
        top_nodes = sorted(
            zip(expandable_nodes, uct_scores, probabilities),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        logger.info("Softmax selection summary (top 10 nodes):")
        for node, score, prob in top_nodes:
            logger.info(
                f"Node {node.node_id}: Visits={node.visits}, "
                f"Reward={node.reward.value if node.reward else '-'}, "
                f"UCTScore={score:.2f}, Probability={prob:.4f}"
            )

        # Select a node based on the probabilities
        selected_node = random.choices(expandable_nodes, weights=probabilities, k=1)[0]
        selected_index = expandable_nodes.index(selected_node)
        selected_score = nodes_with_scores[selected_index][1]

        logger.info(
            f"Selected Node {selected_node.node_id}: "
            f"UCTScore={uct_scores[selected_index]:.2f}, "
            f"Probability={probabilities[selected_index]:.4f}"
        )

        # Return Selection with detailed trace information
        trace = self._create_selection_trace(
            selected_node, 
            selected_score,
            probability=probabilities[selected_index],
            all_probabilities={node.node_id: prob for node, prob in zip(expandable_nodes, probabilities)}
        )
        trace["top_nodes"] = self._get_top_nodes_info(nodes_with_scores, probabilities)

        return Selection(
            node_id=selected_node.node_id,
            reason=f"Selected via softmax with probability {probabilities[selected_index]:.4f}",
            trace=trace
        )
    


def calculate_similarity(context_a: FileContext, context_b: FileContext) -> float:
    """
    Calculates the similarity score between the two FileContext instances.

    Returns:
        float: The similarity score between 0 and 1.
    """
    # Step 1: File path similarity
    files_a = set(context_a._files.keys())
    files_b = set(context_b._files.keys())
    file_path_similarity = jaccard_similarity(files_a, files_b)

    # Step 2: Span similarity
    span_similarities = []
    for file_path in files_a.intersection(files_b):
        spans_a = context_a._files[file_path].span_ids
        spans_b = context_b._files[file_path].span_ids
        if spans_a or spans_b:
            span_similarity = jaccard_similarity(spans_a, spans_b)
            span_similarities.append(span_similarity)
        else:
            # If both have no spans, consider them fully similar
            span_similarities.append(1.0)
    if span_similarities:
        average_span_similarity = sum(span_similarities) / len(span_similarities)
    else:
        average_span_similarity = 1.0  # Default to full similarity if no spans

    # Step 3: Patch similarity
    patch_similarities = []
    for file_path in files_a.intersection(files_b):
        patch_a = context_a._files[file_path].patch or ""
        patch_b = context_b._files[file_path].patch or ""
        if patch_a or patch_b:
            patch_similarity = string_similarity(patch_a, patch_b)
            patch_similarities.append(patch_similarity)
        else:
            # If both have no patches, consider them fully similar
            patch_similarities.append(1.0)
    if patch_similarities:
        average_patch_similarity = sum(patch_similarities) / len(patch_similarities)
    else:
        average_patch_similarity = 1.0  # Default to full similarity if no patches

    # Combine the similarities with weights
    total_similarity = (
        0.4 * file_path_similarity
        + 0.2 * average_span_similarity
        + 0.4 * average_patch_similarity
    )

    return total_similarity


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculates the Jaccard similarity between two sets.

    Returns:
        float: Jaccard similarity score.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0  # Both sets are empty
    return len(intersection) / len(union)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculates the similarity between two strings using difflib.

    Returns:
        float: Similarity score between 0 and 1.
    """
    if not s1 and not s2:
        return 1.0  # Both strings are empty
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()
