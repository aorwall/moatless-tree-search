import logging
import math
import random
from dataclasses import dataclass
from typing import List, Type, Literal, Dict, Any, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from moatless.selector.similarity import calculate_similarity
from moatless.node import Node

logger = logging.getLogger(__name__)


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
    expect_correction_bonus: float = 0.0
    diversity_bonus: float = 0.0
    duplicate_child_penalty: float = 0.0

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
            f"Expect Correction Bonus: {self.expect_correction_bonus:.2f}",
            f"Diversity Bonus: {self.diversity_bonus:.2f}",
            f"Duplicate Child Penalty: {self.duplicate_child_penalty:.2f}",
        ]
        return ", ".join(components)


class Selector(BaseModel):
    type: Literal["BestFirstSelector", "SoftmaxSelector"] = Field(
        ..., description="The type of selector"
    )
    exploration_weight: float = Field(
        default=1.0,
        description="Weight factor for the exploration term in the UCT score calculation. Higher values encourage more exploration of less-visited nodes.",
    )
    depth_weight: float = Field(
        default=0.8,
        description="Weight factor for the depth-based components in the UCT score. Affects both the depth bonus and penalty calculations.",
    )
    depth_bonus_factor: float = Field(
        default=0.0,
        description="Factor used in calculating the depth bonus. Higher values increase the bonus for exploring deeper nodes, especially near the root.",
    )
    high_value_threshold: float = Field(
        default=55.0,
        description="Threshold for considering a node's reward as 'high value'. Used in various bonus calculations.",
    )
    low_value_threshold: float = Field(
        default=50.0,
        description="Threshold for considering a node's reward as 'low value'. Used in various penalty calculations.",
    )
    very_high_value_threshold: float = Field(
        default=75.0,
        description="Threshold for considering a node's reward as 'very high value'. Used in the high value child penalty calculation.",
    )
    high_value_leaf_bonus_constant: float = Field(
        default=20.0,
        description="Constant bonus applied to high-value leaf nodes to encourage their exploration.",
    )
    high_value_bad_children_bonus_constant: float = Field(
        default=20.0,
        description="Constant used in calculating the bonus for high-value nodes with low-value children, encouraging 'auto-correction'.",
    )
    high_value_child_penalty_constant: float = Field(
        default=5.0,
        description="Constant used in penalizing nodes with very high-value children to prevent over-exploitation of a single path.",
    )
    finished_trajectory_penalty: float = Field(
        default=50.0,
        description="Penalty applied to nodes on a trajectory that has already finished with a high reward, discouraging revisiting completed paths.",
    )
    expect_correction_bonus: float = Field(
        default=0.0,
        description="Bonus applied to nodes expecting correction, prioritizing exploration of potential fix paths.",
    )
    check_for_bad_child_actions: List[str] = Field(
        default_factory=lambda: ["RequestCodeChange"],
        description="List of action types to check for when calculating the high value bad children bonus.",
    )
    diversity_weight: float = Field(
        default=100.0,
        description="Weight factor for the diversity bonus. Higher values increase the bonus for nodes with low similarity to other explored nodes."
    )
    duplicate_child_penalty_constant: float = Field(
        default=25.0,
        description="Constant used in penalizing nodes that have duplicate children. Penalty increases with each duplicate."
    )

    _similarity_cache: Dict[Tuple[int, int], float] = PrivateAttr(default_factory=dict)

    def select(self, expandable_nodes: List[Node]) -> Node:
        raise NotImplementedError("Subclasses must implement the select method.")

    def uct_score(self, node: Node) -> UCTScore:
        """
        Compute the UCT score with additional bonuses and penalties based on node characteristics.

        This method combines various components to create a comprehensive score for node selection,
        balancing exploration and exploitation while considering node-specific factors.
        """
        if node.visits == 0:
            return UCTScore(final_score=float("inf"))

        exploitation = self.calculate_exploitation(node)
        exploration = self.calculate_exploration(node)
        depth_bonus = self.calculate_depth_bonus(node)
        depth_penalty = self.calculate_depth_penalty(node)
        high_value_leaf_bonus = self.calculate_high_value_leaf_bonus(node)
        high_value_bad_children_bonus = self.calculate_high_value_bad_children_bonus(
            node
        )
        high_value_child_penalty = self.calculate_high_value_child_penalty(node)
        high_value_parent_bonus = self.calculate_high_value_parent_bonus(node)
        finished_trajectory_penalty = self.calculate_finished_trajectory_penalty(node)
        expect_correction_bonus = self.calculate_expect_correction_bonus(node)
        diversity_bonus = self.calculate_diversity_bonus(node)
        duplicate_child_penalty = self.calculate_duplicate_child_penalty(node)

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
            + expect_correction_bonus
            + diversity_bonus
            - duplicate_child_penalty
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
            expect_correction_bonus=expect_correction_bonus,
            diversity_bonus=diversity_bonus,
            duplicate_child_penalty=duplicate_child_penalty,
        )

    def calculate_exploitation(self, node: Node) -> float:
        """
        Calculate the exploitation component of the UCT score.

        Purpose: Favors nodes with higher rewards, encouraging the algorithm to exploit
        known good paths in the search tree.
        """
        return node.reward.value if node.reward else 0

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
        exploitation = self.calculate_exploitation(node)
        if not node.children and exploitation >= self.high_value_threshold:
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
        exploitation = self.calculate_exploitation(node)
        if not node.children:
            if node.parent and (
                not node.parent.reward
                or node.parent.reward.value > self.high_value_threshold
            ):
                if exploitation <= self.low_value_threshold:
                    return self.high_value_threshold - exploitation
        return 0

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

    def calculate_expect_correction_bonus(self, node: Node) -> float:
        """
        Calculate the bonus for nodes with a parent node that expect correction.

        Purpose: Prioritizes nodes that are marked as expecting correction (e.g., after
        a failed test run or an invalid search request). This bonus decreases rapidly
        as the parent node accumulates more children, encouraging exploration of less-visited
        correction paths.
        """
        if (node.observation and node.observation.expect_correction and
                not (node.parent and node.parent.observation and node.parent.observation.expect_correction)):  # TODO: Set parent as decay factor  instead?
            # Use a more aggressive decay factor
            decay_factor = 1 / (1 + len(node.children) ** 2)
            return self.expect_correction_bonus * decay_factor

        return 0

    def calculate_diversity_bonus(self, node: Node) -> float:
        """
        Calculate the diversity bonus based on the similarity of the node's solution to already expanded nodes.

        Purpose: Boosts the score for nodes whose solutions have low similarity to other explored nodes,
        encouraging the exploration of novel solutions.
        """
        if not self.diversity_weight:
            return 0

        expandable_nodes = [n for n in node.get_root().get_expanded_descendants() if n.node_id != node.node_id]

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
            return self.duplicate_child_penalty_constant * (duplicate_count ** 2)
        return 0

    def get_similarity(self, node_a: Node, node_b: Node) -> float:
        """
        Retrieve the similarity between two nodes from the cache or compute it if not cached.
        """
        if node_a.file_context is None or node_b.file_context is None:
            return 0.0

        node_ids = (min(node_a.node_id, node_b.node_id), max(node_a.node_id, node_b.node_id))
        if node_ids in self._similarity_cache:
            return self._similarity_cache[node_ids]

        similarity = calculate_similarity(node_a.file_context, node_b.file_context)
        self._similarity_cache[node_ids] = similarity
        return similarity

    @classmethod
    def model_validate(cls: Type["Selector"], obj: Any) -> "Selector":
        if isinstance(obj, dict):
            selector_type = obj.get("type")
            if selector_type == "BestFirstSelector":
                return BestFirstSelector(**obj)
            elif selector_type == "SoftmaxSelector":
                return SoftmaxSelector(**obj)
            else:
                raise ValueError(f"Unknown selector type: {selector_type}")
        return super().model_validate(obj)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["type"] = self.__class__.__name__
        return dump


class BestFirstSelector(Selector):
    type: Literal["BestFirstSelector"] = "BestFirstSelector"

    def select(self, expandable_nodes: List[Node]) -> Node:
        if len(expandable_nodes) == 1:
            return expandable_nodes[0]

        # Calculate UCT scores with components
        nodes_with_scores = [(node, self.uct_score(node)) for node in expandable_nodes]
        sorted_nodes = sorted(
            nodes_with_scores, key=lambda x: x[1].final_score, reverse=True
        )

        # Log top nodes with detailed score breakdowns
        top_nodes = sorted_nodes[: min(len(sorted_nodes), 10)]
        logger.info("Comparing top nodes:")
        for i, (node, score) in enumerate(top_nodes):
            logger.info(
                f"Node {node.node_id} - Visits: {node.visits} - "
                f"Reward: {node.reward.value if node.reward else '-'} - "
                f"\nScore components: {score}"
            )

        # Select the node with the highest UCT score
        selected_node = sorted_nodes[0][0]
        selected_score = sorted_nodes[0][1].final_score

        logger.info(
            f"Selected Node {selected_node.node_id} with UCT Score: {selected_score:.2f}"
        )
        return selected_node


class SoftmaxSelector(Selector):
    type: Literal["SoftmaxSelector"] = "SoftmaxSelector"

    def select(self, expandable_nodes: List[Node]) -> Node:
        if len(expandable_nodes) == 1:
            return expandable_nodes[0]

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

        logger.info(
            f"Selected Node {selected_node.node_id}: "
            f"UCTScore={uct_scores[selected_index]:.2f}, "
            f"Probability={probabilities[selected_index]:.4f}"
        )

        return selected_node
