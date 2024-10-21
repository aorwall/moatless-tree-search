import json
import logging
from typing import List, Optional, Dict, Any

from moatless.agent.agent import Agent

from moatless.completion import Usage
from moatless.discriminators import MeanAwardDiscriminator
from moatless.feedback import FeedbackGenerator
from moatless.node import Node, generate_ascii_tree
from moatless.selector import BestFirstSelector, Selector, SoftmaxSelector
from moatless.settings import TreeSearchSettings
from moatless.value_function import ValueFunction
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class SearchTree:
    def __init__(
        self,
        message: str | None = None,
        root: Node | None = None,
        workspace: Workspace | None = None,
        settings: TreeSearchSettings | None = None,
        selector: Selector | None = None,
        agent: Agent | None = None,
        value_function: ValueFunction | None = None,
        feedback_generator: FeedbackGenerator | None = None,
        discriminator: MeanAwardDiscriminator | None = None,
        metadata: Dict[str, Any] | None = None,
        persist_path: str | None = None,
    ):
        """
        Initialize a SearchTree instance.

        Args:
            message (str | None): The incoming request message from the user.
            root (Node | None): The root node of an existing search tree. If not provided, a new root node will be created.
            workspace (Workspace | None): The workspace for file operations.
            settings (TreeSearchSettings | None): Settings for the tree search algorithm.
            selector (Selector | None): Custom selector for node selection.
            agent (Agent | None): Custom agent for generating actions.
            value_function (ValueFunction | None): Custom value function for reward calculation.
            feedback_generator (FeedbackGenerator | None): Custom feedback generator.
            discriminator (MeanAwardDiscriminator | None): Custom discriminator for selecting the best trajectory.
            metadata (Dict[str, Any] | None): Additional metadata for the search tree.
            persist_path (str | None): Path to persist the search tree.

        Raises:
            ValueError: If neither root nor message is provided.
        """
        if not root and not message:
            raise ValueError("Either a root node or a message must be provided.")

        if root:
            self.root = root
        else:
            self.root = Node(
                node_id=0,
                max_expansions=settings.max_expansions,
                message=message,
                file_context=workspace.create_file_context() if workspace else None,
            )
        self.unique_id = 0
        self.settings = settings or TreeSearchSettings()
        self.workspace = workspace
        self.metadata = metadata

        if selector:
            self.selector = selector
        elif settings.best_first:
            self.selector = BestFirstSelector()
        else:
            self.selector = SoftmaxSelector()

        self.agent = agent or Agent(
            workspace=self.workspace, model_settings=settings.agent_model
        )

        self.value_function = value_function

        self.feedback_generator = feedback_generator or FeedbackGenerator()
        self.discriminator = discriminator or MeanAwardDiscriminator()

        self.persist_path = persist_path

    @classmethod
    def from_dict(
        cls,
        dict: Dict[str, Any],
        workspace: Workspace | None = None,
        persist_path: str | None = None,
    ) -> "SearchTree":
        settings = TreeSearchSettings(**dict["settings"])
        root = Node.reconstruct(
            dict["tree"], repo=workspace.file_repo if workspace else None
        )
        tree = cls(
            root=root,
            workspace=workspace,
            settings=settings,
            metadata=dict.get("metadata"),
            persist_path=persist_path,
        )
        tree.unique_id = len(tree.root.get_all_nodes())
        return tree

    @classmethod
    def from_file(
        cls,
        file_path: str,
        workspace: Workspace | None = None,
        persist_path: str | None = None,
    ) -> "SearchTree":
        with open(file_path, "r") as f:
            tree_data = json.load(f)

        return cls.from_dict(
            tree_data, workspace=workspace, persist_path=persist_path or file_path
        )

    def run_search(self) -> Node | None:
        """Run the MCTS algorithm for a specified number of iterations."""

        logger.info(generate_ascii_tree(self.root))

        if len(self.root.get_all_nodes()) > 1:
            logger.info(
                f"Restarting search tree with {len(self.root.get_all_nodes())} nodes"
            )

        while not self.is_finished():
            logger.info(f"Run iteration {len(self.root.get_all_nodes())}")

            node = self._select(self.root)
            if node:
                new_node = self._expand(node)
                self._simulate(new_node)
                self._backpropagate(new_node)
                self.maybe_persist()
                logger.info(generate_ascii_tree(self.root, new_node))
            else:
                logger.info("Search complete: no more nodes to expand.")
                break

        if not len(self.get_finished_nodes()):
            logger.warning(
                f"Search completed with no finished nodes. {len(self.root.get_all_nodes())} nodes created."
            )
        else:
            logger.info(
                f"Search completed with {len(self.get_finished_nodes())} finished nodes. {len(self.root.get_all_nodes())} nodes created."
            )

        return self.get_best_trajectory()

    def _select(self, node: Node) -> Optional[Node]:
        """Select a node for expansion using the UCT algorithm."""
        expandable_nodes = node.get_expandable_descendants()

        filtered_nodes = []
        for node in expandable_nodes:
            if node.get_depth() >= self.settings.max_depth:
                continue

            filtered_nodes.append(node)

        expandable_nodes = filtered_nodes

        if not expandable_nodes:
            logger.info("No expandable nodes found.")
            return None

        return self.selector.select(expandable_nodes)

    def _expand(self, node: Node) -> Node:
        """Expand the node by returning an unexecuted child or adding a new child node."""
        # Check for unexecuted children
        for child in node.children:
            if not child.is_duplicate and child.output is None:
                logger.info(
                    f"Returning unexecuted child Node{child.node_id} of Node{node.node_id}"
                )
                return child

        feedback = self.feedback_generator.generate_feedback(node)
        child_node = Node(
            node_id=self._generate_unique_id(),
            parent=node,
            file_context=node.file_context.clone() if node.file_context else None,
            max_expansions=self.settings.max_expansions,
            feedback=feedback,
        )
        node.add_child(child_node)
        logger.info(f"Expanded Node{node.node_id} to new Node{child_node.node_id}")
        return child_node

    def _simulate(self, node: Node):
        """Simulate a playout by executing the action and evaluating the result."""

        if not node.action:
            action, completion_response = self.agent.generate_action(node)
            node.action = action

            if completion_response:
                node.completions["build_action"] = completion_response

            duplicate_node = node.find_duplicate()
            if duplicate_node:
                logger.info(
                    f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution."
                )
                node.is_duplicate = True
                return
        else:
            # To set workspace on saved but not executed actions
            node.action.set_workspace(self.workspace)

        output = node.action.execute(node.file_context)

        # TODO: Return completion_response from execute instead?
        if output.execution_completion:
            node.completions["execute_action"] = output.execution_completion
            output.execution_completion = None

        node.output = output
        node.message = output.message
        logger.info(
            f"Node{node.node_id}: Executed action: {node.action.name}. "
            f"Terminal: {node.output.terminal}. "
            f"Output: {node.output.message}"
        )

        if self.value_function:
            node.reward, completion_response = self.value_function.get_reward(node=node)
            node.completions["value_function"] = completion_response
            logger.info(f"Node{node.node_id}: Reward = {node.reward.value}.")

    def _backpropagate(self, node: Node):
        """Backpropagate the reward up the tree."""

        if not node.reward:
            logger.info(
                f"Node{node.node_id} has no evaluation. Skipping backpropagation."
            )
            return

        reward = node.reward.value
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_best_trajectory(self) -> Node | None:
        """
        Get the best finished trajectory to return
        """

        nodes = self.get_finished_nodes()
        if not nodes:
            nodes = self.get_leaf_nodes()
            logger.info(
                f"get_best_trajectory() No finished nodes found. Will select from {len(nodes)} leaf nodes."
            )

        if len(nodes) == 1:
            return nodes[0]

        return self.discriminator.select(nodes)

    def is_finished(self):
        if len(self.root.get_all_nodes()) >= self.settings.max_iterations:
            return True

        finished_nodes = self.get_finished_nodes()

        # Expect at least one finished node with reward above threshold
        if self.settings.reward_threshold and not any(
            node.reward and node.reward.value >= self.settings.reward_threshold
            for node in finished_nodes
        ):
            return False

        if (
            self.settings.min_finished_nodes
            and len(self.get_finished_nodes()) >= self.settings.min_finished_nodes
        ):
            return True

        if not self.root.get_expandable_descendants():
            return True

        return False

    def get_finished_nodes(self) -> List[Node]:
        """Get all finished nodes in the search tree by uniqe parent node."""
        parent_ids = set()
        finished_nodes = []
        for node in self.root.get_all_nodes():
            # TODO: Pick finished node with highest/avg/lowest reward?
            if node.is_finished() and node.parent.node_id not in parent_ids:
                parent_ids.add(node.parent.node_id)
                finished_nodes.append(node)

        return finished_nodes

    def get_leaf_nodes(self) -> List[Node]:
        """Get all leaf nodes in the search tree."""
        return [node for node in self.root.get_all_nodes() if node.is_leaf()]

    def total_usage(self) -> Usage:
        total_usage = Usage()
        for node in self.root.get_all_nodes():
            total_usage += node.total_usage()
        return total_usage

    def maybe_persist(self):
        if self.persist_path:
            self.persist(self.persist_path)

    def persist(self, file_path: str, **kwargs):
        """
        Persist the entire SearchTree to a file.

        Args:
            file_path (str): The path to the file where the tree will be saved.
        """
        tree_data = self.model_dump(**kwargs)

        with open(file_path, "w") as f:
            try:
                json.dump(tree_data, f, indent=2)
            except Exception as e:
                logger.exception(
                    f"Error saving search tree to {file_path}: {tree_data}"
                )
                raise e

    def _generate_unique_id(self) -> int:
        self.unique_id += 1
        return self.unique_id

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the SearchTree.

        Returns:
            Dict[str, Any]: A dictionary representation of the search tree.
        """
        return {
            "settings": self.settings.model_dump(**kwargs),
            "tree": self.root.model_dump(**kwargs),
            "metadata": self.metadata,
        }


def find_best_by_mean_award(finished_nodes: List[Node]) -> Node | None:
    best_finish_node: Node | None = None
    best_mean_reward = float("-inf")
    trajectories_mean_rewards = []

    for finished_node in finished_nodes:
        mean_reward = finished_node.calculate_mean_reward()

        trajectories_mean_rewards.append(mean_reward)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_finish_node = finished_node

    logger.info(f"Mean Rewards for finished trajectories: {trajectories_mean_rewards}")

    if best_finish_node:
        logger.info(
            f"Best finished path finished on Node{best_finish_node.node_id} with mean reward: {best_mean_reward}"
        )
        return best_finish_node
    else:
        logger.info(
            "No valid finished path found. This should not happen if there are finished nodes."
        )
        return None
