import json
import logging
from typing import Optional, List, Dict, Any, Type

from instructor import OpenAISchema
from pydantic import BaseModel, Field

from moatless.actions.model import ActionArguments, Observation
from moatless.completion.model import Usage, Completion
from moatless.file_context import FileContext
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class Reward(OpenAISchema):
    explanation: Optional[str] = Field(
        None, description="An explanation and the reasoning behind your decision."
    )
    feedback: Optional[str] = Field(
        None, description="Feedback to the alternative branch."
    )
    value: int = Field(
        ...,
        description="As ingle integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue",
    )


class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")
    parent: Optional["Node"] = Field(None, description="The parent node")
    children: List["Node"] = Field(default_factory=list, description="The child nodes")
    is_duplicate: bool = Field(
        False, description="Flag to indicate if the node is a duplicate"
    )
    action: Optional[ActionArguments] = Field(
        None, description="The action associated with the node"
    )
    observation: Optional[Observation] = Field(
        None, description="The output of the action"
    )
    reward: Optional[Reward] = Field(None, description="The reward of the node")
    visits: int = Field(0, description="The number of times the node has been visited")
    value: float = Field(0.0, description="The total value (reward) of the node")
    max_expansions: int = Field(1, description="The maximum number of expansions")
    file_context: Optional[FileContext] = Field(
        None, description="The file context state associated with the node"
    )
    message: Optional[str] = Field(
        None, description="The message associated with the node"
    )
    feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(
        default_factory=dict, description="The completions used in this node"
    )
    possible_actions: List[str] = Field(
        default_factory=list, description="List of possible action types for this node"
    )

    @classmethod
    def stub(cls, **kwargs):
        return cls(node_id=0, **kwargs)

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0

    def expanded_count(self) -> int:
        """Get the number of expanded children."""
        return len([child for child in self.children])

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried and executed from this node."""
        return self.expanded_count() >= self.max_expansions

    def is_terminal(self) -> bool:
        """Determine if the current state is a terminal state."""
        if self.observation and self.observation.terminal:
            return True

        return False

    def is_finished(self) -> bool:
        """Determine if the node is succesfully finished"""
        if self.action and self.action.name == "Finish":
            return True

        return False

    def add_child(self, child_node: "Node"):
        """Add a child node to this node."""
        child_node.parent = self
        self.children.append(child_node)

    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent:
            depth += 1
            node = node.parent
        return depth

    def is_expandable(self) -> bool:
        """Check if the node can be expanded further."""
        return (
            not self.is_terminal()
            and not self.is_fully_expanded()
            and not self.is_duplicate
        )

    def find_duplicate(self) -> Optional["Node"]:
        if not self.parent:
            return None

        for child in self.parent.children:
            if child.node_id != self.node_id and child.equals(self):
                return child

        return None

    def get_trajectory(self) -> List["Node"]:
        nodes = []
        current_node = self
        while current_node is not None:
            nodes.insert(0, current_node)
            current_node = current_node.parent

        return nodes

    def get_expandable_descendants(self) -> List["Node"]:
        """Get all expandable descendants of this node, including self if expandable."""
        expandable_nodes = []
        if self.is_expandable():
            expandable_nodes.append(self)
        for child in self.children:
            expandable_nodes.extend(child.get_expandable_descendants())
        return expandable_nodes

    def get_all_nodes(self) -> List["Node"]:
        nodes = []
        nodes.append(self)
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_root(self) -> "Node":
        node = self
        while node.parent:
            node = node.parent
        return node

    def calculate_mean_reward(self) -> float:
        """
        Calculate the mean trajectory reward for this node.

        Returns:
            float: The mean reward.
        """
        rewards = []
        node = self
        while node is not None:
            rewards.append(node.value / node.visits if node.visits > 0 else 0)
            node = node.parent

        return sum(rewards) / len(rewards) if rewards else 0

    def total_usage(self) -> Usage:
        total_usage = Usage()

        for completion in self.completions.values():
            total_usage += completion.usage

        return total_usage

    def equals(self, other: "Node"):
        if self.action and not other.action:
            return False

        if not self.action and other.action:
            return False

        if self.action.name != other.action.name:
            return False

        return self.action.equals(other.action)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the node and its descendants.

        Returns:
            Dict[str, Any]: A dictionary representation of the node tree.
        """

        def serialize_node(node: "Node") -> Dict[str, Any]:
            exclude_set = {"parent", "children"}
            if "exclude" in kwargs:
                if isinstance(kwargs["exclude"], set):
                    exclude_set.update(kwargs["exclude"])
                elif isinstance(kwargs["exclude"], dict):
                    exclude_set.update(kwargs["exclude"].keys())

            new_kwargs = {k: v for k, v in kwargs.items() if k != "exclude"}
            node_dict = super().model_dump(exclude=exclude_set, **new_kwargs)

            if node.action and "action" not in exclude_set:
                node_dict["action"] = node.action.model_dump(**kwargs)
                node_dict["action"]["action_args_class"] = f"{node.action.__class__.__module__}.{node.action.__class__.__name__}"

            if node.completions and "completions" not in exclude_set:
                node_dict["completions"] = {
                    key: completion.model_dump(**kwargs)
                    for key, completion in node.completions.items()
                }

            if node.reward and "reward" not in exclude_set:
                node_dict["reward"] = node.reward.model_dump(**kwargs)

            if node.observation and "output" not in exclude_set:
                node_dict["output"] = node.observation.model_dump(**kwargs)

            if node.file_context and "file_context" not in exclude_set:
                node_dict["file_context"] = node.file_context.model_dump(**kwargs)

            if not kwargs.get("exclude") or "children" not in kwargs.get("exclude"):
                node_dict["children"] = [
                    serialize_node(child) for child in node.children
                ]

            return node_dict

        return serialize_node(self)

    def persist_tree(self, file_path: str):
        """
        Persist the node and all its descendants to a file.
        Only works for root nodes (nodes with no parent).

        Args:
            file_path (str): The path to the file where the tree will be saved.

        Raises:
            ValueError: If the node is not a root node.
        """
        if self.parent is not None:
            raise ValueError("Only root nodes can be persisted.")

        tree_data = self.model_dump()

        with open(file_path, "w") as f:
            json.dump(tree_data, f, indent=2)

    @classmethod
    def reconstruct(
        cls,
        node_data: Dict[str, Any],
        parent: Optional["Node"] = None,
        repo: Repository | None = None,
    ) -> "Node":
        node = cls(
            node_id=node_data["node_id"],
            parent=parent,
            visits=node_data["visits"],
            value=node_data["value"],
            max_expansions=node_data["max_expansions"],
            message=node_data["message"],
            feedback=node_data.get("feedback"),
            is_duplicate=node_data.get("is_duplicate", False),
        )

        if node_data.get("possible_actions"):
            node.possible_actions = node_data.get("possible_actions")

        if node_data.get("action"):
            node.action = ActionArguments.model_validate(node_data["action"])

        if node_data.get("output"):
            node.observation = Observation.model_validate(node_data["output"])

        if node_data.get("completions"):
            for key, completion_data in node_data["completions"].items():
                completion = Completion.model_validate(completion_data)
                node.completions[key] = completion

        if node_data.get("reward"):
            node.reward = Reward.model_validate(node_data["reward"])

        if node_data.get("file_context"):
            node.file_context = FileContext.from_dict(
                repo=repo, data=node_data["file_context"]
            )

        for child_data in node_data.get("children", []):
            child = cls.reconstruct(child_data, parent=node, repo=repo)
            node.add_child(child)

        return node


def generate_ascii_tree(root: Node, current: Node | None = None) -> str:
    tree_lines = ["MCTS Tree"]
    _append_ascii_node(root, "", True, tree_lines, current)
    return "\n".join(tree_lines)


def _append_ascii_node(
    node: Node, prefix: str, is_last: bool, tree_lines: list[str], current: Node | None
):
    state_params = []

    if node.action:
        state_params.append(node.action.name)

        if node.observation and node.observation.expect_correction:
            state_params.append("expect_correction")

    state_info = f"Node{node.node_id}"
    if state_params:
        state_info += f"({', '.join(state_params)})"
    else:
        state_info += f"()"

    if current and node.node_id == current.node_id:
        state_info = color_white(state_info)

    if not node.reward:
        reward_str = "0"
    elif node.reward.value >= 75:
        reward_str = color_green(node.reward.value)
    elif node.reward.value <= 0:
        reward_str = color_red(node.reward.value)
    else:
        reward_str = color_yellow(node.reward.value)

    # avg_reward = node.get_mean_traj_reward()
    if not node.reward:
        node_str = f"Node{node.node_id} [-]"
    elif node.reward.value >= 75:
        node_str = color_green(f"Node{node.node_id} [{node.reward.value}]")
    elif node.reward.value < 0:
        node_str = color_red(f"Node{node.node_id} [{node.reward.value}]")
    else:
        node_str = color_yellow(f"Node{node.node_id} [{node.reward.value}]")

    if node.is_duplicate:
        tree_lines.append(
            f"{prefix}{'└── ' if is_last else '├── '}Node{node.node_id} {state_info} (duplicate)"
        )
    else:
        tree_lines.append(
            f"{prefix}{'└── ' if is_last else '├── '}{node_str} {state_info} (expansions: {node.expanded_count()}, reward: {reward_str}, visits: {node.visits})"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = node.children
        for i, child in enumerate(node.children):
            _append_ascii_node(
                child, child_prefix, i == len(children) - 1, tree_lines, current
            )


def color_red(text: Any) -> str:
    return f"\033[91m{text}\033[0m"


def color_green(text: Any) -> str:
    return f"\033[92m{text}\033[0m"


def color_yellow(text: Any) -> str:
    return f"\033[93m{text}\033[0m"


def color_white(text: Any) -> str:
    return f"\033[97m{text}\033[0m"
