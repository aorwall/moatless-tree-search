import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from pydantic import BaseModel, Field

from moatless.actions.model import ActionArguments, Observation
from moatless.agent.settings import AgentSettings
from moatless.completion.model import (
    Usage,
    Completion,
    Message,
    UserMessage,
    AssistantMessage,
)
from moatless.file_context import FileContext
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)


class ActionStep(BaseModel):
    action: ActionArguments
    observation: Optional[Observation] = None
    completion: Optional[Completion] = None


class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")
    parent: Optional["Node"] = Field(None, description="The parent node")
    children: List["Node"] = Field(default_factory=list, description="The child nodes")
    
    user_message: Optional[str] = Field(None, description="The user message for this node")
    assistant_message: Optional[str] = Field(None, description="The assistant response for this node")
    
    action_steps: List[ActionStep] = Field(
        default_factory=list,
        description="The sequence of actions and observations for this node"
    )
    
    file_context: Optional[FileContext] = Field(
        None, description="The file context state associated with the node"
    )
    feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(
        default_factory=dict, description="The completions used in this node"
    )
    possible_actions: List[str] = Field(
        default_factory=list, description="List of possible action types for this node"
    )
    is_duplicate: Optional[bool] = Field(
        None, description="Flag to indicate if the node is a duplicate"
    )
    reward: Optional[Reward] = Field(None, description="The reward of the node")
    visits: int = Field(0, description="The number of times the node has been visited")
    value: Optional[float] = Field(None, description="The total value (reward) of the node")
    max_expansions: Optional[int] = Field(None, description="The maximum number of expansions")
    agent_settings: Optional[AgentSettings] = Field(
        None, description="The agent settings associated with the node"
    )

    @property
    def action(self) -> Optional[ActionArguments]:
        """Backward compatibility: Get action from the latest action step"""
        if not self.action_steps:
            return None
        return self.action_steps[-1].action if self.action_steps else None

    @action.setter
    def action(self, value: Optional[ActionArguments]):
        """Backward compatibility: Set action on the current/new action step"""

        if not self.action_steps:
            self.action_steps.append(ActionStep(action=value))
        else:
            self.action_steps[-1].action = value

    @property 
    def observation(self) -> Optional[Observation]:
        """Backward compatibility: Get observation from the latest action step"""
        if not self.action_steps:
            return None
        return self.action_steps[-1].observation if self.action_steps else None

    @observation.setter
    def observation(self, value: Optional[Observation]):
        """Backward compatibility: Set observation on the current/new action step"""
        if value is None:
            return
            
        if not self.action_steps:
            # Create new action step if setting observation on empty node
            self.action_steps.append(ActionStep(
                action=self.action,
                observation=value
            ))
        else:
            self.action_steps[-1].observation = value

    @property
    def message(self) -> Optional[str]:
        """Backward compatibility: Get message maps to user_message"""
        return self.user_message

    @message.setter
    def message(self, value: Optional[str]):
        """Backward compatibility: Set message maps to user_message"""
        self.user_message = value

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
        return self.expanded_count() >= (self.max_expansions or 1)

    def is_terminal(self) -> bool:
        """Determine if the current state is a terminal state."""
        if not self.action_steps:
            return False
        last_step = self.action_steps[-1]
        return last_step.observation.terminal if last_step.observation else False

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

    def get_sibling_nodes(self) -> List["Node"]:
        if not self.parent:
            return []

        return [
            child for child in self.parent.children if child.node_id != self.node_id
        ]

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

    def get_expanded_descendants(self) -> List["Node"]:
        """Get all expanded descendants of this node, including self if expanded."""
        expanded_nodes = []
        if self.expanded_count() > 0:
            expanded_nodes.append(self)
        for child in self.children:
            expanded_nodes.extend(child.get_expanded_descendants())
        return expanded_nodes

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
        
        # Sum usage across all action steps
        for step in self.action_steps:
            if step.completion:
                total_usage += step.completion.usage
                    
        return total_usage


    def equals(self, other: "Node"):
        if self.action and not other.action:
            return False

        if not self.action and other.action:
            return False

        if self.action.name != other.action.name:
            return False

        return self.action.equals(other.action)

    def reset(self):
        """Reset the node state to be able to execute it again."""
        self.action_steps = []
        self.user_message = None
        self.assistant_message = None
        self.visits = 0
        self.value = 0.0
        self.feedback = None
        self.is_duplicate = False
        if self.parent and self.parent.file_context:
            self.file_context = self.parent.file_context.clone()
        self.children = []

    def clone_and_reset(self) -> "Node":
        """
        Creates a copy of the node and resets its observation and file context.

        Returns:
            Node: A new node instance with reset state
        """
        # Create a new node with same base attributes
        new_node = Node(
            node_id=self.node_id,
            parent=self.parent,
            visits=self.visits,
            value=self.value,
            max_expansions=self.max_expansions,
            user_message=self.user_message,
            feedback=self.feedback,
            is_duplicate=self.is_duplicate,
            action=self.action,
            possible_actions=self.possible_actions.copy()
            if self.possible_actions
            else [],
        )

        new_node.reset()
        return new_node

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
                node_dict["action"]["action_args_class"] = (
                    f"{node.action.__class__.__module__}.{node.action.__class__.__name__}"
                )

            if node.completions and "completions" not in exclude_set:
                node_dict["completions"] = {
                    key: completion.model_dump(**kwargs)
                    for key, completion in node.completions.items()
                    if completion
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

    @classmethod
    def _reconstruct_node(
        cls,
        node_data: Dict[str, Any],
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None
    ) -> "Node":
        """Update reconstruction to handle both old and new formats"""
        
        # Handle legacy format conversion
        if "action" in node_data and not "action_steps" in node_data:
            action = node_data.get("action")
            observation = node_data.get("output")
            completions = node_data.get("completions", {})
            
            if action or observation or completions:
                node_data["action_steps"] = [{
                    "action": action,
                    "observation": observation,
                    "completions": completions
                }]

        if "message" in node_data and not "user_message" in node_data:
            node_data["user_message"] = node_data.pop("message")

        if node_data.get("action_steps"):
            node_data["action_steps"] = [
                ActionStep.model_validate(step) for step in node_data["action_steps"]
            ]

        if node_data.get("file_context"):
            node_data["file_context"] = FileContext.from_dict(
                repo=repo, runtime=runtime, data=node_data["file_context"]
            )

        node_data["visits"] = node_data.get("visits", 0)
        node_data["value"] = node_data.get("value", 0.0)

        if "children" in node_data:
            children = node_data.get("children", [])

            del node_data["children"]
            node = super().model_validate(node_data)

            for child_data in children:
                child = cls._reconstruct_node(child_data, repo=repo, runtime=runtime)
                child.parent = node
                node.children.append(child)

            return node
        else:
            return super().model_validate(node_data)

    @classmethod
    def reconstruct(
        cls,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None
    ) -> "Node":
        """
        Reconstruct a node tree from either dict (tree) or list format.

        Args:
            data: Either a dict (tree format) or list of dicts (list format)
            parent: Optional parent node (used internally)
            repo: Optional repository reference

        Returns:
            Node: Root node of reconstructed tree
        """
        # Handle list format
        if isinstance(data, list):
            return cls._reconstruct_from_list(data, repo=repo, runtime=runtime)

        # Handle single node reconstruction (dict format)
        return cls._reconstruct_node(data, repo=repo, runtime=runtime)

    @classmethod
    def _reconstruct_from_list(
        cls, node_list: List[Dict], repo: Repository | None = None, runtime: RuntimeEnvironment | None = None

    ) -> "Node":
        """
        Reconstruct tree from a flat list of nodes.

        Args:
            node_list: List of serialized nodes
            repo: Optional repository reference

        Returns:
            Node: Root node of reconstructed tree
        """
        # Create nodes without relationships first
        nodes_by_id = {}

        for node_data in node_list:
            parent_id = node_data.pop("parent_id", None)
            # Use the core reconstruct method for each node
            node = cls._reconstruct_node(node_data, repo=repo, runtime=runtime)
            nodes_by_id[node.node_id] = (node, parent_id)

        # Connect parent-child relationships
        for node, parent_id in nodes_by_id.values():
            if parent_id is not None:
                parent_node = nodes_by_id[parent_id][0]
                parent_node.add_child(node)

        root_nodes = [n for n, p_id in nodes_by_id.values() if p_id is None]
        if not root_nodes:
            raise ValueError("No root node found in data")

        return root_nodes[0]

    def dump_as_list(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Dump all nodes as a flat list structure.
        """
        nodes = self.get_all_nodes()
        node_list = []

        for node in nodes:
            node_data = node.model_dump(exclude={"parent", "children"}, **kwargs)
            node_data["parent_id"] = node.parent.node_id if node.parent is not None else None
            node_list.append(node_data)

        return node_list

    @classmethod
    def load_from_file(cls, file_path: str, repo: Repository | None = None) -> "Node":
        """
        Load node tree from file, supporting both old tree format and new list format.

        Args:
            file_path (str): Path to the saved node data
            repo (Repository): Optional repository reference

        Returns:
            Node: Root node of the tree
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return cls.reconstruct_from_list(data, repo=repo)
        else:
            # Old tree format
            return cls.reconstruct(data, repo=repo)

    def persist(self, file_path: str, format: str = "list"):
        """
        Persist the node tree to file.

        Args:
            file_path (str): The path to save to
            format (str): Either "list" (new) or "tree" (legacy)
        """
        if format == "list":
            self.persist_as_list(file_path)
        elif format == "tree":
            self.persist_tree(file_path)
        else:
            raise ValueError("Format must be either 'list' or 'tree'")


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
