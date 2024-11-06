import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Type

from instructor import OpenAISchema
from pydantic import BaseModel, Field

from moatless.actions.model import ActionArguments, Observation
from moatless.completion.model import Usage, Completion, Message, UserMessage, AssistantMessage
from moatless.file_context import FileContext
from moatless.repository.repository import Repository
from moatless.value_function.model import Reward

logger = logging.getLogger(__name__)


class MessageHistoryType(Enum):
    MESSAGES = "messages"  # Provides all messages in sequence
    SUMMARY = "summary"  # Generates one message with summarized history


message_history_type: MessageHistoryType = Field(
    default=MessageHistoryType.SUMMARY,
    description="Determines how message history is generated"
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

        for completion in self.completions.values():
            total_usage += completion.usage

        return total_usage

    def generate_message_history(self, 
                                 message_history_type: MessageHistoryType,
                                 include_file_context: bool = True,
                                 include_extra_history: bool = True,
                                 include_git_patch: bool = True) -> list[Message]:
        messages = [UserMessage(content=f"<task>\n{self.get_root().message}\n</task>\n\n")]

        previous_nodes = self.get_trajectory()[:-1]

        if message_history_type == MessageHistoryType.SUMMARY:
            messages.extend(self.generate_summary_history(previous_nodes, include_extra_history, include_file_context, include_git_patch))
        else:  # MessageHistoryType.MESSAGES
            messages.extend(self.generate_full_history(previous_nodes, include_extra_history, include_file_context, include_git_patch))

        return messages

    def generate_summary_history(self, previous_nodes: List["Node"], include_extra_history: bool = True, include_file_context: bool = True, include_git_patch: bool = True) -> list[Message]:
        """Generate a single message containing summarized history."""
        formatted_history: List[str] = []
        counter = 0

        # Generate history
        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = f"\n## {counter}. Action: {previous_node.action.name}\n"
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    formatted_state += f"\n\nOutput: {previous_node.observation.message}"
                    if (i == len(previous_nodes) - 1) or include_extra_history:
                        if previous_node.observation.extra:
                            formatted_state += "\n\n"
                            formatted_state += previous_node.observation.extra
                    formatted_history.append(formatted_state)
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")

        content = "Below is the history of previously executed actions and their outputs.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"

        if include_file_context:
            content += self._format_file_context()

        if include_git_patch:
            content += self._format_git_patch()

        content += self._format_feedback()
        return [UserMessage(content=content)]

    def generate_full_history(self, previous_nodes: List["Node"], include_extra_history: bool = True, include_file_context: bool = True, include_git_patch: bool = True) -> list[Message]:
        """Generate a sequence of messages representing the full conversation history."""
        messages: list[Message] = []

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                tool_call = previous_node.action.to_tool_call()
                messages.append(AssistantMessage(tool_call=tool_call))

                content = ""
                if previous_node.message:
                    content = f"<issue>\n{previous_node.message}\n</issue>"

                if previous_node.observation:
                    content += previous_node.observation.message
                    if (i == len(previous_nodes) - 1) or include_extra_history:
                        if previous_node.observation.extra:
                            content += "\n" + previous_node.observation.extra

                if not content:
                    logger.warning(
                        f"Node{previous_node.node_id}: No content to add to messages"
                    )

                messages.append(UserMessage(content=content))

        # Add file context, git patch, and feedback as the final message

        if include_file_context:
            context = self._format_file_context()
        else:
            context = ""

        if include_git_patch:
            git_patch = self._format_git_patch()
        else:
            git_patch = ""

        feedback = self._format_feedback()

        if context or git_patch or feedback:
            messages.append(UserMessage(content=context + git_patch + feedback))

        return messages

    def _format_file_context(self) -> str:
        """Generate formatted string for file context."""
        if not self.file_context:
            return ""

        if self.file_context.is_empty():
            file_context_str = "No files added to file context yet."
        else:
            file_context_str = self.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        return f"# Current file context:\n\n<file_context>\n{file_context_str}\n</file_context>\n\n"

    def _format_git_patch(self) -> str:
        """Generate formatted string for git patch."""
        if not self.file_context:
            return ""

        full_patch = self.file_context.generate_git_patch()
        message = "Changes made to the codebase so far:\n"
        if full_patch.strip():
            message += "<git_patch>\n"
            message += full_patch
            message += "\n</git_patch>\n\n"
        else:
            message += "<git_patch>\n"
            message += "No changes made yet."
            message += "\n</git_patch>\n\n"
        return message

    def _format_feedback(self) -> str:
        """Generate formatted string for feedback."""
        if not self.feedback:
            return ""

        return f"\n\n{self.feedback}"

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


def generate_ascii_tree(
    root: Node, 
    current: Node | None = None, 
    include_explanation: bool = False,
    use_color: bool = True
) -> str:
    tree_lines = ["MCTS Tree"]
    _append_ascii_node(
        root, 
        "", 
        True, 
        tree_lines, 
        current, 
        include_explanation,
        use_color
    )
    return "\n".join(tree_lines)


def _append_ascii_node(
    node: Node, 
    prefix: str, 
    is_last: bool, 
    tree_lines: list[str], 
    current: Node | None,
    include_explanation: bool = False,
    use_color: bool = True
) -> None:
    # Build node information
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

    if use_color and current and node.node_id == current.node_id:
        state_info = color_white(state_info)

    # Build reward string
    if not node.reward:
        reward_str = "0"
        node_str = f"Node{node.node_id} [-]"
    else:
        if use_color:
            if node.reward.value >= 75:
                reward_str = color_green(node.reward.value)
                node_str = color_green(f"Node{node.node_id} [{node.reward.value}]")
            elif node.reward.value <= 0:
                reward_str = color_red(node.reward.value)
                node_str = color_red(f"Node{node.node_id} [{node.reward.value}]")
            else:
                reward_str = color_yellow(node.reward.value)
                node_str = color_yellow(f"Node{node.node_id} [{node.reward.value}]")
        else:
            reward_str = str(node.reward.value)
            node_str = f"Node{node.node_id} [{node.reward.value}]"

    # Add expandable status
    expandable_str = "expandable" if node.is_expandable() else "not-expandable"
    if use_color:
        expandable_str = color_green(expandable_str) if node.is_expandable() else color_red(expandable_str)

    # Calculate the current node's connection prefix
    connection = "└── " if is_last else "├── "
    
    # Add the node line with expandable status
    tree_lines.append(f"{prefix}{connection}{node_str} {state_info} "
                     f"(expansions: {node.expanded_count()}, reward: {reward_str}, "
                     f"visits: {node.visits}, {expandable_str})")
    
    # Calculate the explanation prefix - should align with the node's content
    explanation_prefix = prefix + ("    " if is_last else "│   ")
    
    # Add explanation if available
    if include_explanation and node.reward and node.reward.explanation:
        # Split and wrap explanation text
        explanation_text = node.reward.explanation.strip()
        words = explanation_text.split()
        current_line = []
        current_length = 0
        max_line_length = 100
        
        for word in words:
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                tree_lines.append(f"{explanation_prefix}{' '.join(current_line)}")
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            tree_lines.append(f"{explanation_prefix}{' '.join(current_line)}")
    
    # Calculate the prefix for children
    child_prefix = prefix + ("    " if is_last else "│   ")
    
    # Process children
    children = node.children
    for i, child in enumerate(children):
        _append_ascii_node(
            child,
            child_prefix,
            i == len(children) - 1,
            tree_lines,
            current,
            include_explanation,
            use_color
        )


def color_red(text: Any) -> str:
    return f"\033[91m{text}\033[0m"


def color_green(text: Any) -> str:
    return f"\033[92m{text}\033[0m"


def color_yellow(text: Any) -> str:
    return f"\033[93m{text}\033[0m"


def color_white(text: Any) -> str:
    return f"\033[97m{text}\033[0m"
