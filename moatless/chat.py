import logging
from typing import Optional, Dict, Any, Callable

from pydantic import BaseModel, Field

from moatless.agent.agent import ActionAgent
from moatless.completion.model import Usage
from moatless.exceptions import RuntimeError
from moatless.file_context import FileContext
from moatless.node import Node
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class Chat(BaseModel):
    root: Node = Field(..., description="The root node of the chat sequence.")
    agent: ActionAgent = Field(..., description="Agent for generating responses.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the chat."
    )
    persist_path: Optional[str] = Field(
        None, description="Path to persist the chat sequence."
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        message: str | None = None,
        file_context: Optional[FileContext] = None,
        repository: Repository | None = None,
        agent: Optional[ActionAgent] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_path: Optional[str] = None,
    ) -> "Chat":
        """Create a new Chat instance."""
        if not file_context:
            file_context = FileContext(repo=repository)

        root = Node(
            node_id=0,
            user_message=message,
            file_context=file_context,
        )

        return cls(
            root=root,
            agent=agent,
            metadata=metadata or {},
            persist_path=persist_path,
        )

    def send_message(self, message: str) -> str:
        """Send a message and get a response."""
        self.assert_runnable()

        current_node = self._create_next_node(self.get_last_node())
        current_node.message = message

        try:
            self.agent.run(current_node)
            self.maybe_persist()
            return current_node.action or ""
        except RuntimeError as e:
            self.log(logger.error, f"Runtime error: {e.message}")
            return f"Error: {e.message}"

    def _create_next_node(self, parent: Node) -> Node:
        """Create a new node as a child of the parent node."""
        child_node = Node(
            node_id=self._generate_unique_id(),
            parent=parent,
            file_context=parent.file_context.clone() if parent.file_context else None,
        )
        parent.add_child(child_node)
        return child_node

    def get_last_node(self) -> Node:
        """Get the last node in the chat sequence."""
        return self.root.get_all_nodes()[-1]

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        total_usage = Usage()
        for node in self.root.get_all_nodes():
            total_usage += node.total_usage()
        return total_usage

    def maybe_persist(self):
        """Persist the chat state if a persist path is set."""
        if self.persist_path:
            self.persist(self.persist_path)

    def persist(self, file_path: str):
        """Persist the chat state to a file."""
        tree_data = self.model_dump(exclude_none=True)
        with open(file_path, "w") as f:
            import json
            json.dump(tree_data, f, indent=2)

    def _generate_unique_id(self) -> int:
        """Generate a unique ID for a new node."""
        return len(self.root.get_all_nodes())

    def assert_runnable(self):
        """Verify that the chat is properly configured to run."""
        if self.root is None:
            raise ValueError("Chat must have a root node.")

        if self.root.file_context is None:
            raise ValueError("Chat root node must have a file context.")

        if self.agent is None:
            raise ValueError("Chat must have an agent.")

        return True

    def log(self, logger_fn: Callable, message: str, **kwargs):
        """Log a message with metadata."""
        metadata = {**self.metadata, **kwargs}
        metadata_str = " ".join(f"{k}: {str(v)[:20]}" for k, v in metadata.items())
        log_message = f"[{metadata_str}] {message}" if metadata else message
        logger_fn(log_message) 