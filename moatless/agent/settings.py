from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from moatless.completion import CompletionModel


class MessageHistoryType(Enum):
    MESSAGES = "messages"  # Provides all messages in sequence
    SUMMARY = "summary"  # Generates one message with summarized history
    REACT = "react"


class AgentSettings(BaseModel):
    model_config = {"frozen": True}
    
    completion_model: CompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt to be used for generating completions"
    )
    actions: List[str] = Field(default_factory=list)
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Determines how message history is generated",
    )

    def __eq__(self, other):
        if not isinstance(other, AgentSettings):
            return False
        return (self.completion_model == other.completion_model and
                self.system_prompt == other.system_prompt and
                self.actions == other.actions and
                self.message_history_type == other.message_history_type)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["message_history_type"] = self.message_history_type.value
        return dump

    @classmethod
    def model_validate(
        cls,
        obj: Any
    ) -> "AgentSettings":
        if isinstance(obj, dict):
            if "message_history_type" in obj:
                obj["message_history_type"] = MessageHistoryType(
                    obj["message_history_type"]
                )

        return super().model_validate(obj)
