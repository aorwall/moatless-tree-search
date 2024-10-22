from typing import Type

from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.file_context import FileContext


class RejectArgs(ActionArguments):
    """Reject the task and explain why."""

    rejection_reason: str = Field(..., description="Explanation for rejection.")

    class Config:
        title = "Reject"

    def to_prompt(self):
        return f"Reject with reason: {self.rejection_reason}"


class Reject(Action):
    args_schema: Type[ActionArguments] = RejectArgs

    def execute(self, args: RejectArgs, file_context: FileContext | None = None):
        return Observation(message=args.rejection_reason, terminal=True)
